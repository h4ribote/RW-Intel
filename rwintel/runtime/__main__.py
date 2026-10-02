"""Builds the agents, prepares instance directories, and runs game processes under them.

    python -m rwintel.runtime build
    python -m rwintel.runtime instances --count 8
    python -m rwintel.runtime run --count 4 -- learn tactics --save local/models/tactics.pt
    python -m rwintel.runtime run --count 2 -- control --paired --spawn-probe 6 --max-seconds 180
    python -m rwintel.runtime run --count 1 -- control --versus --policy operations:local/models/ops-rl.pt
    python -m rwintel.runtime run --count 2 -- replay play local/replays/*.replay --journal local/episodes/control-<stamp>.jsonl
    python -m rwintel.runtime run --count 24 --controls 4 -- eval --arm script --arm operations:local/models/ops.pt --episodes 6 --record local/episodes/ops.jsonl
    python -m rwintel.runtime agents --count 2
    python -m rwintel.runtime pair
    python -m rwintel.runtime probe --count 8 --seconds 90
    python -m rwintel.runtime probe --count 8 --seconds 90 --clock wall --speed 10 --draw
    python -m rwintel.runtime probe --count 1 --seconds 200 --map Lake --agent-options obs=true
    python -m rwintel.runtime outcomes --count 8 --episodes 6 --map Islands --difficulty 1
    python -m rwintel.runtime lab --map Beach tools/lab-agent/scenarios/hovercraft.txt tools/lab-agent/scenarios/unreachable.txt
    python -m rwintel.runtime timeline local/logs/lab/<stamp>/00.out hc1 b1

`run` starts a control side (`control`, `eval`, `learn` or `replay`, with its arguments after --) and the games that dial in to it, and stops the games when the control side ends; with --controls a measurement is divided between several control sides and their journals pooled. `agents` and `pair` start only the games, for a control process started separately, which has to be started first. `probe` and `outcomes` use the probe agent and need nothing else. `lab` starts one game per experiment script under the experiment agent and stops each as its script ends.

Every launcher starts a private Xvfb for its games, creates any instance directory that is missing and keeps it from uploading crash reports, rebuilds an agent whose jar is older than its sources, and writes the games' output to local/logs/<command>/<stamp>/ beside a launch.json that records what was run. It stays in the foreground: the games stop when it does, at the end of --seconds, on Ctrl-C, or on SIGTERM.

Games run undrawn on the fixed clock unless told otherwise: every frame advances --step-ms of game time, as fast as the processors allow or at most --speed times real time. --clock wall gives the game its own clock back, at --speed times real time under an --fps cap, and --draw draws. A paired match runs on the wall clock, and a match against a person (`-- control --versus`) runs on it at real speed, printing the addresses to join it at. A playback (`-- replay play`) runs on the replay clock, which holds the engine multiplier at one and gives every frame --step-ms of elapsed time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Collection, Dict, Iterator, List, Optional

from .. import paths
from . import build, control, instances, lablog, ports, reports
from .display import DisplayError, VirtualDisplay
from .game import DEFAULT_HEAP, GameInstall, agent_options, command, environment, running_games
from .launch import Fleet, Spec, Terminated, early_failures, terminated_on_sigterm, write_manifest

#: Seconds after starting the games at which the launcher looks for any that died on start.
START_CHECK_SECONDS = 10.0

#: Times `run` starts a game again after it exits or crashes. A game rarely fails to get a usable OpenGL context when many start at once, and a game lost that way would otherwise leave its share of the episodes waiting for ever.
DEFAULT_RESTARTS = 3

#: Seconds between two counts of finished episodes in an outcome run.
OUTCOME_POLL_SECONDS = 20.0

#: Threads Mesa's software renderer gets in each game process. Left to itself it starts one per processor in every process, and with a process per core those threads compete with the simulations for the same cores while drawing a 10x10 window.
DEFAULT_RENDER_THREADS = 1

#: Game time per frame on the fixed clock: finer than the step the game takes when played at thirty frames a second, and a divisor of every control period, so that each period is the same whole number of steps.
DEFAULT_STEP_MS = 25

#: The engine speed multiplier on the wall clock when none is given.
DEFAULT_WALL_SPEED = 10.0

#: The frame rate cap on the wall clock when none is given, which is the cap the game sets itself.
DEFAULT_WALL_FPS = 300

#: The engine speed multiplier of a match against a person when none is given, which is the speed the game is played at.
DEFAULT_VERSUS_SPEED = 1.0


def say(text: str) -> None:
    print(text, flush=True)


def default_count() -> int:
    """One game process per logical processor: on the fixed clock a process keeps one processor busy, so more processes only share the same ones."""
    return max(1, min(64, os.cpu_count() or 1))


def clock(arguments) -> str:
    """The clock the games run on: the one asked for, or the fixed clock when none was."""
    return arguments.clock or "fixed"


def paired_clock(arguments) -> str:
    """Settles the clock of a paired lockstep match, and says why it cannot be had, or an empty string.

    A two process lockstep session does not take a frame's step from the elapsed time the fixed clock writes; each frame advances the engine multiplier's worth of sixtieths of a second instead, so on the fixed clock its steps would come out many times coarser than asked. A paired match therefore runs on the wall clock, which is its default.
    """
    if arguments.clock == "fixed":
        return "a lockstep match takes its steps from the network rather than the fixed clock; leave --clock out or pass --clock wall"
    arguments.clock = "wall"
    return ""


def versus_clock(arguments) -> str:
    """Settles the clock of a match against a person: the paired rule, at the speed the game is played at unless another was asked for."""
    problem = paired_clock(arguments)
    if not problem and arguments.speed is None:
        arguments.speed = DEFAULT_VERSUS_SPEED
    return problem


def replay_clock(arguments) -> str:
    """Settles the clock of a playback, which is the replay clock and nothing else: a recorded match's step scales with the engine multiplier the other clocks set, and a different step is a different match."""
    if arguments.clock not in (None, "replay"):
        return "a replay plays back on the replay clock; leave --clock out"
    arguments.clock = "replay"
    return ""


def replay_problem(arguments) -> str:
    """Why the clock asked for cannot run anything but a playback, or an empty string."""
    if arguments.clock == "replay":
        return "the replay clock is for playing replays back (`-- replay play ...`)"
    return ""


def frame_options(arguments) -> Dict[str, object]:
    """The frame layer's agent options: whether to draw, which clock, and the speed, whose default and meaning depend on the clock."""
    running = clock(arguments)
    speed = arguments.speed
    if speed is None:
        speed = 0.0 if running in ("fixed", "replay") else DEFAULT_WALL_SPEED
    return {"draw": "true" if arguments.draw else "false", "clock": running, "step": arguments.step_ms,
            "fps": arguments.fps, "speed": speed}


def period_problem(arguments) -> str:
    """Why the control agent's periods cannot run on this clock, or an empty string. A period starts on the first frame to reach it, so on the fixed clock a period the step does not divide would come out longer than asked, by up to a step."""
    if clock(arguments) != "fixed":
        return ""
    for name, period in (("--tactical-ms", arguments.tactical_ms), ("--operational-ms", arguments.operational_ms)):
        if period % arguments.step_ms:
            return f"{name} {period} is not a whole number of {arguments.step_ms} ms steps; pick a --step-ms that divides it"
    return ""


def describe_clock(arguments) -> str:
    options = frame_options(arguments)
    pace = f"at most {options['speed']:g}x" if options["speed"] > 0 else "as fast as the processors allow"
    if options["clock"] == "fixed":
        text = f"fixed {options['step']} ms steps, {pace}"
    elif options["clock"] == "replay":
        text = f"replay clock, {options['step']} ms elapsed a frame, {pace}"
    else:
        text = f"wall clock at {options['speed']:g}x, capped at {options['fps']} fps"
    return text + ("" if arguments.draw else ", drawing off")


@dataclass
class Run:
    install: GameInstall
    jar: str
    directories: List[str]
    names: List[str]
    log_directory: str
    display: str
    fleet: Fleet


@contextmanager
def _run(arguments, tool: str, target: str, count: int, offset: int = 0,
         log_directory: Optional[str] = None) -> Iterator[Run]:
    install = GameInstall.at(arguments.game).require()
    jar = build.ensure(build.TARGETS[target], install)
    prepared = instances.prepare(count, install, start=offset)
    if prepared.created or prepared.relinked:
        say(instances.describe(prepared))
    indices = list(range(offset, offset + count))
    directories = instances.require(indices)
    others = running_games()
    if others:
        say(f"warning: {len(others)} game process(es) already running (pid {', '.join(map(str, others))}); "
            "they share the processors with this run")
    log_directory = log_directory or paths.run_log_directory(tool)
    with VirtualDisplay(arguments.display, os.path.join(log_directory, "xvfb.log")) as display, \
            Fleet(log_directory) as fleet:
        yield Run(install, jar, directories, [f"{index:02d}" for index in indices], log_directory, display.name, fleet)


def _extra_environment(arguments) -> Dict[str, str]:
    extra = {}
    if arguments.render_threads and arguments.render_threads > 0:
        extra["LP_NUM_THREADS"] = str(arguments.render_threads)
    return extra


def _start_all(run: Run, arguments, options: List[str], **recorded) -> None:
    env = environment(run.install, run.display, extra=_extra_environment(arguments))
    for name, directory, option in zip(run.names, run.directories, options):
        run.fleet.start(Spec(name, directory, command(run.install, run.jar, option, arguments.heap), env))
    write_manifest(run.log_directory, game=run.install.root, agent=run.jar, display=run.display,
                   instances=dict(zip(run.names, run.directories)), options=dict(zip(run.names, options)),
                   frame=frame_options(arguments), environment=_extra_environment(arguments), heap=arguments.heap,
                   **recorded)


def _check_start(run: Run) -> bool:
    """Waits a moment and reports any game that died on start. False when none is left running."""
    run.fleet.wait(START_CHECK_SECONDS, poll=0.5)
    for line in early_failures(run.fleet):
        say(f"warning: {line}")
    if not run.fleet.alive():
        say(f"every game exited on start; see {run.log_directory}")
        return False
    return True


def _supervise(run: Run, seconds: int) -> str:
    """Waits out the rest of the requested time, or until the games exit or the launcher is stopped when no time was given."""
    remaining = seconds - START_CHECK_SECONDS
    if seconds > 0 and remaining <= 0:
        return "time"
    try:
        return run.fleet.wait(remaining if seconds > 0 else 0)
    except (KeyboardInterrupt, Terminated):
        return "interrupted"


# ---- build and instances ------------------------------------------------------------------------

def cmd_build(arguments) -> int:
    install = GameInstall.at(arguments.game).require()
    names = arguments.targets or sorted(build.TARGETS)
    for name in names:
        jar = build.build(build.TARGETS[name], install)
        say(f"built {jar}")
    return 0


def cmd_instances(arguments) -> int:
    prepared = instances.prepare(arguments.count, GameInstall.at(arguments.game), force=arguments.force)
    say(instances.describe(prepared))
    return 0


# ---- the control agent ----------------------------------------------------------------------------

def _agent_options(arguments, index: int, port: Optional[int] = None) -> str:
    # The instance number is the game's place in the run, from nought whatever directory it runs in, so an offset moves the directory and the log and not the identity the run knows an instance by. A run split between control processes keeps that numbering across them, and `port` is then the part's own control process.
    return agent_options({"host": arguments.control_host, "port": arguments.port if port is None else port, "instance": index,
                          "tactical": arguments.tactical_ms, "operational": arguments.operational_ms,
                          **frame_options(arguments)}, arguments.agent_options)


def _run_agents(arguments, tool: str, count: int, offset: int) -> int:
    problem = period_problem(arguments)
    if problem:
        say(f"error: {problem}")
        return 2
    with _run(arguments, tool, "agent", count, offset) as run:
        _start_all(run, arguments, [_agent_options(arguments, i) for i in range(count)])
        say(f"started {count} instance(s) against {arguments.control_host}:{arguments.port} on display {run.display}, "
            f"{describe_clock(arguments)}, logs in {run.log_directory}")
        if not _check_start(run):
            return 1
        reason = _supervise(run, arguments.seconds)
    say({"time": f"stopped {count} instance(s) after {arguments.seconds}s",
         "exited": "every instance has exited",
         "interrupted": f"stopped {count} instance(s)"}.get(reason, reason))
    return 0


def cmd_agents(arguments) -> int:
    return _run_agents(arguments, "agents", arguments.count, arguments.offset)


def cmd_pair(arguments) -> int:
    problem = paired_clock(arguments)
    if problem:
        say(f"error: {problem}")
        return 2
    problem = ports.check_free(arguments.match_port)
    if problem:
        say(f"{problem}. Stop it or pass a different --match-port, and give the control process the same one.")
        return 1
    say(f"paired match: instances 00 and 01, match port {arguments.match_port} is free")
    return _run_agents(arguments, "pair", 2, 0)


# ---- the control process and its games together -----------------------------------------------------

#: Seconds between two looks at whether the control process or the games have ended.
RUN_POLL_SECONDS = 0.5


def _restart_crashed(run: Run, limit: int, given_up: set, finished: Collection[str] = ()) -> None:
    """Starts again every game that has exited or crashed, up to `limit` times each, except the `finished` ones, whose control process has ended. A game past its limit is reported once and left as it is."""
    for running in list(run.fleet.running):
        if running.spec.name in given_up or running.spec.name in finished:
            continue
        reason = run.fleet.crashed(running)
        if reason is None:
            continue
        if running.restarts >= limit:
            given_up.add(running.spec.name)
            say(f"{running.spec.name} {reason}; not restarting it" + (f" after {limit} restart(s)" if limit else ""))
            continue
        say(f"{running.spec.name} {reason}; restarting it ({running.restarts + 1} of {limit}), "
            f"earlier logs kept as {os.path.basename(running.out)}.N")
        run.fleet.restart(running)


def cmd_run(arguments) -> int:
    try:
        wanted = control.prepare(arguments.control, arguments.count)
    except control.ControlError as error:
        say(f"error: {error}")
        return 2
    settle_clock = replay_clock if wanted.replay else versus_clock if wanted.versus else paired_clock if wanted.paired else None
    problem = (settle_clock(arguments) if settle_clock else replay_problem(arguments)) or period_problem(arguments)
    if problem:
        say(f"error: {problem}")
        return 2
    try:
        control.check_split(wanted, arguments.controls)
    except control.ControlError as error:
        say(f"error: {error}")
        return 2
    # Checked before the control side starts: a stale control process already listening there would answer the readiness check in the new one's place and the games would dial it.
    for port in range(wanted.port, wanted.port + arguments.controls):
        problem = ports.check_listener_free(wanted.host, port)
        if problem:
            say(f"{problem}. Stop it or pass a different --port to the control side"
                + (f"; a split run listens on {wanted.port} to {wanted.port + arguments.controls - 1}." if arguments.controls > 1 else "."))
            return 1
    if wanted.networked:
        problem = ports.check_free(wanted.match_port)
        if problem:
            say(f"{problem}. Stop it or pass a different --match-port to the control side.")
            return 1
    # The games dial the address the control side was told to listen on, whatever the launcher's own defaults are.
    arguments.control_host, arguments.port = wanted.connect_host, wanted.port
    # Built before the control side starts, so a build failure does not leave a control process waiting for games that never come.
    build.ensure(build.TARGETS["agent"], GameInstall.at(arguments.game).require())

    log_directory = paths.run_log_directory("run")
    shards = (control.split(wanted, arguments.controls, log_directory) if arguments.controls > 1
              else [control.whole(wanted)])
    # A run appending to a journal that already holds episodes reports the speed of its own episodes only.
    starts = [control.journal_size(shard.journal) for shard in shards]
    with terminated_on_sigterm(), ExitStack() as stack:
        processes = [stack.enter_context(control.ControlProcess(shard.command)) for shard in shards]
        for shard, process in zip(shards, processes):
            process.start()
            say(f"control side: {shard.command.describe()} (pid {process.pid})")
        for shard, process in zip(shards, processes):
            try:
                listening = process.wait_listening()
            except (KeyboardInterrupt, Terminated):
                say("interrupted")
                return _stop_all(processes) or 1
            if not listening:
                code = process.poll()
                if code is None:
                    say(f"the control side did not listen on {shard.command.connect_host}:{shard.command.port} within {control.LISTEN_SECONDS:.0f}s")
                    return _stop_all(processes) or 1
                say(f"the control side exited with {code} before it listened; no game was started")
                return _stop_all(processes) or code or 1

        reason = "games"
        started = time.monotonic()
        with _run(arguments, "run", "agent", arguments.count, arguments.offset, log_directory) as run:
            options = [_agent_options(arguments, shard.first + i, shard.command.port)
                       for shard in shards for i in range(shard.count)]
            recorded = {"control": wanted.argv}
            if len(shards) > 1:
                recorded["controls"] = [{"argv": shard.command.argv, "port": shard.command.port, "first": shard.first,
                                         "games": shard.count, "journal": shard.journal} for shard in shards]
            _start_all(run, arguments, options, **recorded)
            say(f"started {arguments.count} instance(s) against {wanted.connect_host}:"
                + (f"{wanted.port}" if len(shards) == 1 else f"{wanted.port}-{wanted.port + len(shards) - 1}")
                + f" on display {run.display}, {describe_clock(arguments)}, logs in {run.log_directory}")
            if wanted.versus:
                say("join from your own game client at " + " or ".join(ports.join_addresses(wanted.match_port)))
            # A lockstep match or a playback cannot have one side taken back and started over, so only independent games are restarted.
            restarts = 0 if wanted.networked or wanted.replay else arguments.restarts
            given_up = set()
            # The games of each part, which are stopped as soon as their control process ends so that the parts still running have the processors.
            games = [run.names[shard.first:shard.first + shard.count] for shard in shards]
            ended: List[int] = []
            finished: set = set()
            try:
                if _check_start(run):
                    while True:
                        for index, process in enumerate(processes):
                            if index not in ended and process.poll() is not None:
                                ended.append(index)
                                if len(ended) < len(processes):
                                    say(f"control side {index} exited with {process.poll()}; stopping its {len(games[index])} game(s)")
                                    finished.update(games[index])
                                    run.fleet.stop(games[index])
                        if len(ended) == len(processes):
                            reason = "control"
                            break
                        _restart_crashed(run, restarts, given_up, finished)
                        if not run.fleet.alive():
                            reason = "games"
                            break
                        time.sleep(RUN_POLL_SECONDS)
            except (KeyboardInterrupt, Terminated):
                reason = "interrupted"
            elapsed = time.monotonic() - started
            if reason != "control":
                # The control side goes first and on the Ctrl-C path, because its journal and a learning run's parameters are written on the way out and the games are what it is still talking to.
                say("stopping the control side" if reason == "interrupted" else "every game has exited; stopping the control side")
                _stop_all(processes)
        codes = [process.poll() for process in processes]
    say(f"control side exited with {', '.join(str(code) for code in codes)}; stopped {arguments.count} instance(s)")
    code = next((code for code in codes if code != 0), 0)
    if reason == "control":
        code = code if code is not None else 1
    else:
        code = code or 1
    if len(shards) > 1:
        return _pool(wanted, shards, elapsed, code)
    if shards[0].journal:
        say(reports.throughput(control.journal_entries(shards[0].journal, starts[0]), elapsed).line())
    return code


def _stop_all(processes: List[control.ControlProcess]) -> Optional[int]:
    """Stops every control process, and returns the first exit code that is not nought."""
    codes = [process.stop() for process in processes]
    return next((code for code in codes if code), None)


def _pool(wanted: control.ControlCommand, shards: List[control.Shard], elapsed: float, code: int) -> int:
    """Appends the parts' journals to the run's journal, reports them together and says how fast the run went. Episodes finished before an interruption or a failure are pooled and reported as well; the exit code is then the failure's."""
    journals = [shard.journal for shard in shards if control.journal_size(shard.journal) > 0]
    target = control.record_path(wanted, paths.stamp())
    pooled = control.pool(journals, target)
    if not pooled:
        say("no episode was journalled by any control side")
        return code or 1
    say(f"pooled {pooled} episode(s) from {len(journals)} control side(s) into {target}")
    reported = subprocess.run(control.report_argv(wanted, journals)).returncode
    say(reports.throughput((entry for journal in journals for entry in control.journal_entries(journal)), elapsed).line())
    return code or reported


# ---- the probe agent ------------------------------------------------------------------------------

def _out_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()
    except OSError:
        return []


def cmd_probe(arguments) -> int:
    count = arguments.count
    with _run(arguments, "probe", "probe", count, arguments.offset) as run:
        options = []
        for i in range(count):
            pairs = {"interval": arguments.interval_ms, **frame_options(arguments)}
            if arguments.map:
                # Each instance gets its own seed so that concurrent runs are not all the same match.
                pairs.update({"match": arguments.map, "ai": arguments.opponents, "difficulty": arguments.difficulty,
                              "episodes": arguments.episodes, "seed": 1000 + i, "maxSeconds": arguments.max_seconds})
            options.append(agent_options(pairs, arguments.agent_options))
        _start_all(run, arguments, options)
        say(f"running {count} instance(s) for {arguments.seconds}s on display {run.display}, {describe_clock(arguments)}, "
            f"logs in {run.log_directory}")
        if not _check_start(run):
            return 1
        reason = _supervise(run, arguments.seconds)
        if reason == "interrupted":
            say("interrupted")
    last = []
    for name in run.names:
        samples = reports.rate_samples(_out_lines(os.path.join(run.log_directory, f"{name}.out")))
        if samples:
            last.append(samples[-1])
        else:
            say(f"warning: instance {name} produced no measurement; see {os.path.join(run.log_directory, name + '.out')}")
    summary = reports.summarise_rates(last, frame_options(arguments)["speed"])
    if summary is None:
        say("no instance reported a measurement")
        return 1
    lines = summary.lines()
    with open(os.path.join(run.log_directory, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    for line in lines:
        say(line)
    return 0


def cmd_outcomes(arguments) -> int:
    count, wanted = arguments.count, arguments.count * arguments.episodes
    levels = [int(v) for v in arguments.levels.split(",")] if arguments.levels else []
    with _run(arguments, "outcomes", "probe", count, arguments.offset) as run:
        options = []
        for i in range(count):
            # A different seed per instance only varies the map's own randomisation; the match does not reproduce from a seed anyway.
            pairs = {"interval": 15000, "match": arguments.map, "ai": arguments.opponents,
                     "difficulty": arguments.difficulty, "contestants": 2, "episodes": arguments.episodes,
                     "seed": 1000 + i, "maxSeconds": arguments.max_seconds, **frame_options(arguments)}
            if levels:
                # The agent's options are comma separated, so the per contestant list travels with semicolons.
                pairs["levels"] = ";".join(map(str, levels))
            options.append(agent_options(pairs))
        _start_all(run, arguments, options)
        started = time.monotonic()
        matchup = f"difficulties {' vs '.join(map(str, levels))}" if levels else f"both at difficulty {arguments.difficulty}"
        say(f"running {count} instance(s) x {arguments.episodes} episode(s) = {wanted} episodes, two AI players, "
            f"{matchup}, on {arguments.map}, {describe_clock(arguments)}; logs in {run.log_directory}")
        if not _check_start(run):
            return 1
        outputs = [os.path.join(run.log_directory, f"{name}.out") for name in run.names]

        def finished() -> int:
            return sum(reports.count_results(_out_lines(path)) for path in outputs)

        deadline = time.monotonic() + arguments.timeout_minutes * 60
        try:
            while time.monotonic() < deadline:
                reason = run.fleet.wait(min(OUTCOME_POLL_SECONDS, max(1.0, deadline - time.monotonic())))
                done = finished()
                say(f"  {done}/{wanted} episodes finished")
                if done >= wanted or reason == "exited":
                    break
        except (KeyboardInterrupt, Terminated):
            say("interrupted")
        elapsed = time.monotonic() - started
    found = reports.outcomes(line for path in outputs for line in _out_lines(path))
    if not found:
        say(f"no episode finished; see {run.log_directory}")
        return 1
    lines = reports.outcome_lines(reports.summarise_outcomes(found), reports.game_seconds_per_second(found, elapsed), count)
    with open(os.path.join(run.log_directory, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    for line in lines:
        say(line)
    return 0


#: Marks a lab agent's output carries once its script has ended or its match could not be started.
LAB_DONE = ("[rw-lab] end", "[rw-lab] server failed", "[rw-lab] episode failed", "[rw-lab] boot failed", "[rw-lab] setup failed")


def cmd_lab(arguments) -> int:
    """One game per experiment script under the experiment agent, all on one map, each stopped when its script ends. `{out}` in a script stands for the run's log directory, where files a step writes belong."""
    count = len(arguments.scripts)
    with _run(arguments, "lab", "lab", count, arguments.offset) as run:
        options = []
        for name, script in zip(run.names, arguments.scripts):
            with open(script, encoding="utf-8") as handle:
                text = handle.read().replace("{out}", run.log_directory)
            prepared = os.path.join(run.log_directory, f"{name}-{os.path.basename(script)}")
            with open(prepared, "w", encoding="utf-8") as handle:
                handle.write(text)
            pairs = {**frame_options(arguments), "map": arguments.map, "script": prepared, "seed": arguments.seed,
                     "difficulty": arguments.difficulty, "halt": "false" if arguments.live_ai else "true"}
            options.append(agent_options(pairs, arguments.agent_options))
        _start_all(run, arguments, options, scripts=arguments.scripts)
        say(f"running {count} experiment(s) on {arguments.map}, logs in {run.log_directory}")
        if not _check_start(run):
            return 1

        def ended() -> bool:
            for name in run.names:
                lines = _out_lines(os.path.join(run.log_directory, f"{name}.out"))
                if not any(line.startswith(LAB_DONE) for line in lines):
                    return False
            return True

        try:
            reason = run.fleet.wait(max(1, arguments.seconds - START_CHECK_SECONDS), poll=1.0, until=ended)
        except (KeyboardInterrupt, Terminated):
            reason = "interrupted"
    for name, script in zip(run.names, arguments.scripts):
        lines = _out_lines(os.path.join(run.log_directory, f"{name}.out"))
        state = "ended" if any(line.startswith("[rw-lab] end") for line in lines) else "did not end"
        say(f"{script}: {state}, {sum(1 for line in lines if line.startswith('[rw-lab]'))} line(s) in {name}.out")
    return 0 if reason != "interrupted" else 1


def cmd_timeline(arguments) -> int:
    for line in lablog.read(arguments.output, arguments.labels):
        print(line)
    return 0


# ---- arguments ----------------------------------------------------------------------------------------

def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game", default=None, help=f"the game install, defaulting to local/{paths.GAME_DIRECTORY} or RWINTEL_GAME")
    parser.add_argument("--display", default=None,
                        help="'xvfb' (the default) starts a private virtual display; 'inherit' uses DISPLAY; a name such as :0 uses that display")
    parser.add_argument("--heap", default=DEFAULT_HEAP, help="maximum heap of each game process")
    parser.add_argument("--render-threads", type=int, default=DEFAULT_RENDER_THREADS,
                        help=f"threads Mesa's software renderer may use in each game process, default {DEFAULT_RENDER_THREADS}; 0 leaves it to Mesa")
    parser.add_argument("--draw", action="store_true", help="draw frames, which is off by default because nobody sees them")
    parser.add_argument("--clock", choices=("fixed", "wall", "replay"), default=None,
                        help="'fixed' advances every frame by --step-ms of game time as fast as the processors allow, and is the default; "
                             "'wall' is the game's own clock, driven by real elapsed time under --fps, and is the default and the only choice for a paired or versus lockstep match; "
                             "'replay' plays a recorded match back at its own step as fast as the processors allow, and is the only choice for `-- replay play`")
    parser.add_argument("--step-ms", type=_range(1, 200), default=DEFAULT_STEP_MS,
                        help=f"game time per frame on the fixed clock, and elapsed time per frame on the replay clock, default {DEFAULT_STEP_MS}")
    parser.add_argument("--fps", type=_range(1, 100000), default=DEFAULT_WALL_FPS,
                        help=f"frame rate cap on the wall clock, default {DEFAULT_WALL_FPS}")
    parser.add_argument("--speed", type=float, default=None,
                        help="fixed and replay clock: the most game time per real time, default no limit; "
                             f"wall clock: the engine speed multiplier, default {DEFAULT_WALL_SPEED:g}, or {DEFAULT_VERSUS_SPEED:g} against a person")


def _offset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offset", type=_range(0, 63), default=0,
                        help="instance directory to start numbering at, for a small run beside one already using the first ones")


def _range(low: int, high: int):
    def parse(text: str) -> int:
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"has to be between {low} and {high}")
        return value
    return parse


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m rwintel.runtime", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build", help="build the javaagents")
    p.add_argument("targets", nargs="*", choices=sorted(build.TARGETS), help="which to build, defaulting to all")
    p.add_argument("--game", default=None, help="the install whose Slick jar the agents compile against")
    p.set_defaults(handler=cmd_build)

    p = commands.add_parser("instances", help="create or repair instance directories")
    p.add_argument("--count", type=_range(1, 64), required=True)
    p.add_argument("--force", action="store_true", help="recreate existing instances, discarding their saves and settings")
    p.add_argument("--game", default=None)
    p.set_defaults(handler=cmd_instances)

    for name, handler, text in (("agents", cmd_agents, "start instances under the control agent"),
                                ("pair", cmd_pair, "start the two instances of one paired lockstep match")):
        p = commands.add_parser(name, help=text)
        if name == "agents":
            p.add_argument("--count", type=_range(1, 64), default=1)
            _offset(p)
        else:
            p.add_argument("--match-port", type=_range(1024, 65535), default=5123,
                           help="the port the hosting instance binds; give the control process the same --match-port")
        p.add_argument("--seconds", type=int, default=0, help="stop the instances after this long, 0 to run until Ctrl-C")
        p.add_argument("--control-host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8642, help="the control process's port")
        p.add_argument("--tactical-ms", type=int, default=200)
        p.add_argument("--operational-ms", type=int, default=2000)
        p.add_argument("--agent-options", default="", help="extra comma separated agent options, appended as given")
        _common(p)
        p.set_defaults(handler=handler)

    p = commands.add_parser("run", help="start a control side and its games together, and stop the games when it ends",
                            description="Starts `python -m rwintel.<control side>` with the arguments after --, waits for it to listen, "
                                        "starts --count games against it, and stops the games when it ends. Ctrl-C or SIGTERM stops "
                                        "the control side first, the way Ctrl-C would, and the games after it.")
    p.add_argument("--count", type=_range(1, 64), default=1,
                   help="games to start; the control side is given the same --instances unless it names its own, which has to agree")
    _offset(p)
    p.add_argument("--controls", type=_range(1, 64), default=1,
                   help="control processes to divide the games between, on the control side's --port and the ports after it, "
                        "default 1; only for eval and learn duel, whose journals are then pooled and reported together")
    p.add_argument("--restarts", type=_range(0, 100), default=DEFAULT_RESTARTS,
                   help=f"times each game that exits or crashes is started again, default {DEFAULT_RESTARTS}; "
                        "never for paired, versus or replay runs")
    p.add_argument("--tactical-ms", type=int, default=200)
    p.add_argument("--operational-ms", type=int, default=2000)
    p.add_argument("--agent-options", default="", help="extra comma separated agent options, appended as given")
    _common(p)
    p.add_argument("control", nargs=argparse.REMAINDER,
                   help="after --, the control side and its arguments: control, eval, learn or replay, for example -- learn tactics --save local/models/tactics.pt")
    p.set_defaults(handler=cmd_run)

    p = commands.add_parser("probe", help="run instances under the probe agent and report the simulation rate")
    p.add_argument("--count", type=_range(1, 64), default=1)
    _offset(p)
    p.add_argument("--seconds", type=_range(10, 3600), default=60)
    p.add_argument("--interval-ms", type=int, default=15000)
    p.add_argument("--map", default="", help="run skirmish episodes on the first built-in map whose file name contains this, instead of the menu's background battle")
    p.add_argument("--opponents", type=int, default=1)
    p.add_argument("--difficulty", type=_range(-2, 3), default=1)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--agent-options", default="",
                   help="extra comma separated probe options: catalog, obs, spawn, dump, act (see tools/probe-agent/RwProbeAgent.java)")
    _common(p)
    p.set_defaults(handler=cmd_probe)

    p = commands.add_parser("outcomes", help="run built-in AI against built-in AI and report the spread of outcomes")
    p.add_argument("--count", type=_range(1, 64), default=default_count(),
                   help="instances to run, defaulting to one per logical processor")
    _offset(p)
    p.add_argument("--map", default="Islands", help="needs a start for the local player's slot and both contestants, so a four player map")
    p.add_argument("--difficulty", type=_range(-2, 3), default=1)
    p.add_argument("--levels", default="", help="difficulty per contestant, comma separated, for example 1,0")
    p.add_argument("--opponents", type=int, default=2)
    p.add_argument("--episodes", type=int, default=6, help="episodes each instance runs")
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--timeout-minutes", type=int, default=40)
    _common(p)
    p.set_defaults(handler=cmd_outcomes)

    p = commands.add_parser("lab", help="run experiment scripts under the experiment agent, one game each")
    p.add_argument("scripts", nargs="+", help="experiment scripts, for example tools/lab-agent/scenarios/*.txt")
    _offset(p)
    p.add_argument("--map", default="Lake", help="the first built-in map whose file name contains this")
    p.add_argument("--seconds", type=_range(10, 7200), default=600, help="stop any game whose script has not ended by then")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--difficulty", type=_range(-2, 3), default=-2)
    p.add_argument("--live-ai", action="store_true", help="leave the computer players running instead of halting them")
    p.add_argument("--agent-options", default="", help="extra comma separated agent options, appended as given")
    _common(p)
    p.set_defaults(handler=cmd_lab)

    p = commands.add_parser("timeline", help="condense an experiment's output to the moments each unit changed")
    p.add_argument("output", help="an experiment game's NN.out under local/logs/lab/<stamp>/")
    p.add_argument("labels", nargs="*", help="only these labels' tracking lines")
    p.set_defaults(handler=cmd_timeline)
    return top


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        return arguments.handler(arguments)
    except (build.BuildError, DisplayError, FileNotFoundError, FileExistsError) as error:
        say(f"error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
