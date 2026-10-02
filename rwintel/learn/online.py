"""The actor/learner loop as one command: `python -m rwintel.learn online`.

It starts the following learner (`offline --follow`) and, once the learner has published its first version, the actor fleet through the runtime launcher (`python -m rwintel.runtime run --count N -- learn collect --student <published file> --reload-seconds N`), recording into a fresh run under the directory the learner watches. When the actors have played their episodes the learner is sent SIGTERM and waited for, so that its last version is published before the command returns. Ctrl-C or SIGTERM stops the actors first and the learner after them. A launcher still running when its grace period ends is killed together with the control side, display and games it started, which run in sessions of their own and would otherwise outlive it. Both processes' output goes to one run directory, `local/logs/online/<stamp>/`.

The command takes the learn command line once. The learner's options (the base datasets, the start, the network, the method's constants, publishing, the buffer, the device and `--card-share`) go to the learner, `--layer`, `--seed` and `--verbose` to both, and every other option to the actors; `--actor-device` and `--actor-card-share` (`ACTOR_CARD_SHARE` unless given) become the actors' `--device` and `--card-share`.
"""

from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .. import paths
from .reload import read_sidecar, sidecar_path

#: Options of the learn command line that only the learner reads, by their argparse destination.
LEARNER = frozenset({"dataset_sources", "load", "net", "width", "depth", "device", "batch", "learning_rate", "expectile",
                     "beta", "alpha", "top", "judge", "critic_epochs", "publish_every", "rescan_seconds", "updates", "buffer", "keep_tainted",
                     "fraction", "report", "method", "smoothing", "patience", "epochs", "follow", "watch", "card_share"})
#: Options both sides read. The reward's go to both so that the learner prices the actors' records at the terms they were paid under.
SHARED = frozenset({"layer", "seed", "verbose", "reward", "discount", "terminal_weight", "value_flow", "ground_flow",
                    "local_exchange", "achievement_weight", "predictor"})
#: Options of this command itself, which it passes on in its own form.
OWN = frozenset({"count", "offset", "actors", "save", "reload_seconds", "student", "actor_device", "actor_card_share"})

#: Share of the card the actors' control process takes unless --actor-card-share is given.
ACTOR_CARD_SHARE = 0.1

#: Seconds between two looks at the processes, and the default interval at which the actors look for a newer version.
POLL_SECONDS = 1.0
RELOAD_SECONDS = 30.0

#: Seconds the learner is given to publish and exit after SIGTERM, and the actors to stop their games, before they are killed.
LEARNER_EXIT_SECONDS = 600.0
ACTORS_EXIT_SECONDS = 180.0

#: Seconds the processes the launcher started are given to exit after SIGTERM when the launcher itself has to be killed.
ORPHAN_EXIT_SECONDS = 5.0

#: The methods a following learner learns by, the first being the one it takes when --method is not given.
FOLLOWING_METHODS = ("iql", "awr", "cql")


class Stopped(Exception):
    """SIGTERM reached the command."""


def following_method(method: Optional[str], command: str) -> str:
    """The method a following learner runs: --method when given, which has to be one of FOLLOWING_METHODS, and the first of them otherwise."""
    if method is None:
        return FOLLOWING_METHODS[0]
    if method not in FOLLOWING_METHODS:
        raise SystemExit(f"{command} learns with --method {', '.join(FOLLOWING_METHODS[:-1])} or {FOLLOWING_METHODS[-1]}, "
                         f"not {method}")
    return method


def split(parser, argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """The options of a learn command line meant for the learner and those meant for the actors, each in the order given; positional words and this command's own options are left out."""
    by_flag = {flag: action for action in parser._actions for flag in action.option_strings}
    learner: List[str] = []
    actors: List[str] = []
    words = list(argv)
    index = 0
    while index < len(words):
        word = words[index]
        index += 1
        if not word.startswith("--"):
            continue
        flag, joined, _ = word.partition("=")
        action = by_flag.get(flag)
        if action is None:
            raise SystemExit(f"online: no option {flag}")
        taken = [word]
        if not joined and action.nargs != 0:
            if action.nargs == "+":
                while index < len(words) and not words[index].startswith("--"):
                    taken.append(words[index])
                    index += 1
            elif index < len(words):
                taken.append(words[index])
                index += 1
        if action.dest in OWN:
            continue
        if action.dest in LEARNER or action.dest in SHARED:
            learner.extend(taken)
        if action.dest not in LEARNER:
            actors.extend(taken)
    return learner, actors


def commands(arguments, learner_options: Sequence[str], actor_options: Sequence[str]) -> Dict[str, object]:
    """The learner's and the actors' command lines, the directory the actors record under, and the run they record into."""
    method = following_method(arguments.method, "online")
    if not arguments.save:
        raise SystemExit("online publishes the learner's versions to --save, which was not given")
    if arguments.student:
        raise SystemExit("online has the actors play the learner's --save; leave --student out")
    actors_root = arguments.actors or os.path.join(paths.datasets(), arguments.layer, "actors")
    # The learner publishes its first version only once it has read a shard, and the actors start only after that version.
    if not arguments.dataset_sources and not glob.glob(os.path.join(actors_root, "*", "shard-*.npz")):
        raise SystemExit(f"online needs recorded decisions to start the learner on: give --dataset, or record a run under {actors_root} first")
    record = os.path.join(actors_root, f"online-{arguments.stamp}")
    reload_seconds = arguments.reload_seconds if arguments.reload_seconds is not None else RELOAD_SECONDS
    share = arguments.actor_card_share if arguments.actor_card_share is not None else ACTOR_CARD_SHARE
    placement = ["--device", arguments.actor_device] if arguments.actor_device else []
    learner = [sys.executable, "-m", "rwintel.learn", "offline", "--follow", "--layer", arguments.layer, "--method", method,
               "--save", arguments.save, "--watch", actors_root,
               *_drop(learner_options, ("--layer", "--method", "--follow"))]
    actors = [sys.executable, "-m", "rwintel.runtime", "run", "--count", str(arguments.count), "--offset",
              str(arguments.offset), "--", "learn", "collect", "--layer", arguments.layer, "--student", arguments.save,
              "--reload-seconds", f"{reload_seconds:g}", "--dataset", record, *placement, "--card-share", f"{share:g}",
              *_drop(actor_options, ("--layer",))]
    return {"learner": learner, "actors": actors, "actors_root": actors_root, "record": record}


def _drop(options: Sequence[str], flags: Sequence[str]) -> List[str]:
    """The options without the named flags and their values."""
    out: List[str] = []
    skip = False
    for word in options:
        if skip:
            skip = False
            continue
        name = word.partition("=")[0]
        if name in flags:
            skip = "=" not in word
            continue
        out.append(word)
    return out


def _start(command: Sequence[str], log_path: str) -> subprocess.Popen:
    handle = open(log_path, "ab")
    try:
        # A session of its own, so that a Ctrl-C at the terminal reaches this command alone, which stops the two in order.
        return subprocess.Popen(list(command), stdout=handle, stderr=subprocess.STDOUT, cwd=paths.REPOSITORY,
                                start_new_session=True)
    finally:
        handle.close()


def _stamp(path: str) -> Optional[Tuple[int, int]]:
    """A file's modification time and inode, which change when the learner replaces it; None while there is none."""
    try:
        status = os.stat(path)
    except OSError:
        return None
    return status.st_mtime_ns, status.st_ino


def _stop(process: Optional[subprocess.Popen], seconds: float, name: str, launcher: bool = False) -> Optional[int]:
    """Sends SIGTERM and waits up to `seconds`, then kills, the runtime launcher together with what it started (`_kill_launcher`); the exit code."""
    if process is None:
        return None
    if process.poll() is None:
        print(f"stopping the {name} (pid {process.pid})", flush=True)
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            print(f"the {name} did not stop within {seconds:.0f}s; killing " + ("it and what it started" if launcher else "it"),
                  flush=True)
            if launcher:
                _kill_launcher(process)
            else:
                process.kill()
                process.wait()
    return process.returncode


def _status(pid: int, proc: str = "/proc") -> Optional[Tuple[str, int, int]]:
    """A process's state letter, parent and process group from /proc; None when there is no such process."""
    try:
        with open(os.path.join(proc, str(pid), "stat"), "rb") as handle:
            fields = handle.read().rpartition(b")")[2].split()
    except OSError:
        return None
    if len(fields) < 3:
        return None
    return fields[0].decode("ascii", "replace"), int(fields[1]), int(fields[2])


def _running(pid: int) -> bool:
    """Whether the process exists and has not exited; a zombie waiting for its parent has exited."""
    status = _status(pid)
    return status is not None and status[0] not in ("Z", "X")


def _children(pid: int, proc: str = "/proc") -> List[Tuple[int, int]]:
    """The processes whose parent is `pid`, each with its process group; empty when /proc cannot be read."""
    try:
        entries = os.listdir(proc)
    except OSError:
        return []
    found = []
    for entry in entries:
        if entry.isdigit():
            status = _status(int(entry), proc)
            if status is not None and status[1] == pid:
                found.append((int(entry), status[2]))
    return found


def _signal_group(group: int, signum: int) -> None:
    try:
        os.killpg(group, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_launcher(process: subprocess.Popen) -> None:
    """Kills the runtime launcher without orphaning the control side, display and games it started, each of which runs in a session of its own.

    The launcher is frozen first so that it cannot start a game in place of one that ends; every process group of its children is then sent SIGTERM, and SIGKILL once ORPHAN_EXIT_SECONDS have passed with any of them still running, and the launcher's own group is killed last.
    """
    try:
        process.send_signal(signal.SIGSTOP)
    except ProcessLookupError:
        pass
    children = _children(process.pid)
    groups = sorted({group for _, group in children if group != process.pid})
    if children:
        print(f"ending the {len(children)} process(es) the launcher started", flush=True)
    for group in groups:
        _signal_group(group, signal.SIGTERM)
    deadline = time.monotonic() + ORPHAN_EXIT_SECONDS
    while any(_running(pid) for pid, group in children if group != process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    # A child seen between its fork and its own session is in a group of its own by now; the frozen launcher has started none since.
    groups = sorted(set(groups) | {group for _, group in _children(process.pid) if group != process.pid})
    for group in groups:
        _signal_group(group, signal.SIGKILL)
    _signal_group(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.kill()
    process.wait()


def run(arguments, parser, argv: Sequence[str]) -> int:
    learner_options, actor_options = split(parser, argv)
    planned = commands(arguments, learner_options, actor_options)
    directory = paths.run_log_directory("online")
    with open(os.path.join(directory, "online.json"), "w", encoding="utf-8") as handle:
        json.dump(planned, handle, indent=1)
    learner_log, actors_log = os.path.join(directory, "learner.log"), os.path.join(directory, "actors.log")

    def terminated(*_):
        raise Stopped()

    previous = signal.signal(signal.SIGTERM, terminated)
    interrupt = signal.getsignal(signal.SIGINT)
    learner = actors = None
    code = 1
    try:
        sidecar = sidecar_path(arguments.save)
        before = _stamp(sidecar)
        learner = _start(planned["learner"], learner_log)
        print(f"learner started (pid {learner.pid}), output in {learner_log}", flush=True)
        while _stamp(sidecar) in (None, before):
            if learner.poll() is not None:
                print(f"the learner exited with {learner.returncode} before publishing; see {learner_log}", flush=True)
                return learner.returncode or 1
            time.sleep(POLL_SECONDS)
        print(f"the learner published version {(read_sidecar(arguments.save) or {}).get('version')}; starting "
              f"{arguments.count} actor game(s)", flush=True)
        actors = _start(planned["actors"], actors_log)
        print(f"actors started (pid {actors.pid}), output in {actors_log}", flush=True)
        while actors.poll() is None:
            if learner.poll() is not None:
                print(f"the learner exited with {learner.returncode} while the actors played; stopping them", flush=True)
                _stop(actors, ACTORS_EXIT_SECONDS, "actors", launcher=True)
                return learner.returncode or 1
            time.sleep(POLL_SECONDS)
        print(f"the actors exited with {actors.returncode}", flush=True)
        learned = _stop(learner, LEARNER_EXIT_SECONDS, "learner")
        code = actors.returncode or learned or 0
    except (KeyboardInterrupt, Stopped):
        print("interrupted", flush=True)
    finally:
        # A further Ctrl-C or SIGTERM is ignored while the two are stopped, so that both are always waited for.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _stop(actors, ACTORS_EXIT_SECONDS, "actors", launcher=True)
        _stop(learner, LEARNER_EXIT_SECONDS, "learner")
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, previous)
        published = read_sidecar(arguments.save) or {}
        print(f"weights: {arguments.save} (version {published.get('version')}, {published.get('updates')} update(s) "
              f"over {published.get('decisions')} decision(s))", flush=True)
        print(f"actors' dataset: {planned['record']}", flush=True)
        print(f"logs: {directory}", flush=True)
    return code
