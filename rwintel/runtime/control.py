"""The control process as a child of the launcher, for running it and its games as one command.

The command line after `--` names the control side (`control`, `eval`, `learn` or `replay`) and is passed to it as written, except that the instance count is filled in from the launcher's own when it is missing. The launcher reads the host and port from it so that the games dial the address the control process actually listens on, and a count written on both sides that disagrees is refused rather than left to a run that waits forever for an instance that was never started.

The control process runs in a session of its own, so a Ctrl-C at the terminal reaches the launcher alone, which then passes it on as SIGINT: that is the path on which the control side closes its journal and a learning run saves its parameters. Its SIGINT is reset to the default before it starts, because a launcher started in the background of a non-interactive shell inherits SIGINT ignored and would otherwise hand that on.

A measurement (`eval` or `learn duel`) can be split between several control processes on consecutive ports (`split`), because one Python process answering every game is what limits how fast a machine with processors to spare plays. Each part's games announce the instance numbers they would have had in the same run unsplit, so every seed drawn from an instance number comes out as it would have, and the parts' journals are pooled into one afterwards (`pool`).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import Iterator, List, Optional, Sequence

from .. import paths

#: The control sides a launcher can run, by the name written after `--`.
MODULES = {"control": "rwintel.control", "eval": "rwintel.eval", "learn": "rwintel.learn", "replay": "rwintel.replay"}

#: Runs of `rwintel.learn` that never listen for games.
OFFLINE_LEARNING = ("clone", "dataset")

#: Runs of `rwintel.learn`, which is how its first positional argument is told apart from an option's value.
LEARNING_RUNS = ("tactics", "operations", "economy", "collect", "clone", "duel", "matchups", "dataset")

#: Runs of `rwintel.replay` that read files and never listen for games.
OFFLINE_REPLAY = ("inspect", "verify")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
DEFAULT_MATCH_PORT = 5123

#: Seconds the control process is given to start listening, which covers reading the map and unit files and loading the tensor library.
LISTEN_SECONDS = 60.0

#: Seconds a control process is given after SIGINT to close its records and, for a learning run, finish its last update and save, before it is sent SIGTERM.
STOP_SECONDS = 60.0

#: Options of an evaluation that shape only its report, which the pooled report of a split run is handed as written.
EVAL_REPORT_OPTIONS = ("--reference", "--weights", "--lead", "--fit-out")


class ControlError(ValueError):
    pass


def _option(words: Sequence[str], name: str) -> Optional[str]:
    """The value of the last `--name value` or `--name=value` in the words, or None."""
    value = None
    for index, word in enumerate(words):
        if word == name:
            if index + 1 >= len(words):
                raise ControlError(f"{name} is missing its value")
            value = words[index + 1]
        elif word.startswith(name + "="):
            value = word[len(name) + 1:]
    return value


def _options(words: Sequence[str], name: str) -> List[str]:
    """Every value of `--name value` and `--name=value` in the words, in order."""
    values = []
    for index, word in enumerate(words):
        if word == name:
            if index + 1 >= len(words):
                raise ControlError(f"{name} is missing its value")
            values.append(words[index + 1])
        elif word.startswith(name + "="):
            values.append(word[len(name) + 1:])
    return values


def _replaced(words: Sequence[str], name: str, value: Optional[str]) -> List[str]:
    """The words without any `--name value` or `--name=value`, and with `--name value` at the end unless the value is None."""
    kept, skip = [], False
    for word in words:
        if skip:
            skip = False
        elif word == name:
            skip = True
        elif not word.startswith(name + "="):
            kept.append(word)
    return kept + ([name, value] if value is not None else [])


def _integer(words: Sequence[str], name: str, default: int) -> int:
    written = _option(words, name)
    if written is None:
        return default
    try:
        return int(written)
    except ValueError:
        raise ControlError(f"{name} has to be a whole number, not {written!r}")


def _learning_run(words: Sequence[str]) -> Optional[str]:
    for index, word in enumerate(words):
        if word in LEARNING_RUNS and (index == 0 or words[index - 1] != "--layer"):
            return word
    return None


@dataclass(frozen=True)
class ControlCommand:
    """A control side to start, with what the games need to know about it."""

    name: str
    arguments: List[str]
    host: str
    port: int
    instances: int
    paired: bool
    match_port: int
    #: One instance hosting a match for a person to join from their own game client.
    versus: bool = False
    #: The run of `rwintel.learn` (`duel`, `tactics` ...), empty for any other control side.
    run: str = ""

    @property
    def splittable(self) -> bool:
        """Whether the games can be divided between several control processes: a measurement whose episodes are independent of one another and whose journals can be reported together."""
        if self.networked:
            return False
        if self.name == "eval":
            return _option(self.arguments, "--from") is None
        return self.name == "learn" and self.run == "duel"

    @property
    def replay(self) -> bool:
        """Whether the games play recorded matches back, which fixes their clock."""
        return self.name == "replay"

    @property
    def networked(self) -> bool:
        """Whether the games host or join a lockstep session, which fixes their clock and needs the match port free."""
        return self.paired or self.versus

    @property
    def module(self) -> str:
        return MODULES[self.name]

    @property
    def argv(self) -> List[str]:
        return [sys.executable, "-m", self.module, *self.arguments]

    @property
    def connect_host(self) -> str:
        """Where the games dial: the listening address, or the loopback when it listens on every address."""
        return DEFAULT_HOST if self.host in ("", "0.0.0.0") else self.host

    def describe(self) -> str:
        return " ".join(["python", "-m", self.module, *self.arguments])


def prepare(words: Sequence[str], count: int) -> ControlCommand:
    """Checks the control side's command line against the launcher's instance count and fills in what it leaves out."""
    words = list(words)
    if words and words[0] == "--":
        words = words[1:]
    if not words:
        raise ControlError(f"name the control side after --: one of {', '.join(MODULES)}")
    name, arguments = words[0], words[1:]
    if name not in MODULES:
        raise ControlError(f"no control side named {name!r}: expected one of {', '.join(MODULES)}")
    run = _learning_run(arguments) if name == "learn" else None
    if run in OFFLINE_LEARNING:
        raise ControlError(f"learn {run} reads recorded files and starts no game; run it directly with python -m rwintel.learn {run}")
    if run == "duel" and _option(arguments, "--from") is not None:
        raise ControlError("a duel reported --from its journals starts no game; run it directly with python -m rwintel.learn duel --from")
    if name == "replay" and (not arguments or arguments[0] != "play"):
        run = arguments[0] if arguments else ""
        if run in OFFLINE_REPLAY:
            raise ControlError(f"replay {run} reads files and starts no game; run it directly with python -m rwintel.replay {run}")
        raise ControlError("the replay control side is `replay play <replay>...`")
    written = _option(arguments, "--instances")
    if written is None:
        arguments = [*arguments, "--instances", str(count)]
    else:
        try:
            wanted = int(written)
        except ValueError:
            raise ControlError(f"--instances has to be a whole number, not {written!r}")
        if wanted != count:
            raise ControlError(f"the control side waits for {wanted} instance(s) but {count} would be started; give one --count")
    paired = "--paired" in arguments
    versus = "--versus" in arguments
    if versus and name != "control":
        raise ControlError(f"a match against a person is run by the control side, not {name}")
    if versus and paired:
        raise ControlError("--versus and --paired are different matches; give one of them")
    if paired and count != 2:
        raise ControlError(f"a paired match is two instances, not {count}")
    if versus and count != 1:
        raise ControlError(f"a match against a person is hosted by one instance, not {count}")
    return ControlCommand(
        name=name, arguments=arguments,
        host=_option(arguments, "--host") or DEFAULT_HOST,
        port=_integer(arguments, "--port", DEFAULT_PORT),
        instances=count, paired=paired,
        match_port=_integer(arguments, "--match-port", DEFAULT_MATCH_PORT),
        versus=versus, run=run or "",
    )


@dataclass(frozen=True)
class Shard:
    """One of the control processes a run's games are divided between, and the games it serves."""

    command: ControlCommand
    #: The instance number its first game announces, which is that game's number in the same run unsplit.
    first: int
    count: int
    #: Where its episodes are journalled, or an empty string when the launcher was not told.
    journal: str


def whole(command: ControlCommand) -> Shard:
    """The one part of a run that is not split, journalling wherever its command says."""
    return Shard(command, 0, command.instances, _option(command.arguments, "--record") or "")


def check_split(command: ControlCommand, controls: int) -> None:
    """Refuses a division of the run's games between `controls` control processes that cannot be made."""
    if controls < 1 or controls > command.instances:
        raise ControlError(f"{command.instances} game(s) cannot be divided between {controls} control process(es)")
    if controls > 1 and not command.splittable:
        raise ControlError("only a measurement whose episodes stand alone can be split between control processes: "
                           "`eval` other than --from, or `learn duel`")


def split(command: ControlCommand, controls: int, directory: str) -> List[Shard]:
    """The run's games divided between `controls` control processes on consecutive ports, as evenly as they go with the larger parts first, each journalling to its own file in `directory`.

    A `--card-share` is divided between them, so that the run as a whole takes the share asked for, and a `--fit-out` is left to the pooled report.
    """
    check_split(command, controls)
    written = _option(command.arguments, "--card-share")
    try:
        share = None if written is None else float(written) / controls
    except ValueError:
        raise ControlError(f"--card-share has to be a number, not {written!r}")
    shards, first = [], 0
    for index in range(controls):
        count = command.instances // controls + (1 if index < command.instances % controls else 0)
        journal = os.path.join(directory, f"journal-{index}.jsonl")
        port = command.port + index
        words = _replaced(command.arguments, "--port", str(port))
        words = _replaced(words, "--instances", str(count))
        words = _replaced(words, "--record", journal)
        words = _replaced(words, "--fit-out", None)
        if share is not None:
            words = _replaced(words, "--card-share", f"{share:g}")
        shards.append(Shard(replace(command, arguments=words, port=port, instances=count), first, count, journal))
        first += count
    return shards


def record_path(command: ControlCommand, stamp: str) -> str:
    """Where a split run's pooled journal goes: where its command said to record, or a file of its own under local/episodes."""
    written = _option(command.arguments, "--record")
    if written:
        return written
    return os.path.join(paths.episodes(), f"{'eval' if command.name == 'eval' else 'duel'}-{stamp}.jsonl")


def report_argv(command: ControlCommand, journals: Sequence[str]) -> List[str]:
    """The command that reports the journals of a split run's parts together, with the options of the run that shape its report."""
    if command.name != "eval":
        return [sys.executable, "-m", MODULES["learn"], "duel", "--from", *journals]
    words = ["--from", *journals]
    for arm in _options(command.arguments, "--arm"):
        words += ["--arm", arm]
    for name in EVAL_REPORT_OPTIONS:
        value = _option(command.arguments, name)
        if value is not None:
            words += [name, value]
    if "--verbose" in command.arguments:
        words.append("--verbose")
    return [sys.executable, "-m", MODULES["eval"], *words]


def journal_size(path: str) -> int:
    """The bytes a journal already holds, from which the episodes of a run appending to it start."""
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def journal_entries(path: str, start: int = 0) -> Iterator[dict]:
    """The episodes a journal holds from byte `start` on; nothing when it does not exist."""
    if not path or not os.path.exists(path):
        return
    with open(path, "rb") as handle:
        handle.seek(start)
        for line in handle:
            if line.strip():
                yield json.loads(line.decode("utf-8"))


def pool(journals: Sequence[str], target: str) -> int:
    """Appends the episodes of the parts' journals to `target` in the parts' order, the way a journal is appended to, and returns how many there were."""
    lines = []
    for journal in journals:
        if os.path.exists(journal):
            with open(journal, "rb") as handle:
                lines.extend(line if line.endswith(b"\n") else line + b"\n" for line in handle if line.strip())
    with open(paths.ensure_parent(target), "ab") as handle:
        handle.writelines(lines)
    return len(lines)


def _default_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class ControlProcess:
    """A running control side. Use as a context manager so that it is stopped however the launcher ends."""

    def __init__(self, command: ControlCommand, argv: Optional[List[str]] = None) -> None:
        self.command = command
        self.argv = argv or command.argv
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self) -> "ControlProcess":
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self) -> None:
        self.process = subprocess.Popen(self.argv, preexec_fn=_default_sigint, start_new_session=True)

    def poll(self) -> Optional[int]:
        return self.process.poll() if self.process is not None else None

    @property
    def pid(self) -> int:
        return self.process.pid if self.process is not None else -1

    def wait_listening(self, seconds: float = LISTEN_SECONDS, poll: float = 0.2) -> bool:
        """True once the control side accepts a connection; False if it exits first or the time runs out."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.poll() is not None:
                return False
            try:
                with socket.create_connection((self.command.connect_host, self.command.port), timeout=poll):
                    return True
            except OSError:
                time.sleep(poll)
        return False

    def stop(self, grace: float = STOP_SECONDS) -> Optional[int]:
        """Stops the control side as Ctrl-C would, then harder if it does not go, and returns its exit code."""
        process = self.process
        if process is None:
            return None
        for signum, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, None)):
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        return process.poll()
