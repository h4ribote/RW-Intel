"""Starting a group of game processes, watching them, and stopping them together.

Each process runs in a session of its own with its output in the run's log directory, as NN.out and NN.err after the instance directory it runs in. Stopping sends SIGTERM to every process group and SIGKILL to whatever is still there after a grace period, and it happens however the launcher ends: at the end of the requested time, on Ctrl-C, on SIGTERM, or on an error. A process that has exited or whose engine reports a crash can be started again in its place, with its earlier logs kept beside the new ones.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

#: Seconds a process is given to exit after SIGTERM before it is killed.
GRACE_SECONDS = 5.0

#: What the engine writes to its standard output when an uncaught exception ends the game. The process does not exit after it; it keeps showing a crash screen to nobody.
CRASH_MARKER = b"----------- onGameCrash ----------"


@dataclass
class Spec:
    """One process to start: the name its logs go under, the directory it runs in, and what to run."""

    name: str
    cwd: str
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class Running:
    spec: Spec
    process: subprocess.Popen
    out: str
    err: str
    restarts: int = 0
    #: Bytes of `out` already searched for the crash marker.
    scanned: int = 0


class Terminated(Exception):
    """Raised in the main thread when the launcher is sent SIGTERM, so that the same clean-up runs as on Ctrl-C."""


def _on_terminate(signum, frame):
    raise Terminated()


@contextmanager
def terminated_on_sigterm() -> Iterator[None]:
    """Turns SIGTERM into `Terminated` for the duration, so that the clean-up written for Ctrl-C also runs when the launcher is stopped by a signal."""
    try:
        previous = signal.signal(signal.SIGTERM, _on_terminate)
    except ValueError:
        # Not the main thread, so the handler cannot be installed; the callers' context managers still clean up on the way out.
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


class Fleet:
    """The processes of one run. Use as a context manager so that they are stopped however the run ends."""

    def __init__(self, log_directory: str) -> None:
        self.log_directory = log_directory
        self.running: List[Running] = []
        self._signals = None

    def __enter__(self) -> "Fleet":
        self._signals = terminated_on_sigterm()
        self._signals.__enter__()
        return self

    def __exit__(self, *_) -> None:
        try:
            self.stop()
        finally:
            self._signals.__exit__(None, None, None)

    def start(self, spec: Spec) -> Running:
        running = self._launch(spec)
        self.running.append(running)
        return running

    def _launch(self, spec: Spec) -> Running:
        out = os.path.join(self.log_directory, f"{spec.name}.out")
        err = os.path.join(self.log_directory, f"{spec.name}.err")
        with open(out, "wb") as stdout, open(err, "wb") as stderr:
            process = subprocess.Popen(spec.argv, cwd=spec.cwd, env=spec.env or None, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
        return Running(spec, process, out, err)

    def crashed(self, running: Running) -> Optional[str]:
        """Why a process can no longer play: it has exited, or the engine has written its crash marker since the last look. None while it is healthy.

        The log is searched incrementally, so a marker is reported once; an exit is reported every time it is asked about.
        """
        code = running.process.poll()
        if code is not None:
            return f"exited with {code}"
        start = max(0, running.scanned - len(CRASH_MARKER) + 1)
        try:
            with open(running.out, "rb") as handle:
                handle.seek(start)
                chunk = handle.read()
        except OSError:
            return None
        running.scanned = start + len(chunk)
        return "crashed" if CRASH_MARKER in chunk else None

    def restart(self, running: Running) -> Running:
        """Stops one process and starts its spec again in its place, keeping what it logged as NN.out.K and NN.err.K under the next free K."""
        _signal_group(running.process, signal.SIGTERM)
        _reap(running.process, time.monotonic() + GRACE_SECONDS)
        suffix = 1
        while os.path.exists(f"{running.out}.{suffix}") or os.path.exists(f"{running.err}.{suffix}"):
            suffix += 1
        for path in (running.out, running.err):
            if os.path.exists(path):
                os.replace(path, f"{path}.{suffix}")
        replacement = self._launch(running.spec)
        replacement.restarts = running.restarts + 1
        self.running[self.running.index(running)] = replacement
        return replacement

    def alive(self) -> List[Running]:
        return [r for r in self.running if r.process.poll() is None]

    def wait(self, seconds: float, poll: float = 1.0, until: Optional[Callable[[], bool]] = None) -> str:
        """Waits for the given time, zero meaning for as long as any process lives, and says why it stopped waiting.

        The answer is "time", "exited" when every process has ended by itself, or "done" when `until` said so.
        """
        deadline = time.monotonic() + seconds if seconds > 0 else None
        while True:
            if not self.alive():
                return "exited"
            if until is not None and until():
                return "done"
            if deadline is not None and time.monotonic() >= deadline:
                return "time"
            time.sleep(poll if deadline is None else max(0.0, min(poll, deadline - time.monotonic())))

    def stop(self, names: Optional[Iterable[str]] = None) -> None:
        """Stops every process, or only those whose spec carries one of `names`."""
        chosen = set(names) if names is not None else None
        stopping = [r for r in self.running if chosen is None or r.spec.name in chosen]
        for running in stopping:
            if running.process.poll() is None:
                _signal_group(running.process, signal.SIGTERM)
        deadline = time.monotonic() + GRACE_SECONDS
        for running in stopping:
            _reap(running.process, deadline)

    def exit_codes(self) -> Dict[str, Optional[int]]:
        return {r.spec.name: r.process.poll() for r in self.running}


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _reap(process: subprocess.Popen, deadline: float) -> None:
    """Waits until the deadline for a process already sent SIGTERM, and kills its group if it is still there."""
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait()


def write_manifest(log_directory: str, **fields) -> str:
    """Writes what a run was started with next to its logs, so that a log directory says on its own what produced it."""
    path = os.path.join(log_directory, "launch.json")
    record = {"argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    record.update(fields)
    from .. import paths

    with paths.replacing(path) as handle:
        json.dump(record, handle, indent=1, ensure_ascii=False, default=str)
    return path


def early_failures(fleet: Fleet) -> Sequence[str]:
    """The processes that have already exited, each with the last line of its error log, which is where a failed start says why."""
    lines = []
    for running in fleet.running:
        code = running.process.poll()
        if code is None:
            continue
        last = ""
        try:
            with open(running.err, "r", encoding="utf-8", errors="replace") as handle:
                tail = [line.strip() for line in handle if line.strip()]
            last = tail[-1] if tail else ""
        except OSError:
            pass
        lines.append(f"{running.spec.name} exited with {code}" + (f": {last}" if last else ""))
    return lines
