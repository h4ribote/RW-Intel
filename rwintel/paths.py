"""Where RW-Intel reads and writes on disk.

Everything a run produces goes under one working area, `local/` at the repository root, which is outside version control because it holds the game itself and measurements of one machine. The layout is fixed here so that no module resolves a path against the current directory:

    local/RustedWarfare_Linux/   the game install, read only
    local/instances/NN/          one working directory per game process
    local/logs/<tool>/<stamp>/   a launcher's game process logs, NN.out and NN.err
    local/logs/<tool>/<stamp>.log  a Python process's own log
    local/logs/<tool>/latest     the most recent of either
    local/episodes/              episode journals, one file per run
    local/models/                network parameters
    local/datasets/<layer>/<run>/  recorded decisions, one directory per run
    local/interventions/         a person's interventions beside the board they were taken from
    local/reports/               what the data tools export
    local/replays/               kept replays, and beside each what playing it back produced

RWINTEL_LOCAL moves the working area and RWINTEL_GAME points at a game install elsewhere.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import sys
import time
from contextlib import contextmanager
from typing import IO, Iterator, Optional

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The directory the Linux release unpacks to, which is also the default name of the install under the working area.
GAME_DIRECTORY = "RustedWarfare_Linux"


def local_root() -> str:
    return os.path.abspath(os.environ.get("RWINTEL_LOCAL") or os.path.join(REPOSITORY, "local"))


def game() -> str:
    return os.path.abspath(os.environ.get("RWINTEL_GAME") or os.path.join(local_root(), GAME_DIRECTORY))


def instances() -> str:
    return os.path.join(local_root(), "instances")


def instance(index: int) -> str:
    return os.path.join(instances(), f"{index:02d}")


def logs(tool: str) -> str:
    return os.path.join(local_root(), "logs", tool)


def episodes() -> str:
    return os.path.join(local_root(), "episodes")


def models() -> str:
    return os.path.join(local_root(), "models")


def datasets() -> str:
    return os.path.join(local_root(), "datasets")


def interventions() -> str:
    return os.path.join(local_root(), "interventions")


def reports() -> str:
    return os.path.join(local_root(), "reports")


def replays() -> str:
    return os.path.join(local_root(), "replays")


def stamp() -> str:
    """A name for one run that sorts in the order the runs were started."""
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_parent(path: str) -> str:
    """Creates the directory a file is about to be written into, and returns the path unchanged."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    return path


@contextmanager
def replacing(path: str, mode: str = "w", **options) -> Iterator[IO]:
    """Opens a temporary file beside `path` for writing and puts it in place of `path` in one step when the block ends without an error.

    The new file is synced to the disk before the swap and the directory after it, so a reader of `path` sees either the old file or the whole new one, never a part written, even after the machine itself went down. When the block raises, the temporary file is removed and `path` is left as it was.
    The new file gets the permissions a plain open() would leave it with: those of the file it replaces, or the umask's for a new one.
    """
    ensure_parent(path)
    directory, name = os.path.split(os.path.abspath(path))
    if "b" not in mode:
        options.setdefault("encoding", "utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        temporary = os.path.join(directory, f".{name}.{os.urandom(6).hex()}.tmp")
        try:
            # Created with mode 0o666 so that the umask applies, as it does to open().
            descriptor = os.open(temporary, flags, 0o666)
            break
        except FileExistsError:
            continue
    try:
        with os.fdopen(descriptor, mode, **options) as handle:
            try:
                os.chmod(temporary, stat.S_IMODE(os.stat(path).st_mode))
            except FileNotFoundError:
                pass
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
    _sync_directory(directory)


def _sync_directory(directory: str) -> None:
    """Makes a rename in `directory` durable; skipped where a directory cannot be opened (Windows) or its filesystem refuses to sync one."""
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF):
            raise
    finally:
        os.close(descriptor)


def run_log_directory(tool: str) -> str:
    """A fresh directory for one launcher run's process logs, with `latest` pointed at it."""
    base = logs(tool)
    os.makedirs(base, exist_ok=True)
    name = stamp()
    # Creating the directory is the test for whether the name is free, so that two runs started in the same second take different names rather than one of them failing.
    suffix = 1
    while True:
        path = os.path.join(base, name if suffix == 1 else f"{name}-{suffix}")
        try:
            os.makedirs(path)
            break
        except FileExistsError:
            suffix += 1
    _point_latest(base, path)
    return path


def run_log_file(tool: str) -> str:
    """A fresh file for one Python run's log, with `latest` pointed at it."""
    base = logs(tool)
    os.makedirs(base, exist_ok=True)
    name = stamp()
    suffix = 1
    while True:
        path = os.path.join(base, f"{name}.log" if suffix == 1 else f"{name}-{suffix}.log")
        try:
            open(path, "x", encoding="utf-8").close()
            break
        except FileExistsError:
            suffix += 1
    _point_latest(base, path)
    return path


def _point_latest(base: str, target: str) -> None:
    link = os.path.join(base, "latest")
    temporary = link + ".new"
    try:
        if os.path.lexists(temporary):
            os.remove(temporary)
        os.symlink(os.path.basename(target), temporary)
        os.replace(temporary, link)
    except OSError:
        # A filesystem without symbolic links still gets its logs; only the shortcut is missing.
        pass


def configure_logging(tool: str, verbose: bool = False, fmt: str = "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                      log_file: Optional[str] = None) -> str:
    """Logs to standard error and to a file under local/logs/<tool>, and returns the file's path."""
    path = log_file or run_log_file(tool)
    ensure_parent(path)
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    written = logging.FileHandler(path, encoding="utf-8")
    written.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(console)
    root.addHandler(written)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    return path
