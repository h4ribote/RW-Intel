"""The game install and the command line that starts one game process from it.

The Linux release ships its own Java 8 runtime under `jvm-linux` and native libraries in the install root. Those libraries depend on each other by bare file name, so the dynamic linker has to be pointed at the install root through LD_LIBRARY_PATH as well as the JVM through java.library.path; the release's own start script does the same with `.` because it runs from the install root, which an instance does not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .. import paths

MAIN_CLASS = "com.corrodinggames.rts.java.Main"

#: Headless as far as the engine allows, and with mods off, because locally installed mods rewrite unit definitions and a run has to see the same units every time.
GAME_ARGUMENTS = ("-nodisplay", "-nosound", "-nomusic", "-nomods")

DEFAULT_HEAP = "800M"


@dataclass(frozen=True)
class GameInstall:
    """One unpacked Linux release of the game."""

    root: str

    @staticmethod
    def default() -> "GameInstall":
        return GameInstall(paths.game())

    @staticmethod
    def at(root: Optional[str]) -> "GameInstall":
        return GameInstall(os.path.abspath(root)) if root else GameInstall.default()

    @property
    def java(self) -> str:
        return os.path.join(self.root, "jvm-linux", "bin", "java")

    @property
    def classpath(self) -> str:
        return os.pathsep.join([os.path.join(self.root, "game-lib.jar"), os.path.join(self.root, "libs", "*")])

    def require(self) -> "GameInstall":
        if not os.path.isfile(os.path.join(self.root, "game-lib.jar")):
            raise FileNotFoundError(
                f"no game-lib.jar under {self.root}; unpack the Linux release into local/{paths.GAME_DIRECTORY} or set RWINTEL_GAME"
            )
        if not os.access(self.java, os.X_OK):
            raise FileNotFoundError(f"no executable bundled JVM at {self.java}; the unpacked release looks incomplete")
        return self


def agent_options(pairs: Mapping[str, object], extra: str = "") -> str:
    """The comma separated key=value string a javaagent receives, with anything the caller typed appended as it was typed."""
    text = ",".join(f"{key}={value}" for key, value in pairs.items())
    extra = extra.strip().strip(",")
    if extra:
        text = f"{text},{extra}" if text else extra
    return text


def command(install: GameInstall, agent_jar: str, options: str, heap: str = DEFAULT_HEAP) -> List[str]:
    """The argument vector of one game process under a javaagent."""
    return [
        install.java,
        f"-Xmx{heap}",
        "-Dfile.encoding=UTF-8",
        f"-Djava.library.path={install.root}",
        f"-javaagent:{os.path.abspath(agent_jar)}={options}",
        "-cp", install.classpath,
        MAIN_CLASS,
        *GAME_ARGUMENTS,
    ]


def environment(install: GameInstall, display: str, base: Optional[Mapping[str, str]] = None,
                extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment of one game process: the display it draws its window on and the linker path to the install's native libraries."""
    env = dict(os.environ if base is None else base)
    library_path = [install.root] + [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_path)
    env["DISPLAY"] = display
    if extra:
        env.update(extra)
    return env


def running_games(proc: str = "/proc") -> List[int]:
    """Process ids of every game process on the machine, whoever started it."""
    found: List[int] = []
    try:
        entries = os.listdir(proc)
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc, entry, "cmdline"), "rb") as handle:
                arguments: Sequence[bytes] = handle.read().split(b"\0")
        except OSError:
            continue
        if MAIN_CLASS.encode() in arguments:
            found.append(int(entry))
    return sorted(found)
