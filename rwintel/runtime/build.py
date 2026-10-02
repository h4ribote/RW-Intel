"""Building the javaagents: the control agent, the probe agent and the experiment agent.

The game's bundled runtime is a Java 8 JRE with neither javac nor jar, so the agents are built with a JDK from the system and compiled with `--release 8`, which checks them against the Java 8 API as well as emitting Java 8 class files. Any JDK from 9 on can do that; JAVA_HOME is used when set, the PATH otherwise.

Every agent carries the frame layer in `frame/`, which implements Slick's renderer interface and so compiles against the install's Slick jar. The experiment agent also carries the control agent's sources, for its engine access.

A launcher builds an agent whose jar is missing or older than its sources before starting anything, so a run never carries an agent that no longer matches the code beside it.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

from .. import paths
from .game import GameInstall

JAVA_RELEASE = "8"

#: Sources every agent is built with, beside its own.
FRAME_DIRECTORY = os.path.join(paths.REPOSITORY, "frame")

#: Jars of the install the agents are compiled against, relative to the install root.
COMPILE_CLASSPATH = (os.path.join("libs", "slick.jar"),)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    name: str
    directory: str
    sources: str
    jar_name: str
    shared: Tuple[str, ...] = ()

    @property
    def jar(self) -> str:
        return os.path.join(self.directory, self.jar_name)

    @property
    def manifest(self) -> str:
        return os.path.join(self.directory, "manifest.txt")

    @property
    def classes(self) -> str:
        return os.path.join(self.directory, "classes")

    def source_files(self) -> List[str]:
        found = glob.glob(os.path.join(self.directory, self.sources))
        for directory in self.shared:
            found += glob.glob(os.path.join(directory, "*.java"))
        return sorted(found)


#: The control agent's sources, which the experiment agent is built with as well for its engine access.
AGENT_DIRECTORY = os.path.join(paths.REPOSITORY, "agent")

TARGETS = {
    "agent": Target("agent", AGENT_DIRECTORY, "*.java", "rwagent.jar", (FRAME_DIRECTORY,)),
    "probe": Target("probe", os.path.join(paths.REPOSITORY, "tools", "probe-agent"), "*.java", "rwprobe.jar", (FRAME_DIRECTORY,)),
    "lab": Target("lab", os.path.join(paths.REPOSITORY, "tools", "lab-agent"), "*.java", "rwlab.jar",
                  (FRAME_DIRECTORY, AGENT_DIRECTORY)),
}


def classpath(install: GameInstall) -> str:
    """The compile classpath the agents need from this install, refused by name when a jar is missing."""
    jars = [os.path.join(install.root, jar) for jar in COMPILE_CLASSPATH]
    missing = [jar for jar in jars if not os.path.isfile(jar)]
    if missing:
        raise BuildError(f"the agents compile against {', '.join(missing)}, which the install does not have")
    return os.pathsep.join(jars)


def find_tool(name: str) -> str:
    home = os.environ.get("JAVA_HOME")
    if home:
        candidate = os.path.join(home, "bin", name)
        if os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise BuildError(f"no {name} on the PATH or under JAVA_HOME; install a JDK (for example openjdk-17-jdk-headless), the game's bundled runtime has none")


def stale(target: Target) -> bool:
    if not os.path.isfile(target.jar):
        return True
    built = os.path.getmtime(target.jar)
    inputs = target.source_files() + [target.manifest]
    return any(os.path.getmtime(path) > built for path in inputs if os.path.exists(path))


def build(target: Target, install: GameInstall) -> str:
    javac, jar = find_tool("javac"), find_tool("jar")
    sources = target.source_files()
    if not sources:
        raise BuildError(f"no sources matching {target.sources} in {target.directory}")
    compile_classpath = classpath(install)
    if os.path.isdir(target.classes):
        shutil.rmtree(target.classes)
    os.makedirs(target.classes)
    compiled = subprocess.run([javac, "--release", JAVA_RELEASE, "-Xlint:-options", "-cp", compile_classpath,
                               "-d", target.classes, *sources],
                              capture_output=True, text=True)
    if compiled.returncode != 0:
        raise BuildError(f"compiling {target.name} failed:\n{compiled.stdout}{compiled.stderr}")
    # Written beside the jar and renamed over it, so a game process still running on the previous jar keeps reading the file it opened.
    handle, temporary = tempfile.mkstemp(prefix=f".{target.jar_name}.", suffix=".tmp", dir=target.directory)
    os.close(handle)
    os.remove(temporary)
    try:
        packed = subprocess.run([jar, "--create", "--file", temporary, "--manifest", target.manifest,
                                 "-C", target.classes, "."], capture_output=True, text=True)
        if packed.returncode != 0:
            raise BuildError(f"packaging {target.name} failed:\n{packed.stdout}{packed.stderr}")
        os.replace(temporary, target.jar)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target.jar


def ensure(target: Target, install: GameInstall) -> str:
    """The target's jar, built first if it is missing or older than its sources."""
    return build(target, install) if stale(target) else target.jar
