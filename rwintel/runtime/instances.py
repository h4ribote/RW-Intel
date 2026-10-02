"""Working directories for running several game processes side by side.

The game writes preferences.ini, saves, replays and its cache relative to its current directory, so every process needs a directory of its own. The rest of the install is only read, so an instance links to it instead of copying it: the asset trees are symbolic links to the install, and the native libraries and jars are found through the command line and LD_LIBRARY_PATH (see game.py). An instance therefore costs no meaningful disk space.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List

from .. import paths
from .game import GameInstall

#: Trees the game only reads, shared with the install.
LINKED = ("assets", "font", "res", "mods")

#: Trees the game writes, one per instance.
OWN = ("saves", "cache", "replays")

#: The settings file the game reads from and writes to its working directory.
PREFERENCES = "preferences.ini"

#: The game uploads a report of every uncaught exception to its developer while this setting is on, and it is on by default.
CRASH_REPORT_SETTING = "sendReports"


@dataclass
class Prepared:
    created: int
    reused: int
    relinked: int
    root: str


def directory(index: int) -> str:
    return paths.instance(index)


def _remove(path: str) -> None:
    # The links go first and on their own, so that removing an instance can never reach into the install they point at.
    for name in LINKED:
        link = os.path.join(path, name)
        if os.path.islink(link):
            os.unlink(link)
    shutil.rmtree(path)


def _link(path: str, install: GameInstall) -> bool:
    """Points the shared trees of one instance at the install, and says whether anything had to change."""
    changed = False
    for name in LINKED:
        target = os.path.join(install.root, name)
        link = os.path.join(path, name)
        if not os.path.isdir(target):
            continue
        if os.path.islink(link):
            if os.readlink(link) == target:
                continue
            os.unlink(link)
        elif os.path.exists(link):
            raise FileExistsError(f"{link} is a real {('directory' if os.path.isdir(link) else 'file')}, not a link to the install; recreate the instance with --force")
        os.symlink(target, link)
        changed = True
    for name in OWN:
        os.makedirs(os.path.join(path, name), exist_ok=True)
    return changed


def disable_crash_reports(path: str) -> None:
    """Turns the crash report setting off in an instance's settings file, writing a file with only that setting when the game has not written one yet."""
    preferences = os.path.join(path, PREFERENCES)
    wanted = f"{CRASH_REPORT_SETTING}:false"
    try:
        with open(preferences, "r", encoding="utf-8", newline="") as handle:
            lines = handle.read().splitlines()
    except FileNotFoundError:
        lines = ["[settings]"]
    prefix = f"{CRASH_REPORT_SETTING}:"
    if [line for line in lines if line.startswith(prefix)] == [wanted]:
        return
    kept = [line for line in lines if not line.startswith(prefix)]
    with open(preferences, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(kept + [wanted]) + "\n")


def prepare(count: int, install: GameInstall, force: bool = False, start: int = 0) -> Prepared:
    """Makes sure instances start .. start+count-1 exist, point at this install, and do not upload crash reports."""
    install.require()
    root = paths.instances()
    os.makedirs(root, exist_ok=True)
    created = reused = relinked = 0
    for index in range(start, start + count):
        path = directory(index)
        if os.path.isdir(path) and force:
            _remove(path)
        if os.path.isdir(path):
            reused += 1
            if _link(path, install):
                relinked += 1
        else:
            os.makedirs(path)
            _link(path, install)
            created += 1
        disable_crash_reports(path)
    return Prepared(created, reused, relinked, root)


def require(indices: List[int]) -> List[str]:
    """The directories of these instances, which have to exist already."""
    missing: List[str] = []
    found: List[str] = []
    for index in indices:
        path = directory(index)
        (found if os.path.isdir(path) else missing).append(path)
    if missing:
        raise FileNotFoundError(f"instance directory missing: {', '.join(missing)}; create them with `python -m rwintel.runtime instances --count N`")
    return found


def describe(prepared: Prepared) -> str:
    return (f"instances: {prepared.created} created, {prepared.reused} reused"
            + (f" ({prepared.relinked} relinked to this install)" if prepared.relinked else "")
            + f", root {prepared.root}")
