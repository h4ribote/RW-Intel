"""Reading an experiment agent's output: one line per unit whenever anything about it but its position changes, or it moves far enough to notice, with the script's own lines between them."""

from __future__ import annotations

import re
from typing import Iterable, Iterator, List, Optional, Set, Tuple

#: A tracking line: game seconds since the script began, the label, and the unit's state.
TRACK = re.compile(r"\[rw-lab\] track t=([\d.]+) (\S+) (.*)")
POSITION = re.compile(r"\((-?\d+),(-?\d+)\)")
HEALTH = re.compile(r"hp=-?\d+/")

#: How far a unit has to move, in world units summed over both axes, before its position alone earns a line.
MOVED = 150


def timeline(lines: Iterable[str], labels: Optional[Set[str]] = None) -> Iterator[str]:
    """The condensed record. Tracking lines are kept when the unit's state other than its position and health changed, or it moved more than MOVED since its last kept line; every other line the agent wrote is kept as it is, except the catalogue and the trees `near` lists."""
    last: dict = {}
    for line in lines:
        line = line.rstrip("\n")
        if not line.startswith("[rw-lab]"):
            continue
        match = TRACK.match(line)
        if match is None:
            if "catalog:" in line or (" near: " in line and " tree " in line):
                continue
            yield "    " + line[len("[rw-lab] "):]
            continue
        at, label, state = float(match.group(1)), match.group(2), match.group(3)
        if labels and label not in labels:
            continue
        position = POSITION.search(state)
        key = HEALTH.sub("hp=", POSITION.sub("", state))
        where: Optional[Tuple[int, int]] = (int(position.group(1)), int(position.group(2))) if position else None
        previous = last.get(label)
        moved = (previous is None or where is None or previous[1] is None
                 or abs(where[0] - previous[1][0]) + abs(where[1] - previous[1][1]) > MOVED)
        if previous is None or previous[0] != key or moved:
            last[label] = (key, where)
            yield f"{at:6.1f} {label:6s} {state}"


def read(path: str, labels: Optional[List[str]] = None) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return list(timeline(handle, set(labels or ())))
