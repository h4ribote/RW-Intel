"""Checks the decoder against the engine's own account of the same replay.

While it plays a replay back, the engine writes each command it dispatches to its log: the issuing player's name and slot, how many live units it addresses, and then the order kind, the type being built, the special action, the stance and the switches it carries. Reading the same replay here and setting the two side by side, command by command, is how the decoder is shown to read every field the engine acts on the way the engine reads it.

A playback that stopped early has logged a prefix of the commands, and only that prefix is compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from .commands import STANCES, Command

_PREFIX = "updateGameFrame: "
_COMMAND = re.compile(r"Command: (?P<name>.*) \((?P<slot>-?\d+)\) count:(?P<count>\d+) id:(?P<id>\d+)$")


@dataclass
class Logged:
    """One command as the engine logged it."""

    slot: int
    name: str
    #: Units the command addressed that were still in the world when it ran, which can fall short of those it names.
    count: int
    sequence: int
    kind: Optional[str] = None
    build: Optional[str] = None
    action: Optional[str] = None
    stance: Optional[str] = None
    cancel: bool = False
    step_rate: Optional[float] = None
    system_action: int = 0


@dataclass
class Comparison:
    compared: int
    decoded: int
    logged: int
    #: Each disagreement as the command's position and what differed.
    differences: List[str] = field(default_factory=list)

    @property
    def agrees(self) -> bool:
        return not self.differences and self.compared > 0


def read_log(lines: Iterable[str]) -> List[Logged]:
    """The dispatched commands out of a game's log, in the order they ran."""
    logged: List[Logged] = []
    current: Optional[Logged] = None
    for line in lines:
        at = line.find(_PREFIX)
        if at < 0:
            continue
        text = line[at + len(_PREFIX):].rstrip("\n")
        found = _COMMAND.match(text)
        if found:
            current = Logged(slot=int(found.group("slot")), name=found.group("name"),
                             count=int(found.group("count")), sequence=int(found.group("id")))
            logged.append(current)
            continue
        if current is None:
            continue
        if text.startswith("Waypoint: "):
            current.kind = text[len("Waypoint: "):]
        elif text.startswith("Build Type: "):
            current.build = text[len("Build Type: "):]
        elif text.startswith("SpecialAction: "):
            current.action = text[len("SpecialAction: "):]
        elif text.startswith("SetAttackMode: "):
            current.stance = text[len("SetAttackMode: "):]
        elif text == "stopOrUndo is set":
            current.cancel = True
        elif text.startswith("changeStepRate:"):
            current.step_rate = float(text[len("changeStepRate:"):])
        elif text.startswith("systemAction_action:"):
            current.system_action = int(text[len("systemAction_action:"):])
        elif text == "------":
            current = None
    return logged


def compare(decoded: Sequence[Command], logged: Sequence[Logged]) -> Comparison:
    """Sets the decoded commands against the logged ones, position by position."""
    comparison = Comparison(compared=min(len(decoded), len(logged)), decoded=len(decoded), logged=len(logged))
    if len(logged) > len(decoded):
        comparison.differences.append(f"the engine ran {len(logged)} command(s) and the decoder read {len(decoded)}")
    for index, (ours, theirs) in enumerate(zip(decoded, logged)):
        where = f"command {index + 1} at frame {ours.frame}"
        if theirs.sequence != index + 1:
            comparison.differences.append(f"{where}: the engine numbers it {theirs.sequence}")
        pairs = [
            ("slot", ours.player, theirs.slot),
            ("order", ours.order_kind, theirs.kind),
            ("build type", ours.order.unit_type if ours.order is not None else None, theirs.build),
            ("action", ours.action, theirs.action),
            ("stance", STANCES[ours.stance] if ours.stance is not None else None, theirs.stance),
            ("cancel", ours.cancel, theirs.cancel),
            ("step rate", ours.step_rate, theirs.step_rate),
            ("system action", ours.system_action, theirs.system_action),
        ]
        for name, mine, engine in pairs:
            if mine != engine:
                comparison.differences.append(f"{where}: {name} decoded {mine!r}, engine {engine!r}")
        if theirs.count > len(ours.units):
            comparison.differences.append(
                f"{where}: the engine moved {theirs.count} unit(s) and the decoder read {len(ours.units)}")
    return comparison
