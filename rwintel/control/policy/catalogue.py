"""Sorting unit types into the roles the doctrines are written in.

The test is what a type can do, never what it is called. A definition file may take over a built-in slot and rename it, and mods rename freely, so a rule that read names would classify a renamed tank as nothing at all. Everything used here reaches the control process in the catalogue the game sends at HELLO, and every value in it came out of the running engine rather than out of a file that might not be the one loaded.

The thresholds are opening values. Which side of a line a type falls on decides which squads will accept it, so these are exactly the sort of number the design says to settle by measurement rather than by argument.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from ...data import read_unit_catalog
from .contracts import Doctrine, DOCTRINES, Role

#: Reach at which a unit is treated as a gun rather than as a line unit. The median armed mobile type reaches 200 world units and the longest 400, so this takes the top of that spread.
ARTILLERY_RANGE = 250.0

#: Movement types that make a unit fast enough to raid with, which is what the raiding doctrine is built out of.
FAST_MOVEMENT = frozenset({"AIR", "HOVER"})


def role_of(kind) -> Role:
    """Which role a unit type fills. The order the tests are tried in is the classification: a type that answers to more than one is put where it is scarcest."""
    if kind is None:
        return Role.OTHER
    if kind.builder:
        return Role.BUILDER
    if kind.building:
        return Role.STRUCTURE
    if not kind.armed or not kind.mobile:
        return Role.OTHER
    # Something that can only shoot upwards is worth nothing else, and is the whole reason a garrison wants two of them.
    if kind.hits_air and not kind.hits_land:
        return Role.ANTI_AIR
    if kind.range >= ARTILLERY_RANGE:
        return Role.ARTILLERY
    if kind.movement in FAST_MOVEMENT:
        return Role.FAST
    return Role.ARMOUR


class Catalogue:
    """The type table with a role attached to each entry, and the questions the layers ask of it."""

    def __init__(self, types: Iterable, assets=None) -> None:
        self.types: List = list(types)
        self.roles: Dict[int, Role] = {kind.index: role_of(kind) for kind in self.types}
        self.built_from: Dict[str, Set[str]] = _build_links(assets)

    def role(self, type_index: int) -> Role:
        return self.roles.get(type_index, Role.OTHER)

    def kind(self, type_index: int):
        return self.types[type_index] if 0 <= type_index < len(self.types) else None

    def by_lookup(self, lookup: str):
        for kind in self.types:
            if kind.lookup == lookup:
                return kind
        return None

    def cheapest(self, role: Role, tech: int = 99, producer: Optional[str] = None):
        """The least expensive type filling a role at or below a technology level, which is what a build order asks for when it wants one more of something.

        A producer narrows it to what that building can actually turn out. Without that the answer is whatever is cheapest anywhere in the registry — which is some creature from a scenario nothing on the board can build — and the factory sits idle while the credits pile up.
        """
        candidates = [k for k in self.types
                      if self.roles.get(k.index) == role and k.tech <= tech and k.price > 0
                      and (producer is None or self.builds(producer, k))]
        return min(candidates, key=lambda k: k.price) if candidates else None

    def builds(self, producer: str, kind) -> bool:
        """Whether a producer can make this type.

        The link is read from the definition files, since the type interface the agent reports through carries what a type is and not what makes it. It is read as an exclusion rather than a permission: a definition that names its makers is taken at its word, and one that names none is a type the standard chain produces. The core units — the tank and everything alongside it — have no definition file naming a factory, because the factory that makes them is itself code rather than a definition; whitelisting would throw away exactly the units an opening is built on, while excluding what names a different maker still keeps the factory from being asked for the creatures out of a scenario.
        """
        makers = self.built_from.get(kind.lookup) or self.built_from.get(kind.name)
        return not makers or producer in makers

    def accepts(self, doctrine: Doctrine, type_index: int) -> bool:
        """Whether a squad of this doctrine will take this unit: the role has to be one it is built from, and the movement type one it keeps to, so that a squad moves as one thing."""
        kind = self.kind(type_index)
        if kind is None:
            return False
        spec = DOCTRINES[doctrine]
        return self.role(type_index) in spec.establishment and kind.movement in spec.movement

    def doctrine_for(self, type_index: int) -> Optional[Doctrine]:
        """The doctrine a loose unit would go to if a squad were formed around it. The first that will take it, which orders the doctrines by how much each wants a unit of that kind."""
        for doctrine in (Doctrine.ENGINEER, Doctrine.RAID, Doctrine.GARRISON, Doctrine.VANGUARD):
            if self.accepts(doctrine, type_index):
                return doctrine
        return None

    def value(self, type_index: int) -> float:
        """What a unit of this type is worth. Price, which measurement showed stands in for durability well and for durability times damage nearly as well, and which has the further merit of being exact and in the same currency as the economy."""
        kind = self.kind(type_index)
        return float(kind.price) if kind is not None else 0.0


def _build_links(assets) -> Dict[str, Set[str]]:
    """Which buildings can produce each type, by name.

    Read from the definition files rather than from the running game, because the type interface the agent reports through has no such link on it. These are the same files the engine loads, so the answer is the engine's own; a type with no definition file is a building or the builder, and nothing produces those from a factory anyway.
    """
    links: Dict[str, Set[str]] = {}
    for name, definition in read_unit_catalog(assets).items():
        if definition.built_from:
            links[name] = set(definition.built_from)
    return links
