"""Sorting unit types by what they can do: the domain they move in, the functions they perform, and the one role they are counted under.

The test is what a type can do, never what it is called. A definition file may take over a built-in slot and rename it, and mods rename freely, so a rule that read names would classify a renamed tank as nothing at all. Everything used here reaches the control process in the catalogue the game sends at HELLO, read off the engine's own sample unit of each type: whether it can attack, its reach, its speed, its transport capacity and what it loads, whether it builds, and what its menu makes. Which building makes what is the menu too, not a definition file.

The thresholds are opening values. Which side of a line a type falls on decides which squads will accept it, so these are exactly the sort of number the design says to settle by measurement rather than by argument.
"""

from __future__ import annotations

from typing import Collection, Dict, Iterable, List, Optional, Set

from ...data import read_unit_catalog
from .contracts import DOCTRINES, DOMAIN_OF_MOVEMENT, Doctrine, Domain, Function, Role

#: Reach at which a unit is treated as a gun rather than as a line unit. The median armed mobile type reaches 200 world units and the longest 400, so this takes the top of that spread.
ARTILLERY_RANGE = 250.0

#: Speed, in world units a second, at which a unit on the ground is fast enough to raid with: clearly above the line tank's, so that a raid outruns what it meets.
FAST_SPEED = 75.0

#: The order the doctrines take a loose unit in when more than one would, which orders them by how much each wants a unit of that kind.
DOCTRINE_PREFERENCE = (Doctrine.ENGINEER, Doctrine.FLEET, Doctrine.AIRWING, Doctrine.RAID, Doctrine.GARRISON,
                       Doctrine.VANGUARD)


def domain_of(kind) -> Domain:
    if kind is None or kind.building:
        return Domain.STATIC
    return DOMAIN_OF_MOVEMENT.get(kind.movement, Domain.STATIC)


def functions_of(kind, makes_units: bool = False) -> Function:
    """Everything a type does. `makes_units` says whether its menu holds something that is not a building, which is what a producer is."""
    if kind is None:
        return Function(0)
    found = Function(0)
    domain = domain_of(kind)
    if kind.building:
        found |= Function.STRUCTURE
    if kind.builder:
        found |= Function.BUILDER
    if makes_units:
        found |= Function.PRODUCER
    if kind.transport:
        found |= Function.TRANSPORT
    if kind.armed and kind.mobile:
        found |= Function.COMBAT
        if kind.range >= ARTILLERY_RANGE:
            found |= Function.ARTILLERY
        if kind.hits_air and not kind.hits_land:
            found |= Function.ANTI_AIR
        if kind.speed >= FAST_SPEED and domain in (Domain.GROUND, Domain.AMPHIBIOUS):
            found |= Function.FAST
    if kind.mobile and not kind.armed and not kind.transport and not kind.builder:
        found |= Function.SCOUT
    return found


def role_of(kind, functions: Optional[Function] = None) -> Role:
    """Which role a unit type is counted under. The order the tests are tried in is the classification: a type that answers to more than one is put where it is scarcest, and a transport that can also fight is counted as what it fights as."""
    if kind is None:
        return Role.OTHER
    functions = functions_of(kind) if functions is None else functions
    if functions & Function.BUILDER:
        return Role.BUILDER
    if functions & Function.STRUCTURE:
        return Role.STRUCTURE
    if functions & Function.COMBAT:
        # Something that can only shoot upwards is worth nothing else, and is the whole reason a garrison wants two of them.
        if functions & Function.ANTI_AIR:
            return Role.ANTI_AIR
        if functions & Function.ARTILLERY:
            return Role.ARTILLERY
        if functions & Function.FAST:
            return Role.FAST
        return Role.ARMOUR
    if functions & Function.TRANSPORT:
        return Role.TRANSPORT
    return Role.OTHER


class Catalogue:
    """The type table with each entry's domain, functions and role, and the questions the layers ask of it."""

    def __init__(self, types: Iterable, assets=None) -> None:
        self._index(list(types))
        self.definitions = _definitions(assets)

    @classmethod
    def of_types(cls, types: Iterable) -> "Catalogue":
        """The type table alone, without the definitions the asset tree provides. Enough for everything the encodings read, which is what a recorded state is rebuilt with."""
        catalogue = cls.__new__(cls)
        catalogue._index(list(types))
        catalogue.definitions = {}
        return catalogue

    def _index(self, types: List) -> None:
        self.types = types
        self.domains: Dict[int, Domain] = {kind.index: domain_of(kind) for kind in types}
        self.functions: Dict[int, Function] = {}
        for kind in types:
            makes = any(not getattr(self.kind(i), "building", True) for i in getattr(kind, "menu", ()))
            self.functions[kind.index] = functions_of(kind, makes)
        self.roles: Dict[int, Role] = {kind.index: role_of(kind, self.functions[kind.index]) for kind in types}
        #: Buildings a builder places, which is what tells a factory from a command centre: both make units, and only one is put down by a builder.
        self.placeable: Set[int] = {index for kind in types if kind.builder for index in getattr(kind, "menu", ())}
        #: Buildings a builder places whose menu makes something that fights.
        self.factories: Set[int] = {kind.index for kind in types
                                    if kind.building and kind.index in self.placeable and self._makes_fighters(kind)}
        #: Buildings that make builders and are not placed by one: where the first builders come from.
        self.headquarters: Set[int] = {kind.index for kind in types
                                       if kind.building and kind.index not in self.placeable
                                       and any(getattr(self.kind(i), "builder", False) for i in getattr(kind, "menu", ()))}

    def _makes_fighters(self, kind) -> bool:
        return any(self.functions.get(index, Function(0)) & Function.COMBAT for index in getattr(kind, "menu", ()))

    def role(self, type_index: int) -> Role:
        return self.roles.get(type_index, Role.OTHER)

    def domain(self, type_index: int) -> Domain:
        return self.domains.get(type_index, Domain.STATIC)

    def has(self, type_index: int, function: Function) -> bool:
        return bool(self.functions.get(type_index, Function(0)) & function)

    def kind(self, type_index: int):
        return self.types[type_index] if 0 <= type_index < len(self.types) else None

    def by_lookup(self, lookup: str):
        for kind in self.types:
            if kind.lookup == lookup:
                return kind
        return None

    def cheapest(self, role: Role, tech: int = 99, producer=None, excluding: Collection[int] = ()):
        """The least expensive type filling a role at or below a technology level, which is what a build order asks for when it wants one more of something.

        A producer, a type, narrows it to what its menu makes. Without that the answer is whatever is cheapest anywhere in the registry -which is some creature from a scenario nothing on the board can build -and the factory sits idle while the credits pile up. `excluding` removes types the caller has found the producer does not make after all.
        """
        candidates = [k for k in self.types
                      if self.roles.get(k.index) == role and k.tech <= tech and k.price > 0
                      and k.index not in excluding and (producer is None or self.builds(producer, k))]
        return min(candidates, key=lambda k: k.price) if candidates else None

    def priciest(self, role: Role, tech: int = 99, producer=None, excluding: Collection[int] = (),
                 budget: float = float("inf")):
        """The most expensive type filling a role that fits the budget, under the same conditions as `cheapest`. Worth is price, so this is the most worth one unit can carry, which is what matters once the number of units rather than the credits is what runs out."""
        candidates = [k for k in self.types
                      if self.roles.get(k.index) == role and k.tech <= tech and 0 < k.price <= budget
                      and k.index not in excluding and (producer is None or self.builds(producer, k))]
        return max(candidates, key=lambda k: k.price) if candidates else None

    def builds(self, producer, kind) -> bool:
        """Whether a producer type makes this type at its first tier, as its menu says."""
        return kind.index in getattr(producer, "menu", ())

    def accepts(self, doctrine: Doctrine, type_index: int, domain: Optional[Domain] = None) -> bool:
        """Whether a squad of this doctrine will take this unit: the role has to be one it is built from, and the domain one it is raised in and, for a squad already formed, the squad's own."""
        if self.kind(type_index) is None:
            return False
        spec = DOCTRINES[doctrine]
        own = self.domain(type_index)
        if self.role(type_index) not in spec.establishment or own not in spec.domains:
            return False
        return domain is None or own == domain

    def doctrine_for(self, type_index: int) -> Optional[Doctrine]:
        """The doctrine a loose unit would go to if a squad were formed around it."""
        for doctrine in DOCTRINE_PREFERENCE:
            if self.accepts(doctrine, type_index):
                return doctrine
        return None

    def value(self, type_index: int) -> float:
        """What a unit of this type is worth. Price, which measurement showed stands in for durability well and for durability times damage nearly as well, and which has the further merit of being exact and in the same currency as the economy."""
        kind = self.kind(type_index)
        return float(kind.price) if kind is not None else 0.0


#: Definitions already read, by asset tree. The definition files do not change while the process runs, and every episode of every instance builds a catalogue.
_DEFINITIONS: Dict[str, Dict[str, object]] = {}


def _definitions(assets) -> Dict[str, object]:
    """Every definition by name, for the numbers a fight is decided by. Read once per asset tree and shared, since no caller changes what it is handed."""
    key = repr(assets)
    cached = _DEFINITIONS.get(key)
    if cached is None:
        cached = _DEFINITIONS[key] = read_unit_catalog(assets)
    return cached
