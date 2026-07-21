"""One period's board, sorted into the shapes the layers ask questions of.

Every layer reads the same observation but wants a different cut of it, and each cut is cheap: a few hundred units filtered a few ways is nothing beside a period. Building it once in one place means the layers agree on what "ours" means — in particular that it excludes whatever another commander is holding, which is the difference between planning with the army and planning with the army we can actually move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...wire import NO_SQUAD, Observation, RegionState, UnitState
from .catalogue import Catalogue
from .contracts import Role


@dataclass
class Sighting:
    """A unit as seen this period, with what its type is and what that makes it good for."""

    unit: UnitState
    kind: object
    role: Role

    @property
    def value(self) -> float:
        return float(self.kind.price) if self.kind is not None else 0.0


@dataclass
class WorldView:
    observation: Observation
    catalogue: Catalogue
    ours: List[Sighting] = field(default_factory=list)
    enemies: List[Sighting] = field(default_factory=list)
    buildings: List[Sighting] = field(default_factory=list)
    builders: List[Sighting] = field(default_factory=list)
    fighters: List[Sighting] = field(default_factory=list)
    #: Ours, able to fight or build, and in no squad. This is what the organisation layer forms and reinforces from.
    unassigned: List[Sighting] = field(default_factory=list)
    home: Optional[RegionState] = None

    @property
    def regions(self) -> List[RegionState]:
        return self.observation.regions

    def region(self, region_id: int) -> Optional[RegionState]:
        return next((r for r in self.observation.regions if r.id == region_id), None)

    def from_home(self) -> List[RegionState]:
        """Regions ordered outward from our own base, which is the order every layer names places in.

        Ordering egocentrically rather than by the map's own numbering is what lets a policy read a map it was not written against: the first is always home and the last always the far side, whatever the map. The wire keeps the map's numbering because home is not known until something has been built, so the translation happens here.
        """
        return sorted(self.observation.regions, key=lambda r: (r.distance_from_home, r.id))

    def value_of(self, units: List[int]) -> float:
        by_id = {s.unit.id: s for s in self.ours}
        return sum(by_id[u].value for u in units if u in by_id)

    def enemies_near(self, x: float, y: float, radius: float) -> List[Sighting]:
        return [e for e in self.enemies if math.hypot(e.unit.x - x, e.unit.y - y) <= radius]

    def contact(self, x: float, y: float, radius: float) -> Dict[Role, float]:
        """What has been run into around a point, by role and by worth. This is the only way anything above the fighting learns what the enemy is fielding once the fog is on."""
        found: Dict[Role, float] = {}
        for sighting in self.enemies_near(x, y, radius):
            found[sighting.role] = found.get(sighting.role, 0.0) + sighting.value
        return found


def build(observation: Observation, catalogue: Catalogue, home_id: Optional[int]) -> WorldView:
    view = WorldView(observation=observation, catalogue=catalogue)
    for unit in observation.unit_states:
        kind = catalogue.kind(unit.type_index)
        sighting = Sighting(unit=unit, kind=kind, role=catalogue.role(unit.type_index))
        if unit.hostile:
            view.enemies.append(sighting)
            continue
        view.ours.append(sighting)
        if sighting.role == Role.STRUCTURE:
            view.buildings.append(sighting)
            continue
        if sighting.role == Role.BUILDER:
            view.builders.append(sighting)
        else:
            view.fighters.append(sighting)
        if unit.squad == NO_SQUAD:
            view.unassigned.append(sighting)

    if home_id is not None:
        view.home = view.region(home_id)
    if view.home is None and observation.regions:
        view.home = min(observation.regions, key=lambda r: r.distance_from_home)
    return view


def home_region_id(observation: Observation) -> Optional[int]:
    """The region our own buildings sit in, which is the origin the others are ordered from. Taken from where the buildings actually are rather than from the starting position, because a base that has grown outward is where home is now."""
    ours = [u for u in observation.unit_states if not u.hostile]
    if not ours or not observation.regions:
        return None
    x = sum(u.x for u in ours) / len(ours)
    y = sum(u.y for u in ours) / len(ours)
    return min(observation.regions, key=lambda r: (r.x - x) ** 2 + (r.y - y) ** 2).id
