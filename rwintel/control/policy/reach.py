"""Which regions a unit can get to under its own power, and where a transport can pick it up and set it down.

Built once an episode from the terrain frame (`wire.terrain.Passage`), which is the path finder's own connected components per movement type, and the region table. A region is reached by a movement type when a tile of the region's own component for that type is reachable: the component is the one at the tile of that type nearest the region's centre. Air reaches everything.

A transport sets its passengers down only on land, so a region's landing point for a transport is the land tile nearest the region's centre that the transport can stand on and that lies in the region's own land component. A pick-up point is the tile nearest the passengers that both they and the transport can reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...wire.terrain import AIR, TILE, Passage

#: The land movement types, of which the narrowest is what a landing has to be walkable by.
LAND = "LAND"
HOVER = "HOVER"
WATER = "WATER"

#: How far from a region's centre, in tiles, the tile a movement type reaches it by may lie. Regions are agglomerated at 400 world units, so this covers a region's own ground and the water off its shore.
ANCHOR_TILES = 30

#: How far from a region's centre, in world units, open water has to come for the region to count as coastal.
COAST_REACH = 600.0

Point = Tuple[float, float]


@dataclass
class RegionReach:
    """One region as each movement type reaches it."""

    region: int
    x: float
    y: float
    #: Per movement type, the component that reaches the region, -1 when none comes within ANCHOR_TILES of its centre.
    components: Dict[str, int] = field(default_factory=dict)
    #: Per movement type, the component of the region's centre as the game side resolves a contract's target point (`Passage.component_at` at its own reach), -1 when none lies that close: what a unit has to stand in to be sent to the region under its own power.
    targets: Dict[str, int] = field(default_factory=dict)
    #: Per transport movement type, where a transport of that type sets passengers down in the region, or None.
    landings: Dict[str, Optional[Point]] = field(default_factory=dict)
    #: Whether open water comes within COAST_REACH of the centre.
    coastal: bool = False


class Reach:
    """The reachability of every region of one map."""

    def __init__(self, passage: Passage, regions: Iterable) -> None:
        self.passage = passage
        self._rows, self._columns = np.mgrid[0:passage.height, 0:passage.width]
        #: The answers of `approaches`, by movement type, component, point tile and tiles, which is all an answer depends on.
        self._approaches: Dict[Tuple[str, int, Tuple[int, int], int], bool] = {}
        self.regions: Dict[int, RegionReach] = {}
        for region in regions:
            self.regions[region.id] = self._region(region.id, float(region.x), float(region.y))

    # Positions.

    def component(self, movement: str, x: float, y: float) -> int:
        """The component of a world position for a movement type, as `Passage.component_at` finds it; 0 for air."""
        return self.passage.component_at(movement, x, y)

    def walkable(self, movement: str, x: float, y: float, region: int) -> bool:
        """Whether a unit of the movement type standing at the position can get to the region's centre under its own power, by the test the game side applies before it orders a contract out: a ship cannot be sent to a region whose centre stands inland, however close the water comes."""
        if movement == AIR:
            return True
        target = self.regions.get(region)
        if target is None:
            return False
        wanted = target.targets.get(movement, -1)
        return wanted >= 0 and self.component(movement, x, y) == wanted

    def reachable(self, movement: str, start: Point, end: Point) -> bool:
        return movement == AIR or self.passage.reachable(movement, start, end)

    def approaches(self, movement: str, start: Point, point: Point, tiles: int = ANCHOR_TILES) -> bool:
        """Whether a unit of the movement type standing at `start` can get within `tiles` of a point under its own power: a ship to the water off a landing on the shore, as a region is reached for landings.

        Answered once per movement type, component, point tile and distance for the map, from the tiles within `tiles` of the point only, so that asking for every member of every fleet about every transport costs a lookup each.
        """
        if movement == AIR:
            return True
        grid = self.passage.labels.get(movement)
        component = self.component(movement, *start)
        if grid is None or component < 0:
            return False
        key = (movement, component, self.passage.tile(*point), tiles)
        known = self._approaches.get(key)
        if known is None:
            known = self._approaches[key] = self._component_near(grid, component, key[2], tiles)
        return known

    def landing(self, region: int, transport: str) -> Optional[Point]:
        target = self.regions.get(region)
        return target.landings.get(transport) if target is not None else None

    def pickup(self, passenger: str, at: Point, transport: str, transport_at: Point) -> Optional[Point]:
        """The tile nearest the passengers that both they and the transport can reach, or None when there is none."""
        mask = self._component_mask(passenger, at)
        if mask is None:
            return None
        if transport != AIR:
            reach = self._component_mask(transport, transport_at)
            if reach is None:
                return None
            mask = mask & reach
        return self._nearest(mask, at)

    def ship_site(self, at: Point, ship_at: Point, reach: float) -> Optional[Point]:
        """The water tile nearest a point, within a ship's reach of it, that a ship standing at `ship_at` can get to; None when there is none. What a building ship places from."""
        mask = self._component_mask(WATER, ship_at)
        if mask is None:
            return None
        site = self._nearest(mask, at)
        if site is None or math.hypot(site[0] - at[0], site[1] - at[1]) > reach:
            return None
        return site

    # Building the table.

    def _region(self, region: int, x: float, y: float) -> RegionReach:
        row = RegionReach(region=region, x=x, y=y)
        anchors: Dict[str, Optional[Point]] = {}
        for movement, grid in self.passage.labels.items():
            anchor = self._nearest(grid >= 0, (x, y), ANCHOR_TILES)
            anchors[movement] = anchor
            row.components[movement] = self.passage.component_at(movement, *anchor, reach=0) if anchor else -1
            row.targets[movement] = self.passage.component_at(movement, x, y)
        row.components[AIR] = 0
        row.targets[AIR] = 0
        land = self.passage.labels.get(LAND)
        land_component = row.components.get(LAND, -1)
        if land is not None and land_component >= 0:
            ground = land == land_component
            hover = self.passage.labels.get(HOVER)
            if hover is not None:
                row.landings[HOVER] = self._nearest(ground & (hover >= 0), (x, y), ANCHOR_TILES)
            row.landings[AIR] = self._nearest(ground, (x, y), ANCHOR_TILES)
        water = self.passage.labels.get(WATER)
        if water is not None:
            coast = self._nearest(water >= 0, (x, y))
            row.coastal = coast is not None and math.hypot(coast[0] - x, coast[1] - y) <= COAST_REACH
        return row

    def _component_mask(self, movement: str, at: Point) -> Optional[np.ndarray]:
        if movement == AIR:
            return np.ones((self.passage.height, self.passage.width), dtype=bool)
        grid = self.passage.labels.get(movement)
        component = self.component(movement, *at)
        if grid is None or component < 0:
            return None
        return grid == component

    @staticmethod
    def _component_near(grid: np.ndarray, component: int, tile: Tuple[int, int], tiles: int) -> bool:
        """Whether a tile of the component lies within `tiles` tiles of the tile, by the distance `_nearest` measures, looking only at the square around it."""
        column, row = tile
        top, left = max(0, row - tiles), max(0, column - tiles)
        window = grid[top:row + tiles + 1, left:column + tiles + 1] == component
        if not window.any():
            return False
        rows, columns = np.nonzero(window)
        return bool(((rows + top - row) ** 2 + (columns + left - column) ** 2).min() <= tiles * tiles)

    def _nearest(self, mask: np.ndarray, at: Point, within: Optional[int] = None) -> Optional[Point]:
        """The centre of the masked tile nearest a world position, in world units; None when no masked tile lies within `within` tiles."""
        if not mask.any():
            return None
        column, row = self.passage.tile(*at)
        distance = (self._rows - row) ** 2 + (self._columns - column) ** 2
        distance = np.where(mask, distance, np.iinfo(distance.dtype).max)
        index = int(np.argmin(distance))
        best_row, best_column = divmod(index, self.passage.width)
        if within is not None and distance[best_row, best_column] > within * within:
            return None
        return ((best_column + 0.5) * TILE, (best_row + 0.5) * TILE)


def narrowest(movements: Sequence[str]) -> str:
    """The narrowest of a set of movement types, as the game side reports a squad's passage: the one every other crosses all the ground of, or the empty name when there is none."""
    best: Optional[str] = None
    for movement in movements:
        if best is None or within(movement, best):
            best = movement
    if best is None or any(not within(best, other) for other in movements):
        return ""
    return best


_WIDER = {
    "LAND": {"LAND", "OVER_CLIFF", "HOVER", "OVER_CLIFF_WATER", AIR},
    "OVER_CLIFF": {"OVER_CLIFF", "OVER_CLIFF_WATER", AIR},
    "HOVER": {"HOVER", "OVER_CLIFF_WATER", AIR},
    "WATER": {"WATER", "HOVER", "OVER_CLIFF_WATER", AIR},
    "OVER_CLIFF_WATER": {"OVER_CLIFF_WATER", AIR},
    AIR: {AIR},
}


def within(narrow: str, wide: str) -> bool:
    """Whether every tile the first movement type crosses the second crosses too, as `agent/Passage.java` has it."""
    return wide in _WIDER.get(narrow, {narrow})


def members_reach(reach: Reach, members: Sequence[Tuple[str, float, float]], region: int) -> bool:
    """Whether every member, given as its movement type and position, can walk to the region."""
    return bool(members) and all(reach.walkable(movement, x, y, region) for movement, x, y in members)


def regions_walkable(reach: Reach, members: Sequence[Tuple[str, float, float]]) -> List[int]:
    return [region for region in reach.regions if members_reach(reach, members, region)]
