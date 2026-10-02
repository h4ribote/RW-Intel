"""The ground of a skirmish map: which tiles each movement type can cross, which connected components that makes, and what each side starts with in them.

Passability follows the engine's path finder, read off the map's tile properties by the table below (`ALLOWS`). Only the layers the engine reads count (`ENGINE_LAYERS`); the editor's helper layers (`set` and its spellings) carry properties the engine ignores. Buildings standing on the map block their footprint as well, which this reading does not see.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np

from .assets import AssetPaths
from .maps import _GID_MASK, _decode_layer, _tile_property_table

#: The engine's movement types that move over the ground, in its own order, and air.
LAND, OVER_CLIFF, HOVER, WATER_MOVE, OVER_CLIFF_WATER, AIR = "LAND", "OVER_CLIFF", "HOVER", "WATER", "OVER_CLIFF_WATER", "AIR"
MOVEMENTS = (LAND, OVER_CLIFF, HOVER, WATER_MOVE, OVER_CLIFF_WATER, AIR)

#: What a tile with none of the properties below lets across: everything but boats.
OPEN = frozenset({LAND, OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR})

#: Per tile property, the movement types that may cross a tile carrying it; a tile is crossable by a movement type only when every property on it allows it. A property not listed changes nothing.
ALLOWS: Dict[str, FrozenSet[str]] = {
    "water": frozenset({HOVER, WATER_MOVE, OVER_CLIFF_WATER, AIR}),
    "cliff-soft": frozenset({OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR}),
    "cliff": frozenset({OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR}),
    "large-cliff": frozenset({OVER_CLIFF, OVER_CLIFF_WATER, AIR}),
    "trees": frozenset({OVER_CLIFF, OVER_CLIFF_WATER, AIR}),
    "res_pool": frozenset({OVER_CLIFF, HOVER, OVER_CLIFF_WATER, AIR}),
    "lava": frozenset({AIR}),
    "lava-cliff": frozenset({AIR}),
    "large-rock": frozenset({AIR}),
}

#: The layers whose tile properties the engine reads.
ENGINE_LAYERS = ("Ground", "Items")

#: The kinds a tile is drawn as, from the properties it carries, the first that applies.
WATER, LAVA, ROCK, LARGE_CLIFF, SOFT_CLIFF, GROUND = "water", "lava", "rock", "large-cliff", "cliff-soft", "land"
KINDS = (WATER, LAVA, ROCK, LARGE_CLIFF, SOFT_CLIFF, GROUND)
_KIND_OF = ((LAVA, ("lava", "lava-cliff")), (WATER, ("water",)), (ROCK, ("large-rock",)),
            (LARGE_CLIFF, ("large-cliff", "trees")), (SOFT_CLIFF, ("cliff-soft", "cliff")))

#: What a resource point is called among the marks.
RESOURCE = "resource"

#: Colours of the rendered map, as RGB: the ground by kind, then the marks. A placed unit not named here is drawn in `OTHER_UNIT`.
COLOURS: Dict[str, Tuple[int, int, int]] = {
    WATER: (40, 90, 200), LAVA: (210, 70, 20), ROCK: (110, 100, 90), LARGE_CLIFF: (60, 60, 60),
    SOFT_CLIFF: (150, 120, 80), GROUND: (90, 160, 70),
    RESOURCE: (255, 230, 0), "commandCenter": (230, 30, 30), "builder": (255, 140, 0),
    "hovercraft": (255, 0, 255), "seaFactory": (0, 230, 230),
}
OTHER_UNIT = (255, 255, 255)


@dataclass
class Mark:
    """A resource point or a unit the map places at the start, at its tile."""

    kind: str
    cell: Tuple[int, int]
    #: The side a placed unit belongs to, as the map numbers it; empty for a resource point.
    team: str = ""


@dataclass
class Terrain:
    """A map tile by tile, row by row from the top left: what each tile is drawn as, which movement types cross it, and the marks."""

    name: str
    width: int
    height: int
    tile_size: int
    kinds: List[str]
    #: Per tile, the movement types that may cross it.
    allows: List[FrozenSet[str]]
    marks: List[Mark] = field(default_factory=list)

    def kind_at(self, cell: Tuple[int, int]) -> str:
        return self.kinds[cell[1] * self.width + cell[0]]

    def share(self, kind: str) -> float:
        return self.kinds.count(kind) / max(1, len(self.kinds))

    def passable(self, movement: str) -> np.ndarray:
        """Per tile, as rows, whether this movement type may cross it."""
        return np.asarray([movement in allowed for allowed in self.allows], dtype=bool).reshape(self.height, self.width)


@dataclass
class Component:
    """Tiles one movement type can cross, joined edge to edge, numbered from the largest."""

    index: int
    tiles: int
    x: Tuple[int, int]
    y: Tuple[int, int]
    resources: int = 0
    #: The units the map places in this component, as (unit, team).
    units: List[Tuple[str, str]] = field(default_factory=list)


def allowed_by(properties) -> FrozenSet[str]:
    """The movement types that may cross a tile carrying these properties."""
    listed = [ALLOWS[name] for name in properties if name in ALLOWS]
    if not listed:
        return OPEN
    allowed = frozenset(MOVEMENTS)
    for movements in listed:
        allowed &= movements
    return allowed


def _kind(properties) -> str:
    for kind, names in _KIND_OF:
        if any(name in properties for name in names):
            return kind
    return GROUND


def read_terrain(path: str, paths: Optional[AssetPaths] = None) -> Terrain:
    """The properties of every tile on the layers the engine reads, with every resource point and placed unit of the map. Neutral decorations, which the map places with no side (trees), are left out of the marks."""
    paths = paths or AssetPaths.default()
    root = ElementTree.parse(path).getroot()
    properties = _tile_property_table(root, paths, os.path.dirname(os.path.abspath(path)))
    width, height = int(root.get("width")), int(root.get("height"))
    carried: List[set] = [set() for _ in range(width * height)]
    marks: List[Mark] = []
    for layer in root.findall("layer"):
        read = layer.get("name") in ENGINE_LAYERS
        for index, gid in enumerate(_decode_layer(layer)):
            values = properties.get(gid & _GID_MASK)
            if not values:
                continue
            cell = (index % width, index // width)
            if "res_pool" in values:
                marks.append(Mark(RESOURCE, cell))
            elif values.get("unit"):
                if values.get("team", "none") != "none":
                    marks.append(Mark(values["unit"], cell, values["team"]))
                continue
            if read:
                carried[index].update(name for name in values if name != "team")
    marks.sort(key=lambda mark: (mark.kind, mark.cell))
    return Terrain(name=os.path.splitext(os.path.basename(path))[0], width=width, height=height,
                   tile_size=int(root.get("tilewidth")), kinds=[_kind(names) for names in carried],
                   allows=[allowed_by(names) for names in carried], marks=marks)


def components(terrain: Terrain, movement: str = LAND) -> Tuple[List[Component], np.ndarray]:
    """Every connected component of the tiles this movement type crosses, largest first, and per tile the index of the component it belongs to (-1 where it cannot go)."""
    width, height = terrain.width, terrain.height
    blocked = [movement not in allowed for allowed in terrain.allows]
    found = np.full(width * height, -1, dtype=np.int64)
    sizes: List[int] = []
    for start in range(width * height):
        if found[start] >= 0 or blocked[start]:
            continue
        number = len(sizes)
        found[start] = number
        queue, size = deque([start]), 0
        while queue:
            at = queue.popleft()
            size += 1
            x, y = at % width, at // width
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    other = ny * width + nx
                    if found[other] < 0 and not blocked[other]:
                        found[other] = number
                        queue.append(other)
        sizes.append(size)
    order = sorted(range(len(sizes)), key=lambda number: (-sizes[number], number))
    # One slot more than there are components, so that the -1 of a blocked tile indexes that last slot and stays -1.
    rank = np.full(len(sizes) + 1, -1, dtype=np.int64)
    for position, number in enumerate(order):
        rank[number] = position
    labels = rank[found]
    found_components = []
    for position, number in enumerate(order):
        cells = np.flatnonzero(labels == position)
        xs, ys = cells % width, cells // width
        found_components.append(Component(index=position, tiles=sizes[number], x=(int(xs.min()), int(xs.max())),
                                          y=(int(ys.min()), int(ys.max()))))
    grid = labels.reshape(height, width)
    for mark in terrain.marks:
        component = nearest_component(grid, mark.cell)
        if component < 0:
            continue
        if mark.kind == RESOURCE:
            found_components[component].resources += 1
        else:
            found_components[component].units.append((mark.kind, mark.team))
    for component in found_components:
        component.units.sort()
    return found_components, grid


def nearest_component(labels: np.ndarray, cell: Tuple[int, int], reach: int = 2) -> int:
    """The component a mark belongs to: the one under it, or, for a mark on a tile the movement type cannot cross itself (a resource pool, a building), the one on the nearest crossable tile within `reach` tiles; -1 when there is none."""
    x, y = cell
    if labels[y, x] >= 0:
        return int(labels[y, x])
    height, width = labels.shape
    for distance in range(1, reach + 1):
        for dy in range(-distance, distance + 1):
            for dx in range(-distance, distance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and labels[ny, nx] >= 0:
                    return int(labels[ny, nx])
    return -1


def render_png(terrain: Terrain, path: str, scale: int = 4) -> str:
    """Writes the map as a PNG, `scale` pixels to a tile: the ground coloured by kind and every mark as a square on top (`COLOURS`)."""
    palette = np.asarray([COLOURS[kind] for kind in KINDS], dtype=np.uint8)
    index = {kind: position for position, kind in enumerate(KINDS)}
    tiles = palette[np.asarray([index[kind] for kind in terrain.kinds], dtype=np.int64)].reshape(
        terrain.height, terrain.width, 3)
    image = np.repeat(np.repeat(tiles, scale, axis=0), scale, axis=1)
    for mark in terrain.marks:
        colour = COLOURS.get(mark.kind, OTHER_UNIT)
        half = (3 if mark.kind == "commandCenter" else 2) * scale // 2
        cx, cy = mark.cell[0] * scale + scale // 2, mark.cell[1] * scale + scale // 2
        image[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1] = colour
    height, width = image.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8), image.reshape(height, width * 3)], axis=1)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(path, "wb") as out:
        out.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9)) + chunk(b"IEND", b""))
    return path
