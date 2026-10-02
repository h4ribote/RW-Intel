"""The terrain frame: what each movement type can cross on the map in play, as the path finder's own connected components.

The agent sends it once an episode's map is loaded and again after a HELLO in the middle of one. Two places are reachable from each other by a movement type when they lie in the same component of its grid; air reaches everything and is not sent.

Body, little-endian: width u16, height u16, grids u8; per grid: name length u8, name ASCII, components u16, runs u32, then the runs row by row from the top left, each a label u16 and a length u16, the label 0xFFFF on a tile the movement type cannot cross.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

#: The label of a tile its movement type cannot cross, on the wire.
BLOCKED = 0xFFFF

#: World units to a tile.
TILE = 20.0

#: The movement type that crosses everything, which no grid is sent for.
AIR = "AIR"

_HEADER = struct.Struct("<HHB")
_GRID = struct.Struct("<HI")


@dataclass
class Passage:
    """Per movement type, the component of every tile, -1 where the movement type cannot go."""

    width: int
    height: int
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    components: Dict[str, int] = field(default_factory=dict)

    def tile(self, x: float, y: float) -> Tuple[int, int]:
        return (min(self.width - 1, max(0, int(x // TILE))), min(self.height - 1, max(0, int(y // TILE))))

    def component_at(self, movement: str, x: float, y: float, reach: int = 3) -> int:
        """The component of the world position for this movement type: the one under it, or, for a position on a tile it cannot cross (a building, a resource pool, a unit's footprint at the edge), the first crossable tile's within `reach` tiles, ring by ring and each ring row by row from the top left, as the game side's `Passage.componentAt` scans; -1 when there is none or no grid for the movement type. Air is everywhere one component."""
        if movement == AIR:
            return 0
        grid = self.labels.get(movement)
        if grid is None:
            return -1
        column, row = self.tile(x, y)
        if grid[row, column] >= 0:
            return int(grid[row, column])
        for distance in range(1, reach + 1):
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    nx, ny = column + dx, row + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and grid[ny, nx] >= 0:
                        return int(grid[ny, nx])
        return -1

    def reachable(self, movement: str, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Whether a unit of this movement type can go from one world position to the other under its own power."""
        first = self.component_at(movement, *start)
        return first >= 0 and first == self.component_at(movement, *end)

    def passable(self, movement: str, x: float, y: float) -> bool:
        if movement == AIR:
            return True
        grid = self.labels.get(movement)
        if grid is None:
            return False
        column, row = self.tile(x, y)
        return bool(grid[row, column] >= 0)


def decode_terrain(body: bytes) -> Passage:
    width, height, grids = _HEADER.unpack_from(body, 0)
    offset = _HEADER.size
    passage = Passage(width=width, height=height)
    for _ in range(grids):
        length = body[offset]
        offset += 1
        name = body[offset:offset + length].decode("ascii")
        offset += length
        components, runs = _GRID.unpack_from(body, offset)
        offset += _GRID.size
        pairs = np.frombuffer(body, dtype="<u2", count=runs * 2, offset=offset).reshape(-1, 2)
        offset += runs * 4
        labels = np.repeat(pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int64))
        labels[labels == BLOCKED] = -1
        passage.labels[name] = labels.reshape(height, width)
        passage.components[name] = components
    return passage


def encode_terrain(width: int, height: int, labels: Dict[str, np.ndarray]) -> bytes:
    """The body the agent writes, from per movement type label grids with -1 where it cannot go. What the tests and tools build a frame with."""
    parts = [_HEADER.pack(width, height, len(labels))]
    for name, grid in labels.items():
        flat = np.where(grid.reshape(-1) < 0, BLOCKED, grid.reshape(-1)).astype(np.int64)
        runs = []
        index = 0
        while index < len(flat):
            value = flat[index]
            length = 1
            while index + length < len(flat) and flat[index + length] == value and length < 0xFFFF:
                length += 1
            runs.append((int(value), length))
            index += length
        encoded = name.encode("ascii")
        components = int(grid.max()) + 1 if (grid >= 0).any() else 0
        parts.append(bytes([len(encoded)]) + encoded + _GRID.pack(components, len(runs)))
        parts.append(b"".join(struct.pack("<HH", value, length) for value, length in runs))
    return b"".join(parts)
