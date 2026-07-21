"""Reading the contents of a skirmish map.

Maps are Tiled TMX files. What matters for region decomposition is in two places: tiles carrying the `res_pool` property, which the engine turns into the crystalResource objects that extractors are built on, and tiles in the units tileset carrying `unit` and `team` properties, which is where each player's starting command centre goes.
"""

from __future__ import annotations

import base64
import glob
import gzip
import os
import struct
import zlib
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .assets import AssetPaths

# Tiled stores three transform flags in the top bits of every global tile id.
_GID_MASK = 0x1FFFFFFF


@dataclass
class MapContent:
    """A skirmish map reduced to the things a commander cares about. Tile coordinates, origin at the top left."""

    path: str
    name: str
    width: int
    height: int
    tile_size: int
    resources: List[Tuple[int, int]] = field(default_factory=list)
    spawns: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def players(self) -> int:
        return len(self.spawns)

    def to_world(self, tile: Tuple[int, int]) -> Tuple[float, float]:
        """Tile centre in the world coordinates the engine and the order API use."""
        return ((tile[0] + 0.5) * self.tile_size, (tile[1] + 0.5) * self.tile_size)


def _tile_properties(element: ElementTree.Element) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for tile in element.findall("tile"):
        properties = {p.get("name"): p.get("value") for p in tile.findall("./properties/property")}
        if properties:
            out[int(tile.get("id"))] = properties
    return out


def _load_tileset(element: ElementTree.Element, paths: AssetPaths, map_dir: str) -> Dict[int, Dict[str, str]]:
    source = element.get("source")
    if source is None:
        return _tile_properties(element)
    # External tileset references are written relative to the shared tilesets directory, not to the map.
    for candidate in (os.path.join(paths.tilesets, source), os.path.join(map_dir, source)):
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return _tile_properties(ElementTree.parse(candidate).getroot())
    # Some maps reference tilesets that were never shipped. Those tiles carry no properties we need.
    return {}


def _decode_layer(layer: ElementTree.Element) -> Tuple[int, ...]:
    data = layer.find("data")
    if data is None or data.text is None:
        return ()
    raw = base64.b64decode(data.text.strip())
    compression = data.get("compression")
    if compression == "gzip":
        raw = gzip.decompress(raw)
    elif compression == "zlib":
        raw = zlib.decompress(raw)
    return struct.unpack("<%dI" % (len(raw) // 4), raw)


def read_map(path: str, paths: Optional[AssetPaths] = None) -> MapContent:
    paths = paths or AssetPaths.default()
    map_dir = os.path.dirname(os.path.abspath(path))
    root = ElementTree.parse(path).getroot()

    tilesets = root.findall("tileset")
    first_gids = [int(t.get("firstgid")) for t in tilesets]
    properties: Dict[int, Dict[str, str]] = {}
    for index, tileset in enumerate(tilesets):
        first = first_gids[index]
        # A tileset image can hold more tiles than the map reserved ids for, and the surplus ids belong to the next tileset.
        # Without this clamp the units tileset bleeds into the terrain that follows it and every ground tile reads as a unit.
        limit = first_gids[index + 1] - first if index + 1 < len(tilesets) else 1 << 28
        for tile_id, tile_properties in _load_tileset(tileset, paths, map_dir).items():
            if tile_id < limit:
                properties[first + tile_id] = tile_properties

    width = int(root.get("width"))
    content = MapContent(
        path=os.path.abspath(path),
        name=os.path.splitext(os.path.basename(path))[0],
        width=width,
        height=int(root.get("height")),
        tile_size=int(root.get("tilewidth")),
    )

    for layer in root.findall("layer"):
        for index, gid in enumerate(_decode_layer(layer)):
            if gid == 0:
                continue
            tile_properties = properties.get(gid & _GID_MASK)
            if not tile_properties:
                continue
            cell = (index % width, index // width)
            if "res_pool" in tile_properties:
                content.resources.append(cell)
            if tile_properties.get("unit") == "commandCenter":
                content.spawns.append(cell)

    content.resources.sort()
    content.spawns.sort()
    return content


def list_skirmish_maps(paths: Optional[AssetPaths] = None) -> List[str]:
    paths = (paths or AssetPaths.default()).require()
    return sorted(glob.glob(os.path.join(paths.skirmish_maps, "*.tmx")))
