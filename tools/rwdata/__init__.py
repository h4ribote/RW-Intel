"""Readers for the data Rusted Warfare ships on disk: skirmish maps and unit definitions.

Both are plain files under the game's assets directory, so everything here works without launching the game.
The engine reads the same files at load time, which is what makes these numbers usable as ground truth rather than estimates.
"""

from .assets import AssetPaths
from .maps import MapContent, read_map, list_skirmish_maps
from .regions import Region, decompose
from .units import UnitDefinition, read_unit_catalog

__all__ = [
    "AssetPaths",
    "MapContent",
    "Region",
    "UnitDefinition",
    "decompose",
    "list_skirmish_maps",
    "read_map",
    "read_unit_catalog",
]
