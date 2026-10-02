"""Locating the game's asset tree."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .. import paths


@dataclass(frozen=True)
class AssetPaths:
    """Where the readers look for game data. The default is the assets directory of the game install under the working area."""

    assets: str

    @staticmethod
    def default() -> "AssetPaths":
        return AssetPaths(os.path.join(paths.game(), "assets"))

    @staticmethod
    def at(assets: str) -> "AssetPaths":
        return AssetPaths(os.path.abspath(assets))

    def require(self) -> "AssetPaths":
        if not os.path.isdir(self.assets):
            raise FileNotFoundError(
                f"no asset directory at {self.assets}; unpack the game into local/{paths.GAME_DIRECTORY} as docs/project/02-runtime.md describes, set RWINTEL_GAME, or pass --assets"
            )
        return self

    @property
    def skirmish_maps(self) -> str:
        return os.path.join(self.assets, "maps", "skirmish")

    @property
    def tilesets(self) -> str:
        return os.path.join(self.assets, "tilesets")

    @property
    def units(self) -> str:
        return os.path.join(self.assets, "units")
