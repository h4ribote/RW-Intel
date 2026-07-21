"""Locating the game's asset tree."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _repository_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class AssetPaths:
    """Where the readers look for game data. The default is the master copy the runtime docs describe."""

    assets: str

    @staticmethod
    def default() -> "AssetPaths":
        return AssetPaths(os.path.join(_repository_root(), "local", "rw", "assets"))

    @staticmethod
    def at(assets: str) -> "AssetPaths":
        return AssetPaths(os.path.abspath(assets))

    def require(self) -> "AssetPaths":
        if not os.path.isdir(self.assets):
            raise FileNotFoundError(
                f"no asset directory at {self.assets}; copy the game into local/rw as the runtime document describes, or pass --assets"
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
