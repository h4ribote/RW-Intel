"""What a match looked like as it went, read off a playback of it.

Two things are kept. Once per tactical period, where every side stood: units, worth, income, what each had destroyed and lost, and the credits it held, as the game reports it for all sides at once. And where the fighting happened: every unit that leaves the board between two observations is charged, at its price, to the region it was last seen in and to the side that owned it, which over a match says what each region cost each side.

Losses are read as disappearances, which is exact when the observation is omniscient. From one side's view under fog an enemy unit also leaves the observation by going out of sight, so the enemy's losses are counted only when the observation is omniscient, and the summary says which it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..data import Region
from ..eval.scoring import contestants
from ..wire import Observation


@dataclass
class RegionExchange:
    region: int
    x: float
    y: float
    #: Worth lost here by the observed side and by the others.
    ours: float = 0.0
    theirs: float = 0.0
    #: Units lost here by the observed side and by the others.
    ours_units: int = 0
    theirs_units: int = 0

    def as_dict(self) -> dict:
        return {"region": self.region, "x": round(self.x, 1), "y": round(self.y, 1),
                "ours": round(self.ours, 1), "theirs": round(self.theirs, 1),
                "ours_units": self.ours_units, "theirs_units": self.theirs_units}


@dataclass
class Timeline:
    """Built up over one playback, from the observations and the standings the game sends."""

    regions: Sequence[Region]
    #: The price of each type by the index the observation names it by.
    prices: Sequence[float]
    omniscient: bool = True
    #: One row per tactical period: game time, frame, the standings of every side, and the observed side's own figures.
    rows: List[dict] = field(default_factory=list)
    #: Every checksum the game had taken by some period, as frame and value, in the order they were seen.
    checksums: List[Tuple[int, int]] = field(default_factory=list)
    _last: Dict[int, Tuple[bool, float, float, int]] = field(default_factory=dict)
    _exchange: Dict[int, RegionExchange] = field(default_factory=dict)
    _own: Optional[dict] = None

    def progress(self, payload: Mapping, row: bool = True) -> None:
        """Takes one progress event, or with `row` false only the checksum it carries."""
        if row:
            entry = {"time_ms": int(payload.get("timeMs", 0)), "frame": int(payload.get("frame", 0)),
                     "standing": list(payload.get("standing", []))}
            if self._own is not None:
                entry["own"] = self._own
            self.rows.append(entry)
        frame = int(payload.get("checksumFrame", -1))
        if frame >= 0 and (not self.checksums or self.checksums[-1][0] != frame):
            self.checksums.append((frame, int(payload.get("checksum", 0))))

    def observe(self, observation: Observation) -> None:
        """Takes one observation, charging whatever left the board since the last one to where it was last seen."""
        present: Dict[int, Tuple[bool, float, float, int]] = {}
        for unit in observation.unit_states:
            present[unit.id] = (bool(unit.hostile), unit.x, unit.y, unit.type_index)
        for unit_id, (hostile, x, y, type_index) in self._last.items():
            if unit_id in present or (hostile and not self.omniscient):
                continue
            self._charge(hostile, x, y, self._price(type_index))
        self._last = present
        self._own = {"credits": round(observation.credits, 1), "income": round(observation.income, 2),
                     "units": observation.units,
                     "enemies_seen": sum(1 for unit in observation.unit_states if unit.hostile)}

    def _price(self, type_index: int) -> float:
        return float(self.prices[type_index]) if 0 <= type_index < len(self.prices) else 0.0

    def _charge(self, hostile: bool, x: float, y: float, price: float) -> None:
        region = _nearest(self.regions, x, y)
        if region is None:
            return
        exchange = self._exchange.get(region.id)
        if exchange is None:
            exchange = self._exchange[region.id] = RegionExchange(region=region.id, x=region.x, y=region.y)
        if hostile:
            exchange.theirs += price
            exchange.theirs_units += 1
        else:
            exchange.ours += price
            exchange.ours_units += 1

    def exchange(self) -> List[RegionExchange]:
        """Every region anything was lost in, by region."""
        return [self._exchange[key] for key in sorted(self._exchange)]

    def summary(self) -> dict:
        """The curves by team, for the teams that took part, the exchange by region, and the last standing."""
        curves: Dict[int, List[list]] = {}
        for row in self.rows:
            for entry in contestants(row["standing"]):
                team = int(entry.get("team", -1))
                curves.setdefault(team, []).append([round(row["time_ms"] / 1000.0, 1), entry.get("units", 0),
                                                    entry.get("value", 0), entry.get("income", 0),
                                                    entry.get("killed", 0), entry.get("lost", 0),
                                                    entry.get("credits", 0)])
        return {
            "omniscient": self.omniscient,
            "curve_columns": ["second", "units", "value", "income", "killed", "lost", "credits"],
            "curves": {str(team): points for team, points in sorted(curves.items())},
            "exchange": [e.as_dict() for e in self.exchange()],
            "last_standing": self.rows[-1]["standing"] if self.rows else [],
            "checksums": len(self.checksums),
        }


def _nearest(regions: Sequence[Region], x: float, y: float) -> Optional[Region]:
    best, best_distance = None, math.inf
    for region in regions:
        distance = (region.x - x) ** 2 + (region.y - y) ** 2
        if distance < best_distance:
            best, best_distance = region, distance
    return best
