"""What a unit type is worth in a fight, and what two forces would do to each other.

Price is the measure of military value everywhere else, and it tracks durability well, but it says nothing about which of two types the same credits are better spent on, nor about what a force can reach: a tank does nothing to an aircraft whatever it cost. This table answers those two questions from the types' own numbers, so that the build order can choose what to make and the command layers can judge whether a fight is worth taking.

The model is Lanchester's square law. A force's fighting strength is the geometric mean of the health it can absorb and the damage it deals to what it is fighting, and two forces meeting lose in proportion to the square of each other's strength; so a type is worth what its health times its damage is worth per credit squared, and concentrating a force counts for more than its size. Damage a unit cannot deliver, because it cannot shoot upward or downward at what it is facing, is not counted.

Health comes from the engine's sample unit of each type, and damage per second and range from the definition files. A type that exists only as code has no damage on record, and is given the damage per credit typical of the types that do, which is to say the table has no opinion about it. Fights actually measured in the arena between pairs of types (`matchups`) correct the model for the pairs they cover, weighted by how many fights stand behind each figure.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from ... import paths
from .tuning import Tuning

#: How many measured fights a pair needs before its measurement counts as much as the model's prediction. Opening value.
MATCHUP_PRIOR = 20.0

#: The file the arena's measured matchups are read from, beside the other reports.
MATCHUPS_FILE = "matchups.json"

#: World units the range difference is quoted against: the longest reach among the built-in armed types.
RANGE_SCALE = 400.0


@dataclass(frozen=True)
class Profile:
    """One type's fighting numbers."""

    price: float
    hp: float
    dps: float
    range: float
    flying: bool
    hits_air: bool
    hits_land: bool
    #: Whether the damage came from a definition rather than from the typical figure.
    measured: bool

    @property
    def armed(self) -> bool:
        return self.dps > 0.0 and self.range > 0.0

    def reaches(self, other: "Profile") -> bool:
        return self.hits_air if other.flying else self.hits_land


class CombatTable:
    def __init__(self, catalogue, matchups: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None,
                 tuning: Tuning = Tuning()) -> None:
        self.catalogue = catalogue
        #: How much the reach of a type beyond what it fights adds to its worth, per RANGE_SCALE of difference, since a unit that shoots first shoots for longer.
        self.range_bonus = tuning.range_bonus
        #: The least share of an enemy's damage taken to reach a type that it cannot reach at all. Without a floor an aircraft facing an army without anti-air would be infinitely valuable, and the model would buy nothing else.
        self.unreachable_floor = tuning.unreachable_floor
        definitions = getattr(catalogue, "definitions", {}) or {}
        typical_dps, typical_hp = _typical(catalogue.types, definitions)
        self.profiles: Dict[int, Profile] = {}
        for kind in catalogue.types:
            definition = definitions.get(kind.lookup) or definitions.get(kind.name)
            dps = definition.damage_per_second if definition is not None else 0.0
            measured = dps > 0.0
            armed = kind.range > 0.0 and not kind.builder
            if armed and not measured:
                dps = typical_dps * kind.price
            hp = getattr(kind, "max_hp", 0.0) or (definition.max_hp if definition is not None else 0.0) or typical_hp * kind.price
            self.profiles[kind.index] = Profile(price=float(kind.price), hp=hp, dps=dps if armed else 0.0,
                                                range=kind.range, flying=kind.movement == "AIR",
                                                hits_air=kind.hits_air, hits_land=kind.hits_land, measured=measured)
        #: Measured outcomes by (type, opponent type): the mean outcome for the first at equal credits, and how many fights it is the mean of.
        self.matchups = dict(matchups or {})

    @classmethod
    def load(cls, catalogue, path: Optional[str] = None, tuning: Tuning = Tuning()) -> "CombatTable":
        """The table with whatever matchups the arena has measured, read from the reports directory."""
        return cls(catalogue, read_matchups(path or os.path.join(paths.reports(), MATCHUPS_FILE), catalogue), tuning)

    def snapshot(self) -> dict:
        """Everything the table answers from, as plain data: the profiles, the measured matchups and the two tuned figures. A table rebuilt from it (`from_snapshot`) answers every question the same way without reading the definition files or the matchups report again."""
        return {"range_bonus": self.range_bonus, "unreachable_floor": self.unreachable_floor,
                "profiles": {str(index): asdict(profile) for index, profile in sorted(self.profiles.items())},
                "matchups": [[a, b, mean, fights] for (a, b), (mean, fights) in sorted(self.matchups.items())]}

    @classmethod
    def from_snapshot(cls, snapshot: dict, catalogue=None) -> "CombatTable":
        table = cls.__new__(cls)
        table.catalogue = catalogue
        table.range_bonus = float(snapshot["range_bonus"])
        table.unreachable_floor = float(snapshot["unreachable_floor"])
        table.profiles = {int(index): Profile(**fields) for index, fields in snapshot["profiles"].items()}
        table.matchups = {(int(a), int(b)): (float(mean), float(fights)) for a, b, mean, fights in snapshot["matchups"]}
        return table

    def profile(self, type_index: int) -> Optional[Profile]:
        return self.profiles.get(type_index)

    # ---- forces --------------------------------------------------------------------------

    def strength(self, force: Sequence[Tuple[int, float]], against: Sequence[Tuple[int, float]]) -> float:
        """A force's fighting strength against another: the geometric mean of the health it has left and the damage it can deliver to that opponent. Each entry is a type and how much of it is standing, in units (a unit at half health is half a unit)."""
        health = sum(self._hp(t) * n for t, n in force)
        return math.sqrt(max(0.0, health) * max(0.0, self._damage(force, against)))

    def outcome(self, ours: Sequence[Tuple[int, float]], theirs: Sequence[Tuple[int, float]]) -> float:
        """What a fight between two forces does, from -1 to +1: the share of its strength the winner keeps, signed for the first. Two empty forces, or two that cannot touch each other, are a draw."""
        a, b = self.strength(ours, theirs), self.strength(theirs, ours)
        if a <= 0.0 and b <= 0.0:
            return 0.0
        if a >= b:
            return math.sqrt(max(0.0, 1.0 - (b / a) ** 2))
        return -math.sqrt(max(0.0, 1.0 - (a / b) ** 2))

    def _hp(self, type_index: int) -> float:
        profile = self.profiles.get(type_index)
        return profile.hp if profile is not None else 0.0

    def _damage(self, force: Sequence[Tuple[int, float]], against: Sequence[Tuple[int, float]]) -> float:
        """Damage per second the force delivers to the opponent, each unit counted for the share of the opponent's health it can shoot at."""
        targets = [(self.profiles.get(t), n) for t, n in against]
        total = sum(p.hp * n for p, n in targets if p is not None)
        if total <= 0.0:
            return sum(p.dps * n for p, n in ((self.profiles.get(t), n) for t, n in force) if p is not None)
        damage = 0.0
        for t, n in force:
            p = self.profiles.get(t)
            if p is None or p.dps <= 0.0:
                continue
            reachable = sum(q.hp * m for q, m in targets if q is not None and p.reaches(q)) / total
            damage += p.dps * n * reachable
        return damage

    # ---- types ---------------------------------------------------------------------------

    def efficiency(self, type_index: int, enemy: Sequence[Tuple[int, float]], per: str = "credit") -> float:
        """How much fighting strength one type buys against an enemy of the given composition, per credit or, with `per="unit"`, per unit. A force of one type has strength proportional to its number times the square root of health times damage, and a spend buys the spend over the price in number, so the per-credit figure is health times damage over the price squared: squared strength per credit squared, which is what decides a fight between equal spends. The per-unit figure is health times damage alone.

        An enemy that cannot touch this type at all is taken to reach it with `unreachable_floor` of its damage, and a type that outranges the enemy's average reach is worth `range_bonus` more per RANGE_SCALE of the difference. Measured matchups then bend the figure towards what the arena saw.
        """
        profile = self.profiles.get(type_index)
        if profile is None or not profile.armed or profile.price <= 0.0:
            return 0.0
        enemy = [(t, n) for t, n in enemy if n > 0 and self.profiles.get(t) is not None]
        if not enemy:
            enemy_profiles = []
        else:
            enemy_profiles = [(self.profiles[t], n) for t, n in enemy]
        hp_total = sum(q.hp * n for q, n in enemy_profiles)
        reach = (sum(q.hp * n for q, n in enemy_profiles if profile.reaches(q)) / hp_total) if hp_total > 0 else 1.0
        dps_total = sum(q.dps * n for q, n in enemy_profiles)
        exposure = (sum(q.dps * n for q, n in enemy_profiles if q.reaches(profile)) / dps_total) if dps_total > 0 else 1.0
        exposure = max(self.unreachable_floor, exposure)
        reach_diff = 0.0
        if dps_total > 0:
            enemy_range = sum(q.range * q.dps * n for q, n in enemy_profiles) / dps_total
            reach_diff = max(0.0, profile.range - enemy_range) / RANGE_SCALE
        square = profile.hp * profile.dps * reach / exposure * (1.0 + self.range_bonus * reach_diff)
        value = square / profile.price ** 2 if per == "credit" else square
        return value * self._measured(type_index, enemy)

    def _measured(self, type_index: int, enemy: Sequence[Tuple[int, float]]) -> float:
        """How far the arena's measurements of this type against the enemy's types move the model's figure: the measured strength ratio over the predicted one for each pair, averaged over the enemy by worth and shrunk towards one by how few fights stand behind it."""
        if not self.matchups or not enemy:
            return 1.0
        weight_total = 0.0
        factor = 0.0
        for t, n in enemy:
            entry = self.matchups.get((type_index, t))
            profile = self.profiles.get(t)
            if profile is None:
                continue
            worth = profile.price * n
            weight_total += worth
            if entry is None:
                factor += worth
                continue
            mean, fights = entry
            predicted = _ratio(self.outcome([(type_index, 1.0 / max(1.0, self.profiles[type_index].price))],
                                            [(t, 1.0 / max(1.0, profile.price))]))
            observed = _ratio(mean)
            shrink = fights / (fights + MATCHUP_PRIOR)
            corrected = (observed / predicted) ** 2 if predicted > 0 else 1.0
            factor += worth * (1.0 + shrink * (corrected - 1.0))
        return factor / weight_total if weight_total > 0 else 1.0


def _ratio(outcome: float) -> float:
    """The strength ratio a square-law fight with this outcome implies, first over second."""
    outcome = max(-0.99, min(0.99, outcome))
    if outcome >= 0:
        return 1.0 / math.sqrt(1.0 - outcome ** 2)
    return math.sqrt(1.0 - outcome ** 2)


def _typical(types: Iterable, definitions: Dict) -> Tuple[float, float]:
    """The median damage per second and health per credit among the armed mobile types with definitions, which is what a type with no numbers on record is assumed to have."""
    dps, hp = [], []
    for kind in types:
        definition = definitions.get(kind.lookup) or definitions.get(kind.name)
        if definition is None or kind.price <= 0 or kind.building or kind.range <= 0:
            continue
        if definition.damage_per_second > 0:
            dps.append(definition.damage_per_second / kind.price)
        if definition.max_hp > 0:
            hp.append(definition.max_hp / kind.price)
    return _median(dps) or 0.05, _median(hp) or 0.5


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def read_matchups(path: str, catalogue) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """The measured matchups in a report, keyed by type index. The report names types by lookup so that it stays readable when the catalogue's numbering changes; pairs naming a type this catalogue does not have are skipped."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    index = {kind.lookup: kind.index for kind in catalogue.types}
    found: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for entry in entries.get("pairs", []):
        a, b = index.get(entry.get("type")), index.get(entry.get("against"))
        if a is None or b is None:
            continue
        found[(a, b)] = (float(entry.get("mean", 0.0)), float(entry.get("fights", 0)))
    return found
