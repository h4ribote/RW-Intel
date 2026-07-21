"""Reading the built-in unit definitions.

In 1.15 every unit, including the built-in ones, is generated from an `.ini` file under `assets/units`; the engine's own parser reads exactly these files.
So the catalogue here is the same data the game runs on, not an approximation of it, which is what makes it usable for the things a commander needs: what a unit costs, what it can hurt, how far it reaches, and what has to exist before it can be built.

Two unit systems in these files are deliberately ignored. Distances are mixed: `maxAttackRange` is in world units, `fogOfWarSightRange` is in tiles, and the definitions themselves show the conversion as `${core.fogOfWarSightRange * 20 - 40}`. Everything this module reports is converted to world units, which is what the order API takes.
Rates are per frame at the engine's 60 frames per second, so `shootDelay` and `buildSpeed` are converted to seconds.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .assets import AssetPaths

#: The engine expresses per-frame rates against this rate regardless of how fast frames are actually drawn.
FRAMES_PER_SECOND = 60.0

#: World units per map tile. Fixed by the tile size every shipped map declares, and confirmed by the range expression in the definitions.
WORLD_UNITS_PER_TILE = 20.0

#: Applied when a definition does not set fogOfWarSightRange, per the comment in the shared template.
DEFAULT_SIGHT_TILES = 15.0


@dataclass
class UnitDefinition:
    """One buildable thing. Distances are world units, times are seconds."""

    #: The name the engine registers the type under, which is what build links and the `u_`/`b_` action ids use.
    name: str
    #: The definition's own `name` key, which differs whenever it takes over a built-in slot through `overrideAndReplace`.
    definition_name: str
    source: str
    price: int = 0
    max_hp: float = 0.0
    tech_level: int = 0
    build_speed: float = 0.0
    is_building: bool = False
    movement_type: str = ""
    move_speed: float = 0.0
    attack_range: float = 0.0
    sight_range: float = DEFAULT_SIGHT_TILES * WORLD_UNITS_PER_TILE
    can_attack: bool = False
    hits_air: bool = False
    hits_land: bool = False
    direct_damage: float = 0.0
    area_damage: float = 0.0
    shoot_delay_frames: float = 0.0
    built_from: List[str] = field(default_factory=list)
    builds: List[str] = field(default_factory=list)

    @property
    def damage_per_second(self) -> float:
        if not self.can_attack or self.shoot_delay_frames <= 0:
            return 0.0
        return (self.direct_damage + self.area_damage) * FRAMES_PER_SECOND / self.shoot_delay_frames

    @property
    def build_seconds(self) -> float:
        """How long a single builder takes to finish this, buildSpeed being progress per frame."""
        if self.build_speed <= 0:
            return 0.0
        return 1.0 / (self.build_speed * FRAMES_PER_SECOND)


#: Values the definitions use to mean "no link here", in the spellings they actually appear in.
_ABSENT = ("none", "ignore", "null")

_SECTION = re.compile(r"^\[(?P<name>[^\]]+)\]")
_ENTRY = re.compile(r"^(?P<key>[A-Za-z_0-9]+)\s*:\s*(?P<value>.*)$")


def _read_ini(path: str) -> Dict[str, Dict[str, str]]:
    """A forgiving reader for the game's ini dialect: `#` comments, `key: value`, repeated sections merged."""
    sections: Dict[str, Dict[str, str]] = {}
    current = sections.setdefault("core", {})
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            heading = _SECTION.match(line)
            if heading:
                current = sections.setdefault(heading.group("name").strip(), {})
                continue
            entry = _ENTRY.match(line)
            if entry:
                value = entry.group("value").split("#", 1)[0].strip()
                current.setdefault(entry.group("key").strip(), value)
    return sections


def _flag(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes")


def _number(value: Optional[str], default: float = 0.0) -> float:
    if value is None:
        return default
    # Definitions may compute a value from other keys; those are left at the default rather than evaluated.
    try:
        return float(value.strip())
    except ValueError:
        return default


def _resolve(path: str, cache: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    """Reads a definition with its `copyFrom` chain folded in. Keys already present win, as they do in the engine."""
    key = os.path.normcase(os.path.abspath(path))
    if key in cache:
        return cache[key]
    cache[key] = {}  # guards against a definition chain that loops back on itself
    sections = _read_ini(path)
    # dont_load marks a file as a template to be copied from, so it describes this file and must not reach the files that copy it.
    own_template_flag = sections.get("core", {}).get("dont_load")
    parent_name = sections.get("core", {}).get("copyFrom")
    if parent_name:
        parent_path = os.path.join(os.path.dirname(path), parent_name)
        if os.path.exists(parent_path):
            for section, entries in _resolve(parent_path, cache).items():
                target = sections.setdefault(section, {})
                for entry_key, entry_value in entries.items():
                    target.setdefault(entry_key, entry_value)
    if own_template_flag is None:
        sections.get("core", {}).pop("dont_load", None)
    cache[key] = sections
    return sections


def _definition(path: str, sections: Dict[str, Dict[str, str]]) -> Optional[UnitDefinition]:
    core = sections.get("core", {})
    if _flag(core.get("dont_load")):
        return None
    name = core.get("name")
    if not name:
        return None

    attack = sections.get("attack", {})
    movement = sections.get("movement", {})
    turret = sections.get("turret_1", {})
    projectile = sections.get("projectile_1", {})

    replaces = (core.get("overrideAndReplace") or "").strip()
    unit = UnitDefinition(
        name=replaces if replaces and replaces not in _ABSENT else name,
        definition_name=name,
        source=path,
        price=int(_number(core.get("price"), -1)),
        max_hp=_number(core.get("maxHp")),
        tech_level=int(_number(core.get("techLevel"), 1)),
        build_speed=_number(core.get("buildSpeed")),
        is_building=_flag(core.get("isBuilding")),
        movement_type=movement.get("movementType", core.get("movementType", "")).strip().upper(),
        move_speed=_number(movement.get("moveSpeed")),
        attack_range=_number(attack.get("maxAttackRange")),
        sight_range=_number(core.get("fogOfWarSightRange"), DEFAULT_SIGHT_TILES) * WORLD_UNITS_PER_TILE,
        can_attack=_flag(attack.get("canAttack")),
        hits_air=_flag(attack.get("canAttackFlyingUnits")),
        hits_land=_flag(attack.get("canAttackLandUnits"), True),
        direct_damage=_number(projectile.get("directDamage")),
        area_damage=_number(projectile.get("areaDamage")),
        shoot_delay_frames=_number(attack.get("shootDelay"), _number(turret.get("shootDelay"))),
    )
    if unit.movement_type == "BUILDING":
        unit.is_building = True

    for index in range(1, 9):
        source = (core.get(f"builtFrom_{index}_name") or "").strip()
        if source and source.lower() not in _ABSENT:
            unit.built_from.append(source)
    return unit


def read_unit_catalog(paths: Optional[AssetPaths] = None) -> Dict[str, UnitDefinition]:
    """Every built-in definition, keyed by internal name, with the build links resolved in both directions."""
    paths = (paths or AssetPaths.default()).require()
    cache: Dict[str, Dict[str, Dict[str, str]]] = {}
    catalog: Dict[str, UnitDefinition] = {}

    for path in sorted(glob.glob(os.path.join(paths.units, "**", "*.ini"), recursive=True)):
        try:
            unit = _definition(path, _resolve(path, cache))
        except Exception:  # a malformed definition should not take the whole catalogue down
            continue
        if unit is not None and unit.name not in catalog:
            catalog[unit.name] = unit

    for unit in catalog.values():
        for producer in unit.built_from:
            if producer in catalog:
                catalog[producer].builds.append(unit.name)
    for unit in catalog.values():
        unit.builds.sort()
    return catalog
