"""The tactical layer's state as a set of tokens, built from what each decision was encoded from.

The tactical flat state aggregates the squad and what threatens it into 78 numbers, so a network that reads units one by one needs the units themselves. They are in the recorded materials (`materials.TacticalMaterials`): the squad row, one row per member and per threat, and the target region, with the run's type table as the context. A "set state" is one flat row holding all of it, so that a set network keeps the `forward(state, mask)` signature of the flat one:

    [78 flat features | target offset (2) | member tokens | threat tokens | token-present flags]

Members and threats are each taken nearest the squad's centre first and cut at a cap (`MEMBER_CAP`, `THREAT_CAP`); a slot past the last unit is all zeros with its present flag at nought. `set_state` builds one row from one decision's material rows, `set_states` builds every row of a dataset at once, and `cached_set_states` keeps a run's rows on disk at half precision, one file per shard under a digest of the shard file and of this layout, so a shard is tokenised once and a run that is still growing only has its new shards tokenised.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import paths
from ..control.policy.contracts import Domain, Role
from ..control.policy.encoding import DISTANCE_SCALE, RANGE_SCALE, RECENT_HIT_MS, TACTICAL_SIZE
from . import materials as materials_module

#: Most members and threats a set state carries; nearly every recorded fight fits, and the units past a cap are dropped farthest first.
MEMBER_CAP = 10
THREAT_CAP = 10

#: Bound on the scaled offsets, distances, ranges, prices, health and time since a hit.
TOKEN_CLIP = 4.0

#: Scales for a type's price and maximum health in a unit token.
PRICE_SCALE = 1000.0
HEALTH_SCALE = 1000.0

DOMAINS = tuple(Domain)
ROLES = tuple(Role)

#: What a unit token holds, in order.
UNIT_FEATURES = ("dx", "dy", "distance", "health_share", "range", "hits_air", "hits_land", "price", "max_health",
                 *tuple(f"domain_{d.name.lower()}" for d in DOMAINS), *tuple(f"role_{r.name.lower()}" for r in ROLES),
                 "attacking", "since_hit", "enemy")
UNIT_SIZE = len(UNIT_FEATURES)

#: The squad token's addition to the flat features: the offset to the contract's target region.
SQUAD_EXTRA = ("target_dx", "target_dy")
SQUAD_TOKEN = TACTICAL_SIZE + len(SQUAD_EXTRA)


def set_size(members: int = MEMBER_CAP, threats: int = THREAT_CAP) -> int:
    """How long a set state is for the caps given."""
    return SQUAD_TOKEN + (members + threats) * (UNIT_SIZE + 1)


SET_SIZE = set_size()


def layout(members: int = MEMBER_CAP, threats: int = THREAT_CAP) -> dict:
    return {"flat": TACTICAL_SIZE, "squad_extra": list(SQUAD_EXTRA), "unit": list(UNIT_FEATURES),
            "members": members, "threats": threats, "clip": TOKEN_CLIP, "price_scale": PRICE_SCALE,
            "health_scale": HEALTH_SCALE, "range_scale": RANGE_SCALE, "distance_scale": DISTANCE_SCALE,
            "recent_hit_ms": RECENT_HIT_MS}


def layout_digest(members: int = MEMBER_CAP, threats: int = THREAT_CAP) -> str:
    return hashlib.sha256(json.dumps(layout(members, threats), sort_keys=True).encode("utf-8")).hexdigest()[:16]


_U = {name: index for index, name in enumerate(materials_module.UNIT_COLUMNS)}
_S = {name: index for index, name in enumerate(materials_module.SQUAD_COLUMNS)}
_R = {name: index for index, name in enumerate(materials_module.REGION_COLUMNS)}


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _type_row(catalogue, type_index: int) -> List[float]:
    """The type part of a unit token: range, what it hits, price, maximum health, domain and role."""
    kind = catalogue.kind(type_index) if catalogue is not None else None
    row = [0.0] * 5
    if kind is not None:
        row = [_clip(float(kind.range) / RANGE_SCALE, 0.0, TOKEN_CLIP), 1.0 if kind.hits_air else 0.0,
               1.0 if kind.hits_land else 0.0, _clip(float(kind.price) / PRICE_SCALE, 0.0, TOKEN_CLIP),
               _clip(float(kind.max_hp) / HEALTH_SCALE, 0.0, TOKEN_CLIP)]
    domain = catalogue.domain(type_index) if catalogue is not None else Domain.STATIC
    role = catalogue.role(type_index) if catalogue is not None else Role.OTHER
    return row + [1.0 if d == domain else 0.0 for d in DOMAINS] + [1.0 if r == role else 0.0 for r in ROLES]


def _tokens(rows: Sequence[Sequence[float]], sx: float, sy: float, catalogue, enemy: bool, cap: int) -> List[List[float]]:
    measured = []
    for row in rows:
        dx, dy = float(row[_U["x"]]) - sx, float(row[_U["y"]]) - sy
        measured.append((math.hypot(dx, dy), dx, dy, row))
    measured.sort(key=lambda entry: entry[0])
    tokens = []
    for distance, dx, dy, row in measured[:cap]:
        max_health = float(row[_U["max_health"]])
        tokens.append([_clip(dx / RANGE_SCALE, -TOKEN_CLIP, TOKEN_CLIP), _clip(dy / RANGE_SCALE, -TOKEN_CLIP, TOKEN_CLIP),
                       _clip(distance / RANGE_SCALE, 0.0, TOKEN_CLIP),
                       float(row[_U["health"]]) / max_health if max_health > 0 else 0.0]
                      + _type_row(catalogue, int(row[_U["type_index"]]))
                      + [1.0 if int(row[_U["target"]]) != 0 else 0.0,
                         _clip(float(row[_U["since_hit_ms"]]) / RECENT_HIT_MS, 0.0, TOKEN_CLIP), 1.0 if enemy else 0.0])
    return tokens


def set_state(rows: Dict[str, Sequence[Sequence[float]]], context: Optional[materials_module.Context],
              features: Optional[Sequence[float]] = None, members: int = MEMBER_CAP,
              threats: int = THREAT_CAP) -> List[float]:
    """One decision's set state from its material rows and the run's context. `features` are the 78 flat features when already at hand; otherwise they are rebuilt from the rows."""
    if features is None:
        features = materials_module.rebuild("tactics", rows, context)
    catalogue = context.catalogue if context is not None else None
    squad = rows["squad"][0]
    sx, sy = float(squad[_S["x"]]), float(squad[_S["y"]])
    offset = [0.0, 0.0]
    if len(rows["target"]):
        target = rows["target"][0]
        offset = [_clip((float(target[_R["x"]]) - sx) / DISTANCE_SCALE, -1.0, 1.0),
                  _clip((float(target[_R["y"]]) - sy) / DISTANCE_SCALE, -1.0, 1.0)]
    ours = _tokens(rows["members"], sx, sy, catalogue, False, members)
    theirs = _tokens(rows["threats"], sx, sy, catalogue, True, threats)
    blank = [0.0] * UNIT_SIZE
    state = [float(value) for value in features] + offset
    for tokens, cap in ((ours, members), (theirs, threats)):
        for index in range(cap):
            state.extend(tokens[index] if index < len(tokens) else blank)
    state.extend([1.0 if index < len(ours) else 0.0 for index in range(members)])
    state.extend([1.0 if index < len(theirs) else 0.0 for index in range(threats)])
    return state


# A whole dataset at once.

def _type_tables(dataset) -> tuple:
    """Per decision, which row of one stacked type table its units are looked up in, and that table, whose last row is the unknown type."""
    keys: Dict[tuple, int] = {}
    tables: List[np.ndarray] = []
    of_episode = np.zeros(len(dataset.episodes), dtype=np.int64)
    for index, episode in enumerate(dataset.episodes):
        key = (int(episode["run"]), str(episode.get("context", "")))
        if key not in keys:
            catalogue = None
            if key[1]:
                with open(os.path.join(dataset.runs[key[0]]["path"], "tables", f"{key[1]}.json"), encoding="utf-8") as handle:
                    written = json.load(handle)
                catalogue = materials_module.Context.of(written["types"], written.get("combat")).catalogue
            count = len(catalogue.types) if catalogue is not None else 0
            table = np.asarray([_type_row(catalogue, i) for i in range(count)] + [_type_row(None, -1)], dtype=np.float64)
            keys[key] = len(tables)
            tables.append(table)
        of_episode[index] = keys[key]
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in tables])]).astype(np.int64)
    sizes = np.asarray([len(t) for t in tables], dtype=np.int64)
    return of_episode, np.concatenate(tables) if tables else np.zeros((1, 5 + len(DOMAINS) + len(ROLES))), offsets, sizes


def set_states(dataset, members: int = MEMBER_CAP, threats: int = THREAT_CAP) -> np.ndarray:
    """Every decision's set state, as `set_state` builds it, for a tactical dataset opened with its materials."""
    if dataset.layer != "tactics":
        raise ValueError("set states are built for the tactical layer")
    if not dataset.materials:
        raise ValueError("a set state is built from the materials, and these runs were opened without them")
    count = len(dataset)
    out = np.zeros((count, set_size(members, threats)), dtype=np.float32)
    out[:, :TACTICAL_SIZE] = dataset.arrays["state"]
    if not count:
        return out

    def first_rows(name: str):
        rows, starts = dataset.materials[name]
        has = starts[1:] > starts[:-1]
        return rows, starts, has

    squad_rows, squad_starts, has_squad = first_rows("squad")
    sx = np.zeros(count)
    sy = np.zeros(count)
    sx[has_squad] = squad_rows[squad_starts[:-1][has_squad], _S["x"]]
    sy[has_squad] = squad_rows[squad_starts[:-1][has_squad], _S["y"]]
    target_rows, target_starts, has_target = first_rows("target")
    if has_target.any():
        picked = target_rows[target_starts[:-1][has_target]]
        out[has_target, TACTICAL_SIZE] = np.clip((picked[:, _R["x"]] - sx[has_target]) / DISTANCE_SCALE, -1.0, 1.0)
        out[has_target, TACTICAL_SIZE + 1] = np.clip((picked[:, _R["y"]] - sy[has_target]) / DISTANCE_SCALE, -1.0, 1.0)

    of_episode, table, offsets, sizes = _type_tables(dataset)
    context_of = of_episode[dataset.episode_index()]
    flags = SQUAD_TOKEN + (members + threats) * UNIT_SIZE
    for name, cap, base, flag_base, enemy in (("members", members, SQUAD_TOKEN, flags, 0.0),
                                              ("threats", threats, SQUAD_TOKEN + members * UNIT_SIZE, flags + members, 1.0)):
        rows, starts = dataset.materials[name]
        if not len(rows):
            continue
        counts = np.diff(starts)
        owner = np.repeat(np.arange(count), counts)
        dx = rows[:, _U["x"]] - sx[owner]
        dy = rows[:, _U["y"]] - sy[owner]
        distance = np.hypot(dx, dy)
        order = np.lexsort((np.arange(len(rows)), distance, owner))
        rank = np.empty(len(rows), dtype=np.int64)
        rank[order] = np.arange(len(rows)) - starts[owner[order]]
        keep = rank < cap
        max_health = rows[:, _U["max_health"]]
        types = rows[:, _U["type_index"]].astype(np.int64)
        ctx = context_of[owner]
        known = (types >= 0) & (types < sizes[ctx] - 1)
        lookup = offsets[ctx] + np.where(known, types, sizes[ctx] - 1)
        features = np.concatenate([
            np.clip(dx / RANGE_SCALE, -TOKEN_CLIP, TOKEN_CLIP)[:, None],
            np.clip(dy / RANGE_SCALE, -TOKEN_CLIP, TOKEN_CLIP)[:, None],
            np.clip(distance / RANGE_SCALE, 0.0, TOKEN_CLIP)[:, None],
            np.where(max_health > 0, rows[:, _U["health"]] / np.where(max_health > 0, max_health, 1.0), 0.0)[:, None],
            table[lookup],
            (rows[:, _U["target"]].astype(np.int64) != 0).astype(np.float64)[:, None],
            np.clip(rows[:, _U["since_hit_ms"]] / RECENT_HIT_MS, 0.0, TOKEN_CLIP)[:, None],
            np.full((len(rows), 1), enemy),
        ], axis=1)
        kept_owner, kept_rank = owner[keep], rank[keep]
        columns = base + kept_rank[:, None] * UNIT_SIZE + np.arange(UNIT_SIZE)[None, :]
        out[kept_owner[:, None], columns] = features[keep]
        out[kept_owner, flag_base + kept_rank] = 1.0
    return out


def cache_path(run: str) -> str:
    """The directory a run's set states are cached in, one file per shard, under the local cache."""
    where = os.path.abspath(run)
    digest = hashlib.sha256(where.encode("utf-8")).hexdigest()[:8]
    return os.path.join(paths.local_root(), "cache", "tokens", f"{os.path.basename(os.path.normpath(where))}-{digest}")


def shard_cache_path(shard: str, members: int = MEMBER_CAP, threats: int = THREAT_CAP) -> str:
    """Where one shard's set states are kept, named by the shard's name, size and modification time and the layout's digest, so a shard written again or a changed layout is never served an old file."""
    status = os.stat(shard)
    held = json.dumps([os.path.basename(shard), status.st_size, status.st_mtime_ns, layout_digest(members, threats)])
    digest = hashlib.sha256(held.encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(shard))[0]
    return os.path.join(cache_path(os.path.dirname(os.path.abspath(shard))), f"{stem}-{digest}.npy")


class Stacked:
    """Several shards' cached rows read as one array without joining them in memory: `shape`, and rows taken by an index array, in any order."""

    def __init__(self, parts: Sequence[np.ndarray], width: int) -> None:
        self.parts = list(parts)
        self.starts = np.concatenate([[0], np.cumsum([len(part) for part in self.parts])]).astype(np.int64)
        self.shape = (int(self.starts[-1]), width)
        self.dtype = np.dtype(np.float16)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        index = np.atleast_1d(np.arange(self.shape[0])[index] if isinstance(index, slice) else np.asarray(index))
        out = np.empty((len(index), self.shape[1]), dtype=self.dtype)
        part_of = np.searchsorted(self.starts, index, side="right") - 1
        for part in np.unique(part_of):
            chosen = part_of == part
            out[chosen] = self.parts[part][index[chosen] - self.starts[part]]
        return out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        joined = np.concatenate(self.parts) if self.parts else np.zeros(self.shape, dtype=self.dtype)
        return joined.astype(dtype) if dtype is not None else joined


def cached_set_states(run: str, members: int = MEMBER_CAP, threats: int = THREAT_CAP,
                      shards: Optional[Sequence[str]] = None) -> Stacked:
    """A run's set states at half precision, shard by shard from the cache, each shard tokenised and written there the first time it is read. `shards` limits them to a listing of the run's shard files, so that a run still being written is read as of one listing."""
    from .dataset import Dataset, shard_files

    parts = []
    for shard in (shard_files(run) if shards is None else shards):
        path = shard_cache_path(shard, members, threats)
        if not os.path.exists(path):
            dataset = Dataset.open([run], layer="tactics", with_materials=True, shards=[shard])
            built = set_states(dataset, members, threats).astype(np.float16)
            del dataset
            with paths.replacing(path, "wb") as handle:
                np.save(handle, built)
            del built
        parts.append(np.load(path, mmap_mode="r"))
    return Stacked(parts, set_size(members, threats))
