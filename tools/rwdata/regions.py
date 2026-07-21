"""Cutting a map into the discrete regions the command layers address.

The upper layers must not name places by raw coordinates, or their action space changes shape with every map.
Regions give them a small fixed set of names instead. They are derived from what is actually worth holding: resource pools, which is where extractors and therefore income come from, and the starting positions.

The rule is single-linkage agglomeration under a fixed world distance, with one constraint: two clusters that each contain a starting position never merge, because a region containing two players' bases cannot be attacked or defended as one place.
A fixed distance is used rather than a fixed cluster count. Cutting at a fixed count instead makes the regions of a large map grow until one of them spans a third of the board, which is not a place a squad can hold.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .maps import MapContent

#: Merge distance in world units. Chosen against the reach of the units that would contest a region: a tier 1 tank shoots 130 and sees 640, so 400 is comfortably inside what one squad covers and comfortably outside a single extractor's footprint.
DEFAULT_MERGE_DISTANCE = 400.0


@dataclass
class Region:
    id: int
    x: float
    y: float
    radius: float
    resources: int
    spawn: bool

    def distance_to(self, other: "Region") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def decompose(content: MapContent, merge_distance: float = DEFAULT_MERGE_DISTANCE) -> List[Region]:
    seeds: List[Tuple[float, float]] = [content.to_world(t) for t in content.resources]
    resource_count = len(seeds)
    seeds += [content.to_world(t) for t in content.spawns]
    if not seeds:
        return []

    parent = list(range(len(seeds)))
    members: List[List[int]] = [[i] for i in range(len(seeds))]
    holds_spawn = [i >= resource_count for i in range(len(seeds))]
    alive = set(range(len(seeds)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs: List[Tuple[float, int, int]] = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            distance = math.dist(seeds[i], seeds[j])
            if distance <= merge_distance:
                pairs.append((distance, i, j))
    heapq.heapify(pairs)

    while pairs:
        _, i, j = heapq.heappop(pairs)
        a, b = find(i), find(j)
        if a == b or (holds_spawn[a] and holds_spawn[b]):
            continue
        parent[a] = b
        members[b] += members[a]
        holds_spawn[b] = holds_spawn[b] or holds_spawn[a]
        alive.discard(a)

    regions: List[Region] = []
    for root in alive:
        points = [seeds[i] for i in members[root]]
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        regions.append(
            Region(
                id=0,
                x=cx,
                y=cy,
                radius=max(math.dist((cx, cy), p) for p in points),
                resources=sum(1 for i in members[root] if i < resource_count),
                spawn=holds_spawn[root],
            )
        )

    # A stable order so region ids mean the same thing across runs. Each layer that needs an egocentric order sorts by distance from its own base at observation time.
    regions.sort(key=lambda r: (round(r.y, 3), round(r.x, 3)))
    for index, region in enumerate(regions):
        region.id = index
    return regions


def order_from(regions: Sequence[Region], origin: Optional[Region]) -> List[Region]:
    """Regions sorted by distance from a starting region, which is the order the command layers see them in.

    Ordering egocentrically is what lets a policy trained on one map read another: slot 0 is always home, and the later slots run outward.
    """
    if origin is None:
        return list(regions)
    return sorted(regions, key=lambda r: (origin.distance_to(r), r.id))
