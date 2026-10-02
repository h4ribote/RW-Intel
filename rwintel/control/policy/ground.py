"""Which ground is worth expanding onto, and in what order.

The strategic layer names the regions to expand into and the economy places extractors in them, so both have to rank ground the same way; otherwise the army clears one region while the only builder walks to another. Both read it from here.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

from ...wire import RegionState

#: How far a region's centre may be from ground we hold for it to count as the next step out. Opening value: regions are agglomerated at 400 world units, so this is a few region widths and not a whole board.
FRONTIER_REACH = 1600.0

#: Enemy worth in a region that a builder may be sent into anyway. Opening value, about two of the cheapest things that move: a scout passing through is not a reason to stop expanding, and anything heavier is a reason to send a squad first.
ESTABLISHED_ENEMY = 700.0


def enemy_spawns(spawns: Sequence, home) -> List:
    """The starting positions that are not ours, ours being the one nearest home. Empty when there is no home yet or the map has fewer than two."""
    if home is None or len(spawns) < 2:
        return []
    start = min(spawns, key=lambda r: math.hypot(r.x - home.x, r.y - home.y))
    return [r for r in spawns if r.id != start.id]


def safety_rank(home, enemies: Sequence) -> Callable[[float, float], float]:
    """Ranks a place to expand onto, lower first: how far it is from home, less how much deeper into our own half it lies than the enemy's.

    Distance alone sends an opening into the middle of the board. On a map between two players the centre pool is usually the nearest unclaimed one, and it is nearer the enemy's approach than anything behind it: it is where the first army arrives, and the only builder an opening has is what would be standing there. Subtracting the margin -how much closer the place is to us than to them -makes a slightly further pool deep in our own ground beat a slightly nearer one on the line.
    """

    def rank(x: float, y: float) -> float:
        reach = math.hypot(x - home.x, y - home.y) if home is not None else 0.0
        if not enemies:
            return reach
        theirs = min(math.hypot(x - r.x, y - r.y) for r in enemies)
        return reach - (theirs - reach)

    return rank


def open_points(region: RegionState) -> int:
    """Resource points in a region that nobody has an extractor on."""
    return max(0, region.resources - region.held_by_us - region.held_by_enemy)


def established(region: RegionState) -> bool:
    """Whether the enemy is established in a region: an extractor of theirs on the ground, or more standing there than a builder could be sent past. Merely seen there is not enough, because the observation is omniscient and one scout crossing a region would otherwise veto it for as long as it kept walking."""
    return region.held_by_enemy > 0 or region.enemy_value > max(ESTABLISHED_ENEMY, region.our_value)


def frontier(regions: Sequence[RegionState], home: Optional[RegionState], enemies: Sequence,
             chained: bool = True) -> List[int]:
    """The regions to expand into next, safest first, by id.

    A region qualifies when it has a resource point nobody holds, the enemy is not established there, and its centre is within FRONTIER_REACH of ground we hold. Chained, every region with an extractor of ours counts as held ground, so the frontier moves outward as the expansion does; unchained, only home counts, which caps the economy at a fixed disk around it.
    """
    if home is None:
        return []
    anchors = [home] + ([r for r in regions if r.held_by_us > 0 and r.id != home.id] if chained else [])
    rank = safety_rank(home, enemies)
    candidates = [r for r in regions
                  if r.id != home.id and open_points(r) > 0 and not established(r)
                  and any(math.hypot(r.x - a.x, r.y - a.y) <= FRONTIER_REACH for a in anchors)]
    return [r.id for r in sorted(candidates, key=lambda r: (rank(r.x, r.y), r.id))]
