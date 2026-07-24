"""What the constructed operations arena promises, held to without launching a game.

The arena's whole claim is that a run of the handwritten operational chain against itself pools its side score to nought, so that a policy measured on it is read as the difference from a board that does not lean. That pooled zero is a statistical statement and needs live episodes, but the properties it rests on are arithmetic and geometric and can be pinned here: the side score is exactly antisymmetric under a hostility flip, the board is a single reflection about one centre with equal garrison strength on both sides, the priorities are invariant under the mirror map, and the pre-placed garrisons never reach a command layer. Every one of those is a break in exchange symmetry if it fails, and every one is checkable on a synthetic capture.

The arena is exercised the way the engagement arena is exercised in test_learning: assembled field by field rather than constructed, because what these tests look at needs the type catalogue and the seeded random and nothing the game provides. Nothing here connects to anything.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.session import UnitType
from rwintel.data.regions import Region
from rwintel.learn.ops_arena import (
    CATCHMENT_RADIUS,
    CONTEST_PAIRS,
    HORIZON_MS,
    OUR_SQUADS,
    OURS,
    THEIRS,
    OpsArena,
    OpsStatistics,
    _Contest,
)
from rwintel.wire import (
    BLOCK_REGIONS,
    BLOCK_UNITS,
    NO_SQUAD,
    Action,
    Observation,
    UnitState,
)

#: A catalogue with something in every doctrine's pool, so a draw of any of the three staged doctrines returns a force: an armour tank and an artillery piece the vanguard takes, an anti-air the garrison wants beside its armour, and a hover raider for the raid.
_TYPES = [
    UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", range=130.0, hits_air=False, hits_land=True),
    UnitType(index=1, name="artillery", lookup="artillery", price=700, tech=1, building=False,
             builder=False, movement="LAND", range=320.0, hits_air=False, hits_land=True),
    UnitType(index=2, name="flak", lookup="flak", price=400, tech=1, building=False, builder=False,
             movement="LAND", range=200.0, hits_air=True, hits_land=False),
    UnitType(index=3, name="raider", lookup="raider", price=300, tech=1, building=False, builder=False,
             movement="HOVER", range=100.0, hits_air=False, hits_land=True),
]


class _Catalogue(Catalogue):
    """The type table without the producer links, which are read off the game's asset tree and are not what the arena's geometry or scoring is about."""

    def __init__(self, types):
        self.types = list(types)
        self.roles = {kind.index: role_of(kind) for kind in self.types}
        self.built_from = {}
        self.declares_maker = {}


_CATALOGUE = _Catalogue(_TYPES)


class _Session:
    """Enough of a session for the arena's geometry: the stable region table the map decomposition would provide, whose baseless sparring slot owns the enemy side, and a scenario sink that records the one order the deployment submits."""

    def __init__(self, regions, sparring_slot=1):
        self.regions = regions
        self.sparring_slot = sparring_slot
        self.calls = []

    def scenario(self, spawns, sandbox=None):
        self.calls.append((list(spawns), sandbox))


def _grid(step=400.0, reach=1200.0):
    """A grid of regions dense enough that the two points of a contest pair fall on distinct cells and two pairs do not collide. Ids are the row order, matching the wire's own region numbering."""
    regions = []
    index = 0
    coordinate = -reach
    coords = []
    while coordinate <= reach + 1e-6:
        coords.append(coordinate)
        coordinate += step
    for gx in coords:
        for gy in coords:
            regions.append(Region(id=index, x=gx, y=gy, radius=step / 2, resources=2, spawn=False))
            index += 1
    return regions


#: Four staging corners about the origin, so the board centre is the origin and a staging point is a far corner: every contest sits near the centre, clear of the staging squads by geometry.
_CORNERS = [(4000.0, 4000.0), (4000.0, -4000.0), (-4000.0, 4000.0), (-4000.0, -4000.0)]


def _arena(session=None, seed=0, our_n=OUR_SQUADS, radius=CATCHMENT_RADIUS, pairs=CONTEST_PAIRS,
           sites=None):
    """An arena assembled field by field, as everywhere the arena is exercised without a game. The heavy constructor builds command layers from the type catalogue the game sent at HELLO, which is not what the geometry and the scoring need."""
    arena = OpsArena.__new__(OpsArena)
    arena.session = session
    arena.catalogue = _CATALOGUE
    arena.random = random.Random(seed)
    arena.our_n = our_n
    arena.radius = radius
    arena.contest_pairs = pairs
    arena.horizon_ms = HORIZON_MS
    arena.score_slope = 0.0
    arena.enemy_slot = None
    arena.phase = "opening"
    arena.squads = {}
    arena.enemy = {}
    arena.garrisons = []
    arena.pairs = []
    arena.contests = []
    arena.mirror_of = {}
    arena.orders = None
    arena.priorities = {}
    arena.garrison_share = {}
    arena.our_home_id = None
    arena.their_home_id = None
    arena.our_reports = []
    arena.their_reports = []
    arena.side_score = 0.0
    arena.refused = False
    arena.until_ms = 0
    arena.known = set()
    arena._wanted = {}
    arena._period = 0
    arena._sandbox_sent = True
    arena.centre = (0.0, 0.0)
    arena._our_pt = None
    arena._their_pt = None
    arena.sites = list(sites or [])
    arena.last_regions = []
    arena.statistics = OpsStatistics()
    return arena


def _unit(unit_id, x=0.0, y=0.0, type_index=0, hostile=0, health=100.0, max_health=100.0):
    return UnitState(id=unit_id, squad=NO_SQUAD, type_index=type_index, x=x, y=y, health=health,
                     max_health=max_health, built=255, order=255, queued=0, target=0, stance=5,
                     hostile=hostile, since_hit_ms=9999)


def _observation(units=(), regions=(), slot=0, game_time_ms=1000):
    return Observation(frame=1, game_time_ms=game_time_ms, episode=1,
                       blocks=BLOCK_REGIONS | BLOCK_UNITS, slot=slot, credits=0.0, income=0.0,
                       units=len(units), unit_cap=200, under_construction=0, killed_units=0,
                       killed_buildings=0, lost_units=0, lost_buildings=0,
                       regions=list(regions), unit_states=list(units))


def _rows(flat):
    """The flat spawn order back into rows of type, slot, x, y, count."""
    return [flat[base:base + 5] for base in range(0, len(flat), 5)]


# ---- the side score is exactly antisymmetric -----------------------------------------------

def test_the_side_score_is_exactly_antisymmetric_under_a_hostility_flip():
    """The property the whole measurement rests on, and the reason self-play must average nought.

    A run of the script chain against itself is the only statement there is about whether the arena leans, and it is only a statement about the arena if the two sides' scores are one number and its negation whatever happened on the board. Here the other side's score is this one with every unit's hostility flipped: each contest's our-worth and enemy-worth swap, so its share is one less this one's and its domination the negative, and the priorities are one board statement identical to both. Read at floating precision the two sum to nought.
    """
    arena = _arena()
    arena.contests = [_Contest(region_id=1, point=(0.0, 0.0)),
                      _Contest(region_id=2, point=(2000.0, 0.0))]
    # A mirror pair's two regions carry the same weight, but antisymmetry does not depend on that; unequal weights make the check bite harder.
    arena.priorities = {1: 0.8, 2: 0.5}

    units = [
        # First catchment: our armour whole and half-health against enemy artillery.
        _unit(1, 0.0, 0.0, type_index=0, hostile=0, health=100.0),
        _unit(2, 50.0, 20.0, type_index=0, hostile=0, health=50.0),
        _unit(3, 30.0, 40.0, type_index=1, hostile=1, health=100.0),
        # Second catchment: enemy armour against our artillery at a quarter.
        _unit(4, 2000.0, 10.0, type_index=1, hostile=0, health=25.0),
        _unit(5, 2020.0, 30.0, type_index=0, hostile=1, health=100.0),
        _unit(6, 1980.0, 10.0, type_index=0, hostile=1, health=80.0),
        # A stray well outside every disc, which must count for neither side.
        _unit(7, 9000.0, 9000.0, type_index=0, hostile=0, health=100.0),
    ]
    ours = arena._side_score(units)
    flipped = [dataclasses.replace(unit, hostile=1 - unit.hostile) for unit in units]
    theirs = arena._side_score(flipped)

    assert abs(ours + theirs) < 1e-9
    # And it is a real number, not a trivial nought that any board would give.
    assert abs(ours) > 1e-6

    # An empty catchment reads a half and contributes no domination, which is what keeps a region neither side has reached from leaning the score.
    assert arena._catchment_worths(units, (9000.0, -9000.0)) == (0.0, 0.0)
    empty = _arena()
    empty.contests = [_Contest(region_id=1, point=(0.0, 0.0))]
    empty.priorities = {1: 1.0}
    assert empty._side_score([]) == 0.0
    assert empty._side_score([_unit(1, 9000.0, 9000.0)]) == 0.0


# ---- the mirror layout ----------------------------------------------------------------------

def test_the_mirror_layout_pairs_every_point_and_equalises_the_garrisons():
    """The board is a single reflection about one centre, so every unit this side spawns has a congruent partner the other side spawns, and the two sides' garrison worth is equal.

    This is the structural fix for the dominant exchange-symmetry break the design's review found: all-enemy garrisons make it attack-versus-defend-on-their-ground with the enemy strictly stronger, a lean no within-episode sign check can see. Mirror-paired-by-ownership garrisons of equal value make each side attack one member of every pair and defend the other, and reflecting the whole board about one centre makes the two sides congruent rather than measured in non-congruent map buckets. Both are checked here on the one order the deployment submits.
    """
    session = _Session(_grid())
    arena = _arena(session=session, seed=7, sites=_CORNERS)
    action = Action()
    arena._deploy(_observation(slot=0), action, 0)
    assert not arena.refused

    # One order carries both sides, so neither is submitted ahead of the other.
    assert len(session.calls) == 1
    flat, sandbox = session.calls[0]
    rows = _rows(flat)
    ours = [row for row in rows if int(row[1]) == 0]
    theirs = [row for row in rows if int(row[1]) == 1]
    assert ours and len(ours) == len(theirs)

    centre = arena.centre
    remaining = list(theirs)
    for row in ours:
        target = (2 * centre[0] - row[2], 2 * centre[1] - row[3])
        match = next((other for other in remaining
                      if int(other[0]) == int(row[0])
                      and math.hypot(other[2] - target[0], other[3] - target[1]) < 1e-6), None)
        assert match is not None, "an our-side spawn has no congruent mirror on the other side"
        remaining.remove(match)
    assert not remaining

    # One garrison value per pair, placed once on each side by ownership, so the total garrison worth is equal.
    our_garrison = sum(g.value for g in arena.garrisons if g.side == OURS)
    their_garrison = sum(g.value for g in arena.garrisons if g.side == THEIRS)
    assert our_garrison > 0.0 and abs(our_garrison - their_garrison) < 1e-9

    # Every contest pair's two members are distinct regions, so the two sides' contests are measured in different buckets rather than one.
    assert len(arena.pairs) == CONTEST_PAIRS
    for pair in arena.pairs:
        assert pair.attack_region != pair.defend_region
    contested = {contest.region_id for contest in arena.contests}
    assert len(contested) == 2 * CONTEST_PAIRS

    # Nothing has been commissioned yet, so the taskable dicts are still empty: a garrison is never in them.
    assert arena.squads == {} and arena.enemy == {}


# ---- the priorities are invariant under the mirror map --------------------------------------

def test_the_synthesized_priorities_are_invariant_under_the_mirror_map():
    """A weight drawn once per unordered mirror pair and set on both of its regions, so the priority a region carries equals the priority its mirror carries. Any unpaired or mismatched weight is an asymmetric board statement and a direct lean; drawing region by region instead would re-randomise the second member of a pair."""
    session = _Session(_grid())
    # A seed whose draw places both contest pairs on this grid at the default catchment radius; the invariance under test does not depend on which seed, only on both pairs being placed.
    arena = _arena(session=session, seed=7, sites=_CORNERS)
    arena._deploy(_observation(slot=0), Action(), 0)
    assert not arena.refused

    assert arena.priorities and arena.orders is not None
    # The orders handed to both sides carry exactly these priorities.
    assert arena.orders.priorities is arena.priorities
    for region, weight in arena.priorities.items():
        mirror = arena.mirror_of[region]
        assert arena.priorities[mirror] == weight


# ---- garrisons never reach the command loop -------------------------------------------------

def test_garrisons_never_enter_the_taskable_squad_dicts():
    """A garrison handed to the operational layer would be tasked and finished — GARRISON doctrine has real tasks — giving the enemy more squads than us and polluting a side's trajectories. So a garrison stays out of the taskable dicts entirely.

    The board the deployment submitted is reconstructed unit for unit and commissioned, exactly as it would be off a live observation. The staged squads form from the units near the two staging points; the garrisons, which spawn out at the contest points, are excluded by position and appear in no squad's membership on either side.
    """
    session = _Session(_grid())
    arena = _arena(session=session, seed=11, sites=_CORNERS)
    arena._deploy(_observation(slot=0), Action(), 0)
    assert not arena.refused

    flat, _ = session.calls[0]
    their_slot = arena._their_slot(_observation(slot=0))
    units = []
    for index, row in enumerate(_rows(flat)):
        units.append(_unit(1000 + index, x=row[2], y=row[3], type_index=int(row[0]),
                           hostile=1 if int(row[1]) == their_slot else 0))
    # The units at a contest point are the garrisons; those near a staging point are the squads.
    garrison_ids = {unit.id for unit in units
                    if any(math.hypot(unit.x - g.point[0], unit.y - g.point[1]) <= arena.radius
                           for g in arena.garrisons)}
    assert garrison_ids, "no garrison units were reconstructed to check the exclusion against"

    arena._commission(_observation(units=units, slot=0), Action(), arena.until_ms)

    # Both sides got their full complement of taskable squads.
    assert len(arena.squads) == arena.our_n and len(arena.enemy) == arena.our_n
    # And not one garrison unit is in any of them.
    commissioned = {member for squad in list(arena.squads.values()) + list(arena.enemy.values())
                    for member in squad.members}
    assert commissioned and not (commissioned & garrison_ids)


def test_an_episode_is_refused_when_the_room_exposes_no_sparring_slot():
    """The enemy squads and garrisons need a baseless player to own them, with its built-in AI stopped, or the other side cannot be placed and fights on its own economy and poisons every number. The room settles which slot that is as it fills, so an episode where none was reported is refused rather than run against a slot that is playing its own match. The engagement arena's slot fallback is right for a match and wrong here, so the refusal reads the report and not the fallback."""
    session = _Session(_grid(), sparring_slot=-1)
    arena = _arena(session=session, seed=1, sites=_CORNERS)
    arena._deploy(_observation(slot=0), Action(), 0)

    assert arena.refused and arena.phase == "done"
    assert arena.statistics.refused
    # No board was submitted, because there was nobody to own half of it.
    assert session.calls == []


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
