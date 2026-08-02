"""What the constructed operations arena promises, held to without launching a game.

The arena's whole claim is that a run of the handwritten operational chain against itself pools its side score to nought, so that a policy measured on it is read as the difference from a board that does not lean. That pooled zero is a statistical statement and needs live episodes, but the properties it rests on are arithmetic and geometric and can be pinned here: the side score is exactly antisymmetric under a hostility flip, the board is a single reflection about one centre with equal garrison strength on both sides, the priorities are invariant under the mirror map, and the pre-placed garrisons never reach a command layer. Every one of those is a break in exchange symmetry if it fails, and every one is checkable on a synthetic capture.

The arena is exercised the way the engagement arena is exercised in test_learning: assembled field by field rather than constructed, because what these tests look at needs the type catalogue and the seeded random and nothing the game provides. Nothing here connects to anything.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.session import UnitType
from rwintel.control.policy.contracts import (
    Doctrine,
    OperationsOrders,
    Posture,
    SquadRecord,
    TaskContract,
)
from rwintel.control.policy.operations import Concentrated, Operations
from rwintel.control.policy.tactics import Tactics
from rwintel.control.policy.view import WorldView, build as build_view
from rwintel.wire.action import Deviation
from rwintel.data.regions import Region
from rwintel.learn import ops_run
from rwintel.learn.encoding import OPERATIONAL_SIZE, TACTICAL_SIZE
from rwintel.learn.layers import LearntOperations, LearntTactics
from rwintel.learn.rollout import FIGHT_DISCOUNT, FIGHT_TRACE, Rollout, Step
from rwintel.learn.ops_arena import (
    CATCHMENT_RADIUS,
    CONTEST_PAIRS,
    CREDIT,
    GARRISON_SCALE,
    HORIZON_MS,
    OPENING_BASELINE,
    OUR_SQUADS,
    OURS,
    MIRROR_ARRIVAL,
    SCRIPT_TACTICS,
    STANDING_LEAVE,
    STANDING_MIRROR,
    TENURE,
    THEIRS,
    OpsArena,
    OpsStatistics,
    _Contest,
)
from rwintel.control.policy.strategy import LOSS_ALLOWANCE_FLOOR, OFFENSIVE
from rwintel.wire import (
    BLOCK_REGIONS,
    BLOCK_UNITS,
    NO_SQUAD,
    Action,
    Observation,
    RegionState,
    UnitState,
)
from rwintel.wire.action import Stance, Status, Task

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
    """Enough of a session for the arena's geometry: the stable region table the map decomposition would provide, whose baseless sparring slot owns the enemy side, and a scenario sink that records the one order the deployment submits.

    The type table and the asset tree are here as well, because the arena's real constructor builds its own catalogue out of them and its own command layers out of that; the tests that go through the constructor rather than assembling an arena field by field need both. No asset tree at all is the right answer for a session that never touched a game: the catalogue then carries the types and no producer links, which is everything the layers ask of it here.
    """

    def __init__(self, regions, sparring_slot=1):
        self.regions = regions
        self.sparring_slot = sparring_slot
        self.types = list(_TYPES)
        self.assets = None
        self.calls = []

    def scenario(self, spawns, sandbox=None):
        self.calls.append((list(spawns), sandbox))


@contextlib.contextmanager
def _without_the_game_installed():
    """Builds a real arena without the game's definition files under it.

    The arena's constructor makes its own catalogue, and a catalogue reads which building produces which type out of the files the engine itself loads; with no asset tree named it looks for the master copy. Every test here is meant to pass on a machine that has never had the game on it — a test that quietly needs the install passes where it was written and fails where it is read — and the production links are the one thing an arena test never asks about, so they are handed back empty.
    """
    from rwintel.control.policy import catalogue

    original = catalogue._build_links
    catalogue._build_links = lambda assets: ({}, {})
    try:
        yield
    finally:
        catalogue._build_links = original


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
           sites=None, credit=CREDIT, tenure=TENURE, standing=STANDING_MIRROR):
    """An arena assembled field by field, as everywhere the arena is exercised without a game. The heavy constructor builds command layers from the type catalogue the game sent at HELLO, which is not what the geometry and the scoring need."""
    arena = OpsArena.__new__(OpsArena)
    arena.session = session
    arena.catalogue = _CATALOGUE
    arena.random = random.Random(seed)
    arena.placement = random.Random(seed ^ 0x5EED51E5)
    arena.our_n = our_n
    arena.radius = radius
    arena.contest_pairs = pairs
    arena.horizon_ms = HORIZON_MS
    arena.opening_baseline = OPENING_BASELINE
    arena.credit = credit
    arena.tenure = tenure
    arena.garrison_scale = GARRISON_SCALE
    arena.standing = standing
    arena._mirrored = []
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
    arena.ownership = {}
    arena.garrison_share = {}
    arena._frozen = {}
    arena.our_home_id = None
    arena.their_home_id = None
    arena.our_reports = []
    arena.their_reports = []
    arena.side_score = 0.0
    arena._accrued = {}
    arena._side_accrued = 0.0
    arena._marked_ms = None
    arena.refused = False
    arena.until_ms = 0
    arena.known = set()
    arena._wanted = {}
    arena._drawn = {}
    arena._period = 0
    arena._operational_period = 0
    arena._issued = {}
    arena._at_issue = {}
    arena._balance = {}
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
    """A weight drawn once per unordered mirror pair and set on both of its regions, so the priority a region carries equals the priority its mirror carries. Any unpaired or mismatched weight is an asymmetric board statement and a direct lean; drawing region by region instead would re-randomise the second member of a pair.

    That holds of both dicts and for the same reason. The scored weights carry the contested regions; the dict the layers are told carries those and a weight on every other region the reflection pairs, so that a weight of nought stops being a label for unscored ground. Ground the reflection does not pair carries nothing in either, because a prize one side has and the other does not is the very asymmetry the mirrored draw exists to remove.
    """
    session = _Session(_grid())
    # A seed whose draw places both contest pairs on this grid at the default catchment radius; the invariance under test does not depend on which seed, only on both pairs being placed.
    arena = _arena(session=session, seed=7, sites=_CORNERS)
    arena.last_regions = list(_grid())
    arena._deploy(_observation(slot=0), Action(), 0)
    assert not arena.refused

    assert arena.priorities and arena.orders is not None
    for region, weight in arena.priorities.items():
        mirror = arena.mirror_of[region]
        assert arena.priorities[mirror] == weight

    # The posture is drawn rather than pinned, and the two orders it decides are read off the same tables a match reads them off.
    #
    # Pinned to ARM before, with the other two drawn independently: four of the operational cut's five posture features were nought on every board this arena had drawn, while in a match all five move, and `offensive` and the allowance became free quantities that no strategic layer could have paired that way. A board saying "hold the front, and press" is a board a match cannot emit.
    assert arena.orders.offensive is OFFENSIVE[arena.orders.posture]
    assert arena.orders.loss_allowance >= LOSS_ALLOWANCE_FLOOR
    drawn = set()
    for seed in range(24):
        other = _arena(session=_Session(_grid()), seed=seed, sites=_CORNERS)
        other.last_regions = list(_grid())
        other._deploy(_observation(slot=0), Action(), 0)
        if other.orders is not None:
            drawn.add(other.orders.posture)
            assert other.orders.offensive is OFFENSIVE[other.orders.posture]
    assert len(drawn) >= 4, "the draw reaches only %d posture(s), so the cut's posture block barely moves" % len(drawn)

    # The layers are told about more ground than the episode scores, and every weight in that dict is even under the reflection too.
    wanted = arena.orders.priorities
    assert wanted is arena.wanted and set(arena.priorities) <= set(wanted)
    assert len(wanted) > len(arena.priorities), "the layer is still told that unscored ground is worth nothing"
    for region, weight in arena.priorities.items():
        assert wanted[region] == weight, "a scored region is worth what the episode scores it by"
    by_position = {region.id: (region.x, region.y) for region in _grid()}
    for region, weight in wanted.items():
        point = arena._mirror(by_position[region], arena.centre)
        mirror = min(by_position, key=lambda other: (by_position[other][0] - point[0]) ** 2
                     + (by_position[other][1] - point[1]) ** 2)
        assert abs(wanted.get(mirror, 0.0) - weight) < 1e-12, (
            "region %d carries a weight its reflection does not" % region)


def test_the_opening_board_is_reflected_for_the_side_that_has_no_starting_position():
    """The free base a starting position is given stands on this side alone, so its reflection is placed for the other side and the opening board is congruent unit for unit like everything else on it.

    It is not a nicety about totals. A vanguard's candidate regions are the ones carrying enemy strength, so an unreflected base is a candidate for the seat opposite and for no one else, and every re-task the ladder makes reads that asymmetric candidate set. `leave` is kept as the arm that measures what it was worth, and under it no reflection is placed at all.
    """
    session = _Session(_grid())
    arena = _arena(session=session, seed=13, sites=_CORNERS)
    # The free base: a headquarters and a builder standing where the map put them, before anything is deployed.
    base = [_unit(1, 3600.0, 3600.0, type_index=0), _unit(2, 3660.0, 3540.0, type_index=1)]
    arena._deploy(_observation(units=base, slot=0), Action(), 0)
    assert not arena.refused

    flat, _ = session.calls[0]
    rows = _rows(flat)
    their_slot = arena._their_slot(_observation(slot=0))
    centre = arena.centre
    for unit in base:
        target = (2 * centre[0] - unit.x, 2 * centre[1] - unit.y)
        assert any(int(row[1]) == their_slot and int(row[0]) == unit.type_index
                   and math.hypot(row[2] - target[0], row[3] - target[1]) < 1e-6 for row in rows), \
            "a standing unit of this side has no reflection ordered for the other"
    assert len(arena._mirrored) == len(base)

    # And the BOARD is congruent, which is not the same as the order being congruent: this side's base is already
    # standing and is not ordered, so the other side's order carries exactly those extra rows.
    ours = [row for row in rows if int(row[1]) == 0]
    theirs = [row for row in rows if int(row[1]) == their_slot]
    assert len(theirs) == len(ours) + len(base)
    standing = [(unit.type_index, unit.x, unit.y) for unit in base]
    board = [(int(row[0]), row[2], row[3]) for row in ours] + standing
    remaining = [(int(row[0]), row[2], row[3]) for row in theirs]
    for kind, x, y in board:
        target = (2 * centre[0] - x, 2 * centre[1] - y)
        match = next((other for other in remaining
                      if other[0] == kind and math.hypot(other[1] - target[0], other[2] - target[1]) < 1e-6), None)
        assert match is not None, "a piece of this side's board has no congruent partner on the other"
        remaining.remove(match)
    assert not remaining

    left = _Session(_grid())
    leaving = _arena(session=left, seed=13, sites=_CORNERS, standing=STANDING_LEAVE)
    leaving._deploy(_observation(units=base, slot=0), Action(), 0)
    assert not leaving._mirrored
    kept = _rows(left.calls[0][0])
    assert not any(math.hypot(row[2] - (2 * leaving.centre[0] - base[0].x),
                              row[3] - (2 * leaving.centre[1] - base[0].y)) < 1e-6 for row in kept)


def test_the_board_says_who_holds_each_region_and_says_it_antisymmetrically():
    """A field with the economy taken out has nobody holding a resource point anywhere, so the wire's ownership count was nought on every row while it moves all match — and two of the handwritten ladder's doctrine filters read nothing else, which left every garrison-doctrine squad falling through to the home region.

    The board states the ownership from its own construction instead: a side holds the contest its own garrison opens on, the pairs the reflection makes but no contest was drawn on are dealt out a side each, and the statement is turned over for the seat opposite so the two sides read the same ground from their own side of it.
    """
    session = _Session(_grid())
    arena = _arena(session=session, seed=23, sites=_CORNERS)
    arena._deploy(_observation(slot=0), Action(), 0)
    assert not arena.refused
    assert arena.ownership, "no region was given an owner"

    # A side holds the contest its own garrison opens on, and the mirror member is the other side's.
    for pair in arena.pairs:
        assert arena.ownership[pair.defend_region] == 1.0
        assert arena.ownership[pair.attack_region] == -1.0

    # Every owned region's partner is owned by the other side: the statement is antisymmetric under the mirror map.
    ours = sum(1 for held in arena.ownership.values() if held > 0)
    theirs = sum(1 for held in arena.ownership.values() if held < 0)
    assert ours == theirs and ours > 0

    # And what the two seats read of one region is the exact reverse of each other.
    regions = [RegionState(id=region_id, resources=2, held_by_us=0, held_by_enemy=0,
                           x=0.0, y=0.0, our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0,
                           distance_from_home=0.0)
               for region_id in sorted(arena.ownership)]
    observation = _observation(slot=0)
    ours_view = arena._contacts(WorldView(observation=observation, catalogue=_CATALOGUE, regions=list(regions)),
                                observation, ours=True)
    theirs_view = arena._contacts(WorldView(observation=observation, catalogue=_CATALOGUE, regions=list(regions)),
                                  observation, ours=False)
    for mine, yours in zip(ours_view.regions, theirs_view.regions):
        assert mine.id == yours.id
        assert mine.held_by_us == yours.held_by_enemy and mine.held_by_enemy == yours.held_by_us
        assert (mine.held_by_us > 0) != (mine.held_by_enemy > 0), "a region is held by exactly one of the two"


def test_a_board_whose_reflection_could_not_be_placed_is_refused():
    """Ground that takes no building is a fact about the map, and a board where the reflection of the opening base never appears is one base short on one side. Refused whole rather than run and scored, exactly as a board that cannot carry its contest pairs is.

    The arrival is matched by position and side rather than by type, because the engine settles a building onto its own grid instead of onto the point it was asked for, and one arrival cannot answer for two points.
    """
    session = _Session(_grid())
    arena = _arena(session=session, seed=17, sites=_CORNERS)
    base = [_unit(1, 3600.0, 3600.0, type_index=0)]
    arena._deploy(_observation(units=base, slot=0), Action(), 0)
    assert len(arena._mirrored) == 1
    point = arena._mirrored[0]

    flat, _ = session.calls[0]
    their_slot = arena._their_slot(_observation(slot=0))
    spawned = []
    for index, row in enumerate(_rows(flat)):
        hostile = 1 if int(row[1]) == their_slot else 0
        if hostile and math.hypot(row[2] - point[0], row[3] - point[1]) < 1e-6:
            continue  # the ground the reflection landed on took nothing
        spawned.append(_unit(1000 + index, x=row[2], y=row[3], type_index=int(row[0]), hostile=hostile))

    observation = _observation(units=base + spawned, slot=0)
    arena.known = {unit.id for unit in base}
    arena._commission(observation, Action(), arena.until_ms)
    assert arena.refused and arena.statistics.refused

    # And with the reflection standing — inside the tolerance the engine's own nudge needs — the same board runs.
    session2 = _Session(_grid())
    again = _arena(session=session2, seed=17, sites=_CORNERS)
    again._deploy(_observation(units=base, slot=0), Action(), 0)
    flat2, _ = session2.calls[0]
    arrived = []
    for index, row in enumerate(_rows(flat2)):
        hostile = 1 if int(row[1]) == again._their_slot(_observation(slot=0)) else 0
        nudge = MIRROR_ARRIVAL / 2 if hostile and math.hypot(row[2] - point[0], row[3] - point[1]) < 1e-6 else 0.0
        arrived.append(_unit(1000 + index, x=row[2] + nudge, y=row[3], type_index=int(row[0]), hostile=hostile))
    again.known = {unit.id for unit in base}
    again._commission(_observation(units=base + arrived, slot=0), Action(), again.until_ms)
    assert not again.refused and again.phase == "running"


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

    # And every staged squad carries the doctrine its force was drawn for, on both sides and slot for slot.
    #
    # Not the one its first unit would be offered to, which is a different question with a different answer: `doctrine_for` tries ENGINEER, RAID, GARRISON and VANGUARD in that order, so a tank goes to GARRISON and a vanguard force comes back labelled a garrison. The label is the action space here — `task_mask` reads it — so such a squad would be offered DEFEND and ESCORT and never ATTACK, in an arena built to measure whether ground is taken, while a match musters fighting formations first and gives the same armour a vanguard's head.
    for slot, squad in list(arena.squads.items()) + list(arena.enemy.items()):
        assert squad.doctrine is arena._drawn[slot], (
            "slot %d was drawn as %s and came back as %s" % (slot, arena._drawn[slot], squad.doctrine))
    mirrored = {slot: arena._drawn[slot] for slot in range(arena.our_n)}
    assert all(arena._drawn[arena.our_n + slot] is doctrine for slot, doctrine in mirrored.items()), (
        "the mirror squad is the same force reflected and has to be the same doctrine")


def test_an_episode_is_refused_when_the_room_exposes_no_sparring_slot():
    """The enemy squads and garrisons need a baseless player to own them, with its built-in AI stopped, or the other side cannot be placed and fights on its own economy and poisons every number. The room settles which slot that is as it fills, so an episode where none was reported is refused rather than run against a slot that is playing its own match. The engagement arena's slot fallback is right for a match and wrong here, so the refusal reads the report and not the fallback."""
    session = _Session(_grid(), sparring_slot=-1)
    arena = _arena(session=session, seed=1, sites=_CORNERS)
    arena._deploy(_observation(slot=0), Action(), 0)

    assert arena.refused and arena.phase == "done"
    assert arena.statistics.refused
    # No board was submitted, because there was nobody to own half of it.
    assert session.calls == []


#: A contract to copy for a squad under test. Only the region it names is read by the terminal; the rest is what any contract carries.
_CONTRACT = TaskContract(squad=0, task=Task.ATTACK, target_region=0, stance=Stance.AGGRESSIVE,
                         cost_budget=1000.0, deadline_ms=60000, issued_at_ms=0)


def test_the_side_score_is_antisymmetric_across_a_sweep_of_synthetic_boards():
    """One hand-built board says the arithmetic is right on that board. What the arena's trust rests on is that it is right on every board a run can draw, including the awkward ones: a contest nobody reached, a region carrying no priority, a disc holding only one side, a unit with no maximum health, a board with no contests at all.

    So the same check is swept over two thousand drawn boards and the worst residue is held to floating precision. A sweep is the only form of this claim that can be quoted as coverage; a single board cannot be.
    """
    draw = random.Random(20260725)
    worst = 0.0
    nontrivial = 0
    for _ in range(2000):
        arena = _arena()
        count = draw.randint(0, 4)
        arena.contests = [_Contest(region_id=index, point=(draw.uniform(-3000.0, 3000.0),
                                                          draw.uniform(-3000.0, 3000.0)))
                          for index in range(count)]
        # Some regions carry no priority at all and some priorities name no contest, which are both boards a draw can produce.
        arena.priorities = {index: draw.choice([0.0, draw.uniform(0.3, 1.0)])
                            for index in range(count + 2)}
        units = []
        for unit_id in range(draw.randint(0, 12)):
            near = draw.choice(arena.contests).point if arena.contests and draw.random() < 0.7 else (0.0, 0.0)
            maximum = draw.choice([100.0, 0.0])
            units.append(_unit(unit_id, near[0] + draw.uniform(-600.0, 600.0),
                               near[1] + draw.uniform(-600.0, 600.0),
                               type_index=draw.randrange(len(_TYPES)), hostile=draw.randint(0, 1),
                               health=draw.uniform(0.0, 100.0), max_health=maximum))
        ours = arena._side_score(units)
        theirs = arena._side_score([dataclasses.replace(unit, hostile=1 - unit.hostile) for unit in units])
        worst = max(worst, abs(ours + theirs))
        if abs(ours) > 1e-6:
            nontrivial += 1
    assert worst < 1e-12, "the two sides of a drawn board did not sum to nought"
    assert nontrivial > 500, "a sweep of boards that all score nought says nothing about antisymmetry"


def test_the_drawn_contests_never_overlap_and_never_reach_what_was_standing():
    """Two invariants the score is read under, both of which failed silently: a pair's two points are the reflection of each other, so their separation is twice the offset drawn and was never tested against the catchment diameter that two different pairs are held to; and nothing kept a contest clear of the free command centre and builder this side is given, which the mirrored side has no counterpart for and which the catchment would have counted whole.

    Swept over many draws rather than one, because both failures are conditional on the draw and a single board says nothing about whether they can happen.
    """
    session = _Session(_grid(step=400.0, reach=4000.0))
    standing = [(600.0, 0.0), (-1500.0, 1500.0)]
    boards = 0
    for seed in range(300):
        arena = _arena(session=session, seed=seed, sites=_CORNERS)
        pairs = arena._draw_pairs((0.0, 0.0), standing)
        if len(pairs) < arena.contest_pairs:
            continue
        boards += 1
        points = [point for pair in pairs for point in (pair.attack_point, pair.defend_point)]
        for index, first in enumerate(points):
            for second in points[index + 1:]:
                assert math.hypot(first[0] - second[0], first[1] - second[1]) >= 2 * arena.radius - 1e-9, (
                    "two scored discs overlap, so a unit would be counted in both")
            for x, y in standing:
                assert math.hypot(first[0] - x, first[1] - y) >= arena.radius - 1e-9, (
                    "a scored disc reaches something that was standing before the board was laid out")
    assert boards > 50, "the sweep has to place boards to be saying anything about them"


class _Paid:
    """A command layer that only remembers what it was paid, which is all `_finish_side` asks of one.

    It carries the count of closed errands a real learnt layer carries, because the arena reads its terminal count back off the layer rather than counting its own offers: what the arena knows is how many payments it held out, and only the layer knows how many of them reached a decision. Here every payment reaches one, which is the case of a layer with a decision of that squad still waiting.
    """

    def __init__(self):
        self.paid = {}
        self.terminals = {}

    def finish(self, squad, terminal, reason):
        self.paid[squad.id] = terminal
        self.terminals[reason] = self.terminals.get(reason, 0) + 1


def _contested(arena, region_id, point, weight):
    arena.contests.append(_Contest(region_id=region_id, point=point))
    arena.priorities[region_id] = weight
    arena.garrison_share[region_id] = 0.5


def test_a_scored_episode_says_which_discs_it_had_to_take_and_which_to_hold():
    """A final share on its own cannot say whether a region was one this side had to take from the enemy's garrison or one it only had to hold, and the two are the whole question when an arm's advantage might be that it declined to attack. So the record carries what each contested region started as, beside what it ended as."""
    arena = _arena(seed=7)
    _contested(arena, 4, (0.0, 0.0), 0.9)
    _contested(arena, 9, (2000.0, 0.0), 0.4)
    # Ours stands on one of the pair and the enemy's on its reflection, which is how the draw places them.
    arena.garrison_share = {4: 1.0, 9: 0.0}
    arena.squads = {}
    arena.our_ops = arena.their_ops = None
    arena.enemy = {}

    arena._score(_observation(units=[_unit(1, 0.0, 0.0), _unit(2, 2000.0, 0.0, hostile=1)]))
    record = arena.statistics.as_dict()

    assert sorted(record["shares"]) == sorted(record["held"]) == sorted(record["priorities"]) == [4, 9]
    assert record["held"] == {4: 1.0, 9: 0.0}
    assert record["priorities"] == {4: 0.9, 9: 0.4}
    # And what happened: the disc we held is ours, the one we would have had to take is the enemy's.
    assert record["shares"] == {4: 1.0, 9: 0.0}


def test_a_scored_episode_says_how_far_each_side_ended_from_its_contests():
    """Whether the staged squads reached the scored ground at all, recorded for BOTH sides, because one side's figures cannot say what they were meant to.

    The board is a point reflection, but the ground under it is not: this side stages from a site the map was searched for and the other from that site's reflection, which is wherever the reflection lands, with a march nothing can make congruent. The arena's answer to that has always been the script arm's self-play zero — and that bounds the asymmetry only under the script, since a fighter strong enough to exploit a shorter march would turn it into a score no arm comparison could tell from an operational difference. With both sides written down, a run can be asked whether the two deployments arrived alike instead of being trusted to have.

    Here they plainly did not: our squad ended inside the catchment and theirs a long way outside it, which is what a non-congruent march looks like in the record.
    """
    arena = _arena(seed=11, credit="region")
    _contested(arena, 4, (0.0, 0.0), 1.0)
    arena.our_ops = arena.their_ops = None
    arena.squads = {0: SquadRecord(id=0, doctrine=Doctrine.VANGUARD, members=[1])}
    arena.enemy = {1: SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[2])}

    arena._score(_observation(units=[_unit(1, 100.0, 0.0), _unit(2, 3000.0, 0.0, hostile=1)]))
    record = arena.statistics.as_dict()
    assert (record["our_alive"], record["our_in_catchment"], record["our_reach"]) == (1, 1, 100.0)
    assert (record["their_alive"], record["their_in_catchment"], record["their_reach"]) == (1, 0, 3000.0)

    # A side with nothing left on the board reads minus one rather than nought, which is outside the range of a distance and so cannot be read as having arrived.
    arena.enemy = {1: SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[])}
    arena._score(_observation(units=[_unit(1, 100.0, 0.0)]))
    assert arena.statistics.as_dict()["their_reach"] == -1.0


def test_a_terminal_is_read_from_where_its_disc_started_and_not_from_the_neutral_half():
    """An errand is worth what it changed, and what it changed cannot be read without knowing where the ground started.

    A garrison keeps its own disc through the horizon about eighty-eight times in a hundred whether or not a squad is sent to stand with it, and even four squads massed on an enemy's disc take it only about four times in ten. Read from the neutral half, those two rates pay a redundant defence about +0.38 of a priority and an assault about −0.10, so the most profitable errand a squad can be given is one that was going to be won without it, and the first layer trained on this arena duly learnt to attack nothing and lose nothing. Read from the disc's own opening, the same two rates pay the assault about +0.40 and the redundant defence about nothing, which is the quantity the side score is a priority-weighted mean of.
    """
    arena = _arena(seed=11, credit="region")
    _contested(arena, 4, (0.0, 0.0), 0.8)                 # the enemy's garrison stands here, so this disc has to be taken
    _contested(arena, 9, (2000.0, 0.0), 0.5)              # ours stands here, so this one only has to be kept
    arena.garrison_share = {4: 0.0, 9: 1.0}

    def paid(shares, sign=+1.0, baseline=OPENING_BASELINE):
        arena.opening_baseline = baseline
        squads = {}
        for slot, region in ((1, 4), (2, 9)):
            squad = SquadRecord(id=slot, doctrine=Doctrine.VANGUARD, members=[slot])
            squad.contract = dataclasses.replace(_CONTRACT, squad=slot, target_region=region)
            squads[slot] = squad
        ledger = _Paid()
        arena._finish_side(ledger, squads, arena._standings(squads, shares, sign, [_unit(1, 0.0, 0.0), _unit(2, 2000.0, 0.0)]))
        return ledger.paid

    # Both errands come off: the assault turned a disc from theirs to ours and is paid the whole of its priority, and the defence left a disc exactly where it started and is paid nothing for it.
    won = paid({4: 1.0, 9: 1.0})
    assert abs(won[1] - 0.8) < 1e-9
    assert abs(won[2]) < 1e-9

    # Both errands fail: the assault left the disc where it found it and is charged nothing for having tried, and the defence gave up ground that was already ours and is charged the whole of its priority.
    lost = paid({4: 0.0, 9: 0.0})
    assert abs(lost[1]) < 1e-9
    assert abs(lost[2] + 0.5) < 1e-9

    # The enemy is paid the same figures with the sign turned over, which is what keeps the self-play zero a statement about the board: the two sides' openings on one disc sum to one exactly as their final shares do.
    theirs = paid({4: 1.0, 9: 1.0}, sign=-1.0)
    assert abs(theirs[1] + won[1]) < 1e-9
    assert abs(theirs[2] + won[2]) < 1e-9

    # At the neutral baseline the reading that taught the layer to stay at home comes back, and both errands are paid for how the ground stands rather than for what they did to it.
    neutral = paid({4: 1.0, 9: 1.0}, baseline=0.0)
    assert abs(neutral[1] - 0.8 * 0.5) < 1e-9
    assert abs(neutral[2] - 0.5 * 0.5) < 1e-9


def test_the_two_credit_readings_pay_a_pile_of_squads_differently():
    """Two squads converge on one region and take it between them; a third is sent to a region it never reaches.

    Under the region reading each squad on the taken region is paid the whole of that region's domination, so being the second squad on a won region is worth exactly as much as being the first — the free-rider term. Under the marginal reading each is paid only what its own surviving units account for, so the two divide what they jointly produced, and the squad with nothing in any catchment is paid nothing either way.

    Neither reading pays a squad with nothing left on the board. The free-rider term is what makes the region reading able to teach a concentrated assault — every squad of the pile is paid for the pile's work — but a squad that no longer exists is not part of the pile, and paying it would attribute to a decision about it whatever its allies go on doing without it. Its figure is frozen where its last unit left it instead, which here is nought because it never reached the disc at all.
    """
    arena = _arena(seed=3)
    _contested(arena, 4, (0.0, 0.0), 1.0)

    # Two squads of one tank each inside the disc and an enemy tank beside them, so our share of the catchment is two thirds.
    units = [_unit(1, 0.0, 0.0), _unit(2, 10.0, 0.0), _unit(3, 20.0, 0.0, hostile=1)]
    shares = {4: 2.0 / 3.0}
    squads = {
        1: SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[1]),
        2: SquadRecord(id=2, doctrine=Doctrine.VANGUARD, members=[2]),
        3: SquadRecord(id=3, doctrine=Doctrine.VANGUARD, members=[]),
    }
    for squad in squads.values():
        squad.contract = dataclasses.replace(_CONTRACT, squad=squad.id, target_region=4)

    arena.credit = "region"
    region_ops = _Paid()
    arena._finish_side(region_ops, squads, arena._standings(squads, shares, +1.0, units))
    assert abs(region_ops.paid[1] - (2.0 / 3.0 - 0.5)) < 1e-9
    assert region_ops.paid[1] == region_ops.paid[2], "the region reading pays the pile in full, squad by squad"

    arena.credit = "marginal"
    arena.squads = squads
    marginal_ops = _Paid()
    arena._finish_side(marginal_ops, squads, arena._standings(squads, shares, +1.0, units))
    # With none of this side's staged squads the disc holds one hostile tank alone and reads nought, so the deployment moved two thirds of it and the two squads standing there divide that in proportion to what each still has in the disc.
    assert abs(marginal_ops.paid[1] - 1.0 / 3.0) < 1e-9
    assert abs(marginal_ops.paid[2] - marginal_ops.paid[1]) < 1e-9
    assert abs(marginal_ops.paid[1] + marginal_ops.paid[2] - 2.0 / 3.0) < 1e-9, (
        "the pile is paid exactly what it moved between them, no more and no less")
    # The third squad was sent to the same region and is not on the board at the horizon. Neither reading pays it: it has moved nothing since its last unit went, and the disc it was sent to was taken by others.
    assert region_ops.paid[3] == 0.0
    assert marginal_ops.paid[3] == 0.0

    # A third squad piles onto the same region. The region reading hands each of the three the whole of that region's outcome, so the side is paid three times over for one disc; the marginal reading hands the three one disc's movement to divide, so the total is the movement whatever the size of the pile. That, and not the figure any one squad happens to take, is the difference between the two.
    units.append(_unit(4, 30.0, 0.0))
    squads[3].members = [4]
    shares = {4: 3.0 / 4.0}
    piled = _Paid()
    arena.credit = "region"
    arena._finish_side(piled, squads, arena._standings(squads, shares, +1.0, units))
    assert abs(piled.paid[3] - (3.0 / 4.0 - 0.5)) < 1e-9
    assert abs(sum(piled.paid.values()) - 3.0 * (3.0 / 4.0 - 0.5)) < 1e-9
    marginal_piled = _Paid()
    arena.credit = "marginal"
    arena._finish_side(marginal_piled, squads, arena._standings(squads, shares, +1.0, units))
    assert abs(sum(marginal_piled.paid.values()) - 3.0 / 4.0) < 1e-9
    assert all(abs(value - 0.25) < 1e-9 for value in marginal_piled.paid.values()), (
        "three equal squads on one disc divide its movement equally")


def test_the_marginal_credit_is_the_change_the_squad_made_to_the_side_score():
    """What makes the marginal reading the right terminal is not a symmetry but an identity: what a squad is paid is exactly how much of this side's score its own units account for, region weights and all. Anything else would be paying a squad for something other than the quantity the arena is measured by.

    Two contested regions of different worth, a squad standing in each. The squad's credit has to equal the priority-weighted side score as it stands, less the same score computed with that squad's units off the board.
    """
    arena = _arena(seed=4)
    arena.credit = "marginal"
    _contested(arena, 4, (0.0, 0.0), 0.8)
    _contested(arena, 9, (2000.0, 0.0), 0.4)

    units = [_unit(1, 0.0, 0.0), _unit(2, 10.0, 0.0, hostile=1, health=50.0),
             _unit(3, 2000.0, 0.0, type_index=1), _unit(4, 2010.0, 0.0, hostile=1)]
    shares = {}
    for contest in arena.contests:
        our_worth, enemy_worth = arena._catchment_worths(units, contest.point)
        shares[contest.region_id] = our_worth / (our_worth + enemy_worth)

    squads = {}
    for squad_id, member, region in ((1, 1, 4), (2, 3, 9)):
        squad = SquadRecord(id=squad_id, doctrine=Doctrine.VANGUARD, members=[member])
        squad.contract = dataclasses.replace(_CONTRACT, squad=squad_id, target_region=region)
        squads[squad_id] = squad

    paid = _Paid()
    arena._finish_side(paid, squads, arena._standings(squads, shares, +1.0, units))

    def _weighted(states) -> float:
        """The side score without its division by the total weight, which is the scale the terminal is paid on."""
        total = 0.0
        for contest in arena.contests:
            our_worth, enemy_worth = arena._catchment_worths(states, contest.point)
            both = our_worth + enemy_worth
            share = our_worth / both if both > 0 else 0.5
            total += arena.priorities[contest.region_id] * (share - 0.5)
        return total

    standing = _weighted(units)
    for squad_id, member in ((1, 1), (2, 3)):
        without = _weighted([unit for unit in units if unit.id != member])
        assert abs(paid.paid[squad_id] - (standing - without)) < 1e-9


def test_the_terminal_count_is_the_payments_that_landed_and_not_the_offers_made():
    """A count of how many times the horizon offered a payment is one per staged squad by construction and says nothing at all. What has to be counted is how many of those offers reached a decision, because a payment that reaches no decision teaches nothing and is exactly what happens to a squad whose errand was replaced before the board was scored: its trajectory was cut when the new contract arrived, and there is nothing left for the terminal to be added to.

    So the arena reads the figure back off the layer's own count of closed errands, the way the engagement arena reads its own, rather than counting its own offers. Here two squads are offered a payment and one takes it. The old count would have said two, and a run whose terminals never reach a decision would have looked exactly like a run whose terminals all did.
    """
    arena = _arena(seed=5)
    _contested(arena, 4, (0.0, 0.0), 1.0)
    arena.garrison_share = {4: 0.0}

    rollout = Rollout()
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0)
    arena.our_ops = layer
    squads = {}
    for slot in (1, 2):
        squad = SquadRecord(id=slot, doctrine=Doctrine.VANGUARD, members=[slot])
        squad.contract = dataclasses.replace(_CONTRACT, squad=slot, target_region=4)
        squads[slot] = squad
    # The first squad still has a decision waiting to be paid; the second had its errand replaced earlier in the episode, so its trajectory was cut and it has nothing outstanding.
    layer.pending[1] = Step(state=[0.0], action=0, mask=[1.0], value=0.4, squad=1)
    rollout.add((0, 2), Step(state=[0.0], action=0, mask=[1.0], value=0.4, squad=2))
    rollout.cut((0, 2), reason="renewed")

    arena._finish_side(layer, squads, arena._standings(squads, {4: 1.0}, +1.0, [_unit(1, 0.0, 0.0), _unit(2, 10.0, 0.0)]))
    arena._tally()

    assert len(squads) == 2, "both squads have to be offered a payment or the count is not being told apart from the offers"
    assert arena.statistics.terminals == 1
    assert layer.terminals["horizon"] == 1
    # A layer that keeps no trajectories at all — every arm of the measuring runner — lands none of them, and nought is the truth for it rather than a fault.
    arena.our_ops = LearntOperations(None, None, None, rollout=None, instance=-1)
    arena._finish_side(arena.our_ops, squads, arena._standings(squads, {4: 1.0}, +1.0, [_unit(1, 0.0, 0.0)]))
    arena._tally()
    assert arena.statistics.terminals == 0


def test_an_episode_counts_the_decisions_its_squads_took_and_the_errands_they_were_split_into():
    """The arena pays one terminal per squad, at the horizon, to the errand that squad was on when the board was scored. So how much of an episode that payment can reach is decided by something the record did not contain: how long an errand ran.

    Two episodes that score identically can be completely different instruments. One in which four contracts stood from the staging point to the horizon pays every decision taken; one in which the contracts were re-drawn every period pays four decisions out of hundreds, because a squad handed a new contract has its trajectory cut and a cut trajectory is never paid a terminal at all. Nothing else in the record separates them, so the periods and the errands are counted here, off the contracts the layers wrote onto the squad records — which costs the same and means the same for a handwritten ladder, a pinned deployment and a network alike.
    """
    arena = _arena(seed=13)
    first = SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[1])
    second = SquadRecord(id=2, doctrine=Doctrine.VANGUARD, members=[2])
    arena.squads = {1: first, 2: second}
    first.contract = dataclasses.replace(_CONTRACT, squad=1, target_region=4, issued_at_ms=1000)

    # Two periods in which the first squad held one contract and the second held none at all.
    arena._survey()
    arena._survey()
    assert (arena.statistics.periods, arena.statistics.errands) == (2, 1)

    # A fresh contract for the first squad is a second errand; the same contract standing is not.
    first.contract = dataclasses.replace(first.contract, target_region=9, issued_at_ms=2000)
    second.contract = dataclasses.replace(_CONTRACT, squad=2, target_region=4, issued_at_ms=2000)
    arena._survey()
    arena._survey()
    assert (arena.statistics.periods, arena.statistics.errands) == (6, 3)
    assert arena.statistics.as_dict()["errands"] == 3

    # A squad re-tasked every period turns every decision into its own errand, which is the case the count exists to make visible: the ratio, and not the score, is what says whether the terminal reached anything.
    for issued in (3000, 4000, 5000):
        first.contract = dataclasses.replace(first.contract, issued_at_ms=issued)
        second.contract = dataclasses.replace(second.contract, issued_at_ms=issued)
        arena._survey()
    assert (arena.statistics.periods, arena.statistics.errands) == (12, 9)


def test_the_terminal_is_what_the_ground_came_to_against_where_it_opened_and_not_the_path_it_took():
    """What the arena pays a squad is a statement about two boards — the one its disc opened on and the one it was scored on — and about no board in between. That is what makes the measured rates quotable: a concentrated assault pays about four tenths of a priority and a redundant defence about nothing because those are statements about `priority * (final share − the share the disc opened at)`.

    A dense credit paid every period is the obvious cure for a terminal that reaches one decision in a hundred, and there are two ways to write one that look identical until a squad is re-tasked. Paying each period the movement of the disc the squad's standing contract names — the path the squad walked, region by region — does not sum to the terminal: it re-sets its origin at every change of contract, so a squad banks what its allies won on one disc and then steps onto a disc it cannot lose and keeps both. Paying instead the movement of the squad's own scored figure, each board read under the contract in force at that board and against the disc's own opening, is a difference of one quantity and telescopes to the terminal exactly, after any number of re-taskings.

    Measured here on the arena's own reading rather than argued. A squad is contracted to an enemy-held disc while its allies carry it two thirds of the way, then re-tasked onto a disc of its own that never moves. The horizon pays it nothing, because the ground it was on when the board was scored ended exactly where it opened. The path ledger pays it two thirds of a priority for ground it walked away from; the endpoint ledger pays it nought, which is the terminal.
    """
    arena = _arena(seed=17)
    _contested(arena, 4, (0.0, 0.0), 1.0)        # the enemy's garrison opened here, so this disc reads nought
    _contested(arena, 9, (2000.0, 0.0), 1.0)     # ours opened here, so this one reads whole
    arena.garrison_share = {4: 0.0, 9: 1.0}
    arena.credit = "region"

    # Three boards: the opening, one on which allies have taken two thirds of the enemy's disc, and the horizon on which they hold all of it. The squad's own disc never moves.
    opening = [_unit(1, 0.0, 0.0, hostile=1), _unit(2, 2000.0, 0.0)]
    middle = [_unit(1, 0.0, 0.0, hostile=1), _unit(2, 2000.0, 0.0),
              _unit(10, 20.0, 0.0), _unit(11, 40.0, 0.0)]
    horizon = [_unit(2, 2000.0, 0.0), _unit(10, 20.0, 0.0), _unit(11, 40.0, 0.0)]

    def reading(units, region):
        contest = next(c for c in arena.contests if c.region_id == region)
        our_worth, enemy_worth = arena._catchment_worths(units, contest.point)
        total = our_worth + enemy_worth
        return our_worth / total if total > 0 else 0.5

    def figure(units, region):
        """The squad's own scored figure on one board: what the disc its contract names has moved from the ownership that disc opened at, weighted by what the strategic layer said the region was worth."""
        return arena.priorities[region] * (reading(units, region) - arena.garrison_share[region])

    assert abs(reading(opening, 4)) < 1e-9 and abs(reading(middle, 4) - 2.0 / 3.0) < 1e-9
    assert abs(reading(horizon, 4) - 1.0) < 1e-9 and abs(reading(horizon, 9) - 1.0) < 1e-9

    # The contract in force over each period: the enemy's disc first, this side's own disc after the re-tasking.
    path = [(opening, 4, middle, 4), (middle, 4, horizon, 9)]
    walked = sum(arena.priorities[now_region] * (reading(now, now_region) - reading(before, now_region))
                 for before, before_region, now, now_region in path)
    telescoped = sum(figure(now, now_region) - figure(before, before_region)
                     for before, before_region, now, now_region in path)

    squad = SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[10, 11])
    squad.contract = dataclasses.replace(_CONTRACT, squad=1, target_region=9)
    ledger = _Paid()
    arena._finish_side(ledger, {1: squad}, arena._standings({1: squad}, {4: reading(horizon, 4), 9: reading(horizon, 9)}, +1.0, horizon))
    terminal = ledger.paid[1]

    assert abs(terminal) < 1e-9, "the disc this squad held at the horizon ended where it opened"
    assert abs(telescoped - terminal) < 1e-9, "the endpoint ledger did not sum to the terminal"
    assert abs(walked - 2.0 / 3.0) < 1e-9 and abs(walked - terminal) > 0.6, (
        "the path ledger paid the squad for a disc it was re-tasked away from")

    # The other way a dense credit stops telescoping, and it needs no re-tasking at all: under the movement reading a squad's figure depends on how much of it is standing in the disc, so an earlier board re-read with the members the squad has now is not the reading that board gave when it was current. Anything paying differences of readings has to hold the membership of the board it holds, or a squad that loses a tank is paid for the loss twice over.
    arena.credit = "marginal"
    arena.garrison_share = {4: 0.0, 9: 1.0}
    both = _tasked(1, 4, [10, 11])
    arena.squads = {1: both}
    arena._frozen.clear()
    whole = arena._standing(both, arena._shares(middle), +1.0, middle)
    lost = _tasked(1, 4, [10])
    arena.squads = {1: lost}
    arena._frozen.clear()
    survivor = arena._standing(lost, arena._shares(middle), +1.0, middle)
    assert abs(whole - 2.0 / 3.0) < 1e-9, "the squad holding the disc is paid what its side moved there"
    # Read with one member instead of two, the other tank is on the board and in no staged squad, so it sits inside the origin: the disc without this side's squads reads a half rather than nought, and the squad is paid only the sixth it added on top of it.
    assert abs(survivor - 1.0 / 6.0) < 1e-9
    assert survivor < whole, "the same board re-read with fewer members is a different reading"


# ---- the reading the periods are paid off ---------------------------------------------------

def _standing_board(credit="region"):
    """An arena with two discs of opposite ownership, which is the pair every property here needs: one this side has to take from the enemy's garrison and one it only has to keep.

    Built on the reading that pays the domination of the named region unless a caller asks otherwise, because the properties written about the hand-off — that a figure is read once per board, that it is total over a side's squads, that the two sides' figures are negatives — were all stated about that reading and are checked against its arithmetic by hand.
    """
    arena = _arena(seed=23, credit=credit)
    _contested(arena, 4, (0.0, 0.0), 0.8)              # the enemy's garrison opened here
    _contested(arena, 9, (2000.0, 0.0), 0.5)           # ours opened here
    arena.garrison_share = {4: 0.0, 9: 1.0}
    return arena


def _tasked(squad_id, region, members=()):
    squad = SquadRecord(id=squad_id, doctrine=Doctrine.VANGUARD, members=list(members))
    if region is not None:
        squad.contract = dataclasses.replace(_CONTRACT, squad=squad_id, target_region=region)
    return squad


def test_the_period_reading_and_the_horizon_reading_are_one_expression():
    """The invariant the whole dense credit rests on, and the one thing a later edit could break in silence.

    Every operational period is paid the movement of a squad's scored figure and the horizon is paid the same figure once more, so the payments telescope to the horizon's reading — but only for as long as the two readings are the same reading. Were the horizon to keep an expression of its own, the identity would be an intention that two pieces of code had to be kept in step, and the first time they drifted every episode would return the terminal plus whatever the drift came to, with nothing in any log to say so. So both go through `_read`, and this holds them to it: the horizon is run end to end and what it pays a layer is compared against what the period loop's own expression says of the very same board.
    """
    for credit in ("region", "marginal"):
        units = [_unit(1, 0.0, 0.0), _unit(2, 20.0, 0.0), _unit(3, 40.0, 0.0, hostile=1),
                 _unit(4, 2000.0, 0.0), _unit(5, 2030.0, 0.0, hostile=1, health=40.0)]
        arena = _standing_board()
        arena.credit = credit
        arena.squads = {1: _tasked(1, 4, [1, 2]), 2: _tasked(2, 9, [4]), 3: _tasked(3, 7, [3])}
        arena.enemy = {4: _tasked(4, 4, [3]), 5: _tasked(5, 9, [5])}
        arena.our_ops, arena.their_ops = _Paid(), _Paid()
        arena._score(_observation(units=units, game_time_ms=300000))

        shares = arena._shares(units)
        for ledger, squads, sign in ((arena.our_ops, arena.squads, +1.0),
                                     (arena.their_ops, arena.enemy, -1.0)):
            for squad in squads.values():
                assert ledger.paid[squad.id] == arena._standing(squad, shares, sign, units), (
                    "the horizon paid something the period loop does not read, so the payments cannot telescope")

    # And a squad sent at a region the board put no priority on moves no figure at all, which is what the early return says and what the horizon paid before there was one.
    arena = _standing_board()
    assert arena._standing(_tasked(3, 7), {4: 1.0, 9: 1.0}, +1.0) == 0.0
    assert arena._standing(_tasked(4, None), {4: 1.0, 9: 1.0}, +1.0) == 0.0


def test_a_squads_standing_is_antisymmetric_between_the_sides():
    """What is now paid every period used to be paid once, and the property that made the arena a measurement has to survive being paid a hundred and fifty times instead of once.

    The other side's share of a disc is one less this side's and its opening is one less this side's, so its figure is `w · ((1 − s) − (1 − o))`, the exact negative of `w · (s − o)` at every baseline. That is what the sign the arena carries stands for, and it has to hold on every board the period loop reads, not only on the one the horizon reads — otherwise a period would pay the two sides something other than a number and its negation, and the self-play zero would stop being a statement about the board.
    """
    worst = 0.0
    for trial in range(60):
        seed = random.Random(trial)
        for baseline in (0.0, 0.4, 1.0):
            # The property under test belongs to the reading that pays the whole domination of one named region, so the reading is named rather than taken from whatever the default happens to be.
            arena = _arena(seed=trial, credit="region")
            arena.opening_baseline = baseline
            _contested(arena, 4, (0.0, 0.0), seed.uniform(0.3, 1.0))
            _contested(arena, 9, (2000.0, 0.0), seed.uniform(0.3, 1.0))
            arena.garrison_share = {4: 0.0, 9: 1.0}
            units = []
            for index in range(seed.randrange(1, 7)):
                units.append(_unit(10 + index, seed.uniform(-300.0, 300.0), seed.uniform(-300.0, 300.0),
                                   hostile=seed.randrange(2), health=seed.uniform(1.0, 100.0)))
                units.append(_unit(30 + index, 2000.0 + seed.uniform(-300.0, 300.0),
                                   seed.uniform(-300.0, 300.0), hostile=seed.randrange(2),
                                   health=seed.uniform(1.0, 100.0)))
            shares = arena._shares(units)

            for region in (4, 9):
                squad = _tasked(1, region, [10])
                ours = arena._standing(squad, shares, +1.0, units)
                theirs = arena._standing(squad, shares, -1.0, units)
                worst = max(worst, abs(ours + theirs))
                # And read the long way round, as the mirror actually reads it: the other side's own share of the disc is one less ours and its own opening is one less ours, which is what the sign stands in for.
                mirror = _arena(seed=trial, credit="region")
                mirror.opening_baseline = baseline
                mirror.contests = list(arena.contests)
                mirror.priorities = dict(arena.priorities)
                mirror.garrison_share = {r: 1.0 - s for r, s in arena.garrison_share.items()}
                flipped = {r: 1.0 - s for r, s in shares.items()}
                worst = max(worst, abs(ours + mirror._standing(squad, flipped, +1.0, units)))
    assert worst < 1e-15, "a period's figures are not exact negatives between the sides: %.3e" % worst


def test_a_period_reads_every_disc_off_one_board():
    """One reading of one board, covering every disc and every squad, and taken before either side decides.

    Read per squad instead, a disc that two squads were sent to would be read twice; read per side, the leader alternation would hand the two sides boards a decision apart and their figures would stop being exact negatives. And the dict has to be total over the side's squads, including one contracted to ground the board put no priority on, because the layer must never have to fall back on a quantity of its own for a squad it cannot find — a trajectory paid partly in the arena's disc reading and partly in the game's region block sums to neither.
    """
    arena = _standing_board()
    units = [_unit(1, 0.0, 0.0), _unit(2, 20.0, 0.0, hostile=1), _unit(3, 2000.0, 0.0),
             _unit(4, 9000.0, 0.0)]
    shares = arena._shares(units)

    assert sorted(shares) == [4, 9]
    for contest in arena.contests:
        our_worth, enemy_worth = arena._catchment_worths(units, contest.point)
        total = our_worth + enemy_worth
        assert shares[contest.region_id] == (our_worth / total if total > 0 else 0.5)

    squads = {1: _tasked(1, 4, [1]), 2: _tasked(2, 9, [3]), 3: _tasked(3, 7, [4]), 4: _tasked(4, None)}
    standings = arena._standings(squads, shares, +1.0, units)
    assert sorted(standings) == [1, 2, 3, 4], "every squad the side has, or the layer needs a fallback of its own"
    assert standings[3] == 0.0 and standings[4] == 0.0
    assert abs(standings[1] - 0.8 * (0.5 - 0.0)) < 1e-9      # one of ours against one hostile is a half share
    assert abs(standings[2] - 0.5 * (1.0 - 1.0)) < 1e-9      # the disc we opened whole is still whole


class _Chain:
    """An operational layer that records the order it was called in and what it was handed, so the arena's hand-off can be checked rather than assumed."""

    def __init__(self, calls, name):
        self.calls = calls
        self.name = name
        self.standings = []

    def standing(self, figures):
        self.calls.append((self.name, "standing"))
        self.standings.append(dict(figures))

    def decide(self, view, orders, squads, reports, now):
        self.calls.append((self.name, "decide"))
        return [], []


class _Still:
    """A tactical layer that moves nothing, so a period can be driven without one."""

    def decide(self, view, squads, now):
        return [], []


def test_the_arena_hands_the_layer_its_standing_before_it_decides():
    """The hand-off has to arrive before the decision, because settling is what pays the decision the last period left waiting and settling happens at the top of the layer's own decide. Handed over afterwards it would pay every period out of the board of the period before, and the last one out of nothing at all.

    Both sides are handed the same period's reading of the same board, which is what the alternating leader would otherwise break: the two sides decide one after the other, and a reading taken inside that loop would give the second side a board the first has already acted on.
    """
    arena = _standing_board()
    calls = []
    arena.our_ops = _Chain(calls, "ours")
    arena.their_ops = _Chain(calls, "theirs")
    arena.our_tac = arena.their_tac = _Still()
    arena.orders = OperationsOrders(posture=Posture.ARM, priorities=dict(arena.priorities),
                                    offensive=True, loss_allowance=1000.0)
    # One squad a side, both sent at the disc the enemy's garrison opened on, so the two figures are read off one disc and must come out as a number and its negation.
    arena.squads = {1: _tasked(1, 4, [1])}
    arena.enemy = {5: _tasked(5, 4, [2])}
    arena.until_ms = 999999

    units = [_unit(1, 0.0, 0.0), _unit(2, 30.0, 0.0, hostile=1), _unit(3, 60.0, 0.0, hostile=1)]
    observation = _observation(units=units)
    view = WorldView(observation=observation, catalogue=_CATALOGUE, regions=[])
    arena._run(observation, view, Action(), 5000)

    assert calls == [("ours", "standing"), ("ours", "decide"),
                     ("theirs", "standing"), ("theirs", "decide")], calls
    ours, = arena.our_ops.standings
    theirs, = arena.their_ops.standings
    assert sorted(ours) == [1] and sorted(theirs) == [5]
    assert abs(ours[1] + theirs[5]) < 1e-15, "the two sides were paid off different boards"
    # And the figure is the board's own reading: one of ours against two hostiles is a third of the disc, off an opening of nought.
    assert abs(ours[1] - 0.8 * (1.0 / 3.0 - 0.0)) < 1e-9

    # The leader alternates period by period, and the reading is taken outside that alternation, so the second period's figures are still exact negatives.
    arena._run(observation, view, Action(), 7000)
    assert [name for name, _ in calls[4:]] == ["theirs", "theirs", "ours", "ours"]
    assert abs(arena.our_ops.standings[-1][1] + arena.their_ops.standings[-1][5]) < 1e-15


def test_the_horizon_pays_only_what_the_periods_have_not():
    """End to end on the real layer and the real buffer: the arena reads its discs every period, the layer is paid the movement of each squad's own figure, and the horizon adds what is left. What the whole episode returns has to be the horizon's reading and nothing more, or the dense credit has changed the objective instead of only its density.

    And nothing is cut. The layer re-draws its region every period and used to have its trajectory cut every time it did, which is what left four fifths of a batch carrying an advantage of exactly nought; paid off one quantity from the first decision to the last there is no boundary left to cut at, so the census reads one finished trajectory a squad and every step paid.
    """
    arena = _standing_board()
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    # The real layer over the real inherited ladder, so the contracts are written the way a run writes them.
    layer = LearntOperations(None, _CATALOGUE, None, rollout=rollout, instance=0, discount=1.0)
    arena.our_ops = layer
    arena.their_ops = _Chain([], "theirs")
    arena.our_tac = arena.their_tac = _Still()
    arena.orders = OperationsOrders(posture=Posture.ARM, priorities=dict(arena.priorities),
                                    offensive=True, loss_allowance=1000.0)
    squad = _tasked(1, 4, [1])
    arena.squads = {1: squad}
    arena.enemy = {}
    arena.until_ms = 999999

    def board(ours_in_disc, enemy_in_disc, region):
        """A board with a stated number of tanks a side inside the disc the enemy opened on, and the squad re-tasked as asked."""
        units = [_unit(10 + index, index * 20.0, 0.0) for index in range(ours_in_disc)]
        units += [_unit(50 + index, 100.0 + index * 20.0, 0.0, hostile=1) for index in range(enemy_in_disc)]
        units.append(_unit(90, 2000.0, 0.0))          # our garrison, still holding the disc we opened whole
        squad.members = [10 + index for index in range(ours_in_disc)]
        squad.contract = dataclasses.replace(_CONTRACT, squad=1, target_region=region,
                                             issued_at_ms=1000 * region)
        return units

    # Five periods: the squad arrives at the enemy's disc and takes ground, is re-tasked onto its own, and is re-tasked back.
    for period, (ours_in, enemy_in, region) in enumerate(
            ((0, 3, 4), (1, 3, 4), (2, 2, 4), (2, 2, 9), (3, 1, 4))):
        units = board(ours_in, enemy_in, region)
        observation = _observation(units=units, game_time_ms=2000 * period)
        view = WorldView(observation=observation, catalogue=_CATALOGUE, regions=[])
        # A decision of this squad's is recorded every period, as a layer with a decider takes one.
        arena._run(observation, view, Action(), 2000 * period)
        layer.pending[1] = Step(state=[0.0], action=0, mask=[1.0], value=0.1 * (period + 1), squad=1)

    # And the horizon, on a board that has moved again since the last period.
    units = board(4, 1, 4)
    horizon = _observation(units=units, game_time_ms=999999)
    arena._score(horizon)

    terminal = arena._standing(squad, arena._shares(units), +1.0, units)
    trajectory, = rollout.done
    assert trajectory.finished and trajectory.reason == "" and trajectory.steps[-1].done
    assert abs(sum(step.reward for step in trajectory.steps) - terminal) < 1e-12, (
        "the episode returned something other than the horizon's own reading")
    assert arena.statistics.terminals == 1 and layer.terminals["horizon"] == 1

    rollout.drain()
    census = rollout.census
    assert census.cut == {} and census.finished == 1
    assert census.paid_steps == census.steps == 5
    assert census.zero_advantage == 0


def test_finishing_the_fight_pays_a_squad_more_and_not_less():
    """The credit has to move the same way the score does, and under the leave-one-out counterfactual it moved the opposite way on the errand this arena is for.

    One disc a garrison opened, three tanks of ours in it against its one. The disc reads three quarters, the deployment moved it three quarters from an opening of nought, and the one squad standing there takes all of that. The same three then destroy the garrison: the disc reads whole and the side score rises from a quarter to a half. Asked what the disc would read with THIS squad's units removed, the answer was an empty disc at a neutral half, so the squad's pay FELL to a half — a third of its pay taken away for winning the fight, in the very quantity it is paid in. Asked instead what the disc moved and how much of that move this squad is standing on, it rises to the whole priority, which is the direction the score moved.
    """
    for credit in ("board", "marginal"):
        arena = _arena(seed=5, credit=credit)
        _contested(arena, 4, (0.0, 0.0), 1.0)
        arena.garrison_share = {4: 0.0}
        squad = _tasked(1, 4, [10, 11, 12])
        arena.squads = {1: squad}
        ours = [_unit(10, 0.0, 0.0), _unit(11, 20.0, 0.0), _unit(12, 40.0, 0.0)]

        contested = ours + [_unit(50, 60.0, 0.0, hostile=1)]
        arena._frozen.clear()
        before = arena._standing(squad, arena._shares(contested), +1.0, contested)
        arena._frozen.clear()
        after = arena._standing(squad, arena._shares(ours), +1.0, ours)

        assert abs(arena._side_score(contested) - 0.25) < 1e-9
        assert abs(arena._side_score(ours) - 0.5) < 1e-9, "the score has to rise when the defender dies"
        assert abs(before - 0.75) < 1e-9 and abs(after - 1.0) < 1e-9, (
            "%s: the squad was not paid more for finishing the fight" % credit)

    # A disc taken outright, by squads of several units each with health left over, is the case the whole reading exists for and the one an arithmetic shortcut loses.
    #
    # The origin is the disc read with none of this side's staged squads in it. Written as this side's worth less what its squads hold, it is two sums accumulated in different groupings — one running total over the board's rows against a total per squad added up afterwards — and their difference carries a residue of the last bits. Over six hundred random boards of two to four squads of two to four units, the two forms disagree on more than two in five, and the residue divides one leftover by another: the worst produces an origin of one where the honest answer is nought, which pays a side that has just taken a disc outright exactly nothing. Summed over the units that are left, an emptied disc is empty.
    for trial in range(30):
        arena = _arena(seed=200 + trial, credit="board")
        _contested(arena, 4, (0.0, 0.0), 1.0)
        arena.garrison_share = {4: 0.0}
        rng = random.Random(trial)
        squads, units, unit_id = {}, [], 10
        for index in range(rng.randrange(2, 5)):
            members = []
            for _ in range(rng.randrange(2, 5)):
                units.append(_unit(unit_id, (unit_id % 7) * 9.0, 0.0, health=rng.uniform(0.5, 100.0)))
                members.append(unit_id)
                unit_id += 1
            squads[index] = _tasked(index, 4, members)
        arena.squads = squads
        figures = arena._standings(squads, arena._shares(units), +1.0, units)
        assert abs(sum(figures.values()) - 1.0) < 1e-9, (
            "board %d: a disc taken outright paid its takers %.6f of the priority between them"
            % (trial, sum(figures.values())))
        assert all(value > 0.0 for value in figures.values())

    # And the other half of the same statement: a pile divides one disc's movement rather than each member taking it whole, which is what the leave-one-out reading was built for and what it stopped doing the moment the pile succeeded.
    arena = _arena(seed=5, credit="marginal")
    _contested(arena, 4, (0.0, 0.0), 1.0)
    arena.garrison_share = {4: 0.0}
    taken = [_unit(10, 0.0, 0.0), _unit(20, 20.0, 0.0)]      # an ally's tank is in the disc as well
    piled, ally = _tasked(1, 4, [10]), _tasked(2, 4, [20])
    arena.squads = {1: piled, 2: ally}
    figures = arena._standings(arena.squads, arena._shares(taken), +1.0, taken)
    assert abs(figures[1] - figures[2]) < 1e-9 and abs(sum(figures.values()) - 1.0) < 1e-9


# ---- the reading that prices when the ground was taken --------------------------------------

def _held(arena, at_ms, ours_in_disc, enemy_in_disc):
    """Drives one operational period on a board carrying a stated number of tanks a side inside the disc the enemy opened on, and returns the board it was driven on. Each side's tanks are that side's squad's members, so both sides' figures are read off units that are actually there."""
    units = [_unit(10 + index, index * 20.0, 0.0) for index in range(ours_in_disc)]
    units += [_unit(50 + index, 100.0 + index * 20.0, 0.0, hostile=1) for index in range(enemy_in_disc)]
    units.append(_unit(90, 2000.0, 0.0))              # our garrison, still holding the disc we opened whole
    arena.squads[1].members = [10 + index for index in range(ours_in_disc)]
    if 5 in arena.enemy:
        arena.enemy[5].members = [50 + index for index in range(enemy_in_disc)]
    observation = _observation(units=units, game_time_ms=at_ms)
    view = WorldView(observation=observation, catalogue=_CATALOGUE, regions=[])
    arena._run(observation, view, Action(), at_ms)
    return units


def _tenure_board(tenure="tenure", credit="region"):
    """An arena of two discs with one squad a side, wound up so that periods can be driven straight into it."""
    arena = _standing_board(credit=credit)
    arena.tenure = tenure
    arena.our_ops, arena.their_ops = _Chain([], "ours"), _Chain([], "theirs")
    arena.our_tac = arena.their_tac = _Still()
    arena.orders = OperationsOrders(posture=Posture.ARM, priorities=dict(arena.priorities),
                                    offensive=True, loss_allowance=1000.0)
    arena.squads = {1: _tasked(1, 4, [])}
    arena.enemy = {5: _tasked(5, 4, [])}
    arena.until_ms = 999999
    arena.horizon_ms = 8000
    arena._marked_ms = 0
    return arena


def test_the_tenure_reading_is_the_mean_of_the_periods_weighted_by_how_long_each_stood():
    """What the tenure reading is, stated as arithmetic and checked against the readings it is a mean of.

    Each stretch of the horizon counts once, weighted by its own share of the horizon, and the weight is time rather than a count of readings — a mean over readings would depend on how often the region block happened to ride a frame, which is a property of the wire and not of the deployment. So a figure that stood for a quarter of the horizon enters at a quarter, whether it was read once in that quarter or ten times.
    """
    arena = _tenure_board()
    readings = []
    for at_ms, ours_in in ((2000, 0), (4000, 1), (6000, 2), (8000, 3)):
        units = _held(arena, at_ms, ours_in, 3)
        readings.append(arena._standing(arena.squads[1], arena._shares(units), +1.0, units))

    # Four equal stretches of a two-thousand-millisecond horizon eighth apiece: the mean is the plain mean of the four.
    assert abs(arena._accrued[1] - sum(readings) / 4.0) < 1e-12, (
        "the tenure is not the time-weighted mean of the readings it was accrued from")
    # And the last reading alone is not it, which is the whole difference between the two readings of an episode.
    assert abs(arena._accrued[1] - readings[-1]) > 0.05

    # The same ground held over stretches of unequal length is weighted by the length and not by the count.
    uneven = _tenure_board()
    _held(uneven, 6000, 0, 3)                          # three quarters of the horizon holding nothing
    first = uneven._accrued[1]
    units = _held(uneven, 7000, 3, 1)
    last = uneven._standing(uneven.squads[1], uneven._shares(units), +1.0, units)
    _held(uneven, 8000, 3, 1)
    assert abs(uneven._accrued[1] - (first + 0.25 * last)) < 1e-12


def test_a_squad_that_is_gone_accrues_no_further_tenure():
    """A wiped squad's figure is frozen where its last unit left it, and that freeze belongs to the horizon reading alone.

    There it prevents a punishment, since the horizon pays differences and a figure falling to nought at the moment of death would charge a squad for dying after it had earned. Carried into the tenure reading it becomes a payment for ground the squad is not standing on: a squad that takes a disc in the first period and is annihilated would accrue, over the rest of the horizon, exactly what a squad that held the same disc all the way accrues, and the one distinction the reading exists to make would not be made for anything that dies. At the shipped three-hundred-second horizon a squad wiped at ten seconds would keep about all of a full hold.
    """
    holds, dies = _tenure_board(), _tenure_board()
    for at_ms in (2000, 4000, 6000, 8000):
        _held(holds, at_ms, 3, 0)
    _held(dies, 2000, 3, 0)
    for at_ms in (4000, 6000, 8000):
        _held(dies, at_ms, 0, 0)

    assert dies._accrued[1] < holds._accrued[1] - 0.3, (
        "a squad annihilated after one period accrued what a squad that held the disc to the horizon accrued")
    # And it is not charged for having died either: the total stops growing rather than falling back, so the quarter of the horizon it did hold the disc for is still in it.
    assert abs(dies._accrued[1] - 0.25 * holds._accrued[1]) < 1e-12


def test_the_tenure_reading_is_exactly_antisymmetric_between_the_sides():
    """The self-play zero is what makes this arena a measurement, and it has to hold of the tenure reading exactly as it holds of the horizon reading, or a run paid off the tenure could not be gated at all.

    It holds for the reason the horizon's does and needs no separate argument: every reading the mean is accrued from is taken off one board, before either side decides, and each of those readings is already a number and its negation. A mean of exact negatives is an exact negative. What this checks is that the accrual really is fed one reading of one board — an accrual driven per side, or one that read the board twice, would break it while every figure on its own still looked right.
    """
    arena = _tenure_board()
    scores = []
    for at_ms, ours_in, enemy_in in ((2000, 1, 3), (4000, 2, 3), (6000, 3, 2), (8000, 4, 1)):
        units = _held(arena, at_ms, ours_in, enemy_in)
        scores.append(arena._side_score(units))
        assert abs(arena._accrued[1] + arena._accrued[5]) < 1e-15, (
            "the two sides' tenures are not exact negatives, so the self-play zero is not a statement about the board")
        assert abs(arena.our_ops.standings[-1][1] + arena.their_ops.standings[-1][5]) < 1e-15

    # And the side score's own tenure, which is what a run pools, is the same mean of the same stretches. Antisymmetric for the reason each of its terms is: the other side's side score is this one's negated, so a mean of them is too.
    assert abs(arena._side_accrued - sum(scores) / 4.0) < 1e-12


def test_a_run_paid_the_tenure_returns_the_mean_and_not_the_last_reading():
    """End to end on the real layer and the real buffer, exactly as the horizon reading is checked: what the whole episode returns has to be the quantity the run says it pays, and under the tenure setting that is the mean over the horizon and not the reading at it.

    The two are made to differ by construction — the squad holds the disc for most of the episode and is driven off it at the end — so an episode that returned the last reading would return a visibly different number rather than the same one by luck.
    """
    arena = _tenure_board(credit="region")
    arena.horizon_ms = 10000
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    layer = LearntOperations(None, _CATALOGUE, None, rollout=rollout, instance=0, discount=1.0)
    arena.our_ops = layer

    # The disc stays in the enemy's hands for three quarters of the horizon and is taken at the last, so the mean over the episode and the reading at the end of it are visibly different numbers.
    for period, (at_ms, ours_in, enemy_in) in enumerate(
            ((2000, 0, 3), (4000, 0, 3), (6000, 0, 3), (8000, 3, 0))):
        _held(arena, at_ms, ours_in, enemy_in)
        layer.pending[1] = Step(state=[0.0], action=0, mask=[1.0], value=0.1 * (period + 1), squad=1)

    units = [_unit(10, 0.0, 0.0), _unit(11, 20.0, 0.0), _unit(12, 40.0, 0.0), _unit(90, 2000.0, 0.0)]
    arena.squads[1].members = [10, 11, 12]
    arena._score(_observation(units=units, game_time_ms=10000))

    trajectory, = rollout.done
    total = sum(step.reward for step in trajectory.steps)
    horizon = arena._standing(arena.squads[1], arena._shares(units), +1.0, units)
    assert abs(total - arena._accrued[1]) < 1e-12, "the episode returned something other than its own tenure"
    assert abs(horizon - 0.8) < 1e-9 and abs(total - 0.8 * 2.0 / 5.0) < 1e-9, (
        "the two readings were built to differ, and the episode returned the reading at the horizon")
    assert arena.statistics.side_score != arena.statistics.side_tenure


def test_two_deployments_that_end_alike_are_told_apart_by_when_they_took_the_ground():
    """The property the reading exists for, and the one the horizon reading cannot have.

    Two deployments end the horizon on the same ground: one takes the disc in the first period and keeps it, the other spends the episode elsewhere and walks onto it at the last. A match pays these differently — ground is upstream of income, and a region held from the third minute pays its owner for the rest of the match — while an arena that reads the discs only when the clock stops calls them equal. So the horizon figures must agree and the tenures must not.
    """
    early, late = _tenure_board(), _tenure_board()
    for at_ms, ours_in in ((2000, 3), (4000, 3), (6000, 3), (8000, 3)):
        held = _held(early, at_ms, ours_in, 0)
    for at_ms, ours_in in ((2000, 0), (4000, 0), (6000, 0), (8000, 3)):
        walked = _held(late, at_ms, ours_in, 0)

    assert abs(early._standing(early.squads[1], early._shares(held), +1.0, held)
               - late._standing(late.squads[1], late._shares(walked), +1.0, walked)) < 1e-12, (
        "the two deployments were built to end the horizon alike")
    assert early._accrued[1] > late._accrued[1] + 0.3, (
        "the tenure reading does not tell an errand that held the ground from one that arrived at the horizon")
    assert early._side_accrued > late._side_accrued + 0.15


def test_the_concentrating_arm_sends_every_squad_at_the_one_region_most_wanted():
    """The arm that actually concentrates replaces the region and keeps the doctrine's task, so squads of different doctrines converge on one place while still doing different things there. Setting the ladder's crowding discount to nought does not do this on the arena, where that discount is already nought — which is why this arm exists as well as that one."""
    regions = [
        RegionState(id=1, resources=3, held_by_us=1, held_by_enemy=0, x=0.0, y=0.0,
                    our_value=500.0, enemy_value=200.0, enemy_seen_at_ms=0, distance_from_home=300.0),
        RegionState(id=2, resources=1, held_by_us=0, held_by_enemy=1, x=100.0, y=0.0,
                    our_value=0.0, enemy_value=1000.0, enemy_seen_at_ms=0, distance_from_home=900.0),
    ]
    view = WorldView(observation=_observation(), catalogue=_CATALOGUE, regions=regions)
    # Region 2 is what the strategic layer wants; the ladder's own discounts would send a vanguard elsewhere for its distance.
    orders = OperationsOrders(posture=Posture.ARM, priorities={1: 0.2, 2: 1.0}, offensive=True, loss_allowance=1000.0)

    ladder, massed = Operations(None, _CATALOGUE), Concentrated(None, _CATALOGUE)
    for doctrine in (Doctrine.VANGUARD, Doctrine.GARRISON, Doctrine.RAID):
        squad = SquadRecord(id=1, doctrine=doctrine, value=1000.0)
        plain = ladder._pick(view, orders, squad, None)
        massed_pick = massed._pick(view, orders, squad, None)
        assert massed_pick is not None and massed_pick[1].id == 2, doctrine
        # The task is the doctrine's own, untouched: only where it goes is overridden.
        assert plain is not None and massed_pick[0] == plain[0], doctrine

    # A board on which the strategic layer wants nothing leaves the ladder's answer alone.
    barren = OperationsOrders(posture=Posture.ARM, priorities={}, offensive=True, loss_allowance=1000.0)
    squad = SquadRecord(id=1, doctrine=Doctrine.VANGUARD, value=1000.0)
    assert massed._pick(view, barren, squad, None) == ladder._pick(view, barren, squad, None)


def test_the_massed_arm_is_the_ladder_with_only_its_spreading_term_removed():
    """The arena's massed arm has to differ from the script arm in exactly one thing: the discount a region takes for the strength we already have standing in it. Everything else the ladder weighs — the priority, the march, the resources, the threat — must still be weighed, or the arm would answer a different question from the one it was built to ask.

    Two contested regions, the second worth more to the strategic layer and already holding a squad of ours. With the spreading term the ladder goes to the emptier one; with it at nought the ladder goes to the one it ranks higher, which is what massing on the best region means.
    """
    # The gap in priority is deliberately smaller than the discount a squad-and-a-half of our own strength earns the second region, so the two arms are made to disagree by the term under test and by nothing else.
    priorities = {1: 0.8, 2: 0.9}
    regions = [
        RegionState(id=1, resources=0, held_by_us=0, held_by_enemy=0, x=0.0, y=0.0,
                    our_value=0.0, enemy_value=1000.0, enemy_seen_at_ms=0, distance_from_home=500.0),
        RegionState(id=2, resources=0, held_by_us=0, held_by_enemy=0, x=100.0, y=0.0,
                    our_value=4000.0, enemy_value=1000.0, enemy_seen_at_ms=0, distance_from_home=500.0),
    ]
    view = WorldView(observation=_observation(), catalogue=_CATALOGUE, regions=regions)
    orders = OperationsOrders(posture=Posture.ARM, priorities=priorities, offensive=True, loss_allowance=1000.0)
    squad = SquadRecord(id=1, doctrine=Doctrine.VANGUARD, value=1000.0)

    spread = Operations(None, _CATALOGUE)._pick(view, orders, squad, None)
    massed = Operations(None, _CATALOGUE, crowding=0.0)._pick(view, orders, squad, None)
    assert spread is not None and massed is not None
    assert spread[1].id == 1, "the ladder as written spreads away from the region it already stands in"
    assert massed[1].id == 2, "with the spreading term at nought it takes the region it ranks highest"

    # The priority is still what decides between two regions neither of which we stand in, so removing the term removed the spreading and nothing else.
    regions[1] = dataclasses.replace(regions[1], our_value=0.0)
    assert Operations(None, _CATALOGUE, crowding=0.0)._pick(view, orders, squad, None)[1].id == 2
    assert Operations(None, _CATALOGUE)._pick(view, orders, squad, None)[1].id == 2


def test_the_priority_admits_a_region_the_doctrine_would_not_have_considered():
    """A doctrine's candidate filter is written about what it likes, not about what it may do, so a region the strategic layer asked for can fail the filter and never be scored at all. The priority admits it instead: it joins the candidates wherever the doctrine could act there, and the scores decide as before.

    A garrison holding two regions, one that pays and that nobody asked for, one that pays nothing and that the strategic layer wants. Before admission the wanted region is not a candidate at all, so no amount of priority could have chosen it.
    """
    regions = [
        RegionState(id=1, resources=4, held_by_us=1, held_by_enemy=0, x=0.0, y=0.0,
                    our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=0.0),
        RegionState(id=2, resources=0, held_by_us=1, held_by_enemy=0, x=100.0, y=0.0,
                    our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=0.0),
    ]
    view = WorldView(observation=_observation(), catalogue=_CATALOGUE, regions=regions)
    orders = OperationsOrders(posture=Posture.ARM, priorities={2: 0.9}, offensive=True, loss_allowance=1000.0)
    squad = SquadRecord(id=1, doctrine=Doctrine.GARRISON, value=1000.0)

    assert Operations(None, _CATALOGUE, admit=False)._pick(view, orders, squad, None)[1].id == 1
    assert Operations(None, _CATALOGUE)._pick(view, orders, squad, None)[1].id == 2

    # Admission adds and never removes: a priority too small to beat what the region pays leaves the ladder's own answer standing.
    faint = OperationsOrders(posture=Posture.ARM, priorities={2: 0.1}, offensive=True, loss_allowance=1000.0)
    assert Operations(None, _CATALOGUE)._pick(view, faint, squad, None)[1].id == 1


def test_admission_keeps_each_doctrine_inside_what_it_may_legally_do():
    """What a doctrine may be sent to is not what it prefers. A garrison may hold any ground of ours; a raider may go anywhere that is not ours; a vanguard on the defensive may press only where we already stand. Admission widens a candidate set up to those limits and no further, or it would hand a raider an errand on ground we are already holding."""
    regions = [
        # The first is in contact as well as ours, so a vanguard told not to press has an errand of its own and the ladder's "nothing in contact" fallback does not fire.
        RegionState(id=1, resources=4, held_by_us=1, held_by_enemy=0, x=0.0, y=0.0,
                    our_value=0.0, enemy_value=500.0, enemy_seen_at_ms=0, distance_from_home=0.0),
        RegionState(id=2, resources=0, held_by_us=1, held_by_enemy=0, x=100.0, y=0.0,
                    our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=0.0),
        RegionState(id=3, resources=4, held_by_us=0, held_by_enemy=0, x=200.0, y=0.0,
                    our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=200.0),
    ]
    view = WorldView(observation=_observation(), catalogue=_CATALOGUE, regions=regions)
    ladder = Operations(None, _CATALOGUE)

    # The raider is asked for ground we hold. It may not go there, so it keeps the ground it could take.
    wants_ours = OperationsOrders(posture=Posture.ARM, priorities={2: 0.9}, offensive=True, loss_allowance=1000.0)
    raider = SquadRecord(id=1, doctrine=Doctrine.RAID, value=1000.0)
    assert ladder._pick(view, wants_ours, raider, None)[1].id == 3

    # The vanguard is asked for ground we do not hold while the strategic layer is not pressing. Not pressing is the strategic layer's own switch, so a priority may not be read as permission to press.
    wants_theirs = OperationsOrders(posture=Posture.ARM, priorities={3: 0.9}, offensive=False, loss_allowance=1000.0)
    vanguard = SquadRecord(id=2, doctrine=Doctrine.VANGUARD, value=1000.0)
    assert ladder._pick(view, wants_theirs, vanguard, None)[1].id != 3

    # And a vanguard that is pressing may, since the whole board is legal to it then.
    pressing = OperationsOrders(posture=Posture.ARM, priorities={3: 0.9}, offensive=True, loss_allowance=1000.0)
    assert ladder._pick(view, pressing, vanguard, None)[1].id == 3


# ---- the tactical layer that fights beneath both sides --------------------------------------

class _OneNetwork:
    """Stands in for the single network and batching server a frozen tactical layer is read through, so a test can say whether both sides of the board were built off one of them or off two."""


def test_both_sides_of_the_mirror_are_built_from_the_one_tactical_factory():
    """The arena is handed one tactical factory and calls it once for each side, so trained parameters frozen under it fight for the enemy exactly as they fight for us. There is deliberately no way to put a fighter under one side alone: the board is a single reflection about one centre and the trust gate is the pooled self-play zero of the script arm, so two sides that fought differently would stop being exchangeable and that mean would no longer have to be nought.

    What one factory must not mean is one layer. A tactical layer keeps per-side state — what each of its squads has destroyed, and the board it last saw — and the two sides are handed different boards, so the two objects have to be distinct while the network behind them is one. Both halves of that are checked here, because either one alone would look right.
    """
    session = _Session(_grid())
    decider = _OneNetwork()
    built = []

    def tactics(given, catalogue):
        built.append((given, catalogue))
        return LearntTactics(given, catalogue, decider, None, -1)

    with _without_the_game_installed():
        arena = OpsArena(session, tactics=tactics, seed=5)
    assert len(built) == 2, "the arena did not build both of its tactical layers from the factory it was handed"
    # Both are handed the arena's own session and the arena's own catalogue, which is why the layers are built in here at all: a layer classifying a unit from some other type table would sort the same tank into a different role from the arena that spawned it.
    assert all(given is session and catalogue is arena.catalogue for given, catalogue in built)
    assert isinstance(arena.our_tac, LearntTactics) and isinstance(arena.their_tac, LearntTactics)
    assert arena.our_tac is not arena.their_tac, "one shared layer would fold the two sides' bookkeeping together"
    assert arena.our_tac.decider is decider and arena.their_tac.decider is decider, (
        "the two sides are reading different networks, so they are not fighting under one frozen layer")

    # Named no factory, the arena builds the handwritten ladder for itself on both sides, which is what every measurement taken on this arena so far was made under — and two of it, for the same reason.
    with _without_the_game_installed():
        plain = OpsArena(session, seed=5)
    assert type(plain.our_tac) is Tactics and type(plain.their_tac) is Tactics
    assert plain.our_tac is not plain.their_tac
    # And nothing about the operational layers moved: with no operational factory both sides are still the script chain.
    assert type(plain.our_ops) is Operations and type(plain.their_ops) is Operations


def test_an_episode_says_which_tactical_layer_it_was_made_under_before_it_has_scored_anything():
    """Two runs made under different tactical layers are two different instruments, and pairing them board by board would read the change of fighter as a difference between the operational arms. So the episode record has to carry which layer was beneath it, and carry it from construction rather than from scoring: an episode cut off before its horizon still has to say what instrument it was run on."""
    session = _Session(_grid())
    with _without_the_game_installed():
        arena = OpsArena(session, tactics_name="sha256:0123456789abcdef", seed=5)
    assert not arena.statistics.scored
    assert arena.statistics.tactics == "sha256:0123456789abcdef"
    assert arena.statistics.as_dict()["tactics"] == "sha256:0123456789abcdef"

    # And a run that named nothing says so in the one word a comparison reads as the handwritten ladder.
    assert OpsStatistics().tactics == SCRIPT_TACTICS
    with _without_the_game_installed():
        assert OpsArena(session, seed=5).statistics.as_dict()["tactics"] == SCRIPT_TACTICS


def test_an_episode_says_which_operational_policy_played_it():
    """An arm's name is a nickname and the policy is the identity, so the episode carries the identity.

    A learnt arm is named after the file its parameters were read from, and that file changes underneath itself: a training run overwrites whatever its save names. Two runs a week apart therefore write one arm name over two networks, and a comparison pairing them would report the change of policy as a difference between two arms that are the same arm. Carried from construction, exactly as the tactical layer beneath the board is, so an episode cut off before its horizon still says what played it.
    """
    session = _Session(_grid())
    with _without_the_game_installed():
        arena = OpsArena(session, operations_name="sha256:fedcba9876543210", seed=5)
    assert not arena.statistics.scored
    assert arena.statistics.as_dict()["operations"] == "sha256:fedcba9876543210"

    # A run that named nothing carries nothing, which is what every journal written before the field existed looks like, and a comparison has to read that as saying nothing rather than as disagreeing.
    assert OpsStatistics().operations == ""
    with _without_the_game_installed():
        assert OpsArena(session, seed=5).statistics.as_dict()["operations"] == ""


def _tactical_board():
    """A board with two regions and one enemy in front of one of our units, which is the least that makes both layers decide: the operational one needs somewhere legal to send a squad, and the tactical one needs something to depart from its contract about."""
    regions = [
        RegionState(id=1, resources=3, held_by_us=1, held_by_enemy=0, x=0.0, y=0.0,
                    our_value=500.0, enemy_value=200.0, enemy_seen_at_ms=0, distance_from_home=300.0),
        RegionState(id=2, resources=1, held_by_us=0, held_by_enemy=1, x=600.0, y=0.0,
                    our_value=0.0, enemy_value=1000.0, enemy_seen_at_ms=0, distance_from_home=900.0),
    ]
    units = [_unit(1, 0.0, 0.0), _unit(2, 20.0, 0.0), _unit(9, 200.0, 0.0, hostile=1, type_index=1)]
    observation = _observation(units=units, regions=regions)
    return build_view(observation, _CATALOGUE, None, regions)


def test_the_frozen_tactical_layer_is_handed_no_rollout_because_the_buffer_belongs_to_the_operational_one():
    """Why the layer the runners freeze under the arena is built with no rollout, which is a correctness requirement and not thrift.

    A trajectory is keyed by the instance and the squad, and the tactical layer under this board sees the very same squad records, with the very same slot ids, that the operational layer above is deciding about. Handed the training run's buffer it would file its own decisions under the operational layer's keys — a state of the tactical width and one action, spliced into a trajectory of operational states and region-and-task pairs — and the trainer would feed the lot to a network that reads neither. Its flush would also cut, and its close seal, whatever the operational layer had left live. The first half of this drives one decision of each layer about one squad through one buffer and shows the collision; nothing but the missing rollout prevents it.
    """
    view = _tactical_board()
    orders = OperationsOrders(posture=Posture.ARM, priorities={1: 0.4, 2: 1.0}, offensive=True,
                              loss_allowance=4000.0)
    squad = SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[1, 2], value=1000.0)
    shared = Rollout()
    # No decider on either: the inherited rule then decides and the layer writes down what it chose, which records a step of exactly the shape a learnt layer emits and needs no network here.
    tactical = LearntTactics(None, _CATALOGUE, None, rollout=shared, instance=0)
    operational = LearntOperations(None, _CATALOGUE, None, rollout=shared, instance=0)
    for now in (1000, 2000):
        # Twice, because a decision is paid and filed by the period after it: one period leaves it waiting.
        operational.decide(view, orders, [squad], [], now)
        tactical.decide(view, [squad], now)

    trajectories = list(shared.done) + list(shared.live.values())
    assert {trajectory.key for trajectory in trajectories} == {(0, squad.id)}, (
        "the two layers filed under different keys, and the splice this argument prevents is not what is being shown")
    widths = sorted(len(step.state) for trajectory in trajectories for step in trajectory.steps)
    assert widths == sorted([TACTICAL_SIZE, OPERATIONAL_SIZE]), (
        "one buffer under one key really does hold two action spaces' decisions, which is what the frozen layer must not do")

    # And the layer as the runners build it: no rollout, so a whole episode of it records nothing, and what the operational buffer holds at the end is the operational decisions and only those.
    arena = _standing_board()
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    layer = LearntOperations(None, _CATALOGUE, None, rollout=rollout, instance=0, discount=1.0)
    arena.our_ops = layer
    arena.their_ops = _Chain([], "theirs")
    # Built as the runners' loader builds it — the fourth argument, the rollout, left out. The decider is left out too, because what is under test is what the layer records and not what answers it.
    arena.our_tac = LearntTactics(None, _CATALOGUE, None, None, -1)
    arena.their_tac = LearntTactics(None, _CATALOGUE, None, None, -1)
    assert arena.our_tac.rollout is None and arena.their_tac.rollout is None
    arena.orders = OperationsOrders(posture=Posture.ARM, priorities=dict(arena.priorities),
                                    offensive=True, loss_allowance=4000.0)
    arena.squads = {1: _tasked(1, 4, [1])}
    arena.enemy = {}
    arena.until_ms = 999999
    regions = [RegionState(id=region, resources=1, held_by_us=0, held_by_enemy=1, x=0.0, y=0.0,
                           our_value=0.0, enemy_value=1000.0, enemy_seen_at_ms=0, distance_from_home=400.0)
               for region in (4, 9)]

    for period in range(3):
        units = [_unit(1, 0.0, 0.0), _unit(2, 60.0, 0.0, hostile=1), _unit(3, 2000.0, 0.0)]
        observation = _observation(units=units, regions=regions, game_time_ms=2000 * period)
        arena._run(observation, build_view(observation, _CATALOGUE, None, regions), Action(), 2000 * period)
    arena._score(_observation(units=[_unit(1, 0.0, 0.0), _unit(3, 2000.0, 0.0)], regions=regions,
                              game_time_ms=999999))
    arena.close()

    steps = [step for trajectory in rollout.done for step in trajectory.steps]
    assert steps, "the operational layer recorded nothing, so there is no buffer to say anything about"
    assert all(len(step.state) == OPERATIONAL_SIZE for step in steps), (
        "a tactical decision is in the operational buffer, which is what a shared rollout would put there")
    assert rollout.live == {}, "the episode closed with a trajectory still open"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)


def test_the_two_seats_departures_are_counted_apart():
    """The one diagnostic that can tell an even board fought unevenly by chance from two seats being played differently.

    The arena runs one tactical policy beneath both sides of a board that is one reflection of the other, so a fighter reading only distances and strengths gives two counts that differ by the draw. A trained one need not: the reflection is congruent in what the layer is handed and not in the ground it is handed it about, so a decision boundary can fall between the two seats and make them play differently on every board of a run. Measured, the self-play mean leans by about +0.017 under two sets of trained parameters and by nothing under a third or under the handwritten ladder.

    The counts must therefore never be pooled. This drives the pair through the record and through the reporting that reads it back.
    """
    statistics = OpsStatistics()
    statistics.our_departures[int(Deviation.HOLD)] = 7
    statistics.their_departures[int(Deviation.CLOSE)] = 5
    written = statistics.as_dict()
    assert written["our_departures"] == {int(Deviation.HOLD): 7}
    assert written["their_departures"] == {int(Deviation.CLOSE): 5}

    # And the report reads them back apart, naming the departure the two seats disagree most about rather than the commonest one.
    class _Record:
        def __init__(self, statistics):
            self.arm = "script"
            self.statistics = statistics

    class _Session:
        def __init__(self, records):
            self.records = records

    said = []
    sessions = [_Session([
        _Record({"our_departures": {int(Deviation.HOLD): 60, int(Deviation.CLOSE): 40},
                 "their_departures": {int(Deviation.HOLD): 90, int(Deviation.CLOSE): 10}}),
        _Record({"our_departures": {int(Deviation.HOLD): 40, int(Deviation.CLOSE): 60},
                 "their_departures": {int(Deviation.HOLD): 10, int(Deviation.CLOSE): 90}}),
        # Another arm's episodes must not be pooled into this arm's reading.
        _Record({"our_departures": {int(Deviation.SPREAD): 1000}, "their_departures": {}}),
    ])]
    sessions[0].records[2].arm = "pin"
    original = ops_run.log.info
    ops_run.log.info = lambda message, *values: said.append(message % values)
    try:
        ops_run._report_departures(sessions, "script")
    finally:
        ops_run.log.info = original

    assert any("hold 50.0%/50.0%" in line for line in said), said
    assert any("close 50.0%/50.0%" in line for line in said), said
    assert any("of 200 and 200 departure(s)" in line for line in said), said
    assert not any("spread" in line for line in said), "another arm's episodes were pooled in"


def test_the_board_reading_cannot_be_changed_by_renaming_the_errand():
    """The hole the board reading exists to close, stated as the property that closes it.

    Both older readings take the region the CONTRACT names, and a region the board put no priority on moves no figure, so a squad's figure goes to nought the moment its layer points it at worthless ground. Re-contracting costs nothing. A squad losing a disc therefore carries a figure below nought and can take it to nought by naming an unwanted region, which is paid as a positive movement: the layer is paid for walking away from what it is losing. The per-period payments telescope to the last figure, so whoever picks the last region picks the total, and the layer picks the last region.

    Under the board reading the contract is not read at all. The same units in the same places are paid the same figure whatever errand they hold, including no errand, so there is nothing to abstain into.
    """
    arena = _standing_board(credit="board")
    units = [_unit(1, 0.0, 0.0), _unit(2, 20.0, 0.0, hostile=1), _unit(3, 2000.0, 0.0)]
    shares = arena._shares(units)

    # One squad standing in the disc it is losing, asked for under four different errands and under none.
    figures = []
    for region in (4, 9, 7, None):
        squad = _tasked(1, region, [1])
        arena.squads = {1: squad}
        figures.append(arena._standing(squad, shares, +1.0, units))
    assert max(figures) - min(figures) < 1e-12, (
        "renaming the errand moved the pay by %.3e, so the layer can still choose its own terminal"
        % (max(figures) - min(figures)))

    # And the figure is not merely constant: it is what this squad's units account for across the scored board, so a squad that is holding something is paid and a squad standing nowhere is not.
    idle = _tasked(2, 4, [9])
    arena.squads = {2: idle}
    idle = arena._standing(idle, shares, +1.0, units)
    assert abs(idle) < 1e-12, "a squad with nothing inside any catchment is paid for standing nowhere"
    assert abs(figures[0]) > 1e-6, "a squad inside a contested catchment is paid nothing at all"

    # The older reading is the one that has the hole, and the test says so rather than assuming it: the same squad's pay moves when the errand is renamed.
    named = _standing_board(credit="region")
    named.garrison_share = dict(arena.garrison_share)
    on_disc = named._standing(_tasked(1, 4, [1]), shares, +1.0, units)
    off_disc = named._standing(_tasked(1, 7, [1]), shares, +1.0, units)
    assert abs(on_disc - off_disc) > 1e-6 and off_disc == 0.0, (
        "the region reading no longer pays differently for renaming the errand, so this record is stale")


def test_the_wanted_arm_reweighs_the_request_and_nothing_else():
    """What the strategic layer asked for is scored on the same scale as the terms this layer reads for itself, and that scale was never swept.

    A priority never exceeds one, while four resource points are worth 0.6 and enemy strength on ground we hold another 0.3, so a contested point drawing no income can lose the scoring to a quiet mine. Admitting the wanted ground to the candidates was implemented and measured and moved nothing, which leaves the scale as the standing explanation for where the ladder goes. The arm exists to measure it: every term is still read and only the request's weight moves, so a difference between it and the script arm is about the scale and about nothing else.
    """
    from types import SimpleNamespace

    from rwintel.learn.ops_run import arms_of

    region = SimpleNamespace(id=4)
    orders = SimpleNamespace(priorities={4: 0.5})
    # The default weight is one, which is the scale every measurement on file was taken at.
    assert abs(Operations._priority(SimpleNamespace(wanted=1.0), orders, region) - 0.5) < 1e-12
    assert abs(Operations._priority(SimpleNamespace(wanted=3.0), orders, region) - 1.5) < 1e-12
    # Ground nobody asked for is worth nothing whatever the weight, so re-weighting cannot invent a request.
    assert Operations._priority(SimpleNamespace(wanted=9.0), orders, SimpleNamespace(id=7)) == 0.0

    arms = arms_of(SimpleNamespace(our=["script", "wanted:3"], load=None))
    assert [arm.label for arm in arms] == ["script", "wanted-3"]
    assert arms[1].weight == 3.0

    # A weight is the whole of what the arm is: without one it would be the script arm under a second name, and two arms of one run cannot be one policy.
    for written in ("wanted", "wanted:x", "wanted:-1"):
        try:
            arms_of(SimpleNamespace(our=[written], load=None))
        except SystemExit:
            pass
        else:
            raise AssertionError("%r was accepted as a re-weighted arm" % written)


def test_the_episode_counts_how_often_squads_share_a_region():
    """The region block was given a count of how many of this side's squads hold a contract on each region, so a layer can see the allocation. Nothing in the record then said whether a layer that could see it did anything with it.

    That question cannot be answered by the score, which says only that the layer got better, nor by the errand length or the priority share, both of which have been refused as quality measures. This is the figure that says whether massing is what it started doing, and it is counted against the contracts standing in the period rather than off any layer's own bookkeeping, so it costs the same and means the same for the handwritten ladder, the pin, the concentrating rule and a network alike.
    """
    from rwintel.control.policy.contracts import Stance, Task, TaskContract

    with _without_the_game_installed():
        arena = OpsArena(_Session(_grid()), seed=5)

    def contracted(squad_id, region):
        squad = _tasked(squad_id, region, [squad_id])
        squad.contract = TaskContract(squad=squad_id, task=Task.ATTACK, target_region=region,
                                     stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                     deadline_ms=90000, issued_at_ms=1000 * squad_id)
        return squad

    arena.orders = object()
    arena.priorities = {4: 1.0, 7: 1.0}

    # Two squads on one region and one on its own: two of the three decisions are about a shared region.
    arena.squads = {0: contracted(0, 4), 1: contracted(1, 4), 2: contracted(2, 7)}
    arena._survey()
    assert arena.statistics.periods == 3
    assert arena.statistics.massed == 2, (
        "a period with two squads on one region should count both of them as massed, not %d"
        % arena.statistics.massed)

    # Spread out, nobody shares, and the count does not move even though every squad still holds a contract.
    arena.squads = {0: contracted(0, 4), 1: contracted(1, 7), 2: contracted(2, 9)}
    arena._survey()
    assert arena.statistics.periods == 6 and arena.statistics.massed == 2, (
        "a period in which no two squads share a region added to the massing count")

    # It is about the allocation and not about the ground being worth anything: a shared region the board puts no priority on still counts as massing, and still names no priority.
    before = arena.statistics.on_priority
    arena.squads = {0: contracted(0, 9), 1: contracted(1, 9)}
    arena._survey()
    assert arena.statistics.massed == 4 and arena.statistics.on_priority == before


def test_a_side_wider_than_the_squad_block_is_refused_before_the_board_is_built():
    """The operational cut has eight squad rows and the network names a squad to itself by a one-hot of the same width, so a ninth squad has no row to be described in and no slot to be asked about.

    Staged anyway, it would be deployed, fought, scored and paid its terminal while the layer deciding for it was handed a board on which it does not appear — a number in a journal that reads like every other number. The bound is the organisation layer's cap in a match, and this board builds its squads itself and never passes that layer, so this is the only place the cap can be kept.
    """
    from rwintel.wire import SQUAD_SLOTS

    refused = ""
    try:
        OpsArena(None, our_squads=SQUAD_SLOTS + 1)
    except ValueError as complaint:
        refused = str(complaint)
    assert "squad row" in refused, "a side wider than the squad block has to be refused, not staged"

    # The width the block does have is not refused, and the refusal happens before the session is touched at all.
    assert "squad row" not in _refusal_of(lambda: OpsArena(None, our_squads=SQUAD_SLOTS))


def _refusal_of(call):
    try:
        call()
    except Exception as complaint:      # the session is None here, so anything past the check is this test's own doing
        return str(complaint)
    return ""


def test_both_sides_squads_carry_their_scatter_their_losses_and_where_their_errand_stands():
    """What the fold hands the layers, on a board where a squad has been shot at and its deadline has passed.

    The fold wrote membership, worth and position and nothing else, so every squad on this board read as unscattered, unhurt and ACTIVE from the first frame to the last. Six of the inputs both frozen layers read were pinned at their opening values — the tactical cut carries a squad's spread, how much of its allowance is gone and a one-hot of its status, and the operational cut carries the same three per squad row — and the inherited operational rule re-tasks a squad on exactly those statuses, so the ladder this arena measures learnt policies against never re-tasked anything for a whole horizon.
    """
    arena = _arena()
    arena.squads = {0: _tasked(0, 4, [1, 2])}
    arena.enemy = {1: _tasked(1, 4, [3, 4])}
    for squad in list(arena.squads.values()) + list(arena.enemy.values()):
        squad.value = 700.0
    # Both sides' squads are strung out, which is a fact about the units and not about which seat is reading them.
    units = [_unit(1, 0.0, 0.0), _unit(2, 300.0, 0.0), _unit(3, 1000.0, 0.0, hostile=1),
             _unit(4, 1300.0, 0.0, hostile=1)]
    arena._at_issue = {0: (0, 1500.0), 1: (0, 1500.0)}
    arena._fold_all(_observation(units))

    for squad in (arena.squads[0], arena.enemy[1]):
        assert squad.spread > 0.0, "a squad whose members stand 300 apart is not unscattered"
        assert squad.losses == 800.0, "the worth gone since the contract was issued has to reach the allowance it is spent against"

    # And where the errand stands is written from the board, by the ladder the game side uses for a match.
    board = build_view(_observation(units, [RegionState(id=4, resources=1, held_by_us=0, held_by_enemy=1,
                                                        x=0.0, y=0.0, our_value=0.0, enemy_value=900.0,
                                                        enemy_seen_at_ms=0, distance_from_home=100.0)]),
                       _CATALOGUE, None)
    arena._fold_status(board, board, now=30000)
    assert arena.squads[0].status is Status.LOSING, (
        "losses past the share of the allowance the game side calls losing have to read as losing, not %s"
        % arena.squads[0].status)

    # An allowance that has not gone, with the deadline past, is the other ending the rule reads off the contract.
    arena._at_issue = {0: (0, 700.0), 1: (0, 700.0)}
    arena._fold_all(_observation(units))
    arena._fold_status(board, board, now=10 ** 9)
    assert arena.squads[0].status is Status.EXPIRED
