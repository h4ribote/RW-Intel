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
from rwintel.data.regions import Region
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
    SCRIPT_TACTICS,
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
    RegionState,
    UnitState,
)
from rwintel.wire.action import Stance, Task

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
    arena.opening_baseline = OPENING_BASELINE
    arena.credit = CREDIT
    arena.garrison_scale = GARRISON_SCALE
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
    arena._frozen = {}
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
    arena._issued = {}
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


def test_a_terminal_is_read_from_where_its_disc_started_and_not_from_the_neutral_half():
    """An errand is worth what it changed, and what it changed cannot be read without knowing where the ground started.

    A garrison keeps its own disc through the horizon about eighty-eight times in a hundred whether or not a squad is sent to stand with it, and even four squads massed on an enemy's disc take it only about four times in ten. Read from the neutral half, those two rates pay a redundant defence about +0.38 of a priority and an assault about −0.10, so the most profitable errand a squad can be given is one that was going to be won without it, and the first layer trained on this arena duly learnt to attack nothing and lose nothing. Read from the disc's own opening, the same two rates pay the assault about +0.40 and the redundant defence about nothing, which is the quantity the side score is a priority-weighted mean of.
    """
    arena = _arena(seed=11)
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
        arena._finish_side(ledger, squads, shares, sign, [_unit(1, 0.0, 0.0), _unit(2, 2000.0, 0.0)])
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
    arena._finish_side(region_ops, squads, shares, +1.0, units)
    assert abs(region_ops.paid[1] - (2.0 / 3.0 - 0.5)) < 1e-9
    assert region_ops.paid[1] == region_ops.paid[2], "the region reading pays the pile in full, squad by squad"

    arena.credit = "marginal"
    marginal_ops = _Paid()
    arena._finish_side(marginal_ops, squads, shares, +1.0, units)
    # Without either squad the disc reads one tank ours against one hostile, which is a half; each is credited the sixth it added.
    assert abs(marginal_ops.paid[1] - (2.0 / 3.0 - 0.5)) < 1e-9
    assert abs(marginal_ops.paid[2] - marginal_ops.paid[1]) < 1e-9
    # The third squad was sent to the same region and is not on the board at the horizon. Neither reading pays it: it has moved nothing since its last unit went, and the disc it was sent to was taken by others.
    assert region_ops.paid[3] == 0.0
    assert marginal_ops.paid[3] == 0.0

    # A third squad piled onto the same taken region is paid in full by the region reading and almost nothing by the marginal one, which is the whole difference between them.
    units.append(_unit(4, 30.0, 0.0))
    squads[3].members = [4]
    shares = {4: 3.0 / 4.0}
    piled = _Paid()
    arena.credit = "region"
    arena._finish_side(piled, squads, shares, +1.0, units)
    assert abs(piled.paid[3] - (3.0 / 4.0 - 0.5)) < 1e-9
    marginal_piled = _Paid()
    arena.credit = "marginal"
    arena._finish_side(marginal_piled, squads, shares, +1.0, units)
    assert 0.0 < marginal_piled.paid[3] < piled.paid[3]


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
    arena._finish_side(paid, squads, shares, +1.0, units)

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

    arena._finish_side(layer, squads, {4: 1.0}, +1.0, [_unit(1, 0.0, 0.0), _unit(2, 10.0, 0.0)])
    arena._tally()

    assert len(squads) == 2, "both squads have to be offered a payment or the count is not being told apart from the offers"
    assert arena.statistics.terminals == 1
    assert layer.terminals["horizon"] == 1
    # A layer that keeps no trajectories at all — every arm of the measuring runner — lands none of them, and nought is the truth for it rather than a fault.
    arena.our_ops = LearntOperations(None, None, None, rollout=None, instance=-1)
    arena._finish_side(arena.our_ops, squads, {4: 1.0}, +1.0, [_unit(1, 0.0, 0.0)])
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
    arena._finish_side(ledger, {1: squad}, {4: reading(horizon, 4), 9: reading(horizon, 9)}, +1.0, horizon)
    terminal = ledger.paid[1]

    assert abs(terminal) < 1e-9, "the disc this squad held at the horizon ended where it opened"
    assert abs(telescoped - terminal) < 1e-9, "the endpoint ledger did not sum to the terminal"
    assert abs(walked - 2.0 / 3.0) < 1e-9 and abs(walked - terminal) > 0.6, (
        "the path ledger paid the squad for a disc it was re-tasked away from")

    # The other way a dense credit stops telescoping, and it needs no re-tasking at all: under the marginal reading a board's reading depends on which of the squad's units were standing in the disc, so an earlier board re-read with the members the squad has now is not the reading that board gave when it was current. Anything paying differences of readings has to hold the membership of the board it holds, or a squad that loses a tank is paid for the loss twice over.
    contest = next(c for c in arena.contests if c.region_id == 4)
    whole = reading(middle, 4) - arena._share_without(middle, contest, [10, 11])
    survivor = reading(middle, 4) - arena._share_without(middle, contest, [10])
    assert abs(whole - 2.0 / 3.0) < 1e-9 and abs(survivor - 1.0 / 6.0) < 1e-9


# ---- the reading the periods are paid off ---------------------------------------------------

def _standing_board():
    """An arena with two discs of opposite ownership, which is the pair every property here needs: one this side has to take from the enemy's garrison and one it only has to keep."""
    arena = _arena(seed=23)
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

    Every operational period is paid the movement of a squad's scored figure and the horizon is paid the same figure once more, so the payments telescope to the horizon's reading — but only for as long as the two readings are the same reading. Were the horizon to keep an expression of its own, the identity would be an intention that two pieces of code had to be kept in step, and the first time they drifted every episode would return the terminal plus whatever the drift came to, with nothing in any log to say so. So the horizon calls `_standing` and this holds it to that: what `_finish_side` hands a layer is exactly what the period loop reads, under either credit.
    """
    for credit in ("region", "marginal"):
        arena = _standing_board()
        arena.credit = credit
        units = [_unit(1, 0.0, 0.0), _unit(2, 20.0, 0.0), _unit(3, 40.0, 0.0, hostile=1),
                 _unit(4, 2000.0, 0.0), _unit(5, 2030.0, 0.0, hostile=1, health=40.0)]
        squads = {1: _tasked(1, 4, [1, 2]), 2: _tasked(2, 9, [4]), 3: _tasked(3, 7, [3])}
        shares = arena._shares(units)

        for sign in (+1.0, -1.0):
            ledger = _Paid()
            arena._finish_side(ledger, squads, shares, sign, units)
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
            arena = _arena(seed=trial)
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
                mirror = _arena(seed=trial)
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

    arena = OpsArena(session, tactics=tactics, seed=5)
    assert len(built) == 2, "the arena did not build both of its tactical layers from the factory it was handed"
    # Both are handed the arena's own session and the arena's own catalogue, which is why the layers are built in here at all: a layer classifying a unit from some other type table would sort the same tank into a different role from the arena that spawned it.
    assert all(given is session and catalogue is arena.catalogue for given, catalogue in built)
    assert isinstance(arena.our_tac, LearntTactics) and isinstance(arena.their_tac, LearntTactics)
    assert arena.our_tac is not arena.their_tac, "one shared layer would fold the two sides' bookkeeping together"
    assert arena.our_tac.decider is decider and arena.their_tac.decider is decider, (
        "the two sides are reading different networks, so they are not fighting under one frozen layer")

    # Named no factory, the arena builds the handwritten ladder for itself on both sides, which is what every measurement taken on this arena so far was made under — and two of it, for the same reason.
    plain = OpsArena(session, seed=5)
    assert type(plain.our_tac) is Tactics and type(plain.their_tac) is Tactics
    assert plain.our_tac is not plain.their_tac
    # And nothing about the operational layers moved: with no operational factory both sides are still the script chain.
    assert type(plain.our_ops) is Operations and type(plain.their_ops) is Operations


def test_an_episode_says_which_tactical_layer_it_was_made_under_before_it_has_scored_anything():
    """Two runs made under different tactical layers are two different instruments, and pairing them board by board would read the change of fighter as a difference between the operational arms. So the episode record has to carry which layer was beneath it, and carry it from construction rather than from scoring: an episode cut off before its horizon still has to say what instrument it was run on."""
    session = _Session(_grid())
    arena = OpsArena(session, tactics_name="sha256:0123456789abcdef", seed=5)
    assert not arena.statistics.scored
    assert arena.statistics.tactics == "sha256:0123456789abcdef"
    assert arena.statistics.as_dict()["tactics"] == "sha256:0123456789abcdef"

    # And a run that named nothing says so in the one word a comparison reads as the handwritten ladder.
    assert OpsStatistics().tactics == SCRIPT_TACTICS
    assert OpsArena(session, seed=5).statistics.as_dict()["tactics"] == SCRIPT_TACTICS


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
