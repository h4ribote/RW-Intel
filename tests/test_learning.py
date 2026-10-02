"""What the learning side promises, held to.

Three things are pinned here and they are the three that fail silently rather than loudly. The encodings, because a feature vector that has quietly shifted by one, or that carries a raw credit total among ratios, produces a policy that trains and is worthless on the next map rather than one that crashes. The rewards, because potential-based shaping is only safe if it telescopes, and a shaping term that does not is a term that changes which policy is optimal while looking exactly like one that does not. And the buffers, because an errand that was cut off when the match ended is not an errand that failed, and scoring the two the same way teaches a policy to end matches.

A fourth was added once a run of the arena had been measured rather than assumed: what a fight was worth, and where one errand stops and the next begins. A score that is not exactly antisymmetric makes self-play average to something other than nought, and then no comparison drawn against a baseline means anything, because the baseline is measuring the arena's own left-right lean instead of the policy. A trajectory that is not closed when the fight it belongs to is called runs on into the next fight built on the same squad number, and then the advantage earned in one fight flows backwards into decisions taken in another. Both produce a policy that trains perfectly smoothly and has learnt the wrong thing.

Last, the two ways a run is started from something rather than from noise: a fit to what the handwritten layer already does, and a few updates that fit the critic while the policy is held still. Those three need the tensor library, which everything above them deliberately does not.

Nothing here launches a game. The arena is exercised only where it can be: the mirrored view, which is the whole of how one process drives both sides of a fight, and the arithmetic that turns a finished fight into a score.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwintel.control.policy.catalogue import Catalogue
from rwintel.control.policy.contracts import Doctrine, SquadRecord, TaskContract
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.learn.arena import STRENGTH_SLOPE, Engagement
from rwintel.learn.deciders import Choice
from rwintel.control.policy.encoding import (
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    REGION_FEATURES,
    SQUAD_FEATURES,
    TACTICAL_ACTIONS,
    TACTICAL_FEATURES,
    TACTICAL_SIZE,
    operational_state,
    plan_masks,
    plan_of,
    region_mask,
    squad_mask,
    tactical_state,
    task_mask,
)
from rwintel.learn.layers import LearntTactics
from rwintel.learn.net import TacticalNet
from rwintel.learn.reward import (
    BY_HEALTH,
    BY_KILLS,
    SCORES,
    COMPLETE_REWARD,
    DISCOUNT,
    EXCHANGE_PRIOR,
    EXCHANGE_WEIGHT,
    HOLDING_WEIGHT,
    LOSING_REWARD,
    OperationalReward,
    OperationalTerms,
    SHARE,
    SPENDING_WEIGHT,
    TacticalReward,
    WIPED_REWARD,
)
from rwintel.learn.rollout import Rollout, Step
from rwintel.learn.train import Optimiser
from rwintel.control.policy.tactics import Tactics, _Track
from rwintel.eval.sampling import Summary
from rwintel.wire import (
    BLOCK_REGIONS,
    BLOCK_SQUADS,
    BLOCK_UNITS,
    Action,
    Deviation,
    Observation,
    RegionState,
    SquadDeviation,
    Stance,
    Status,
    Task,
    UnitState,
)

_TYPES = [
    UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0, hits_air=False, hits_land=True),
    UnitType(index=1, name="artillery", lookup="artillery", price=700, tech=1, building=False,
             builder=False, movement="LAND", can_attack=True, range=320.0, hits_air=False, hits_land=True),
    UnitType(index=2, name="builder", lookup="builder", price=500, tech=1, building=False,
             builder=True, movement="LAND"),
]


def _Catalogue(types):
    """The type table without the definitions the asset tree provides, which are not what any of this is about."""
    return Catalogue.of_types(types)


def _unit(unit_id, x=0.0, y=0.0, type_index=0, squad=0xFFFF, hostile=0, health=100.0, hit=9999,
          max_health=100.0):
    return UnitState(id=unit_id, squad=squad, type_index=type_index, x=x, y=y, health=health,
                     max_health=max_health, built=255, order=255, queued=0, target=0, stance=5,
                     hostile=hostile, since_hit_ms=hit)


def _region(region_id, x=0.0, y=0.0, ours=0.0, theirs=0.0, resources=2, distance=500.0):
    return RegionState(id=region_id, resources=resources, held_by_us=1, held_by_enemy=0, x=x, y=y,
                       our_value=ours, enemy_value=theirs, enemy_seen_at_ms=1000,
                       distance_from_home=distance)


def _observation(units=(), regions=(), game_time_ms=30000):
    return Observation(frame=10, game_time_ms=game_time_ms, episode=1,
                       blocks=BLOCK_REGIONS | BLOCK_SQUADS | BLOCK_UNITS, slot=0, credits=1500.0,
                       income=22.0, units=len(units), unit_cap=100, under_construction=1,
                       killed_units=3, killed_buildings=1, lost_units=2, lost_buildings=0,
                       regions=list(regions), unit_states=list(units))


def _view(units, regions):
    return build_view(_observation(units, regions), _CATALOGUE, None)


_CATALOGUE = _Catalogue(_TYPES)


def _squad(members=(1, 2, 3), contract=True, **overrides):
    fields = dict(id=0, doctrine=Doctrine.VANGUARD, members=list(members), value=1050.0,
                  formed_value=1400.0, x=100.0, y=100.0, spread=60.0, losses=350.0)
    fields.update(overrides)
    squad = SquadRecord(**fields)
    if contract:
        squad.contract = TaskContract(squad=squad.id, task=Task.ATTACK, target_region=1,
                                      stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                      deadline_ms=90000, issued_at_ms=20000)
    return squad


def _refusal(exception, call, *arguments):
    """What a call complained about, or nothing at all if it went through. The runner at the foot of this file is a for loop rather than a framework, so this is where an assertion about a refusal comes from."""
    try:
        call(*arguments)
    except exception as refused:
        return str(refused) or repr(refused)
    return ""


# ---- the encodings ------------------------------------------------------------------------

def test_the_feature_names_and_the_vectors_are_the_same_length():
    """Named rather than merely counted, because a vector that has silently shifted by one trains perfectly well and means nothing."""
    units = [_unit(1, 100, 100), _unit(2, 120, 100), _unit(9, 300, 100, hostile=1, type_index=1)]
    regions = [_region(0), _region(1, 400.0, 100.0, ours=200.0, theirs=900.0)]
    view = _view(units, regions)
    squad = _squad()
    threats = [s for s in view.enemies]
    members = [s for s in view.ours]

    state = tactical_state(squad, members, threats, 350.0, 200.0, view, 30000)
    assert len(state) == TACTICAL_SIZE == len(TACTICAL_FEATURES)
    assert len(operational_state(view, None, [squad], 30000)) == OPERATIONAL_SIZE


def test_every_feature_is_a_ratio_and_stays_inside_its_range():
    """The one property that makes a policy readable on a map it was not trained on. A raw credit total or a raw world coordinate anywhere in here would tie the policy to the size of the match it saw."""
    units = [_unit(i, 100.0 + 10 * i, 100.0) for i in range(1, 9)]
    units += [_unit(50 + i, 900.0, 900.0, hostile=1, type_index=1) for i in range(6)]
    regions = [_region(i, 1000.0 * i, 0.0, ours=5000.0, theirs=90000.0, distance=9000.0)
               for i in range(6)]
    view = _view(units, regions)
    squad = _squad(members=[1, 2, 3, 4, 5, 6, 7, 8], value=99999.0, formed_value=1.0, spread=5000.0)

    state = tactical_state(squad, [s for s in view.ours], [s for s in view.enemies],
                           999999.0, 999999.0, view, 10 ** 7)
    assert all(math.isfinite(value) for value in state)
    assert all(-1.0 <= value <= 1.0 for value in state), [
        (name, value) for name, value in zip(TACTICAL_FEATURES, state) if not -1.0 <= value <= 1.0]

    board = operational_state(view, None, [squad], 10 ** 7)
    assert all(math.isfinite(value) for value in board)
    assert all(-1.0 <= value <= 1.0 for value in board)


def test_a_board_with_nothing_on_it_still_encodes():
    """The first frame of an arena episode is exactly this, and a division by nought in the encoder would end the run before the first engagement was built."""
    view = _view([], [])
    squad = _squad(members=[], contract=False, value=0.0, formed_value=0.0)
    assert len(tactical_state(squad, [], [], 0.0, 0.0, view, 0)) == TACTICAL_SIZE
    assert len(operational_state(view, None, [], 0)) == OPERATIONAL_SIZE


def test_the_region_block_is_always_twenty_four_rows_whatever_the_map():
    """Fixed width with a validity flag, never a packed list: a slot has to mean the same place from one decision to the next, and a packed list renumbers everything the moment a region appears."""
    small = operational_state(_view([], [_region(0)]), None, [], 0)
    large = operational_state(_view([], [_region(i) for i in range(20)]), None, [], 0)
    assert len(small) == len(large) == OPERATIONAL_SIZE
    assert sum(region_mask(_view([], [_region(0), _region(7)]))) == 2


def test_a_squad_may_only_be_sent_where_a_region_is_and_given_what_its_doctrine_allows():
    assert region_mask(_view([], [_region(0), _region(3)]))[:4] == [1.0, 0.0, 0.0, 1.0]
    vanguard = task_mask(Doctrine.VANGUARD)
    assert vanguard[int(Task.ATTACK)] == 1.0 and vanguard[int(Task.DEFEND)] == 0.0
    # Engineers are what an escort escorts, not what is sent anywhere: a contract on one would land on top of the placement a builder is walking to.
    assert sum(task_mask(Doctrine.ENGINEER)) == 0.0


def test_a_squad_someone_else_holds_is_not_one_this_layer_may_task():
    mine = _squad()
    theirs = _squad(id=1, commander=1)
    empty = _squad(id=2, members=[])
    mask = squad_mask([mine, theirs, empty])
    assert mask[0] == 1.0 and mask[1] == 0.0 and mask[2] == 0.0


# ---- the rewards --------------------------------------------------------------------------

def test_the_first_period_of_an_errand_pays_nothing():
    """The step that merely received a contract is not paid for the board it arrived on, or every errand would begin with a reward for wherever the squad happened to be standing."""
    reward = TacticalReward()
    view = _view([], [_region(1, ours=500.0, theirs=500.0)])
    assert reward.step(_squad(), view, 21000).reward == 0.0


def test_the_shaping_telescopes_so_a_board_that_does_not_move_pays_almost_nothing():
    """Potential-based shaping is only safe because it sums to the difference of two potentials. A term that paid for standing still would be a term that changes what the optimal policy is."""
    reward = TacticalReward()
    view = _view([], [_region(1, ours=500.0, theirs=500.0)])
    squad = _squad()
    reward.step(squad, view, 21000)
    outcome = reward.step(squad, view, 21200)
    # What is left over is one discount step on an unchanged potential, and nothing else. Nothing has been destroyed and nothing has been lost, so the exchange stands at even.
    potential = (HOLDING_WEIGHT * 0.5
                 + SPENDING_WEIGHT * (1.0 - squad.losses / squad.contract.cost_budget)
                 + EXCHANGE_WEIGHT * (EXCHANGE_PRIOR / (2 * EXCHANGE_PRIOR + squad.losses)))
    assert abs(outcome.reward - (DISCOUNT - 1.0) * potential) < 1e-9
    assert not outcome.done


def test_the_shaping_telescopes_at_whatever_discount_it_was_handed():
    """The condition under which shaping stays harmless, now that the discount is an argument rather than a constant: the term has to telescope at exactly the figure the returns are discounted at. Two numbers that were one constant can drift apart, and a pair that differ leave a residue which depends on where the errand went -which is the single dependence potential shaping is chosen to rule out.

    Telescoping means that the payments of an errand, discounted at that same figure and summed, come to the terminal discounted back to the first paid decision less the potential the errand opened on, and to nothing else at all: no board the errand passed through survives the sum. Pinned at both figures actually in use -the hundredth off that an errand which is a fragment of a match takes, and nothing at all, which is what a constructed fight is discounted at because it is one whole finite episode with a real terminal. At a discount of one the property reads as the plain sum of the payments being the terminal less the opening potential.
    """
    for discount in (1.0, 0.99):
        reward = TacticalReward(discount=discount)
        squad = _squad()
        boards = [_view([], [_region(1, ours=ours, theirs=900.0 - ours)])
                  for ours in (100.0, 700.0, 300.0, 500.0, 200.0)]
        reward.step(squad, boards[0], 21000)
        held = reward.missions[squad.id].potential
        assert held > 0.0

        payments = []
        for index, board in enumerate(boards[1:], start=1):
            # The board moves under the errand on every count the potential is made of -the ground, the allowance and the exchange -so that a term which failed to telescope would leave a residue rather than a nought.
            squad.losses = 100.0 * index
            payments.append(reward.step(squad, board, 21000 + index * 200,
                                        killed=250.0 * index).reward)
        squad.status = Status.COMPLETE
        ending = reward.step(squad, boards[-1], 22000)
        assert ending.done and ending.reason == "complete"
        payments.append(ending.reward)

        summed = sum(payment * discount ** index for index, payment in enumerate(payments))
        assert abs(summed - (discount ** (len(payments) - 1) * COMPLETE_REWARD - held)) < 1e-9
        if discount == 1.0:
            assert abs(sum(payments) - (COMPLETE_REWARD - held)) < 1e-9


def test_taking_the_ground_pays_the_terminal_less_the_potential_it_was_holding():
    """The completion, and one last shaping term taken against a terminal potential of nought.

    That convention is the whole of why shaping is safe: the terms over an errand have to telescope to the difference between where it started and nothing, and a last term paid against the real potential of the board the errand ended on leaves a residue proportional to it instead. Taking the ground scores near the top of the potential, so paying that residue would make a completion worth well over the terminal it is defined to be worth, and worth a different amount according to how much of the budget went -which is a reward that depends on the ending, which is what shaping is chosen not to be.
    """
    reward = TacticalReward()
    view = _view([], [_region(1, ours=800.0, theirs=0.0)])
    squad = _squad(status=Status.COMPLETE)
    reward.step(squad, view, 21000)
    held = reward.missions[squad.id].potential
    outcome = reward.step(squad, view, 40000)
    assert outcome.done and outcome.reason == "complete"
    assert held > 0.0 and abs(outcome.reward - (COMPLETE_REWARD - held)) < 1e-9


def test_losing_the_budget_without_the_ground_is_the_expensive_outcome():
    reward = TacticalReward()
    view = _view([], [_region(1, ours=0.0, theirs=900.0)])
    squad = _squad(status=Status.LOSING, losses=900.0)
    reward.step(squad, view, 21000)
    outcome = reward.step(squad, view, 40000)
    assert outcome.done and outcome.reason == "losing"
    assert outcome.reward < LOSING_REWARD / 2


def test_a_squad_destroyed_under_contract_is_the_worst_outcome_of_all():
    """Distinguished from being reported as losing because a squad that is gone cannot be withdrawn, and the layer had every period until then to withdraw it."""
    reward = TacticalReward()
    view = _view([], [_region(1, ours=0.0, theirs=900.0)])
    squad = _squad(members=[])
    reward.step(squad, view, 21000)
    outcome = reward.step(squad, view, 40000)
    assert outcome.done and outcome.reason == "wiped"
    assert outcome.reward < WIPED_REWARD / 2


def test_the_operational_layer_is_paid_for_the_ground_its_orders_named_only_when_that_is_weighed_in():
    class _Orders:
        posture = 0
        priorities = {1: 1.0}
        offensive = True
        loss_allowance = 2000.0

    losing = _view([], [_region(1, ours=100.0, theirs=900.0)])
    winning = _view([], [_region(1, ours=900.0, theirs=100.0)])
    for weight in (1.0, 0.0):
        terms = OperationalTerms(achievement_weight=weight, shaping=SHARE, share_weight=0.0)
        reward = OperationalReward(terms)
        reward.signal(losing, _Orders(), [])
        to_winning = terms.pay(reward.signal(winning, _Orders(), []))
        reward.reset()
        reward.signal(winning, _Orders(), [])
        to_losing = terms.pay(reward.signal(losing, _Orders(), []))
        if weight:
            assert to_winning > 0.0 > to_losing
        else:
            assert to_winning == to_losing == 0.0


# ---- the buffers --------------------------------------------------------------------------

def _steps(count, reward=0.0, value=0.0, done_last=True):
    return [Step(state=[0.0], action=0, mask=[1.0], value=value, reward=reward,
                 done=done_last and index == count - 1, squad=0)
            for index in range(count)]


def test_a_finished_errand_and_a_cut_off_one_are_scored_differently():
    """An errand still running when the match was called did not fail. Treating the two the same teaches a policy that ending the match is worth something."""
    finished = Rollout()
    for step in _steps(3, reward=1.0):
        finished.add("a", step)
    cut = Rollout()
    for step in _steps(3, reward=1.0, done_last=False):
        cut.add("b", step)
    cut.cut("b", tail_value=10.0)

    a = finished.drain()
    b = cut.drain()
    assert len(a) == len(b) == 3
    # The bootstrap is the whole of the difference, and it only reaches the last step and those before it.
    assert b[-1].ret > a[-1].ret


def test_the_trainer_thread_finishes_and_is_joined():
    """Finishing a training run stops the update thread and joins it, which is the last thing every training run does before saving its parameters."""
    from rwintel.learn.train import Trainer

    trainer = Trainer(Rollout(), Optimiser(TacticalNet()), batch=4)
    trainer.start()
    assert trainer.finish() is None
    assert not trainer.is_alive()


def test_decisions_about_a_squad_somebody_interfered_with_are_not_learnt_from():
    """The design's rule: praising a policy for what a person's squad achieved, or blaming it for what a person's squad lost, teaches the wrong thing. Marking happens late because which squads were interfered with is not known when the decision was taken."""
    rollout = Rollout()
    for index in range(3):
        rollout.add("a", Step(state=[0.0], action=0, mask=[1.0], squad=0, done=index == 2))
        rollout.add("b", Step(state=[0.0], action=0, mask=[1.0], squad=1, done=index == 2))
    rollout.taint([1])
    kept = rollout.drain()
    assert {step.squad for step in kept} == {0}
    assert len(kept) == 3


def test_a_trajectory_that_is_still_running_is_left_alone_by_a_drain():
    rollout = Rollout()
    rollout.add("a", Step(state=[0.0], action=0, mask=[1.0]))
    assert rollout.drain() == []
    assert len(rollout) == 1


# ---- driving both sides of a fight ---------------------------------------------------------

def test_the_board_read_from_the_other_side_exchanges_the_sides_and_nothing_else():
    """How one process fights both sides of a constructed engagement: the opposing tactical layer reads the same frame with the sides the other way round, and everything else a tactical layer looks at is symmetric already.

    Read rather than copied. Copying every unit with its hostility flipped is the obvious way and it measured three to five times the cost of building the whole view, twice a period, which made it the most expensive thing in a frame.
    """
    observation = _observation(
        units=[_unit(1, 10.0, 10.0), _unit(2, 20.0, 20.0, hostile=1)],
        regions=[_region(0, ours=100.0, theirs=900.0)])
    ours = build_view(observation, _CATALOGUE, None)
    theirs = build_view(observation, _CATALOGUE, None, invert=True)

    assert [s.unit.id for s in ours.ours] == [1] and [s.unit.id for s in ours.enemies] == [2]
    assert [s.unit.id for s in theirs.ours] == [2] and [s.unit.id for s in theirs.enemies] == [1]
    assert (theirs.regions[0].our_value, theirs.regions[0].enemy_value) == (900.0, 100.0)
    # Nothing is copied, so the two views are of the very same units and the observation is untouched.
    assert ours.ours[0].unit is theirs.enemies[0].unit
    assert [unit.hostile for unit in observation.unit_states] == [0, 1]


def test_the_two_sides_spawn_orders_are_interleaved_so_neither_leads():
    """A fight is two spawn orders sent down one command queue, and the queue is drained in the order it was filled. Sending one side's whole order before the other's leaves that side more completely on the board when the spawn wait is called, which was measured as three points of a left-right lean in the strength that forms into a squad -enough to bias the self-play score the arena is scored against. Interleaving keeps both orders at the same depth in the queue throughout, so a wait that runs out cuts both to the same degree."""
    from rwintel.learn.arena import Arena

    def rows(slot, count):
        # One five-float spawn row per unit, the second field the player slot that names which side it is.
        return [[float(slot * 100 + index), float(slot), 0.0, 0.0, 1.0] for index in range(count)]

    for our_count, their_count in ((6, 6), (12, 4), (3, 9), (1, 5), (7, 1)):
        flat = Arena._interleave(rows(0, our_count), rows(1, their_count))
        assert len(flat) == (our_count + their_count) * 5
        # The slot field of each row, in the order the queue would drain them.
        order = [flat[base + 1] for base in range(0, len(flat), 5)]

        # Every row of both orders is present exactly once.
        assert order.count(0.0) == our_count and order.count(1.0) == their_count

        # While both orders still have units to place, the queue never runs more than one unit ahead on either side: this is the property that removes the systematic lead. Past that depth the shorter order is spent and its tail runs on alone, which leans on the larger force rather than on a fixed side.
        for depth in range(1, 2 * min(our_count, their_count) + 1):
            assert abs(order[:depth].count(0.0) - order[:depth].count(1.0)) <= 1

        # Neither side is the one that always leads: which side is submitted first alternates pair by pair.
        if our_count >= 2 and their_count >= 2:
            assert order[0] != order[2]


# ---- what a fight was worth ----------------------------------------------------------------

def _fight(our_value, their_value, our_left, their_left, our_health=None, their_health=None):
    """One called fight, as far as the arithmetic that scores it reads one.

    The two health figures default to the worth left standing, which is what a side whose survivors are all untouched is worth on either reading and is therefore the fixture for a fight the two readings have to agree on.
    """
    return Engagement(index=0, site=(0.0, 0.0), our_value=our_value, their_value=their_value,
                      our_left_value=our_left, their_left_value=their_left,
                      our_left_health=our_left if our_health is None else our_health,
                      their_left_health=their_left if their_health is None else their_health)


def test_the_score_of_a_fight_read_from_the_other_side_is_the_same_number_negated():
    """The property the whole measurement rests on, and the reason the score is written in shares rather than in credits.

    Self-play has to average to nought, so that a run against the handwritten layer that averages above nought is the same statement as having beaten it and needs no correction for anything. A score built from a difference of worth would instead pay for having been dealt the stronger side, and the arena deals deliberately uneven sides.

    The term that takes out what the draw was worth has to be antisymmetric too, and it is, because one side's share of the total strength is one less the other's. It is what puts the score a little outside minus one to plus one: a massacre against the odds scores above one, which is the point of it.
    """
    draw = random.Random(5)
    for _ in range(64):
        our_value = draw.uniform(1.0, 5000.0)
        their_value = draw.uniform(1.0, 5000.0)
        our_left = draw.uniform(0.0, our_value)
        their_left = draw.uniform(0.0, their_value)
        ours = _fight(our_value, their_value, our_left, their_left)
        theirs = _fight(their_value, our_value, their_left, our_left)
        assert abs(ours.outcome + theirs.outcome) < 1e-12
        # Bounded by the shares, which run from minus one to plus one, less a term that cannot exceed half the slope and at present is nought.
        assert -1.0 - STRENGTH_SLOPE / 2 <= ours.outcome <= 1.0 + STRENGTH_SLOPE / 2

    # A side that was never built at all is worth nothing and has lost nothing, which has to be a number rather than a division by nought: an engagement whose spawns never arrived on one side still reaches the point where it is scored.
    empty = _fight(0.0, 1200.0, 0.0, 0.0)
    assert empty.outcome + _fight(1200.0, 0.0, 0.0, 0.0).outcome == 0.0


def test_destroying_the_other_side_without_a_loss_is_the_top_of_the_scale():
    """What fixes the size of the scale, and with it how much a called fight is worth against the errand's own conclusions: a massacre is paid exactly what taking the contracted ground is paid, and no more, so that a layer is never taught to prefer the one to the other.

    Stated on an even draw, so that it holds whatever the term that takes out what the draw was worth is set to. That term is nought at present, having been measured to inject a bias larger than anything it was meant to help see, but the scale is exactly one on an even draw either way.
    """
    assert _fight(3000.0, 3000.0, 3000.0, 0.0).outcome == 1.0
    assert _fight(3000.0, 3000.0, 0.0, 3000.0).outcome == -1.0
    assert _fight(2000.0, 4000.0, 2000.0, 0.0).outcome >= 1.0
    # And the middle of it is an even trade between sides of equal worth.
    assert _fight(2000.0, 2000.0, 1000.0, 1000.0).outcome == 0.0


def test_the_health_reading_of_a_fight_is_the_same_number_negated_as_well():
    """The second reading of a fight has to hold the property the first one does or it cannot be used for anything. A run of the handwritten layer against itself is the only statement there is about whether the arena leans, and it is only a statement about the arena if the two sides' figures are one number and its negation whatever happened in the fight.

    It holds for the same reason the sparse reading's does, and the reason is worth being able to see: both are the same subtraction of two shares with the terms exchanged, and counting a survivor at the health it has left changes what goes into the subtraction rather than the shape of it.
    """
    draw = random.Random(17)
    for _ in range(64):
        our_value = draw.uniform(1.0, 5000.0)
        their_value = draw.uniform(1.0, 5000.0)
        our_left = draw.uniform(0.0, our_value)
        their_left = draw.uniform(0.0, their_value)
        # Never above what is standing: a survivor at full health is worth its price and no more, so the health worth of a side lies between nothing and the worth of its survivors.
        ours = _fight(our_value, their_value, our_left, their_left,
                      draw.uniform(0.0, our_left), draw.uniform(0.0, their_left))
        theirs = _fight(their_value, our_value, their_left, our_left,
                        ours.their_left_health, ours.our_left_health)
        assert abs(ours.outcome + theirs.outcome) < 1e-12
        assert abs(ours.outcome_health + theirs.outcome_health) < 1e-12
        assert -1.0 - STRENGTH_SLOPE / 2 <= ours.outcome_health <= 1.0 + STRENGTH_SLOPE / 2

    # A side whose spawns never arrived reaches the point where it is scored like any other, and on this reading too that has to be a number rather than a division by nought.
    empty = _fight(0.0, 1200.0, 0.0, 0.0)
    assert empty.outcome_health + _fight(1200.0, 0.0, 0.0, 0.0).outcome_health == 0.0


def test_the_two_readings_agree_on_a_body_count_and_part_company_on_damage():
    """What the health reading is for, stated as the difference between the two.

    A fight that ended with somebody destroyed is worth the same under both, because a dead unit is worth nothing whichever way it is counted. That is what keeps every ceiling this project has quoted readable: the second reading existing renumbers none of them.

    Where the two part company is the common ending -both sides still standing and one of them shot to pieces. A fight is called twelve seconds after the last casualty, so three fights in four end that way, and under the sparse reading every one of those is worth precisely nothing to either side however one-sided the damage was. A side left at half health on every survivor loses half of that survivor's worth on the health reading and none of it on the other.
    """
    massacre = _fight(3000.0, 3000.0, 3000.0, 0.0)
    assert massacre.outcome == massacre.outcome_health == 1.0
    even = _fight(2000.0, 2000.0, 1000.0, 1000.0)
    assert even.outcome == even.outcome_health == 0.0

    # Nobody died and their survivors are at half health apiece. On the sparse reading that fight did not happen; on the other it cost them half of what they brought and is worth half the scale.
    halved = _fight(3000.0, 3000.0, 3000.0, 3000.0, our_health=3000.0, their_health=1500.0)
    assert halved.outcome == 0.0
    assert halved.outcome_health == 0.5
    # Which of the two is paid is what the arena was asked for; both are always computed and both go into the history row a run is read back from.
    assert halved.scored(BY_KILLS) == 0.0 and halved.scored(BY_HEALTH) == 0.5
    assert halved.as_dict()["outcome"] == 0.0 and halved.as_dict()["outcome_health"] == 0.5


def test_what_a_side_is_worth_on_health_is_its_own_survivors_weighted_by_what_is_left_of_them():
    """Taken from the unit rows, because the squad block carries the price of what is standing and nothing about how much of it is standing.

    Three things have to hold together or the figure is not a worth at all: only the squad's own units count, only units the board still reports count, and each counts for the share of its health it has left. A type the game reported no maximum health for counts whole, which is what the sparse reading says about it too and is therefore the only answer that cannot make the two readings differ for a reason that is not about the fight.
    """
    from rwintel.learn.arena import Arena

    # Assembled field by field rather than constructed, as everywhere else the arena is exercised without a game: what a side is worth needs the type catalogue and nothing else the constructor builds.
    arena = Arena.__new__(Arena)
    arena.catalogue = _CATALOGUE
    board = _observation(units=[_unit(1, health=100.0), _unit(2, health=50.0),
                                _unit(3, type_index=1, health=25.0), _unit(4, health=100.0)])
    # A tank whole, a tank at half and an artillery piece at a quarter. The fourth tank is on the board and in nobody's membership, so it is the other side's problem and not part of this figure.
    assert arena._health_worth(_squad(members=[1, 2, 3]), board) == 350.0 + 175.0 + 175.0
    # A member the board no longer reports is a member that is dead, and a dead unit is worth nothing on this reading exactly as on the other.
    assert arena._health_worth(_squad(members=[1, 2, 3, 99]), board) == 350.0 + 175.0 + 175.0
    assert arena._health_worth(_squad(members=[]), board) == 0.0

    # No maximum health recorded, so nothing about this unit is known to be missing and it counts for its price.
    unmeasured = _observation(units=[_unit(5, health=0.0, max_health=0.0)])
    assert arena._health_worth(_squad(members=[5]), unmeasured) == 350.0


def test_a_squad_is_filled_against_the_order_that_was_placed_and_not_with_whatever_appeared():
    """Two kinds of stranger reach the board while a fight is being formed, and neither was commissioned for it.

    One is this side's own base. A spawn-point player is given a headquarters and a builder and they land a step apart, so the wait for the opening board to settle can pass in the gap between the two; the headquarters is then inside the snapshot of what was already standing and the builder is not, and a squad filled with everything freshly arrived takes the builder in. Measured over eleven hundred episodes it happened in seven of every ten, always on this side and never on the baseless one, always worth exactly the builder. The other is an engagement abandoned as stillborn, whose spawn commands are never withdrawn and whose units surface later.

    Taking the arrivals against the order closes both, and the count matters as much as the type: a leftover of a type that was ordered is still a unit nobody ordered.
    """
    from rwintel.learn.arena import Arena

    ours = _CATALOGUE.types[0]
    arrivals = [_unit(1, type_index=ours.index), _unit(2, type_index=ours.index),
                # The builder: it is this side's, it has just appeared, and it is of no type the fight asked for.
                _unit(3, type_index=1), _unit(4, type_index=ours.index),
                _unit(5, type_index=ours.index, hostile=True)]
    assert Arena._commissioned(arrivals, False, {ours.index: 2}) == [1, 2]
    assert Arena._commissioned(arrivals, True, {ours.index: 1}) == [5]
    # With nothing to take the arrivals against there is nothing that says what does not belong, so everything of the right side is taken.
    assert Arena._commissioned(arrivals, False, None) == [1, 2, 3, 4]


def test_the_worth_a_squad_was_formed_with_is_not_overwritten_by_the_game_side():
    """The game keeps that figure as a running maximum over the life of a squad number and never lowers it, which is right for a squad reinforced over a match and wrong here.

    The arena hands the same two numbers to every fight of an episode, so after one big fight the game's figure is the largest force either slot ever held rather than the force standing in this one. It is the denominator of the squad's health, which is one of the layer's fifty-eight inputs, so leaving it alone is the difference between an input that says a fresh force is whole and one that says it is already half destroyed.
    """
    from rwintel.learn.arena import Arena

    import dataclasses

    from rwintel.wire.observation import SquadState

    arena = Arena.__new__(Arena)
    arena.squads = {0: _squad(members=[1], value=1000.0, formed_value=1000.0)}
    # The figure the game reports is deliberately the worth of a far bigger fight this squad number held earlier in the episode.
    reported = SquadState(id=0, commander=0, units=1, value=600.0, formed_value=9000.0, x=0.0, y=0.0,
                          spread=0.0, task_type=0, stance=0, target_region=0, status=0,
                          cost_budget=1000.0, budget_share=0.5, deadline_ms=60000, issued_at_ms=0,
                          losses=400.0)
    board = dataclasses.replace(_observation(units=[_unit(1)]), squads=[reported])
    arena._fold(board)
    assert arena.squads[0].value == 600.0
    # Whatever the game reports, the worth this fight was formed with is the arena's own figure.
    assert arena.squads[0].formed_value == 1000.0
    assert arena.squads[0].health == 0.6


def test_a_layer_refuses_a_score_it_was_not_taught():
    """A misspelt score would otherwise fall through to whichever reading the code tests for by name, and the run would be paid on one reading while whoever started it believed it was paying the other. Both readings are reported either way, so nothing in the log would say which had been paid. Refused where it is still one line rather than a measurement nobody can interpret afterwards."""
    refused = _refusal(ValueError, lambda: LearntTactics(None, _CATALOGUE, None, score="bodies"))
    assert "bodies" in refused
    # Named with what it would have taken, because the first thing anyone does with a refusal is go looking for the spelling.
    assert all(name in refused for name in SCORES)


# ---- where one errand stops and the next begins ---------------------------------------------

class _Fixed:
    """A decider that answers the same way whatever it is shown, so that what these tests look at is the bookkeeping around a decision rather than the decision."""

    def __init__(self, action=0):
        self.action = action

    def choose(self, state, mask):
        return Choice(action=self.action, log_prob=-1.6, value=0.25)


def _skirmish():
    """A squad of three with an enemy inside its engagement radius, which is the least that makes the tactical layer take a decision at all rather than report and stand by."""
    units = [_unit(1, 100.0, 100.0), _unit(2, 120.0, 100.0), _unit(3, 140.0, 100.0),
             _unit(9, 300.0, 120.0, hostile=1, type_index=1)]
    return _view(units, [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])


def test_ending_a_fight_pays_the_decision_it_was_still_owed_and_closes_the_trajectory():
    """The layer only ever sees periods; whoever is running the fight is the only one who knows it is over. So the last decision of a fight has to be paid from outside, or it is never paid at all.

    The figure it is paid is the outcome handed in plus one last shaping term taken against a terminal potential of nought, which is what makes the shaping over an errand telescope away to nothing and so leave the best policy where it was.
    """
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    squad = _squad()
    layer.decide(_skirmish(), [squad], 21000)
    held = layer.reward.missions[squad.id].potential
    assert layer.pending and held > 0.0

    layer.finish(squad, 0.0, 0.75, "called", engagement=4)
    trajectory, = rollout.done
    step, = trajectory.steps
    assert trajectory.finished and step.done and step.engagement == 4
    # The health reading is the one paid unless the layer was told otherwise, and the sparse one rides along in the signal.
    assert abs(step.reward - (0.75 - held)) < 1e-9
    assert not layer.pending and squad.id not in layer.reward.missions
    assert layer.terminals["called"] == 1
    kills_paid = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=Rollout(), instance=0, score=BY_KILLS)
    kills_paid.decide(_skirmish(), [squad], 21000)
    kills_paid.finish(squad, 0.25, 0.75, "called")
    assert abs(kills_paid.rollout.done[0].steps[0].reward - (0.25 - held)) < 1e-9


def test_a_decision_taken_after_the_fight_was_called_belongs_to_a_new_trajectory():
    """Squad numbers are handed round: the arena has two of them and uses them for every fight it builds, and a match has eight for as many errands as are ever run.

    So the decision left outstanding when one fight ends must not be paid out of the first period of the next one. Doing that strings two fights into a single trajectory, and generalised advantage estimation then runs the advantage of the second backwards into the decisions of the first -which teaches a layer that what it did in a fight it has already finished was answerable for what happened in a fight it had not yet begun.
    """
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    squad = _squad()
    view = _skirmish()
    layer.decide(view, [squad], 21000)
    layer.finish(squad, 0.0, 0.75, "called")
    layer.decide(view, [squad], 22000)
    layer.decide(view, [squad], 23000)

    closed, = rollout.done
    assert closed.finished and [step.at_ms for step in closed.steps] == [21000]
    running = rollout.live[(0, squad.id)]
    assert running is not closed
    assert [step.at_ms for step in running.steps] == [22000]


def test_a_squad_handed_a_new_contract_begins_a_new_trajectory():
    """A contract is the unit of work and therefore the unit of pay, so an errand ends when the layer above replaces it. The decisions of the errand before and of the errand after must not share a trajectory, or advantage estimation runs what the second one earned backwards into decisions taken for the first.

    Cut rather than closed. The errand that was replaced did not fail and did not finish; it stopped being observed, so its last decision is bootstrapped from its own value estimate exactly as an errand still running when the episode is called is. What that decision is paid is nothing, because the potentials of two contracts are measured against different ground and a difference between them is not a shaping term.
    """
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    squad = _squad()
    view = _skirmish()
    layer.decide(view, [squad], 21000)
    layer.decide(view, [squad], 22000)
    squad.contract = TaskContract(squad=squad.id, task=Task.ATTACK, target_region=1,
                                  stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                  deadline_ms=90000, issued_at_ms=23000)
    layer.decide(view, [squad], 23000)
    layer.decide(view, [squad], 24000)

    ended, = rollout.done
    assert not ended.finished and [step.at_ms for step in ended.steps] == [21000, 22000]
    assert ended.steps[-1].reward == 0.0 and ended.tail_value == ended.steps[-1].value
    assert [step.at_ms for step in rollout.live[(0, squad.id)].steps] == [23000]


def test_a_layer_built_with_a_discount_pays_its_shaping_at_that_discount():
    """One figure discounts the returns and telescopes the shaping, and a run states it once. It reaches the shaping only by being handed to the layer and passed on from there, so this is the join where the two could silently come apart -and a shaping term telescoping at one figure while the returns are discounted at another is a term that moves which policy is best.

    A layer nobody told is a layer inside a match, whose errand is a fragment of one, and the constant is that case.
    """
    assert LearntTactics(None, _CATALOGUE, _Fixed(), discount=1.0).reward.discount == 1.0
    assert LearntTactics(None, _CATALOGUE, _Fixed(), discount=0.5).reward.discount == 0.5
    assert LearntTactics(None, _CATALOGUE, _Fixed()).reward.discount == DISCOUNT


# ---- starting from something rather than from noise -----------------------------------------

def _decisions(seed, count=16):
    """One finished trajectory of tactical decisions, drained so that its advantages and returns are filled in.

    The boards differ from step to step on purpose. With the same board every time the trunk emits the same row every time, the value head's weights have no gradient at all, and a test that asked whether the critic moved would be answered by the bias alone.
    """
    draw = random.Random(seed)
    rollout = Rollout()
    for index in range(count):
        rollout.add("a", Step(state=[draw.uniform(-1.0, 1.0) for _ in range(TACTICAL_SIZE)],
                              action=draw.randrange(TACTICAL_ACTIONS),
                              mask=[1.0] * TACTICAL_ACTIONS, log_prob=-1.6, value=0.1,
                              reward=-1.0 if index % 3 == 0 else 1.0, done=index == count - 1))
    return rollout.drain()


def test_a_warming_update_moves_the_critic_and_leaves_the_policy_exactly_where_it_was():
    """A policy that arrived from an imitation did not arrive with a critic, so its first advantages are noise of about the size of the returns and a gradient taken against them takes the policy apart before it has been paid for anything.

    The trunk has to be held as well as the action head, and that is the point of the exercise rather than a precaution: it is shared, so a value loss let through to it moves the very features the action head reads. What is wanted at the end of a warm-up is a critic that has caught up with an actor that has not moved. And it has to be let go afterwards, because a run that left the trunk frozen would go on collecting, go on reporting, and never move the policy again.
    """
    torch.manual_seed(0)
    net = TacticalNet()
    optimiser = Optimiser(net, warmup=1)

    before = {name: parameter.detach().clone() for name, parameter in net.named_parameters()}
    warming = optimiser.update(_decisions(1))
    assert warming.warming and warming.policy_loss == 0.0 and warming.entropy == 0.0
    assert warming.value_loss > 0.0
    moved = {name for name, parameter in net.named_parameters()
             if not torch.equal(parameter, before[name])}
    assert moved == {"value.weight", "value.bias"}, sorted(moved)

    before = {name: parameter.detach().clone() for name, parameter in net.named_parameters()}
    after = optimiser.update(_decisions(2))
    assert not after.warming
    moved = {name for name, parameter in net.named_parameters()
             if not torch.equal(parameter, before[name])}
    # Everything but the action-value heads, which only offline reinforcement learning reads.
    assert moved == {name for name, _ in net.named_parameters() if not name.startswith("q.")}, sorted(moved)


# ---- the widened tactical action space ------------------------------------------------------

def test_the_handwritten_layer_reaches_the_two_added_departures():
    """The action space grew from five departures to seven, and the two added ones carry a choice a rule on the game side used to make: how far a withdrawal commits and which enemy a concentration goes onto. The imitation clones the handwritten layer and the reinforcement is measured against it, so the wider space is only worth training on if the handwritten layer actually reaches the two: a withdrawal is the whole way out when the squad is reported losing and a short step otherwise, and a concentration goes onto a longer-ranged enemy in reach and onto the weakest when none is."""
    tactics = Tactics(None, _CATALOGUE)

    def departure(view, squad):
        tactics._view, tactics._now = view, 0
        return tactics._departure(squad, [s for s in view.ours], [s for s in view.enemies],
                                  squad.losses, _Track())

    # Freshly hit members, so the squad reads as under fire and the choice is a withdrawal.
    beaten = [_unit(1, 100, 100, hit=100), _unit(2, 120, 100, hit=100), _unit(3, 140, 100, hit=100)]
    enemy = _unit(9, 210, 110, type_index=0, hostile=1)
    fight = _view(beaten + [enemy], [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])
    assert departure(fight, _squad(status=Status.LOSING, losses=900.0)) == Deviation.WITHDRAW_FAR
    assert departure(fight, _squad(status=Status.ACTIVE, losses=900.0)) == Deviation.WITHDRAW

    # Members not freshly hit, so a bunched squad is not read as covered by area fire and the choice reaches a concentration. Two enemies so that concentrating has a target to choose.
    ours = [_unit(1, 100, 100), _unit(2, 120, 100), _unit(3, 140, 100)]
    second = _unit(10, 180, 120, type_index=0, hostile=1)
    artillery = _view(ours + [_unit(9, 200, 100, type_index=1, hostile=1), second],
                      [_region(1, 400.0, 100.0, ours=200.0, theirs=300.0)])
    tanks_only = _view(ours + [_unit(9, 210, 110, type_index=0, hostile=1), second],
                       [_region(1, 400.0, 100.0, ours=200.0, theirs=300.0)])
    assert departure(artillery, _squad(status=Status.ACTIVE, losses=0.0)) == Deviation.FOCUS_THREAT
    assert departure(tanks_only, _squad(status=Status.ACTIVE, losses=0.0)) == Deviation.FOCUS


# ---- which side of a fight is decided first --------------------------------------------------

class _Recorder:
    """A tactical layer that answers the same way whatever it is shown and writes down that it was asked, so that what is under test is the order the two sides of a fight are decided and submitted in."""

    def __init__(self, name: str, calls: list) -> None:
        self.name, self.calls = name, calls

    def decide(self, view, squads, game_time_ms):
        self.calls.append(self.name)
        return [SquadDeviation(squad=squad.id, deviation=Deviation.HOLD) for squad in squads], []


def _arena_at(order: str, calls: list):
    """An arena far enough built to fight one period of one engagement, without a session, a map or a game.

    Assembled field by field rather than constructed, because everything the constructor does -the type catalogue, the two layers, the sandbox -needs a live session, and none of it bears on which of two already-built layers is asked first.
    """
    from rwintel.learn.arena import Arena, OURS, THEIRS, Statistics

    arena = Arena.__new__(Arena)
    arena.catalogue, arena.statistics = _CATALOGUE, Statistics()
    arena.decision_order, arena.stall_ms = order, 12000
    arena.tactics, arena.opponent = _Recorder("ours", calls), _Recorder("theirs", calls)
    arena.squads = {OURS: _squad(members=(1, 2, 3)), THEIRS: _squad(members=(9,))}
    arena.squads[THEIRS].id = THEIRS
    arena.engagement, arena.last_regions = None, []
    arena.until_ms, arena._alive, arena._changed_ms = 10 ** 9, 4, 0
    return arena


def test_the_two_sides_of_a_fight_are_decided_in_the_order_the_arena_was_asked_for():
    """Both sides read the same frame and neither can see what the other chose, so the order they are decided in ought not to matter. A left-right lean the score cannot absorb has been measured in the fighting all the same, and the order is the one thing about a period that is not symmetric between the sides, so it has to be something a run can set: fixed either way to measure whether the lean follows it, and alternating to answer a lean that does.

    Alternating is per period rather than per fight. A fight is a few hundred periods, so the side that leads has to change inside a fight for the advantage of leading to be dealt evenly within the fight that is being scored.
    """
    from rwintel.learn.arena import ALTERNATING, OURS_FIRST, THEIRS_FIRST

    observation = _observation(units=[_unit(1, 100.0, 100.0), _unit(2, 120.0, 100.0),
                                      _unit(3, 140.0, 100.0), _unit(9, 300.0, 120.0, hostile=1)],
                               regions=[_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])
    view = build_view(observation, _CATALOGUE, None)

    for order, expected in ((OURS_FIRST, ["ours", "theirs"]), (THEIRS_FIRST, ["theirs", "ours"])):
        calls: list = []
        arena = _arena_at(order, calls)
        action = Action()
        arena._fight(observation, view, action, 21000)
        assert calls == expected
        # The departures reach the action in the order they were decided, which is the order the game side applies them in.
        assert [deviation.squad for deviation in action.deviations] == ([0, 1] if order == OURS_FIRST else [1, 0])

    calls = []
    arena = _arena_at(ALTERNATING, calls)
    for period in range(6):
        arena._fight(observation, view, Action(), 21000 + period * 200)
    assert calls == ["ours", "theirs", "theirs", "ours"] * 3
    # Only this side's decisions are the run's own output; the opposing layer's are the environment.
    assert arena.statistics.tactical == 6 and arena.statistics.decisions == 6


def test_a_layer_pinned_to_one_departure_answers_with_it_and_writes_nothing_down():
    """The ablation the band a policy plays inside is measured with: give up the choice, keep everything else, and see what the score loses. It was kept as hand-made parameter files until the action space went from five departures to seven and they stopped loading, so it lives here now, where it cannot go stale and where naming a departure that does not exist is refused rather than measured."""
    from rwintel.learn.__main__ import _pinned
    from rwintel.learn.deciders import PinnedDeparture

    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, PinnedDeparture(Deviation.WITHDRAW_FAR.value),
                          rollout=None, instance=0)
    view, squad = _skirmish(), _squad()
    deviations, _ = layer.decide(view, [squad], 21000)
    assert [deviation.deviation for deviation in deviations] == [Deviation.WITHDRAW_FAR]
    # Nothing to learn from and nowhere to put it: an ablation is read, not trained.
    assert not rollout.live and not rollout.done

    class _Asked:
        pin = "hold, withdraw_far"

    assert _pinned(_Asked()) == [Deviation.HOLD, Deviation.WITHDRAW_FAR]
    _Asked.pin = "sidestep"
    assert "sidestep" in _refusal(SystemExit, _pinned, _Asked())


def test_every_episode_draws_a_different_fight_and_the_arms_of_a_run_draw_the_same_ones():
    """An arena is built afresh for every episode and draws its sites, budgets, imbalances and forces from the seed it is built with. Built from the instance alone, every episode of an instance drew the identical sequence: a run of fifty episodes on seven instances was about fifty distinct fights fought forty times over, and its two thousand fight rows were reported as two thousand samples. That is a sample size overstated by a factor of thirty to fifty, and it is what every left-right lean the arena was charged with turned out to be made of.

    The arms of a comparison are the exception, and deliberately: they are given the same draw as each other in the same round, so that a policy and its baseline meet the same sites, the same budgets and the same forces and what is left between them is the play.
    """
    from rwintel.learn.__main__ import _arena_seed

    arguments = _Arguments(seed=4242)
    for instances in (1, 7):
        for arms in (1, 2):
            seeds = [[_arena_seed(arguments, _Session(instance, arms, record))
                      for record in range(arms * 6)] for instance in range(instances)]
            for stream in seeds:
                rounds = [stream[index * arms:(index + 1) * arms] for index in range(6)]
                # Every arm of one round draws the same fights, and every round draws different ones.
                assert all(len(set(round_)) == 1 for round_ in rounds)
                assert len({round_[0] for round_ in rounds}) == 6
            # No two instances share a draw, whatever episode either of them is on.
            assert len({seed for stream in seeds for seed in stream}) == instances * 6


class _Arguments:
    def __init__(self, seed: int) -> None:
        self.seed = seed


class _Session:
    """A session as far as the arena's seed reads one: which instance it is, how many arms the run has, and how many episodes have finished."""

    def __init__(self, instance: int, arms: int, records: int) -> None:
        self.instance, self.arms, self.records = instance, [("arm", None)] * arms, [None] * records


# ---- reading the fights a run has already scored ----------------------------------------------

def test_a_runs_fights_are_summarised_from_its_episodes_without_keeping_every_fight():
    """A run reports the mean and the spread of its fights, and it holds neither: an episode record carries its own count, mean and spread, and the run is put back together from those. That is what lets a training run be read for the fights it has already scored -three times as many as the duel that measured the same policy -and it has to give exactly what the flat list of fights would.

    The spread is on the sample, matching every other scatter quoted in this project and the sample sizes computed from them.
    """
    from rwintel.learn.__main__ import _summarise

    draw = random.Random(11)
    episodes = [[draw.uniform(-1.0, 1.0) for _ in range(draw.randint(1, 9))] for _ in range(20)]
    records = []
    for index, fights in enumerate(episodes):
        summary = Summary.of(fights)
        # Written the way an arena episode writes itself, spread included, and rounded the way the journal rounds it.
        records.append(_Record(index + 1, {"fought": summary.n, "outcome_mean": round(summary.mean, 4),
                                           "outcome_sd": round(_population_sd(fights), 4)}))

    flat = [outcome for fights in episodes for outcome in fights]
    whole = _summarise(records)
    assert whole.n == len(flat)
    assert abs(whole.mean - Summary.of(flat).mean) < 1e-4
    assert abs(whole.sd - Summary.of(flat).sd) < 1e-4
    # The later half is the run's own second half by episode, which is what a training run reports beside the whole because its policy moved while it was scoring.
    later = _summarise([record for record in records if record.episode * 2 > len(episodes)])
    assert later.n == sum(len(fights) for fights in episodes[len(episodes) // 2:])


def _population_sd(values):
    """The spread an arena episode reports, which divides by n rather than by n-1: it is the spread of the fights that episode had, not an estimate drawn from them."""
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


class _Record:
    """An episode record as far as a summary reads one."""

    def __init__(self, episode: int, statistics: dict) -> None:
        self.episode, self.statistics = episode, statistics
        self.arm, self.instance = "duel", 0


# Batched inference.

def _batched(window: float):
    from rwintel.learn.inference import Batcher

    calls = []

    def evaluate(requests):
        calls.append(list(requests))
        return [request * 10 for request in requests]

    return Batcher(evaluate, window=window), calls


def test_a_batch_goes_out_as_soon_as_every_deciding_thread_is_waiting():
    """Nothing more can arrive once every thread in a decision is waiting, so the batch goes then rather than at the end of the window."""
    import threading
    import time

    from rwintel.control.deciding import deciding

    batcher, calls = _batched(window=30.0)
    together = threading.Barrier(2)
    answers = {}

    def decide(value):
        with deciding():
            together.wait()
            answers[value] = batcher.submit(value)

    try:
        started = time.monotonic()
        threads = [threading.Thread(target=decide, args=(value,)) for value in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        assert time.monotonic() - started < 5.0
        assert answers == {1: 10, 2: 20}
        assert len(calls) == 1 and sorted(calls[0]) == [1, 2]
    finally:
        batcher.stop()


def test_a_request_from_outside_any_decision_waits_out_the_window():
    import time

    batcher, calls = _batched(window=0.2)
    try:
        started = time.monotonic()
        assert batcher.submit(3) == 30
        assert time.monotonic() - started >= 0.15
        assert calls == [[3]]
    finally:
        batcher.stop()


# ---- the operational layer -------------------------------------------------------------------

class _Orders:
    posture = 0
    priorities = {1: 1.0}
    offensive = True
    loss_allowance = 2000.0
    expansion = ()


def _ops_region(region_id, ours=0.0, theirs=0.0, held=0, enemy_held=0, distance=500.0):
    return RegionState(id=region_id, resources=2, held_by_us=held, held_by_enemy=enemy_held,
                       x=400.0 * region_id, y=0.0, our_value=ours, enemy_value=theirs,
                       enemy_seen_at_ms=0, distance_from_home=distance)


def test_an_unchanging_board_pays_nothing_by_default_and_the_achievement_flow_when_it_is_weighed_in():
    """The default pay is the match score and the change in the score expected, so a board that stands still pays nothing until the match ends. Weighed in, the achievement of the strategic orders is paid as a flow: a board standing on ground the strategic layer asked for pays every period, and one standing on the enemy's ground charges every period."""
    from rwintel.learn.reward import ACHIEVEMENT_RATE

    ours = _view([], [_ops_region(1, ours=900.0, held=2)])
    theirs = _view([], [_ops_region(1, theirs=900.0, enemy_held=2)])
    for board, sign in ((ours, 1.0), (theirs, -1.0)):
        for weight in (0.0, 1.0):
            terms = OperationalTerms(achievement_weight=weight)
            reward = OperationalReward(terms)
            assert terms.pay(reward.signal(board, _Orders(), [], 600.0)) == 0.0
            paid = [terms.pay(reward.signal(board, _Orders(), [], 600.0)) for _ in range(3)]
            assert all(abs(payment - weight * sign * ACHIEVEMENT_RATE) < 1e-9 for payment in paid), paid


def test_a_match_layers_pay_adds_up_to_the_score_less_the_expectation_it_opened_on():
    """Undiscounted and shaped by the expected score, the payments of a match, its end included, add up to the score less the expectation of the first board, whatever the boards in between were and however the expectation moves with the time left. Shaped by the worth share at a discount below one, they add up to the discounted score less the first potential, plus the flows: the shaping telescopes either way."""
    from rwintel.learn.predictor import Predictor
    from rwintel.learn.reward import CLOSED, EconomicReward, EconomicTerms, value_share

    predictor = Predictor(knots=(0.0, 300.0, 600.0), value=(1.0, 0.6, 0.2), ground=(0.0, 0.3, 0.5))
    units = [[_unit(1)], [_unit(1), _unit(2)], [_unit(1), _unit(9, hostile=1)], [_unit(3), _unit(8, hostile=1)]]
    boards = [_view(u, [_ops_region(1, ours=100.0 * i, theirs=50.0 * i, held=i % 2, enemy_held=1)])
              for i, u in enumerate(units, start=1)]
    times = (650.0, 420.0, 180.0, 20.0)
    score = 0.35
    for terms in (OperationalTerms(predictor=predictor), EconomicTerms(predictor=predictor),
                  OperationalTerms(discount=0.9, shaping=SHARE, value_flow=0.05),
                  EconomicTerms(discount=0.9, shaping=SHARE, ground_flow=0.02)):
        operational = isinstance(terms, OperationalTerms)
        reward = OperationalReward(terms) if operational else EconomicReward(terms)
        rows = [reward.signal(board, _Orders(), [], t) if operational else reward.step(board, t).signal
                for board, t in zip(boards, times)]
        rows.append(reward.end(score))
        assert int(rows[-1][0]) == CLOSED
        value, ground = (6, 8) if operational else (4, 6)
        paid = terms.reward(rows[1:])
        if terms.shaping == SHARE:
            d = terms.discount
            flows = sum(d ** k * terms.flows(row[value], row[ground]) for k, row in enumerate(rows[1:-1]))
            assert flows != 0.0
            expected = flows + d ** (len(rows) - 2) * score - terms.share_weight * value_share(boards[0])
        else:
            expected = score - predictor.potential(times[0], rows[0][value], rows[0][ground])
        assert abs(paid - expected) < 1e-9


def test_the_worth_share_is_the_same_cut_on_both_sides_and_is_paid_as_a_flow_only_when_asked():
    """Both sides' units and buildings count, so the board seen from the other side reads one less our share; a flow of it is paid only when the terms ask for one."""
    from rwintel.learn.reward import value_share

    units = [_unit(1), _unit(2, type_index=1), _unit(8, hostile=1), _unit(9, hostile=1, type_index=1)]
    observation = _observation(units, [_ops_region(1)])
    ours = build_view(observation, _CATALOGUE, None)
    theirs = build_view(observation, _CATALOGUE, None, invert=True)
    assert abs(value_share(ours) + value_share(theirs) - 1.0) < 1e-12
    ahead = _view([_unit(1), _unit(2)], [_ops_region(1)])
    assert OperationalTerms().flows(value_share(ahead), 0.5) == 0.0
    assert abs(OperationalTerms(value_flow=0.05).flows(value_share(ahead), 0.5) - 0.05) < 1e-9


def test_losses_beyond_the_allowance_are_charged_when_the_orders_are_weighed_in():
    from rwintel.learn.reward import ELAPSED, operational_row

    board = (600.0, 0.5, 0.5)
    within = [_squad(losses=1000.0)]
    beyond = [_squad(losses=5000.0)]
    assert OperationalReward.overspend(_Orders(), within) == 0.0
    terms = OperationalTerms(achievement_weight=1.0)

    def paid(squads):
        return terms.pay(operational_row(ELAPSED, board, board, overspend=OperationalReward.overspend(_Orders(), squads)))

    assert paid(beyond) < paid(within) == 0.0


def test_a_decision_held_for_several_periods_is_discounted_as_several():
    """One period apiece is the ordinary estimator; a decision that stood for k periods discounts what follows it by the discount to the k."""
    rollout = Rollout(discount=0.9, trace=1.0)
    first = Step(state=[0.0], action=0, mask=[1.0], reward=1.0, value=0.0, squad=0, periods=3)
    second = Step(state=[0.0], action=0, mask=[1.0], reward=0.0, value=2.0, squad=0, periods=1)
    rollout.add((0, 0), first)
    rollout.add((0, 0), second)
    rollout.cut((0, 0), tail_value=2.0)
    rollout.drain()
    assert abs(first.ret - (1.0 + 0.9 ** 3 * (0.0 + 0.9 * 2.0))) < 1e-9

    single = Rollout(discount=0.9, trace=0.5)
    steps = _steps(3, reward=1.0, value=0.5)
    for step in steps:
        single.add((0, 0), step)
    single.drain()
    assert abs(steps[-1].advantage - 0.5) < 1e-9
    assert abs(steps[-2].advantage - ((1.0 + 0.9 * 0.5 - 0.5) + 0.9 * 0.5 * 0.5)) < 1e-9


class _Counting:
    """A decider that sends every squad to region 2 under the first task it may take, and counts how often it was asked."""

    def __init__(self):
        self.asked = 0

    def choose(self, state, slot, region_mask, task_mask):
        self.asked += 1
        return Choice(action=2, second=[i for i, allowed in enumerate(task_mask) if allowed][0],
                      log_prob=-1.0, value=0.25)


def _ops_board(t, status=Status.ACTIVE):
    units = [_unit(1, 100, 100, squad=0), _unit(2, 120, 100, squad=0), _unit(3, 140, 100, squad=0)]
    regions = [_ops_region(1, distance=0.0), _ops_region(2, theirs=300.0, distance=800.0)]
    return build_view(_observation(units, regions, game_time_ms=t), _CATALOGUE, 1)


def _ops_layer(decider, rollout=None):
    from rwintel.learn.layers import LearntOperations

    return LearntOperations(None, _CATALOGUE, decider, rollout, instance=0)


def test_the_learnt_operational_layer_decides_only_when_something_calls_for_it():
    """A running errand is left alone until its report says it stalled, went badly, finished or ran out, or until the review interval comes round; in between no decision is taken and none is recorded."""
    from rwintel.learn.layers import REVIEW_MS

    decider = _Counting()
    rollout = Rollout()
    layer = _ops_layer(decider, rollout)
    squad = _squad(contract=False, value=1050.0, formed_value=1050.0, losses=0.0)
    layer.decide(_ops_board(30000), _Orders(), [squad], [], 30000)
    assert decider.asked == 1 and squad.contract is not None and squad.contract.target_region == 2

    for t in (32000, 34000, 30000 + REVIEW_MS - 2000):
        layer.decide(_ops_board(t), _Orders(), [squad], [], t)
    assert decider.asked == 1
    assert layer.pending[0].periods == 3

    squad.status = Status.STALLED
    layer.decide(_ops_board(30000 + REVIEW_MS), _Orders(), [squad], [], 30000 + REVIEW_MS)
    assert decider.asked == 2
    filed = rollout.live[(0, 0)].steps
    assert len(filed) == 1 and filed[0].periods == 4

    squad.status = Status.ACTIVE
    layer.decide(_ops_board(30000 + 2 * REVIEW_MS), _Orders(), [squad], [], 30000 + 2 * REVIEW_MS)
    assert decider.asked == 3


def test_a_decision_can_be_paid_its_own_squads_exchange():
    """With the local exchange on, what a squad destroyed less what it lost in a period is paid to the decision standing for it, and only what changed since the period before; a new contract counts from nought."""
    from rwintel.control.policy.contracts import MissionReport
    from rwintel.learn.layers import LearntOperations
    from rwintel.learn.reward import EXCHANGE_SCALE

    def run(weight, reports_by_period):
        layer = LearntOperations(None, _CATALOGUE, _Counting(), Rollout(), instance=0,
                                 terms=OperationalTerms(local_exchange=weight))
        squad = _squad(contract=False, losses=0.0)
        for index, reports in enumerate(reports_by_period):
            t = 30000 + 2000 * index
            layer.decide(_ops_board(t), _Orders(), [squad], reports, t)
        return layer.pending[0].reward, squad

    quiet = [[], [], []]
    base, squad = run(0.1, quiet)
    fought = [[], [MissionReport(squad=0, status=Status.ACTIVE, losses=100.0, destroyed=600.0)],
              [MissionReport(squad=0, status=Status.ACTIVE, losses=100.0, destroyed=900.0)]]
    paid, _ = run(0.1, fought)
    discount = OperationalTerms().discount
    expected = 0.1 * (500.0 / EXCHANGE_SCALE) + discount * 0.1 * (300.0 / EXCHANGE_SCALE)
    assert abs((paid - base) - expected) < 1e-9
    unpaid, _ = run(0.0, fought)
    assert abs(unpaid - base) < 1e-12


def test_a_squad_that_leaves_the_board_is_cut_rather_than_ended():
    """The payment is for the board, which goes on after the squad, so its last decision is bootstrapped from its value rather than closed at nought."""
    rollout = Rollout()
    layer = _ops_layer(_Counting(), rollout)
    squad = _squad(contract=False, losses=0.0)
    layer.decide(_ops_board(30000), _Orders(), [squad], [], 30000)
    layer.decide(_ops_board(32000), _Orders(), [], [], 32000)
    assert not rollout.live and len(rollout.done) == 1
    assert not rollout.done[0].finished and rollout.done[0].tail_value == 0.25


def test_a_collected_decision_is_the_contract_the_script_actually_issues():
    """With no decider the script decides, and what is written down has to be what it issued, with the judge's scores softened into distributions beside it."""
    rollout = Rollout()
    layer = _ops_layer(None, rollout)
    squad = _squad(contract=True, losses=0.0)
    squad.contract.target_region = 1
    squad.contract.task = Task.ATTACK
    layer.decide(_ops_board(30000), _Orders(), [squad], [], 30000)
    recorded = layer.pending[0]
    assert (recorded.action, recorded.second) == (squad.contract.target_region, plan_of(squad.contract.task, -1))
    assert abs(sum(recorded.soft) - 1.0) < 1e-9 and max(recorded.soft) == recorded.soft[recorded.action]
    assert recorded.second_soft[recorded.second] == max(recorded.second_soft)


def test_the_plan_head_reads_the_region_it_was_chosen_for():
    from rwintel.learn.net import OperationalNet, one_hot_slot

    torch.manual_seed(0)
    net = OperationalNet()
    with torch.no_grad():
        net.plan.weight.normal_()
    state = torch.zeros(1, OPERATIONAL_SIZE)
    slot = one_hot_slot(0).unsqueeze(0)
    plans = torch.tensor([plan_masks(_view([], [_ops_region(1)]), Doctrine.VANGUARD)[1]])
    with torch.no_grad():
        _, near, _ = net(state, slot, None, plans, torch.tensor([1]))
        _, far, _ = net(state, slot, None, plans, torch.tensor([7]))
    assert not torch.allclose(near, far)
    assert float(near[0, plan_of(Task.DEFEND, -1)]) < -1e8
    assert float(near[0, plan_of(Task.ATTACK, 0)]) < -1e8 and float(near[0, plan_of(Task.ATTACK, -1)]) > -1e8


def test_a_report_about_a_superseded_contract_reads_as_a_fresh_errand():
    """The game reports on the contract it holds, which for one period after a new one is issued is the old one; its status and losses belong to the errand that was replaced."""
    from rwintel.control.policy.organisation import Organisation
    from rwintel.wire.observation import SquadState

    organisation = Organisation(None, None)
    squad = _squad(losses=0.0)
    organisation.squads[0] = squad
    organisation.free_ids.remove(0)

    def report(issued_at):
        state = SquadState(id=0, commander=0, units=3, value=1000.0, formed_value=1400.0, x=0.0, y=0.0,
                           spread=10.0, task_type=0, stance=5, target_region=1, status=int(Status.STALLED),
                           cost_budget=1000.0, budget_share=0.5, deadline_ms=90000, issued_at_ms=issued_at,
                           losses=400.0)
        return Observation(frame=1, game_time_ms=30000, episode=1, blocks=BLOCK_SQUADS, slot=0, credits=0.0,
                           income=0.0, units=3, unit_cap=100, under_construction=0, killed_units=0,
                           killed_buildings=0, lost_units=0, lost_buildings=0, squads=[state])

    organisation._fold(report(squad.contract.issued_at_ms - 2000))
    assert squad.status is Status.ACTIVE and squad.losses == 0.0
    organisation._fold(report(squad.contract.issued_at_ms))
    assert squad.status is Status.STALLED and squad.losses == 400.0


def test_a_garrison_holding_its_ground_is_running_and_an_attack_that_took_it_is_not():
    from rwintel.control.policy.operations import running

    garrison = _squad(status=Status.COMPLETE)
    garrison.contract.task = Task.DEFEND
    attack = _squad(status=Status.COMPLETE)
    assert running(garrison) and not running(attack)
    assert running(_squad(status=Status.ACTIVE)) and not running(_squad(status=Status.STALLED))


def test_an_intervention_spoils_only_the_operational_decision_standing_when_it_happened():
    """Decisions paid for periods nobody interfered with stay in the signal; the one standing when a squad was touched, and the one standing for any squad units were moved into, do not."""
    from rwintel.control.intruder import Log
    from rwintel.learn.policy import OPERATIONAL, LearningPolicy

    class _Commander:
        log = Log()

    rollout = Rollout()
    layer = _ops_layer(_Counting(), rollout)
    squads = [_squad(id=0, contract=False, losses=0.0), _squad(id=1, contract=False, losses=0.0),
              _squad(id=2, contract=False, losses=0.0)]
    layer.decide(_ops_board(30000), _Orders(), squads, [], 30000)
    earlier = dict(layer.pending)

    policy = LearningPolicy.__new__(LearningPolicy)
    policy.layer, policy.operations, policy._seen_events = OPERATIONAL, layer, {}
    commander = _Commander()
    policy.outside = [commander]
    commander.log.events.append({"kind": "reassign", "squad": 0, "into": 1, "at_ms": 31000})
    policy._taint_standing()
    assert earlier[0].tainted and earlier[1].tainted and not earlier[2].tainted

    for squad in squads:
        squad.status = Status.STALLED
    layer.decide(_ops_board(32000), _Orders(), squads, [], 32000)
    policy._taint_standing()
    assert not any(step.tainted for step in layer.pending.values())


def test_the_anchor_holds_a_policy_nearer_the_one_it_started_from():
    """Updates driven by noise move a policy away from where it started; the anchor charges that divergence, so the same updates leave an anchored policy nearer its start, for both layers."""
    from rwintel.learn.net import OperationalNet, TacticalNet, one_hot_slot
    from rwintel.learn.train import _kl

    def drift(anchor, two_headed):
        torch.manual_seed(3)
        net = OperationalNet() if two_headed else TacticalNet()
        start = [p.detach().clone() for p in net.parameters()]
        optimiser = Optimiser(net, two_headed=two_headed, anchor_weight=anchor, learning_rate=3e-3)
        rng = random.Random(5)
        size = OPERATIONAL_SIZE if two_headed else TACTICAL_SIZE
        actions = OPERATIONAL_REGIONS if two_headed else TACTICAL_ACTIONS
        for _ in range(6):
            steps = [Step(state=[rng.uniform(-1, 1) for _ in range(size)], action=rng.randrange(actions),
                          mask=[1.0] * actions, second=rng.randrange(2) if two_headed else -1,
                          second_mask=([1.0, 1.0] + [0.0] * (OPERATIONAL_PLANS - 2)) * actions if two_headed else (),
                          advantage=rng.gauss(0, 1), ret=0.0, squad=0) for _ in range(64)]
            optimiser.update(steps)
        fresh = OperationalNet() if two_headed else TacticalNet()
        with torch.no_grad():
            for parameter, value in zip(fresh.parameters(), start):
                parameter.copy_(value)
        states = torch.stack([torch.tensor(step.state) for step in steps])
        masks = torch.ones(len(steps), actions)
        with torch.no_grad():
            if two_headed:
                slots = torch.stack([one_hot_slot(0) for _ in steps])
                now, _, _ = net(states, slots, masks)
                then, _, _ = fresh(states, slots, masks)
            else:
                now, _ = net(states, masks)
                then, _ = fresh(states, masks)
        return float(_kl(now, then).mean())

    for two_headed in (False, True):
        assert drift(5.0, two_headed) < drift(0.0, two_headed)


def test_the_operational_ablations_answer_from_the_encoded_board():
    from rwintel.learn.deciders import PinnedOperations

    regions = [_ops_region(0, distance=0.0), _ops_region(3, theirs=500.0, distance=900.0),
               _ops_region(5, distance=2500.0)]
    view = _view([], regions)
    state = operational_state(view, None, [], 30000)
    mask = region_mask(view)
    plans = plan_masks(view, Doctrine.GARRISON)
    assert PinnedOperations("home").choose(state, 0, mask, plans).action == 0
    assert PinnedOperations("nearest").choose(state, 0, mask, plans).action == 3

    rich = [_ops_region(0, distance=0.0), _ops_region(3, theirs=900.0, enemy_held=1, distance=900.0),
            _ops_region(5, theirs=100.0, enemy_held=2, distance=2500.0), _ops_region(7, distance=3000.0)]
    board = _view([_unit(1)], rich)
    both = operational_state(board, None, [], 30000, spawns=(0, 7))
    wide = region_mask(board)
    many = plan_masks(board, Doctrine.GARRISON)
    assert PinnedOperations("weakest").choose(both, 0, wide, many).action == 5
    assert PinnedOperations("richest").choose(both, 0, wide, many).action == 5
    assert PinnedOperations("spawn").choose(both, 0, wide, many).action == 7
    assert PinnedOperations("random", seed=1).choose(state, 0, mask, plans).action in (0, 3, 5)
    assert PinnedOperations("home").choose(state, 0, mask, plans).second == plan_of(Task.DEFEND, -1)


def test_the_carry_ablation_takes_a_lift_where_one_is_open_and_walks_otherwise():
    from rwintel.control.policy.encoding import MEANS
    from rwintel.learn.deciders import PinnedOperations

    regions = [_ops_region(0, distance=0.0), _ops_region(3, theirs=500.0, distance=900.0),
               _ops_region(5, distance=2500.0)]
    view = _view([], regions)
    state = operational_state(view, None, [], 30000)
    mask = region_mask(view)
    walking = plan_masks(view, Doctrine.GARRISON)
    walk = next(i for i, allowed in enumerate(walking[3]) if allowed > 0)
    assert walk % MEANS == 0

    choice = PinnedOperations("carry").choose(state, 0, mask, walking)
    assert (choice.action, choice.second) == (3, walk)

    lifted = [list(row) for row in walking]
    lift = walk + 2
    lifted[3][lift] = 1.0
    choice = PinnedOperations("carry").choose(state, 0, mask, lifted)
    assert (choice.action, choice.second) == (3, lift)
    assert choice.second_probabilities[lift] == 1.0
    assert PinnedOperations("nearest").choose(state, 0, mask, lifted).second == walk


# ---- the economy ----------------------------------------------------------------------------

_ECONOMY_TYPES = [
    UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0, hits_air=False, hits_land=True),
    UnitType(index=1, name="builder", lookup="builder", price=500, tech=1, building=False, builder=True,
             movement="LAND", menu=(2,)),
    UnitType(index=2, name="landFactory", lookup="landFactory", price=700, tech=1, building=True, builder=False,
             movement="NONE", menu=(1, 0)),
]

_ECONOMY_CATALOGUE = _Catalogue(_ECONOMY_TYPES)


class _EconomySession:
    regions = []
    map_content = None

    def type_by_lookup(self, lookup):
        return next((kind for kind in _ECONOMY_TYPES if kind.lookup == lookup), None)


def _economy_view(t, credits=1200.0):
    """One builder with nothing to do and one idle factory, on ground that is all ours."""
    units = [_unit(1, 100.0, 100.0, type_index=1), _unit(10, 150.0, 100.0, type_index=2)]
    region = RegionState(id=0, resources=2, held_by_us=2, held_by_enemy=0, x=100.0, y=100.0, our_value=0.0,
                         enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=0.0)
    observation = Observation(frame=10, game_time_ms=t, episode=1, blocks=BLOCK_REGIONS | BLOCK_UNITS, slot=0,
                              credits=credits, income=10.0, units=len(units), unit_cap=100, under_construction=0,
                              killed_units=0, killed_buildings=0, lost_units=0, lost_buildings=0,
                              regions=[region], unit_states=units)
    return build_view(observation, _ECONOMY_CATALOGUE, 0)


def _economy_orders():
    from rwintel.control.policy.contracts import ALLOCATION, EconomyOrders, Posture, Role

    return EconomyOrders(posture=Posture.ARM, allocation=ALLOCATION[Posture.ARM], tech_cap=0.0,
                         target_mix={Role.ARMOUR: 1.0})


def _economic_board(**fields):
    from rwintel.control.policy.contracts import Posture
    from rwintel.control.policy.encoding import EconomicBoard

    values = dict(credits=0.0, reserve=0.0, income=0.0, units=0, unit_cap=0, near_cap=False, at_cap=False,
                  factories=0, pending_factory=False, free_factories=0, factory_raising=False, builders=0,
                  builder_shortfall=0, idle_builders=0, extractors=0, home_open=0, plan_open=0, posture=Posture.EXPAND,
                  economy_share=0.7, military_share=0.2, tech_share=0.1, tech_cap=0.0, tech_fund=0.0,
                  tech_fund_limit=8000.0, tech_level=1)
    values.update(fields)
    return EconomicBoard(**values)


def test_the_economic_feature_names_and_vectors_agree_and_stay_in_range_whatever_the_treasury():
    """The same two promises as the other cuts: named features the vector is exactly as long as, and every one a ratio inside its range, however rich or empty the board."""
    from rwintel.control.policy.contracts import Role
    from rwintel.control.policy.encoding import (
        ECONOMIC_CONTEXT,
        ECONOMIC_SIZE,
        INVESTMENT_FEATURES,
        INVESTMENT_SLOTS,
        Investment,
        Offer,
        economic_state,
        lay_out,
    )

    assert ECONOMIC_SIZE == len(ECONOMIC_CONTEXT) + INVESTMENT_SLOTS * len(INVESTMENT_FEATURES)
    assert len(economic_state(_economic_board(), lay_out([]))) == ECONOMIC_SIZE
    offers = [Offer(kind=Investment.UNIT, price=90000.0, type_index=i, efficiency=1e9 * (i + 1),
                    unit_efficiency=1e12, role=Role.ARMOUR, role_rank=1.0, range=9999.0, tier=9, army_share=5.0)
              for i in range(40)]
    offers += [Offer(kind=Investment.EXTRACTOR, price=700.0, region=i, safety=-1e6, distance=1e6, open_points=99)
               for i in range(9)]
    offers += [Offer(kind=Investment.RAISE, price=1e6, stage=1, gain=50.0)]
    rich = _economic_board(credits=1e7, reserve=5e7, income=1e5, units=900, unit_cap=10, factories=90,
                           free_factories=90, builders=90, builder_shortfall=-90, idle_builders=90, extractors=900,
                           home_open=90, plan_open=90, tech_cap=1e6, tech_fund=1e6, tech_level=9, picks=900,
                           game_time_ms=10 ** 9, target_mix={Role.ARMOUR: 5.0})
    state = economic_state(rich, lay_out(offers))
    assert len(state) == ECONOMIC_SIZE
    assert all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in state)


def test_an_economic_decision_scores_every_offer_by_one_network_and_never_an_empty_slot():
    from rwintel.control.policy.encoding import ECONOMIC_SIZE, INVESTMENT_SLOTS
    from rwintel.learn.net import MASKED, EconomicNet

    net = EconomicNet()
    mask = torch.zeros(2, INVESTMENT_SLOTS)
    mask[:, :3] = 1.0
    logits, values = net(torch.randn(2, ECONOMIC_SIZE), mask)
    assert logits.shape == (2, INVESTMENT_SLOTS) and values.shape == (2,)
    assert bool((logits[:, 3:] == MASKED).all()) and bool((logits[:, :3] > MASKED).all())


def test_a_decision_followed_within_its_own_period_is_not_discounted():
    """Nought periods between two decisions is no game time between them, so what the second earns reaches the first whole."""
    for periods, expected in ((0, 2.0), (1, 1.5)):
        rollout = Rollout(discount=0.5, trace=1.0)
        rollout.add("a", Step(state=[0.0], action=0, mask=[1.0], reward=1.0, value=0.0, periods=periods))
        rollout.add("a", Step(state=[0.0], action=0, mask=[1.0], reward=1.0, value=0.0, periods=1, done=True))
        first = rollout.drain()[0]
        assert abs(first.ret - expected) < 1e-9, periods


def test_a_learnt_economy_files_every_investment_and_pays_each_period_to_the_last_of_it():
    """Two periods of the judge answering: a builder then stopping, a tank then stopping. The builder is followed within its period and filed at nought periods; stopping stands until the next period and is paid that period's figure."""
    from dataclasses import replace

    from rwintel.control.policy.options import Options
    from rwintel.learn.layers import ECONOMY_KEY, LearntEconomy
    from rwintel.learn.reward import CLOSED, EconomicTerms

    rollout = Rollout()
    terms = EconomicTerms(discount=0.99, shaping=SHARE, value_flow=0.05, ground_flow=0.02)
    layer = LearntEconomy(_EconomySession(), _ECONOMY_CATALOGUE, None, rollout, instance=0,
                          options=replace(Options(), choose=False), terms=terms)
    first = layer.decide(_economy_view(30000), _economy_orders(), [])
    assert [_ECONOMY_TYPES[p.type_index].lookup for p in first] == ["builder"]
    second = layer.decide(_economy_view(32000), _economy_orders(), [])
    assert [_ECONOMY_TYPES[p.type_index].lookup for p in second] == ["tank"]
    assert layer.decisions == 4

    steps = rollout.live[(0, ECONOMY_KEY)].steps
    assert [step.periods for step in steps] == [0, 1, 0]
    assert steps[0].reward == 0.0 and steps[1].action == 0
    # The board did not move: everything on it is ours, so the flows are paid whole and the shaping is only the discount's share of the potential.
    assert abs(steps[1].reward - (0.05 + 0.02 - (1.0 - 0.99) * terms.share_weight)) < 1e-9
    assert all(step.soft and abs(sum(step.soft) - 1.0) < 1e-9 for step in steps)

    layer.close(score=0.5)
    drained = rollout.drain()
    assert len(drained) == 4 and not rollout.live
    last = drained[-1]
    assert last.done and int(last.signals[-1][0]) == CLOSED
    assert abs(terms.reward(last.signals) - last.reward) < 1e-12


def test_a_match_ended_with_a_score_ends_the_trajectories_and_one_stopped_cuts_them():
    """A match that ended, decided or cut off by the clock, pays its score once to the decisions standing at its end and ends their trajectories there; a match stopped or lost to the connection has no score, so they are cut and bootstrapped."""
    from rwintel.learn.layers import LearntOperations
    from rwintel.learn.reward import CLOSED

    for score in (0.75, None):
        rollout = Rollout()
        layer = LearntOperations(None, _CATALOGUE, _Counting(), rollout, instance=0)
        squad = _squad(contract=False, losses=0.0)
        for index in range(4):
            t = 30000 + 2000 * index
            layer.decide(_ops_board(t), _Orders(), [squad], [], t)
        layer.close(score=score)
        assert not rollout.live and len(rollout.done) == 1
        trajectory = rollout.done[0]
        ends = [row for step in trajectory.steps for row in step.signals if int(row[0]) == CLOSED]
        if score is None:
            assert not trajectory.finished and not ends
        else:
            assert trajectory.finished and len(ends) == 1 and ends[0][10] == score
            # The board never moved and the expectation of it is all the shaping holds, so the decision is paid the score less that expectation.
            held = layer.reward.terms.potential(*(ends[0][1], ends[0][5], ends[0][7]))
            assert abs(trajectory.steps[-1].reward - (score - held)) < 1e-12


def test_the_opening_prediction_is_the_worth_edge_and_coefficients_interpolate_between_knots():
    from rwintel.learn.predictor import OPENING, Predictor, load, save

    assert abs(OPENING.potential(123.0, 0.8, 0.1) - 0.6) < 1e-12
    assert OPENING.potential(5.0, 1.0, 0.0) == 1.0
    two = Predictor(knots=(0.0, 100.0), value=(1.0, 0.0), ground=(0.0, 1.0))
    assert two.coefficients(50.0) == (0.5, 0.5)
    assert two.coefficients(500.0) == two.coefficients(-1.0) == (0.0, 1.0)
    assert two.potential(80.0, 1.0, 1.0) == 1.0
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "predictor.json")
        save(two, path)
        assert load(path) == two
    for broken in (dict(knots=(), value=(), ground=()), dict(knots=(0.0, 0.0), value=(1.0, 1.0), ground=(0.0, 0.0)),
                   dict(knots=(0.0,), value=(1.0, 2.0), ground=(0.0,))):
        with pytest.raises(ValueError):
            Predictor(**broken)


def test_a_fit_recovers_the_coefficients_the_scores_were_made_with():
    """Boards drawn at every knot with scores made from known coefficients and a little noise give those coefficients back, and the held-out agreement says the scores are predictable; a knot with too few boards keeps the opening coefficients."""
    from rwintel.learn.predictor import OPENING, Samples, fit

    draw = random.Random(7)
    knots = (0.0, 300.0, 600.0, 900.0)
    truth = {0.0: (1.0, 0.0), 300.0: (0.7, 0.3), 600.0: (0.4, 0.5)}
    samples = Samples()
    for match in range(300):
        knot = knots[match % 3]
        a, b = truth[knot]
        for _ in range(4):
            value, ground = draw.random(), draw.random()
            samples.remaining.append(knot + draw.uniform(-20.0, 20.0) if knot else draw.uniform(0.0, 20.0))
            samples.value.append(value)
            samples.ground.append(ground)
            samples.score.append(a * (2 * value - 1) + b * (2 * ground - 1) + draw.gauss(0.0, 0.02))
            samples.match.append(match)
            samples.group.append("run|map")
    predictor, reports = fit(samples, knots=knots)
    for index, knot in enumerate(knots[:3]):
        assert abs(predictor.value[index] - truth[knot][0]) < 0.05 and abs(predictor.ground[index] - truth[knot][1]) < 0.05
        assert reports[index].r2 > 0.9
    assert (predictor.value[3], predictor.ground[3]) == (OPENING.value[0], OPENING.ground[0]) and reports[3].r2 is None


def test_the_named_rewards_set_the_terms_and_a_figure_given_is_laid_over_them():
    """The default is the expected score undiscounted with the match score at the end; the flows are the score-aligned flows it replaced, each layer at the discount and trace it was run at; and the strategic orders and a squad's own exchange are refused for the economy, which has neither."""
    from rwintel.learn.__main__ import build_parser, match_terms
    from rwintel.learn.reward import PREDICTED

    def terms(*words):
        arguments = build_parser().parse_args(["collect", *words])
        return match_terms(arguments, arguments.layer)

    default, trace = terms("--layer", "economy")
    assert (default.discount, trace, default.shaping, default.terminal_weight, default.value_flow) == (1.0, 0.95, PREDICTED, 1.0, 0.0)
    flows, trace = terms("--layer", "operations", "--reward", "flows")
    assert (flows.discount, trace, flows.shaping, flows.value_flow, flows.local_exchange, flows.achievement_weight) == \
        (0.95, 0.9, SHARE, 0.05, 0.1, 0.0)
    economy, trace = terms("--layer", "economy", "--reward", "flows")
    assert (economy.discount, trace, economy.value_flow, economy.ground_flow) == (0.99, 0.95, 0.05, 0.02)
    laid, trace = terms("--layer", "operations", "--reward", "flows", "--discount", "0.97", "--trace", "0.5")
    assert (laid.discount, trace, laid.local_exchange) == (0.97, 0.5, 0.1)
    with pytest.raises(SystemExit):
        terms("--layer", "economy", "--local-exchange", "0.1")


def test_checkpoints_overwrite_the_saved_parameters_and_snapshots_keep_each_one():
    """A checkpoint goes over --save so that a stopped run keeps its latest parameters, and with --snapshots each one is kept under its own name as well, so that a policy a run passed through can be played again afterwards."""
    import threading
    from types import SimpleNamespace

    from rwintel.learn.__main__ import _checkpointer, build_parser

    net = TacticalNet()
    optimiser = SimpleNamespace(lock=threading.Lock())
    with tempfile.TemporaryDirectory() as folder:
        save, snapshots = os.path.join(folder, "tactics-rl.pt"), os.path.join(folder, "snap")
        arguments = build_parser().parse_args(["tactics", "--save", save, "--checkpoint-every", "2",
                                               "--snapshots", snapshots])
        on_update = _checkpointer(net, optimiser, arguments)
        for updates in range(1, 6):
            with torch.no_grad():
                next(net.parameters()).fill_(float(updates))
            on_update(SimpleNamespace(updates=updates))
        assert sorted(os.listdir(snapshots)) == ["tactics-rl-u00002.pt", "tactics-rl-u00004.pt"]
        from rwintel.learn import models

        kept = models.read(os.path.join(snapshots, "tactics-rl-u00002.pt"))
        assert float(next(iter(kept["state"].values())).flatten()[0]) == 2.0 and kept["version"] == 2
        latest = models.read(save)
        assert float(next(iter(latest["state"].values())).flatten()[0]) == 4.0 and latest["version"] == 4
        bare = build_parser().parse_args(["tactics", "--checkpoint-every", "1", "--snapshots", snapshots])
        _checkpointer(net, optimiser, bare)(SimpleNamespace(updates=7))
        assert os.path.exists(os.path.join(snapshots, "tactics-u00007.pt"))
    assert _checkpointer(net, optimiser, build_parser().parse_args(["tactics", "--checkpoint-every", "2"])) is None


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
