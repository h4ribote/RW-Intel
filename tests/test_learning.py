"""What the learning side promises, held to.

Three things are pinned here and they are the three that fail silently rather than loudly. The encodings, because a feature vector that has quietly shifted by one, or that carries a raw credit total among ratios, produces a policy that trains and is worthless on the next map rather than one that crashes. The rewards, because potential-based shaping is only safe if it telescopes, and a shaping term that does not is a term that changes which policy is optimal while looking exactly like one that does not. And the buffers, because an errand that was cut off when the match ended is not an errand that failed, and scoring the two the same way teaches a policy to end matches.

A fourth was added once a run of the arena had been measured rather than assumed: what a fight was worth, and where one errand stops and the next begins. A score that is not exactly antisymmetric makes self-play average to something other than nought, and then no comparison drawn against a baseline means anything, because the baseline is measuring the arena's own left-right lean instead of the policy. A trajectory that is not closed when the fight it belongs to is called runs on into the next fight built on the same squad number, and then the advantage earned in one fight flows backwards into decisions taken in another. Both produce a policy that trains perfectly smoothly and has learnt the wrong thing.

Last, the two ways a run is started from something rather than from noise: a fit to what the handwritten layer already does, and a few updates that fit the critic while the policy is held still. Those three need the tensor library, which everything above them deliberately does not.

Nothing here launches a game. The arena is exercised only where it can be: the mirrored view, which is the whole of how one process drives both sides of a fight, and the arithmetic that turns a finished fight into a score.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwintel.control.policy.catalogue import Catalogue
from rwintel.control.policy.contracts import (
    ALLOCATION,
    Doctrine,
    FrontReport,
    OperationsOrders,
    Posture,
    SquadRecord,
    TaskContract,
)
from rwintel.control.policy.operations import Operations
from rwintel.control.policy.strategy import (
    LOSS_ALLOWANCE_FLOOR,
    LOSS_ALLOWANCE_SHARE,
    OFFENSIVE,
    Strategy,
)
from rwintel.control.policy.view import build as build_view, rehome
from rwintel.control.session import UnitType
from rwintel.learn.arena import (
    BY_HEALTH,
    BY_KILLS,
    SCORES,
    STRENGTH_SLOPE_HEALTH,
    STRENGTH_SLOPE_KILLS,
    Engagement,
)
from rwintel.learn.deciders import Choice, NetworkTactics, evaluate_operational
from rwintel.learn.encoding import (
    GLOBAL_SIZE,
    OPERATIONAL_REGIONS,
    OPERATIONAL_TASKS,
    OPERATIONAL_FEATURES,
    OPERATIONAL_RECIPE,
    OPERATIONAL_SIZE,
    REGION_FEATURES,
    REGION_SIZE,
    SQUAD_FEATURES,
    STRATEGIC_FEATURES,
    STRATEGIC_RECIPE,
    STRATEGIC_SIZE,
    TACTICAL_ACTIONS,
    TACTICAL_FEATURES,
    TACTICAL_RECIPE,
    TACTICAL_SIZE,
    operational_state,
    operational_slots,
    region_mask,
    squad_mask,
    squad_slots,
    strategic_state,
    tactical_state,
    task_mask,
)
from rwintel.learn.imitation import (Sample, TeacherMismatch, feature_recipe, fit, read_teacher,
                                     rule_recipe)
from rwintel.learn.frozen import frozen_layers
from rwintel.learn.layers import LearntOperations, LearntStrategy, LearntTactics
from rwintel.learn.net import (
    AVOWAL_KEY,
    ENCODING_KEY,
    EncodingRefused,
    OperationalNet,
    StrategicNet,
    TacticalNet,
    _bytes,
    avowed,
    encoding_avowal,
    encoding_features,
    encoding_recipe,
    encoding_stamp,
    one_hot_slot,
)
from rwintel.learn.ops_arena import SCRIPT_TACTICS, OpsArena
from rwintel.learn.ops_run import arms_of, frozen_tactics, load_arms
from rwintel.learn import policy as eval_arms_policy
from rwintel.learn.policy import OPERATIONAL, STRATEGIC, TACTICAL, LearningPolicy
from rwintel.learn.reward import (
    COMPLETE_REWARD,
    DISCOUNT,
    EXCHANGE_PRIOR,
    EXCHANGE_WEIGHT,
    HOLDING_WEIGHT,
    LOSING_REWARD,
    OperationalReward,
    Outcome,
    SPENDING_WEIGHT,
    StrategicReward,
    TacticalReward,
    WIPED_REWARD,
)
from rwintel.learn.rollout import FIGHT_DISCOUNT, FIGHT_TRACE, Rollout, Step, normalise
from rwintel.learn.train import Optimiser, Trainer
from rwintel.control.policy.tactics import Tactics, _Track
from rwintel.eval import arms as eval_arms
from rwintel.eval.sampling import Summary
from rwintel.eval.scoring import score
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
             movement="LAND", range=130.0, hits_air=False, hits_land=True),
    UnitType(index=1, name="artillery", lookup="artillery", price=700, tech=1, building=False,
             builder=False, movement="LAND", range=320.0, hits_air=False, hits_land=True),
    UnitType(index=2, name="builder", lookup="builder", price=500, tech=1, building=False,
             builder=True, movement="LAND"),
]


class _Catalogue(Catalogue):
    """The type table without the producer links, which are read off the game's asset tree and are not what any of this is about."""

    def __init__(self, types):
        self.types = list(types)
        self.roles = {kind.index: self._role(kind) for kind in self.types}
        self.built_from = {}
        self.declares_maker = {}

    @staticmethod
    def _role(kind):
        from rwintel.control.policy.catalogue import role_of

        return role_of(kind)


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
    """The mask runs over the slots the state is written in, which are the regions in order out from this side's own home. So the live slots are the first however many there are, whatever the map numbered them, and a map with two regions offers slots nought and one however far apart their numbers are."""
    board = _view([], [_region(0, distance=900.0), _region(3, distance=100.0)])
    assert region_mask(board)[:4] == [1.0, 1.0, 0.0, 0.0]
    # And the slot a policy would be answering with is the region that distance orders there, not the region the map numbered so.
    assert [region.id for region in operational_slots(board)] == [3, 0]
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
    """The condition under which shaping stays harmless, now that the discount is an argument rather than a constant: the term has to telescope at exactly the figure the returns are discounted at. Two numbers that were one constant can drift apart, and a pair that differ leave a residue which depends on where the errand went — which is the single dependence potential shaping is chosen to rule out.

    Telescoping means that the payments of an errand, discounted at that same figure and summed, come to the terminal discounted back to the first paid decision less the potential the errand opened on, and to nothing else at all: no board the errand passed through survives the sum. Pinned at both figures actually in use — the hundredth off that an errand which is a fragment of a match takes, and nothing at all, which is what a constructed fight is discounted at because it is one whole finite episode with a real terminal. At a discount of one the property reads as the plain sum of the payments being the terminal less the opening potential.
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
            # The board moves under the errand on every count the potential is made of — the ground, the allowance and the exchange — so that a term which failed to telescope would leave a residue rather than a nought.
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

    That convention is the whole of why shaping is safe: the terms over an errand have to telescope to the difference between where it started and nothing, and a last term paid against the real potential of the board the errand ended on leaves a residue proportional to it instead. Taking the ground scores near the top of the potential, so paying that residue would make a completion worth well over the terminal it is defined to be worth, and worth a different amount according to how much of the budget went — which is a reward that depends on the ending, which is what shaping is chosen not to be.
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


def test_the_operational_layer_is_paid_for_the_ground_its_orders_named():
    """The dense term is per-squad and region-specific: a squad is paid the priority-weighted domination of the one region its contract named, so the reward moves when that region does. That this was a single global figure written into every squad's step was what made the operational gradient dead."""
    class _Orders:
        posture = 0
        priorities = {1: 1.0}
        offensive = True
        loss_allowance = 2000.0

    squad = _squad(id=0)  # its contract names region 1
    reward = OperationalReward()
    losing = _view([], [_region(1, ours=100.0, theirs=900.0)])
    winning = _view([], [_region(1, ours=900.0, theirs=100.0)])
    reward.step(squad, losing, _Orders())  # opens the mission on the losing board
    assert reward.step(squad, winning, _Orders()).reward > 0.0  # the region swung our way

    reward.reset()
    reward.step(squad, winning, _Orders())
    assert reward.step(squad, losing, _Orders()).reward < 0.0


def test_two_operational_squads_on_different_regions_are_paid_differently():
    """The whole point of making the operational reward per-squad: a global figure paid every squad the same number, so the advantage barely depended on which region a squad was sent to. Now, on one board, a squad contracted to a region going our way and one contracted to a region going theirs are paid opposite signs."""
    class _Orders:
        posture = 0
        priorities = {1: 1.0, 2: 1.0}
        offensive = True
        loss_allowance = 2000.0

    ours = _squad(id=0)  # contract names region 1
    theirs = _squad(id=1)
    theirs.contract = TaskContract(squad=1, task=Task.ATTACK, target_region=2,
                                   stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                   deadline_ms=90000, issued_at_ms=20000)
    reward = OperationalReward()
    even = _view([], [_region(1, ours=500.0, theirs=500.0), _region(2, ours=500.0, theirs=500.0)])
    swung = _view([], [_region(1, ours=900.0, theirs=100.0), _region(2, ours=100.0, theirs=900.0)])
    reward.step(ours, even, _Orders())
    reward.step(theirs, even, _Orders())
    held_ours = reward.step(ours, swung, _Orders()).reward     # region 1 went our way
    held_theirs = reward.step(theirs, swung, _Orders()).reward  # region 2 went theirs
    assert held_ours > 0.0 > held_theirs


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


def test_decisions_about_a_squad_somebody_interfered_with_are_not_learnt_from():
    """The design's rule: praising a policy for what a person's squad achieved, or blaming it for what a person's squad lost, teaches the wrong thing. Marking happens late because which squads were interfered with is not known when the decision was taken.

    Scoped to the instance the interference happened on. One buffer serves every instance of a run and each keys its trajectories by its own (instance, squad); the squad-number pool is identical on every instance, so tainting by the bare number would drop every other instance's clean decisions about the same number. Here instance 0's squad 1 is interfered with, and instance 1's squad 1 — a different trajectory that happens to reuse the number — must survive.
    """
    rollout = Rollout()
    for index in range(3):
        rollout.add((0, 0), Step(state=[0.0], action=0, mask=[1.0], squad=0, at_ms=0, done=index == 2))
        rollout.add((0, 1), Step(state=[0.0], action=0, mask=[1.0], squad=1, at_ms=1, done=index == 2))
        rollout.add((1, 1), Step(state=[0.0], action=0, mask=[1.0], squad=1, at_ms=2, done=index == 2))
    rollout.taint(0, [1])
    kept = rollout.drain()
    # Tagged by at_ms: 0 is instance 0's clean squad 0, 1 is instance 0's interfered squad 1, 2 is instance 1's clean squad 1.
    assert sorted(step.at_ms for step in kept) == [0, 0, 0, 2, 2, 2]
    assert all(step.at_ms != 1 for step in kept)


def test_a_trajectory_that_is_still_running_is_left_alone_by_a_drain():
    rollout = Rollout()
    rollout.add("a", Step(state=[0.0], action=0, mask=[1.0]))
    assert rollout.drain() == []
    assert len(rollout) == 1


def test_a_finished_trajectory_is_held_from_the_gated_drain_until_its_episode_is_sealed():
    """The trainer takes only sealed trajectories, and a trajectory is sealed only when its episode closes. Between a mid-episode finish — a squad wiped, a contract renewed, a squad off the board — and the close, the finished trajectory sits in the buffer but must not be drained: the close is where the episode's interference is marked, and a decision drained before then would enter the update before it could be told it had been interfered with."""
    rollout = Rollout()
    for index in range(3):
        rollout.add((0, 1), Step(state=[0.0], action=0, mask=[1.0], squad=1, done=index == 2))
    # Finished and in the buffer, but the episode has not closed.
    assert len(rollout.done) == 1 and not rollout.done[0].sealed
    assert rollout.drain(sealed_only=True) == []
    assert len(rollout.done) == 1  # still held
    rollout.seal(0)
    drained = rollout.drain(sealed_only=True)
    assert len(drained) == 3 and not rollout.done


def test_a_drain_empties_the_finished_list_rather_than_replacing_it():
    """The instance threads file finished trajectories into this list while the trainer drains it on its own clock, so a drain that builds a replacement list and binds it over the attribute drops whatever was filed after the replacement was built. What would be dropped is a finished trajectory — decisions collected, paid, and gone from the run with nothing in any census to say so.

    Pinned as identity rather than as a race, because a race is exactly what a test cannot pin: the window is a few bytecodes wide under one interpreter lock, and a loop filing eight thousand trajectories against a draining thread lost none. What can be pinned is the property that makes the window impossible, which is that the list a filer holds is the list the drain empties.
    """
    rollout = Rollout()
    held = rollout.done
    for key in (1, 2):
        rollout.add((0, key), Step(state=[0.0], action=0, mask=[1.0], squad=key, done=True))
    rollout.seal(0)
    assert len(rollout.drain(sealed_only=True)) == 2
    assert rollout.done is held, "the gated drain replaced the list its filers are appending to"

    rollout.add((0, 3), Step(state=[0.0], action=0, mask=[1.0], squad=3, done=True))
    assert len(rollout.drain()) == 1
    assert rollout.done is held, "the ungated drain replaced the list its filers are appending to"


def test_an_interfered_decision_that_finished_mid_episode_is_tainted_before_it_can_be_drained():
    """The race the seal closes: a squad the intruder touched finishes its errand in the middle of an episode, so its trajectory joins the finished set several periods before the episode closes and the tainting runs, and the trainer thread drains on its own clock. Held unsealed, the trajectory cannot be drained in that window; when the episode closes the taint marks it and the seal releases it, and the gated drain then drops it as interfered with, so the decision never reaches an update. Squad 2, untouched, is the control that proves the gate releases what it should."""
    rollout = Rollout()
    for index in range(3):
        rollout.add((0, 1), Step(state=[0.0], action=0, mask=[1.0], squad=1, done=index == 2))
        rollout.add((0, 2), Step(state=[0.0], action=0, mask=[1.0], squad=2, done=index == 2))
    # Before the episode closes nothing is drainable, so the touched squad cannot leak.
    assert rollout.drain(sealed_only=True) == []
    # The episode closes: squad 1 was interfered with, and only then is everything sealed.
    rollout.taint(0, [1])
    rollout.seal(0)
    drained = rollout.drain(sealed_only=True)
    assert {step.squad for step in drained} == {2}


def test_the_seal_releases_only_the_instance_whose_episode_closed():
    """One buffer serves every instance. An episode closing on one instance says nothing about the fight another is in the middle of, so the seal is scoped to the instance that closed, exactly as the taint and the cut are. Here instance 0 closes and instance 1 is still fighting; only instance 0's finished trajectory becomes drainable."""
    rollout = Rollout()
    rollout.add((0, 1), Step(state=[0.0], action=0, mask=[1.0], squad=1, at_ms=0, done=True))
    rollout.add((1, 1), Step(state=[0.0], action=0, mask=[1.0], squad=1, at_ms=9, done=True))
    rollout.seal(0)
    drained = rollout.drain(sealed_only=True)
    assert [step.at_ms for step in drained] == [0]
    assert len(rollout.done) == 1 and rollout.done[0].key == (1, 1)


def test_a_layers_flush_leaves_its_work_unsealed_and_only_its_close_seals_it():
    """A layer's close is two things: flush, which ends the episode's errands, and seal, which releases them to the trainer. They are separate because the operational chain marks the episode's interference in between — flush, then taint, then seal — and a flush that sealed would let a decision the intruder touched be drained before it was tainted. So flush alone must leave the work unsealed and undrainable, and close (here with no intruder above it) must seal it."""
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    squad = _squad()
    layer.decide(_skirmish(), [squad], 21000)
    layer.finish(squad, 0.75, "called")
    layer.flush()
    assert rollout.done and not rollout.done[0].sealed
    assert rollout.drain(sealed_only=True) == []
    layer.close()
    assert len(rollout.drain(sealed_only=True)) == 1


def test_a_decision_in_a_cut_errand_is_anchored_by_the_critic_alone_and_its_last_one_by_nothing():
    """The signature to watch for in any layer, and the one worth having a test for whatever else is being measured: what a batch of decisions that nothing terminal ever reached actually teaches.

    An errand that is cut — because a new contract replaced it, or the squad left the board, or the episode closed around it — is bootstrapped from its last decision's own value estimate. At a discount and a trace of one that makes the last decision's advantage exactly its own reward, and a period that pays nothing makes it identically nought; every earlier decision of that errand comes out at the sum of the rewards after it plus the difference between two of the critic's own estimates, which contains no terminal at all. The critic's own regression target on those steps is its own later estimate, so a run of them fits nothing and drifts, and the mean return drifts with it.

    Both halves are pinned here because the two are easy to confuse and the difference decides what a run's numbers mean. With a critic that answers the same everywhere — a fresh value head, or one replayed at nought — every decision of every cut errand is exactly nought and normalising turns the whole mass into one identical nonzero number pushed onto whatever the policy happened to draw. With a critic that has fitted anything at all, only the last decision of each cut errand is exactly nought and the rest carry real differences. A count of exact noughts is therefore a measurement of the critic as much as of the buffer, and quoting one without the other says more than it knows.
    """
    def batch(values):
        rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
        for squad in range(4):
            for value in values:
                rollout.add((0, squad), Step(state=[0.0], action=0, mask=[1.0], value=value,
                                             reward=0.0, squad=squad))
            # Every period of these errands paid nothing, which is what a period that renews a contract pays.
            rollout.cut((0, squad), reason="renewed")
        # And one errand that ran to a terminal and was paid there, which is what every trajectory of a contest that reads its own scored ground now looks like.
        rollout.add((0, 9), Step(state=[0.0], action=0, mask=[1.0], value=values[0], reward=0.5,
                                 done=True, squad=9))
        return rollout, rollout.drain()

    flat, steps = batch([0.3, 0.3, 0.3, 0.3, 0.3])
    cut = [step for step in steps if step.squad != 9]
    assert len(cut) == 20 and all(step.advantage == 0.0 for step in cut)
    assert flat.census.zero_advantage == 20 and flat.census.distinct == 2
    assert flat.census.paid_steps == 1 and flat.census.cut == {"renewed": 4}
    normalise(steps)
    spread = {step.advantage for step in cut}
    assert len(spread) == 1 and abs(spread.pop()) > 0.1, (
        "a batch of identical noughts comes out of normalisation as one identical nonzero push on whatever was drawn")

    # The same buffer under a critic that has fitted something: only the last decision of each cut errand is exactly nought, and every other one carries what the critic's estimate moved between there and the end of what was observed.
    values = [0.1, 0.25, 0.4, 0.55, 0.7]
    fitted, steps = batch(values)
    cut = [step for step in steps if step.squad != 9]
    assert fitted.census.zero_advantage == 4
    assert fitted.census.distinct == len(values) + 1
    for index, step in enumerate(cut):
        assert abs(step.advantage - (values[-1] - values[index % len(values)])) < 1e-9


def test_a_drained_batch_says_how_much_of_itself_a_payment_ever_reached():
    """The census exists because none of the figures an update reports can say this. A batch mostly made of errands nothing ever paid reports a policy loss, a value loss, an entropy and a mean return exactly like a batch that was paid throughout, and the mean return of the unpaid part is the critic's own estimate at the point observation stopped — so a run of them looks like a return climbing.

    Counted over the batch as the optimiser will see it, so decisions dropped for having been interfered with are not in it: what is wanted is the make-up of the gradient, not of the buffer.
    """
    rollout = Rollout()
    for index in range(3):
        rollout.add((0, 1), Step(state=[0.0], action=0, mask=[1.0], reward=0.1, squad=1,
                                 done=index == 2))
    for index in range(4):
        rollout.add((0, 2), Step(state=[0.0], action=0, mask=[1.0], reward=0.0, squad=2))
    rollout.cut((0, 2), reason="renewed")
    for index in range(2):
        rollout.add((0, 3), Step(state=[0.0], action=0, mask=[1.0], reward=0.0, squad=3))
    rollout.cut((0, 3), reason="left")
    rollout.taint(0, [3])

    steps = rollout.drain()
    census = rollout.census
    assert len(steps) == 7, "the tainted decisions are dropped from the batch"
    assert census.steps == 7 and census.paid_steps == 3
    assert census.finished == 1 and census.cut == {"renewed": 1, "left": 1}
    assert census.as_dict()["cut"] == {"renewed": 1, "left": 1}


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


#: The point a constructed board is reflected about, as the arenas reflect their own draws about the mean of the places an engagement may be built on. Any centre does; this one is round.
MIRROR_CENTRE = (500.0, 500.0)


def _reflected(x, y):
    """A point taken through the same half turn the arenas take their mirror force through: both coordinates negated about the centre."""
    return (2 * MIRROR_CENTRE[0] - x, 2 * MIRROR_CENTRE[1] - y)


#: Where this side is staged from on the constructed board, which is the anchor its bearing is read against. Behind the squad and off the line it is marching on, so that a bearing measured from it has a lateral part on both sides and a test of the sign is a test of something.
STAGE = (120.0, 180.0)


def _mirrored_board():
    """One board laid out as a point reflection about a centre, read once from each side, which is the situation the constructed arenas put a layer in.

    Four groups of units and two regions, in exactly reflected pairs: our squad and its reflection, which is the enemy's squad; the enemy garrison our squad is walking into and its reflection, which is our own garrison standing where the enemy's squad is walking; the region our squad was sent to and its reflection, which is the region the enemy's squad was sent to. Every reflected unit is of the SAME TYPE as the unit it reflects, since what makes the two sides exchangeable is one force mirrored and not two different forces facing each other, and a difference of type would show up as a difference of worth, of role and of weapon reach in a dozen slots that have nothing to do with the geometry.

    Deliberately chiral: the squad's members, the enemies shooting at it and the region it was sent to are nowhere near collinear, so a bearing measured across the fight has a large lateral part. On a collinear board that part is nought on both sides and a test of it would pass whichever way round its sign was written.

    The fog record is set the way a match sets it and NOT symmetrically: hostiles are standing in the region our squad is attacking, so the enemy has been seen there, and nothing has been seen where our own garrison stands. Setting it recent on both members of a pair is the one setting under which the region row's contact flag cannot disagree between the sides, which would hide a defect this board exists to state.

    The two regions are likewise given DIFFERENT distances from home, which is what a real map gives them: the wire measures every region's distance from this process's own base, so a region and its reflection are near and far from it respectively. Giving a mirrored pair the same distance is the one setting under which the row's distance cannot disagree between the sides either, and it would hide the second of the two defects below in the same way.

    Each side is then anchored where it came from, as both arenas anchor theirs: a staging point behind our squad, and its reflection behind the enemy's. That step is the caller's and not the builder's — the builder can only offer the region nearest this process's base, which is ONE point for the whole board, and reading the bearing from one point on a mirrored board is the frame all over again. The anchors are points rather than regions because a pair of exact reflections can sit nearest to two regions that are not a pair, which is what a map with two regions in it and no symmetry gives here.
    """
    members = [(1, 200.0, 300.0, 0), (2, 232.0, 312.0, 0), (3, 214.0, 348.0, 1)]
    threats = [(9, 300.0, 480.0, 0), (10, 340.0, 520.0, 1)]

    units = []
    for unit_id, x, y, kind in members:
        units.append(_unit(unit_id, x, y, type_index=kind))
        units.append(_unit(unit_id + 100, *_reflected(x, y), type_index=kind, hostile=1))
    for unit_id, x, y, kind in threats:
        units.append(_unit(unit_id, x, y, type_index=kind, hostile=1))
        units.append(_unit(unit_id + 100, *_reflected(x, y), type_index=kind))

    target = (760.0, 250.0)
    ours_region = _region(1, target[0], target[1], ours=200.0, theirs=900.0, distance=640.0)
    theirs_region = _region(2, *_reflected(*target), ours=900.0, theirs=200.0, distance=1480.0)
    theirs_region.held_by_us, theirs_region.held_by_enemy = 0, 1
    ours_region.enemy_seen_at_ms, theirs_region.enemy_seen_at_ms = 29000, 0

    observation = _observation(units, [ours_region, theirs_region])
    ours = rehome(build_view(observation, _CATALOGUE, None), point=STAGE)
    theirs = rehome(build_view(observation, _CATALOGUE, None, invert=True), point=_reflected(*STAGE))

    centre = (sum(m[1] for m in members) / len(members), sum(m[2] for m in members) / len(members))
    their_centre = _reflected(*centre)
    our_squad = _squad(id=0, members=[m[0] for m in members], x=centre[0], y=centre[1])
    their_squad = _squad(id=1, members=[m[0] + 100 for m in members],
                         x=their_centre[0], y=their_centre[1])
    their_squad.contract.target_region = 2
    return observation, ours, theirs, our_squad, their_squad


def _contacts(view, now=30000):
    """The contact record the operations arena writes for each side as it builds its view: an enemy is in contact where an enemy is standing, read off the force totals that the inversion has already turned over.

    Applied to both sides and not only the mirror. The wire's own record is kept for the seat this process occupies and cannot be turned over — the row has no counterpart field — so a mirrored board is congruent here only once both sides' records are written by one rule.
    """
    from dataclasses import replace

    view.regions = [replace(region, enemy_seen_at_ms=now if region.enemy_value > 0 else 0)
                    for region in view.regions]
    return view


def _squad_fight(view, squad):
    """One squad's fight as the tactical layer cuts it out of the board: the members of the squad that are on it, and what is near enough to be shooting at them. The same rule on both sides, so the cut is congruent whenever the board is."""
    members = [s for s in view.ours if s.unit.id in squad.members]
    return members, view.enemies_near(squad.x, squad.y, 400.0)


def test_the_tactical_state_of_a_mirrored_board_reads_the_same_from_both_sides():
    """The property one process driving both sides of a constructed board depends on, stated at the encoding: a squad and its exact reflection must read as the same fight.

    They face congruent situations at reflected positions, and the inverted view turns the ownership over without touching a coordinate — which is right, because the coordinates on the wire are the world's and not the seat's, and a process seated at the other slot would be handed exactly these. So the congruence has to be in what the features are made of: a feature measured against the map's own axes reads one thing for a squad and the opposite for its reflection, and a network fed the two answers one fight two different ways depending on which way round the board happens to be numbered. Half its training experience then goes on learning the same thing twice in a different frame, and where one process drives both sides the two sides are not exchangeable at all.

    Compared within a tolerance rather than exactly because the reflected coordinates are exact while the threats' centre is accumulated over the same offsets in a different order: the two vectors agree here to 1.1e-16, which is the last bit of a double and not a difference in what was computed. The names are zipped onto the values so that a failure says which feature moved rather than that two lists differ.
    """
    _, ours, theirs, our_squad, their_squad = _mirrored_board()
    our_members, our_threats = _squad_fight(ours, our_squad)
    their_members, their_threats = _squad_fight(theirs, their_squad)
    assert [s.unit.id for s in our_members] == [1, 2, 3] and [s.unit.id for s in our_threats] == [9, 10]
    assert [s.unit.id for s in their_members] == [101, 102, 103]
    assert [s.unit.id for s in their_threats] == [109, 110]

    mine = tactical_state(our_squad, our_members, our_threats, 350.0, 200.0, ours, 30000)
    yours = tactical_state(their_squad, their_members, their_threats, 350.0, 200.0, theirs, 30000)
    apart = [(name, a, b) for name, a, b in zip(TACTICAL_FEATURES, mine, yours) if abs(a - b) > 1e-9]
    assert not apart, "these features read differently from the two sides of one mirrored board: " + ", ".join(
        f"{name} {a:+.6f} against {b:+.6f}" for name, a, b in apart)


def test_a_network_answers_both_sides_of_a_mirrored_board_the_same_way():
    """The same property in the terms that decide whether the arena measures anything: not that the features are tidy, but that the layer answers one situation the same way whichever seat it is in.

    This is what a handwritten ladder gives for nothing and a network does not. The ladder reads distances and strengths and is indifferent to which way round the board is numbered; a network reads whatever it was handed, so a feature carrying the map's frame makes it a different fighter on the two sides of a board that is one force mirrored. Where a trained tactical layer is frozen beneath both sides of the operations arena, that difference is the arena's own lean, and every arm measured on it is measured against a board that is not exchangeable between the sides.
    """
    _, ours, theirs, our_squad, their_squad = _mirrored_board()
    mine = tactical_state(our_squad, *_squad_fight(ours, our_squad), 350.0, 200.0, ours, 30000)
    yours = tactical_state(their_squad, *_squad_fight(theirs, their_squad), 350.0, 200.0, theirs, 30000)

    torch.manual_seed(4321)
    net = TacticalNet()
    with torch.no_grad():
        logits, value = net(torch.tensor([mine, yours], dtype=torch.float32))
    assert torch.allclose(logits[0], logits[1], atol=1e-6), (
        "the same fight read from the two sides is answered differently, by up to %.3e"
        % float((logits[0] - logits[1]).abs().max()))
    assert torch.allclose(value[0], value[1], atol=1e-6), (
        "the same fight read from the two sides is valued differently: %.6f against %.6f"
        % (float(value[0]), float(value[1])))


def test_a_squad_marching_with_nothing_shooting_at_it_still_carries_a_bearing():
    """What the anchor buys back, and the reason it is home and not the fight.

    A bearing read from the centre of what is shooting is undefined for a squad nobody is shooting at, and a squad marching on a contest is exactly that — which is most of what an operational layer sends squads to do. Measured, that cost was a tactical layer sitting 0.0398 below the handwritten ladder where the frame-carrying encoding it replaced had held nought. Home does not come and go with the fighting, so the pair here is read off a board with no threats on it at all and must still say where the errand points.

    The lateral term is checked to be substantially non-nought as well, because a board on which the anchor, the squad and its target happen to be collinear would give a nought there whatever the code did.
    """
    _, ours, _, our_squad, _ = _mirrored_board()
    ahead = TACTICAL_FEATURES.index("target_ahead_of_home")
    abeam = TACTICAL_FEATURES.index("target_abeam_of_home")

    members = [s for s in ours.ours if s.unit.id in our_squad.members]
    alone = tactical_state(our_squad, members, [], 350.0, 200.0, ours, 30000)
    assert abs(alone[ahead]) > 0.1 and abs(alone[abeam]) > 0.1, (
        "a squad with nothing shooting at it reads no direction at all: %+.6f, %+.6f"
        % (alone[ahead], alone[abeam]))
    # And it is the same direction whether or not anything is shooting, since what it is measured from has not moved.
    fought = tactical_state(our_squad, *_squad_fight(ours, our_squad), 350.0, 200.0, ours, 30000)
    assert abs(fought[ahead] - alone[ahead]) < 1e-12 and abs(fought[abeam] - alone[abeam]) < 1e-12

    # Nowhere to have come from is the one case that still reads nought, and it is honest: before anything is built there is no home to measure from.
    ours.home_point = None
    unanchored = tactical_state(our_squad, members, [], 350.0, 200.0, ours, 30000)
    assert (unanchored[ahead], unanchored[abeam]) == (0.0, 0.0)


def test_an_anchor_the_two_sides_share_puts_the_frame_straight_back():
    """The requirement the anchor lays on whoever builds the views, stated as a failure rather than as a comment.

    The bearing is invariant because the anchor reflects along with everything else, which is true of each side's OWN staging point and false of any single point on the board. The view builder can only offer the latter — the wire measures every distance from this process's base, so an inverted view has nothing in it that says where the other side came from — and an arena that forgets to anchor the mirror side would leave both sides reading from one point. That does not fail loudly anywhere else: the features stay finite, the run trains, and the only symptom is the arena leaning toward the side the process plays.
    """
    _, ours, theirs, our_squad, their_squad = _mirrored_board()
    theirs.home_point = ours.home_point  # the mistake: one anchor for a board with two sides

    mine = tactical_state(our_squad, *_squad_fight(ours, our_squad), 350.0, 200.0, ours, 30000)
    yours = tactical_state(their_squad, *_squad_fight(theirs, their_squad), 350.0, 200.0, theirs, 30000)
    apart = [name for name, a, b in zip(TACTICAL_FEATURES, mine, yours) if abs(a - b) > 1e-9]
    assert apart == ["target_ahead_of_home", "target_abeam_of_home"], (
        "sharing one anchor between the sides should move the bearing and nothing else, and it moved %s" % apart)


def test_the_engagement_arena_anchors_each_side_where_it_was_put_down():
    """The wiring the property above depends on, at the arena that trains the tactical layer.

    A constructed fight has no base: both sides are put down beside a site, and the two staging points are its exact reflections through it. So they are what each side's view is anchored to, and this is the test that the arena hands each layer its own rather than leaving both on the builder's fallback, which is the region nearest THIS process's base and therefore one point for the whole board.
    """
    from rwintel.learn.arena import Engagement, OURS_FIRST

    observation = _observation(units=[_unit(1, 100.0, 100.0), _unit(2, 120.0, 100.0),
                                      _unit(3, 140.0, 100.0), _unit(9, 300.0, 120.0, hostile=1)],
                               regions=[_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])
    view = build_view(observation, _CATALOGUE, None)

    site = (220.0, 110.0)
    our_place, their_place = (100.0, 60.0), (2 * site[0] - 100.0, 2 * site[1] - 60.0)
    arena = _arena_at(OURS_FIRST, [])
    arena.engagement = Engagement(index=0, site=site, our_place=our_place, their_place=their_place)
    # The fight has just moved, so that this period is fought rather than called off as stalled before either layer is asked.
    arena._changed_ms = 21000
    view.home_point = our_place  # as `decide` sets it, before the period reaches the fighting
    arena._fight(observation, view, Action(), 21000)

    assert arena.tactics.board.home_point == our_place
    assert arena.opponent.board.home_point == their_place
    assert arena.tactics.board.home_point != arena.opponent.board.home_point
    # The pair is the half turn about the site, which is what makes the two anchors a mirror rather than merely two points.
    assert their_place == (2 * site[0] - our_place[0], 2 * site[1] - our_place[1])


def test_the_board_says_where_the_squad_being_decided_is_already_going():
    """The one region feature that is about the decision rather than about the board, and why it is there.

    A squad's own row says how long its mission has been running and how much of its budget is gone, and nothing anywhere said WHERE it was going. So a layer could not see the errand it was already on, and the only way it had to keep one was to re-choose the same region from nothing every period. Measured, its errands run about a fifth as long as the handwritten ladder's, and shorten further the longer it is trained while its score falls.

    The flag marks the slot the region sits in rather than the region's own number, which is what keeps it readable from either side of a mirrored board: the slots run outward from each side's own home, so a squad and its mirror mark the same row.
    """
    regions = [_region(1, 0.0, 0.0, distance=100.0), _region(2, 600.0, 0.0, distance=700.0)]
    view = _view([_unit(1, 10.0, 0.0)], regions)
    squad = _squad(members=(1,))

    def row(state, slot):
        start = GLOBAL_SIZE + slot * REGION_SIZE
        return dict(zip(REGION_FEATURES, state[start:start + REGION_SIZE]))

    plain = operational_state(view, None, [squad], 30000)
    assert row(plain, 0)["contracted"] == 0.0 and row(plain, 1)["contracted"] == 0.0, (
        "a state built for no particular errand marks nothing")

    near = operational_state(view, None, [squad], 30000, contracted=1)
    far = operational_state(view, None, [squad], 30000, contracted=2)
    assert row(near, 0)["contracted"] == 1.0 and row(near, 1)["contracted"] == 0.0
    assert row(far, 0)["contracted"] == 0.0 and row(far, 1)["contracted"] == 1.0
    # Nothing else moves: it is one flag on one row, not a re-reading of the board.
    assert sum(1 for a, b in zip(near, far) if a != b) == 2


def test_a_layer_asks_about_each_squad_from_that_squads_own_errand():
    """Two squads deciding on one board are handed two states, differing in the one feature that says where each is already going. It is built per squad because that is what it is; the rest of the board is read once for the period, since it is the same board for every squad."""
    regions = [_region(1, 0.0, 0.0, distance=100.0), _region(2, 600.0, 0.0, distance=700.0)]
    view = _view([_unit(1, 10.0, 0.0), _unit(2, 20.0, 0.0)], regions)
    first, second = _squad(id=0, members=(1,)), _squad(id=1, members=(2,))
    first.contract.target_region, second.contract.target_region = 1, 2

    asked = []

    class _Watching:
        def choose(self, state, slot, region_mask, task_mask):
            asked.append(list(state))
            return Choice(action=0, second=0)

    layer = LearntOperations(None, _CATALOGUE, _Watching())
    orders = OperationsOrders(posture=Posture.ARM, priorities={1: 1.0, 2: 1.0}, offensive=True,
                              loss_allowance=1000.0)
    layer.decide(view, orders, [first, second], [], 30000)
    assert len(asked) == 2, "each squad is asked once"
    assert asked[0] != asked[1], "two squads on one board were handed the same errand"
    differ = [index for index, (a, b) in enumerate(zip(*asked)) if a != b]
    assert len(differ) == 2, "the two states should differ in the contracted flag of two rows and nothing else"


def test_the_operational_teacher_records_the_contract_that_actually_goes_out():
    """What a teacher writes down has to be what the layer did, and for the operational layer it was not.

    The decision is recorded where the choice is made, and that call is the scoring of the board; the rule then overrules the scoring whenever the squad is running a mission that has not stalled, failed, finished or expired. That overruling is what makes the handwritten ladder hold an errand for a hundred-odd decisions instead of re-scoring every period, so it fires on most of them — and a teacher written from the scoring alone teaches a policy that never holds an errand at all.

    Driven here on a board where the two differ by construction: the squad holds a running contract on one region while the scoring prefers the other.
    """
    rollout = Rollout()
    layer = LearntOperations(None, _CATALOGUE, None, rollout=rollout, instance=0)
    # Two regions, the far one worthless and the near one wanted, so the scoring prefers the near one.
    regions = [_region(1, 0.0, 0.0, distance=100.0, theirs=800.0),
               _region(2, 900.0, 0.0, distance=900.0, theirs=0.0)]
    view = _view([_unit(1, 880.0, 0.0), _unit(2, 900.0, 0.0)], regions)
    squad = _squad(id=0, members=(1, 2))
    squad.doctrine = Doctrine.VANGUARD
    squad.status = Status.ACTIVE
    squad.contract.target_region = 2                      # already running an errand on the far region
    orders = OperationsOrders(posture=Posture.ARM, priorities={1: 1.0, 2: 0.0}, offensive=True,
                              loss_allowance=1000.0)

    # The two really do differ on this board, or the test would pass on a policy that never corrected anything.
    scored = Operations(None, _CATALOGUE)._pick(view, orders, squad, None)
    assert scored is not None and scored[1].id != squad.contract.target_region, (
        "the board was built so that the scoring and the running errand name different regions")

    contracts, _ = layer.decide(view, orders, [squad], [], 30000)
    step = layer.pending.get(0)
    assert step is not None, "the teacher recorded nothing for a squad it decided about"
    settled = next((index for index, row in enumerate(operational_slots(view))
                    if row.id == squad.contract.target_region), -1)
    assert step.action == settled, (
        "the teacher wrote down the region the scoring preferred rather than the errand the rule kept")
    assert not contracts, "an unchanged contract is not re-issued, which is the whole of what the rule did here"


def test_a_decision_is_updated_against_the_row_it_was_asked_about():
    """The row of the squad block a decision was asked about is not the squad's own number, and writing down only the number made the optimiser rebuild the one-hot from the wrong thing.

    One process drives both sides of the constructed arena out of one numbering, so a side's squads are some run of numbers that need not begin at nought, and every layer offsets its rows by the first number it was ever handed. The decider is asked with the offset row. The step used to record the raw number, and the trainer built its one-hot from that, so wherever a side's numbering did not begin at nought the network was asked about one row and updated against another. Both are written down now, and each is read by whatever needs it — the raw number by the tainting, which has to find a squad an intruder named, and the row by the optimiser.
    """
    asked = []

    class _Watching:
        def choose(self, state, slot, region_mask, task_mask):
            asked.append(slot)
            return Choice(action=0, second=0)

    rollout = Rollout()
    layer = LearntOperations(None, _CATALOGUE, _Watching(), rollout=rollout, instance=0)
    regions = [_region(1, 0.0, 0.0, distance=100.0), _region(2, 600.0, 0.0, distance=700.0)]
    view = _view([_unit(1, 10.0, 0.0), _unit(2, 20.0, 0.0)], regions)
    # A side whose numbering starts at four, which is what the arena hands its second side.
    first, second = _squad(id=4, members=(1,)), _squad(id=5, members=(2,))
    first.contract.target_region, second.contract.target_region = 1, 2
    orders = OperationsOrders(posture=Posture.ARM, priorities={1: 1.0, 2: 1.0}, offensive=True,
                              loss_allowance=1000.0)
    layer.decide(view, orders, [first, second], [], 30000)

    assert asked == [0, 1], "the decider is asked about the row, offset by where this side's numbering starts"
    assert sorted(step.slot for step in layer.pending.values()) == [0, 1], (
        "the optimiser would rebuild the one-hot from a row the network was never asked about")
    assert sorted(step.squad for step in layer.pending.values()) == [4, 5], (
        "the tainting has to find a squad by the number an intruder names it by")


def test_the_operational_cut_reads_one_mirrored_board_the_same_way_from_either_side():
    """The property the operations arena depends on at the layer it is actually training: one board laid out as a point reflection has to read identically from the two seats, block for block and number for number.

    Three things had to be true for that, and each was false in a different way. The rows were laid out by the MAP's numbering, so congruent ground sat at different offsets and the two sides read different rows for the same place; they are laid out from each side's own home now. The squad rows were laid out by the global squad number, and one process drives both sides out of one numbering, so this side's squads were the first few rows and the other side's the next few; each side is offset by its own first squad now. And `distance` and the contact record were the wire's, measured for the seat this process occupies, so the mirror side read our marches and our fog as its own; the arena rewrites both as it builds the two views, which is what this test applies.

    What is NOT shown here is that the ordering is congruent when two of a side's regions sit at exactly the same distance from its home. The order breaks that tie by the map's number, and the mirror side can number the tied pair the other way round. Their rows are congruent — same distance, and on a mirrored board the same standing — so the state is unchanged by the swap; what can differ is which of two congruent places a chosen slot names.
    """
    _, ours, theirs, our_squad, their_squad = _mirrored_board()

    def state(view, squad, base):
        return operational_state(view, None, [squad], 30000, base=base)

    def rows(state_vector, slot):
        start = GLOBAL_SIZE + slot * REGION_SIZE
        return state_vector[start:start + REGION_SIZE]

    # Unrepaired, the two sides order their rows by the wire's distances, which are both measured from THIS process's base. So they agree about the order and disagree about the ground: slot nought is our own region for us and the enemy's region for them.
    mine, yours = state(ours, our_squad, 0), state(theirs, their_squad, 1)
    apart = [name for name, a, b in zip(REGION_FEATURES, rows(mine, 0), rows(yours, 0)) if abs(a - b) > 1e-9]
    assert "held_by_enemy" in apart and "force_edge" in apart, (
        "before the arena repairs the two views, slot nought should be different ground for the two sides, "
        "and here it differs only in %s" % apart)

    # What the arena does as it builds them: each side's distances measured again from the region it stages out of, its anchor put at its own staging point, and its contact record written from what is standing there for that side rather than from this process's own fog.
    ours = _contacts(rehome(ours, 1, STAGE))
    theirs = _contacts(rehome(theirs, 2, _reflected(*STAGE)))
    mine, yours = state(ours, our_squad, 0), state(theirs, their_squad, 1)

    for slot in (0, 1):
        apart = [name for name, a, b in zip(REGION_FEATURES, rows(mine, slot), rows(yours, slot))
                 if abs(a - b) > 1e-9]
        assert not apart, "region slot %d differs between the sides in %s" % (slot, apart)
    # The one feature that is about the decision rather than about the board is congruent too. Each squad is marked at the region its own contract names — ours at region one, its mirror at region two — and because the slots run outward from each side's own home, the two marks land in the same row. A mark laid out by the map's numbering would not.
    marked = operational_state(ours, None, [our_squad], 30000, base=0, contracted=1)
    theirs_marked = operational_state(theirs, None, [their_squad], 30000, base=1, contracted=2)
    assert marked == theirs_marked, "the contracted flag is not congruent between the two sides"
    assert marked != mine, "marking the contracted region changed nothing"

    assert mine == yours, (
        "the whole operational cut of a mirrored board should read the same from either side, and the first "
        "of %d numbers to differ is at %s"
        % (len(mine), next(i for i, (a, b) in enumerate(zip(mine, yours)) if a != b)))

    # The squad offset is load-bearing and not tidying: read the mirror side's squad by its global number, as the cut did before, and its row lands somewhere else entirely.
    assert state(theirs, their_squad, 0) != yours


def test_the_region_block_says_how_many_squads_are_committed_to_each_region():
    """The region block carried where the squad being decided about was going and nothing about where the others were sent, so a layer could not price a second squad onto ground a first was already taking, nor massing three on one contest.

    The reward it is fitted against prices no coordination either -- a squad is paid what its own surviving units account for -- so the count is what makes the allocation visible at all. It is counted off the contracts in force, which is this process's own statement about where a squad was sent and nothing the game reports, and it has to be congruent under the mirror for the same reason every other feature does.
    """
    _, ours, theirs, our_squad, their_squad = _mirrored_board()
    ours = _contacts(rehome(ours, 1, STAGE))
    theirs = _contacts(rehome(theirs, 2, _reflected(*STAGE)))

    def rows(state_vector, slot):
        start = GLOBAL_SIZE + slot * REGION_SIZE
        return dict(zip(REGION_FEATURES, state_vector[start:start + REGION_SIZE]))

    # One squad each, contracted to congruent ground: ours names region one, its mirror names region two.
    mine = operational_state(ours, None, [our_squad], 30000, base=0)
    yours = operational_state(theirs, None, [their_squad], 30000, base=1)
    assert mine == yours, "the count is not congruent between the two sides"
    # Which row the contracted region lands in is the ordering's business — the rows run outward from where the side
    # staged, so it is the row that says it is contracted, not a fixed slot number.
    order = sorted(ours.regions, key=lambda region: region.distance_from_home)
    sent = next(slot for slot, region in enumerate(order)
                if region.id == our_squad.contract.target_region)
    other = 1 - sent
    committed = [rows(mine, sent)["committed"], rows(mine, other)["committed"]]
    assert committed[0] > 0.0, "the region a squad is contracted to is not counted"
    assert committed[1] == 0.0, "a region nobody is contracted to is counted"

    # A second squad onto the same ground moves the number, which is the whole point: two squads on one contest read differently from one.
    second = _squad(id=4)
    both = operational_state(ours, None, [our_squad, second], 30000, base=0)
    assert rows(both, sent)["committed"] > committed[0], (
        "a second squad contracted to the same region did not raise the count")

    # A squad under no contract has not been sent anywhere and counts nowhere, which is a different statement from having been sent home.
    idle = _squad(id=5, contract=False)
    with_idle = operational_state(ours, None, [our_squad, idle], 30000, base=0)
    assert rows(with_idle, sent)["committed"] == committed[0], "an uncontracted squad was counted somewhere"


def test_the_operational_squad_rows_are_offset_by_the_sides_own_first_squad():
    """Why the offset is fixed once rather than recomputed from whoever is alive.

    A match hands squads out from nought, so the offset is nought and a squad's row is its own number — which is what makes a row mean the same thing from one period to the next. A constructed arena numbers the second side's squads after the first side's, so without the offset the two sides' congruent squads are written into different rows and named to the network by different one-hot slots. Recomputing the offset from the living squads each period would give a third behaviour, worse than both: the first squad of a side dying would renumber every squad above it, and a row would stop meaning one squad.
    """
    ours = [_squad(id=0), _squad(id=1)]
    theirs = [_squad(id=2), _squad(id=3)]
    assert squad_slots(ours, 0) == {0: 0, 1: 1}
    assert squad_slots(theirs, 2) == {2: 0, 3: 1}
    # A death does not renumber what is left, because the offset is the side's first squad ever and not its first squad now.
    assert squad_slots([theirs[1]], 2) == {3: 1}
    assert squad_mask(ours, 0)[:2] == [1.0, 1.0]
    assert squad_mask(theirs, 2)[:2] == [1.0, 1.0]


def test_the_two_sides_spawn_orders_are_interleaved_so_neither_leads():
    """A fight is two spawn orders sent down one command queue, and the queue is drained in the order it was filled. Sending one side's whole order before the other's leaves that side more completely on the board when the spawn wait is called, which was measured as three points of a left-right lean in the strength that forms into a squad — enough to bias the self-play score the arena is scored against. Interleaving keeps both orders at the same depth in the queue throughout, so a wait that runs out cuts both to the same degree."""
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
        # Bounded by the shares, which run from minus one to plus one, less the draw term, which cannot move the score by more than half the slope.
        assert -1.0 - STRENGTH_SLOPE_KILLS / 2 <= ours.outcome <= 1.0 + STRENGTH_SLOPE_KILLS / 2

    # A side that was never built at all is worth nothing and has lost nothing, which has to be a number rather than a division by nought: an engagement whose spawns never arrived on one side still reaches the point where it is scored.
    empty = _fight(0.0, 1200.0, 0.0, 0.0)
    assert empty.outcome + _fight(1200.0, 0.0, 0.0, 0.0).outcome == 0.0


def test_destroying_the_other_side_without_a_loss_is_the_top_of_the_scale():
    """What fixes the size of the scale, and with it how much a called fight is worth against the errand's own conclusions: a massacre is paid exactly what taking the contracted ground is paid, and no more, so that a layer is never taught to prefer the one to the other.

    Stated on an even draw, so that it holds whatever the term that takes out what the draw was worth is set to. On an even draw the strength share is a half and that term is nought regardless of the multiple, so the scale is exactly one whatever the multiple is; away from an even draw the term moves the score, which is the next test.
    """
    assert _fight(3000.0, 3000.0, 3000.0, 0.0).outcome == 1.0
    assert _fight(3000.0, 3000.0, 0.0, 3000.0).outcome == -1.0
    assert _fight(2000.0, 4000.0, 2000.0, 0.0).outcome >= 1.0
    # And the middle of it is an even trade between sides of equal worth.
    assert _fight(2000.0, 2000.0, 1000.0, 1000.0).outcome == 0.0


def test_the_draw_term_charges_the_stronger_side_for_the_advantage_it_was_dealt():
    """A fight the two sides destroy each other in is an even result, but not from an even start: the side dealt the larger force was the one expected to win it and only broke even, so the draw term marks its score down by what the advantage was worth and the weaker side's up by the same. What it subtracts is exactly the multiple times how far the share sat from a half, which is the part of the score that was the draw rather than the play, and it leaves the antisymmetry the measurement rests on untouched.

    On the sparse and the health readings alike, because each carries its own multiple and the draw is a property of the start rather than of which reading scores the end.
    """
    share = 4000.0 / 6000.0
    strong = _fight(4000.0, 2000.0, 0.0, 0.0)  # dealt two thirds of the strength, traded down to nothing
    weak = _fight(2000.0, 4000.0, 0.0, 0.0)
    assert abs(strong.outcome - (-STRENGTH_SLOPE_KILLS * (share - 0.5))) < 1e-9
    assert abs(strong.outcome_health - (-STRENGTH_SLOPE_HEALTH * (share - 0.5))) < 1e-9
    assert strong.outcome < 0.0 < weak.outcome
    assert abs(strong.outcome + weak.outcome) < 1e-12
    assert abs(strong.outcome_health + weak.outcome_health) < 1e-12


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
        assert -1.0 - STRENGTH_SLOPE_HEALTH / 2 <= ours.outcome_health <= 1.0 + STRENGTH_SLOPE_HEALTH / 2

    # A side whose spawns never arrived reaches the point where it is scored like any other, and on this reading too that has to be a number rather than a division by nought.
    empty = _fight(0.0, 1200.0, 0.0, 0.0)
    assert empty.outcome_health + _fight(1200.0, 0.0, 0.0, 0.0).outcome_health == 0.0


def test_the_two_readings_agree_on_a_clean_kill_and_part_company_on_damage():
    """What the health reading is for, stated as the difference between the two.

    A fight that ended with one side destroyed and the other untouched is worth the same under both, because a dead unit is worth nothing whichever way it is counted and an untouched survivor is worth its whole price under both. That is the case pinned below, and it is narrower than it looks: a winner that took damage on the way is worth less on the health reading, so the two readings do part company on most real annihilations, and only the clean kill is the fixture they have to agree on.

    Where the two part company is the damage left standing on whoever survived. A fight is called twelve seconds after its last casualty rather than for want of one, so a called fight has almost always had its dead and the sparse reading scores it on them; what it cannot see at any point is a survivor at half health, which is worth half its price on the health reading and its whole price on the other. The fight built below isolates that difference by killing nobody, which is why it is a fixture and not a common fight.
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


def test_an_arena_refuses_a_score_it_was_not_taught():
    """A misspelt score would otherwise fall through to whichever reading the code tests for by name, and the run would be paid on one reading while whoever started it believed it was paying the other. Both readings are reported either way, so nothing in the log would say which had been paid. Refused where it is still one line rather than a measurement nobody can interpret afterwards."""
    from rwintel.learn.arena import Arena

    refused = _refusal(ValueError, lambda: Arena(None, score="bodies"))
    assert "bodies" in refused
    # Named with what it would have taken, because the first thing anyone does with a refusal is go looking for the spelling.
    assert all(name in refused for name in SCORES)


# ---- where one errand stops and the next begins ---------------------------------------------

class _Fixed:
    """A decider that answers the same way whatever it is shown, so that what these tests look at is the bookkeeping around a decision rather than the decision."""

    def __init__(self, action=0):
        self.action = action

    def choose_many(self, requests):
        return [Choice(action=self.action, log_prob=-1.6, value=0.25) for _ in requests]


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

    layer.finish(squad, 0.75, "called")
    trajectory, = rollout.done
    step, = trajectory.steps
    assert trajectory.finished and step.done
    assert abs(step.reward - (0.75 - held)) < 1e-9
    assert not layer.pending and squad.id not in layer.reward.missions
    assert layer.terminals["called"] == 1


def test_a_decision_taken_after_the_fight_was_called_belongs_to_a_new_trajectory():
    """Squad numbers are handed round: the arena has two of them and uses them for every fight it builds, and a match has eight for as many errands as are ever run.

    So the decision left outstanding when one fight ends must not be paid out of the first period of the next one. Doing that strings two fights into a single trajectory, and generalised advantage estimation then runs the advantage of the second backwards into the decisions of the first — which teaches a layer that what it did in a fight it has already finished was answerable for what happened in a fight it had not yet begun.
    """
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    squad = _squad()
    view = _skirmish()
    layer.decide(view, [squad], 21000)
    layer.finish(squad, 0.75, "called")
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


def test_an_operational_squad_that_leaves_the_board_is_cut_rather_than_ended():
    """The operational reward is a statement about the whole board, so a squad leaving it — folded into another by the organisation layer, disbanded, or wiped — does not end the board or the errand the reward is about: the orders and every other squad go on. Its last decision is therefore bootstrapped from its own value estimate, as any decision that merely stopped being observed is, and not closed against a continuation of nought.

    Marking it done would bootstrap from nought and assert the world ended where a squad turned over, which a routine merge of a healthy squad does several times a match. That would teach the critic that the states before every merge are worth nothing from here, corrupting the baseline the operational advantage of every other squad is taken against — the one thing a benign merge must leave untouched.
    """
    class _PerSquadReward:
        # The figure is what a contest that reads its own scored ground hands in for the period, and it is None here because this is the match path, where no such board exists and the region block is the whole of the signal. Taken and ignored so that the double has the signature the real reward has: the layer passes it on every path, and a double that could not accept it would be testing a call the layer does not make.
        def step(self, squad, view, orders, figure=None):
            return Outcome(reward=0.1)
        def forget(self, squad_id):
            pass

    rollout = Rollout()
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0)
    layer.reward = _PerSquadReward()
    # Squad 1 has a decision already accruing in the buffer from an earlier period, plus one still pending.
    rollout.add((0, 1), Step(state=[0.0], action=0, mask=[1.0], value=0.7, reward=0.05, squad=1))
    layer.pending[0] = Step(state=[0.0], action=0, mask=[1.0], value=0.4, squad=0)
    layer.pending[1] = Step(state=[0.0], action=0, mask=[1.0], value=0.7, squad=1)

    # Squad 0 is still on the board this period; squad 1 has left it.
    layer._settle(None, None, [_squad(id=0, contract=False)])

    # The squad still present is paid and keeps a live trajectory that goes on accruing; the one that left has its accrued trajectory cut, not finished, so its last decision bootstraps from its own value rather than from nought. A squad off the board cannot be paid from anything, so its still-pending decision is dropped rather than scored against a board it is no longer on.
    assert (0, 0) in rollout.live and not rollout.live[(0, 0)].finished
    assert rollout.live[(0, 0)].steps[-1].reward == 0.1
    cut, = rollout.done
    step, = cut.steps
    assert not cut.finished and not step.done
    assert cut.tail_value == step.value == 0.7
    assert step.reward == 0.05
    assert not layer.pending


def test_shaping_earned_in_a_period_with_no_decision_waiting_is_carried_and_not_dropped():
    """A layer does not take a decision about every squad every period — the inherited operational rule leaves a squad worn below the health it will task at all out of the decision, with its contract still standing — while the reward advances that squad's potential regardless.

    The shaping only means anything because it telescopes: what an errand returns is its terminal less the potential it opened at, and only if every increment in between reached a step. A period whose increment was computed and thrown away punches a hole in that sum, and the return of the errand is then wrong by whatever the board did over that period. So an unpaid increment is carried to the next decision that is paid, and to the terminal if none is.
    """
    class _Ticking:
        """A reward that pays a tenth every period, so what arrived and what was paid can be told apart by counting."""

        # The figure a contest hands in for the period, None on the match path this test drives, taken so the double matches the real reward's signature.
        def step(self, squad, view, orders, figure=None):
            return Outcome(reward=0.1)

        def forget(self, squad_id):
            pass

        def ended(self, squad_id):
            return False

        def close(self, squad_id):
            return 0.0

    rollout = Rollout()
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0)
    layer.reward = _Ticking()
    squad = _squad(id=0)

    # Three periods in which the layer took no decision about this squad, then one in which it did.
    for _ in range(3):
        layer._settle(None, None, [squad])
    assert abs(layer.owed[squad.id] - 0.3) < 1e-9
    layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=squad.id)
    layer._settle(None, None, [squad])

    paid, = rollout.live[(0, squad.id)].steps
    assert abs(paid.reward - 0.4) < 1e-9, "the periods without a decision were dropped instead of carried"
    assert squad.id not in layer.owed

    # And what is still carried when the errand ends from outside goes into the terminal payment, which with no decision waiting is reached back to the step already in the buffer and added to what it holds.
    for _ in range(2):
        layer._settle(None, None, [squad])
    layer.finish(squad, 0.6, "horizon")
    trajectory, = rollout.done
    assert abs(trajectory.steps[-1].reward - (0.4 + 0.6 + 0.2)) < 1e-9
    assert not layer.owed


def test_finishing_an_operational_errand_pays_its_terminal_net_of_the_last_potential():
    """The terminal comes from outside — the constructed operations arena at its horizon, which knows the region domination the whole errand is scored on — not from the board. What the errand returns has to be that terminal and nothing else: the shaping cancels over the errand, both its last term and its first.

    Cancelling only the last one is what the code used to do, and it left the opening potential standing in every return. That residue is not a constant — the opening potential is read off the region the decision itself named — so it was a term of the action: a squad sent at ground the enemy held opened near the bottom and kept its whole terminal, while a squad sent to hold ground already ours opened near the top and had that much taken away. The arena's score says the second is worth half the region's priority and the first is worth nothing, so the signal was pointed the other way round from the quantity being measured.

    Here the board moves under the errand, so the shaping is not zero, and the sum of everything the errand was paid still has to come to the terminal exactly.
    """
    class _Orders:
        priorities = {1: 1.0}

    rollout = Rollout()
    # Built at the discount the arena uses, which is the only place a terminal is paid at all: an arena contest is one whole bounded errand and is discounted at nothing, and it is at nothing that the shaping cancels exactly.
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0, discount=1.0)
    squad = _squad(id=0)
    orders = _Orders()
    # Opens at 1.0 * (share(100, 900) - a half) = -0.4, a region we are being beaten in.
    layer.reward.step(squad, _view([], [_region(1, ours=100.0, theirs=900.0)]), orders)
    assert abs(layer.reward.missions[squad.id].opening - -0.4) < 1e-9

    # Two periods in which the region comes our way, each paying its own shaping term.
    for ours, theirs in ((500.0, 500.0), (900.0, 100.0)):
        layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=squad.id)
        layer._settle(_view([], [_region(1, ours=ours, theirs=theirs)]), orders, [squad])
    layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=squad.id)

    layer.finish(squad, 0.6, "dominated")
    trajectory, = rollout.done
    assert trajectory.finished and trajectory.steps[-1].done
    assert abs(sum(step.reward for step in trajectory.steps) - 0.6) < 1e-9, (
        "the errand returned something other than its terminal, so the shaping did not cancel")
    assert layer.terminals["dominated"] == 1
    assert squad.id not in layer.reward.missions


def test_a_wiped_operational_squad_takes_its_terminal_on_the_last_step_it_left_behind():
    """A squad gone between periods has no decision waiting to hang the terminal on: the period that finds it gone takes no decision, and the last one there was has already gone into the buffer as an ordinary step. So the terminal is reached back to that step through the buffer's close-with, exactly as the tactical layer does for a squad wiped between periods."""
    class _Orders:
        priorities = {1: 1.0}

    rollout = Rollout()
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0)
    squad = _squad(id=0)
    view = _view([], [_region(1, ours=100.0, theirs=900.0)])
    # Opens the mission at 1.0 * (share(100, 900) - a half) = -0.4, the priority-weighted domination of a region we are being beaten in.
    layer.reward.step(squad, view, _Orders())
    rollout.add((0, squad.id), Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=squad.id))

    layer.finish(squad, 0.6, "wiped")
    trajectory, = rollout.done
    step, = trajectory.steps
    assert trajectory.finished and step.done
    # Nothing moved under this errand, so its shaping is nought either way and the return is the terminal alone.
    assert abs(step.reward - 0.6) < 1e-9
    assert layer.terminals["wiped"] == 1


def test_re_tasking_an_operational_squad_hands_back_the_ground_it_is_leaving():
    """What the match's operational ledger pays in the period a layer changes its mind, on the real reward and the real buffer rather than by argument.

    One ledger a squad from its first period to its last, so a fresh contract is a change of density and not a change of objective: the period that re-tasks a squad pays the new region's valuation less the old region's, which hands back everything the ground it is leaving had earned it and takes on what the ground it is going to is worth. That is the property that makes the whole episode telescope to its last reading less its first, and it is what a re-based ledger gave away — under that one this same squad kept the two quarters it had gained on a region and paid nothing at all for walking off it, so a policy could bank a rise and duck the fall that was coming by writing itself a new contract.

    The trajectory is not cut here either, and for the same reason: there is no boundary to cut at when both errands are priced out of the one region table. A cut arrives when the squad leaves the board, which is the test above, and a terminal only ever from outside.
    """
    class _Orders:
        priorities = {1: 1.0, 2: 1.0}

    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0, discount=1.0)
    squad = _squad(id=0)
    orders = _Orders()

    def board(ours, theirs):
        return _view([], [_region(1, ours=ours, theirs=theirs), _region(2, ours=500.0, theirs=500.0)])

    # Opens at 1.0 * (share(100, 900) - a half) = -0.4, a region we are being beaten in.
    layer.reward.step(squad, board(100.0, 900.0), orders)
    assert abs(layer.reward.missions[squad.id].opening - -0.4) < 1e-9

    # Two periods in which the region comes our way, each paying its own shaping term to the decision that was waiting.
    for value, (ours, theirs) in ((0.2, (500.0, 500.0)), (0.6, (900.0, 100.0))):
        layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=value, squad=squad.id)
        layer._settle(board(ours, theirs), orders, [squad])

    # And a third in which the layer sends the squad somewhere else, which is what a policy that re-draws every period does whenever it changes its mind. The region it is sent to is level, so it is worth nought against the +0.4 it is walking away from.
    layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.3, squad=squad.id)
    squad.contract = TaskContract(squad=squad.id, task=Task.ATTACK, target_region=2,
                                  stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                  deadline_ms=90000, issued_at_ms=24000)
    layer._settle(board(900.0, 100.0), orders, [squad])

    # Still one live trajectory: nothing ended, and the decisions of the two contracts belong to the one ledger.
    assert not rollout.done
    trajectory = rollout.live[(0, squad.id)]
    assert [round(step.reward, 9) for step in trajectory.steps] == [0.4, 0.4, -0.4]
    # The whole of what has been paid is the movement of one quantity: the valuation now, less the one the ledger opened at.
    assert abs(sum(step.reward for step in trajectory.steps) - (0.0 - -0.4)) < 1e-9
    assert not layer.terminals

    # And the period that re-tasked is charged rather than being free, which is the whole of the difference from the ledger this replaced.
    assert trajectory.steps[-1].reward < 0.0


# ---- a board that reads its own scored ground ----------------------------------------------

#: What the strategic layer said each of the two discs was worth, and the ownership each opened at. The first is a disc the enemy's garrison stood on, so it opens at nought and can only be gained by taking it; the second is one ours stood on, so it opens whole and can only be lost.
_SCORED_PRIORITY = {1: 0.8, 2: 0.5}
_SCORED_OPENING = {1: 0.0, 2: 1.0}

#: The board period by period: which disc the squad's contract named, and this side's share of that disc's catchment as that board read. The squad works the enemy's disc and takes half of it, is re-tasked onto its own, loses units inside it so it slips away, and is re-tasked back to finish the first — two re-taskings and a casualty, which is what a policy that re-draws its region every period actually does.
_SCORED_BOARD = ((1, 0.00), (1, 0.25), (1, 0.50), (2, 0.95), (2, 0.90), (2, 0.70), (1, 0.75), (1, 0.60))

#: And the board the contest is scored on, which has moved again since the last decision was taken.
_SCORED_HORIZON = (1, 0.65)


def _figure(period) -> float:
    """One squad's scored figure on one board: what the disc its contract names has moved from the ownership that disc opened at, weighted by what the disc was said to be worth. This is the whole of the quantity — the arena reads it off a health-weighted catchment, and what reaches the layer is this number."""
    region, share = period
    return _SCORED_PRIORITY[region] * (share - _SCORED_OPENING[region])


def _scored_episode(skip=(), squads=(0,)):
    """Drives `_SCORED_BOARD` through the real layer, the real reward and the real buffer, one decision a squad a period except where `skip` says the layer passed the squad over, and closes at the horizon the way the arena does."""
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0, discount=1.0)
    records = {squad_id: _squad(id=squad_id) for squad_id in squads}
    issued = {squad_id: 0 for squad_id in squads}
    for index, period in enumerate(_SCORED_BOARD):
        region = period[0]
        for squad_id, squad in records.items():
            if squad.contract is None or squad.contract.target_region != region:
                # A fresh contract, exactly as the inherited rule writes one when the region changes. Under the region block this is what cuts the trajectory; here it must not.
                issued[squad_id] += 1000
                squad.contract = TaskContract(squad=squad_id, task=Task.ATTACK, target_region=region,
                                              stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                              deadline_ms=90000, issued_at_ms=issued[squad_id])
        layer.standing({squad_id: _figure(period) for squad_id in records})
        layer._settle(None, None, list(records.values()))
        if index + 1 not in skip:
            for squad_id in records:
                layer.pending[squad_id] = Step(state=[0.0], action=0, mask=[1.0],
                                               value=0.05 * (index + 1) + 0.01 * squad_id,
                                               squad=squad_id)
    for squad in records.values():
        layer.finish(squad, _figure(_SCORED_HORIZON), "horizon")
    return rollout, layer


def test_a_scored_board_pays_a_squad_the_movement_of_its_own_figure():
    """Every period pays the movement of the squad's own scored figure, each board read under the contract in force at that board and against that disc's own opening. The payments are differences of one quantity, so they telescope: the episode sums to the last reading of the figure, which is exactly the terminal the contest pays at its horizon. The objective is unchanged and only its density changes, which is what makes this a credit assignment rather than a new objective."""
    rollout, _ = _scored_episode()
    trajectory, = rollout.done
    paid = [step.reward for step in trajectory.steps]

    figures = [_figure(period) for period in _SCORED_BOARD] + [_figure(_SCORED_HORIZON)]
    expected = [after - before for before, after in zip(figures, figures[1:])]
    assert len(paid) == len(expected)
    for index, (got, want) in enumerate(zip(paid, expected)):
        assert abs(got - want) < 1e-12, "period %d paid %r rather than the movement of the figure %r" % (
            index + 1, got, want)
    assert abs(sum(paid) - _figure(_SCORED_HORIZON)) < 1e-12, (
        "the episode returned something other than the terminal")


def test_the_payments_telescope_to_the_terminal_across_two_re_taskings():
    """The property a weaker dense credit lacks, and the whole reason this one is written as it is.

    On the period a squad is re-tasked it hands back the entire figure it had banked on the disc it is leaving and takes on the new disc's standing measured from that disc's own opening. It is not paid for having re-tasked and it is not charged for it; the sum depends only on where the disc it ends on ends. A ledger that re-set its origin at each change of contract would instead let the squad keep what it had gained on one disc and open clean on another — banking a rise and ducking a fall, which is a change of objective smuggled in as a change of density and one a policy can help itself to by re-tasking. Measured on this board that ledger returns something other than the terminal, and it is the difference between the two that this pins.
    """
    rollout, _ = _scored_episode()
    trajectory, = rollout.done
    paid = [step.reward for step in trajectory.steps]

    # The first hand-off: the squad had taken half of a disc worth 0.8 and is sent to one it opened whole.
    assert abs(paid[2] - (_figure(_SCORED_BOARD[3]) - _figure(_SCORED_BOARD[2]))) < 1e-12
    assert abs(paid[2] - -0.425) < 1e-12, "the squad did not hand back what it had banked on the disc it left"
    # The second: back onto the first disc, which its allies have carried further while it was away.
    assert abs(paid[5] - (_figure(_SCORED_BOARD[6]) - _figure(_SCORED_BOARD[5]))) < 1e-12
    assert abs(paid[5] - 0.75) < 1e-12
    assert abs(sum(paid) - _figure(_SCORED_HORIZON)) < 1e-12

    # And what a re-set origin would have returned on the same board: each errand paid its own disc's movement from wherever it stood when the errand opened, so the rise on the first disc is banked and the fall on the second is ducked.
    rebased = 0.0
    previous = 0.0
    for index, period in enumerate(_SCORED_BOARD):
        changed = index > 0 and period[0] != _SCORED_BOARD[index - 1][0]
        rebased += 0.0 if changed else _figure(period) - previous
        previous = _figure(period)
    rebased += _figure(_SCORED_HORIZON) - previous
    assert abs(rebased - _figure(_SCORED_HORIZON)) > 0.3, (
        "this board no longer separates the two ledgers, so it cannot pin the one that telescopes")


def test_a_squad_losing_its_units_is_charged_as_it_loses_them():
    """The squad's units die inside the disc it is holding and the disc slips from nine tenths to seven. Under a terminal paid once that arrives as a lump at the horizon, tens of decisions after the ones that led to it; paid every period it is charged to the decision that was in force while it happened, which is the point of a dense credit. Nothing about the sum changes — that is the point of this one."""
    rollout, _ = _scored_episode()
    trajectory, = rollout.done
    paid = [step.reward for step in trajectory.steps]

    losing = _figure(_SCORED_BOARD[5]) - _figure(_SCORED_BOARD[4])
    assert abs(losing - 0.5 * (0.70 - 0.90)) < 1e-12
    assert abs(paid[4] - losing) < 1e-12, "the charge did not land on the period the units were lost on"
    assert abs(sum(paid) - _figure(_SCORED_HORIZON)) < 1e-12


def test_a_scored_board_leaves_no_errand_boundary_to_cut_at():
    """The direct regression on the defect. The squad is handed a fresh contract twice over this episode, and under the region block each of those cut its trajectory: the period that did the replacing was written a reward of nought, its advantage came out at exactly nought at a discount and a trace of one, every earlier decision was anchored by nothing but two of the critic's own estimates, and the terminal reached none of them.

    Paid off one quantity from the first decision to the last there is no boundary left to cut at. One trajectory, finished, no reason, and every decision of the episode inside it.
    """
    rollout, layer = _scored_episode()
    assert not rollout.live, "the episode left a trajectory open"
    trajectory, = rollout.done
    assert trajectory.finished and trajectory.reason == ""
    assert trajectory.steps[-1].done and len(trajectory.steps) == len(_SCORED_BOARD)
    assert layer.terminals["horizon"] == 1
    assert not layer.owed and not layer.reward.missions


def test_a_scored_board_replaces_the_region_block_rather_than_joining_it():
    """Which of the two quantities a period is paid in is a decision about the board and not about the squad, and where a contest reads its own ground its reading replaces the region block rather than being added to it.

    Added, the block's own movement would stand in the return as well, and that residue is not a constant: the block is re-based at every contract, only the last errand's movement is ever cancelled, and what is left over is chosen by the policy — a squad can bank a rise on one block and re-task away before the fall, which is the very thing the scored ledger is written to forbid. It would also be worth nothing here: the block is the whole Voronoi cell at unit prices with this side's free base standing in it, while the figure is a health-weighted disc about a contest point some hundreds of units away.

    So the block is made to move sharply under the squad while the figure moves a little, and what the squad is paid has to be the figure's movement alone.
    """
    class _Orders:
        priorities = {1: 1.0}

    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0, discount=1.0)
    squad = _squad(id=0)
    orders = _Orders()

    # The block swings from a region we are being beaten in, through level, to one we hold: four tenths of its own quantity a period. The scored figure moves a tenth and then two tenths.
    for figure, (ours, theirs) in ((0.0, (100.0, 900.0)), (0.1, (500.0, 500.0)), (0.3, (900.0, 100.0))):
        layer.standing({squad.id: figure})
        layer._settle(_view([], [_region(1, ours=ours, theirs=theirs)]), orders, [squad])
        layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=squad.id)

    paid = [step.reward for step in rollout.live[(0, squad.id)].steps]
    assert [round(reward, 12) for reward in paid] == [0.1, 0.2], (
        "the region block reached a period the contest was paying for")
    # The scored ledger opens at nought and holds the last figure, which is what makes the horizon payment the last difference rather than the whole of it.
    mission = layer.reward.missions[squad.id]
    assert mission.opening == 0.0 and abs(mission.potential - 0.3) < 1e-12
    assert abs(layer.reward.close(squad.id) - 0.3) < 1e-12


def test_a_squad_the_layer_passed_over_carries_its_scored_credit_forward():
    """The inherited rule leaves a squad worn below the health it will task at all out of the decision while its contract stands, so no step is recorded for it that period and the movement of its figure has nothing to be paid to. Dropped, the sum stops telescoping and the episode returns the terminal less whatever those periods moved. Carried to the next decision that is paid — and to the terminal if none is — the telescope closes again."""
    rollout, layer = _scored_episode(skip=(3, 4))
    trajectory, = rollout.done
    paid = [step.reward for step in trajectory.steps]

    assert len(paid) == len(_SCORED_BOARD) - 2, "a decision was recorded on a period the layer passed the squad over"
    assert abs(sum(paid) - _figure(_SCORED_HORIZON)) < 1e-12
    # The decision that follows the passed-over periods is paid what they moved as well as what it moved itself.
    assert abs(paid[2] - (_figure(_SCORED_BOARD[5]) - _figure(_SCORED_BOARD[2]))) < 1e-12
    assert not layer.owed


def test_a_squad_between_contracts_hands_its_ground_back_rather_than_forgetting_it():
    """A squad holding no contract has moved no ground, so its figure is nought and the honest payment is nought less whatever the ledger held — the same hand-back a re-tasking makes.

    This is why the scored branch is taken before the guard that drops a contract-less squad's mission and not after it. Dropped there, the ledger would restart at nought with the hand-back unpaid, and the next contracted period would pay its whole figure again: the episode would return the terminal plus everything banked before the gap, which is the re-basing this whole ledger exists to forbid arriving through the back door.
    """
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0, discount=1.0)
    squad = _squad(id=0)

    # The squad takes three tenths of a disc's worth, then holds no contract for a period, then is sent out again and takes half.
    for index, figure in enumerate((0.0, 0.3, 0.0, 0.5)):
        squad.contract = None if index == 2 else _squad(id=0).contract
        layer.standing({squad.id: figure})
        layer._settle(None, None, [squad])
        layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.1 * index, squad=squad.id)
    layer.finish(squad, 0.5, "horizon")

    trajectory, = rollout.done
    paid = [step.reward for step in trajectory.steps]
    assert abs(paid[1] - -0.3) < 1e-12, "the ground held before the gap was not handed back"
    assert abs(paid[2] - 0.5) < 1e-12
    assert abs(sum(paid) - 0.5) < 1e-12, "the episode paid what was banked before the gap a second time"


def test_the_scored_batch_is_paid_and_distinct():
    """The census counterpart of the batch that could not learn, on the same four squads and the same buffer.

    Under the region block a batch of this shape read `paid_steps 0`, `cut {'renewed': …}` and an advantage of exactly nought on nearly every step — and normalisation turns a mass of identical noughts into one identical nonzero number, which is a uniform push on whatever the policy happened to draw, on no merit at all. Paid every period, every step lies in a trajectory a payment reached, nothing is cut, and the advantages are as many different numbers as there are decisions.
    """
    rollout, _ = _scored_episode(squads=(0, 1, 2, 3))
    steps = rollout.drain()
    census = rollout.census

    assert census.steps == len(steps) == 4 * len(_SCORED_BOARD)
    assert census.paid_steps == census.steps and census.cut == {} and census.finished == 4
    assert census.zero_advantage == 0
    assert census.distinct == census.steps, "the batch holds fewer advantages than decisions"


def test_a_layer_built_with_a_discount_pays_its_shaping_at_that_discount():
    """One figure discounts the returns and telescopes the shaping, and a run states it once. It reaches the shaping only by being handed to the layer and passed on from there, so this is the join where the two could silently come apart — and a shaping term telescoping at one figure while the returns are discounted at another is a term that moves which policy is best.

    A layer nobody told is a layer inside a match, whose errand is a fragment of one, and the constant is that case.
    """
    assert LearntTactics(None, _CATALOGUE, _Fixed(), discount=1.0).reward.discount == 1.0
    assert LearntTactics(None, _CATALOGUE, _Fixed(), discount=0.5).reward.discount == 0.5
    assert LearntTactics(None, _CATALOGUE, _Fixed()).reward.discount == DISCOUNT


# ---- starting from something rather than from noise -----------------------------------------

def test_a_teacher_whose_choice_follows_from_its_board_is_learnt_almost_exactly():
    """A floor rather than a measurement of the real teacher. The rule ladder is a function of the board and so is this, so a fit that cannot recover a plainly separable one has something wrong with it that no amount of real data would fix."""
    draw = random.Random(11)
    samples = []
    for _ in range(600):
        action = draw.randrange(TACTICAL_ACTIONS)
        state = [draw.uniform(-0.2, 0.2) for _ in range(TACTICAL_SIZE)]
        state[action] = 1.0
        samples.append(Sample(state=state, action=action))

    _, cloning = fit(samples, seed=3)
    assert cloning.training.accuracy > 0.9
    assert cloning.validation.accuracy > 0.9
    # And it is still a distribution rather than a lookup table, which is what the label smoothing is there for: a policy that arrives at the reinforcement run certain of everything produces no evidence that would ever argue it out of an opinion.
    assert cloning.training.entropy > 0.0


def _teacher(folder, *rows, name="teacher.jsonl"):
    """A teacher file made of the rows given, in the form the collecting run writes one: a JSON object per line, the feature list at the head and a decision on every line after it."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_a_teacher_written_by_a_different_feature_list_is_refused_rather_than_fitted():
    """A teacher file records an encoding as much as it records a policy, and the numbers in it cannot say which one, so it states its feature list at its head and that is what is checked before a decision is taken from it.

    Three refusals, and the middle one is why the list is written down at all: a file whose features were renamed has rows of exactly the right length, so nothing about their shape would ever have caught it, and a fit to it yields a network reading two slots as something they no longer are while reporting a perfectly ordinary accuracy. A file stating no list is refused as well, since it was collected before the list was recorded and the only alternative to refusing it is to assume the answer. The length of a row is still checked, as the backstop against a truncated file, and it is named to the line, because the first thing anyone does with a file that has been refused is go and look at it.
    """
    assert "decision 0" in _refusal(TeacherMismatch, fit, [Sample(state=[0.0] * (TACTICAL_SIZE - 1), action=0)])

    stated = {"layer": "tactics", "encoding": list(TACTICAL_FEATURES), "recipe": TACTICAL_RECIPE,
              "rule": rule_recipe(TACTICAL)}
    decision = {"state": [0.0] * TACTICAL_SIZE, "action": 0}
    with tempfile.TemporaryDirectory() as folder:
        refusal = _refusal(TeacherMismatch, read_teacher, _teacher(folder, decision))
        assert "line 1" in refusal and "does not state the feature list" in refusal

        renamed = dict(stated, encoding=["target_dx" if name == "target_ahead_of_home" else name
                                         for name in TACTICAL_FEATURES])
        refusal = _refusal(TeacherMismatch, read_teacher, _teacher(folder, renamed, decision))
        assert "'target_dx'" in refusal and "'target_ahead_of_home'" in refusal

        short = {"state": [0.0] * (TACTICAL_SIZE + 1), "action": 0}
        assert "line 3" in _refusal(TeacherMismatch, read_teacher, _teacher(folder, stated, decision, short))
        # And the file the collecting run actually writes is read back whole.
        assert len(read_teacher(_teacher(folder, stated, decision, decision))) == 2


def test_what_a_policy_would_choose_is_readable_off_a_teacher_without_starting_a_game():
    """A score says whether a run moved; it does not say whether the policy moved, and the two have different fixes.

    A generation that scores the same as the one before it may be answering the same boards the same way, or may have moved a great deal of its mass and landed somewhere worth exactly as much. From a score alone the first reads as a learner that is stuck and the second as one looking in the wrong place. The tactical plateau was read as the first for weeks and is the second: over three generations the share of boards answered `close` falls from 49 to 46 to 35 per cent while `hold`, which is the top of the measured band, never moves off 5.

    So the distribution is readable directly, off the boards a collecting run already wrote down, with no game and no arena. What it reports is the likeliest action, because that is what a measurement run plays.
    """
    from rwintel.learn.imitation import chosen, teacher_shares

    # A separable teacher, for the reason the fit's own floor test uses one: what is being checked is the
    # reading, so the thing being read has to be something a fit is known to recover.
    draw = random.Random(11)
    samples = []
    for _ in range(400):
        action = draw.randrange(TACTICAL_ACTIONS)
        state = [draw.uniform(-0.2, 0.2) for _ in range(TACTICAL_SIZE)]
        state[action] = 1.0
        samples.append(Sample(state=state, action=action))
    share, second = teacher_shares(samples, TACTICAL)
    assert abs(sum(share) - 1.0) < 1e-9 and not second
    assert len(share) == TACTICAL_ACTIONS

    net = TacticalNet()
    row = chosen(samples, TACTICAL, net, "fresh")
    assert row.name == "fresh"
    assert abs(sum(row.share) - 1.0) < 1e-9, "every board is answered by exactly one action"
    assert 0.0 <= row.agreement <= 1.0 and row.entropy > 0.0
    assert not row.second_share, "the tactical layer chooses one thing"

    # A policy fitted to the teacher answers the teacher's own boards the teacher's way, which is what makes a
    # departure from it later readable as the reinforcement having moved something.
    fitted, _ = fit(samples, TACTICAL, net=TacticalNet(), seed=7, epochs=60, smoothing=0.0, patience=60)
    after = chosen(samples, TACTICAL, fitted, "fitted")
    assert after.agreement > 0.95, "the fit recovers a separable teacher, so the reading of it has to say so"
    assert after.entropy < row.entropy, "fitting a deterministic teacher narrows the distribution"
    # And the distribution it reports is the teacher's own, which is what makes a later generation's departure from it readable.
    assert max(abs(a - b) for a, b in zip(after.share, share)) < 0.05


def test_a_teacher_whose_slots_kept_their_names_and_changed_what_they_hold_is_refused_too():
    """The half of the claim the names cannot make, on the side where it is least visible.

    Every state in a teacher is a row of numbers already computed, so nothing below the head can be re-derived and compared with anything: if a slot kept its name and changed what it holds, the head is identical, the row lengths are identical, and a fit to it reports a perfectly ordinary accuracy for a network reading the board wrongly. That happened — the strategic cut compared this side's army with the enemy's whole side until it was corrected to armies against armies, and the teacher collected fifteen minutes before the correction states exactly the list it states today.

    So a head states the recipe beside the names, and it is the same digest a set of parameters carries: the two go stale together. A head stating none has to be collected again, and there is no avowal here as there is for parameters, because a teacher is a recording of a rule that is still standing in this tree while a set of parameters is the output of a run that is not.
    """
    decision = {"state": [0.0] * STRATEGIC_SIZE, "action": 0}
    named = {"layer": "strategy", "encoding": list(STRATEGIC_FEATURES)}
    with tempfile.TemporaryDirectory() as folder:
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, named, decision), "strategy")
        assert "not the recipe" in refusal and "collected again" in refusal

        moved = dict(named, recipe="0123456789abcdef")
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, moved, decision), "strategy")
        assert "different recipe" in refusal and STRATEGIC_RECIPE in refusal

        whole = dict(named, recipe=STRATEGIC_RECIPE)
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, whole, decision), "strategy")
        assert "which rule chose the actions" in refusal and "collected again" in refusal

        other = dict(whole, rule="0123456789abcdef")
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, other, decision), "strategy")
        assert "different rule" in refusal and rule_recipe(STRATEGIC) in refusal

        both = dict(whole, rule=rule_recipe(STRATEGIC))
        assert len(read_teacher(_teacher(folder, both, decision), "strategy")) == 1


def test_an_operational_teacher_is_refused_by_the_block_that_moved_and_not_by_a_slot_of_its_state():
    """The operational feature list does not name the state slot by slot and a refusal must not talk as though it did.

    Its state is a handful of aggregates, then one fixed block repeated over twenty-four region slots, then another over eight squad slots — four hundred numbers, described by forty-odd names because each block is named once with the slot counts alongside. So the index in a refusal is an index into the blocks and the length in one is a count of blocks, and reporting either as features would send somebody looking through a four-hundred-wide vector for a forty-fifth slot that decides nothing. The tactical list is the other case and the plain word is exact there, since it names one number of the state apiece; both are checked here so that neither wording can be changed to the other's without this failing.
    """
    moved = "region.distance"
    assert moved in OPERATIONAL_FEATURES, "the block this test renames has itself been renamed"
    decision = {"state": [0.0] * OPERATIONAL_SIZE, "action": 0, "second": 0}
    head = {"layer": "operations", "encoding": list(OPERATIONAL_FEATURES),
            "recipe": OPERATIONAL_RECIPE, "rule": rule_recipe(OPERATIONAL)}

    with tempfile.TemporaryDirectory() as folder:
        renamed = dict(head, encoding=["region.range" if name == moved else name
                                       for name in OPERATIONAL_FEATURES])
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, renamed, decision), "operations")
        index = OPERATIONAL_FEATURES.index(moved)
        assert f"block {index}" in refusal and "'region.range'" in refusal and f"'{moved}'" in refusal
        assert f"feature {index}" not in refusal, (
            "a block of the operational list was reported as a feature of its state: " + refusal)

        # The last entry of the operational list is the slot counts, so a list without it agrees name for name and differs only in length, which is the other branch of the refusal.
        shorter = dict(head, encoding=list(OPERATIONAL_FEATURES[:-1]))
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, shorter, decision), "operations")
        said = refusal.split(" line 1: ", 1)[-1]
        assert f"of {len(OPERATIONAL_FEATURES) - 1} blocks" in said
        assert str(OPERATIONAL_SIZE) not in said, (
            "the width of the state was quoted as though it were the length of the list: " + said)

        # And the tactical list, whose names are one per number, is quoted as features.
        head = {"layer": "tactics", "encoding": list(TACTICAL_FEATURES), "recipe": TACTICAL_RECIPE,
            "rule": rule_recipe(TACTICAL)}
        renamed = dict(head, encoding=["target_dx" if name == "target_ahead_of_home" else name
                                       for name in TACTICAL_FEATURES])
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, renamed, {"state": [0.0] * TACTICAL_SIZE, "action": 0}))
        assert f"feature {TACTICAL_FEATURES.index('target_ahead_of_home')}" in refusal


def test_two_collecting_runs_joined_into_one_teacher_are_read_and_re_checked_at_the_join():
    """The only way a teacher file gets bigger is by joining two of them, since the collecting run opens its file with truncation and cannot append. A joined file therefore carries the second run's head in the middle of it, and what happens at that line decides whether joining is a thing anybody can do.

    Three things happen there. The join reads back whole, which is what it did before a file stated anything and has to go on doing. The second head is CHECKED rather than waved through, because it is the only line in the file that says whether the two halves were collected under the same encoding — a join across an encoding change is two different readings of the board in one file, and fitting to it produces a network that is wrong about half of what it saw. And a line that is neither a decision nor a head is refused by name and line like everything else malformed here, rather than surfacing as a missing key from somewhere inside the reader with nothing said about which file it came from.
    """
    head = {"layer": "tactics", "encoding": list(TACTICAL_FEATURES), "recipe": TACTICAL_RECIPE,
            "rule": rule_recipe(TACTICAL)}
    decision = {"state": [0.0] * TACTICAL_SIZE, "action": 0}
    with tempfile.TemporaryDirectory() as folder:
        joined = _teacher(folder, head, decision, decision, head, decision, decision)
        assert len(read_teacher(joined)) == 4, "a teacher joined from two collecting runs lost decisions"

        renamed = dict(head, encoding=["target_dx" if name == "target_ahead_of_home" else name
                                       for name in TACTICAL_FEATURES])
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, head, decision, renamed, decision))
        assert "line 3" in refusal and "'target_dx'" in refusal, (
            "a join across an encoding change was not caught at the line that states the second encoding")

        # A line that is neither, which is what the head of another layer's teacher looks like from here.
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, head, decision, {"layer": "tactics", "count": 2}))
        assert "line 3" in refusal and "teacher.jsonl" in refusal and "'state'" in refusal

        # And a run killed while it was writing, whose last line stops in the middle.
        path = _teacher(folder, head, decision)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"state": [0.0, 0.0')
        refusal = _refusal(TeacherMismatch, read_teacher, path)
        assert "line 3" in refusal and "half finished" in refusal


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
    assert moved == {name for name, _ in net.named_parameters()}, sorted(moved)


def test_the_trainer_thread_finishes_without_shadowing_the_threads_own_shutdown():
    """A run ends by joining the trainer thread, and the join is where the collection this exercises would fail.

    The Trainer is a Thread, and the standard library calls Thread._stop on itself from inside join, the instant the thread has ended, through _wait_for_tstate_lock. An instance attribute named _stop shadows that method, so the join tries to call it, and if it is anything other than a method — an Event, say — the join raises TypeError rather than returning. That is exactly the shape of finish: it sets its stop flag and then joins, so the failure lands after the last episode has been fought and before the trained parameters have been saved, which is the most expensive place in the whole run for it to land. This starts a trainer over a rollout with finished work in it, lets it run, and finishes it, which is the sequence a training run performs on its way out.
    """
    torch.manual_seed(0)
    net = TacticalNet()
    optimiser = Optimiser(net)
    rollout = Rollout()
    draw = random.Random(0)
    for trajectory in range(4):
        for index in range(8):
            rollout.add(trajectory, Step(state=[draw.uniform(-1.0, 1.0) for _ in range(TACTICAL_SIZE)],
                                         action=draw.randrange(TACTICAL_ACTIONS),
                                         mask=[1.0] * TACTICAL_ACTIONS, log_prob=-1.6, value=0.1,
                                         reward=1.0 if index % 2 else -1.0, done=index == 7))
    trainer = Trainer(rollout, optimiser, batch=8)
    trainer.start()
    # finish() sets the stop flag and joins; the join is the call that raised before the fix.
    report = trainer.finish()
    assert not trainer.is_alive()
    # The last partial batch is spent on the way out, so at least one update was taken over the finished work.
    assert report is not None and report.updates >= 1


# ---- the widened tactical action space ------------------------------------------------------

def test_the_handwritten_layer_reaches_the_two_added_departures():
    """The action space grew from five departures to seven, and the two added ones carry a choice a rule on the game side used to make: how far a withdrawal commits and which enemy a concentration goes onto. The imitation clones the handwritten layer and the reinforcement is measured against it, so the wider space is only worth training on if the handwritten layer actually reaches the two: a withdrawal is the whole way out when the squad is reported losing and a short step otherwise, and a concentration goes onto a longer-ranged enemy in reach and onto the weakest when none is."""
    tactics = Tactics(None, _CATALOGUE)

    def departure(view, squad):
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


def test_the_handwritten_layer_stands_rather_than_closing_when_it_cannot_shoot_back():
    """The departure the ladder was given for this board, and then measured out of it again.

    The engine halts a unit when it acquires a target and acquisition happens at sight range, while shooting needs weapon range, so two forces walking at each other come to rest in the gap between the two. A squad that outranges what is shooting at it keeps that gap, which is kiting. A squad that is outranged stands in it and is shot at without ever firing, and the argument for closing was that nothing else in the set answers that: withdrawing abandons the errand, holding leaves the engine's own halt in force, and concentrating is refused outright because the rule that picks a target only counts enemies already inside our reach.

    The argument was right about the board and wrong about the answer. The ladder against itself with this branch taken away on one side stands +0.0675 with two standard errors of 0.0254 over 1,081 paired fights, where claiming that difference takes 153: walking in on a squad that is outranged is walking into the fire it could not answer, and standing there is cheaper. So the ladder answers this board with holding, which is what it answered before the departure existed, and the departure stays in the action space for a layer that can tell where it is right.
    """
    tactics = Tactics(None, _CATALOGUE)

    def departure(view, squad, layer=None):
        return (layer or tactics)._departure(squad, [s for s in view.ours], [s for s in view.enemies],
                                             squad.losses, _Track())

    ours = [_unit(1, 100, 100), _unit(2, 120, 100), _unit(3, 140, 100)]
    place = [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)]
    # Our tanks reach 130 and the gun reaches 320, and it stands 200 out from the squad's centre: it can shoot and we cannot.
    outranged = _view(ours + [_unit(9, 320, 100, type_index=1, hostile=1)], place)
    assert departure(outranged, _squad(status=Status.ACTIVE, losses=0.0)) == Deviation.HOLD
    # And the move itself is still in the space, which is what the pinned arm and every learnt layer reach for.
    assert Deviation.CLOSE in tuple(Deviation)

    # The same fight with the gun inside our own reach is a fight we can answer where we stand, and the ladder answers it as it always did.
    engaged = _view(ours + [_unit(9, 220, 100, type_index=1, hostile=1),
                            _unit(10, 200, 130, type_index=0, hostile=1)], place)
    assert departure(engaged, _squad(status=Status.ACTIVE, losses=0.0)) == Deviation.FOCUS_THREAT

    # And a squad that outreaches what is shooting at it keeps the gap rather than closing it.
    kiting = _view([_unit(1, 100, 100, type_index=1), _unit(2, 120, 100, type_index=1)]
                   + [_unit(9, 300, 100, type_index=0, hostile=1)], place)
    assert departure(kiting, _squad(members=(1, 2), status=Status.ACTIVE, losses=0.0)) == Deviation.KITE


def test_the_handwritten_ladder_decides_a_whole_side_as_it_decides_one_squad():
    """A period is read squad by squad and then answered in one call for the whole side, which is the seam a learnt layer replaces so that a side of several squads costs one batching window rather than one window each. The handwritten ladder has no decider and must be entirely unaffected by that: it is the baseline every measurement in this project is taken against, so a change to what it decides would invalidate all of them.

    This is the golden statement of that. One board, five squads landing on five different rungs at once, plus the two kinds of squad the layer reports on and does not order: one whose tactical command a human holds, and one with nothing of it left on the board. What is pinned is the exact list of deviations in squad order, that the two unordered squads appear among the reports and in no deviation, and that a report is filed for every squad in the order the squads were handed in.
    """
    import dataclasses

    from rwintel.control.policy.tactics import HUMAN_TACTICS
    from rwintel.wire.observation import SquadState

    units = []
    squads = []

    def band(index, members, hit, enemy_type, enemy=True, **overrides):
        """One squad well away from every other, so that no squad's threats are another's."""
        base = index * 3000.0
        ids = [10 * index + n for n in range(members)]
        for offset, unit_id in enumerate(ids):
            units.append(_unit(unit_id, base + 100.0 + 20.0 * offset, 100.0, type_index=enemy_type[0],
                               hit=hit))
        if enemy:
            units.append(_unit(900 + index, base + 210.0, 110.0, type_index=enemy_type[1], hostile=1))
        squads.append(_squad(id=index, members=ids, x=base + 120.0, y=100.0, contract=True,
                             **overrides))

    # Reported losing and heavily spent, which is the whole way out of the fight rather than a step back.
    band(0, 3, 100, (0, 0), status=Status.LOSING)
    # Spent against its budget but not reported losing, which is the short step back.
    band(1, 3, 100, (0, 0), status=Status.ACTIVE)
    # Bunched, freshly hit, and an artillery in range: one weapon covering the squad.
    band(2, 3, 100, (0, 1), status=Status.ACTIVE, spread=60.0)
    # Artillery of our own out-reaching the tank shooting at it, which is worth backing away while firing.
    band(3, 3, 9999, (1, 0), status=Status.ACTIVE)
    # Nothing shooting and nothing near, which is the quiet case: the squad is still ordered, and what it is ordered is to hold.
    band(4, 3, 9999, (0, 0), enemy=False, status=Status.ACTIVE)
    # A human is driving these units, so the layer reports and orders nothing.
    band(5, 3, 100, (0, 0), status=Status.ACTIVE, commander=HUMAN_TACTICS)
    # Listed members, none of them on the board.
    squads.append(_squad(id=6, members=[70, 71], x=18000.0, y=100.0))

    # The losses each mission has cost, which is what puts the first two squads over their budgets and leaves the rest under theirs.
    spent = {0: 900.0, 1: 900.0}
    rows = [SquadState(id=squad.id, commander=squad.commander, units=len(squad.members),
                       value=squad.value, formed_value=squad.formed_value, x=squad.x, y=squad.y,
                       spread=squad.spread, task_type=0, stance=0, target_region=1,
                       status=int(squad.status), cost_budget=1000.0, budget_share=0.5,
                       deadline_ms=90000, issued_at_ms=20000, losses=spent.get(squad.id, 0.0))
            for squad in squads]
    observation = dataclasses.replace(
        _observation(units, [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)]), squads=rows)
    view = build_view(observation, _CATALOGUE, None)

    deviations, reports = Tactics(None, _CATALOGUE).decide(view, squads, 21000)
    assert [(d.squad, d.deviation) for d in deviations] == [
        (0, Deviation.WITHDRAW_FAR),
        (1, Deviation.WITHDRAW),
        (2, Deviation.SPREAD),
        (3, Deviation.KITE),
        (4, Deviation.HOLD),
    ]
    # Reported on but never ordered, which is the one asymmetry between the two lists.
    assert [report.squad for report in reports] == [0, 1, 2, 3, 4, 5, 6]
    assert 5 not in [d.squad for d in deviations] and 6 not in [d.squad for d in deviations]


# ---- which side of a fight is decided first --------------------------------------------------

class _Recorder:
    """A tactical layer that answers the same way whatever it is shown and writes down that it was asked, so that what is under test is the order the two sides of a fight are decided and submitted in."""

    def __init__(self, name: str, calls: list) -> None:
        self.name, self.calls = name, calls

    def decide(self, view, squads, game_time_ms):
        self.calls.append(self.name)
        #: The last board this side was handed, which is how a test can ask what a layer was actually shown rather than what the arena meant to show it.
        self.board = view
        return [SquadDeviation(squad=squad.id, deviation=Deviation.HOLD) for squad in squads], []


def _arena_at(order: str, calls: list):
    """An arena far enough built to fight one period of one engagement, without a session, a map or a game.

    Assembled field by field rather than constructed, because everything the constructor does — the type catalogue, the two layers, the sandbox — needs a live session, and none of it bears on which of two already-built layers is asked first.
    """
    from rwintel.learn.arena import Arena, OURS, THEIRS, Statistics

    arena = Arena.__new__(Arena)
    arena.catalogue, arena.statistics = _CATALOGUE, Statistics()
    arena.decision_order, arena.stall_ms, arena.outcome_weight = order, 12000, 1.0
    arena.tactics, arena.opponent = _Recorder("ours", calls), _Recorder("theirs", calls)
    arena.squads = {OURS: _squad(members=(1, 2, 3)), THEIRS: _squad(members=(9,))}
    arena.squads[THEIRS].id = THEIRS
    arena.engagement, arena.last_regions = None, []
    arena.until_ms, arena._alive, arena._changed_ms = 10 ** 9, 4, 0
    arena._balance = {}
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


def test_the_ladder_with_one_branch_taken_away_falls_through_rather_than_answering_it():
    """The other ablation, and a different question from pinning.

    Pinning asks what a departure is worth as a constant policy, over every board there is. Withholding asks what it is worth INSIDE the ladder, on the boards the ladder's own tests send to it. The eighth departure is where the two come apart: pinned, closing loses 0.044 against the ladder, and the ladder answers with it on 49 per cent of its decisions, so either its condition for it is wrong or the boards it fires on are ones where closing is right and the constant arm is dragged down by the boards it is not. Nothing before this could tell those apart.

    What a withheld branch does is fall through to the next test, which is exactly what the ladder did before that departure existed. Anything else would be a second change measured as one.
    """
    from rwintel.learn.__main__ import _withheld

    ours = [_unit(1, 100, 100), _unit(2, 120, 100), _unit(3, 140, 100)]
    place = [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)]
    # A gun inside our own reach with a tank beside it: a fight this squad can answer where it stands.
    board = _view(ours + [_unit(9, 220, 100, type_index=1, hostile=1),
                          _unit(10, 200, 130, type_index=0, hostile=1)], place)
    squad = _squad(status=Status.ACTIVE, losses=0.0)

    def departure(layer):
        return layer._departure(squad, list(board.ours), list(board.enemies), squad.losses, _Track())

    whole = LearntTactics(None, _CATALOGUE, None, rollout=None, instance=0)
    answered = departure(whole)
    assert answered is Deviation.FOCUS_THREAT

    lessened = LearntTactics(None, _CATALOGUE, None, rollout=None, instance=0, withhold=(answered,))
    assert departure(lessened) != answered, "the branch that answered was not taken away"
    assert whole.withhold == frozenset() and lessened.withhold == {answered}

    # Taking away every departure leaves the ladder at the one that is not a departure at all.
    nothing = LearntTactics(None, _CATALOGUE, None, rollout=None, instance=0, withhold=tuple(Deviation))
    assert departure(nothing) is Deviation.HOLD

    class _Asked:
        without = "spread, kite"

    assert _withheld(_Asked()) == [Deviation.SPREAD, Deviation.KITE]
    _Asked.without = None
    assert _withheld(_Asked()) == []
    _Asked.without = "sidestep"
    assert "sidestep" in _refusal(SystemExit, _withheld, _Asked())

    # A departure the ladder has no branch for is refused rather than run, because the arm it would build decides
    # exactly what the baseline decides and would report a difference of nought as that departure's contribution
    # inside the ladder. Closing has had no branch since it was measured out of the ladder, and holding is what the
    # ladder falls through to rather than a branch of it; both are still pinnable, which is the other ablation.
    for asked in ("close", "hold"):
        _Asked.without = asked
        assert "no branch" in _refusal(SystemExit, _withheld, _Asked())


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
    """A run reports the mean and the spread of its fights, and it holds neither: an episode record carries its own count, mean and spread, and the run is put back together from those. That is what lets a training run be read for the fights it has already scored — three times as many as the duel that measured the same policy — and it has to give exactly what the flat list of fights would.

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


def test_a_learnt_operational_arm_loads_and_names_itself_after_its_file():
    """The operational layer is measured on whole matches, so its arm is built for the match runner rather than the arena: a network loaded once, a batching server that answers every instance's decisions through it, and the rest of the chain left the script it is measured against. The server is handed back for the run to stop, because a (name, build) pair has nowhere to keep it, and the arm is named after its file so that the number carries what produced it."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "operations.pt")
        torch.save(OperationalNet().state_dict(), path)
        arms, batchers = eval_arms.build_all(["script", f"ops:{path}"])
        try:
            assert [name for name, _ in arms] == ["script", "operations"]
            assert len(batchers) == 1
        finally:
            for batcher in batchers:
                batcher.stop()


def test_the_separation_a_fight_is_put_down_at_runs_as_arms_like_the_other_two_draw_settings():
    """Three settings decide how a fight is drawn — the imbalance floor, the stall time and the separation — and two of them could be swept as arms of one run while the third took a single figure for the whole run. The odd one out was the separation, which is the one the record names as the first thing to sweep: the arena's own 250 sits inside the gap the engine halts two converging forces in, so both sides are exchanging fire from the first period and the departures have the least of the fight left to decide.

    Run as arms it is also a paired comparison, which is the reason it is worth doing this way rather than as two runs. The separation enters the placement alone and never touches the arena's own random stream, so two arms under one seed draw the same site, the same two budgets, the same imbalance, the same angle and the same two forces: the same fight, begun from a different distance. Each separation still carries its own baseline, because what the arena leans by is a property of the instrument and a wider board is a different instrument.
    """
    from rwintel.learn.__main__ import _separations

    class _Asked:
        separation = "250, 700"

    assert _separations(_Asked()) == [250.0, 700.0]
    # Named once so a plain run keeps the names the journal has always carried, and twice over so a sweep says which arm was which.
    _Asked.separation = "700"
    assert _separations(_Asked()) == [700.0]
    _Asked.separation = None
    assert _separations(_Asked()) == [None], "unasked for, the arena's own figure stands"
    _Asked.separation = "0"
    assert "0" in _refusal(SystemExit, _separations, _Asked())
    _Asked.separation = "wide"
    assert "wide" in _refusal(SystemExit, _separations, _Asked())

    # Two arms at two separations draw the same fight from different distances, which is what makes them pairable.
    from rwintel.learn.arena import Arena

    drawn = []
    for separation in (250.0, 700.0):
        with _without_the_game_installed():
            arena = Arena(_Chained(), seed=4242, separation=separation)
        drawn.append((arena.random.uniform(0.0, 1.0), arena.separation))
    assert drawn[0][0] == drawn[1][0], "the separation must not move the stream the forces are drawn from"
    assert drawn[0][1] == 250.0 and drawn[1][1] == 700.0


def test_both_seats_of_a_fight_read_their_own_mission_status():
    """The game side writes a squad's status out of the region force totals as THIS process sees them, so a region's enemy worth is whatever is hostile to us. Read off that block for both seats, completion could fire for our squad on the winning period of a fight and could never fire for the mirror squad at all — the only way for the ground it stands on to hold nothing hostile to us is for that squad itself to be dead, and a squad with nothing left reads ACTIVE.

    Status is five of the fifty-eight tactical features. Two seats that cannot read the same statuses are two seats that are not exchangeable, which is the property the self-play zero and every paired comparison rest on. So it is written here, off each side's own view, by one ladder.
    """
    from rwintel.learn.arena import OURS, OURS_FIRST, THEIRS
    from rwintel.wire import Status

    calls: list = []
    arena = _arena_at(OURS_FIRST, calls)
    for squad_id, squad in arena.squads.items():
        squad.contract = TaskContract(squad=squad_id, task=Task.ATTACK, target_region=1,
                                      stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                      deadline_ms=90000, issued_at_ms=0)
        squad.losses = 0.0

    # One board seen from the two seats: on ours the contested region holds no enemy worth and our squad stands on it;
    # the inverted view says exactly the same thing to the other seat about its own squad.
    ours = build_view(_observation(units=[_unit(1, 400.0, 100.0), _unit(2, 410.0, 100.0), _unit(3, 420.0, 100.0)],
                                   regions=[_region(1, 400.0, 100.0, ours=900.0, theirs=0.0)]), _CATALOGUE, None)
    theirs = build_view(_observation(units=[_unit(9, 400.0, 100.0, hostile=1)],
                                     regions=[_region(1, 400.0, 100.0, ours=0.0, theirs=900.0)]),
                        _CATALOGUE, None, invert=True)
    for squad in arena.squads.values():
        squad.x, squad.y = 400.0, 100.0

    arena._fold_status(ours, theirs, 1000)
    assert arena.squads[OURS].status == Status.COMPLETE
    assert arena.squads[THEIRS].status == Status.COMPLETE, (
        "the mirror seat cannot reach the status its own board says it is in")

    # And the ladder is the game's own: with the ground still contested, an allowance spent past its share reads LOSING.
    contested = build_view(_observation(units=[_unit(1, 400.0, 100.0), _unit(9, 405.0, 100.0, hostile=1)],
                                        regions=[_region(1, 400.0, 100.0, ours=900.0, theirs=900.0)]),
                           _CATALOGUE, None)
    arena.squads[OURS].contract = TaskContract(squad=OURS, task=Task.ATTACK, target_region=1,
                                               stance=Stance.AGGRESSIVE, cost_budget=100.0,
                                               deadline_ms=90000, issued_at_ms=0)
    arena.squads[OURS].losses = 100.0
    arena._fold_status(contested, theirs, 1000)
    assert arena.squads[OURS].status == Status.LOSING


def test_a_fight_is_drawn_under_a_contract_a_match_could_emit():
    """Every fight used to be ATTACK with AGGRESSIVE, which left thirteen of the tactical cut's fifty-eight features constant for the whole of training and moving in every match decision. The fight is drawn under a contract now, and the contracts it may be drawn under are the operational layer's own table read straight off: no pair can be drawn that the chain cannot issue.

    The draw comes out of the fight's own stream, which is what makes two arms of a paired run answer the same contracts on the same fights. The geometry is untouched — the target is the other side's ground whatever the task — because the engine's default under any contract is to advance on the target with the contract's stance, so a fight drawn under a guarding or a withholding contract is still a fight.
    """
    import random as _random

    from rwintel.control.policy.operations import STANCE_FOR
    from rwintel.learn.arena import CONTRACTS

    # The table itself, not a copy of it: a pair the arena can draw is a pair the chain can issue.
    assert dict(CONTRACTS) == STANCE_FOR and len(CONTRACTS) == len(STANCE_FOR)
    assert (Task.ATTACK, Stance.AGGRESSIVE) in CONTRACTS
    assert (Task.WITHDRAW, Stance.HOLD_FIRE) in CONTRACTS

    # Drawn out of the paired stream, so two arms at the same seed draw the same sequence of contracts.
    first = [_random.Random(4242).choice(CONTRACTS) for _ in range(1)]
    second = [_random.Random(4242).choice(CONTRACTS) for _ in range(1)]
    assert first == second

    # And over many draws every one of them turns up, which is the whole point: the layer has to have seen all of them.
    stream = _random.Random(7)
    seen = {stream.choice(CONTRACTS) for _ in range(400)}
    assert seen == set(CONTRACTS)


def test_an_engagement_episode_says_how_its_fights_were_drawn():
    """Two runs drawn under different settings are two different instruments, and a comparison that pooled them would read the change of instrument as a difference between the arms. The constructed operations arena writes its draw into every episode for that reason; the engagement arena wrote none of its own, so a journal of it could not say what it was measuring.

    The separation is the sharpest of the four. It is how far apart the two sides are put down, and the arena's own figure is deliberately inside the gap the engine halts two converging forces in — which is what makes a fight begin at all, and also what leaves the departures a small share of the fight to decide. It has to be movable to ask how much of the score the tactical choice can move, and once it is movable it has to be written down.
    """
    from rwintel.learn.arena import IMBALANCE, SEPARATION, STALL_MS, Arena, Statistics

    plain = Statistics().as_dict()
    assert plain["separation"] == SEPARATION and plain["stall_ms"] == STALL_MS
    assert plain["imbalance_floor"] == IMBALANCE[0] and plain["score"] == BY_HEALTH

    # Written at construction rather than at scoring, so an episode that produced no fight at all still says what it was run under.
    with _without_the_game_installed():
        arena = Arena(_Chained(), separation=600.0, stall_ms=8000, imbalance_floor=0.3, score=BY_KILLS)
    drawn = arena.statistics.as_dict()
    assert drawn["separation"] == 600.0 and drawn["stall_ms"] == 8000
    assert drawn["imbalance_floor"] == 0.3 and drawn["score"] == BY_KILLS
    assert arena.separation == 600.0, "the figure the fights are actually placed at is the one asked for"


class _Chained:
    """A session as far as the whole script chain reads one when it is built: the type table, no asset tree, no regions and no map. The chain's constructor makes its own catalogue out of the first two and the economy layer works its resource points out of the last, which is everything the five layers ask of a session before a board has arrived."""

    types = list(_TYPES)
    assets = None
    regions = ()
    map_content = None


@contextlib.contextmanager
def _without_the_game_installed():
    """Builds a real chain without the game's definition files under it.

    A catalogue reads which building produces which type out of the files the engine itself loads, and with no asset tree named it looks for the master copy. That is right in a run and wrong in this suite: everything here is meant to pass on a machine that has never had the game on it, and a test that quietly needs the install passes where it was written and fails where it is read. The links are what a test of the layers does not use, so they are handed back empty.
    """
    from rwintel.control.policy import catalogue

    original = catalogue._build_links
    catalogue._build_links = lambda assets: ({}, {})
    try:
        yield
    finally:
        catalogue._build_links = original


def test_a_learnt_strategic_arm_is_built_the_same_way_and_can_be_read_either_way():
    """The strategic layer is measured on whole matches like the operational one, through the same wiring: one network read off a file, one batching server, and the rest of the chain left the script it is measured against.

    How the policy is read is the run's to say. Drawing is the default because every match measurement so far was taken that way, and `--greedy` makes every learnt arm take its likeliest action instead — which is how the arena's measuring runner reads its own learnt arm, so the flag is what makes a number from one place comparable with a number from the other.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "strategy.pt")
        torch.save(StrategicNet().state_dict(), path)
        arms, batchers = eval_arms.build_all(["script", "defend", f"strategy:{path}"])
        try:
            assert [name for name, _ in arms] == ["script", "defend", "strategy"]
            assert len(batchers) == 1, "only the learnt arm needs an inference server"
        finally:
            for batcher in batchers:
                batcher.stop()

        # A freshly built policy is nearly uniform on purpose, so a server that drew from it would not answer alike twenty times running; one that takes the likeliest posture will.
        request = ([0.0] * STRATEGIC_SIZE, [1.0] * len(Posture))
        for greedy, alike in ((True, True), (False, False)):
            _, batcher = eval_arms.learnt(f"strategy:{path}", greedy=greedy)
            try:
                answers = {batcher.submit(request).action for _ in range(20)}
            finally:
                batcher.stop()
            assert (len(answers) == 1) is alike


def test_a_layer_frozen_beneath_the_one_being_trained_records_nothing():
    """The other half of what the design means by learning one layer against frozen neighbours.

    A training run has always frozen the script beneath itself, which is what a `LearningPolicy` is; what had no wiring at all in a match was freezing the TRAINED layers beneath it, so the second half of the learning order could only ever be run against the handwritten layers it was supposed to have improved on. A frozen layer is built by the very same code as the layer being trained and differs in exactly one thing: it is handed no rollout, so with nowhere to record a decision it records none.

    Freezing the layer the run is training is refused rather than allowed to mean something: it would leave the gradient going to a network nothing reads.
    """
    session = _Chained()
    training, held = _Fixed(action=0), _Fixed(action=1)
    rollout = Rollout()
    with _without_the_game_installed():
        policy = LearningPolicy(session, STRATEGIC, training, rollout, 0, frozen={TACTICAL: held})
    assert isinstance(policy.strategy, LearntStrategy) and policy.strategy.rollout is rollout
    assert isinstance(policy.tactics, LearntTactics) and policy.tactics.rollout is None
    assert policy.tactics.decider is held and policy.strategy.decider is training
    # The layers not named are the script's own, unchanged.
    assert type(policy.operations).__name__ == "Operations"

    with _without_the_game_installed():
        assert "cannot be both trained and frozen" in _refusal(
            ValueError, LearningPolicy, session, STRATEGIC, training, rollout, 0, {STRATEGIC: held})


def test_freezing_a_layer_names_the_parameters_and_refuses_what_it_cannot_read():
    """A frozen layer is part of the instrument, so it is named by a digest of the parameters themselves rather than by the path they came from — a path is a nickname that a later training run overwrites. What cannot be read is refused before any inference thread exists, for the reason the arena refuses a mistyped tactical path: an instrument built from noise is a whole run measured against nothing and journalled under a trained layer's name."""
    threads = threading.active_count()
    with tempfile.TemporaryDirectory() as directory:
        tactics = os.path.join(directory, "tactics.pt")
        operations = os.path.join(directory, "operations.pt")
        torch.save(TacticalNet().state_dict(), tactics)
        torch.save(OperationalNet().state_dict(), operations)

        frozen = frozen_layers(f"tactics:{tactics},operations:{operations}", training=STRATEGIC)
        try:
            assert sorted(frozen.deciders) == ["operations", "tactics"]
            assert len(frozen.batchers) == 2
            assert all(name.startswith("sha256:") for name in frozen.names.values())
            assert frozen.names["tactics"] != frozen.names["operations"]
            built = frozen.build()
            assert sorted(built) == ["operations", "tactics"] and built["tactics"].greedy
        finally:
            frozen.stop()

        assert "there are no parameters at" in _refusal(
            ValueError, frozen_layers, f"tactics:{os.path.join(directory, 'absent.pt')}")
        assert "cannot be frozen as the tactics layer" in _refusal(
            ValueError, frozen_layers, f"tactics:{operations}")
        assert "was named twice" in _refusal(
            ValueError, frozen_layers, f"tactics:{tactics},tactics:{tactics}")
        assert "layer:path" in _refusal(ValueError, frozen_layers, tactics)
        # Nothing is left running behind any of those refusals.
        assert threading.active_count() == threads


def test_a_missing_operational_file_is_refused_rather_than_started_from_nothing():
    """A duel refuses to measure parameters that are not there rather than starting from a fresh policy, because a plausible number about a policy nobody asked about is worse than an error. The match arm refuses for the same reason, and before it stands up any inference thread."""
    try:
        eval_arms.build_all(["ops:local/does-not-exist.pt"])
    except ValueError:
        return
    raise AssertionError("a missing operational file was not refused")


def test_two_arms_of_one_name_are_refused():
    """Journalled and reported under one name, two arms of a comparison merge into one and the run silently measures half of what it was asked for; two learnt arms loaded from the same file would share the file's name, so the pair is refused."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "operations.pt")
        torch.save(OperationalNet().state_dict(), path)
        try:
            _, batchers = eval_arms.build_all([f"ops:{path}", f"ops:{path}"])
        except ValueError:
            return
        for batcher in batchers:
            batcher.stop()
    raise AssertionError("two arms of one name were not refused")


def test_a_later_arm_failing_stops_the_servers_already_started():
    """A later arm failing must not leave an earlier learnt arm's inference thread running against a run that will never start. The failure has to reach the caller, and the earlier server has to be stopped on the way out."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "operations.pt")
        torch.save(OperationalNet().state_dict(), path)
        try:
            eval_arms.build_all([f"ops:{path}", "ops:local/does-not-exist.pt"])
        except ValueError:
            return
    raise AssertionError("a failure after a learnt arm was built did not raise")


def test_a_pinned_operational_arm_needs_no_network_and_no_teardown():
    """The operational analogue of the arena's --pin: the layer pinned to one legal region and task, as a match arm. It loads no network, so build_all returns it with no batcher to stop. The floor it measures is what the region-and-task choice is worth at all: if a constant deployment scores the same as the script's careful one, the choice was not moving the match."""
    from rwintel.learn.deciders import PinnedRegion

    # The decider itself picks the lowest-numbered legal region and task, whatever board it is shown.
    choice = PinnedRegion().choose([0.0] * 5, 0, [0, 0, 1, 1, 0, 1], [0, 1, 1])
    assert choice.action == 2 and choice.second == 1
    # With nothing legal there is no deployment to make.
    assert PinnedRegion().choose([0.0], 0, [0, 0], [0, 0]) is None

    arms, batchers = eval_arms.build_all(["script", "ops-pin"])
    assert [name for name, _ in arms] == ["script", "ops-pin"]
    assert batchers == []


def test_script_and_posture_arms_still_build_without_a_tensor_library():
    """The two arms that were always here are unchanged and start no server: 'script' is the chain as it decides for itself, and a posture name pins the strategic layer. build_all returns them with no batchers to tear down."""
    arms, batchers = eval_arms.build_all(["script", "defend"])
    assert [name for name, _ in arms] == ["script", "defend"]
    assert batchers == []


# ---- the tactical layer frozen under the operations arena -----------------------------------

def test_naming_no_tactical_parameters_leaves_the_arena_its_handwritten_fighter():
    """The half of the option that has to cost nothing: no path, no file read, no network built, no inference thread started, and no factory handed to the arena, which then builds the handwritten ladder for itself on both sides. That is the layer every measurement taken on this arena so far was made under, and the name a journal writes for it is the word a comparison reads as exactly that."""
    threads = threading.active_count()
    frozen = frozen_tactics(None)
    assert frozen.build is None and frozen.batcher is None
    assert frozen.name == SCRIPT_TACTICS
    assert threading.active_count() == threads
    # Torn down the same way whether or not it started anything, so neither runner has to ask which case it is in.
    frozen.stop()


def test_a_frozen_tactical_layer_reads_greedily_records_nothing_and_shares_one_network():
    """What the runners put under both sides of the arena: one network behind one batching server, a fresh layer object per side, read at its likeliest departure, and handed no rollout at all.

    Each of those is load-bearing. One network is what makes every request from both sides of every instance meet in one batch, where two would halve the batch per call and buy nothing. Separate layer objects are forced by the per-side state a tactical layer keeps. Greedy on the batcher as well as on the decider, because the batcher's flag is what answers when there is a batcher and the decider's when there is not, and drawing is exploration this layer is not being asked for. No rollout, because the operational layer above records against the very same squad ids and a shared buffer would splice two action spaces into one trajectory.
    """
    torch.manual_seed(1234)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tactics.pt")
        torch.save(TacticalNet().state_dict(), path)
        frozen = frozen_tactics(path)
        try:
            assert frozen.build is not None and frozen.batcher is not None
            ours = frozen.build(None, _CATALOGUE)
            theirs = frozen.build(None, _CATALOGUE)
            assert isinstance(ours, LearntTactics) and isinstance(theirs, LearntTactics)
            assert ours is not theirs
            for layer in (ours, theirs):
                assert layer.rollout is None, "a layer that is read rather than learnt from must record nothing"
                assert layer.instance == -1
                assert isinstance(layer.decider, NetworkTactics)
                assert layer.decider.greedy and layer.decider.batcher is frozen.batcher
            assert ours.decider.net is theirs.decider.net, "the two sides are fighting under two networks"

            # The server itself answers greedily, which is the flag that decides on the live path. A freshly built policy is initialised nearly uniform on purpose, so a server that drew from it would not answer alike twenty times running.
            request = ([0.0] * TACTICAL_SIZE, [1.0] * TACTICAL_ACTIONS)
            answers = {frozen.batcher.submit(request).action for _ in range(20)}
            assert len(answers) == 1, "the inference server is drawing rather than taking the likeliest departure"

            # The name written into every episode is the parameters and not the path: the same file twice is one instrument, and different parameters are a different one however they are named on disk.
            again = frozen_tactics(path)
            try:
                assert again.name == frozen.name and frozen.name.startswith("sha256:")
            finally:
                again.stop()
            other = os.path.join(directory, "other.pt")
            torch.save(TacticalNet().state_dict(), other)
            different = frozen_tactics(other)
            try:
                assert different.name != frozen.name
            finally:
                different.stop()
        finally:
            frozen.stop()


def test_missing_tactical_parameters_are_refused_before_any_inference_thread_exists():
    """Refused rather than started from nothing, unlike the tolerant load a training run gives the layer it is about to train. Starting from a fresh policy is meaningful for the layer being learnt and never for the instrument beneath it: a mistyped path would leave a randomly initialised fighter under the whole arena, and the run would be journalled as having been made under trained parameters with nothing downstream able to tell.

    And refused first, before anything is built, so the failure leaves no inference thread running against a run that will never start.
    """
    threads = threading.active_count()
    try:
        frozen_tactics("local/there-are-no-tactical-parameters-here.pt")
    except SystemExit as refusal:
        assert "there-are-no-tactical-parameters-here.pt" in str(refusal)
        assert threading.active_count() == threads
        return
    raise AssertionError("a missing tactical file was not refused")


def test_another_layers_parameters_handed_to_the_arena_are_refused_by_name():
    """An operational network's file carries a `body.0.weight` too, so what tells the two apart is the width of the input the first layer reads. Handed the wrong file, the run says which file, how many features it found and how many a tactical layer reads, rather than failing somewhere inside the tensor library where the answer is a shape mismatch and the question is which flag was mistyped."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "operations.pt")
        torch.save(OperationalNet().state_dict(), path)
        found = int(torch.load(path, map_location="cpu")["body.0.weight"].shape[1])
        try:
            frozen_tactics(path)
        except SystemExit as refusal:
            message = str(refusal)
            assert path in message and str(found) in message and str(TACTICAL_SIZE) in message
            return
        raise AssertionError("an operational network was accepted as a tactical one")


def test_parameters_fitted_to_a_different_feature_list_are_refused_by_the_feature_that_moved():
    """The failure no width can catch, and the one that matters most where a layer is frozen: the features were renamed rather than renumbered, so every shape in the file still fits and the network would read two slots as something they are no longer.

    The file therefore carries the feature list it was fitted to among its parameters, as a buffer, so that it is saved and loaded by the ordinary means and cannot be copied away from them. Two files are refused here: one fitted to a list in which a feature has a different name, which is named in the refusal so a person is told what moved rather than that something did; and one carrying no list at all, which is every set of parameters written before the list was recorded and which cannot be shown to read the features now in force.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "stale.pt")
        state = TacticalNet().state_dict()
        state[ENCODING_KEY] = encoding_stamp(
            ["target_dx" if name == "target_ahead_of_home" else name for name in TACTICAL_FEATURES],
            TACTICAL_RECIPE)
        torch.save(state, path)
        refusal = _refusal(SystemExit, frozen_tactics, path)
        assert path in refusal and "'target_dx'" in refusal and "'target_ahead_of_home'" in refusal

        older = os.path.join(directory, "unstamped.pt")
        state = TacticalNet().state_dict()
        del state[ENCODING_KEY]
        torch.save(state, older)
        refusal = _refusal(SystemExit, frozen_tactics, older)
        assert older in refusal and "no feature list" in refusal

        # And an operational file is refused the same way by the match runner that measures one.
        operational = os.path.join(directory, "operations.pt")
        state = OperationalNet().state_dict()
        del state[ENCODING_KEY]
        torch.save(state, operational)
        assert "no feature list" in _refusal(ValueError, eval_arms.build_all, [f"ops:{operational}"])
        # The refusal is also where the way out is named, because a person holding parameters that cost a training run to make will otherwise conclude from it that they have to make them again.
        assert "avow" in _refusal(ValueError, eval_arms.build_all, [f"ops:{operational}"])


def test_a_slot_that_changed_what_it_holds_retires_the_parameters_fitted_to_the_old_holding():
    """The failure the feature list itself could not catch, measured on the case that happened.

    The strategic cut compared this side's fighting strength with the enemy's WHOLE side — buildings and builders counted into theirs and out of ours — and the day it was corrected to fighters against fighters, the slot stayed named `military_edge`, the list stayed identical byte for byte, and every width stayed where it was. The imitation the strategic layer was bootstrapped from had been fitted fifteen minutes earlier. It loaded in silence onto a quantity that had moved beneath it, and nothing in the file, the loader or the run could have said so.

    So a stamp states two things: what the slots are CALLED, and a digest of the code that FILLS them. A file whose recipe disagrees is refused for good, exactly as one whose list disagrees is. A file that states a list and no recipe was written while the stamp carried only names, and is refused until a person avows it — the same shape of answer a file with no stamp at all already got, for the same reason: it cannot prove what it was fitted to, and refusing is not the same as knowing it is wrong.
    """
    with tempfile.TemporaryDirectory() as directory:
        # What a stamp says, and that the two claims come apart cleanly.
        assert encoding_recipe(getattr(StrategicNet(), ENCODING_KEY)) == STRATEGIC_RECIPE
        assert encoding_features(getattr(StrategicNet(), ENCODING_KEY)) == STRATEGIC_FEATURES
        assert TACTICAL_RECIPE != STRATEGIC_RECIPE, "the three cuts retire separately or they retire together"

        # A file whose names are this layer's and whose recipe is not: the very failure above.
        moved = os.path.join(directory, "moved.pt")
        state = TacticalNet().state_dict()
        state[ENCODING_KEY] = encoding_stamp(TACTICAL_FEATURES, "0123456789abcdef")
        torch.save(state, moved)
        refusal = _refusal(SystemExit, frozen_tactics, moved)
        assert "different recipe" in refusal and "0123456789abcdef" in refusal and TACTICAL_RECIPE in refusal

        # A file written while the stamp carried names alone. Refused, and the refusal names the way back.
        half = os.path.join(directory, "half.pt")
        state = TacticalNet().state_dict()
        state[ENCODING_KEY] = _bytes("\n".join(TACTICAL_FEATURES))
        torch.save(state, half)
        refusal = _refusal(SystemExit, frozen_tactics, half)
        assert "not the recipe" in refusal and "avow" in refusal

        # And an avowal adds the missing half without touching the half the file already states.
        recorded = torch.load(half, map_location="cpu")
        words = "fitted after the last change to the tactical cut"
        torch.save(avowed(recorded, TacticalNet(), words), half)
        after = torch.load(half, map_location="cpu")
        assert encoding_recipe(after[ENCODING_KEY]) == TACTICAL_RECIPE
        assert encoding_avowal(after) == words
        assert all(torch.equal(after[name], value) for name, value in recorded.items()
                   if name != ENCODING_KEY)
        frozen = frozen_tactics(half)
        try:
            assert frozen.build is not None
        finally:
            frozen.batcher.stop()


def test_the_three_rules_digest_apart_and_a_change_to_one_leaves_the_others_alone():
    """A teacher records a rule as much as it records an encoding, and the rule is the half nothing could state.

    The three layers' rules are walked from their own deciding entry points, so a change to one does not retire the other two's teachers. What is deliberately outside the walk is the reporting half of each layer: a teacher records the choice, so what has to be digested is what makes the choice.
    """
    from rwintel.learn.recipe import digest, rule_digests
    from rwintel.control.policy import tactics as tactics_module

    rules = rule_digests()
    assert set(rules) == {"tactics", "operations", "strategy"}
    assert len(set(rules.values())) == 3, "three rules that digest alike are three that retire together"
    assert all(len(value) == 16 for value in rules.values())
    # Twice in one process is twice the same answer, which is what a stamp anything checks has to be.
    assert rule_digests() == rules

    # And the walk reaches the helpers rather than stopping at the entry: taking the ladder's own gate away
    # moves the digest, though `_departure` reaches it only by name.
    class _Ungated(tactics_module.Tactics):
        def _allowed(self, departure):
            return True

    moved = digest([_Ungated, tactics_module], ["_departure"])
    assert moved != rules["tactics"], "a helper rewritten under an unchanged caller has to move the digest"


def _unstamped(net, path):
    """A set of parameters as everything recorded before the feature list was recorded looks: the right shapes, and nothing in the file saying which encoding fitted them."""
    state = net.state_dict()
    del state[ENCODING_KEY]
    torch.save(state, path)
    return path


def test_parameters_recorded_before_the_feature_list_are_read_again_once_a_person_avows_them():
    """The way back for a set of parameters recorded before a file stated anything about its own encoding, which is every set this project made before the list was first written down and several of which cost a training run apiece.

    Refusing them is right and refusing them for good is not. Such a file cannot PROVE what it was fitted to — that is the whole point of writing the list down — but a person who knows the layer's encoding has not moved since they were fitted knows something true that the file does not say, and throwing that work away because nothing in the file can say it is a worse answer than letting the person say it. So they say it into the file: the list they swear it was fitted to, and the words for why, both written where they are saved and copied with the parameters and cannot be mislaid.

    What every loader does afterwards is load it and say out loud that the list beside those parameters is somebody's word and not a fit's record, which is the difference between a number measured on a claim and a number measured on a file that could prove what it was.
    """
    words = "the operational encoding did not move when the list was first recorded"
    with tempfile.TemporaryDirectory() as directory:
        path = _unstamped(OperationalNet(), os.path.join(directory, "operations.pt"))
        assert "no feature list" in _refusal(ValueError, eval_arms.build_all, [f"ops:{path}"])

        recorded = torch.load(path, map_location="cpu")
        torch.save(avowed(recorded, OperationalNet(), words), path)
        arms, batchers = eval_arms.build_all([f"ops:{path}"])
        try:
            assert [name for name, _ in arms] == ["operations"]
        finally:
            for batcher in batchers:
                batcher.stop()

        # The words are in the file and stay there, so that every later reader of it, and not only the run that was standing there when it was avowed, is told the list is a claim.
        after = torch.load(path, map_location="cpu")
        assert encoding_avowal(after) == words
        # And the avowal is a statement ABOUT the parameters: it adds the list and the words, and moves not one number of what was recorded.
        assert sorted(set(after) - set(recorded)) == sorted([ENCODING_KEY, AVOWAL_KEY])
        assert all(torch.equal(after[name], value) for name, value in recorded.items())

        # The same route under the arena, where a tactical layer is frozen beneath both sides of every board.
        tactics = _unstamped(TacticalNet(), os.path.join(directory, "tactics.pt"))
        assert "no feature list" in _refusal(SystemExit, frozen_tactics, tactics)
        torch.save(avowed(torch.load(tactics, map_location="cpu"), TacticalNet(), words), tactics)
        frozen = frozen_tactics(tactics)
        try:
            assert frozen.build is not None
        finally:
            frozen.batcher.stop()


def test_an_avowal_cannot_be_made_over_a_stated_list_or_for_a_layer_the_file_is_not():
    """What the way back is not allowed to be, which matters more than what it is: it must not become a way of waving any file through.

    A file that already states a list cannot be avowed at all, and that is what keeps the refusal of a file fitted to a DIFFERENT list absolute. If an avowal could write over a list, then every refusal in this suite would last exactly as long as it took somebody to run one, and the list would be a thing you clear rather than a thing that decides.

    A file whose first layer does not read this layer's width cannot be avowed as this layer's. That is the whole of what a width proves — which layer's parameters these are, and nothing whatever about what the numbers in their slots mean — and it is what stops one layer's name pointed at a heap of recorded files from stamping the other layer's with a list they were never fitted to. It matters here because the two encodings do not move together: the tactical list changed when the fight-relative bearing replaced the target offsets and the operational list did not, so a tactical file avowed under today's tactical list would afterwards load in silence and be measured under its own trained name.

    And a person has to write down why. The sentence is the only evidence the file will ever carry for a claim nothing in it can check, so an avowal without one is not a claim, it is a shrug.
    """
    words = "the encoding did not move"
    with tempfile.TemporaryDirectory() as directory:
        state = torch.load(_unstamped(TacticalNet(), os.path.join(directory, "tactics.pt")),
                           map_location="cpu")

        refusal = _refusal(EncodingRefused, avowed, state, OperationalNet(), words)
        assert "another layer's parameters" in refusal and str(TACTICAL_SIZE) in refusal

        assert "reason" in _refusal(EncodingRefused, avowed, state, TacticalNet(), "   ")

        stale = TacticalNet().state_dict()
        stale[ENCODING_KEY] = encoding_stamp(
            ["target_dx" if name == "target_ahead_of_home" else name for name in TACTICAL_FEATURES],
            TACTICAL_RECIPE)
        refusal = _refusal(EncodingRefused, avowed, stale, TacticalNet(), words)
        assert "refused for good" in refusal
        # Nor over a file that states this layer's list AND the recipe those features were made by, which is what a fit writes: there is nothing left for a person to add.
        whole = TacticalNet().state_dict()
        refusal = _refusal(EncodingRefused, avowed, whole, TacticalNet(), words)
        assert "nothing here a person can add" in refusal
        # Nor twice, which is the same refusal reached from the other side: a file that has been avowed carries somebody's word and is protected by it.
        once = avowed(state, TacticalNet(), words)
        assert "already carries somebody's avowal" in _refusal(EncodingRefused, avowed, once, TacticalNet(), words)


def test_avowing_will_not_guess_which_layer_a_file_belongs_to():
    """The command that writes an avowal has no default layer, where every other command here has one.

    The width of a file refuses one layer's parameters offered as another's, and it can only do that once a layer has been named. A default would let the wrong layer's list be written by saying nothing at all — which is the accident worth being safe against, since a file avowed under the wrong list afterwards loads in silence and is measured under the name of whatever it was trained to be.
    """
    from rwintel.learn.__main__ import main

    with tempfile.TemporaryDirectory() as directory:
        path = _unstamped(OperationalNet(), os.path.join(directory, "operations.pt"))
        # The parser refuses with a status and prints what it wanted, which is where the words are.
        said = io.StringIO()
        with contextlib.redirect_stderr(said):
            refused = _refusal(SystemExit, main, ["avow", "--load", path, "--because", "it did not move"])
        assert refused and "--layer" in said.getvalue()
        assert ENCODING_KEY not in torch.load(path, map_location="cpu"), (
            "a file was avowed by a command that had not been told which layer it belonged to")

        # Named, and named as the layer it is not: the width refuses it file by file, which is what a run over a directory of recorded parameters meets.
        tactics = _unstamped(TacticalNet(), os.path.join(directory, "tactics.pt"))
        refused = _refusal(SystemExit, main,
                           ["avow", "--layer", "operations", "--load", f"{path},{tactics}",
                            "--because", "the operational encoding did not move"])
        assert tactics in refused and "another layer's parameters" in refused
        # And the operational file named beside it was avowed before the refusal, since each is judged on its own.
        assert ENCODING_KEY in torch.load(path, map_location="cpu")


def test_the_tactical_width_is_read_off_the_file_rather_than_asked_for_as_a_flag():
    """The first layer is one map from the tactical features to the width, so the file states its own width and there is no flag to keep in step with it. A flag would only ever be a way to fail a load that was going to succeed — and unlike the duel, which can also build a fresh network, there is never a fresh network here."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "narrow.pt")
        torch.save(TacticalNet(width=32).state_dict(), path)
        frozen = frozen_tactics(path)
        try:
            layer = frozen.build(None, _CATALOGUE)
            assert layer.decider.net.body[0].out_features == 32
            assert layer.decider.net.body[0].in_features == TACTICAL_SIZE
        finally:
            frozen.stop()


# ---- the strategic layer, which is the one layer paid the match --------------------------------

class _Ending:
    """An episode as far as the score reads one, which is what the session hands the policy when a match ends."""

    def __init__(self, winner, team, timeout, standing):
        self.winner, self.team, self.timeout, self.standing = winner, team, timeout, standing


def _report(**overrides):
    fields = dict(income=30.0, credits=1500.0, military_value=4000.0, enemy_military_value=4000.0,
                  our_value=7000.0, enemy_value=7000.0,
                  held=3, enemy_held=3, lost_regions=0, enemy_bases=2,
                  units=20, unit_cap=100, under_construction=1)
    fields.update(overrides)
    return FrontReport(**fields)


def _places():
    return [_region(1, 0.0, 0.0, ours=800.0, theirs=0.0),
            _region(2, 900.0, 0.0, ours=0.0, theirs=900.0),
            _region(3, 400.0, 400.0, ours=200.0, theirs=200.0)]


def test_the_strategic_state_is_its_feature_list_and_nothing_leaves_the_range():
    """The same three rules the other two cuts follow, checked the same way. A vector one slot out of step fits every width and trains without complaint, so the length is pinned against the names rather than against a number written twice; and every entry is a share, a ratio against a stated scale or a flag, so a match with a hundred times the credits of another cannot produce a feature a hundred times larger."""
    state = strategic_state(_report(), _places(), 300000, income_history=[20.0, 22.0, 25.0],
                            loss_history=[0, 1, 1], most_enemy_bases=3, posture=Posture.ARM)
    assert len(state) == STRATEGIC_SIZE == len(STRATEGIC_FEATURES)
    assert all(math.isfinite(value) for value in state)
    assert all(-1.0 <= value <= 1.0 for value in state)

    named = dict(zip(STRATEGIC_FEATURES, state))
    # An even board reads nought on the edge, whatever the armies are worth, which is what makes the potential the same number the score is.
    assert named["military_edge"] == 0.5, "the edge feature is a share and an even board is a half of it"
    assert named["posture_arm"] == 1.0 and named["posture_expand"] == 0.0
    # The growth across the window as a share of its oldest sample, which is the quantity the rule thresholds — handed over as the number rather than as the rule's verdict on it, so that a policy can put the threshold where it likes.
    assert abs(named["income_growth"] - (25.0 - 20.0) / 20.0) < 1e-9
    # Two of the last three periods saw ground go, which is exactly what the rule reacts at, and it is reported as the count rather than as the reaction.
    assert abs(named["losing_periods"] - 2.0 / 3.0) < 1e-9
    assert named["driven_back"] == 0.0
    assert named["bias"] == 1.0

    # Driven back is a verdict and has to be: an enemy on their last base reads the same count as one that only ever had one.
    driven = strategic_state(_report(enemy_bases=1), _places(), 300000, most_enemy_bases=3)
    untouched = strategic_state(_report(enemy_bases=1), _places(), 300000, most_enemy_bases=1)
    assert dict(zip(STRATEGIC_FEATURES, driven))["driven_back"] == 1.0
    assert dict(zip(STRATEGIC_FEATURES, untouched))["driven_back"] == 0.0

    # An empty board is an ordinary board here: the first frames of a match have nothing standing and nothing contacted, and a division by nothing in this cut would end the run before the first decision.
    empty = strategic_state(_report(military_value=0.0, enemy_military_value=0.0, our_value=0.0,
                                    enemy_value=0.0, held=0, enemy_held=0, units=0, unit_cap=0), [], 0)
    assert len(empty) == STRATEGIC_SIZE and all(math.isfinite(value) for value in empty)


def test_the_two_pairs_of_values_each_compare_like_with_like():
    """The defect this pair of fields was split to remove, pinned so that it cannot come back quietly.

    The report used to carry our fighters and their whole side under one pair of names, so anything reading the two as a comparison compared our army with their army plus their buildings plus their builders, and read us as worse off than we were. Nothing read it that way while the transition rule was the only reader; the strategic layer's reward is a comparison, and so is the operational cut's edge.

    So there are two pairs now, and each is tested by moving one side of it. Adding buildings to the enemy moves the whole-standing pair, which is what the match is scored on, and must not move the fighting pair, which is what an edge between two armies means.
    """
    lean = _report(military_value=4000.0, enemy_military_value=4000.0, our_value=7000.0, enemy_value=7000.0)
    built = _report(military_value=4000.0, enemy_military_value=4000.0, our_value=7000.0, enemy_value=12000.0)

    def edge(report):
        return dict(zip(STRATEGIC_FEATURES, strategic_state(report, _places(), 30000)))["military_edge"]

    assert edge(lean) == edge(built) == 0.5, "the edge feature moved when only the enemy's buildings did"

    reward = StrategicReward(discount=1.0)
    assert abs(reward._potential(lean)) < 1e-9
    assert reward._potential(built) < -0.2, (
        "the potential is the running form of the match score, which counts everything standing, "
        "so buildings the enemy put up have to move it")


def test_the_strategic_layer_is_the_script_with_one_method_replaced():
    """What makes a learnt posture comparable with the rule's: everything except which posture is chosen is the same code.

    With no decider the layer falls through to the inherited rule and writes down what the rule chose, which is how the script becomes this layer's teacher. With one, the posture is the decider's and the orders that come out are the inherited tables read at that posture — the allocation, whether it presses, the loss allowance and the priorities are not re-decided anywhere.
    """
    script = Strategy(None, _CATALOGUE)
    teacher = LearntStrategy(None, _CATALOGUE, None, rollout=Rollout(), instance=0)
    report, regions = _report(), _places()
    for now in (10000, 20000, 30000):
        expected = script.decide(report, regions, now)
        written = teacher.decide(report, regions, now)
        assert teacher.posture is script.posture, "the layer with no decider is not deciding as the rule does"
        assert written[0].allocation == expected[0].allocation
        assert written[1].priorities == expected[1].priorities
    assert teacher.pending is not None and teacher.pending.action == int(script.posture)

    # Handed a decider, the posture is the decider's answer and everything downstream is that posture's own row of the inherited tables.
    learnt = LearntStrategy(None, _CATALOGUE, _Fixed(action=int(Posture.DECIDE)), rollout=Rollout(), instance=0)
    economy, operations = learnt.decide(report, regions, 10000)
    assert learnt.posture is Posture.DECIDE
    assert economy.allocation == ALLOCATION[Posture.DECIDE]
    assert operations.offensive is OFFENSIVE[Posture.DECIDE]
    assert operations.loss_allowance == max(LOSS_ALLOWANCE_FLOOR,
                                            LOSS_ALLOWANCE_SHARE[Posture.DECIDE] * report.military_value)


def test_the_chain_tells_the_strategic_layer_what_the_enemy_is_fielding():
    """Seven of the strategic cut's twenty-nine features are the shares of the enemy's worth by role, and nothing in the chain used to hand that reading over, so all seven were nought in every frame of every match — a quarter of the cut, structurally dead, and dead in a way no range or length check can see.

    It is the same hole on the rule's side. The handwritten layer's answer to an air-heavy enemy is written as a condition on exactly this reading, so it could never fire either. Both are closed by the chain reading the board and handing it down, which is what this drives end to end: a board with the enemy's worth split between two roles, through the real `ScriptPolicy`, and the layer must have been told.
    """
    from rwintel.control.policy import ScriptPolicy

    told = []

    class _Watched(Strategy):
        def decide(self, report, regions, game_time_ms, contact=None):
            told.append(contact)
            return super().decide(report, regions, game_time_ms, contact)

    class _Session:
        types = _TYPES
        assets = {}
        regions = []
        map_content = None

    session = _Session()
    policy = ScriptPolicy(session)
    policy.strategy = _Watched(session, policy.catalogue)
    # Two of ours and two of the enemy's, the enemy's split across two type indices so the reading is a mix and not one role at whole.
    units = [_unit(1, 0.0, 0.0), _unit(2, 100.0, 0.0),
             _unit(3, 900.0, 0.0, type_index=0, hostile=1),
             _unit(4, 950.0, 0.0, type_index=len(_TYPES) - 1, hostile=1)]
    policy.plan(_observation(units, [_region(1, 0.0, 0.0)]))

    assert told and told[0], "the chain decided a posture without saying what the enemy is fielding"
    contact = told[0]
    assert abs(sum(contact.values()) - sum(_CATALOGUE.value(unit.type_index)
                                           for unit in units if unit.hostile)) < 1e-9, (
        "the reading is not the enemy's whole standing worth")
    # And it reaches the cut as shares that are not all nought, which is the state the layer is actually trained on.
    state = dict(zip(STRATEGIC_FEATURES, strategic_state(_report(), _places(), 30000, contact=contact)))
    assert sum(value for name, value in state.items() if name.startswith("contact_")) > 0.0


def test_a_match_returns_its_own_score_and_the_shaping_cancels_whole():
    """The property every layer here is built around, on the layer that is paid the match.

    The shaping is only harmless if it telescopes over the errand, and an errand's return has to be the terminal alone. So a match's decisions must sum to the score that match came to, less the potential it opened at — whatever the board did in between, and however many periods it took. Checked at a discount of one, which is what a match is discounted at from this layer's seat, over a board that moves both ways.
    """
    rollout = Rollout(discount=1.0, trace=1.0)
    layer = LearntStrategy(None, _CATALOGUE, _Fixed(action=int(Posture.ARM)), rollout=rollout,
                           instance=0, discount=1.0)
    edges = [(4000.0, 4000.0), (5000.0, 3000.0), (3000.0, 6000.0), (7000.0, 2000.0)]
    for period, (ours, theirs) in enumerate(edges):
        layer.decide(_report(our_value=ours, enemy_value=theirs), _places(), 10000 * (period + 1))
    opening = 2.0 * (4000.0 / 8000.0 - 0.5)
    layer.conclude(0.6, "match")

    trajectory, = rollout.done
    assert trajectory.finished and len(trajectory.steps) == len(edges)
    assert abs(sum(step.reward for step in trajectory.steps) - (0.6 - opening)) < 1e-9
    assert layer.terminals["match"] == 1

    # And a second match on the same layer opens a fresh errand rather than continuing the first: the ledger the terminal closed is gone, a new one opens at the new board, and the decision now waiting is not appended to the trajectory that has already been paid.
    assert layer.reward.mission is None and layer.pending is None
    layer.decide(_report(our_value=1000.0, enemy_value=9000.0), _places(), 10000)
    assert len(rollout.done) == 1 and not rollout.live
    assert layer.pending is not None
    assert abs(layer.reward.mission.potential - 2.0 * (1000.0 / 10000.0 - 0.5)) < 1e-9


def test_a_posture_a_human_pinned_is_not_recorded_and_its_periods_are_still_carried():
    """A pinned posture is somebody else's decision, so nothing is recorded for it — the design says the results of what another commander decided are kept out of the learning signal, and this is that rule at the one place this layer can be taken over.

    What must not happen is that the shaping earned while it was pinned is dropped. The telescope only closes if every increment reached some decision, so those periods are carried and paid into the next decision this layer does take. The match still returns its terminal less what it opened at.
    """
    rollout = Rollout(discount=1.0, trace=1.0)
    layer = LearntStrategy(None, _CATALOGUE, _Fixed(action=int(Posture.ARM)), rollout=rollout,
                           instance=0, discount=1.0)
    layer.forced = Posture.DEFEND
    for period, (ours, theirs) in enumerate([(4000.0, 4000.0), (6000.0, 2000.0)]):
        layer.decide(_report(our_value=ours, enemy_value=theirs), _places(), 10000 * (period + 1))
    assert layer.posture is Posture.DEFEND
    assert not rollout.live and not rollout.done, "a pinned posture recorded a decision of its own"
    assert layer.owed != 0.0, "the shaping earned while the posture was pinned was dropped"

    layer.forced = None
    layer.decide(_report(our_value=5000.0, enemy_value=3000.0), _places(), 30000)
    layer.conclude(0.25, "match")
    trajectory, = rollout.done
    opening = 2.0 * (4000.0 / 8000.0 - 0.5)
    assert abs(sum(step.reward for step in trajectory.steps) - (0.25 - opening)) < 1e-9


def test_the_match_result_reaches_the_layer_through_the_chain_and_only_that_layer():
    """How the terminal arrives: the session hands the episode to the policy, the policy scores it and hands the number to the layer that is paid the match. Only the strategic layer defines the method, so the same call is a no-op for the other two — which is the design's rule about credit assignment not crossing a layer boundary, made mechanical rather than remembered."""
    rollout = Rollout(discount=1.0, trace=1.0)
    layer = LearntStrategy(None, _CATALOGUE, _Fixed(action=int(Posture.EXPAND)), rollout=rollout,
                           instance=0, discount=1.0)
    layer.decide(_report(), _places(), 10000)

    # A decided match saturates the score, whatever the board looked like when it ended.
    won = _Ending(winner=0, team=0, timeout=False,
                  standing=[{"team": 0, "value": 10.0, "income": 1.0, "killed": 0, "lost": 0},
                            {"team": 1, "value": 9000.0, "income": 90.0, "killed": 0, "lost": 0}])
    assert score(won) == 1.0
    layer.conclude(score(won), "match")
    trajectory, = rollout.done
    assert trajectory.steps[-1].done and trajectory.steps[-1].reward == 1.0

    # The same call against a layer that is not paid the match does nothing at all rather than raising.
    tactical = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=Rollout(), instance=0)
    assert getattr(tactical, "conclude", None) is None


def test_the_strategic_trajectory_cannot_collide_with_a_squads():
    """One buffer serves a run, and a trajectory is keyed by the instance and the thing it is about. The strategic layer commands no squad, so its key has to be a number the squad pool never issues, or a match's strategic decisions and some squad's errand would be appended into one trajectory and the advantage of each would run into the other."""
    assert LearntStrategy.KEY < 0
    rollout = Rollout(discount=1.0, trace=1.0)
    strategic = LearntStrategy(None, _CATALOGUE, _Fixed(action=int(Posture.ARM)), rollout=rollout,
                              instance=0, discount=1.0)
    tactical = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    strategic.decide(_report(), _places(), 10000)
    tactical.decide(_skirmish(), [_squad()], 21000)
    strategic.decide(_report(our_value=6000.0), _places(), 20000)
    tactical.decide(_skirmish(), [_squad()], 21200)
    assert sorted(rollout.live) == [(0, LearntStrategy.KEY), (0, 0)]


# ---- the operational arms a measuring run puts on one set of boards ---------------------------

class _Arms:
    """Only what `arms_of` reads of a run's arguments."""

    def __init__(self, our, load=None, device=None):
        self.our = list(our)
        self.load = load
        self.device = device


def test_two_learnt_arms_carry_their_own_parameters_and_meet_the_same_boards():
    """Why one run has to be able to hold two learnt arms at once.

    The question an arm comparison most often has to resolve is whether a longer training run produced a stronger layer, and that is two generations of one line. Measured in two separate runs the difference pays the arena's board scatter twice — one arm's episodes scatter by about 0.11 while the differences being looked for are around 0.04 — and it has to assume two runs made at different moments on a machine doing different things were otherwise alike. In one run they alternate on held boards and every difference is paired.

    What makes that safe is naming. Each arm is labelled after the file it reads, since two of them under one name would be written into one journal as one arm that played every board twice, which the pairing drops; and each carries the digest of its own parameters as its identity, because a path is a nickname that a later training run overwrites underneath itself.

    Nothing is loaded and no thread is started while the arms are only being named, so that every refusal below happens before the run holds a network or a game.
    """
    threads = threading.active_count()
    with tempfile.TemporaryDirectory() as directory:
        first = os.path.join(directory, "ops-arena-c1.pt")
        second = os.path.join(directory, "ops-arena-c2.pt")
        torch.save(OperationalNet().state_dict(), first)
        torch.save(OperationalNet().state_dict(), second)

        arms = arms_of(_Arms(["script", "concentrate", f"learnt:{first}", f"learnt:{second}"]))
        assert [arm.label for arm in arms] == ["script", "concentrate", "learnt-ops-arena-c1", "learnt-ops-arena-c2"]
        assert [arm.kind for arm in arms] == ["script", "concentrate", "learnt", "learnt"]
        # The handwritten arms are named by the rule they run; a learnt one by what its parameters are, which two files never share.
        assert [arm.name for arm in arms[:2]] == ["script", "concentrate"]
        assert arms[2].name.startswith("sha256:") and arms[2].name != arms[3].name
        assert all(arm.net is None and arm.batcher is None for arm in arms)
        assert threading.active_count() == threads

        # The bare word is the form every measurement so far was taken with, and it keeps the name those journals carry.
        bare = arms_of(_Arms(["script", "learnt"], load=first))
        assert [arm.label for arm in bare] == ["script", "learnt"]
        assert bare[1].name == arms[2].name

        loaded = load_arms(arms)
        try:
            assert [arm.label for arm in loaded] == [arm.label for arm in arms]
            assert all(arm.net is None and arm.batcher is None for arm in loaded[:2])
            # One network and one batching server per learnt arm: two arms are two policies, and a batch is one forward pass of one network.
            assert loaded[2].net is not loaded[3].net
            assert loaded[2].batcher is not None and loaded[2].batcher is not loaded[3].batcher
        finally:
            for arm in loaded:
                if arm.batcher is not None:
                    arm.batcher.stop()


def test_arms_that_could_not_be_told_apart_afterwards_are_refused_before_the_run_starts():
    """Every way two arms could end up indistinguishable in the journal, refused while the run still holds nothing.

    Two arms under one label would be one name over two policies, which a later comparison reads as one arm that played every board twice and drops. Two labels over one file is the same fault the other way round: one policy under two names, which the run would then report as differing from itself by the engine's own scatter. And parameters that are not there are refused rather than loaded blind, for the reason the duel refuses them — a mistyped path leaves a freshly initialised network in place and the run measures a random policy under the trained one's name.
    """
    threads = threading.active_count()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "ops-arena-c1.pt")
        torch.save(OperationalNet().state_dict(), path)

        assert "two arms would be journalled as" in _refusal(
            SystemExit, arms_of, _Arms([f"learnt:{path}", f"learnt:{path}"]))
        # The bare word and the same file named outright are two labels over one policy.
        assert "one policy under two names" in _refusal(
            SystemExit, arms_of, _Arms(["learnt", f"learnt:{path}"], load=path))
        assert "no such operational arm" in _refusal(SystemExit, arms_of, _Arms(["ladder"]))
        assert "only a learnt arm carries parameters" in _refusal(
            SystemExit, arms_of, _Arms([f"script:{path}"]))
        assert "no parameters at" in _refusal(
            SystemExit, arms_of, _Arms([f"learnt:{os.path.join(directory, 'absent.pt')}"]))
        assert "has no parameters to measure" in _refusal(SystemExit, arms_of, _Arms(["learnt"]))
        assert threading.active_count() == threads


# ---- one ask a side, not one a squad ---------------------------------------------------------

class _Counting:
    """A batching server's evaluate that writes down every batch it was handed and answers each row from its own place in that batch, so a test can tell one call of four rows from four calls of one and can tell each caller's answer from its neighbour's."""

    def __init__(self):
        self.batches = []

    def __call__(self, requests):
        self.batches.append(list(requests))
        return [(len(self.batches), index) for index in range(len(requests))]


def test_a_sides_squads_are_one_ask_of_the_inference_server_rather_than_one_each():
    """The defect this exists to keep out: a tactical layer that submitted its squads one at a time paid a full batching window for each, because a submission queues its request and then blocks on it, so the second was not even in the queue until the first had been answered. The window is sized against a twenty millisecond wall period, so a side of four squads spent four windows a frame and the decision lag stopped being the one period the interface promises and became a lag that depends on how many squads a side happens to have. There is no timing here: what is measured is that the whole side arrives at the server in one batch.

    The rows are checked to be in squad order and each squad's departure to be its own row's answer, because a fix that handed the side one answer for all of them, or that zipped the answers onto the squads the wrong way round, would batch just as well and decide wrongly.
    """
    from rwintel.learn.inference import Batcher

    batches = []

    def evaluate(requests):
        batches.append(list(requests))
        # A different departure for every row, so that an answer handed to the wrong squad is visible.
        return [Choice(action=index % len(Deviation)) for index in range(len(requests))]

    batcher = Batcher(evaluate)
    try:
        layer = LearntTactics(None, _CATALOGUE, NetworkTactics(None, None, batcher), None, 0)
        units, squads = [], []
        # Squads of different sizes, because the first tactical feature is the member count and it is therefore what says which row of the batch belongs to which squad.
        for index, size in enumerate((2, 3, 4, 5)):
            base = index * 3000.0
            ids = [10 * index + n for n in range(size)]
            units += [_unit(unit_id, base + 100.0 + 20.0 * n, 100.0, hit=100)
                      for n, unit_id in enumerate(ids)]
            units.append(_unit(900 + index, base + 210.0, 110.0, hostile=1))
            squads.append(_squad(id=index, members=ids, x=base + 120.0, y=100.0))
        view = _view(units, [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])

        deviations, _ = layer.decide(view, squads, 21000)

        assert batcher.calls == 1, "the side was submitted one squad at a time, a batching window each"
        assert batcher.served == 4 and batcher.batch_size == 4.0
        batch, = batches
        assert len(batch) == 4
        # Ascending because the squads were built with ascending member counts and read in that order.
        counts = [state[0] for state, _ in batch]
        assert counts == sorted(counts) and len(set(counts)) == 4
        # Each squad answered from its own row of the batch rather than all of them from one.
        assert [(d.squad, int(d.deviation)) for d in deviations] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    finally:
        batcher.stop()


def test_the_steps_of_a_period_are_recorded_one_a_squad_and_in_the_order_they_were_read():
    """What the trainer is shown must not depend on the side being asked in one call rather than in several. A rollout that reordered its steps, or that filed one squad's state under another squad's number, would train perfectly smoothly on decisions credited to the wrong board — so the recording is pinned here to one step a squad, filed under that squad's own number, holding that squad's own state, in the order the squads were read."""
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, _Fixed(action=int(Deviation.KITE)), rollout=rollout,
                          instance=0)
    units, squads = [], []
    for index, size in enumerate((2, 3, 4)):
        base = index * 3000.0
        ids = [10 * index + n for n in range(size)]
        units += [_unit(unit_id, base + 100.0 + 20.0 * n, 100.0, hit=100)
                  for n, unit_id in enumerate(ids)]
        units.append(_unit(900 + index, base + 210.0, 110.0, hostile=1))
        squads.append(_squad(id=index, members=ids, x=base + 120.0, y=100.0))
    view = _view(units, [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])

    layer.decide(view, squads, 21000)

    assert list(layer.pending) == [0, 1, 2]
    steps = [layer.pending[squad_id] for squad_id in (0, 1, 2)]
    assert [step.squad for step in steps] == [0, 1, 2]
    # The member count is the first tactical feature, so an ascending run of it says each step holds the state of the squad it is filed under.
    sizes = [step.state[0] for step in steps]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    # A mask of its own per step, because a shared list object would let one step's mask be edited through another's.
    assert len({id(step.mask) for step in steps}) == 3
    assert all(step.mask == [1.0] * len(Deviation) for step in steps)


def test_the_batching_server_answers_a_group_queued_together_in_one_call():
    """The half of the fix that lives in the server. A caller with several requests cannot loop over the single submission, because that one queues a ticket and then blocks on it; the group has to be queued before any of it is waited on. Queued that way it meets the window once and is answered in one call, and a group larger than the batch cap is answered over consecutive calls rather than refused, with every answer still coming back to the request that asked for it."""
    from rwintel.learn.inference import Batcher

    evaluate = _Counting()
    batcher = Batcher(evaluate)
    try:
        assert batcher.submit_many([("a", i) for i in range(5)]) == [(1, i) for i in range(5)]
        assert batcher.calls == 1 and batcher.served == 5
        # Nothing asked is nothing waited for: an empty side must not spend a window being handed back nothing.
        assert batcher.submit_many([]) == [] and batcher.calls == 1
    finally:
        batcher.stop()

    evaluate = _Counting()
    capped = Batcher(evaluate, max_batch=2)
    try:
        # Answered over three calls of two, two and one, and still one answer per request and in order.
        assert capped.submit_many([("b", i) for i in range(5)]) == [
            (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
        assert capped.calls == 3 and capped.served == 5
    finally:
        capped.stop()


def test_a_failed_batch_reaches_every_caller_of_a_group():
    """A forward pass that raises has to reach whoever was waiting on it, or every caller of the group waits for ever. The failure is raised only after every ticket of the group has been waited on, so a group is never left half collected with tickets still to be answered, and a server that has been stopped refuses a group at once rather than queueing it behind a thread that has gone."""
    from rwintel.learn.inference import Batcher

    def broken(requests):
        raise ZeroDivisionError("the forward pass fell over")

    batcher = Batcher(broken)
    try:
        assert "fell over" in _refusal(ZeroDivisionError, batcher.submit_many, [1, 2, 3])
        # The single submission is the plural one with a group of one, so it fails the same way it always has.
        assert "fell over" in _refusal(ZeroDivisionError, batcher.submit, 4)
    finally:
        batcher.stop()
    assert "stopped" in _refusal(RuntimeError, batcher.submit_many, [1, 2])


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)


def test_an_arm_can_carry_the_whole_learnt_chain():
    """The design says a layer is learnt against frozen neighbours, and the training runners freeze them; until this the measuring runner could not, so a chain with two layers trained could be built and never measured.

    An arm may therefore name several learnt layers at once. Every one of them is read and none is learnt from, which is what a measurement is: the layer named first stands as the arm's own and the rest are frozen beside it, and that is the same construction either way since a frozen layer and a measured one differ only in the rollout and neither has one here.
    """
    with tempfile.TemporaryDirectory() as folder:
            operational = os.path.join(folder, "operations.pt")
            strategic = os.path.join(folder, "strategy.pt")
            torch.save(OperationalNet().state_dict(), operational)
            torch.save(StrategicNet().state_dict(), strategic)

            name = f"ops:{operational}+strategy:{strategic}"
            assert eval_arms._is_learnt(name)
            (arm_name, build), batcher = eval_arms.learnt(name)
            try:
                # Named for both files, so a journal can never merge the chain with either layer measured alone.
                assert arm_name == "operations+strategy"
                # Both layers are put in place and neither is handed a rollout, which is what a measurement is: the policy is read, not fitted.
                session = _Chained()
                session.instance = 0
                with _without_the_game_installed():
                    policy = build(session)
                assert isinstance(policy.operations, LearntOperations)
                assert isinstance(policy.strategy, LearntStrategy)
                assert policy.operations.rollout is None and policy.strategy.rollout is None
                # The layer nobody named is the script's own, unchanged, so the arm is still the chain with the named decisions replaced and nothing else moved.
                assert type(policy.tactics).__name__ == "Tactics"
                # And the several servers stand in for one everywhere the run touches them, including the figure it reports at the end: a wrapper that could only be stopped crashed a completed measurement on its last line.
                assert batcher.batch_size == 0.0 and batcher.calls == 0
                batcher.batchers[0].calls, batcher.batchers[0].served = 3, 12
                batcher.batchers[1].calls, batcher.batchers[1].served = 1, 4
                assert abs(batcher.batch_size - 4.0) < 1e-12
            finally:
                batcher.stop()

            # A layer named twice would leave one of the two never playing, and is refused rather than silently dropped.
            try:
                eval_arms.learnt(f"ops:{operational}+ops:{operational}")
            except ValueError as refused:
                assert "named twice" in str(refused)
            else:
                raise AssertionError("naming one layer twice in an arm was accepted")


def test_an_arm_can_carry_a_learnt_fighter_under_the_chain():
    """A layer trained with a trained fighter frozen beneath it was trained in that environment, so measuring it with the handwritten fighter underneath measures it somewhere else.

    The training runners have been able to freeze a tactical layer under a match since `--frozen` existed, and the strategic generations on file were trained that way. The measuring runner could name only the operational and strategic layers, so the chain it scored was never the chain those layers were fitted in. It can name all three now, and the tactical layer's own strength is still read where it always was -- on the engagement arena, against the ladder, off any match at all.
    """
    with tempfile.TemporaryDirectory() as folder:
            tactical = os.path.join(folder, "tactics.pt")
            operational = os.path.join(folder, "operations.pt")
            strategic = os.path.join(folder, "strategy.pt")
            torch.save(TacticalNet().state_dict(), tactical)
            torch.save(OperationalNet().state_dict(), operational)
            torch.save(StrategicNet().state_dict(), strategic)

            name = f"tactics:{tactical}+ops:{operational}+strategy:{strategic}"
            assert eval_arms._is_learnt(name)
            (arm_name, build), batcher = eval_arms.learnt(name)
            try:
                assert arm_name == "tactics+operations+strategy"
                session = _Chained()
                session.instance = 0
                with _without_the_game_installed():
                    policy = build(session)
                # All three decisions come from networks, and none of the three records anything: an arm reads a policy, it does not fit one.
                assert isinstance(policy.tactics, LearntTactics)
                assert isinstance(policy.operations, LearntOperations)
                assert isinstance(policy.strategy, LearntStrategy)
                assert policy.tactics.rollout is None
                assert policy.operations.rollout is None and policy.strategy.rollout is None
            finally:
                batcher.stop()

            # A fighter alone is a legal arm too, since the same chain with one decision replaced is what every other arm here is.
            (alone, _), only = eval_arms.learnt(f"tactics:{tactical}")
            try:
                assert alone == "tactics"
            finally:
                only.stop()


# ---- what this session's repairs are held to ----------------------------------------------

def test_the_exposure_features_are_about_the_enemy_and_not_about_our_own_scatter():
    """`ours_in_their_reach` says how much of this squad the enemy can presently shoot, and nothing else can say it.

    It was measured as how many of our members stood within the enemy's longest reach OF OUR OWN CENTRE, so the count moved when the squad spread out and did not move at all when the enemy walked up to it. That is the squad's scatter, which the cut already carries as `spread`, wearing the name of exposure — and the eighth departure exists for exactly the board it could not describe: a squad that is being shot at from outside its own reach and has to walk in to answer.
    """
    def board(enemy_x):
        ours = [_unit(1, 100, 100), _unit(2, 130, 100), _unit(3, 160, 100)]
        # An artillery piece, reach 320, put where it covers the whole squad and then where it covers none of it.
        return _view(ours + [_unit(9, enemy_x, 100, type_index=1, hostile=1)],
                     [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])

    def exposure(view):
        squad = _squad()
        members, threats = _squad_fight(view, squad)
        state = tactical_state(squad, members, threats, squad.losses, 0.0, view, 30000)
        return dict(zip(TACTICAL_FEATURES, state))["ours_in_their_reach"]

    # Everything of ours inside the gun's reach, then the same squad with the gun beyond it.
    assert exposure(board(300)) == 1.0
    assert exposure(board(900)) == 0.0

    # And with nothing in reach of anybody, spreading the squad out does not move it: what it reports is the enemy's position, not ours.
    def scattered(spread):
        ours = [_unit(1, 100, 100), _unit(2, 100 + spread, 100), _unit(3, 100 + 2 * spread, 100)]
        return _view(ours + [_unit(9, 3000, 3000, type_index=1, hostile=1)],
                     [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])

    assert exposure(scattered(20)) == exposure(scattered(300)) == 0.0


def test_a_figure_read_only_inside_a_comprehension_is_still_in_the_stamp():
    """The stamp is a digest of the code that fills the slots, and a fair share of that code lives inside comprehensions.

    Walked from the outermost code object alone, a name read only from inside a comprehension or a generator expression never entered the queue, so the figure behind it was never digested by value. `RECENT_HIT_MS` is exactly that: it is read inside the generator expression that counts the members under fire, so the day it changed the tactical stamp would not have moved and every set of parameters fitted to the old meaning would have loaded in silence — which is the one failure the stamp exists to refuse.
    """
    from rwintel.learn import encoding as cut
    from rwintel.learn.recipe import digest

    def taken():
        return (digest([cut], ["tactical_state"]), digest([cut], ["strategic_state"]),
                digest([cut], ["operational_state", "operational_slots", "squad_slots",
                               "region_mask", "task_mask", "squad_mask"]))

    before = taken()
    held = cut.RECENT_HIT_MS
    try:
        cut.RECENT_HIT_MS = held + 1000
        after = taken()
    finally:
        cut.RECENT_HIT_MS = held
    assert after[0] != before[0], "a figure read inside a comprehension has to reach the stamp"
    # And it retires the one cut that reads it rather than all three, which is what walking from an entry point outward is for.
    assert after[1] == before[1] and after[2] == before[2]

    # A helper every cut runs its values through retires all three.
    kept = cut._finite
    try:
        cut._finite = lambda value: value
        both = taken()
    finally:
        cut._finite = kept
    assert all(now != then for now, then in zip(both, before))


def test_the_growth_of_an_economy_starting_from_nothing_is_not_a_plateau():
    """`income_growth` is the quantity the rule thresholds, handed over as a number. A window whose oldest sample is nought is an economy that started inside the window, which is the opposite of a plateau, and both used to read exactly nought."""
    def growth(history):
        state = strategic_state(_report(), _places(), 300000, income_history=history,
                                loss_history=[0], most_enemy_bases=2)
        return dict(zip(STRATEGIC_FEATURES, state))["income_growth"]

    started = growth([0.0, 0.0, 12.0, 20.0, 30.0])
    flat = growth([30.0, 30.0, 30.0, 30.0, 30.0])
    assert started > 0.0 and flat == 0.0, "an economy building itself must not read as one that has levelled off"
    # The rule reads the same window and answers the same way, which is what makes the feature the rule's own quantity.
    rule = Strategy.__new__(Strategy)
    rule.income_history = [0.0, 0.0, 12.0, 20.0, 30.0]
    assert not rule._income_levelled_off()
    rule.income_history = [30.0, 30.0, 30.0, 30.0, 30.0]
    assert rule._income_levelled_off()


def test_a_terminal_that_arrives_with_no_decision_waiting_closes_the_errand():
    """An errand ends in a period this layer took no decision in whenever the ending is not a fight: a squad that walks onto the ground it was sent to and holds it is not in contact, so no departure is chosen for it and there is nothing waiting to be paid.

    Carried into the ledger of unpaid shaping, as every other unpaid period is, the terminal was handed to whatever decision came next — and that decision belongs to no errand, because this one is over and the next contract has not arrived. The trajectory then never closed on a terminal at all, so nothing in it was anchored by anything but the critic.
    """
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, None, rollout=rollout, instance=0)
    squad = _squad(status=Status.COMPLETE, losses=0.0)
    # A region we hold outright, which is what the contract asked for.
    view = _view([_unit(1, 100, 100)], [_region(1, 400.0, 100.0, ours=900.0, theirs=0.0)])

    # One decision already filed from an earlier period, and nothing waiting: the squad is out of contact.
    rollout.add((0, squad.id), Step(state=[0.0], action=0, mask=[1.0], value=0.3, squad=squad.id))
    layer.reward.step(squad, view, 30000)          # opens the errand
    layer._settle(view, [squad], 32000)            # and this period finds it complete

    trajectory, = rollout.done
    assert trajectory.finished, "the errand ended, so its trajectory has to close rather than run on"
    assert layer.terminals["complete"] == 1
    assert squad.id not in layer.owed, "the terminal is a payment and not a carried shaping term"


def test_a_squad_that_leaves_with_nothing_waiting_is_still_cut_and_forgotten():
    """A squad leaves the board in a period this layer took no decision about it as often as not, and everything keyed by its number has to go with it: the trajectory, the carried payment and the reward's own ledger.

    Squad numbers come out of a pool of eight and the arena hands the same few round fight after fight, so a ledger left standing is a ledger the next squad to be given that number inherits — it files its decisions into the dead squad's trajectory and opens its errand against a potential measured on the dead squad's ground.
    """
    for layer in (LearntTactics(None, _CATALOGUE, None, rollout=Rollout(), instance=0),
                  LearntOperations(None, _CATALOGUE, None, rollout=Rollout(), instance=0)):
        rollout = layer.rollout
        squad = _squad(id=5)
        view = _view([_unit(1, 100, 100)], [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])
        rollout.add((0, squad.id), Step(state=[0.0], action=0, mask=[1.0], value=0.3, squad=squad.id))

        settle = ((lambda: layer._settle(view, [squad], 30000)) if isinstance(layer, LearntTactics)
                  else (lambda: layer._settle(view, None, [squad])))
        settle()
        assert squad.id in layer.tracked and layer.reward.missions

        gone = ((lambda: layer._settle(view, [], 32000)) if isinstance(layer, LearntTactics)
                else (lambda: layer._settle(view, None, [])))
        gone()
        cut, = rollout.done
        assert not cut.finished and cut.reason == "left"
        assert not layer.reward.missions, "the shaping ledger of a squad that is gone cannot be left for the next one"
        assert squad.id not in layer.tracked


def test_the_match_terminal_telescopes_at_every_discount():
    """What a match returns to the strategic layer, on a board that opened lopsided so the two conventions can be told apart.

    Paid against a terminal potential of nought, the shaping telescopes at whatever discount the run takes its returns at and the match returns the result less the potential it opened at. The convention this replaced handed back the movement since the opening instead, which made the return come to the result exactly and only telescoped at a discount of one — below one it left a residue that depended on the path the match took, which is the single thing potential-based shaping is chosen to rule out.
    """
    for discount in (1.0, 0.99):
        rollout = Rollout(discount=discount, trace=1.0)
        layer = LearntStrategy(None, _CATALOGUE, None, rollout=rollout, instance=0, discount=discount)
        # An opening the match is losing, so the opening potential is a long way from nought.
        opening = _report(our_value=1000.0, enemy_value=9000.0)
        layer._settle(opening)
        opened = layer.reward.mission.potential
        assert opened < -0.5

        for value, board in ((0.2, _report(our_value=5000.0, enemy_value=5000.0)),
                             (0.4, _report(our_value=8000.0, enemy_value=2000.0))):
            layer.pending = Step(state=[0.0], action=0, mask=[1.0], value=value, squad=layer.KEY)
            layer._settle(board)

        layer.pending = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=layer.KEY)
        layer.conclude(1.0, "match")

        trajectory, = rollout.done
        # The identity potential-based shaping promises is about the DISCOUNTED sum, which is the sum a return is: every shaping term is `discount x potential after - potential before`, so the discounted sum of them telescopes to minus the opening potential at any discount, and only at a discount of one does the plain sum do it too.
        paid = sum(discount ** index * step.reward for index, step in enumerate(trajectory.steps))
        earned = discount ** (len(trajectory.steps) - 1) * 1.0
        assert trajectory.finished and layer.terminals["match"] == 1
        assert abs(paid - (earned - opened)) < 1e-9, "the match returns the result less the potential it opened at"


def test_an_errand_out_of_time_pays_the_expired_terminal_less_the_potential_it_was_holding():
    """The fourth of the four discrete endings, held to the same identity as the other three."""
    reward = TacticalReward()
    squad = _squad(status=Status.ACTIVE, losses=0.0)
    view = _view([_unit(1, 100, 100)], [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])
    reward.step(squad, view, 30000)
    held = reward.missions[squad.id].potential

    squad.status = Status.EXPIRED
    outcome = reward.step(squad, view, 32000)
    assert outcome.done and outcome.reason == "expired"
    assert abs(outcome.reward - (-0.5 - held)) < 1e-9


def test_the_two_headed_update_scores_the_very_distribution_the_decision_was_drawn_from():
    """The operational layer's own update path, which nothing exercised at all.

    Two things are pinned and the second is the one that catches a whole class of silent failure. The update completes and moves the parameters; and on the first pass over a freshly collected batch the importance ratio is one, which is only true if the optimiser reads the same state, the same squad row and the same two masks the decider was asked with. A row confused with a squad number, or a mask rebuilt rather than recorded, shows up here and nowhere else.
    """
    net = OperationalNet()
    optimiser = Optimiser(net, two_headed=True)
    regions = [1.0] * 6 + [0.0] * (OPERATIONAL_REGIONS - 6)
    tasks = [1.0, 1.0] + [0.0] * (OPERATIONAL_TASKS - 2)

    steps = []
    for slot in range(4):
        state = [0.01 * (slot + 1)] * OPERATIONAL_SIZE
        choice = evaluate_operational(net, [(state, slot, regions, tasks)])[0]
        steps.append(Step(state=state, action=choice.action, mask=list(regions), second=choice.second,
                          second_mask=list(tasks), log_prob=choice.total_log_prob, value=choice.value,
                          reward=0.5, done=True, squad=slot, slot=slot))

    before = [parameter.detach().clone() for parameter in net.parameters()]
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    for step in steps:
        rollout.add((0, step.squad), step)
    drained = rollout.drain()

    # The ratio the objective is built on, computed the way the update computes it, before any parameter has moved.
    with torch.no_grad():
        states = torch.tensor([step.state for step in drained], dtype=torch.float32)
        slots = torch.stack([one_hot_slot(step.slot) for step in drained])
        masks = torch.tensor([step.mask for step in drained], dtype=torch.float32)
        second_masks = torch.tensor([list(step.second_mask) for step in drained], dtype=torch.float32)
        region_logits, task_logits, _ = net(states, slots, masks, second_masks)
        now = (torch.distributions.Categorical(logits=region_logits)
               .log_prob(torch.tensor([step.action for step in drained]))
               + torch.distributions.Categorical(logits=task_logits)
               .log_prob(torch.tensor([step.second for step in drained])))
        old = torch.tensor([step.log_prob for step in drained], dtype=torch.float32)
        assert torch.allclose(torch.exp(now - old), torch.ones(len(drained)), atol=1e-5)

    report = optimiser.update(drained)
    assert report.steps == len(drained)
    assert any(not torch.equal(was, is_now) for was, is_now in zip(before, net.parameters()))


def test_the_arms_of_a_run_draw_the_same_fights_however_differently_they_fight():
    """Two arms of one run meet the same fights, which is the whole of what makes a duel paired.

    The draw of a fight — the two budgets, which side is the stronger, the angle and the two forces — must not depend on what is standing on the board, because what is standing is what the policies made of the last fight. Drawn out of one stream with the placement it did depend on it: choosing a site is a choice among the places clearest of the survivors, `random.choice` over a list of n consumes an amount of the stream that depends on n, and one arm therefore stepped off the other's stream part way through an episode.
    """
    from rwintel.learn.arena import Arena

    def drawn(standing):
        arena = Arena.__new__(Arena)
        arena.random = random.Random(4242)
        arena.placement = random.Random(4242 ^ 0x5EED51E5)
        arena.sites = [(float(x), 0.0) for x in range(0, 2000, 100)]
        arena.catalogue = _CATALOGUE
        drawn_fights = []
        for _ in range(6):
            arena._site(_observation(units=standing))
            drawn_fights.append((round(arena.random.uniform(1200.0, 5000.0), 6),
                                 round(arena.random.uniform(0.5, 1.0), 6)))
        return drawn_fights

    # One arm leaves nothing standing between fights, the other leaves a field of survivors: the same six draws either way.
    assert drawn([]) == drawn([_unit(i, 50.0 * i, 40.0) for i in range(1, 12)])


def test_interference_in_one_episode_leaves_the_episodes_already_sealed_alone():
    """An episode's interference marks that episode's decisions and no others.

    A sealed trajectory belongs to an episode that has already closed and already had its own interference marked; it is only still in the buffer because the trainer has not drained it yet. Squad numbers come out of a pool of eight and are handed round fight after fight, so tainting by number alone dropped every clean decision about that number still waiting from earlier episodes.
    """
    rollout = Rollout()
    earlier = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=3, done=True)
    rollout.add((0, 3), earlier)
    rollout.seal(0)

    later = Step(state=[0.0], action=0, mask=[1.0], value=0.5, squad=3)
    rollout.add((0, 3), later)
    rollout.taint(0, [3])

    assert not earlier.tainted, "an episode that has already closed cannot be tainted again by the next one"
    assert later.tainted


def test_a_squad_destroyed_in_a_match_is_paid_the_terminal_the_design_names():
    """Being destroyed is one of the four endings the design says this layer is paid for, and in a match it could not fire.

    The organisation layer retires a squad in the same period its last unit dies, so the tactical layer is never handed a board with an empty squad standing on it: the destruction read as a squad that stopped being reported, its trajectory was cut and bootstrapped from its own value estimate, and losing a whole squad on an errand cost the layer nothing at all.
    """
    rollout = Rollout()
    layer = LearntTactics(None, _CATALOGUE, None, rollout=rollout, instance=0)
    squad = _squad(id=2)
    view = _view([_unit(1, 100, 100)], [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])
    layer._settle(view, [squad], 30000)
    rollout.add((0, squad.id), Step(state=[0.0], action=0, mask=[1.0], value=0.4, squad=squad.id))

    layer.wiped([squad.id])
    trajectory, = rollout.done
    assert trajectory.finished and layer.terminals["wiped"] == 1
    assert abs(trajectory.steps[-1].reward - (WIPED_REWARD - layer.reward.missions.get(squad.id, None).potential
                                              if squad.id in layer.reward.missions else WIPED_REWARD
                                              - 0.0)) < 1.0
    # And a squad that was merged away rather than destroyed is not paid it: that is what the ordinary settle does.
    assert squad.id not in layer.tracked
