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
from rwintel.control.policy.contracts import Doctrine, SquadRecord, TaskContract
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
from rwintel.learn.deciders import Choice, NetworkTactics
from rwintel.learn.encoding import (
    GLOBAL_SIZE,
    OPERATIONAL_FEATURES,
    OPERATIONAL_SIZE,
    REGION_FEATURES,
    REGION_SIZE,
    SQUAD_FEATURES,
    TACTICAL_ACTIONS,
    TACTICAL_FEATURES,
    TACTICAL_SIZE,
    operational_state,
    region_mask,
    squad_mask,
    tactical_state,
    task_mask,
)
from rwintel.learn.imitation import Sample, TeacherMismatch, fit, read_teacher
from rwintel.learn.layers import LearntOperations, LearntTactics
from rwintel.learn.net import (
    AVOWAL_KEY,
    ENCODING_KEY,
    EncodingRefused,
    OperationalNet,
    TacticalNet,
    avowed,
    encoding_avowal,
    encoding_stamp,
)
from rwintel.learn.ops_arena import SCRIPT_TACTICS, OpsArena
from rwintel.learn.ops_run import frozen_tactics
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
    TacticalReward,
    WIPED_REWARD,
)
from rwintel.learn.rollout import FIGHT_DISCOUNT, FIGHT_TRACE, Rollout, Step, normalise
from rwintel.learn.train import Optimiser, Trainer
from rwintel.control.policy.tactics import Tactics, _Track
from rwintel.eval import arms as eval_arms
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


def test_the_operational_region_rows_are_the_same_ground_read_from_either_side():
    """What holds of the operational cut and what does not, written down so the part that does not cannot rot into folklore.

    The VALUES of a region row are congruent between the two sides with TWO exceptions, and neither is a rounding. They are not the same kind of defect and they do not have the same remedy, which is the whole reason for naming both.

    `distance` is the wire's record of how far a region is from home, and the game measures it from THIS process's base for every region on the board. `build(invert=True)` turns the ownership and the force totals over and leaves that measurement pointing where it pointed, so the inverted side reads our marches as its own and finds the ground it is standing beside on the far side of the map. That one IS repaired, though not here and not by the view builder, which cannot: only the arena knows where each side stages from. The operations arena rewrites every region's distance from the side's own staging region as it builds the other side's view, and the second half of this test applies exactly that rewrite and shows the two sides' rows agreeing afterwards.

    `seen_recently` is the exception NOTHING repairs. It reads the wire's record of when the enemy was last run into in that region, and `build(invert=True)` does not exchange it and cannot, because the region row carries no counterpart field — there is no record of when WE were last seen. The inverted side therefore reads this process's own fog record as its own contact record, and in a constructed arena that leans the same way every period: hostiles stand continuously in the regions this process is attacking and never in the ones it garrisons, so the mirror side is told the enemy is standing on the ground it holds and nowhere near the ground it is attacking. No reordering of the rows repairs that — the field travels with its row — so it needs a field on the wire, or an inverted view that rebuilds the record, or an arena that rebuilds it per side as it already rebuilds the distance from home.

    The row INDEXING is not congruent either, and that is a third defect with a third remedy: the encoder lays the regions out by the map's own numbering and the squads by their global slot, so congruent ground sits at different offsets for the two sides. Hence the whole vectors differ even where every value in them is a pair. A learnt operational layer therefore still reads a frame. When somebody makes the ordering egocentric this test is what will fail, which is exactly when they should be made to come back and read the paragraphs above.
    """
    _, ours, theirs, our_squad, their_squad = _mirrored_board()
    mine = operational_state(ours, None, [our_squad], 30000)
    yours = operational_state(theirs, None, [their_squad], 30000)

    def row(state, slot):
        start = GLOBAL_SIZE + slot * REGION_SIZE
        return state[start:start + REGION_SIZE]

    def differ(ours_view, theirs_view, ours_slot=1, theirs_slot=2):
        """One mirrored pair of region rows read from the two sides: every feature of the pair, and the names of the ones that disagree."""
        left = row(operational_state(ours_view, None, [our_squad], 30000), ours_slot)
        right = row(operational_state(theirs_view, None, [their_squad], 30000), theirs_slot)
        pairs = dict(zip(REGION_FEATURES, zip(left, right)))
        return pairs, [name for name, (a, b) in pairs.items() if abs(a - b) > 1e-9]

    pairs, apart = differ(ours, theirs)
    assert apart == ["distance", "seen_recently"], (
        "the region rows of one mirrored pair differ in %s, where the distance from home and the contact "
        "record are the two that should" % apart)
    assert pairs["seen_recently"] == (1.0, 0.0)
    assert pairs["distance"][0] < pairs["distance"][1], (
        "the inverted side should be reading our own distance from home, which on this board is the longer one")
    assert mine != yours, "the region and squad rows are indexed by the map's numbering, so they cannot agree"

    # Now the repair the operations arena applies as it builds the two views, which is the only one of the two defects that has one: each side's regions measured again from the region that side stages out of, which here is the mirrored pair of regions themselves, and each side's anchor put at its own staging point.
    ours, theirs = rehome(ours, 1, STAGE), rehome(theirs, 2, _reflected(*STAGE))
    pairs, apart = differ(ours, theirs)
    assert apart == ["seen_recently"], (
        "once each side measures from its own staging region only the contact record should still differ, "
        "and these do: %s" % apart)
    # Both ways round, so that what has been shown is congruent rows and not two noughts: our row for the region we stage from against their row for the one they stage from, and our row for theirs against their row for ours.
    assert differ(ours, theirs, ours_slot=2, theirs_slot=1)[1] == ["seen_recently"]


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


def test_an_operational_errand_replaced_by_a_new_contract_takes_no_terminal_and_keeps_its_shaping():
    """Where the operational signal goes when a layer re-draws its region every period, on the real reward and the real buffer rather than by argument.

    Two things are true at once here and each is half of the diagnosis. The shaping of a replaced errand is not cancelled — `close` is only reached from `finish`, and a renewal never reaches `finish` — so what those decisions were paid is the whole movement of the potential over the errand, in the region block's own quantity. And no terminal ever arrives for them: a terminal only comes from outside, at the horizon, and by then this trajectory has been cut and is no longer the squad's. So the errand is paid something, and it is paid nothing of what the arena is scored on.

    The period that does the replacing is paid exactly nought, by construction, and it is the last step of the trajectory. At the discount and trace a whole bounded contest is run at, that leaves its advantage exactly nought and the rest of the errand anchored by the critic's own estimates.

    This is the match's case and only the match's. In a match the region block is the whole of the operational signal and it really is re-based against fresh ground at every contract, so two errands' payments have origins that cannot be compared and running one's advantage backwards into the other's decisions would be wrong. Where a contest reads its own scored ground and hands the reading in, every period is paid the movement of one quantity from the first decision to the last, `renewed` is never set, and none of this happens — see the scored-board tests below, which drive the same layer and the same buffer with a figure handed in.
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

    # And a third in which the layer sends the squad somewhere else, which is what a policy that re-draws every period does whenever it changes its mind.
    layer.pending[squad.id] = Step(state=[0.0], action=0, mask=[1.0], value=0.3, squad=squad.id)
    squad.contract = TaskContract(squad=squad.id, task=Task.ATTACK, target_region=2,
                                  stance=Stance.AGGRESSIVE, cost_budget=1000.0,
                                  deadline_ms=90000, issued_at_ms=24000)
    layer._settle(board(900.0, 100.0), orders, [squad])

    trajectory, = rollout.done
    assert not trajectory.finished and trajectory.reason == "renewed"
    assert [round(step.reward, 9) for step in trajectory.steps] == [0.4, 0.4, 0.0]
    # The errand returned the whole movement of its potential and not nought, which is what an errand ended by a re-tasking is often taken to return; and it returned nothing of the terminal, which is the half that matters.
    assert abs(sum(step.reward for step in trajectory.steps) - 0.8) < 1e-9
    assert not layer.terminals

    steps = rollout.drain()
    assert [round(step.advantage, 9) for step in steps] == [0.9, 0.1, 0.0]
    assert rollout.census.paid_steps == 0 and rollout.census.cut == {"renewed": 1}
    assert rollout.census.zero_advantage == 1


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

    stated = {"layer": "tactics", "encoding": list(TACTICAL_FEATURES)}
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


def test_an_operational_teacher_is_refused_by_the_block_that_moved_and_not_by_a_slot_of_its_state():
    """The operational feature list does not name the state slot by slot and a refusal must not talk as though it did.

    Its state is a handful of aggregates, then one fixed block repeated over twenty-four region slots, then another over eight squad slots — four hundred numbers, described by forty-odd names because each block is named once with the slot counts alongside. So the index in a refusal is an index into the blocks and the length in one is a count of blocks, and reporting either as features would send somebody looking through a four-hundred-wide vector for a forty-fifth slot that decides nothing. The tactical list is the other case and the plain word is exact there, since it names one number of the state apiece; both are checked here so that neither wording can be changed to the other's without this failing.
    """
    moved = "region.distance"
    assert moved in OPERATIONAL_FEATURES, "the block this test renames has itself been renamed"
    decision = {"state": [0.0] * OPERATIONAL_SIZE, "action": 0, "second": 0}
    head = {"layer": "operations", "encoding": list(OPERATIONAL_FEATURES)}

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
        head = {"layer": "tactics", "encoding": list(TACTICAL_FEATURES)}
        renamed = dict(head, encoding=["target_dx" if name == "target_ahead_of_home" else name
                                       for name in TACTICAL_FEATURES])
        refusal = _refusal(TeacherMismatch, read_teacher,
                           _teacher(folder, renamed, {"state": [0.0] * TACTICAL_SIZE, "action": 0}))
        assert f"feature {TACTICAL_FEATURES.index('target_ahead_of_home')}" in refusal


def test_two_collecting_runs_joined_into_one_teacher_are_read_and_re_checked_at_the_join():
    """The only way a teacher file gets bigger is by joining two of them, since the collecting run opens its file with truncation and cannot append. A joined file therefore carries the second run's head in the middle of it, and what happens at that line decides whether joining is a thing anybody can do.

    Three things happen there. The join reads back whole, which is what it did before a file stated anything and has to go on doing. The second head is CHECKED rather than waved through, because it is the only line in the file that says whether the two halves were collected under the same encoding — a join across an encoding change is two different readings of the board in one file, and fitting to it produces a network that is wrong about half of what it saw. And a line that is neither a decision nor a head is refused by name and line like everything else malformed here, rather than surfacing as a missing key from somewhere inside the reader with nothing said about which file it came from.
    """
    head = {"layer": "tactics", "encoding": list(TACTICAL_FEATURES)}
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
            ["target_dx" if name == "target_ahead_of_home" else name for name in TACTICAL_FEATURES])
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
            ["target_dx" if name == "target_ahead_of_home" else name for name in TACTICAL_FEATURES])
        refusal = _refusal(EncodingRefused, avowed, stale, TacticalNet(), words)
        assert "already states the feature list" in refusal
        # Nor twice, which is the same refusal reached from the other side: a file that has been avowed states a list like any other and is protected by it like any other.
        once = avowed(state, TacticalNet(), words)
        assert "already states the feature list" in _refusal(EncodingRefused, avowed, once, TacticalNet(), words)


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
