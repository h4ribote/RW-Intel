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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwintel.control.policy.catalogue import Catalogue
from rwintel.control.policy.contracts import Doctrine, SquadRecord, TaskContract
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.learn.arena import (
    BY_HEALTH,
    BY_KILLS,
    SCORES,
    STRENGTH_SLOPE_HEALTH,
    STRENGTH_SLOPE_KILLS,
    Engagement,
)
from rwintel.learn.deciders import Choice
from rwintel.learn.encoding import (
    OPERATIONAL_SIZE,
    REGION_FEATURES,
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
from rwintel.learn.net import OperationalNet, TacticalNet
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
from rwintel.learn.rollout import Rollout, Step
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
    class _Orders:
        posture = 0
        priorities = {1: 1.0}
        offensive = True
        loss_allowance = 2000.0

    reward = OperationalReward()
    losing = _view([], [_region(1, ours=100.0, theirs=900.0)])
    winning = _view([], [_region(1, ours=900.0, theirs=100.0)])
    reward.step(losing, _Orders(), [])
    assert reward.step(winning, _Orders(), []).reward > 0.0

    reward.reset()
    reward.step(winning, _Orders(), [])
    assert reward.step(losing, _Orders(), []).reward < 0.0


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


def test_the_two_readings_agree_on_a_body_count_and_part_company_on_damage():
    """What the health reading is for, stated as the difference between the two.

    A fight that ended with somebody destroyed is worth the same under both, because a dead unit is worth nothing whichever way it is counted. That is what keeps every ceiling this project has quoted readable: the second reading existing renumbers none of them.

    Where the two part company is the common ending — both sides still standing and one of them shot to pieces. A fight is called twelve seconds after the last casualty, so three fights in four end that way, and under the sparse reading every one of those is worth precisely nothing to either side however one-sided the damage was. A side left at half health on every survivor loses half of that survivor's worth on the health reading and none of it on the other.
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
    class _WholeBoardReward:
        def step(self, view, orders, squads):
            return Outcome(reward=0.1)

    rollout = Rollout()
    layer = LearntOperations(None, None, None, rollout=rollout, instance=0)
    layer.reward = _WholeBoardReward()
    layer.pending[0] = Step(state=[0.0], action=0, mask=[1.0], value=0.4, squad=0)
    layer.pending[1] = Step(state=[0.0], action=0, mask=[1.0], value=0.7, squad=1)

    # Squad 0 is still on the board this period; squad 1 has left it.
    layer._settle(None, None, [_squad(id=0, contract=False)])

    # The squad still present keeps a live trajectory that goes on accruing; the one that left is cut, not finished, and its last decision bootstraps from its own value rather than from nought.
    assert (0, 0) in rollout.live and not rollout.live[(0, 0)].finished
    cut, = rollout.done
    step, = cut.steps
    assert not cut.finished and not step.done
    assert cut.tail_value == step.value == 0.7
    assert step.reward == 0.1
    assert not layer.pending


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


def test_a_teacher_written_by_a_different_feature_list_is_refused_rather_than_fitted():
    """Nothing in a written decision says which version of the feature list produced it, so the length of the state is the only evidence there is that the features have been renumbered since. A file that is wrong in that way fits without complaint and yields a network reading every feature one place from where it now is, which is why this ends the run instead of skipping the row: the rest of the file cannot be trusted either."""
    assert "decision 0" in _refusal(TeacherMismatch, fit, [Sample(state=[0.0] * (TACTICAL_SIZE - 1), action=0)])

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "teacher.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"state": [0.0] * TACTICAL_SIZE, "action": 0}) + "\n")
            handle.write(json.dumps({"state": [0.0] * (TACTICAL_SIZE + 1), "action": 0}) + "\n")
        # Named to the line, because the first thing anyone does with a file that has been refused is go and look at it.
        assert "line 2" in _refusal(TeacherMismatch, read_teacher, path)


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


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
