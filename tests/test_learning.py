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
from rwintel.learn.arena import Engagement
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
from rwintel.learn.layers import LearntTactics
from rwintel.learn.net import TacticalNet
from rwintel.learn.reward import (
    COMPLETE_REWARD,
    DISCOUNT,
    EXCHANGE_PRIOR,
    EXCHANGE_WEIGHT,
    HOLDING_WEIGHT,
    LOSING_REWARD,
    OperationalReward,
    SPENDING_WEIGHT,
    TacticalReward,
    WIPED_REWARD,
)
from rwintel.learn.rollout import Rollout, Step
from rwintel.learn.train import Optimiser
from rwintel.wire import (
    BLOCK_REGIONS,
    BLOCK_SQUADS,
    BLOCK_UNITS,
    Observation,
    RegionState,
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


def _unit(unit_id, x=0.0, y=0.0, type_index=0, squad=0xFFFF, hostile=0, health=100.0, hit=9999):
    return UnitState(id=unit_id, squad=squad, type_index=type_index, x=x, y=y, health=health,
                     max_health=100.0, built=255, order=255, queued=0, target=0, stance=5,
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


# ---- what a fight was worth ----------------------------------------------------------------

def _fight(our_value, their_value, our_left, their_left):
    return Engagement(index=0, site=(0.0, 0.0), our_value=our_value, their_value=their_value,
                      our_left_value=our_left, their_left_value=their_left)


def test_the_score_of_a_fight_read_from_the_other_side_is_the_same_number_negated():
    """The property the whole measurement rests on, and the reason the score is written in shares rather than in credits.

    Self-play has to average to nought, so that a run against the handwritten layer that averages above nought is the same statement as having beaten it and needs no correction for anything. A score built from a difference of worth would instead pay for having been dealt the stronger side, and the arena deals deliberately uneven sides.
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
        assert -1.0 <= ours.outcome <= 1.0

    # A side that was never built at all is worth nothing and has lost nothing, which has to be a number rather than a division by nought: an engagement whose spawns never arrived on one side still reaches the point where it is scored.
    empty = _fight(0.0, 1200.0, 0.0, 0.0)
    assert empty.outcome == 1.0 and empty.outcome + _fight(1200.0, 0.0, 0.0, 0.0).outcome == 0.0


def test_destroying_the_other_side_without_a_loss_is_the_top_of_the_scale():
    """What fixes the size of the scale, and with it how much a called fight is worth against the errand's own conclusions: a massacre is paid exactly what taking the contracted ground is paid, and no more, so that a layer is never taught to prefer the one to the other."""
    assert _fight(3200.0, 2500.0, 3200.0, 0.0).outcome == 1.0
    assert _fight(2500.0, 3200.0, 0.0, 3200.0).outcome == -1.0
    # And the middle of it is an even trade, whatever the two sides were built to be worth.
    assert _fight(4000.0, 1000.0, 2000.0, 500.0).outcome == 0.0


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


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
