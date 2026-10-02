"""What a recorded run promises, held to.

A dataset is only worth keeping if it can be learnt from after the code that wrote it has moved on, so the promises pinned here are the ones that make that true and that would otherwise fail silently: that a run written by another encoding is refused rather than read; that what each state was encoded from rebuilds it exactly, so a changed encoding costs a pass over the data rather than the data; that every reward can be priced again from its signals under other terms and comes out as the layer would have paid it; that returns and advantages computed from the record are the ones the run trained on; that the split a fit is judged on is by episode; and that what the policy drew from, what interfered and how each episode ended survive into the record.

Nothing here launches a game.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rwintel.control.policy.contracts import Role
from rwintel.control.policy.options import Options
from rwintel.control.policy.encoding import (
    ECONOMIC_REVISION,
    OPERATIONAL_REVISION,
    TACTICAL_ACTIONS,
    TACTICAL_REVISION,
    TACTICAL_SIZE,
    Access,
    Investment,
    Offer,
    TransportView,
    economic_state,
    fingerprint,
    forces,
    lay_out,
    operational_state,
    tactical_state,
)
from rwintel.control.policy.view import Sighting, build as build_view
from rwintel.control.session import EpisodeSettings
from rwintel.learn import materials
from rwintel.learn.dataset import (
    FORMAT,
    SIGNALS,
    Dataset,
    DatasetMismatch,
    Recorder,
    fold,
    inspect,
    widths,
    reencode,
    verify,
)
from rwintel.learn.deciders import Choice, Labelled, PinnedDeparture, PinnedOperations, ScriptChoice
from rwintel.learn.layers import ECONOMY_KEY, LearntEconomy, LearntOperations, LearntTactics, context_of
from rwintel.learn.reward import BY_KILLS, CLOSED, SHARE, EconomicTerms, OperationalTerms, TacticalTerms
from rwintel.learn.rollout import Rollout, Step
from rwintel.wire import Status, Task

from test_learning import (
    _CATALOGUE,
    _ECONOMY_CATALOGUE,
    _ECONOMY_TYPES,
    _Counting,
    _EconomySession,
    _Fixed,
    _Orders,
    _economic_board,
    _economy_orders,
    _economy_view,
    _observation,
    _ops_board,
    _refusal,
    _region,
    _skirmish,
    _squad,
    _unit,
    _view,
)


def _episode(instance: int, attempt: int = 1, ending: str = "finished", context: dict = None) -> dict:
    episode = {"instance": instance, "attempt": attempt, "episode": attempt, "arm": "test", "map": "",
               "seed": -1, "ending": ending}
    if context is not None:
        episode["context"] = context
    return episode


def _decision(state, label=-1, reward=0.0, value=0.0, periods=1, done=False, squad=0, meta=None, soft=()):
    return Step(state=list(state), action=max(0, label), mask=[1.0] * TACTICAL_ACTIONS, label=label, reward=reward,
                value=value, periods=periods, done=done, squad=squad, meta=dict(meta or {}), soft=list(soft),
                signals=[], probabilities=[1.0 / TACTICAL_ACTIONS] * TACTICAL_ACTIONS)


# ---- the encoding a run was written by -------------------------------------------------------

def _access():
    """Region 0 walked to, region 1 across water and reached by the hovercraft in transport slot 1."""
    return Access(walk={0}, lift={1: [1]}, coastal={0, 1},
                  transports=[TransportView(slot=0),
                              TransportView(slot=1, valid=True, x=60.0, y=40.0, aboard=1, capacity=4, movement="HOVER",
                                            carries=True)])


def _fixed_boards():
    """One board per layer, made of fixed numbers, encoded with no combat table so that nothing outside this file moves them."""
    units = [_unit(1, 100.0, 100.0, squad=0), _unit(2, 120.0, 110.0, squad=0, health=60.0, hit=500),
             _unit(3, 140.0, 95.0, squad=0), _unit(9, 300.0, 120.0, hostile=1, type_index=1),
             _unit(10, 280.0, 160.0, hostile=1)]
    regions = [_region(0, 0.0, 0.0, ours=400.0, distance=0.0), _region(1, 400.0, 100.0, ours=200.0, theirs=900.0)]
    view = _view(units, regions)
    squad = _squad()
    members = [s for s in view.ours if s.unit.id in squad.members]
    threats = list(view.enemies)
    tactical = tactical_state(squad, members, threats, 350.0, 700.0, view, 30000, None)
    orders = _Orders()
    operational = operational_state(view, orders, [squad], 30000, (1,), squad=squad, combat=None, access=_access())
    offers = [Offer(kind=Investment.UNIT, price=350.0, type_index=0, efficiency=0.5, unit_efficiency=2.0, fighting=True,
                    role=Role.ARMOUR, role_rank=1.0, range=130.0, tier=1),
              Offer(kind=Investment.EXTRACTOR, price=700.0, region=1, home=True, safety=120.0, open_points=2),
              Offer(kind=Investment.RAISE, price=2000.0, stage=0, gain=0.4, funded=True)]
    economic = economic_state(_economic_board(credits=1800.0, income=12.0, units=7, unit_cap=100, factories=1,
                                              builders=2, idle_builders=1, extractors=2, target_mix={Role.ARMOUR: 1.0}),
                              lay_out([Offer(kind=Investment.STOP)] + offers))
    return {"tactics": tactical, "operations": operational, "economy": economic}


def _digest(state) -> str:
    return hashlib.sha256(json.dumps([round(value, 9) for value in state]).encode("utf-8")).hexdigest()[:16]


#: What each layer's encoding writes for its fixed board, by revision. A change to what a feature means that leaves this as it was cannot happen without the change being seen here, and a change that alters it is a new revision.
PINNED = {
    ("tactics", 4): "74dbcd9336506794",
    ("operations", 5): "626ef8cd3b546285",
    ("economy", 2): "20e0999b55992d3c",
}


def test_the_encoding_of_a_fixed_board_is_pinned_to_its_revision():
    """A state is only readable beside the encoding that wrote it, and the lengths of the vectors say nothing about a feature whose meaning changed in place. So each layer's encoding of one fixed board is pinned to the encoding's revision: an encoding that writes anything else for it is a new revision, its fingerprint changes, and every run written by the old one is refused until it is encoded again from its materials."""
    revisions = {"tactics": TACTICAL_REVISION, "operations": OPERATIONAL_REVISION, "economy": ECONOMIC_REVISION}
    for layer, state in _fixed_boards().items():
        assert _digest(state) == PINNED[(layer, revisions[layer])], (layer, _digest(state))
    assert len({fingerprint(layer) for layer in revisions}) == 3


# ---- writing and reading a run ---------------------------------------------------------------

def _record(directory, layer, episodes, discount=1.0, trace=1.0, header=None, shard_decisions=7):
    """Writes the given episodes, each a list of trajectories, through a recorder with small shards so that reading has to join several."""
    recorder = Recorder(directory, layer, header or {}, shard_decisions=shard_decisions)
    for episode, trajectories in episodes:
        recorder.accept(trajectories, episode, discount, trace)
    recorder.close()
    return recorder


def test_a_recorded_run_reads_back_with_the_returns_and_advantages_it_was_collected_with():
    """The estimates a learner takes from a record have to be the ones the run's own optimiser took: an errand that ended closes at nought, one cut off is bootstrapped from the value it was cut at, a decision followed within its own period is not discounted, one held for several is discounted for all of them, and a squad found destroyed after its last decision closes the trajectory on that decision."""
    draw = random.Random(3)
    sealed = []
    rollout = Rollout(discount=0.9, trace=0.8, sink=lambda trajectories, episode, d, t: sealed.append(
        (trajectories, episode, d, t)))

    def step(periods=1, done=False):
        return _decision([draw.uniform(-1, 1)] * TACTICAL_SIZE, label=draw.randrange(TACTICAL_ACTIONS),
                         reward=draw.uniform(-1, 1), value=draw.uniform(-1, 1), periods=periods, done=done)

    for periods in (1, 0, 3, 1):
        rollout.add((0, 0), step(periods))
    rollout.add((0, 0), step(done=True))
    for _ in range(3):
        rollout.add((0, 1), step())
    rollout.cut((0, 1), tail_value=0.4)
    for _ in range(2):
        rollout.add((0, 2), step())
    assert rollout.close_with((0, 2), -1.0)
    rollout.add((1, 0), step())
    rollout.cut_all(owner=1)
    rollout.seal(0, _episode(0))
    rollout.seal(1, _episode(1))
    online = rollout.drain(keep_tainted=True)

    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", [(episode, trajectories) for trajectories, episode, _, _ in sealed],
                discount=0.9, trace=0.8)
        dataset = Dataset.open([folder])
        assert len(dataset) == len(online) == 11 and len(dataset.episodes) == 2
        advantages, returns = dataset.estimates()
        # The record keeps rewards and values at single precision, so the estimates agree to that precision.
        recorded = [s for ts, _, _, _ in sealed for t in ts for s in t.steps]
        for index, original in enumerate(recorded):
            assert abs(advantages[index] - original.advantage) < 1e-5
            assert abs(returns[index] - original.ret) < 1e-5
        assert list(dataset.arrays["t_finished"]) == [True, False, True, False]


def test_every_layers_reward_is_priced_again_from_its_signals():
    """The signals beside each decision are what its reward was priced from, so priced again at the run's own terms they give the reward the layer paid, and priced at other terms they give what the layer would have paid had it been run under those: other weights, another discount, the other reading of a fight. That is what lets runs paid differently be learnt from together."""
    def tactical(terms):
        rollout = Rollout()
        layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0, terms=terms)
        squad = _squad()
        for index, ours in enumerate((200.0, 600.0, 300.0)):
            squad.losses = 100.0 * index
            layer.decide(_view(_skirmish().observation.unit_states, [_region(1, 400.0, 100.0, ours=ours, theirs=900.0)]),
                         [squad], 21000 + 200 * index)
        layer.finish(squad, -0.25, 0.5, "called", engagement=2)
        return [s for t in rollout.done for s in t.steps]

    paid_at = TacticalTerms(discount=1.0)
    other = TacticalTerms(discount=0.9, holding_weight=0.2, outcome_weight=0.5, score=BY_KILLS)
    steps = tactical(paid_at)
    assert steps and all(abs(paid_at.reward(s.signals) - s.reward) < 1e-12 for s in steps)
    for priced, actual in zip(steps, tactical(other)):
        assert abs(other.reward(priced.signals) - actual.reward) < 1e-12

    def operational(terms):
        from rwintel.control.policy.contracts import MissionReport

        rollout = Rollout()
        layer = LearntOperations(None, _CATALOGUE, _Counting(), rollout, instance=0, terms=terms, review_ms=4000)
        squad = _squad(contract=False, losses=0.0)
        for index in range(6):
            t = 30000 + 2000 * index
            reports = [MissionReport(squad=0, status=Status.ACTIVE, losses=50.0 * index, destroyed=120.0 * index)]
            layer.decide(_ops_board(t), _Orders(), [squad], reports, t)
        layer.close(score=0.4)
        return [s for t in rollout.done for s in t.steps]

    paid_at = OperationalTerms()
    other = OperationalTerms(discount=0.9, shaping=SHARE, value_flow=0.05, ground_flow=0.02, local_exchange=0.1,
                             achievement_weight=1.0, terminal_weight=0.5)
    steps = operational(paid_at)
    assert len(steps) >= 2 and all(abs(paid_at.reward(s.signals) - s.reward) < 1e-12 for s in steps)
    assert int(steps[-1].signals[-1][0]) == CLOSED and steps[-1].done
    for priced, actual in zip(steps, operational(other)):
        assert abs(other.reward(priced.signals) - actual.reward) < 1e-12

    def economic(terms):
        rollout = Rollout()
        layer = LearntEconomy(_EconomySession(), _ECONOMY_CATALOGUE, None, rollout, instance=0,
                              options=replace(Options(), choose=False), terms=terms)
        for t in (30000, 32000, 34000):
            layer.decide(_economy_view(t), _economy_orders(), [])
        layer.close(score=-0.3)
        return [s for t in rollout.done for s in t.steps]

    paid_at = EconomicTerms()
    other = EconomicTerms(discount=0.5, shaping=SHARE, value_flow=0.2, ground_flow=0.0, terminal_weight=2.0)
    steps = economic(paid_at)
    assert len(steps) >= 3 and all(abs(paid_at.reward(s.signals) - s.reward) < 1e-12 for s in steps)
    for priced, actual in zip(steps, economic(other)):
        assert abs(other.reward(priced.signals) - actual.reward) < 1e-12


def test_the_boards_a_predictor_is_fitted_on_are_the_periods_of_matches_that_ended_with_a_score():
    """Each period's board is taken once with the score its match ended with and the time left in it; a match stopped before its end has no score and gives nothing, and neither does a board whose time left is not known."""
    from rwintel.learn.predictor import samples_of

    sealed = []
    times = (30000, 32000, 34000, 36000)
    _recorded_economy(sealed, score=0.25, times=times, session=_TimedSession())
    _recorded_economy(sealed, score=None, times=times, session=_TimedSession())
    _recorded_economy(sealed, score=0.5, times=times)
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "economy", [(dict(episode, episode=index + 1), trajectories)
                                    for index, (episode, trajectories) in enumerate(sealed)])
        samples = samples_of([Dataset.open([folder])])
    assert set(samples.score) == {0.25} and set(samples.match) == {0}
    assert samples.remaining == [900.0 - t / 1000.0 for t in times[1:]]


# ---- what a state was encoded from -----------------------------------------------------------

def _recorded_tactics(sealed):
    rollout = Rollout(discount=1.0, trace=1.0, sink=lambda ts, e, d, t: sealed.append((e, ts)))
    layer = LearntTactics(None, _CATALOGUE, _Fixed(), rollout=rollout, instance=0)
    squad = _squad()
    for index in range(4):
        squad.losses = 80.0 * index
        layer.decide(_skirmish(), [squad], 21000 + 200 * index)
    layer.finish(squad, 0.1, 0.3, "called", engagement=5)
    layer.close()
    return layer


def _recorded_operations(sealed):
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append((e, ts)))
    layer = LearntOperations(None, _CATALOGUE, _Counting(), rollout, instance=0, review_ms=4000)
    squad = _squad(contract=False, losses=0.0)
    for index in range(6):
        t = 30000 + 2000 * index
        layer.decide(_ops_board(t), _Orders(), [squad], [], t)
    layer.close()
    return layer


class _TimedSession(_EconomySession):
    """A session that states the match's time limit, so that the layer knows the time left."""

    settings = EpisodeSettings(max_seconds=900)


def _recorded_economy(sealed, terms=None, score=None, times=(30000, 32000), session=None):
    rollout = Rollout(discount=terms.discount if terms is not None else 1.0,
                      sink=lambda ts, e, d, t: sealed.append((e, ts)))
    layer = LearntEconomy(session or _EconomySession(), _ECONOMY_CATALOGUE, None, rollout, instance=0,
                          options=replace(Options(), choose=False), terms=terms)
    for t in times:
        layer.decide(_economy_view(t), _economy_orders(), [])
    layer.close(score=score)
    return layer


def test_the_materials_rebuild_every_layers_state_exactly():
    """Each layer's materials, written as rows and read back, rebuild by the encoding in force to exactly the state that was decided on: the tactical layer's with the combat table's prediction in it, the operational layer's with the forces, the orders and the starting regions, the economy's with every comparison between offers. The type and combat tables travel with the episode, so a report the combat table reads that has grown since cannot change what is rebuilt."""
    for layer_name, record in (("tactics", _recorded_tactics), ("operations", _recorded_operations),
                               ("economy", _recorded_economy)):
        sealed = []
        layer = record(sealed)
        episode, trajectories = sealed[0]
        assert episode["context"]["types"]
        context = materials.Context.of(episode["context"]["types"], episode["context"]["combat"])
        steps = [s for t in trajectories for s in t.steps]
        assert steps and all(s.materials is not None for s in steps)
        for step in steps:
            rows = {name: np.asarray(table, dtype=np.float64).reshape(-1, len(materials.TABLES[layer_name][name]))
                    for name, table in materials.tables(step.materials).items()}
            assert materials.rebuild(layer_name, rows, context) == step.state, layer_name

        with tempfile.TemporaryDirectory() as folder:
            _record(folder, layer_name, [(episode, trajectories)])
            dataset = Dataset.open([folder], with_materials=True)
            assert verify(dataset) == (len(steps), 0, 0)
            target = os.path.join(folder, "again")
            assert reencode(folder, target) == len(steps)
            assert np.array_equal(Dataset.open([target]).arrays["state"], dataset.arrays["state"])


def test_where_a_squad_could_get_to_and_by_which_transport_is_rebuilt_from_its_materials():
    """The operational state reads which regions the squad could walk to, which transport could carry it where, and the transport slots themselves; all of it is kept beside the decision, so the state rebuilds exactly, and the board from a run that kept none rebuilds as every region walkable and no transport anywhere."""
    units = [_unit(1, 100.0, 100.0, squad=0), _unit(2, 120.0, 110.0, squad=0), _unit(9, 300.0, 120.0, hostile=1)]
    regions = [_region(0, 0.0, 0.0, ours=400.0, distance=0.0), _region(1, 400.0, 100.0, ours=200.0, theirs=900.0)]
    view = _view(units, regions)
    squad = _squad()
    context = materials.Context(catalogue=_CATALOGUE, combat=None)
    for access in (_access(), None):
        taken = materials.operational(view, _Orders(), [squad], (1,), squad, 30000, access)
        rows = {name: np.asarray(table, dtype=np.float64).reshape(-1, len(materials.TABLES["operations"][name]))
                for name, table in materials.tables(taken).items()}
        expected = operational_state(view, _Orders(), [squad], 30000, (1,), squad=squad, combat=None,
                                     present=forces(view, [squad]), access=access)
        assert materials.rebuild("operations", rows, context) == expected


def test_a_state_rebuilt_differently_is_counted_as_a_mismatch():
    """The check that a dataset can be encoded again is only a check if it can fail: a recorded state that no longer matches what its materials rebuild to is counted rather than passed over."""
    sealed = []
    _recorded_tactics(sealed)
    episode, trajectories = sealed[0]
    trajectories[0].steps[0].state[0] += 0.5
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", [(episode, trajectories)])
        checked, mismatched, unbuildable = verify(Dataset.open([folder], with_materials=True))
        assert mismatched == 1 and unbuildable == 0 and checked > 1


def test_a_run_written_by_another_encoding_or_layer_is_refused():
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", [(_episode(0), [_trajectory(3)])])
        Dataset.open([folder])
        with open(os.path.join(folder, "run.json"), encoding="utf-8") as handle:
            header = json.load(handle)
        header["fingerprint"] = "0" * 64
        with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(header, handle)
        refused = _refusal(DatasetMismatch, Dataset.open, [folder])
        assert "reencode" in refused
        # Asked for explicitly, as the tools that encode it again do, it opens.
        assert len(Dataset.open([folder], check=False)) == 3
        assert "operations" in _refusal(DatasetMismatch, Dataset.open, [folder], "operations", False)


#: Each layer's layout -the widths of its decision arrays and the columns of its materials tables- by layout version. A layout that changes without its version changing is caught here.
LAYOUTS = {
    ("tactics", 2): "a13031f84820f47d",
    ("operations", 3): "d67e8460f8598a4d",
    ("economy", 2): "deed7f9552511eb4",
}


def _layout(layer):
    _, first, second = widths(layer)
    return [first, first * second, {name: list(columns) for name, columns in materials.TABLES[layer].items()},
            list(SIGNALS[layer])]


def test_each_layers_layout_is_pinned_to_its_layout_version():
    for layer in ("tactics", "operations", "economy"):
        digest = hashlib.sha256(json.dumps(_layout(layer), sort_keys=True).encode("utf-8")).hexdigest()[:16]
        assert (layer, FORMAT[layer]) in LAYOUTS and LAYOUTS[(layer, FORMAT[layer])] == digest, (layer, digest)


def test_a_run_in_another_layout_of_its_layer_is_refused_for_reading_and_for_encoding_again():
    """Each layer has its own layout version, so a change to one layer's arrays leaves the others' runs readable; a run in another layout of its own layer is refused by name rather than failing on a missing array."""
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", [(_episode(0), [_trajectory(3)])])
        with open(os.path.join(folder, "run.json"), encoding="utf-8") as handle:
            header = json.load(handle)
        assert header["format"] == FORMAT["tactics"]
        header["format"] = FORMAT["tactics"] + 1
        with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as handle:
            json.dump(header, handle)
        assert "layout" in _refusal(DatasetMismatch, Dataset.open, [folder], None, False)
        assert "layout" in _refusal(DatasetMismatch, reencode, folder, os.path.join(folder, "again"))


def _trajectory(count, label=0, key=(0, 0), finished=True):
    from rwintel.learn.rollout import Trajectory

    steps = [_decision([0.1 * i] * TACTICAL_SIZE, label=label, done=finished and i == count - 1) for i in range(count)]
    return Trajectory(key=key, steps=steps, finished=finished)


# ---- the split a fit is judged on ------------------------------------------------------------

def test_episodes_are_split_whole_and_keep_their_side_as_data_is_added():
    """A fit is judged on episodes it was not fitted to, never on rows: two decisions of one fight are nearly one decision. Which side an episode is on is a digest of its name, so adding data never moves an episode across."""
    sides = [fold("run", instance, attempt) for instance in range(40) for attempt in range(1, 26)]
    assert 0.05 < sum(sides) / len(sides) < 0.15

    with tempfile.TemporaryDirectory() as folder:
        first, second = os.path.join(folder, "a"), os.path.join(folder, "b")
        _record(first, "tactics", [(_episode(i), [_trajectory(4, key=(i, 0))]) for i in range(30)])
        _record(second, "tactics", [(_episode(i), [_trajectory(2, key=(i, 0))]) for i in range(30)])
        alone = Dataset.open([first])
        together = Dataset.open([first, second])
        held = together.held_out()
        episodes = together.episode_index()
        for episode in set(episodes.tolist()):
            assert len(set(held[episodes == episode].tolist())) == 1
        assert np.array_equal(alone.held_out(), held[:len(alone)])
        assert held.any() and not held.all()


def test_a_run_encoded_again_keeps_the_split_it_was_written_with():
    """Encoding a run again writes it under another name, and the split is a digest of the episode's name, so the name the episodes are split by is the run that first wrote them: what a fit was judged on before is what it is judged on after."""
    sealed = []
    for _ in range(12):
        _recorded_tactics(sealed)
    with tempfile.TemporaryDirectory() as folder:
        first = os.path.join(folder, "first")
        _record(first, "tactics", [(dict(episode, instance=index), trajectories)
                                   for index, (episode, trajectories) in enumerate(sealed)])
        again, twice = os.path.join(folder, "again"), os.path.join(folder, "twice")
        reencode(first, again)
        reencode(again, twice)
        original = Dataset.open([first])
        for target in (again, twice):
            encoded = Dataset.open([target])
            assert encoded.runs[0]["lineage"] == "first" and encoded.runs[0]["run"] != "first"
            assert np.array_equal(encoded.held_out(salt="x"), original.held_out(salt="x"))
            assert np.array_equal(encoded.held_out(0.5), original.held_out(0.5))
        assert original.held_out(0.5).any() and not original.held_out(0.5).all()


# ---- what the record says about how a decision was taken --------------------------------------

def test_every_decider_says_what_distribution_it_drew_from():
    """A rule is a distribution that puts everything on its answer, a rule that draws evenly is the even distribution, and an exploring pupil is a mixture whose probability for the pair it played, region and plan together, is the mixture's, the plan drawn evenly under the plan mask of the region played."""
    mask = [1.0] * TACTICAL_ACTIONS
    pinned = PinnedDeparture(3).choose([0.0] * TACTICAL_SIZE, mask)
    assert pinned.probabilities == [1.0 if i == 3 else 0.0 for i in range(TACTICAL_ACTIONS)]

    from rwintel.control.policy.encoding import OPERATIONAL_PLANS, OPERATIONAL_SIZE

    regions = [1.0, 1.0, 0.0, 1.0] + [0.0] * 20
    # Two plans are open in regions 0 and 1 and three in region 3, so how evenly a plan is drawn depends on the region.
    plans = [[1.0 if (p in (0, 2) and r in (0, 1)) or (p in (0, 2, 5) and r == 3) else 0.0
              for p in range(OPERATIONAL_PLANS)] for r in range(24)]
    drawn = PinnedOperations("random", seed=1).choose([0.0] * OPERATIONAL_SIZE, 0, regions, plans)
    assert abs(sum(drawn.probabilities) - 1.0) < 1e-12 and drawn.probabilities[2] == 0.0
    assert abs(math.exp(drawn.log_prob) - 1.0 / 3.0) < 1e-12

    table = [[0.25 if p == 0 else 0.75 if p == 2 else 0.0 for p in range(OPERATIONAL_PLANS)] for _ in regions]
    first = [0.5, 0.5, 0.0, 0.0] + [0.0] * 20

    class _Pupil:
        def choose(self, state, slot, region_mask, plan_masks):
            return Choice(action=0, second=2, probabilities=list(first), second_probabilities=list(table[0]),
                          second_by_first=table)

    exploring = Labelled(_Pupil(), explore=0.4, seed=5)
    seen = set()
    for _ in range(80):
        choice = exploring.choose([0.0] * OPERATIONAL_SIZE, 0, regions, plans)
        region, plan = choice.action, choice.second
        even = 1.0 / sum(plans[region])
        joint = 0.6 * first[region] * table[region][plan] + 0.4 * (1.0 / 3.0) * even
        assert plans[region][plan] > 0
        assert abs(math.exp(choice.total_log_prob) - joint) < 1e-12
        assert abs(sum(choice.second_probabilities) - 1.0) < 1e-12
        seen.add(region)
    assert 3 in seen


def test_what_was_played_and_what_the_judge_would_have_chosen_are_written_apart():
    """Every recorded decision carries the layer's judge's answer beside what was played, so a record of a network or of an exploring pupil is also a record of what the script would have done; a person's decision from a replay is its own label."""
    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append(ts))
    layer = LearntTactics(None, _CATALOGUE, _Fixed(action=6), rollout=rollout, instance=0)
    layer.decide(_skirmish(), [_squad()], 21000)
    layer.decide(_skirmish(), [_squad()], 21200)
    layer.close()
    step = sealed[0][0].steps[0]
    assert step.action == 6 and step.label == layer.judge.choose(step.state)
    assert step.soft and abs(sum(step.soft) - 1.0) < 1e-9

    class _Person:
        teaches = True

        def choose(self, state, slot, region_mask, plan_masks):
            return Choice(action=2, second=[i for i, a in enumerate(plan_masks[2]) if a][0], meta={"source": "human"})

    rollout = Rollout()
    layer = LearntOperations(None, _CATALOGUE, _Person(), rollout, instance=0)
    layer.decide(_ops_board(30000), _Orders(), [_squad(contract=False, losses=0.0)], [], 30000)
    taken = layer.pending[0]
    assert (taken.label, taken.second_label) == (taken.action, taken.second) and not taken.soft


def test_the_script_answers_as_its_layers_judge_and_explores_around_it_as_a_mixture():
    """The script made into a decider plays the very judge of the layer it is handed to, so with no exploration it is the script; explored at a share, what each decision records is the mixture it was drawn from, the judge's answer at what is left plus the share spread evenly over the legal actions, in every layer."""
    state = _fixed_boards()["tactics"]
    script = ScriptChoice()
    layer = LearntTactics(None, _CATALOGUE, script)
    assert script.choose(state, [1.0] * TACTICAL_ACTIONS).action == int(layer.judge.choose(state))
    assert "bound" in _refusal(RuntimeError, ScriptChoice().choose, state, [1.0] * TACTICAL_ACTIONS)

    def mixed(label, mask, share=0.3):
        legal = sum(1 for allowed in mask if allowed > 0)
        return [(1.0 - share) * (1.0 if index == label else 0.0) + (share / legal if allowed > 0 else 0.0)
                for index, allowed in enumerate(mask)]

    def check(steps, layer_name):
        assert steps, layer_name
        for step in steps:
            expected = mixed(step.label, step.mask)
            assert np.allclose(step.probabilities, expected), layer_name
        return steps

    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append(ts))
    layer = LearntTactics(None, _CATALOGUE, Labelled(ScriptChoice(), explore=0.3, seed=2), rollout=rollout, instance=0)
    for index in range(30):
        layer.decide(_skirmish(), [_squad()], 21000 + 200 * index)
    layer.close()
    for step in check([s for ts in sealed for t in ts for s in t.steps], "tactics"):
        assert abs(math.exp(step.log_prob) - step.probabilities[step.action]) < 1e-9
    assert any(step.action != step.label for ts in sealed for t in ts for step in t.steps)

    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append(ts))
    layer = LearntOperations(None, _CATALOGUE, Labelled(ScriptChoice(), explore=0.3, seed=4), rollout, instance=0,
                             review_ms=2000)
    squad = _squad(contract=False, losses=0.0)
    for index in range(40):
        t = 30000 + 2000 * index
        squad.contract = None
        layer.decide(_ops_board(t), _Orders(), [squad], [], t)
    layer.close()
    steps = check([s for ts in sealed for t in ts for s in t.steps], "operations")
    for step in steps:
        region, plan = step.action, step.second
        width = len(step.second_mask) // len(step.mask)
        row = step.second_mask[region * width:(region + 1) * width]
        plans = [1.0 / sum(row) if allowed > 0 else 0.0 for allowed in row]
        regions = [1.0 / sum(step.mask) if allowed > 0 else 0.0 for allowed in step.mask]
        script = 1.0 if (region, plan) == (step.label, step.second_label) else 0.0
        joint = 0.7 * script + 0.3 * regions[region] * plans[plan]
        assert abs(math.exp(step.log_prob) - joint) < 1e-9
        assert abs(sum(step.second_probabilities) - 1.0) < 1e-9
    assert any(step.action != step.label for step in steps)

    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append(ts))
    layer = LearntEconomy(_EconomySession(), _ECONOMY_CATALOGUE, Labelled(ScriptChoice(), explore=0.3, seed=6), rollout,
                          instance=0, options=replace(Options(), choose=False))
    for index in range(12):
        layer.decide(_economy_view(30000 + 2000 * index), _economy_orders(), [])
    layer.close()
    for step in check([s for ts in sealed for t in ts for s in t.steps], "economy"):
        assert abs(math.exp(step.log_prob) - step.probabilities[step.action]) < 1e-9


def test_a_pinned_departure_is_recorded_beside_what_the_judge_would_have_done():
    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append(ts))
    layer = LearntTactics(None, _CATALOGUE, PinnedDeparture(6), rollout=rollout, instance=0)
    for index in range(3):
        layer.decide(_skirmish(), [_squad()], 21000 + 200 * index)
    layer.close()
    steps = [s for ts in sealed for t in ts for s in t.steps]
    assert len(steps) == 3
    for step in steps:
        assert step.action == 6 and step.probabilities == [1.0 if i == 6 else 0.0 for i in range(TACTICAL_ACTIONS)]
        assert step.label == int(layer.judge.choose(step.state)) and step.label != 6 and step.log_prob == 0.0


def test_a_collecting_run_takes_one_player():
    """What played a collected run is named once in its header, so a run that names two players, pins two departures, or pins one in a layer that has none is refused before anything is started."""
    from rwintel.learn.__main__ import _collect_checks, build_parser
    from rwintel.wire import Deviation

    def checked(argv):
        return _collect_checks(build_parser().parse_args(argv))

    assert checked(["collect", "--layer", "tactics", "--pin", "hold", "--both-sides", "--explore", "0.1"]) == Deviation.HOLD
    assert checked(["collect", "--layer", "operations", "--rule", "home", "--explore", "0.3"]) is None
    for argv in (["collect", "--layer", "tactics", "--pin", "hold,withdraw"],
                 ["collect", "--layer", "economy", "--pin", "hold"],
                 ["collect", "--layer", "tactics", "--pin", "hold", "--student", "x.pt"],
                 ["collect", "--layer", "operations", "--rule", "home", "--student", "x.pt"],
                 ["collect", "--layer", "tactics", "--rule", "home"],
                 ["collect", "--layer", "operations", "--both-sides"],
                 ["collect", "--layer", "tactics", "--explore", "1.5"]):
        assert _refusal(SystemExit, checked, argv), argv


def test_both_sides_of_an_arena_are_sealed_once_with_each_sides_last_decision_in_its_own_trajectory():
    """Both sides of an arena record into one rollout under one instance, and the episode is sealed by whichever side closes first. Each side's last decision is still waiting on payment when the episode ends, so both are filed before either closes: the episode is sealed once, whole, with each side's decisions in one trajectory."""
    from rwintel.learn.arena import Arena, Statistics

    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append((e, ts)), retain=False)
    arena = Arena.__new__(Arena)
    arena.statistics = Statistics()
    arena.tactics = LearntTactics(None, _CATALOGUE, None, rollout=rollout, instance=0, status_terminals=False)
    arena.opponent = LearntTactics(None, _CATALOGUE, None, rollout=rollout, instance=0, status_terminals=False)
    observation = _contact()
    ours, theirs = _squad(), _squad(members=(7, 8, 9), id=1, x=300.0, y=120.0)
    for index in range(4):
        arena.tactics.decide(build_view(observation, _CATALOGUE, None), [ours], 21000 + 200 * index)
        arena.opponent.decide(build_view(observation, _CATALOGUE, None, invert=True), [theirs], 21000 + 200 * index)
    arena.close()
    assert len(sealed) == 1
    _, trajectories = sealed[0]
    assert sorted(t.key for t in trajectories) == [(0, 0), (0, 1)]
    assert all(len(t.steps) == 4 for t in trajectories)


def test_a_decision_taken_on_the_board_read_from_the_other_side_rebuilds_from_its_materials():
    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append((e, ts)))
    layer = LearntTactics(None, _CATALOGUE, None, rollout=rollout, instance=0, status_terminals=False)
    theirs = _squad(members=(7, 8, 9), id=1, x=300.0, y=120.0)
    for index in range(3):
        theirs.losses = 50.0 * index
        layer.decide(build_view(_contact(), _CATALOGUE, None, invert=True), [theirs], 21000 + 200 * index)
    layer.close()
    episode, trajectories = sealed[0]
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", [(episode, trajectories)])
        assert verify(Dataset.open([folder], with_materials=True)) == (3, 0, 0)


def test_each_decision_is_named_by_the_policy_that_played_its_side():
    with tempfile.TemporaryDirectory() as folder:
        both, one = os.path.join(folder, "both"), os.path.join(folder, "one")
        _record(both, "tactics", [(_episode(0), [_trajectory(3, key=(0, 0)), _trajectory(2, key=(0, 1))])],
                header={"behaviour": {"name": "pin:hold", "kind": "deterministic", "explore": 0.0},
                        "opponent": {"name": "script", "kind": "deterministic", "explore": 0.0}})
        _record(one, "tactics", [(_episode(0), [_trajectory(2, key=(0, 0)), _trajectory(1, key=(0, 1))])],
                header={"behaviour": {"name": "network", "parameters": "a.pt", "explore": 0.1}})
        names = Dataset.open([both, one]).behaviours().tolist()
    assert names == ["pin:hold"] * 3 + ["script"] * 2 + ["network[a.pt]@explore0.1"] * 3


def test_inspecting_several_runs_reports_each_run_each_policy_and_each_map():
    """What off-policy learning needs to know of a set of runs is which policy played what, how far it strayed from the judge and how widely it drew, so the report breaks the whole down by run, by the policy that played each decision and by map."""
    with tempfile.TemporaryDirectory() as folder:
        both, one = os.path.join(folder, "both"), os.path.join(folder, "one")
        lake = dict(_episode(0), map="maps/skirmish/[p2]Lake (2p).tmx")
        _record(both, "tactics", [(lake, [_trajectory(3, label=2, key=(0, 0)), _trajectory(2, label=1, key=(0, 1))])],
                header={"behaviour": {"name": "pin:hold", "kind": "deterministic", "explore": 0.0},
                        "opponent": {"name": "script", "kind": "deterministic", "explore": 0.0}})
        beach = dict(_episode(1), map="maps/skirmish/Beach.tmx")
        _record(one, "tactics", [(beach, [_trajectory(4, label=0, key=(1, 0))]), (dict(beach, instance=2),
                                                                                [_trajectory(1, label=0, key=(2, 0))])],
                header={"behaviour": {"name": "script", "kind": "deterministic", "explore": 0.0}})
        report = inspect(Dataset.open([both, one]))
    assert [(r["run"], r["decisions"], r["episodes"], r["opponent"]) for r in report["by_run"]] == [
        ("both", 5, 1, "script"), ("one", 5, 2, None)]
    assert {name: entry["decisions"] for name, entry in report["by_behaviour"].items()} == {"pin:hold": 3, "script": 7}
    # Every decision here is drawn evenly over the seven departures, which is the widest a distribution can be.
    assert abs(report["by_behaviour"]["script"]["entropy"] - math.log(TACTICAL_ACTIONS)) < 1e-3
    assert report["by_behaviour"]["pin:hold"]["agreement"] == 1.0
    assert report["by_map"] == {"[p2]Lake (2p)": {"episodes": 1, "decisions": 5},
                                "Beach": {"episodes": 2, "decisions": 5}}


def _contact():
    """Two squads of three in contact, ours and the enemy's, for fighting both sides of one fight."""
    units = [_unit(1, 100.0, 100.0, squad=0), _unit(2, 120.0, 100.0, squad=0), _unit(3, 140.0, 100.0, squad=0),
             _unit(7, 300.0, 120.0, hostile=1), _unit(8, 310.0, 130.0, hostile=1, type_index=1),
             _unit(9, 290.0, 110.0, hostile=1)]
    return _observation(units, [_region(1, 400.0, 100.0, ours=200.0, theirs=900.0)])


def test_interference_found_at_the_end_reaches_the_record_of_its_own_instance_only():
    """Which squads somebody interfered with is known when an episode ends, after the decisions about them were closed, so the record is sealed after the marking; and a squad number names a different squad on every instance sharing the buffer, so the marking keeps to its own."""
    sealed = []
    rollout = Rollout(sink=lambda ts, e, d, t: sealed.append(ts), retain=False)
    for instance in (0, 1):
        rollout.add((instance, 3), _decision([0.0] * TACTICAL_SIZE, squad=3, done=True))
    rollout.taint([3], owner=0)
    rollout.seal(0, _episode(0))
    rollout.seal(1, _episode(1))
    assert [t.steps[0].tainted for ts in sealed for t in ts] == [True, False]
    # Nothing is retained for an optimiser in a run that has none.
    assert not rollout.done


def test_a_called_fight_is_joined_to_its_trajectory_with_both_readings():
    from rwintel.learn.arena import Engagement

    sealed = []
    _recorded_tactics(sealed)
    episode, trajectories = sealed[0]
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", [(episode, trajectories)])
        dataset = Dataset.open([folder])
        assert int(dataset.arrays["t_engagement"][0]) == 5
        last = int(dataset.arrays["t_start"][0] + dataset.arrays["t_length"][0] - 1)
        row = dataset.signal_rows(last)[-1]
        assert (row[0], row[9], row[10]) == (3.0, 0.1, 0.3)
    assert "formed_at_ms" in Engagement(index=0, site=(0.0, 0.0), formed_at_ms=21000).as_dict()


# ---- how an episode ended ----------------------------------------------------------------------

class _Connection:
    def sendall(self, data: bytes) -> None:
        pass


class _Closing:
    """A policy that notes how the session said its episode ended when it was closed."""

    def __init__(self, session, notes):
        self.session, self.notes = session, notes

    def close(self):
        self.notes.append((self.session.attempt, self.session.ending))


def test_a_game_lost_with_its_episode_and_a_reconnection_are_each_told_apart():
    """A game lost part way is played again under the same episode number, so the record names an episode by the attempt as well, and closes the lost one as lost; a reconnection replaces the policy part way through an episode and closes the old one as rejoined rather than dropping what it held."""
    from rwintel.control.session import EpisodeSettings, Session

    notes = []
    session = Session(_Connection(), None, EpisodeSettings(), [("arm", lambda s: _Closing(s, notes))])
    hello = json.dumps({"instance": 0, "unitTypes": []}).encode("utf-8")
    session.on_hello(hello)
    session._on_started({"map": ""})
    session.on_hello(hello)
    session._on_started({"map": ""})
    session.on_hello(json.dumps({"instance": 0, "unitTypes": [], "running": True}).encode("utf-8"))
    session.close_episode("stopped")
    assert notes == [(1, "lost"), (2, "rejoined"), (2, "stopped")]


# ---- fitting to a record ------------------------------------------------------------------------

def _labelled_episodes(draw, count, label_of, source=None, weight=None, instance_base=0):
    episodes = []
    for index in range(count):
        steps = []
        for _ in range(10):
            action = draw.randrange(TACTICAL_ACTIONS)
            state = [draw.uniform(-0.2, 0.2) for _ in range(TACTICAL_SIZE)]
            state[action] = 1.0
            meta = {}
            if source is not None:
                meta = {"source": source, "weight": weight}
            steps.append(_decision(state, label=label_of(action), meta=meta))
        steps[-1].done = True
        from rwintel.learn.rollout import Trajectory

        episodes.append((_episode(instance_base + index), [Trajectory(key=(instance_base + index, 0), steps=steps,
                                                                       finished=True)]))
    return episodes


def test_a_teacher_whose_choice_follows_from_its_board_is_learnt_almost_exactly():
    """A floor rather than a measurement of the real teacher. The rule ladder is a function of the board and so is this, so a fit that cannot recover a plainly separable one has something wrong with it that no amount of real data would fix."""
    from rwintel.learn.imitation import Teacher, fit

    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", _labelled_episodes(random.Random(11), 80, lambda action: action), shard_decisions=200)
        teacher = Teacher.of([(Dataset.open([folder]), 1.0)])
    assert teacher.held_out.any() and not teacher.held_out.all()
    _, cloning = fit(teacher, seed=3)
    assert cloning.training.accuracy > 0.9 and cloning.validation.accuracy > 0.9
    # And it is still a distribution rather than a lookup table, which is what the label smoothing is there for.
    assert cloning.training.entropy > 0.0


def test_a_fit_on_part_of_the_data_keeps_a_nested_part_and_the_whole_held_out_side():
    """A learning curve over the amount of data compares fits on halves and quarters, so a quarter is a subset of the half and every fraction is judged on the same held-out episodes."""
    from rwintel.learn.imitation import Teacher

    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "tactics", _labelled_episodes(random.Random(13), 200, lambda action: action), shard_decisions=500)
        dataset = Dataset.open([folder])
        teachers = {fraction: Teacher.of([(dataset, 1.0)], fraction=fraction) for fraction in (1.0, 0.5, 0.25)}
        assert "fraction" in _refusal(ValueError, Teacher.of, [(dataset, 1.0)], False, 0.1, 0.0)

    def rows(teacher, held):
        side = teacher.held_out if held else ~teacher.held_out
        return {tuple(row) for row in teacher.states[side].tolist()}

    whole, half, quarter = (rows(teachers[f], False) for f in (1.0, 0.5, 0.25))
    assert quarter < half < whole
    assert 0.35 < len(half) / len(whole) < 0.65 and 0.1 < len(quarter) / len(whole) < 0.4
    assert rows(teachers[1.0], True) == rows(teachers[0.5], True) == rows(teachers[0.25], True)


def test_a_decision_at_weight_nought_teaches_nothing_and_each_source_is_reported():
    """Two teachers that disagree about the same boards: the one at weight nought is not learnt at all, and the fit follows the other exactly as it would alone. A run's weight multiplies the weight each decision was written with."""
    from rwintel.learn.imitation import Teacher, fit

    draw = random.Random(5)
    with tempfile.TemporaryDirectory() as folder:
        script, person = os.path.join(folder, "script"), os.path.join(folder, "person")
        _record(script, "tactics", _labelled_episodes(draw, 40, lambda action: 0), shard_decisions=100)
        _record(person, "tactics", _labelled_episodes(draw, 40, lambda action: 1, source="human", weight=0.5,
                                                      instance_base=100), shard_decisions=100)
        teacher = Teacher.of([(Dataset.open([script]), 1.0), (Dataset.open([person]), 0.0)])
        assert set(teacher.weights[teacher.sources == "human"].tolist()) == {0.0}
        scaled = Teacher.of([(Dataset.open([person]), 0.2)])
        assert np.allclose(scaled.weights, 0.1)
        assert "negative" in _refusal(ValueError, Teacher.of, [(Dataset.open([person]), -1.0)])
    _, cloning = fit(teacher, seed=3, epochs=10)
    assert cloning.training.by_source["script"]["accuracy"] > 0.95
    assert cloning.training.by_source["human"]["accuracy"] < 0.05


def test_an_economic_teacher_is_recorded_read_back_and_fitted():
    """A floor, as for the tactical teacher: offers laid out at random with one marked best, and the teacher always taking it. What the record carries -the state, the slots that held an offer, the distribution- comes back as written, and a fit recovers the choice."""
    from rwintel.control.policy.encoding import INVESTMENT_SLOTS, investment_mask
    from rwintel.learn.imitation import Teacher, fit
    from rwintel.learn.rollout import Trajectory

    draw = random.Random(7)
    episodes = []
    for index in range(60):
        steps = []
        for _ in range(8):
            offers = [Offer(kind=Investment.UNIT, price=draw.uniform(200, 2000), type_index=i,
                            efficiency=draw.uniform(0.1, 0.9), role=Role.ARMOUR, fighting=True)
                      for i in range(draw.randrange(2, 8))]
            best = draw.choice(offers)
            best.efficiency = 1.0
            slots = lay_out([Offer(kind=Investment.STOP)] + offers)
            soft = [0.0] * INVESTMENT_SLOTS
            soft[slots.index(best)] = 1.0
            steps.append(Step(state=economic_state(_economic_board(credits=draw.uniform(0, 5000)), slots),
                              action=slots.index(best), label=slots.index(best), mask=investment_mask(slots), squad=0,
                              soft=soft, periods=0))
        steps[-1].done = True
        episodes.append((_episode(index), [Trajectory(key=(index, ECONOMY_KEY), steps=steps, finished=True)]))
    with tempfile.TemporaryDirectory() as folder:
        _record(folder, "economy", episodes, shard_decisions=100)
        dataset = Dataset.open([folder])
        assert np.array_equal(dataset.arrays["mask"][0], np.asarray(episodes[0][1][0].steps[0].mask, dtype=np.uint8))
        report = inspect(dataset)
        assert report["decisions"] == 480 and report["fingerprint_current"]
        teacher = Teacher.of([(dataset, 1.0)])
    _, cloning = fit(teacher, seed=3)
    assert cloning.training.accuracy > 0.9 and cloning.validation.accuracy > 0.9


def test_datasets_on_the_command_line_are_paths_with_an_optional_weight():
    from rwintel.learn.__main__ import weighted

    assert weighted(["a", "local/datasets/operations/replay-1:0.2", "c:d"]) == [
        ("a", 1.0), ("local/datasets/operations/replay-1", 0.2), ("c:d", 1.0)]


# ---- re-reading finished duels ------------------------------------------------------------------

def test_duels_read_from_their_journals_pair_fights_within_one_run_only():
    """Two runs are separate draws even under the same seed, so the fight with the same numbers in another run is not the same fight; read together, the runs' instances are kept apart and only fights of one run are paired."""
    from rwintel.learn.__main__ import _fights_by_draw, duel_sessions

    def entry(arm, instance, outcomes):
        return {"arm": arm, "instance": instance,
                "statistics": {"history": [{"index": i, "outcome": o} for i, o in enumerate(outcomes)]}}

    with tempfile.TemporaryDirectory() as folder:
        sources = []
        for run in range(2):
            path = os.path.join(folder, f"duel-{run}.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for instance in range(2):
                    for arm in ("duel", "duel-baseline"):
                        handle.write(json.dumps(entry(arm, instance, [0.1 * run, -0.2, 0.3])) + "\n")
            sources.append(path)
        fights = _fights_by_draw(duel_sessions(sources))
    shared = set(fights["duel"]) & set(fights["duel-baseline"])
    assert len(shared) == 2 * 2 * 3
    assert {key[0][0] for key in shared} == {0, 1}


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
