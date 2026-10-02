"""The set networks, the model files, the tactical set state and the offline methods, held to on tiny data.

Nothing here launches a game, and everything runs on the processor; the one test of the bf16 path on a graphics card is skipped where there is none.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

from rwintel.control.policy.encoding import (
    ECONOMIC_CONTEXT_SIZE,
    ECONOMIC_SIZE,
    GLOBAL_SIZE,
    INVESTMENT_SIZE,
    INVESTMENT_SLOTS,
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    REGION_SIZE,
    SQUAD_SIZE,
    SQUAD_SLOTS,
    TACTICAL_ACTIONS,
    TACTICAL_SIZE,
    TRANSPORT_SIZE,
)
from rwintel.learn import models, offline, tokens
from rwintel.learn.dataset import Dataset
from rwintel.learn.net import MASKED, EconomicNet, OperationalNet, TacticalNet
from rwintel.learn.setnet import EconomicSetNet, OperationalSetNet, TacticalSetNet

from test_dataset import _labelled_episodes, _record, _recorded_tactics


@pytest.fixture(autouse=True)
def _local_area(monkeypatch, tmp_path):
    monkeypatch.setenv("RWINTEL_LOCAL", str(tmp_path / "local"))


def _set_rows(count: int, generator: torch.Generator) -> torch.Tensor:
    rows = torch.randn(count, tokens.SET_SIZE, generator=generator)
    flags = tokens.SQUAD_TOKEN + (tokens.MEMBER_CAP + tokens.THREAT_CAP) * tokens.UNIT_SIZE
    rows[:, flags:] = 0.0
    rows[:, flags:flags + 3] = 1.0
    rows[:, flags + tokens.MEMBER_CAP:flags + tokens.MEMBER_CAP + 2] = 1.0
    return rows


# The set networks.

def test_the_tactical_set_net_masks_and_ignores_padded_tokens():
    torch.manual_seed(0)
    net = TacticalSetNet(depth=2).eval()
    rows = _set_rows(6, torch.Generator().manual_seed(1))
    mask = torch.ones(6, TACTICAL_ACTIONS)
    mask[:, 2] = 0
    logits, value = net(rows, mask)
    assert logits.shape == (6, TACTICAL_ACTIONS) and value.shape == (6,)
    assert bool((logits[:, 2] == MASKED).all())
    changed = rows.clone()
    # Members 3.. and threats 2.. are absent; what their slots hold must not matter.
    start = tokens.SQUAD_TOKEN + 3 * tokens.UNIT_SIZE
    changed[:, start:start + 2 * tokens.UNIT_SIZE] = 7.0
    start = tokens.SQUAD_TOKEN + (tokens.MEMBER_CAP + 2) * tokens.UNIT_SIZE
    changed[:, start:start + tokens.UNIT_SIZE] = -5.0
    again, value_again = net(changed, mask)
    assert torch.allclose(logits, again, atol=1e-5) and torch.allclose(value, value_again, atol=1e-5)
    q, v = net.critic(rows)
    assert q.shape == (2, 6, TACTICAL_ACTIONS) and v.shape == (6,)


def test_a_fresh_residual_set_net_answers_exactly_as_its_flat_part():
    from rwintel.learn.net import actor_critic_parameters, value_parameters

    torch.manual_seed(0)
    net = TacticalSetNet(depth=1).eval()
    assert net.residual and net.config["residual"] and net.config["flat_width"] == net.flat.config["width"]
    rows = _set_rows(16, torch.Generator().manual_seed(5))
    mask = torch.ones(16, TACTICAL_ACTIONS)
    mask[:, 4] = 0
    with torch.no_grad():
        logits, value = net(rows, mask)
        flat_logits, flat_value = net.flat(rows[:, :TACTICAL_SIZE], mask)
        q, v = net.critic(rows)
        flat_q, flat_v = net.flat.critic(rows[:, :TACTICAL_SIZE])
    assert torch.equal(logits, flat_logits) and torch.equal(value, flat_value)
    assert torch.equal(q, flat_q) and torch.equal(v, flat_v)
    moving = {id(p) for p in actor_critic_parameters(net)}
    assert not any(id(p) in moving for p in net.flat.q.parameters()) and id(net.flat.body[0].weight) in moving
    spared = {id(p) for p in value_parameters(net)}
    assert {id(p) for p in net.flat.value.parameters()} <= spared and {id(p) for p in net.value.parameters()} <= spared


def test_init_flat_starts_the_residual_part_from_a_flat_model_file_and_refuses_any_other(tmp_path):
    torch.manual_seed(0)
    flat = TacticalNet(width=32).eval()
    path = str(tmp_path / "flat.pt")
    models.save(flat, path)
    settings = offline.Settings(layer="tactics", sources=[], net="set", depth=1, init_flat=path)
    net = offline._fresh(settings).eval()
    assert net.residual and net.config["flat_width"] == 32
    rows = _set_rows(32, torch.Generator().manual_seed(6))
    with torch.no_grad():
        assert torch.equal(net(rows)[0], flat(rows[:, :TACTICAL_SIZE])[0])
    saved = str(tmp_path / "residual.pt")
    models.save(net, saved)
    loaded = models.load(saved)
    assert loaded.residual and loaded.config["flat_width"] == 32
    with torch.no_grad():
        assert torch.equal(loaded(rows)[0], flat(rows[:, :TACTICAL_SIZE])[0])
    others = {"set": TacticalSetNet(depth=1), "operations": OperationalNet()}
    for name, other in others.items():
        models.save(other, str(tmp_path / f"{name}.pt"))
        with pytest.raises(ValueError):
            offline._fresh(offline.Settings(layer="tactics", sources=[], net="set", init_flat=str(tmp_path / f"{name}.pt")))
    with pytest.raises(ValueError):
        offline._fresh(offline.Settings(layer="tactics", sources=[], net="flat", init_flat=path))
    with pytest.raises(ValueError):
        offline.run(offline.Settings(layer="tactics", sources=[("x", 1.0)], net="set", load=saved, init_flat=path))


def test_a_set_model_file_written_before_the_residual_mode_loads_as_a_plain_set_net(tmp_path):
    torch.manual_seed(0)
    net = TacticalSetNet(depth=1, residual=False).eval()
    path = str(tmp_path / "old.pt")
    models.save(net, path)
    record = torch.load(path, map_location="cpu", weights_only=True)
    del record["config"]["residual"]
    torch.save(record, path)
    loaded = models.load(path)
    assert not loaded.residual and not hasattr(loaded, "flat") and loaded.file_config["residual"] is False
    rows = _set_rows(4, torch.Generator().manual_seed(7))
    with torch.no_grad():
        assert torch.equal(loaded(rows)[0], net(rows)[0])


def _operational_rows(count: int, generator: torch.Generator):
    state = torch.randn(count, OPERATIONAL_SIZE, generator=generator)
    at = GLOBAL_SIZE
    for slot in range(OPERATIONAL_REGIONS):
        state[:, at + slot * REGION_SIZE] = 1.0 if slot < 5 else 0.0
    at += OPERATIONAL_REGIONS * REGION_SIZE
    for slot in range(SQUAD_SLOTS):
        state[:, at + slot * SQUAD_SIZE] = 1.0 if slot < 2 else 0.0
    at += SQUAD_SLOTS * SQUAD_SIZE
    for slot in range(4):
        state[:, at + slot * TRANSPORT_SIZE] = 1.0 if slot < 1 else 0.0
    squad = torch.zeros(count, SQUAD_SLOTS)
    squad[:, 1] = 1.0
    return state, squad


def test_the_operational_set_net_keeps_the_flat_interface_and_ignores_absent_slots():
    torch.manual_seed(0)
    net = OperationalSetNet(depth=2).eval()
    state, squad = _operational_rows(4, torch.Generator().manual_seed(2))
    region_mask = torch.zeros(4, OPERATIONAL_REGIONS)
    region_mask[:, :5] = 1.0
    plan_mask = torch.ones(4, OPERATIONAL_PLANS)
    plan_mask[:, 7] = 0.0
    region = torch.tensor([0, 1, 2, 3])
    regions, plans, value = net(state, squad, region_mask, plan_mask, region)
    assert regions.shape == (4, OPERATIONAL_REGIONS) and plans.shape == (4, OPERATIONAL_PLANS) and value.shape == (4,)
    assert bool((regions[:, 5:] == MASKED).all()) and bool((plans[:, 7] == MASKED).all())
    hidden = net.hidden(state, squad)
    assert net.plans_all(hidden).shape == (4, OPERATIONAL_REGIONS, OPERATIONAL_PLANS)
    assert net.critic_regions(hidden).shape == (2, 4, OPERATIONAL_REGIONS)
    assert net.critic_plans(hidden).shape == (2, 4, OPERATIONAL_REGIONS, OPERATIONAL_PLANS)
    assert net.critic_plans(hidden, region).shape == (2, 4, OPERATIONAL_PLANS)
    # The deciders repeat the packed hidden tensor and read plans per region from it.
    every = torch.arange(OPERATIONAL_REGIONS).repeat(4)
    repeated = net.plans(hidden.repeat_interleave(OPERATIONAL_REGIONS, dim=0), every)
    assert torch.allclose(repeated.reshape(4, OPERATIONAL_REGIONS, -1), net.plans_all(hidden), atol=1e-5)
    changed = state.clone()
    squads_at = GLOBAL_SIZE + OPERATIONAL_REGIONS * REGION_SIZE
    changed[:, squads_at + 5 * SQUAD_SIZE + 1:squads_at + 6 * SQUAD_SIZE] = 9.0
    transports_at = squads_at + SQUAD_SLOTS * SQUAD_SIZE
    changed[:, transports_at + 2 * TRANSPORT_SIZE + 1:transports_at + 3 * TRANSPORT_SIZE] = -9.0
    again = net(changed, squad, region_mask, plan_mask, region)
    assert torch.allclose(regions, again[0], atol=1e-4) and torch.allclose(plans, again[1], atol=1e-4)
    assert torch.allclose(value, again[2], atol=1e-4)


def test_the_economic_set_net_scores_offers_and_ignores_empty_slots():
    torch.manual_seed(0)
    net = EconomicSetNet().eval()
    state = torch.randn(3, ECONOMIC_SIZE)
    for slot in range(INVESTMENT_SLOTS):
        state[:, ECONOMIC_CONTEXT_SIZE + slot * INVESTMENT_SIZE] = 1.0 if slot < 4 else 0.0
    mask = torch.zeros(3, INVESTMENT_SLOTS)
    mask[:, :4] = 1.0
    logits, value = net(state, mask)
    assert logits.shape == (3, INVESTMENT_SLOTS) and bool((logits[:, 4:] == MASKED).all())
    changed = state.clone()
    begin = ECONOMIC_CONTEXT_SIZE + 6 * INVESTMENT_SIZE
    changed[:, begin + 1:begin + INVESTMENT_SIZE] = 3.0
    again, value_again = net(changed, mask)
    assert torch.allclose(logits[:, :4], again[:, :4], atol=1e-5) and torch.allclose(value, value_again, atol=1e-5)
    assert net.critic(state)[0].shape == (2, 3, INVESTMENT_SLOTS)


# Model files.

@pytest.mark.parametrize("net", [TacticalNet(width=32), OperationalNet(), EconomicNet(), TacticalSetNet(depth=1),
                                 OperationalSetNet(depth=1), EconomicSetNet(depth=1)])
def test_a_model_file_rebuilds_the_network_it_was_written_from(net, tmp_path):
    path = str(tmp_path / "model.pt")
    models.save(net, path, version=4, extra={"reward_scale": 0.5})
    loaded = models.load(path)
    assert type(loaded) is type(net) and loaded.version == 4 and loaded.file_config["reward_scale"] == 0.5
    for name, value in net.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    with pytest.raises(ValueError):
        models.load(path, layer="economy" if net.LAYER != "economy" else "tactics")


def test_an_old_bare_state_dict_loads_as_the_flat_network_of_its_layer(tmp_path):
    for net in (TacticalNet(width=48), OperationalNet(), EconomicNet(width=96, offer_width=32)):
        bare = {name: value for name, value in net.state_dict().items() if not name.startswith(("q.", "q_region.", "q_plan."))}
        path = str(tmp_path / f"{net.LAYER}.pt")
        torch.save(bare, path)
        loaded = models.load(path)
        assert type(loaded) is type(net) and loaded.config == net.config
        for name, value in bare.items():
            assert torch.equal(value, loaded.state_dict()[name])


def test_a_set_model_file_loads_through_the_runners_paths(tmp_path):
    """Duel's --load, collect's --student and the evaluation's arms go through the model files, so a set network file loads wherever a flat one did."""
    from rwintel.learn.__main__ import _network, _pupil, build_parser
    from rwintel.eval import arms

    path = str(tmp_path / "set.pt")
    models.save(TacticalSetNet(depth=1), path)
    assert isinstance(_network("tactics", path, torch.device("cpu"), TacticalNet), TacticalSetNet)
    arguments = build_parser().parse_args(["collect", "--layer", "tactics", "--student", path])
    decider = _pupil(arguments)
    assert isinstance(decider.net, TacticalSetNet) and decider.set_input
    decider.batcher.stop()
    flat = str(tmp_path / "flat.pt")
    models.save(TacticalNet(width=16), flat)
    decider = _pupil(build_parser().parse_args(["collect", "--layer", "tactics", "--student", flat]))
    assert not decider.set_input
    decider.batcher.stop()
    ops = str(tmp_path / "ops.pt")
    models.save(OperationalSetNet(depth=1), ops)
    assert callable(arms.learnt("operations", ops, greedy=True))
    decider = arms._loaded[("operations", os.path.abspath(ops), True)](None)
    assert isinstance(decider.net, OperationalSetNet)
    decider.batcher.stop()


# The tactical set state.

def test_the_set_state_of_one_decision_is_the_row_built_for_the_whole_dataset(tmp_path):
    sealed = []
    _recorded_tactics(sealed)
    folder = str(tmp_path / "run")
    _record(folder, "tactics", sealed)
    dataset = Dataset.open([folder], with_materials=True)
    built = tokens.set_states(dataset)
    assert built.shape == (len(dataset), tokens.SET_SIZE)
    for decision in range(len(dataset)):
        one = np.asarray(tokens.set_state(dataset.material_rows(decision), dataset.context(decision)), dtype=np.float32)
        assert np.allclose(one, built[decision], atol=1e-6)
        assert one[tokens.SET_SIZE - tokens.MEMBER_CAP - tokens.THREAT_CAP] == 1.0
    cached = tokens.cached_set_states(folder)
    assert cached.dtype == np.float16 and np.allclose(cached, built, atol=1e-2, rtol=1e-3)
    assert os.path.exists(tokens.cache_path(folder))


def _played_by(decider, sealed):
    """The synthetic squad of `_recorded_tactics`, decided by `decider`."""
    from rwintel.learn.layers import LearntTactics
    from rwintel.learn.rollout import Rollout
    from test_learning import _CATALOGUE, _skirmish, _squad

    rollout = Rollout(discount=1.0, trace=1.0, sink=lambda ts, e, d, t: sealed.append((e, ts)))
    layer = LearntTactics(None, _CATALOGUE, decider, rollout=rollout, instance=0)
    squad = _squad()
    for index in range(4):
        squad.losses = 80.0 * index
        layer.decide(_skirmish(), [squad], 21000 + 200 * index)
    layer.finish(squad, 0.1, 0.3, "called", engagement=5)
    layer.close()


def test_a_tactical_set_net_plays_online_on_the_row_the_dataset_rebuilds(tmp_path):
    from rwintel.learn.deciders import Labelled, NetworkChoice

    torch.manual_seed(0)
    decider = NetworkChoice(TacticalSetNet(depth=1).eval(), torch.device("cpu"))
    assert decider.set_input and Labelled(decider, explore=0.1).set_input
    sealed = []
    _played_by(decider, sealed)
    steps = [step for _, trajectories in sealed for trajectory in trajectories for step in trajectory.steps]
    assert steps and all(step.net_state is not None and len(step.net_state) == tokens.SET_SIZE for step in steps)
    assert all(len(step.state) == TACTICAL_SIZE for step in steps)
    folder = str(tmp_path / "run")
    _record(folder, "tactics", sealed)
    built = tokens.set_states(Dataset.open([folder], with_materials=True))
    assert np.allclose(np.asarray([step.net_state for step in steps], dtype=np.float32), built, atol=1e-5)


def test_the_optimiser_updates_a_set_net_on_the_rows_it_read_and_refuses_a_mixed_batch():
    from rwintel.learn.rollout import Step
    from rwintel.learn.train import Optimiser

    torch.manual_seed(0)
    net = TacticalSetNet(depth=1)
    rows = _set_rows(8, torch.Generator().manual_seed(3))
    steps = [Step(state=[0.0] * TACTICAL_SIZE, action=index % TACTICAL_ACTIONS, mask=[1.0] * TACTICAL_ACTIONS,
                  log_prob=-1.9, advantage=float(index % 3) - 1.0, ret=0.5, net_state=rows[index].tolist())
             for index in range(8)]
    before = [parameter.detach().clone() for parameter in net.parameters()]
    optimiser = Optimiser(net)
    report = optimiser.update(steps)
    assert report.updates == 1 and report.steps == 8
    assert any(not torch.equal(old, new) for old, new in zip(before, net.parameters()))
    steps[0].net_state = None
    with pytest.raises(ValueError):
        optimiser.update(steps)


# Inference limits, serving and the bench.

def test_the_processor_keeps_its_batching_limits():
    from rwintel.learn.inference import limits

    assert limits("cpu") == (0.004, 64) and limits(torch.device("cpu")) == (0.004, 64) and limits(None) == (0.004, 64)
    window, max_batch = limits("cuda:0")
    assert window > 0 and max_batch >= 1


def test_the_bench_rule_caps_where_a_call_doubles_and_widens_the_window_only_for_a_slow_single_call():
    from rwintel.learn.bench import suggested
    from rwintel.learn.inference import LIMITS

    def rows(medians):
        return [{"size": size, "median_ms": median} for size, median in medians.items()]

    window_ms = LIMITS["cpu"][0] * 1000.0
    fast, slow = window_ms / 2, window_ms * 1.5
    assert suggested(rows({1: fast, 8: fast * 1.1, 64: fast * 2, 128: fast * 3, 256: fast * 7.5})) == {"window": LIMITS["cpu"][0], "max_batch": 64}
    assert suggested(rows({1: slow, 16: slow * 1.8, 32: slow * 2.2})) == {"window": round(slow / 1000.0, 6), "max_batch": 16}
    assert suggested(rows({8: 1.0})) == {}


def test_a_set_net_served_through_a_batcher_answers_as_a_direct_greedy_forward():
    from rwintel.learn.deciders import NetworkChoice, choice_batcher

    torch.manual_seed(0)
    net = TacticalSetNet(depth=1).eval()
    rows = _set_rows(5, torch.Generator().manual_seed(4))
    mask = torch.ones(5, TACTICAL_ACTIONS)
    with torch.no_grad():
        direct = net(rows, mask)[0].argmax(dim=-1).tolist()
    batcher = choice_batcher(net, device=torch.device("cpu"), greedy=True, lock=__import__("threading").Lock())
    decider = NetworkChoice(net, torch.device("cpu"), batcher, greedy=True)
    try:
        served = [decider.choose(rows[index].tolist(), [1.0] * TACTICAL_ACTIONS).action for index in range(5)]
    finally:
        batcher.stop()
    assert served == direct


@pytest.mark.parametrize("layer,kind", [("tactics", "flat"), ("tactics", "set"), ("operations", "flat"),
                                        ("operations", "set"), ("economy", "flat"), ("economy", "set")])
def test_the_bench_times_every_size_for_each_layer_and_kind(layer, kind, tmp_path):
    from rwintel.learn import bench
    from rwintel.learn.__main__ import main

    result = bench.run(layer, kind, device="cpu", sizes=(1, 4), repeats=3)
    assert [row["size"] for row in result["rows"]] == [1, 4] and result["kind"] == kind
    assert all(row["median_ms"] > 0 and row["p95_ms"] >= row["median_ms"] for row in result["rows"])
    if layer == "tactics" and kind == "set":
        report = str(tmp_path / "bench.json")
        assert main(["bench", "--layer", layer, "--net", kind, "--device", "cpu", "--sizes", "1,4", "--repeats", "3",
                     "--report", report]) == 0
        with open(report, encoding="utf-8") as handle:
            assert len(json.load(handle)["rows"]) == 2


# Actors following a learner.

def test_the_reloader_swaps_in_a_newer_published_version_and_refuses_another_kind(tmp_path):
    import threading

    from rwintel.learn.deciders import choice_batcher
    from rwintel.learn.reload import Reloader, sidecar_path

    path = str(tmp_path / "actor.pt")
    torch.manual_seed(0)
    served = TacticalNet(width=16).eval()
    models.save(served, path, version=3)
    served = models.load(path)
    lock = threading.Lock()
    reloader = Reloader(served, path, lock, seconds=60.0)
    batcher = choice_batcher(served, device=torch.device("cpu"), version=lambda: reloader.version, lock=lock)
    try:
        request = ([0.1] * TACTICAL_SIZE, [1.0] * TACTICAL_ACTIONS)
        assert batcher.submit(request).version == 3
        assert not reloader.poll()
        newer = TacticalNet(width=16)
        models.save(newer, path, version=7)
        with open(sidecar_path(path), "w", encoding="utf-8") as handle:
            json.dump({"version": 9}, handle)
        assert not reloader.poll() and reloader.version == 3
        with open(sidecar_path(path), "w", encoding="utf-8") as handle:
            json.dump({"version": 7}, handle)
        assert reloader.poll() and reloader.version == 7
        for name, value in newer.state_dict().items():
            assert torch.equal(value, served.state_dict()[name])
        assert batcher.submit(request).version == 7
        models.save(TacticalSetNet(depth=1), path, version=8)
        with open(sidecar_path(path), "w", encoding="utf-8") as handle:
            json.dump({"version": 8}, handle)
        assert not reloader.poll() and reloader.refused and reloader.version == 7
    finally:
        batcher.stop()


def test_the_reloader_serves_a_set_model_file_written_before_the_residual_mode(tmp_path):
    import threading

    from rwintel.learn.reload import Reloader, sidecar_path

    def publish(net, version):
        models.save(net, path, version=version)
        record = torch.load(path, map_location="cpu", weights_only=True)
        del record["config"]["residual"]
        torch.save(record, path)
        with open(sidecar_path(path), "w", encoding="utf-8") as handle:
            json.dump({"version": version}, handle)

    path = str(tmp_path / "actor.pt")
    torch.manual_seed(0)
    publish(TacticalSetNet(depth=1, residual=False), 1)
    served = models.load(path)
    reloader = Reloader(served, path, threading.Lock(), seconds=60.0)
    newer = TacticalSetNet(depth=1, residual=False)
    publish(newer, 2)
    assert reloader.poll() and reloader.version == 2 and not served.residual
    for name, value in newer.state_dict().items():
        assert torch.equal(value, served.state_dict()[name])
    models.save(TacticalSetNet(depth=1), path, version=3)
    with open(sidecar_path(path), "w", encoding="utf-8") as handle:
        json.dump({"version": 3}, handle)
    assert not reloader.poll() and reloader.refused and reloader.version == 2


def test_the_token_cache_is_kept_per_shard_and_extended_when_a_run_grows(tmp_path):
    import shutil

    from rwintel.learn.dataset import shard_files

    sealed = []
    _recorded_tactics(sealed)
    folder = str(tmp_path / "run")
    _record(folder, "tactics", sealed * 2, shard_decisions=3)
    shards = shard_files(folder)
    assert len(shards) == 2
    tokens.cached_set_states(folder)
    cache = tokens.cache_path(folder)
    written = {name: os.stat(os.path.join(cache, name)).st_mtime_ns for name in os.listdir(cache)}
    assert len(written) == 2
    shutil.copy(shards[0], os.path.join(folder, "shard-00002.npz"))
    grown = tokens.cached_set_states(folder)
    after = os.listdir(cache)
    assert len(after) == 3 and all(os.stat(os.path.join(cache, name)).st_mtime_ns == stamp for name, stamp in written.items())
    built = tokens.set_states(Dataset.open([folder], with_materials=True))
    assert grown.shape == built.shape and np.allclose(np.asarray(grown), built, atol=1e-2, rtol=1e-3)
    picked = np.array([len(built) - 1, 0, 5])
    assert np.allclose(grown[picked], built[picked], atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("method", ["iql", "awr"])
def test_a_following_learner_publishes_rising_versions_and_takes_in_a_shard_added_while_it_runs(tmp_path, method):
    import shutil
    import threading
    import time

    from rwintel.learn.dataset import shard_files
    from rwintel.learn.reload import read_sidecar

    watch = tmp_path / "watch"
    growing = str(watch / "growing")
    _record(growing, "tactics", _labelled_episodes(random.Random(5), 20, lambda action: action), shard_decisions=200)
    spare = str(tmp_path / "spare")
    _record(spare, "tactics", _labelled_episodes(random.Random(6), 20, lambda action: action), shard_decisions=200)
    save = str(tmp_path / "actor.pt")
    report_path = str(tmp_path / "report.json")
    settings = offline.Settings(layer="tactics", sources=[], method=method, device="cpu", batch=32, save=save,
                                watch=[str(watch)], publish_every=5, rescan_seconds=0.0, report=report_path)
    stop = threading.Event()
    result = {}
    thread = threading.Thread(target=lambda: result.update(offline.follow(settings, stop)), daemon=True)
    thread.start()
    deadline = time.monotonic() + 120.0
    while (read_sidecar(save) or {}).get("version", 0) <= 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    first = read_sidecar(save)
    assert first is not None and first["version"] > 0
    shutil.copy(shard_files(spare)[0], os.path.join(growing, f"shard-{len(shard_files(growing)):05d}.npz"))
    while (read_sidecar(save) or {}).get("shards", 0) <= first["shards"] and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=60.0)
    last = read_sidecar(save)
    assert last["shards"] > first["shards"] and last["decisions"] > first["decisions"]
    assert result["published"] == sorted(set(result["published"])) and len(result["published"]) >= 2
    assert result["published"][0] == 0 and result["appends"] and result["appends"][0]["shards"] == 1
    loaded = models.load(save)
    assert loaded.version == last["version"] == result["published"][-1] and "reward_scale" in loaded.file_config
    with open(report_path, encoding="utf-8") as handle:
        written = json.load(handle)
    assert result["report"] == report_path and written["updates"] == result["updates"] > 0
    assert written["seconds_per_update"] > 0 and written["learning_seconds"] > 0 and written["published"] == result["published"]


def test_the_online_command_gives_the_learner_its_options_and_the_actors_the_rest():
    from rwintel.learn import online
    from rwintel.learn.__main__ import build_parser

    parser = build_parser()
    argv = ["online", "--layer", "tactics", "--dataset", "base", "--dataset", "more:0.5", "--load", "start.pt",
            "--save", "actor.pt", "--count", "2", "--offset", "30", "--port", "9700", "--seed", "7", "--episodes", "3",
            "--publish-every=100", "--buffer", "5000", "--both-sides", "--actors", "watched", "--reload-seconds", "10",
            "--judge", "0.3"]
    arguments = parser.parse_args(argv)
    arguments.stamp = "stamp"
    learner, actors = online.split(parser, argv)
    planned = online.commands(arguments, learner, actors)
    assert planned["learner"][1:] == ["-m", "rwintel.learn", "offline", "--follow", "--layer", "tactics", "--method", "iql",
                                      "--save", "actor.pt", "--watch", "watched", "--dataset", "base", "--dataset",
                                      "more:0.5", "--load", "start.pt", "--seed", "7", "--publish-every=100", "--buffer",
                                      "5000", "--judge", "0.3"]
    assert planned["actors"][1:] == ["-m", "rwintel.runtime", "run", "--count", "2", "--offset", "30", "--", "learn",
                                     "collect", "--layer", "tactics", "--student", "actor.pt", "--reload-seconds", "10",
                                     "--dataset", os.path.join("watched", "online-stamp"), "--card-share", "0.1", "--port",
                                     "9700", "--seed", "7", "--episodes", "3", "--both-sides"]
    with pytest.raises(SystemExit):
        online.commands(parser.parse_args(["online", "--layer", "tactics", "--dataset", "base"]), [], [])
    # Without a base dataset or a shard under the actors' directory the learner would never publish, so the command refuses to start.
    with pytest.raises(SystemExit, match="--dataset"):
        online.commands(parser.parse_args(["online", "--save", "actor.pt", "--actors", "no-such-directory"]), [], [])


def test_a_following_learner_takes_iql_unless_told_and_refuses_methods_it_cannot_follow_with():
    from rwintel.learn import online
    from rwintel.learn.__main__ import _resolve_method, build_parser

    parser = build_parser()
    base = ["online", "--dataset", "base", "--save", "actor.pt"]

    def learner_method(*extra):
        arguments = parser.parse_args(base + list(extra))
        arguments.stamp = "stamp"
        learner = online.commands(arguments, [], [])["learner"]
        return learner[learner.index("--method") + 1]

    assert learner_method() == "iql" and learner_method("--method", "cql") == "cql"
    assert learner_method("--method", "awr") == "awr"
    for method in ("bc", "distill", "topbc", "fqe"):
        with pytest.raises(SystemExit, match=f"online learns with --method iql, awr or cql, not {method}"):
            learner_method("--method", method)

    def resolved(*argv):
        arguments = parser.parse_args(list(argv))
        _resolve_method(arguments)
        return arguments.method

    assert resolved("offline", "--dataset", "base") == "bc"
    assert resolved("offline", "--dataset", "base", "--method", "iql") == "iql"
    assert resolved("offline", "--follow", "--watch", "actors") == "iql"
    assert resolved("offline", "--follow", "--watch", "actors", "--method", "cql") == "cql"
    assert resolved("offline", "--follow", "--watch", "actors", "--method", "awr") == "awr"
    assert resolved("online", "--save", "actor.pt") == "iql"
    with pytest.raises(SystemExit, match="offline --follow learns with --method iql, awr or cql, not bc"):
        resolved("offline", "--follow", "--watch", "actors", "--method", "bc")
    with pytest.raises(SystemExit, match="online learns with --method iql, awr or cql, not bc"):
        resolved("online", "--save", "actor.pt", "--method", "bc")


_FAKE_LEARNER = """
import json, os, signal, sys, time
path = sys.argv[1] + ".json"
def publish(version):
    with open(path + ".tmp", "w") as handle:
        json.dump({"version": version, "updates": version, "decisions": 1}, handle)
    os.replace(path + ".tmp", path)
stopped = []
signal.signal(signal.SIGTERM, lambda *_: stopped.append(1))
time.sleep(0.3)
publish(1)
while not stopped:
    time.sleep(0.02)
publish(2)
"""

_FAKE_ACTORS = """
import json, sys
with open(sys.argv[1] + ".json") as handle:
    played = json.load(handle)["version"]
with open(sys.argv[2], "w") as handle:
    handle.write(str(played))
"""


def test_the_online_loop_starts_the_actors_on_the_first_version_and_stops_the_learner_after_them(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from rwintel.learn import online
    from rwintel.learn.__main__ import build_parser
    from rwintel.learn.reload import read_sidecar

    save, played = str(tmp_path / "actor.pt"), str(tmp_path / "played")
    monkeypatch.setattr(online, "POLL_SECONDS", 0.05)
    monkeypatch.setattr(online, "commands", lambda arguments, learner, actors: {
        "learner": [sys.executable, "-c", _FAKE_LEARNER, save], "actors": [sys.executable, "-c", _FAKE_ACTORS, save, played],
        "actors_root": str(tmp_path), "record": str(tmp_path / "run")})
    arguments = SimpleNamespace(save=save, count=1)
    assert online.run(arguments, build_parser(), ["online", "--save", save]) == 0
    with open(played, encoding="utf-8") as handle:
        assert handle.read() == "1"
    assert read_sidecar(save)["version"] == 2
    logs = os.path.join(str(tmp_path / "local"), "logs", "online")
    assert any(os.path.exists(os.path.join(logs, name, "learner.log")) for name in os.listdir(logs))


_FAKE_LAUNCHER = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
stubborn = "import signal, time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\nwhile True: time.sleep(0.05)"
calm = "import time\\nwhile True: time.sleep(0.05)"
games = [subprocess.Popen([sys.executable, "-c", code], start_new_session=True) for code in (stubborn, calm, calm)]
started = list(games)
time.sleep(0.3)
with open(sys.argv[1] + ".tmp", "w") as handle:
    handle.write(" ".join(str(game.pid) for game in games))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
while True:
    for index, game in enumerate(games):
        if game.poll() is not None:
            games[index] = subprocess.Popen([sys.executable, "-c", calm], start_new_session=True)
            started.append(games[index])
            with open(sys.argv[1], "w") as handle:
                handle.write(" ".join(str(game.pid) for game in started))
    time.sleep(0.02)
"""


def test_a_launcher_that_will_not_stop_is_killed_with_the_games_it_started_in_their_own_sessions(tmp_path, monkeypatch):
    """No game: the launcher is a script that ignores SIGTERM, starts three processes in sessions of their own, one of which ignores SIGTERM too, and starts a fresh one whenever one ends."""
    import sys
    import time

    from rwintel.learn import online

    monkeypatch.setattr(online, "ORPHAN_EXIT_SECONDS", 0.5)
    started = str(tmp_path / "started")
    launcher = online._start([sys.executable, "-c", _FAKE_LAUNCHER, started], str(tmp_path / "actors.log"))
    deadline = time.monotonic() + 30.0
    while not os.path.exists(started) and time.monotonic() < deadline:
        time.sleep(0.05)
    with open(started, encoding="utf-8") as handle:
        games = [int(pid) for pid in handle.read().split()]
    assert len(games) == 3 and all(online._running(pid) for pid in games)
    assert sorted(pid for pid, _ in online._children(launcher.pid)) == sorted(games)
    assert online._stop(launcher, 0.3, "actors", launcher=True) is not None
    while any(online._running(pid) for pid in games) and time.monotonic() < deadline:
        time.sleep(0.05)
    with open(started, encoding="utf-8") as handle:
        assert [int(pid) for pid in handle.read().split()] == games
    assert not any(online._running(pid) for pid in games)


def test_a_launcher_with_no_children_or_none_left_running_is_stopped_without_a_wait(tmp_path, monkeypatch):
    """A launcher that ignores SIGTERM and started nothing, one whose only child shares its own group, and one that has already exited."""
    import sys
    import time

    from rwintel.learn import online

    monkeypatch.setattr(online, "ORPHAN_EXIT_SECONDS", 30.0)
    stubborn = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    alone = online._start([sys.executable, "-c", stubborn + "print('up', flush=True)\nwhile True: time.sleep(0.05)"],
                          str(tmp_path / "alone.log"))
    shared = online._start([sys.executable, "-c", stubborn + "import subprocess, sys\n"
                            "child = subprocess.Popen([sys.executable, '-c', 'import signal, time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\nwhile True: time.sleep(0.05)'])\n"
                            "print(child.pid, flush=True)\nwhile True: time.sleep(0.05)"], str(tmp_path / "shared.log"))
    deadline = time.monotonic() + 30.0
    while not all(os.path.getsize(tmp_path / name) for name in ("alone.log", "shared.log")) and time.monotonic() < deadline:
        time.sleep(0.05)
    child = int((tmp_path / "shared.log").read_text().split()[0])
    assert online._children(alone.pid) == [] and online._children(shared.pid) == [(child, shared.pid)]
    began = time.monotonic()
    assert online._stop(alone, 0.3, "actors", launcher=True) == -9
    assert online._stop(shared, 0.3, "actors", launcher=True) == -9
    assert time.monotonic() - began < 10.0
    while online._running(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not online._running(child)
    done = online._start([sys.executable, "-c", "pass"], str(tmp_path / "done.log"))
    done.wait()
    assert online._stop(done, 0.3, "actors", launcher=True) == 0


_STUB_ACTORS = """
import json, os, shutil, sys, time
record, source, save = sys.argv[1:4]
def sidecar():
    try:
        with open(save + ".json") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}
seen = [sidecar().get("version")]
os.makedirs(record)
shutil.copy(os.path.join(source, "run.json"), os.path.join(record, "run.json"))
shutil.copytree(os.path.join(source, "tables"), os.path.join(record, "tables"))
shutil.copy(os.path.join(source, "shard-00000.npz"), os.path.join(record, "partial.tmp"))
os.replace(os.path.join(record, "partial.tmp"), os.path.join(record, "shard-00000.npz"))
deadline = time.monotonic() + 120.0
while sidecar().get("shards", 0) < 2 and time.monotonic() < deadline:
    time.sleep(0.05)
seen.append(sidecar().get("version"))
print(json.dumps(seen))
sys.exit(0 if sidecar().get("shards", 0) >= 2 else 1)
"""


def test_the_online_loop_runs_the_real_learner_beside_stub_actors_and_takes_its_last_version_on_the_way_out(tmp_path, monkeypatch):
    """No game: the actor side is a script that writes one shard into the watched directory as a recording run would and exits once the learner has appended it."""
    import sys

    from rwintel.learn import online
    from rwintel.learn.__main__ import build_parser
    from rwintel.learn.reload import read_sidecar

    base = str(tmp_path / "base")
    _record(base, "tactics", _labelled_episodes(random.Random(5), 20, lambda action: action), shard_decisions=10 ** 6)
    source = str(tmp_path / "source")
    _record(source, "tactics", _labelled_episodes(random.Random(6), 20, lambda action: action), shard_decisions=10 ** 6)
    save, actors_root = str(tmp_path / "actor.pt"), str(tmp_path / "actors")
    planned = online.commands
    monkeypatch.setattr(online, "POLL_SECONDS", 0.05)

    def stubbed(arguments, learner, actors):
        found = planned(arguments, learner, actors)
        found["actors"] = [sys.executable, "-c", _STUB_ACTORS, found["record"], source, save]
        return found

    monkeypatch.setattr(online, "commands", stubbed)
    argv = ["online", "--layer", "tactics", "--dataset", base, "--save", save, "--actors", actors_root, "--device", "cpu",
            "--batch", "32", "--publish-every", "5", "--rescan-seconds", "0", "--seed", "3"]
    arguments = build_parser().parse_args(argv)
    arguments.stamp = "stamp"
    assert online.run(arguments, build_parser(), argv) == 0
    logs = os.path.join(str(tmp_path / "local"), "logs", "online")
    run_directory = os.path.join(logs, os.listdir(logs)[0])
    with open(os.path.join(run_directory, "learner.log"), encoding="utf-8") as handle:
        learnt = handle.read()
    with open(os.path.join(run_directory, "actors.log"), encoding="utf-8") as handle:
        seen = json.loads(handle.read().strip().splitlines()[-1])
    assert "appended" in learnt and "published version 0 " in learnt
    last = read_sidecar(save)
    # The actors start once a version is published; the learner may have published later ones before they read it.
    assert seen[0] is not None and 0 <= seen[0] <= seen[1]
    assert last["shards"] == 2 and last["version"] >= seen[1] > 0
    assert models.load(save).version == last["version"]


def _cut_every_other(episodes):
    """The episodes with the trajectory of every other one cut off rather than ended."""
    for number, (_, trajectories) in enumerate(episodes):
        if number % 2:
            for trajectory in trajectories:
                trajectory.finished = False
                trajectory.steps[-1].done = False
    return episodes


def _three_shards(tmp_path):
    """A named run of one shard and a watched run of two, recorded from different episodes, every other trajectory cut off."""
    base = str(tmp_path / "base")
    _record(base, "tactics", _cut_every_other(_labelled_episodes(random.Random(5), 20, lambda action: action)),
            shard_decisions=10 ** 6)
    actors = str(tmp_path / "actors")
    _record(actors, "tactics", _cut_every_other(_labelled_episodes(random.Random(6), 30, lambda action: action)),
            shard_decisions=80)
    return base, actors


@pytest.mark.parametrize("with_sets", [False, True])
def test_appending_shards_to_the_buffer_gives_the_rows_of_reading_them_at_once(tmp_path, with_sets):
    from rwintel.learn.dataset import shard_files

    if with_sets:
        sealed = []
        _recorded_tactics(sealed)
        base, actors = str(tmp_path / "base"), str(tmp_path / "actors")
        _record(base, "tactics", sealed)
        _record(actors, "tactics", sealed * 3, shard_decisions=3)
    else:
        base, actors = _three_shards(tmp_path)
    assert len(shard_files(actors)) >= 2
    sources = [(base, 1.0), (actors, 0.5)]
    settings = offline.Settings(layer="tactics", sources=sources, method="iql")
    device = torch.device("cpu")
    whole = offline.load_data(settings, device, with_sets=with_sets, with_rewards=True)
    shards = offline.listing(sources)
    # The scale is fixed by the first append, so the one reading everything at once is given to compare the rows.
    buffer = offline.Buffer(settings, device, with_sets=with_sets, with_rewards=True)
    buffer.scale = whole.reward_scale
    buffer.append(shards[:2])
    buffer.append(shards[2:3])
    buffer.append(shards[3:])
    grown = buffer.data()
    assert len(grown) == len(whole) and grown.reward_scale == whole.reward_scale
    names = [name for name in offline.Data.__dataclass_fields__ if name not in ("layer", "reward_scale")]
    for name in names:
        mine, theirs = getattr(grown, name), getattr(whole, name)
        if mine is None or theirs is None:
            assert mine is None and theirs is None, name
        elif isinstance(mine, np.ndarray):
            assert np.array_equal(mine, theirs) if mine.dtype == object else np.allclose(mine, theirs), name
        else:
            assert mine.dtype == theirs.dtype and torch.equal(torch.nan_to_num(mine), torch.nan_to_num(theirs)), name
            assert torch.equal(mine.isnan(), theirs.isnan()) if mine.is_floating_point() else True, name
    following = grown.next[grown.next >= 0]
    assert bool((grown.trajectory[following.numpy()] == grown.trajectory[(grown.next >= 0).nonzero().squeeze(-1).numpy()]).all())
    if not with_sets:
        _assert_tails_close_their_own_cut_trajectories(grown)


def test_an_offline_run_prices_recorded_rewards_at_the_terms_it_is_given(tmp_path):
    """Unless told otherwise each run is priced at the terms it was paid under; given terms, every reward is priced again at them and the returns are discounted at their discount, so the shaping still telescopes."""
    from rwintel.learn.reward import SHARE, EconomicTerms
    from test_dataset import _TimedSession, _recorded_economy

    sealed = []
    paid = EconomicTerms(discount=0.9, shaping=SHARE, value_flow=0.05, ground_flow=0.02)
    for score in (0.5, -0.25):
        _recorded_economy(sealed, terms=paid, score=score, times=(30000, 32000, 34000), session=_TimedSession())
    folder = str(tmp_path / "economy")
    _record(folder, "economy", [(dict(episode, episode=index + 1), trajectories)
                                for index, (episode, trajectories) in enumerate(sealed)],
            discount=paid.discount, trace=0.95, header={"terms": paid.as_dict()})
    dataset = Dataset.open([folder])
    other = EconomicTerms()
    for given, expected, discount in ((None, dataset.rewards(), 0.9), (other.as_dict(), dataset.rewards(other), 1.0)):
        settings = offline.Settings(layer="economy", sources=[(folder, 1.0)], method="awr", terms=given)
        buffer = offline.Buffer(settings, torch.device("cpu"), with_sets=False, with_rewards=True)
        buffer.scale = 1.0
        buffer.append(offline.listing(settings.sources))
        data = buffer.data()
        assert np.allclose(data.reward.double().numpy(), expected[dataset.usable()], atol=1e-6)
        periods = dataset.arrays["periods"][dataset.usable()].astype(np.float64)
        assert np.allclose(data.gamma.double().numpy(), discount ** periods, atol=1e-6)
    assert not np.allclose(dataset.rewards(), dataset.rewards(other))


def _assert_tails_close_their_own_cut_trajectories(data):
    """Every decision of a cut trajectory names the last decision of its own trajectory as its tail, and no decision of an ended one names any."""
    tails = data.tail.numpy()
    linked = np.flatnonzero(tails >= 0)
    assert len(linked) and (tails < 0).any()
    assert np.array_equal(data.trajectory[tails[linked]], data.trajectory[linked])
    assert bool((data.next[tails[linked]] < 0).all()) and not bool(data.done[tails[linked]].any())
    ended = data.done.numpy()
    assert (tails[np.isin(data.trajectory, data.trajectory[ended])] < 0).all()


def test_the_capped_buffer_drops_the_oldest_actor_shards_and_keeps_the_named_runs(tmp_path):
    from rwintel.learn.dataset import shard_files

    base, actors = _three_shards(tmp_path)
    actor_shards = shard_files(actors)
    assert len(actor_shards) >= 3
    sources = [(base, 1.0), (actors, 1.0)]
    settings = offline.Settings(layer="tactics", sources=sources, method="iql")
    device = torch.device("cpu")
    every = offline.listing(sources)
    sizes = [len(offline.load_data(replace_sources(settings, [(path, 1.0)], shard), device, False, False))
             for path, _, shard in every]
    cap = sizes[0] + sizes[-1] + sizes[-2]
    buffer = offline.Buffer(settings, device, with_sets=False, with_rewards=True, cap=cap, kept=[base])
    for entry in every:
        buffer.append([entry])
    data = buffer.data()
    assert [segment[0] for segment in buffer.segments] == [every[0][2]] + actor_shards[-2:]
    assert len(data) == cap and buffer.dropped == len(actor_shards) - 2
    # The rows left are the named run's and the newest shards', whole, with their links renumbered.
    alone = offline.Buffer(settings, device, with_sets=False, with_rewards=True)
    alone.scale = buffer.scale
    alone.append([every[0]] + every[-2:])
    kept = alone.data()
    for name in ("flat", "action", "reward", "next", "done", "start", "held", "mc", "tail", "tail_gamma"):
        assert torch.equal(getattr(data, name), getattr(kept, name)), name
    _assert_tails_close_their_own_cut_trajectories(data)
    small = offline.Buffer(settings, device, with_sets=False, with_rewards=True, cap=1, kept=[base])
    small.append(every)
    assert len(small) == sizes[0] and len(small.segments) == 1


def _buffer_runs(tmp_path, with_sets: bool, shard_decisions: int = 80):
    """A named run of one shard and a watched run of several, with set states when asked."""
    if not with_sets:
        base = str(tmp_path / "base")
        _record(base, "tactics", _labelled_episodes(random.Random(5), 20, lambda action: action), shard_decisions=10 ** 6)
        actors = str(tmp_path / "actors")
        _record(actors, "tactics", _labelled_episodes(random.Random(6), 30, lambda action: action),
                shard_decisions=shard_decisions)
        return base, actors
    sealed = []
    _recorded_tactics(sealed)
    base, actors = str(tmp_path / "base"), str(tmp_path / "actors")
    _record(base, "tactics", sealed)
    _record(actors, "tactics", sealed * 4, shard_decisions=3)
    return base, actors


def _assert_same_rows(mine, theirs, skip=()):
    """Every field of two `Data` but `skip` holds the same values."""
    assert len(mine) == len(theirs) and mine.reward_scale == theirs.reward_scale
    for name in offline.Data.__dataclass_fields__:
        if name in ("layer", "reward_scale") or name in skip:
            continue
        a, b = getattr(mine, name), getattr(theirs, name)
        if a is None or b is None:
            assert a is None and b is None, name
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b) if a.dtype == object else np.allclose(a, b), name
        else:
            assert a.dtype == b.dtype and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)), name
            assert torch.equal(a.isnan(), b.isnan()) if a.is_floating_point() else True, name


@pytest.mark.parametrize("with_sets", [False, True])
def test_appending_to_a_capped_buffer_writes_in_place_without_reallocating(tmp_path, with_sets):
    base, actors = _buffer_runs(tmp_path, with_sets)
    sources = [(base, 1.0), (actors, 1.0)]
    settings = offline.Settings(layer="tactics", sources=sources, method="iql")
    device = torch.device("cpu")
    every = offline.listing(sources)
    assert len(every) >= 5
    counted = offline.Buffer(settings, device, with_sets=False, with_rewards=True)
    counted.append(every)
    sizes = [rows for _, rows, _ in counted.segments]
    cap = sizes[0] + sizes[-1] + sizes[-2]
    buffer = offline.Buffer(settings, device, with_sets=with_sets, with_rewards=True, cap=cap, kept=[base])

    def pointers():
        return buffer.storage["flat"].data_ptr(), buffer.sets.data_ptr() if with_sets else None

    buffer.append(every[:2])
    first = pointers()
    assert buffer.capacity == cap + offline.APPEND_HEADROOM
    for entry in every[2:]:
        buffer.append([entry])
        assert pointers() == first
    assert buffer.dropped == len(every) - 3 and len(buffer) == buffer.length == cap
    # The rows left equal a fresh read of the kept shards; episode and trajectory numbers keep counting the dropped shards.
    alone = offline.Buffer(settings, device, with_sets=with_sets, with_rewards=True)
    alone.scale = buffer.scale
    alone.append([every[0]] + every[-2:])
    _assert_same_rows(buffer.data(), alone.data(), skip=("episode", "trajectory"))


def test_an_uncapped_buffer_grows_its_storage_by_half_at_a_time(tmp_path):
    _, actors = _buffer_runs(tmp_path, with_sets=False, shard_decisions=20)
    sources = [(actors, 1.0)]
    settings = offline.Settings(layer="tactics", sources=sources, method="iql")
    device = torch.device("cpu")
    every = offline.listing(sources)
    assert len(every) >= 10
    whole = offline.load_data(settings, device, with_sets=False, with_rewards=True)
    buffer = offline.Buffer(settings, device, with_sets=False, with_rewards=True)
    buffer.scale = whole.reward_scale
    seen = []
    for entry in every:
        buffer.append([entry])
        seen.append(buffer.storage["flat"].data_ptr())
        assert buffer.capacity >= buffer.length
    first = len(buffer.segments) and buffer.segments[0][1]
    assert first > 0 and len(set(seen)) <= math.ceil(math.log(len(buffer) / first, 1.5)) + 1
    _assert_same_rows(buffer.data(), whole)
    # Reading at once reserves exactly the rows read.
    reading = offline.Buffer(settings, device, with_sets=False, with_rewards=True)
    reading.append(every)
    assert reading.capacity == reading.length == len(whole)


def test_the_card_share_limits_the_process_only_on_a_graphics_card(monkeypatch):
    from types import SimpleNamespace

    from rwintel.learn.__main__ import _serving_device

    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda share, device=None: calls.append((share, device)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=12 * 2 ** 30))
    models.limit_card(torch.device("cuda"), 0.5)
    assert calls == [(0.5, None)]
    models.limit_card(torch.device("cuda", 1), 1)
    assert calls[-1] == (1.0, 1) and isinstance(calls[-1][0], float)
    models.limit_card(torch.device("cpu"), 0.5)
    models.limit_card(torch.device("cuda"), None)
    assert len(calls) == 2
    for share in (0, 1.5):
        with pytest.raises(ValueError):
            models.limit_card(torch.device("cuda"), share)
    assert _serving_device("cpu", None, 0.3).type == "cpu" and len(calls) == 2
    assert _serving_device("cuda", None, 0.3).type == "cuda" and calls[-1] == (0.3, None)
    # A following learner takes FOLLOW_CARD_SHARE unless told; a one-shot run takes no share unless told.
    settings = offline.Settings(layer="tactics", sources=[])
    assert settings.card_share is None and offline.follow_card_share(settings) == offline.FOLLOW_CARD_SHARE == 0.5
    settings.card_share = 0.3
    assert offline.follow_card_share(settings) == 0.3


def test_online_gives_the_learner_and_the_actors_their_card_shares_and_the_actor_device():
    from rwintel.learn import online
    from rwintel.learn.__main__ import build_parser

    parser = build_parser()

    def planned(*extra):
        argv = ["online", "--layer", "tactics", "--dataset", "base", "--save", "actor.pt", "--device", "cuda", *extra]
        arguments = parser.parse_args(argv)
        arguments.stamp = "stamp"
        learner, actors = online.split(parser, argv)
        return online.commands(arguments, learner, actors)

    def value(command, flag):
        return command[command.index(flag) + 1] if flag in command else None

    plain = planned()
    # Without --card-share the learner is left to its own default (FOLLOW_CARD_SHARE), the actors take ACTOR_CARD_SHARE.
    assert value(plain["learner"], "--card-share") is None and value(plain["learner"], "--device") == "cuda"
    assert value(plain["actors"], "--card-share") == "0.1" and value(plain["actors"], "--device") is None
    assert online.ACTOR_CARD_SHARE == 0.1
    given = planned("--card-share", "0.4", "--actor-device", "cpu", "--actor-card-share", "0.2")
    assert value(given["learner"], "--card-share") == "0.4" and given["learner"].count("--card-share") == 1
    assert value(given["learner"], "--device") == "cuda" and "--actor-device" not in given["learner"]
    assert value(given["actors"], "--device") == "cpu" and value(given["actors"], "--card-share") == "0.2"
    assert given["actors"].count("--card-share") == 1 and given["actors"].count("--device") == 1
    assert "--actor-device" not in given["actors"] and "--actor-card-share" not in given["actors"]


def replace_sources(settings, sources, shard):
    """Settings reading only one shard of one run, for counting a shard's usable decisions."""
    import shutil

    folder = os.path.join(os.path.dirname(os.path.dirname(shard)), "one-" + os.path.basename(shard)[:-4])
    if not os.path.exists(folder):
        os.makedirs(folder)
        shutil.copy(os.path.join(os.path.dirname(shard), "run.json"), folder)
        shutil.copy(shard, os.path.join(folder, "shard-00000.npz"))
    from dataclasses import replace

    return replace(settings, sources=[(folder, 1.0)])


# The offline runner.

def _labelled_run(tmp_path, count=40):
    folder = str(tmp_path / "labelled")
    _record(folder, "tactics", _labelled_episodes(random.Random(5), count, lambda action: action), shard_decisions=200)
    return folder


def test_offline_bc_and_distill_fit_a_plain_teacher(tmp_path):
    folder = _labelled_run(tmp_path)
    report_path = str(tmp_path / "bc.json")
    teacher_path = str(tmp_path / "teacher.pt")
    report = offline.run(offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="bc", device="cpu", epochs=40,
                                          batch=64, learning_rate=5e-3, patience=40, save=teacher_path,
                                          report=report_path, seed=1))
    assert report["validation"]["accuracy"] > 0.9 and report["epochs"][0]["seconds"] >= 0.0
    with open(report_path, encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["method"] == "bc" and written["behaviour_distance"]["count"] > 0 and "by_behaviour" in written["validation"]
    student = offline.run(offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="distill", device="cpu",
                                           epochs=40, batch=64, learning_rate=5e-3, patience=40,
                                           teacher=teacher_path, width=16,
                                           report=str(tmp_path / "distill.json"), seed=1))
    assert student["validation"]["teacher_agreement"] > 0.8 and student["config"]["width"] == 16


def test_offline_bc_trains_a_tactical_set_net_from_the_materials(tmp_path):
    sealed = []
    _recorded_tactics(sealed)
    folder = str(tmp_path / "run")
    _record(folder, "tactics", sealed)
    path = str(tmp_path / "set.pt")
    report = offline.run(offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="bc", net="set", depth=1,
                                          device="cpu", epochs=2, batch=8, save=path, report=str(tmp_path / "r.json")))
    assert report["net"] == "set" and report["epochs"] and isinstance(models.load(path), TacticalSetNet)
    flat_path = str(tmp_path / "flat.pt")
    models.save(TacticalNet(width=16), flat_path)
    residual = str(tmp_path / "residual.pt")
    report = offline.run(offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="bc", net="set", depth=1,
                                          device="cpu", epochs=2, batch=8, save=residual, init_flat=flat_path,
                                          report=str(tmp_path / "r2.json")))
    assert report["init_flat"] == flat_path and report["held_flat"]
    trained, flat = models.load(residual), models.load(flat_path)
    assert all(torch.equal(value, flat.state_dict()[name]) for name, value in trained.flat.state_dict().items())
    assert any(bool(p.abs().sum() > 0) for p in trained.action.parameters())


def test_the_imitation_rate_warms_up_then_decays_to_its_floor():
    batch, total = 8, 50

    def rates(decay):
        params = [torch.nn.Parameter(torch.zeros(3))]
        settings = offline.Settings(layer="tactics", sources=[], learning_rate=1e-3, decay=decay)
        optimiser, schedule, rate = offline._optimiser(params, settings, "flat", batch, 10 * batch, total_steps=total)
        seen = [optimiser.param_groups[0]["lr"]]
        for _ in range(total):
            optimiser.step()
            schedule.step()
            seen.append(optimiser.param_groups[0]["lr"])
        return seen, rate

    seen, rate = rates("cosine")
    warmup = 10
    peak = seen.index(max(seen))
    assert peak == warmup - 1 and math.isclose(seen[peak], rate)
    assert all(later <= earlier for earlier, later in zip(seen[peak:], seen[peak + 1:]))
    assert abs(seen[-1] - offline.DECAY_FLOOR * rate) < 1e-6
    held, rate = rates("none")
    assert all(math.isclose(value, rate) for value in held[warmup - 1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graphics card")
def test_the_bf16_path_runs_on_a_graphics_card(tmp_path):
    folder = _labelled_run(tmp_path, count=20)
    report = offline.run(offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="bc", net="flat", device="cuda",
                                          epochs=2, report=str(tmp_path / "gpu.json")))
    assert report["peak_gpu_mb"] > 0.0 and report["device"].startswith("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a graphics card")
def test_set_states_that_do_not_fit_on_the_card_are_copied_over_per_batch(tmp_path, monkeypatch):
    sealed = []
    _recorded_tactics(sealed)
    folder = str(tmp_path / "run")
    _record(folder, "tactics", sealed)
    monkeypatch.setattr(offline, "SET_SHARE", 0.0)
    settings = offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="bc", net="set", depth=1, device="cuda")
    data = offline.load_data(settings, torch.device("cuda"), with_sets=True, with_rewards=False)
    assert data.sets.device.type == "cpu" and data.sets.is_pinned() and data.label.device.type == "cuda"
    report = offline.run(offline.Settings(layer="tactics", sources=[(folder, 1.0)], method="bc", net="set", depth=1,
                                          device="cuda", epochs=1, report=str(tmp_path / "r.json")))
    assert report["epochs"]


# Offline reinforcement learning on toy data.

def _toy(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, following: np.ndarray, done: np.ndarray,
         start: np.ndarray, episode: np.ndarray, gamma: float = 1.0, behaviour=None, held=None, labels=None,
         weights=None) -> offline.Data:
    count = len(actions)
    flat = torch.zeros(count, TACTICAL_SIZE)
    flat[:, :states.shape[1]] = torch.as_tensor(states, dtype=torch.float32)
    beta = np.full((count, TACTICAL_ACTIONS), 1.0 / TACTICAL_ACTIONS) if behaviour is None else behaviour
    held = np.zeros(count, dtype=bool) if held is None else held
    labels = actions if labels is None else labels
    weights = np.ones(count) if weights is None else weights
    gammas = np.full(count, gamma)
    summed, tail, factor = offline._bootstrapped(np.asarray(rewards, dtype=np.float64), gammas, following, done)

    def t(values, dtype):
        return torch.as_tensor(np.asarray(values), dtype=dtype)

    return offline.Data(
        layer="tactics", flat=flat, sets=None, mask=torch.ones(count, TACTICAL_ACTIONS, dtype=torch.uint8), table=None,
        label=t(labels, torch.long), second_label=torch.full((count,), -1), action=t(actions, torch.long),
        second=torch.full((count,), -1), squad=torch.zeros(count, dtype=torch.long), weight=t(weights, torch.float32),
        held=t(held, torch.bool), soft=torch.full((count, TACTICAL_ACTIONS), float("nan")),
        second_soft=torch.zeros(count, 0), behaviour=t(beta, torch.float32), second_behaviour=torch.zeros(count, 0),
        reward=t(rewards, torch.float32), gamma=t(gammas, torch.float32), next=t(following, torch.long),
        done=t(done, torch.bool), start=t(start, torch.bool), episode=t(episode, torch.long),
        mc=t(summed, torch.float32), tail=t(tail, torch.long), tail_gamma=t(factor, torch.float32),
        sources=np.asarray(["script"] * count, dtype=object), behaviours=np.asarray(["b"] * count, dtype=object),
        returns=np.asarray(rewards, dtype=np.float64), trajectory=np.arange(count))


def _two_state_toy(rewarded, labels=None, weights=None, seed=0):
    """Single-decision episodes in one of two states with uniformly drawn actions, paid `rewarded(state, action)`."""
    draw = np.random.default_rng(seed)
    count = 2048
    which = draw.integers(0, 2, count)
    actions = draw.integers(0, TACTICAL_ACTIONS, count)
    rewards = rewarded(which, actions, draw).astype(np.float64)
    label = None if labels is None else labels(which)
    weight = None if weights is None else weights(which, actions)
    return _toy(np.eye(2)[which], actions, rewards, np.full(count, -1), np.ones(count, bool), np.ones(count, bool),
                np.arange(count), held=np.arange(count) % 10 == 0, labels=label, weights=weight)


def _greedy_on_both_states(settings, data):
    torch.manual_seed(0)
    report = {"epochs": []}
    actor = offline._advantage_learning(settings, data, TacticalNet(width=32), 256, report)
    probe = torch.zeros(2, TACTICAL_SIZE)
    probe[0, 0] = probe[1, 1] = 1.0
    with torch.no_grad():
        logits, _ = actor(probe, None)
    return logits.argmax(dim=-1).tolist(), report


def _paid_for(first, second):
    return lambda which, actions, draw: actions == np.where(which == 0, first, second)


@pytest.mark.parametrize("method", ["iql", "awr"])
def test_the_advantage_learners_recover_the_better_action_of_a_two_state_toy(method):
    settings = offline.Settings(layer="tactics", sources=[], method=method, device="cpu", epochs=40, batch=256, seed=0,
                                judge=0.0)
    chosen, report = _greedy_on_both_states(settings, _two_state_toy(_paid_for(3, 5)))
    assert chosen == [3, 5] and report["judge"] == 0.0


@pytest.mark.parametrize("method", ["iql", "awr"])
def test_the_judge_weight_decides_between_the_judge_and_a_clear_advantage(method):
    data = _two_state_toy(_paid_for(3, 5), labels=lambda which: np.where(which == 0, 2, 4))
    light = offline.Settings(layer="tactics", sources=[], method=method, device="cpu", epochs=40, batch=256, seed=0,
                             judge=0.1)
    heavy = offline.Settings(layer="tactics", sources=[], method=method, device="cpu", epochs=40, batch=256, seed=0,
                             judge=10.0)
    assert _greedy_on_both_states(light, data)[0] == [3, 5]
    assert _greedy_on_both_states(heavy, data)[0] == [2, 4]


@pytest.mark.parametrize("method", ["iql", "awr"])
def test_the_judge_holds_the_policy_where_the_returns_are_noise(method):
    data = _two_state_toy(lambda which, actions, draw: draw.normal(0.0, 1.0, len(actions)),
                          labels=lambda which: np.where(which == 0, 2, 4))
    settings = offline.Settings(layer="tactics", sources=[], method=method, device="cpu", epochs=40, batch=256, seed=0,
                                judge=1.0)
    assert _greedy_on_both_states(settings, data)[0] == [2, 4]


@pytest.mark.parametrize("method", ["iql", "awr", "cql"])
def test_the_critic_epochs_fit_the_critic_and_leave_the_actor_where_it_started(method):
    data = _two_state_toy(_paid_for(3, 5))
    settings = offline.Settings(layer="tactics", sources=[], method=method, device="cpu", epochs=0, critic_epochs=2,
                                batch=256, seed=0)
    torch.manual_seed(0)
    actor = TacticalNet(width=32)
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    report = {"epochs": []}
    offline._advantage_learning(settings, data, actor, 256, report)
    assert all(torch.equal(value, before[name]) for name, value in actor.state_dict().items())
    assert [entry["critic_only"] for entry in report["epochs"]] == [True, True]
    assert report["epochs"][-1]["v_loss"] < report["epochs"][0]["v_loss"] or method != "awr"


def test_an_actor_read_from_a_file_moves_at_the_rate_imitation_ended_at():
    data = _two_state_toy(_paid_for(3, 5))
    fresh = offline.Settings(layer="tactics", sources=[], method="iql", device="cpu", learning_rate=1e-3)
    loaded = offline.Settings(layer="tactics", sources=[], method="iql", device="cpu", learning_rate=1e-3, load="actor.pt")
    learner = offline.AdvantageLearner(fresh, data, TacticalNet(width=8), 256, len(data))
    assert learner.actor_rate == learner.rate == 1e-3
    learner = offline.AdvantageLearner(loaded, data, TacticalNet(width=8), 256, len(data))
    assert math.isclose(learner.actor_rate, offline.DECAY_FLOOR * 1e-3) and learner.rate == 1e-3


def test_a_decision_of_weight_nought_moves_no_advantage_weighted_policy():
    # In state 0 the weighted decisions pay action 3 and the unweighted ones pay action 1.
    def rewarded(which, actions, draw):
        unweighted = np.arange(len(actions)) % 2 == 1
        return np.where(unweighted, actions == 1, actions == np.where(which == 0, 3, 5))

    data = _two_state_toy(rewarded, weights=lambda which, actions: (np.arange(len(actions)) % 2 == 0).astype(float))
    settings = offline.Settings(layer="tactics", sources=[], method="awr", device="cpu", epochs=40, batch=256, seed=0,
                                judge=0.0)
    assert _greedy_on_both_states(settings, data)[0] == [3, 5]


def test_a_cut_trajectory_is_bootstrapped_from_the_value_of_its_last_decision():
    # Rows 0-2 end; rows 3-5 are cut at row 5, whose value stands for its own reward and everything after it.
    rewards = np.asarray([1.0, 2.0, 4.0, 1.0, 2.0, 4.0])
    gamma = np.asarray([0.5, 0.5, 0.5, 0.5, 0.25, 0.5])
    following = np.asarray([1, 2, -1, 4, 5, -1])
    done = np.asarray([False, False, True, False, False, False])
    summed, tail, factor = offline._bootstrapped(rewards, gamma, following, done)
    assert np.allclose(summed, [1 + 0.5 * (2 + 0.5 * 4), 2 + 0.5 * 4, 4, 1 + 0.5 * 2, 2, 0])
    assert tail.tolist() == [-1, -1, -1, 5, 5, 5]
    assert np.allclose(factor, [0, 0, 0, 0.5 * 0.25, 0.25, 1])


def test_return_alignment_compares_agreement_on_the_top_and_bottom_trajectories():
    count = 8
    actions = np.asarray([0, 3, 0, 0, 0, 0, 0, 0])
    data = _toy(np.zeros((count, 1)), actions, np.arange(count, dtype=np.float64), np.full(count, -1),
                np.ones(count, bool), np.ones(count, bool), np.arange(count))
    logits = np.log(np.asarray([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02]))
    result = offline.return_alignment(_Fixed(logits), data, torch.arange(count), top=0.25)
    assert result["by_behaviour"]["b"]["top_agreement"] == 1.0 and result["by_behaviour"]["b"]["bottom_agreement"] == 0.5
    assert result["gap"] == 0.5 and result["count"] == 4


def test_fqe_of_a_fixed_policy_on_a_chain_matches_its_value():
    # Three decisions per episode with reward one each and a discount of a half: the value at the start is 1.75.
    # Every action is played, so the evaluated policy's choices all have data behind them.
    episodes = 600
    count = 3 * episodes
    position = np.tile(np.arange(3), episodes)
    states = np.eye(3)[position]
    following = np.where(position < 2, np.arange(count) + 1, -1)
    actions = np.random.default_rng(1).integers(0, TACTICAL_ACTIONS, count)
    data = _toy(states, actions, np.ones(count), following, position == 2, position == 0,
                np.repeat(np.arange(episodes), 3), gamma=0.5, held=np.repeat(np.arange(episodes) % 4 == 0, 3))
    settings = offline.Settings(layer="tactics", sources=[], method="fqe", device="cpu", epochs=80, batch=256,
                                learning_rate=3e-3, tau=0.05)
    torch.manual_seed(0)
    report = {"epochs": []}
    offline._evaluate(settings, data, TacticalNet(width=32), TacticalNet(width=32), 256, report)
    assert abs(report["fqe"]["mean"] - 1.75) < 0.1
    assert report["fqe"]["low"] <= report["fqe"]["mean"] <= report["fqe"]["high"]


def test_topbc_keeps_the_top_trajectories_of_each_behaviour():
    count = 8
    data = _toy(np.zeros((count, 1)), np.zeros(count, dtype=np.int64), np.arange(count, dtype=np.float64),
                np.full(count, -1), np.ones(count, bool), np.ones(count, bool), np.arange(count))
    data.behaviours = np.asarray(["a"] * 4 + ["b"] * 4, dtype=object)
    kept = offline._top_rows(data, np.arange(count), 0.5)
    assert kept.tolist() == [2, 3, 6, 7]


class _Fixed(torch.nn.Module):
    """A policy with the same logits on every board."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.as_tensor(logits, dtype=torch.float32)

    def forward(self, state, mask=None):
        logits = self.logits.expand(state.shape[0], -1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, torch.zeros(state.shape[0])


def test_the_distance_from_the_behaviour_policy_is_kl_and_the_off_data_share():
    logits = np.log(np.asarray([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02]))
    pi = np.exp(logits)
    beta = np.asarray([[0.0, 0.4, 0.2, 0.2, 0.1, 0.05, 0.05],
                       [0.3, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]])
    data = _toy(np.zeros((2, 1)), np.zeros(2, dtype=np.int64), np.zeros(2), np.full(2, -1), np.ones(2, bool),
                np.ones(2, bool), np.arange(2), behaviour=beta)
    result = offline.behaviour_distance(_Fixed(logits), data, torch.arange(2))
    expected = np.mean([np.sum(pi * (np.log(pi) - np.log(np.maximum(row, 1e-8)))) for row in beta])
    assert result["count"] == 2 and abs(result["kl"] - expected) < 1e-3
    assert result["off_data_share"] == 0.5 and result["untrusted"]
