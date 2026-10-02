"""What the vast.ai fleet tool decides without reaching vast.ai, held to.

The fleet tool spends money and destroys machines, so the decisions pinned here are the ones whose mistakes cost either: which offers are worth renting, how fast a machine is expected to be and when it is judged a dud, where a machine is reached, how spending is counted against the budget, when runs count as done or failed, which status changes wake the lead, whether a pull is whole, and that a machine is never destroyed before its results were pulled.

Nothing here talks to vast.ai or to a machine.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "vast-fleet" / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fleet = _load("vast_fleet", SKILL / "fleet.py")
runs = _load("vast_fleet_runs", SKILL / "remote" / "runs.py")

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _offer(**fields):
    base = {"id": 1, "machine_id": 10, "cpu_cores_effective": 32, "cpu_ram": 64 * 1024, "dph_total": 0.12, "reliability2": 0.99,
            "cpu_ghz": 4.5, "cpu_name": "AMD Ryzen 9 5950X 16-Core Processor", "gpu_name": "RTX 3060", "gpu_ram": 12 * 1024}
    base.update(fields)
    return base


def test_offers_are_ranked_by_price_per_thread_and_unfit_ones_dropped():
    offers = [
        _offer(id=1, dph_total=0.16),
        _offer(id=2, dph_total=0.30, cpu_cores_effective=64, cpu_ram=128 * 1024),
        _offer(id=3, cpu_ram=31 * 1024),
        _offer(id=4, reliability2=0.9),
        _offer(id=5, cpu_ghz=2.2),
        _offer(id=6, gpu_name="RTX 4060 laptop"),
        _offer(id=7, cpu_name="Core i9-14900HX"),
        _offer(id=8, cpu_cores_effective=16, cpu_ram=64 * 1024),
        _offer(id=9, gpu_ram=8 * 1024),
    ]
    ranked = fleet.rank_offers(offers, min_vram_gb=11)
    assert [o["offer"] for o in ranked] == [2, 1]
    assert ranked[0]["per_thread"] < ranked[1]["per_thread"]


def test_a_machine_is_reached_on_its_direct_port_before_the_proxy():
    direct = {"public_ipaddr": " 1.2.3.4 ", "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41278"}]}, "ssh_host": "ssh5.vast.ai", "ssh_port": 36446}
    assert fleet.ssh_address(direct) == ("1.2.3.4", 41278)
    proxied = {"public_ipaddr": "1.2.3.4", "ports": {}, "ssh_host": "ssh5.vast.ai", "ssh_port": 36446}
    assert fleet.ssh_address(proxied) == ("ssh5.vast.ai", 36446)
    assert fleet.ssh_address({}) == (None, None)


def test_labels_name_their_fleet_and_machine_even_when_the_fleet_has_dashes():
    text = fleet.label("1002-2105", "m3")
    assert text == "rwf-1002-2105-m3"
    assert fleet.fleet_of_label(text) == ("1002-2105", "m3")
    assert fleet.fleet_of_label("rw-a") is None
    assert fleet.fleet_of_label(None) is None


def test_a_clock_deadline_is_the_next_such_time():
    local = timezone(timedelta(hours=9))
    morning = datetime(2026, 1, 1, 8, 0, tzinfo=local)
    assert fleet.parse_deadline("09:30", morning) == datetime(2026, 1, 1, 9, 30, tzinfo=local).astimezone(timezone.utc)
    evening = datetime(2026, 1, 1, 22, 0, tzinfo=local)
    assert fleet.parse_deadline("09:30", evening) == datetime(2026, 1, 2, 9, 30, tzinfo=local).astimezone(timezone.utc)
    assert fleet.parse_deadline(None) is None


def test_a_clock_deadline_is_read_in_the_zone_of_the_moment_whatever_the_host_zone(monkeypatch):
    utc_morning = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    expected = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    for zone in ("UTC", "Asia/Tokyo", "America/New_York"):
        monkeypatch.setenv("TZ", zone)
        if hasattr(time_module, "tzset"):
            time_module.tzset()
        assert fleet.parse_deadline("09:30", utc_morning) == expected
    monkeypatch.delenv("TZ")
    if hasattr(time_module, "tzset"):
        time_module.tzset()


def test_paths_under_local_are_taken_with_or_without_their_prefix():
    assert fleet.relative_to_local("local/episodes/a.jsonl") == "episodes/a.jsonl"
    assert fleet.relative_to_local("./local/datasets/x") == "datasets/x"
    assert fleet.relative_to_local("models/m.pt") == "models/m.pt"
    for outside in ("/etc/passwd", "../secret", "episodes/../../x"):
        with pytest.raises(fleet.FleetError):
            fleet.relative_to_local(outside)


def test_spending_takes_the_larger_of_prices_over_time_and_the_fall_in_credit():
    machines = [
        {"dph": 0.2, "rented": fleet.stamp(T0), "destroyed": None},
        {"dph": 0.4, "rented": fleet.stamp(T0), "destroyed": fleet.stamp(T0 + timedelta(minutes=30))},
        {"dph": 1.0, "rented": None},
    ]
    later = T0 + timedelta(hours=2)
    assert fleet.spend(machines, later, 10.0, 9.9) == pytest.approx(0.2 * 2 + 0.4 * 0.5)
    assert fleet.spend(machines, later, 10.0, 8.0) == pytest.approx(2.0)
    assert fleet.spend(machines, later, None, None) == pytest.approx(0.6)


def test_runs_are_done_failed_or_going_by_their_exit_codes():
    text = "\n".join([
        "noise before",
        json.dumps({"name": "collect-ops-0", "exit": 0, "finished": 48, "journal": 48, "tracebacks": 0}),
        json.dumps({"name": "collect-ops-1", "exit": None, "finished": 20, "journal": 20, "tracebacks": 0}),
        json.dumps({"name": "fit-ops", "exit": 1, "finished": 0, "journal": 0, "tracebacks": 1}),
        json.dumps({"host": "h", "load": 20.0, "mem_used_gb": 21.0, "disk_free_gb": 30.0}),
    ])
    parsed = fleet.parse_runs(text)
    assert len(parsed) == 4
    code, line = fleet.judge_runs(parsed, "collect-", 2)
    assert code == fleet.TIMED_OUT and "1/2 ended" in line and "68 finished" in line
    finished = [dict(r, exit=0) if r.get("name") == "collect-ops-1" else r for r in parsed]
    assert fleet.judge_runs(finished, "collect-", 2)[0] == fleet.DONE
    code, line = fleet.judge_runs(parsed, "fit-", 1)
    assert code == fleet.FAILED and "fit-ops=1" in line
    assert fleet.judge_runs(parsed, "eval-", 0)[0] == fleet.TIMED_OUT


def test_the_lead_is_woken_by_settled_stalled_or_vanished_machines_and_once_by_the_budget():
    seen = {"m1": {"state": "running"}, "m2": {"state": "done"}}
    fresh = fleet.stamp(T0)
    now = {"m1": {"state": "done", "message": "pulled", "updated": fresh}, "m2": {"state": "done", "updated": fresh},
           "m3": {"state": "running", "updated": fleet.stamp(T0 - timedelta(hours=1))}}
    events = fleet.watch_events(seen, now, T0, spent=1.0, budget=10.0, vanished=["m4"])
    assert events == ["m1 done: pulled", "m3 stale: no status update for 60 minutes while running", "m4 vanished: its instance is no longer running"]
    assert fleet.watch_events(now, now, T0, 1.0, 10.0) == ["m3 stale: no status update for 60 minutes while running"]
    quiet = {"m1": {"state": "running", "updated": fresh}}
    assert fleet.watch_events(quiet, quiet, T0, 8.5, 10.0) == ["budget: $8.50 spent of $10.00"]
    assert fleet.watch_events(quiet, quiet, T0, 8.5, 10.0, warned=True) == []


def test_a_pull_is_whole_only_when_every_remote_file_arrived_unchanged(tmp_path):
    (tmp_path / "datasets" / "run").mkdir(parents=True)
    (tmp_path / "datasets" / "run" / "a.bin").write_bytes(b"alpha")
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / "j.jsonl").write_bytes(b"{}\n")
    local = fleet.local_sums(tmp_path, ["datasets", "episodes/j.jsonl"])
    listing = "\n".join(f"{digest}  ./{path}" for path, digest in local.items())
    remote = fleet.parse_sums(listing)
    assert fleet.compare_manifests(remote, local) == []
    remote["datasets/run/b.bin"] = "0" * 64
    remote["episodes/j.jsonl"] = "f" * 64
    assert fleet.compare_manifests(remote, local) == ["datasets/run/b.bin", "episodes/j.jsonl"]


@pytest.fixture
def fleet_root(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, "FLEETS", tmp_path)
    calls = []
    monkeypatch.setattr(fleet, "vast", lambda *a, **k: calls.append(a) or {})
    return tmp_path, calls


def _machine(root: Path, name: str = "m1") -> Path:
    workdir = root / "f1" / "machines" / name
    fleet.write_json(workdir / "machine.json", {"fleet": "f1", "machine": name, "instance": 42, "dph": 0.1, "rented": fleet.stamp(T0), "destroyed": None})
    return workdir


def test_a_machine_is_not_destroyed_before_a_final_verified_pull(fleet_root):
    root, calls = fleet_root
    workdir = _machine(root)
    args = type("A", (), {"fleet": "f1", "machine": "m1", "force": False})
    with pytest.raises(fleet.FleetError) as refused:
        fleet.cmd_destroy(args)
    assert refused.value.code == fleet.UNSAFE and calls == []
    fleet.write_json(workdir / "pulled.json", {"ok": True, "final": False, "pulls": []})
    with pytest.raises(fleet.FleetError):
        fleet.cmd_destroy(args)
    fleet.write_json(workdir / "pulled.json", {"ok": True, "final": True, "pulls": []})
    assert fleet.cmd_destroy(args) == fleet.DONE
    assert calls == [("destroy", "instance", "42", "-y")]
    assert fleet.read_json(workdir / "machine.json")["destroyed"]
    assert fleet.cmd_destroy(args) == fleet.DONE and len(calls) == 1


def test_a_forced_destroy_goes_through_without_a_pull(fleet_root):
    root, calls = fleet_root
    _machine(root)
    assert fleet.cmd_destroy(type("A", (), {"fleet": "f1", "machine": "m1", "force": True})) == fleet.DONE
    assert calls == [("destroy", "instance", "42", "-y")]


def test_notes_keep_the_status_whole_and_log_every_change(fleet_root):
    root, _ = fleet_root
    workdir = _machine(root)
    note = lambda **k: fleet.cmd_note(type("A", (), {"workdir": str(workdir), "task": None, "progress": None, "message": None, **k}))
    note(state="running", task="collect", progress="0/6")
    note(state="running", progress="3/6")
    status = fleet.read_json(workdir / "status.json")
    assert status["state"] == "running" and status["task"] == "collect" and status["progress"] == "3/6"
    assert len((workdir / "log.md").read_text().splitlines()) == 2
    with pytest.raises(fleet.FleetError):
        note(state="sleeping")


def test_the_remote_summary_reads_each_run_and_its_journal(tmp_path, monkeypatch):
    root = tmp_path / "rw-intel"
    vps = root / "local" / "vps"
    vps.mkdir(parents=True)
    (root / "local" / "episodes").mkdir()
    (root / "local" / "episodes" / "c0.jsonl").write_text("{}\n{}\n{}\n")
    (vps / "collect-0.pid").write_text(str(os.getpid()))
    (vps / "collect-0.cmd").write_text("python -m rwintel.learn collect --record local/episodes/c0.jsonl --episodes 3 ")
    (vps / "collect-0.out").write_text("x finished: 1\nx finished: 2\nlast line\n")
    (vps / "collect-1.pid").write_text("999999999")
    (vps / "collect-1.exit").write_text("1\n")
    (vps / "collect-1.out").write_text("Traceback (most recent call last):\nboom\n")
    (vps / "collect-2.pid").write_text("999999998")
    (vps / "collect-2.exit").write_text("0\n")
    (vps / "collect-2.cmd").write_text("python -m rwintel.runtime run --count 24 --controls 6 -- eval ")
    (vps / "collect-2.out").write_text("throughput: 10 episode(s), 6000 game seconds in 30s, 200.0 game seconds per second\n"
                                       "throughput: 48 episode(s), 28800 game seconds in 41s, 702.4 game seconds per second\n")
    (vps / "fit-0.pid").write_text("1")
    monkeypatch.setattr(runs, "ROOT", str(root))
    monkeypatch.setattr(runs, "VPS", str(vps))
    monkeypatch.setattr(sys, "argv", ["runs.py", "collect-"])
    lines = []
    monkeypatch.setattr("builtins.print", lambda text: lines.append(text))
    runs.main()
    parsed = fleet.parse_runs("\n".join(lines))
    first, second, third, host = parsed
    assert first == {"name": "collect-0", "exit": None, "alive": True, "finished": 2, "tracebacks": 0, "journal": 3, "last": "last line",
                     "command": "python -m rwintel.learn collect --record local/episodes/c0.jsonl --episodes 3", "throughput": None}
    assert second["exit"] == 1 and second["alive"] is False and second["tracebacks"] == 1
    assert third["throughput"] == {"episodes": 48, "game_seconds": 28800, "wall_seconds": 41, "game_seconds_per_second": 702.4}
    assert "host" in host and "disk_free_gb" in host
    assert fleet.judge_runs(parsed, "collect-", 3)[0] == fleet.FAILED


def _bench(machine_id, cpu, ghz, value, games=24, kind="bench"):
    return {"kind": kind, "machine_id": machine_id, "cpu": cpu, "ghz": ghz, "value": value, "games": games}


def test_a_speed_is_estimated_from_the_same_machine_then_the_same_processor_then_the_model():
    ledger = [_bench(1, "Ryzen 9 5900XT", 5.0, 720.0), _bench(1, "Ryzen 9 5900XT", 5.0, 680.0), _bench(2, "EPYC 7B13", 3.5, 480.0),
              _bench(3, "EPYC 7B13", 3.5, 500.0, games=20), _bench(4, "Xeon E5-2690", 3.0, 999.0, kind="boot")]
    machine = fleet.estimate_speed({"machine_id": 1, "cpu": "EPYC 7B13", "ghz": 3.5, "threads": 32}, ledger)
    assert machine == {"speed": 700.0, "basis": "machine", "samples": 2, "games": 24}
    cpu = fleet.estimate_speed({"machine_id": 9, "cpu": " EPYC 7B13 ", "ghz": 3.5, "threads": 32}, ledger)
    assert cpu["basis"] == "cpu" and cpu["speed"] == pytest.approx((480 / 24 + 500 / 20) / 2 * 24, abs=0.1)
    model = fleet.estimate_speed({"machine_id": 9, "cpu": "Core i9-14900K", "ghz": 6.0, "threads": 16}, ledger)
    per_ghz = sorted([720 / 24 / 5, 680 / 24 / 5, 480 / 24 / 3.5, 500 / 20 / 3.5])
    assert model["basis"] == "model" and model["games"] == 12 and model["speed"] == pytest.approx((per_ghz[1] + per_ghz[2]) / 2 * 6.0 * 12, abs=0.1)
    assert fleet.estimate_speed({"cpu": "x", "ghz": 4.0, "threads": 32}, []) == {"speed": None, "basis": "none", "samples": 0, "games": 24}


def test_games_per_machine_are_three_for_every_four_threads_in_fours():
    assert [fleet.default_games(t) for t in (8, 18, 28, 32, 48, 64, 256, None)] == [4, 12, 20, 24, 36, 48, 48, 4]


def test_a_bench_is_ok_slow_or_a_dud_by_its_fraction_of_the_expected_speed():
    assert fleet.judge_ratio(700, 700) == ("ok", 1.0)
    assert fleet.judge_ratio(560, 700) == ("ok", 0.8)
    assert fleet.judge_ratio(500, 700)[0] == "slow"
    assert fleet.judge_ratio(420, 700) == ("slow", 0.6)
    assert fleet.judge_ratio(300, 700)[0] == "dud"
    assert fleet.judge_ratio(300, None) == ("ok", None)
    assert fleet.judge_ready_time(fleet.HOST_READY_LIMIT) == "ok"
    assert fleet.judge_ready_time(fleet.HOST_READY_LIMIT + 1) == "dud"


def test_a_bench_is_held_to_the_estimate_else_to_the_fleet_else_to_nothing():
    estimated = {"machine": "m2", "games": 24, "estimate": {"speed": 600.0, "basis": "cpu", "games": 24}}
    assert fleet.bench_reference(estimated, []) == (600.0, "cpu")
    regamed = {"machine": "m2", "games": 12, "estimate": {"speed": 600.0, "basis": "machine", "games": 24}}
    assert fleet.bench_reference(regamed, []) == (300.0, "machine")
    peers = [{"machine": "m1", "bench": {"value": 720.0, "games": 24}}, {"machine": "m3", "bench": {"value": 360.0, "games": 12}},
             {"machine": "m2", "bench": {"value": 10.0, "games": 24}}]
    unknown = {"machine": "m2", "games": 24, "estimate": {"speed": None, "basis": "none", "games": 24}}
    assert fleet.bench_reference(unknown, peers) == (pytest.approx(720.0), "fleet")
    assert fleet.bench_reference(unknown, []) == (None, "none")


def test_offers_with_an_estimate_come_first_cheapest_per_game_second():
    ranked = [{"offer": 1, "machine": 10, "cpu": "Xeon", "ghz": 3.5, "threads": 32, "price": 0.10, "reliability": 0.99, "per_thread": 0.003},
              {"offer": 2, "machine": 20, "cpu": "Ryzen", "ghz": 5.0, "threads": 32, "price": 0.20, "reliability": 0.99, "per_thread": 0.006},
              {"offer": 3, "machine": 30, "cpu": "Ryzen", "ghz": 5.0, "threads": 32, "price": 0.12, "reliability": 0.99, "per_thread": 0.004}]
    ledger = [_bench(20, "Ryzen", 5.0, 900.0), _bench(30, "Ryzen", 5.0, 900.0)]
    out = fleet.with_estimates([dict(o) for o in ranked], ledger)
    # The Xeon has no bench of its own, so the model prices it from its clock: slower than the Ryzens but cheaper per game second than offer 2.
    assert [(o["offer"], o["basis"]) for o in out] == [(3, "machine"), (1, "model"), (2, "machine")]
    assert out[0]["per_mgs"] == pytest.approx(0.12 / (900 * 3600) * 1e6, abs=1e-3)
    assert out[1]["est"] == pytest.approx(900 / 24 / 5.0 * 3.5 * 24, abs=0.1)
    unknown = fleet.with_estimates([dict(ranked[0], ghz=0)], ledger)
    assert unknown[0]["basis"] == "none" and unknown[0]["per_mgs"] is None
    empty = fleet.with_estimates([dict(o) for o in ranked], [])
    assert [o["offer"] for o in empty] == [1, 2, 3] and all(o["basis"] == "none" and o["per_mgs"] is None for o in empty)


def test_the_ledger_keeps_every_measurement_with_the_machine_it_was_taken_on(fleet_root):
    root, _ = fleet_root
    machine = {"fleet": "f1", "machine": "m1", "instance": 42, "machine_id": 7, "cpu": "Ryzen", "ghz": 5.0, "threads": 32, "ram_gb": 63,
               "gpu": "RTX 3060", "vram_gb": 12, "dph": 0.11, "games": 24}
    assert fleet.read_ledger() == []
    fleet.append_speed(machine, "boot", 95, "seconds")
    fleet.append_speed(machine, "bench", 702.44449, "game seconds per second", games=24)
    entries = fleet.read_ledger()
    assert [e["kind"] for e in entries] == ["boot", "bench"] and entries[1]["value"] == 702.4445 and entries[1]["games"] == 24
    assert all(e["machine_id"] == 7 and e["cpu"] == "Ryzen" and e["dph"] == 0.11 and e["fleet"] == "f1" for e in entries)
    assert fleet.ledger_path() == root / "speeds.jsonl"


def test_a_rented_offer_takes_its_clock_and_ram_share_from_the_last_listing(fleet_root):
    assert fleet.offer_spec(50747399) == {}
    fleet.write_json(fleet.offers_path(), {"when": fleet.stamp(T0), "offers": [{"offer": 50747399, "machine": 150645, "ghz": 3.5, "ram_gb": 63, "threads": 32.0}]})
    assert fleet.offer_spec(50747399)["ghz"] == 3.5 and fleet.offer_spec(50747399)["ram_gb"] == 63
    assert fleet.offer_spec(1) == {}


def test_a_finished_run_is_written_to_the_ledger_once_and_the_bench_never(fleet_root):
    root, _ = fleet_root
    workdir = _machine(root)
    machine = fleet.read_json(workdir / "machine.json")
    speed = {"episodes": 48, "game_seconds": 28800, "wall_seconds": 41, "game_seconds_per_second": 702.4}
    runs_seen = [{"name": "eval-g1", "exit": 0, "throughput": speed, "command": "python -m rwintel.runtime run --count 24 --controls 6 -- eval"},
                 {"name": "eval-g2", "exit": 1, "throughput": speed}, {"name": "eval-g3", "exit": None, "throughput": None},
                 {"name": fleet.BENCH_RUN, "exit": 0, "throughput": speed}]
    fleet.record_runs(workdir, machine, runs_seen)
    fleet.record_runs(workdir, fleet.read_json(workdir / "machine.json"), runs_seen)
    entries = fleet.read_ledger()
    assert len(entries) == 1 and entries[0]["kind"] == "run" and entries[0]["run"] == "eval-g1" and entries[0]["run_games"] == 24
    assert fleet.read_json(workdir / "machine.json")["recorded_runs"] == ["eval-g1"]


def test_the_lead_is_woken_once_by_each_dud_verdict():
    events = fleet.watch_events({}, {}, T0, 0.0, 10.0, duds=[("m2", "setup", "boot and setup passed 16 minutes")])
    assert events == ["m2 dud (setup): boot and setup passed 16 minutes"]
    machine = {}
    fleet.set_verdict(machine, "setup", "ok", "boot 90s and setup 300s")
    fleet.set_verdict(machine, "bench", "dud", "bench 200 game s/s")
    assert fleet.duds_of(machine) == [("bench", "bench 200 game s/s")]
