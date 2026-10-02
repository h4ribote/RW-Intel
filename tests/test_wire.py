"""Holds the two halves of the wire format to the same layout.

The Java side writes these blocks by hand, so a field added on one side and forgotten on the other would not fail to compile; it would decode into the wrong fields and produce a plausible looking observation. Pinning the block sizes here makes that a test failure instead. The sizes are also stated in `docs/project/05-interface.md`.

The decode test builds a body with struct.pack rather than going through an encoder, because there is no Python encoder for observations: the only writer is the Java agent, so a round trip against our own decoder would prove nothing about the layout. Hand packed bytes are the closest stand-in for what the agent actually sends.
"""

from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.wire import Kind, decode_header, encode
from rwintel.wire.action import (
    Action,
    CargoKind,
    Commander,
    Contract,
    Deviation,
    Lift,
    Production,
    ProductionKind,
    SquadAssignment,
    SquadDeviation,
    Stance,
    Status,
    TargetKind,
    Task,
    UnitAction,
    _CONTRACT,
    _DEVIATION,
    _LIFT as _LIFT_ROW,
    _PRODUCTION,
    _SQUAD_HEADER,
    _UNIT_ACTION,
    decode_action,
    encode_action,
)
from rwintel.wire.frames import FLAGS_MASK, HEADER_SIZE, PROTOCOL_VERSION
from rwintel.wire.observation import (
    AI_ORDER_NO_UNIT,
    AiOrderKind,
    BLOCK_AI_ORDERS,
    BLOCK_EVENTS,
    BLOCK_LIFTS,
    BLOCK_MENUS,
    BLOCK_REGIONS,
    BLOCK_SQUADS,
    BLOCK_TIMING,
    BLOCK_UNITS,
    NO_LIFT,
    PASSAGE_CLASSES,
    REGION_SLOTS,
    SQUAD_SLOTS,
    LiftFailure,
    LiftPhase,
    _COMMON,
    _EVENT,
    _LIFT,
    _REGION,
    _SQUAD,
    _UNIT,
    decode_observation,
)


def test_protocol_version_is_ten_on_both_sides():
    import re

    assert PROTOCOL_VERSION == 10
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "Wire.java"),
              encoding="utf-8") as handle:
        source = handle.read()
    assert int(re.search(r"PROTOCOL_VERSION = (\d+);", source).group(1)) == PROTOCOL_VERSION
    assert int(re.search(r"BLOCK_TIMING = (\d+);", source).group(1)) == BLOCK_TIMING
    assert int(re.search(r"BLOCK_AI_ORDERS = (\d+);", source).group(1)) == BLOCK_AI_ORDERS


def test_block_sizes_match_the_agent():
    assert HEADER_SIZE == 16
    assert _COMMON.size == 40
    assert _REGION.size == 28
    assert _SQUAD.size == 60
    assert _UNIT.size == 44
    assert _LIFT.size == 20
    assert _EVENT.size == 16


def test_action_section_sizes_match_the_agent():
    assert _SQUAD_HEADER.size == 8
    assert _CONTRACT.size == 24
    assert _DEVIATION.size == 4
    assert _PRODUCTION.size == 16
    assert _LIFT_ROW.size == 28
    assert _UNIT_ACTION.size == 6


def test_the_numbering_shared_with_the_agent():
    """Statuses, lift phases and failures, and the movement types of a squad's passage byte, as `agent/World.java`, `agent/Lift.java` and `agent/Passage.java` number them."""
    assert [int(s) for s in (Status.UNREACHABLE, Status.AWAITING_LIFT, Status.LIFTING)] == [5, 6, 7]
    assert [p.name for p in LiftPhase] == ["APPROACH", "LOADING", "CARRYING", "UNLOADING", "DONE", "FAILED"]
    assert [f.name for f in LiftFailure] == ["NONE", "SUNK", "UNREACHABLE_PICKUP", "UNREACHABLE_DROP", "REFUSED",
                                             "EXPIRED", "CANCELLED", "CARGO_LOST"]
    assert PASSAGE_CLASSES == ("", "LAND", "OVER_CLIFF", "HOVER", "WATER", "OVER_CLIFF_WATER", "AIR")
    assert [int(k) for k in TargetKind] == [0, 1, 2] and [int(k) for k in CargoKind] == [0, 1]


def test_header_round_trip():
    frame = encode(Kind.OBSERVATION, 3, b"abcd")
    kind, instance, flags, length = decode_header(frame[:HEADER_SIZE])
    assert kind == Kind.OBSERVATION
    assert instance == 3
    assert flags == 0
    assert length == 4
    assert frame[HEADER_SIZE:] == b"abcd"


def test_the_flags_field_carries_an_observation_number_both_ways():
    frame = encode(Kind.ACTION, 2, b"", FLAGS_MASK)
    kind, instance, flags, length = decode_header(frame[:HEADER_SIZE])
    assert (kind, instance, flags, length) == (Kind.ACTION, 2, 0xFFFF, 0)


def test_action_round_trip():
    action = Action(
        squads=[SquadAssignment(squad=0, commander=Commander.MACHINE, units=[7, 9, 11]),
                SquadAssignment(squad=1, commander=Commander.OPERATIONS, units=[])],
        contracts=[Contract(squad=0, task=Task.ATTACK, stance=Stance.AGGRESSIVE, target=5,
                            cost_budget=2000.5, deadline_ms=123456, issued_at_ms=98765),
                   Contract(squad=1, task=Task.ESCORT, stance=Stance.GUARD_AREA, target_kind=TargetKind.UNIT,
                            target=0x80000001)],
        deviations=[SquadDeviation(squad=0, deviation=Deviation.KITE),
                    SquadDeviation(squad=1, deviation=Deviation.HOLD)],
        production=[Production(producer=42, type_index=8, kind=ProductionKind.BUILDING, x=1.5, y=-2.5)],
        lifts=[Lift(lift=3, transports=[500, 501], cargo_kind=CargoKind.SQUAD, cargo=[2], pickup_x=10.0,
                    pickup_y=20.0, drop_region=7, drop_x=2400.0, drop_y=1800.0, deadline_ms=90000),
               Lift(lift=4, transports=[502], cargo_kind=CargoKind.UNITS, cargo=[61, 62], cancel=True, override=True)],
        unit_actions=[UnitAction(unit=500, action_id="109"), UnitAction(unit=501, action_id="110", append=True)],
    )
    restored = decode_action(encode_action(action))

    assert [(s.squad, s.commander, s.units, s.owner) for s in restored.squads] == [
        (0, Commander.MACHINE, [7, 9, 11], -1), (1, Commander.OPERATIONS, [], -1)]
    contract = restored.contracts[0]
    assert (contract.squad, contract.task, contract.stance) == (0, Task.ATTACK, Stance.AGGRESSIVE)
    assert (contract.target_kind, contract.target) == (TargetKind.REGION, 5)
    # A unit id uses the whole of its 32 bits.
    assert (restored.contracts[1].target_kind, restored.contracts[1].target) == (TargetKind.UNIT, 0x80000001)
    assert [(lift.lift, lift.transports, lift.cargo, lift.cancel, lift.override) for lift in restored.lifts] == [
        (3, [500, 501], [2], False, False), (4, [502], [61, 62], True, True)]
    assert [(a.unit, a.action_id, a.append) for a in restored.unit_actions] == [(500, "109", False), (501, "110", True)]
    # A fraction of a credit survives, so the encoder is not quietly flooring the budget to an int.
    assert (contract.cost_budget, contract.deadline_ms) == (2000.5, 123456)
    assert contract.issued_at_ms == 98765
    assert [(d.squad, d.deviation) for d in restored.deviations] == [
        (0, Deviation.KITE), (1, Deviation.HOLD)]
    item = restored.production[0]
    assert (item.producer, item.type_index, item.kind) == (42, 8, ProductionKind.BUILDING)
    assert (item.x, item.y) == (1.5, -2.5)
    assert restored == action


def test_empty_action_round_trip():
    assert decode_action(encode_action(Action())) == Action()


def test_an_outside_commanders_decisions_are_marked_as_such():
    """The one thing on the wire that separates a decision by whoever holds a squad from a decision by the layer whose job it ordinarily is. Without it the game side, which refuses the layer's decisions about a squad someone has taken over, would refuse the holder's too, and taking a squad over would silence it rather than transfer it."""
    action = Action(
        contracts=[Contract(squad=2, task=Task.DEFEND, target=3, override=True),
                   Contract(squad=3, task=Task.RAID, target=4)],
        deviations=[SquadDeviation(squad=2, deviation=Deviation.SPREAD, override=True),
                    SquadDeviation(squad=3, deviation=Deviation.FOCUS)],
    )
    restored = decode_action(encode_action(action))
    assert [row.override for row in restored.contracts] == [True, False]
    assert [row.override for row in restored.deviations] == [True, False]
    assert restored == action


def test_a_squad_may_be_owned_by_another_player():
    """How one process drives both sides of a constructed engagement. Nought on the wire is this process's own player, so the ordinary case never has to say anything, and a slot travels as itself plus one."""
    action = Action(squads=[SquadAssignment(squad=0, units=[1], owner=-1),
                            SquadAssignment(squad=1, units=[2], owner=0),
                            SquadAssignment(squad=2, units=[3], owner=4)])
    restored = decode_action(encode_action(action))
    assert [row.owner for row in restored.squads] == [-1, 0, 4]


_COUNT = struct.Struct("<H")


def _common(blocks: int) -> bytes:
    return _COMMON.pack(11, 22000, 3, blocks, 1, 0, 1500.0, 42.5, 17, 100, 2, 5, 1, 4, 0)


def _regions() -> bytes:
    """Three real regions parked in slots 0, 5 and 23 of a full 24 slot block. The count is the rows that follow, which is what the agent writes."""
    real = {
        0: (1, 0, 1, 0, 100.0, 200.0, 300.0, 0.0, 0, 0.0),
        5: (1, 2, 0, 1, 900.0, 150.0, 0.0, 640.0, 19000, 850.0),
        23: (1, 1, 0, 0, 500.0, 500.0, 0.0, 0.0, 0, 400.0),
    }
    rows = [_COUNT.pack(REGION_SLOTS)]
    for slot in range(REGION_SLOTS):
        rows.append(_REGION.pack(*real.get(slot, (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0))))
    return b"".join(rows)


def _squads() -> bytes:
    """Two live squads, in slots 1 and 4, so a decoder that walked the count instead of the slots would pick up blanks."""
    real = {
        1: (1, 7, 3, 6, 2, 1, 5, 2100.0, 2400.0, 640.0, 720.0, 55.5, 0, 5, 0, 7, 3000.0, 0.25, 20000, 12000, 300.0,
            5, 3),
        4: (1, 2, 0, 2, 0, 3, 23, 700.0, 700.0, 120.0, 90.0, 12.0, 4, 2, 2, 1, 900.0, 0.1, 8000, 11000, 0.0,
            0x80000001, NO_LIFT),
    }
    blank = (0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0, 0)
    rows = [_COUNT.pack(SQUAD_SLOTS)]
    for slot in range(SQUAD_SLOTS):
        rows.append(_SQUAD.pack(*real.get(slot, blank)))
    return b"".join(rows)


def _units() -> bytes:
    return b"".join([
        _COUNT.pack(2),
        _UNIT.pack(101, 7, 4, 640.0, 720.0, 250.0, 400.0, 202, 1200, 1, 3, 5, 0, 2, 2, 3500, 0, 303),
        _UNIT.pack(202, 0xFFFF, 9, 800.0, 810.0, 90.0, 90.0, 0, 0, 1, 255, 2, 1, 0, 0, 0, 3, 0),
    ])


def _lifts() -> bytes:
    return b"".join([
        _COUNT.pack(2),
        _LIFT.pack(3, int(LiftPhase.CARRYING), 0, 3, 4, 1, 220.0, 41000, 7, 9),
        _LIFT.pack(5, int(LiftPhase.FAILED), int(LiftFailure.SUNK), 0, 2, 1, 0.0, 0, 0xFFFF, 4),
    ])


def _events() -> bytes:
    return b"".join([
        _COUNT.pack(2),
        _EVENT.pack(1, 0, 7, 101, 4, 0, 350.0),
        _EVENT.pack(3, 0, 2, 0, 0, 0, 0.5),
    ])


def test_decode_observation_of_a_hand_packed_body():
    blocks = BLOCK_REGIONS | BLOCK_SQUADS | BLOCK_UNITS | BLOCK_LIFTS | BLOCK_EVENTS
    body = _common(blocks) + _regions() + _squads() + _units() + _lifts() + _events()
    observation = decode_observation(body)

    assert (observation.frame, observation.game_time_ms, observation.episode) == (11, 22000, 3)
    assert (observation.blocks, observation.slot) == (blocks, 1)
    assert (observation.credits, observation.income) == (1500.0, 42.5)
    assert (observation.units, observation.unit_cap, observation.under_construction) == (17, 100, 2)
    assert (observation.killed_units, observation.killed_buildings) == (5, 1)
    assert (observation.lost_units, observation.lost_buildings) == (4, 0)

    # A 24 slot block holding 3 real regions decodes to 3 rows, keeping the slot as the id.
    assert len(observation.regions) == 3
    assert [r.id for r in observation.regions] == [0, 5, 23]
    far = observation.regions[1]
    assert (far.resources, far.held_by_us, far.held_by_enemy) == (2, 0, 1)
    assert (far.x, far.y) == (900.0, 150.0)
    assert (far.our_value, far.enemy_value) == (0.0, 640.0)
    assert (far.enemy_seen_at_ms, far.distance_from_home) == (19000, 850.0)

    assert len(observation.squads) == 2
    assert [s.id for s in observation.squads] == [7, 2]
    squad = observation.squads[0]
    assert (squad.commander, squad.units) == (3, 6)
    assert (squad.value, squad.formed_value) == (2100.0, 2400.0)
    assert (squad.x, squad.y, squad.spread) == (640.0, 720.0, 55.5)
    assert (squad.task_type, squad.stance) == (0, 5)
    assert (squad.target_region, squad.status) == (5, int(Status.LIFTING))
    assert (squad.cost_budget, squad.budget_share) == (3000.0, 0.25)
    assert (squad.deadline_ms, squad.issued_at_ms) == (20000, 12000)
    assert squad.losses == 300.0
    assert (squad.target_kind, squad.target, squad.aboard, squad.passage, squad.lift) == (0, 5, 2, 1, 3)
    escort = observation.squads[1]
    assert (escort.target_kind, escort.target, escort.target_region) == (2, 0x80000001, 23)
    assert (PASSAGE_CLASSES[escort.passage], escort.lift) == ("HOVER", NO_LIFT)

    assert len(observation.unit_states) == 2
    unit = observation.unit_states[0]
    assert (unit.id, unit.squad, unit.type_index) == (101, 7, 4)
    assert (unit.x, unit.y) == (640.0, 720.0)
    assert (unit.health, unit.max_health) == (250.0, 400.0)
    assert (unit.target, unit.since_hit_ms) == (202, 1200)
    assert (unit.built, unit.order, unit.stance, unit.hostile, unit.queued) == (1, 3, 5, 0, 2)
    assert (unit.level, unit.upgrade_price) == (2, 3500)
    assert (unit.aboard, unit.carrier) == (0, 303)
    idle = observation.unit_states[1]
    assert (idle.target, idle.order, idle.hostile) == (0, 255, 1)
    assert (idle.aboard, idle.carrier) == (3, 0)

    carrying, sunk = observation.lifts
    assert (carrying.lift, carrying.phase, carrying.loaded, carrying.expected, carrying.transports) == (3, 2, 3, 4, 1)
    assert (carrying.health, carrying.eta_ms, carrying.squad, carrying.drop_region) == (220.0, 41000, 7, 9)
    assert (sunk.phase, sunk.reason, sunk.squad) == (int(LiftPhase.FAILED), int(LiftFailure.SUNK), 0xFFFF)

    assert len(observation.events) == 2
    completed = observation.events[0]
    assert (completed.kind, completed.squad, completed.unit) == (1, 7, 101)
    assert (completed.type_index, completed.value) == (4, 350.0)
    weak = observation.events[1]
    assert (weak.kind, weak.squad, weak.unit) == (3, 2, 0)
    assert weak.value == 0.5


def test_absent_blocks_leave_empty_lists():
    observation = decode_observation(_common(0))
    assert observation.regions == []
    assert observation.squads == []
    assert observation.unit_states == []
    assert observation.lifts == []
    assert observation.events == []


def test_production_menus_follow_the_events_and_name_types_by_index():
    menus = b"".join([_COUNT.pack(2), struct.pack("<IB3H", 40, 3, 1, 0, 6), struct.pack("<IB", 41, 0)])
    observation = decode_observation(_common(BLOCK_EVENTS | BLOCK_MENUS) + _events() + menus)
    assert [e.kind for e in observation.events] == [1, 3]
    assert observation.menus == {40: [1, 0, 6], 41: []}
    assert decode_observation(_common(0)).menus == {}


def test_the_timing_block_comes_last_and_names_the_answered_observation():
    menus = b"".join([_COUNT.pack(1), struct.pack("<IB1H", 40, 1, 6)])
    observation = decode_observation(_common(BLOCK_EVENTS | BLOCK_MENUS | BLOCK_TIMING) + _events() + menus
                                     + struct.pack("<i", 41))
    assert observation.answered == 41 and observation.menus == {40: [6]}
    assert decode_observation(_common(BLOCK_TIMING) + struct.pack("<i", -1)).answered == -1
    assert decode_observation(_common(BLOCK_EVENTS) + _events()).answered == -1


def test_the_ai_orders_block_follows_the_timing_block_with_each_orders_units():
    orders = b"".join([_COUNT.pack(2),
                       struct.pack("<IBBBxffIH", 61000, 2, int(AiOrderKind.LOAD_INTO), 1, float("nan"), float("nan"), 77, 2),
                       struct.pack("<2I", 10, 11),
                       struct.pack("<IBBBxffIH", 61500, 2, int(AiOrderKind.UNLOAD), 0, float("nan"), float("nan"),
                                   AI_ORDER_NO_UNIT, 0)])
    observation = decode_observation(_common(BLOCK_TIMING | BLOCK_AI_ORDERS) + struct.pack("<i", 5) + orders)
    assert observation.answered == 5
    boarding, unload = observation.ai_orders
    assert (boarding.time_ms, boarding.issuer, boarding.kind, boarding.append, boarding.target, boarding.units) == (
        61000, 2, AiOrderKind.LOAD_INTO, True, 77, (10, 11))
    assert (unload.kind, unload.append, unload.target, unload.units) == (AiOrderKind.UNLOAD, False, AI_ORDER_NO_UNIT, ())
    moved = struct.pack("<IBBBxffIH", 100, 3, int(AiOrderKind.MOVE), 0, 950.0, 120.5, AI_ORDER_NO_UNIT, 1) + struct.pack("<I", 4)
    only, = decode_observation(_common(BLOCK_AI_ORDERS) + _COUNT.pack(1) + moved).ai_orders
    assert (only.x, only.y, only.units) == (950.0, 120.5, (4,))
    assert decode_observation(_common(BLOCK_TIMING) + struct.pack("<i", -1)).ai_orders == []


def test_the_latency_meter_measures_lags_misses_and_the_operational_subset():
    from rwintel.control.latency import Latency

    meter = Latency()
    for number in range(6):
        meter.answered(number - 1 if number else -1, 1000 + 200 * number)
        meter.observed(number, 1000 + 200 * number, operational=number % 2 == 0)
    summary = meter.summary(200)
    assert summary["tactics"] == {"answers": 5, "missed": 0, "mean_ms": 200.0, "p95_ms": 200.0, "max_ms": 200.0,
                                  "mean_periods": 1.0, "p95_periods": 1.0, "max_periods": 1.0}
    assert summary["operations"]["answers"] == 3 and summary["operations"]["missed"] == 0
    assert meter.steady(200) == (True, None)

    meter.reset()
    meter.observed(10, 0, True)
    meter.answered(-1, 200)
    meter.observed(11, 200, False)
    meter.answered(10, 400)
    meter.observed(12, 400, False)
    meter.answered(12, 600)
    meter.observed(13, 600, True)
    summary = meter.summary(200)
    # 11 was never answered; 13 is the episode's last observation and is not counted as missed.
    assert summary["tactics"]["answers"] == 2 and summary["tactics"]["missed"] == 1
    assert summary["tactics"]["max_ms"] == 400.0 and summary["tactics"]["max_periods"] == 2.0
    assert summary["tactics"]["mean_ms"] == 300.0
    assert summary["operations"] == {"answers": 1, "missed": 0, "mean_ms": 400.0, "p95_ms": 400.0, "max_ms": 400.0,
                                     "mean_periods": 2.0, "p95_periods": 2.0, "max_periods": 2.0}
    assert meter.steady(200) == (False, 400)
    assert Latency().summary(200)["tactics"]["answers"] == 0


def test_a_session_feeds_the_latency_meter_from_the_frames_it_answers():
    import socket

    from rwintel.control.session import EpisodeSettings, Session
    from rwintel.wire import read_frame

    ours, theirs = socket.socketpair()
    try:
        session = Session(ours, None, EpisodeSettings(), [("script", lambda s: None)])
        for number, answered in ((3, -1), (4, 3), (5, 4)):
            body = _COMMON.pack(11, 1000 + 200 * number, 3, BLOCK_TIMING, 1, 0, 1500.0, 42.5, 17, 100, 2, 5, 1, 4, 0)
            session.on_observation(body + struct.pack("<i", answered), number)
            read_frame(theirs)
        summary = session.latency.summary(session.tactical_ms)
        assert summary["tactics"]["answers"] == 2 and summary["tactics"]["mean_ms"] == 200.0
    finally:
        ours.close()
        theirs.close()


def test_blocks_are_skipped_by_the_mask_not_by_position():
    """Only the events bit is set, so the decoder has to read the events block straight after the common block."""
    observation = decode_observation(_common(BLOCK_EVENTS) + _events())
    assert observation.regions == []
    assert [e.kind for e in observation.events] == [1, 3]


def test_every_observation_is_answered_once_with_its_number_even_when_there_is_nothing_to_change():
    """A game on the fixed clock waits for the answer to its last observation, so an observation the policy has nothing to say about still gets one."""
    import socket

    from rwintel.control import deciding
    from rwintel.control.session import EpisodeSettings, Session
    from rwintel.wire import read_frame

    class Policy:
        def __init__(self, answers):
            self.answers = list(answers)
            self.inside = []

        def decide(self, observation):
            self.inside.append(deciding.inside())
            return self.answers.pop(0)

    ours, theirs = socket.socketpair()
    try:
        session = Session(ours, None, EpisodeSettings(), [("script", lambda s: None)])
        session.policy = Policy([None, b"act"])
        body = _common(BLOCK_EVENTS) + _events()
        session.on_observation(body, 7)
        session.on_observation(body, 0x10008)
        first, second = read_frame(theirs), read_frame(theirs)
        assert (first.kind, first.flags, first.body) == (Kind.ACTION, 7, b"")
        assert (second.kind, second.flags, second.body) == (Kind.ACTION, 8, b"act")
        assert session.policy.inside == [True, True] and not deciding.inside()
        session.policy = None
        session.on_observation(body, 9)
        assert read_frame(theirs).flags == 9
    finally:
        ours.close()
        theirs.close()


def test_the_terrain_frame_says_what_each_movement_type_can_reach():
    """The grid travels as runs of labels; read back, a position names its component, a position on a blocked tile takes the nearest component in reach, and air reaches everything."""
    import numpy as np

    from rwintel.wire.terrain import TILE, decode_terrain, encode_terrain

    land = np.array([[0, 0, -1, 1, 1],
                     [0, 0, -1, 1, 1],
                     [0, -1, -1, -1, 1]])
    hover = np.zeros_like(land)
    passage = decode_terrain(encode_terrain(5, 3, {"LAND": land, "HOVER": hover}))
    assert passage.width == 5 and passage.height == 3 and passage.components == {"LAND": 2, "HOVER": 1}
    assert np.array_equal(passage.labels["LAND"], land)
    left, right = (0.5 * TILE, 0.5 * TILE), (4.5 * TILE, 0.5 * TILE)
    assert not passage.reachable("LAND", left, right) and passage.reachable("HOVER", left, right)
    assert passage.reachable("AIR", left, right)
    # The blocked middle column belongs to whichever land is nearest; a type with no grid reaches nothing.
    assert passage.component_at("LAND", 2.5 * TILE, 0.5 * TILE) in (0, 1)
    assert passage.component_at("LAND", 2.5 * TILE, 0.5 * TILE, reach=0) == -1
    assert passage.component_at("WATER", *left) == -1
    assert Kind.TERRAIN == 0x03


def test_a_blocked_position_takes_the_component_the_game_side_finds_first():
    """Between two components in the same ring, a blocked position takes the first crossable tile, rows from the top and columns from the left, as `Passage.componentAt` on the game side does; both halves then agree on whether a contract's target can be reached."""
    import numpy as np

    from rwintel.wire.terrain import TILE, decode_terrain, encode_terrain

    land = np.array([[-1, 1, -1],
                     [-1, -1, -1],
                     [0, -1, -1]])
    passage = decode_terrain(encode_terrain(3, 3, {"LAND": land}))
    assert passage.component_at("LAND", 1.5 * TILE, 1.5 * TILE) == 1


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
