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
    Commander,
    Contract,
    Deviation,
    Production,
    ProductionKind,
    SquadAssignment,
    SquadDeviation,
    Stance,
    Task,
    _CONTRACT,
    _DEVIATION,
    _PRODUCTION,
    _SQUAD_HEADER,
    decode_action,
    encode_action,
)
from rwintel.wire.frames import HEADER_SIZE, PROTOCOL_VERSION
from rwintel.wire.observation import (
    BLOCK_EVENTS,
    BLOCK_REGIONS,
    BLOCK_SQUADS,
    BLOCK_UNITS,
    REGION_SLOTS,
    SQUAD_SLOTS,
    _COMMON,
    _EVENT,
    _REGION,
    _SQUAD,
    _UNIT,
    decode_observation,
)


def test_protocol_version_is_two():
    assert PROTOCOL_VERSION == 2


def test_block_sizes_match_the_agent():
    assert HEADER_SIZE == 16
    assert _COMMON.size == 40
    assert _REGION.size == 28
    assert _SQUAD.size == 52
    assert _UNIT.size == 40
    assert _EVENT.size == 16


def test_action_section_sizes_match_the_agent():
    assert _SQUAD_HEADER.size == 8
    assert _CONTRACT.size == 20
    assert _DEVIATION.size == 4
    assert _PRODUCTION.size == 16


def test_header_round_trip():
    frame = encode(Kind.OBSERVATION, 3, b"abcd")
    kind, instance, flags, length = decode_header(frame[:HEADER_SIZE])
    assert kind == Kind.OBSERVATION
    assert instance == 3
    assert flags == 0
    assert length == 4
    assert frame[HEADER_SIZE:] == b"abcd"


def test_action_round_trip():
    action = Action(
        squads=[SquadAssignment(squad=0, commander=Commander.MACHINE, units=[7, 9, 11]),
                SquadAssignment(squad=1, commander=Commander.OPERATIONS, units=[])],
        contracts=[Contract(squad=0, task=Task.ATTACK, stance=Stance.AGGRESSIVE, target_region=5,
                            cost_budget=2000.5, deadline_ms=123456, issued_at_ms=98765)],
        deviations=[SquadDeviation(squad=0, deviation=Deviation.KITE),
                    SquadDeviation(squad=1, deviation=Deviation.HOLD)],
        production=[Production(producer=42, type_index=8, kind=ProductionKind.BUILDING, x=1.5, y=-2.5)],
    )
    restored = decode_action(encode_action(action))

    assert [(s.squad, s.commander, s.units) for s in restored.squads] == [
        (0, Commander.MACHINE, [7, 9, 11]), (1, Commander.OPERATIONS, [])]
    contract = restored.contracts[0]
    assert (contract.squad, contract.task, contract.stance) == (0, Task.ATTACK, Stance.AGGRESSIVE)
    assert contract.target_region == 5
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
        1: (1, 7, 3, 6, 2100.0, 2400.0, 640.0, 720.0, 55.5, 0, 5, 5, 0, 3000.0, 0.25, 20000, 12000, 300.0),
        4: (1, 2, 0, 2, 700.0, 700.0, 120.0, 90.0, 12.0, 1, 2, 23, 1, 900.0, 0.1, 8000, 11000, 0.0),
    }
    blank = (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0.0)
    rows = [_COUNT.pack(SQUAD_SLOTS)]
    for slot in range(SQUAD_SLOTS):
        rows.append(_SQUAD.pack(*real.get(slot, blank)))
    return b"".join(rows)


def _units() -> bytes:
    return b"".join([
        _COUNT.pack(2),
        _UNIT.pack(101, 7, 4, 640.0, 720.0, 250.0, 400.0, 202, 1200, 1, 3, 5, 0, 2),
        _UNIT.pack(202, 0xFFFF, 9, 800.0, 810.0, 90.0, 90.0, 0, 0, 1, 255, 2, 1, 0),
    ])


def _events() -> bytes:
    return b"".join([
        _COUNT.pack(2),
        _EVENT.pack(1, 0, 7, 101, 4, 0, 350.0),
        _EVENT.pack(3, 0, 2, 0, 0, 0, 0.5),
    ])


def test_decode_observation_of_a_hand_packed_body():
    blocks = BLOCK_REGIONS | BLOCK_SQUADS | BLOCK_UNITS | BLOCK_EVENTS
    body = _common(blocks) + _regions() + _squads() + _units() + _events()
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
    assert (squad.target_region, squad.status) == (5, 0)
    assert (squad.cost_budget, squad.budget_share) == (3000.0, 0.25)
    assert (squad.deadline_ms, squad.issued_at_ms) == (20000, 12000)
    assert squad.losses == 300.0

    assert len(observation.unit_states) == 2
    unit = observation.unit_states[0]
    assert (unit.id, unit.squad, unit.type_index) == (101, 7, 4)
    assert (unit.x, unit.y) == (640.0, 720.0)
    assert (unit.health, unit.max_health) == (250.0, 400.0)
    assert (unit.target, unit.since_hit_ms) == (202, 1200)
    assert (unit.built, unit.order, unit.stance, unit.hostile) == (1, 3, 5, 0)
    idle = observation.unit_states[1]
    assert (idle.target, idle.order, idle.hostile) == (0, 255, 1)

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
    assert observation.events == []


def test_blocks_are_skipped_by_the_mask_not_by_position():
    """Only the events bit is set, so the decoder has to read the events block straight after the common block."""
    observation = decode_observation(_common(BLOCK_EVENTS) + _events())
    assert observation.regions == []
    assert [e.kind for e in observation.events] == [1, 3]


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
