"""Holds the two halves of the wire format to the same layout.

The Java side writes these blocks by hand, so a field added on one side and forgotten on the other would not fail to compile; it would decode into the wrong fields and produce a plausible looking observation. Pinning the block sizes here makes that a test failure instead. The sizes are also stated in `docs/project/05-interface.md`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.wire import Kind, decode_header, encode
from rwintel.wire.action import (
    Action,
    Contract,
    Deviation,
    Production,
    ProductionKind,
    SquadAssignment,
    Stance,
    Task,
    decode_action,
    encode_action,
)
from rwintel.wire.frames import HEADER_SIZE
from rwintel.wire.observation import _COMMON, _REGION, _SQUAD, _UNIT


def test_block_sizes_match_the_agent():
    assert HEADER_SIZE == 16
    assert _COMMON.size == 40
    assert _REGION.size == 28
    assert _SQUAD.size == 36
    assert _UNIT.size == 32


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
        squads=[SquadAssignment(squad=0, units=[7, 9, 11]), SquadAssignment(squad=1, units=[])],
        contracts=[Contract(squad=0, task=Task.ATTACK, stance=Stance.AGGRESSIVE, target_region=5,
                            deviation=Deviation.KITE, cost_budget=2000, deadline_ms=123456)],
        production=[Production(producer=42, type_index=8, kind=ProductionKind.BUILDING, x=1.5, y=-2.5)],
    )
    restored = decode_action(encode_action(action))

    assert [(s.squad, s.units) for s in restored.squads] == [(0, [7, 9, 11]), (1, [])]
    contract = restored.contracts[0]
    assert (contract.squad, contract.task, contract.stance) == (0, Task.ATTACK, Stance.AGGRESSIVE)
    assert (contract.target_region, contract.deviation) == (5, Deviation.KITE)
    assert (contract.cost_budget, contract.deadline_ms) == (2000, 123456)
    item = restored.production[0]
    assert (item.producer, item.type_index, item.kind) == (42, 8, ProductionKind.BUILDING)
    assert (item.x, item.y) == (1.5, -2.5)


def test_empty_action_round_trip():
    assert decode_action(encode_action(Action())) == Action()


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
