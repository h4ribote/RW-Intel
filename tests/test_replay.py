"""Holds the replay reader to the engine's format, and the analyses built on it to what the records say.

The replays here are written by a small encoder that follows the engine's writer field by field, so nothing needs the game or a recorded match.
"""

from __future__ import annotations

import gzip
import os
import struct
import sys
from typing import List, Optional, Sequence, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.policy.contracts import DOCTRINES, Doctrine, SquadRecord
from rwintel.control.policy.encoding import MEANS, OPERATIONAL_PLANS, plan_of
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.data import Region
from rwintel.replay.analysis import match_end, render, summarise
from rwintel.replay.commands import NO_UNIT, ORDER_KINDS, STANCES, SYSTEM, Command, Order
from rwintel.replay.container import read_replay
from rwintel.control.policy.logistics import Logistics
from rwintel.replay.commands import CANCEL_UNLOAD_ACTION, UNLOAD_ACTION
from rwintel.replay.human import (AGREEMENT_FLOOR, IDLE, IDLE_WEIGHT, LIFT, LIFT_MASKED, LIFT_UNSLOTTED, ORDERED, STANDING,
                                  HumanOperations, OrderBook, infer_decision, infer_lift, infer_region, infer_task)
from rwintel.replay.playback import Job, check, find_reference
from rwintel.replay.stream import MARK, ReplayFormatError
from rwintel.replay.timeline import Timeline
from rwintel.replay.timing import EXTRA_CHECKSUMS, Clock, step_ms
from rwintel.replay.verify import compare, read_log
from rwintel.wire import BLOCK_REGIONS, BLOCK_SQUADS, BLOCK_UNITS, Observation, RegionState, Status, Task, UnitState


# ---- a writer for the engine's format --------------------------------------------------------

class Out:
    def __init__(self) -> None:
        self.parts: List[bytes] = []

    def raw(self, data: bytes) -> "Out":
        self.parts.append(data)
        return self

    def byte(self, value: int) -> "Out":
        return self.raw(struct.pack(">b", value))

    def boolean(self, value: bool) -> "Out":
        return self.raw(b"\x01" if value else b"\x00")

    def short(self, value: int) -> "Out":
        return self.raw(struct.pack(">h", value))

    def int(self, value: int) -> "Out":
        return self.raw(struct.pack(">i", value))

    def long(self, value: int) -> "Out":
        return self.raw(struct.pack(">q", value))

    def float(self, value: float) -> "Out":
        return self.raw(struct.pack(">f", value))

    def utf(self, text: str) -> "Out":
        encoded = text.encode("utf-8")
        return self.raw(struct.pack(">H", len(encoded)) + encoded)

    def sized(self, data: bytes) -> "Out":
        return self.int(len(data)).raw(data)

    def block(self, name: str, contents: "Out") -> "Out":
        return self.utf(name).sized(contents.bytes())

    def bytes(self) -> bytes:
        return b"".join(self.parts)


def order(kind: str = "move", unit_type: Optional[str] = None, x: float = 100.0, y: float = 200.0,
          target: int = NO_UNIT, action: Optional[str] = None) -> Out:
    out = Out().int(ORDER_KINDS.index(kind))
    if unit_type is None:
        out.int(-1)
    else:
        out.int(-2).utf(unit_type)
    out.float(x).float(y).long(target).byte(0).float(-1.0).float(-1.0).boolean(False).boolean(False).boolean(False)
    out.boolean(action is not None)
    if action is not None:
        out.utf(action)
    return out


def command(player: int = 1, units: Sequence[int] = (), waypoint: Optional[Out] = None, action: str = "-1",
            stance: int = -1, cancel: bool = False, step_rate: Optional[float] = None, moves: int = 0,
            path: bool = False) -> Out:
    out = Out().byte(player).boolean(waypoint is not None)
    if waypoint is not None:
        out.raw(waypoint.bytes())
    out.boolean(False).boolean(cancel).int(-1).int(stance).boolean(False).boolean(False)
    out.int(len(units))
    for unit in units:
        out.long(unit)
    out.boolean(False)                      # no acting player
    out.boolean(False).long(NO_UNIT)        # no second point, no second unit
    out.utf(action).boolean(False).short(0)
    out.boolean(step_rate is not None)
    if step_rate is not None:
        out.byte(0).float(step_rate).float(0.0).int(0)
    out.int(moves)
    for _ in range(moves):
        out.long(7).float(1.0).float(2.0).float(3.0).float(4.0).int(1).int(0)
        out.boolean(path)
        if path:
            out.boolean(True).block("p", Out().raw(gzip.compress(b"\x00\x02\x00\x01\x00\x01")))
    out.boolean(False)
    return out


def save(map_path: str = "maps/skirmish/[p2]Lake (2p).tmx") -> Out:
    body = gzip.compress(b"customUnitsBlock\x00" + map_path.encode("ascii") + b"\x00<rest of the save>")
    return (Out().utf("rustedWarfareSave").int(176).int(96).boolean(False)
            .block("saveCompression", Out().raw(body)).short(MARK).utf("<SAVE END>"))


def replay(records: Sequence[Tuple[str, Out]], version: int = 96, closed: bool = False) -> bytes:
    out = Out().utf("rustedWarfareReplay").int(176).int(version).utf("1.15").boolean(False)
    out.block("gamesave", save())
    for tag, contents in records:
        out.block(tag, contents)
    if closed:
        out.block("end", Out())
    return out.bytes()


def rc(frame: int, body: Out) -> Tuple[str, Out]:
    return "rc", Out().int(frame).block("c", body)


def es(frame: int, values: Sequence[int]) -> Tuple[str, Out]:
    blob = Out().byte(0).int(len(values))
    for value in values:
        blob.long(value)
    return "es", Out().int(frame).sized(blob.bytes())


def chat(frame: int, message: str, slot: int = SYSTEM) -> Tuple[str, Out]:
    return "chat", Out().int(frame).int(slot).boolean(False).boolean(True).utf(message)


def credits_record(frame: int, slot0: int, slot1: int) -> Tuple[str, Out]:
    values = [0] * len(EXTRA_CHECKSUMS)
    values[EXTRA_CHECKSUMS.index("slot0_credits")] = slot0
    values[EXTRA_CHECKSUMS.index("slot1_credits")] = slot1
    return es(frame, values)


# ---- the container ---------------------------------------------------------------------------

def test_a_replay_is_read_back_record_by_record():
    data = replay([
        chat(0, "<All players ready>"),
        ("wait", Out().int(0)),
        rc(20, command(player=0, units=[11], waypoint=order("build", unit_type="extractorT1", x=1170, y=2050),
                       action="b_extractorT1")),
        rc(170, command(player=SYSTEM, step_rate=2.0)),
        ("cs", Out().int(190).long(12325041)),
        credits_record(300, 4000, 3500),
    ], closed=True)
    read = read_replay(data)
    assert read.header.save_version == 96 and read.header.version_name == "1.15"
    assert read.map_path == "maps/skirmish/[p2]Lake (2p).tmx"
    assert read.blocks == {"chat": 1, "wait": 1, "rc": 2, "cs": 1, "es": 1, "end": 1}
    assert read.unknown == {}
    assert read.ended
    assert read.checksums == [(190, 12325041)]
    assert read.waits == [0]
    assert read.players() == [0]
    placed = read.commands[0]
    assert placed.frame == 20 and placed.player == 0 and placed.units == (11,)
    assert placed.order.kind == "build" and placed.order.unit_type == "extractorT1"
    assert placed.places == "extractorT1" and placed.action == "b_extractorT1"
    assert read.commands[1].step_rate == 2.0 and read.commands[1].player == SYSTEM


def test_a_block_of_an_unknown_name_is_counted_and_skipped():
    read = read_replay(replay([("novel", Out().int(5).utf("anything")), rc(1, command(units=[3]))]))
    assert read.unknown == {"novel": 1}
    assert [c.units for c in read.commands] == [(3,)]


def test_a_command_that_leaves_bytes_unread_is_refused():
    body = command(units=[1]).raw(b"\x00")
    with pytest.raises(ReplayFormatError):
        read_replay(replay([rc(5, body)]))


def test_a_file_that_is_not_a_replay_is_refused():
    with pytest.raises(ReplayFormatError):
        read_replay(Out().utf("rustedWarfareSave").bytes())


def test_movement_records_and_their_compressed_paths_are_skipped_to_the_right_place():
    read = read_replay(replay([rc(40, command(units=[1, 2], waypoint=order("attackMove"), moves=2, path=True)),
                               rc(41, command(units=[5], moves=1, path=False))]))
    assert [c.moves for c in read.commands] == [2, 1]
    assert read.commands[0].order_kind == "attackMove" and read.commands[1].units == (5,)


def test_fields_added_after_the_recording_version_are_not_read():
    """At version 30 a command stops after the second unit reference; nothing later in the engine's reader exists yet."""
    body = (Out().byte(1).boolean(False).boolean(False).boolean(False).int(-1).int(-1).boolean(False).boolean(False)
            .int(1).long(9).boolean(False).boolean(False).long(NO_UNIT))
    read = read_replay(replay([rc(3, body)], version=30))
    assert read.commands[0].units == (9,) and read.commands[0].action is None


def test_stance_and_cancel_are_read_where_the_engine_puts_them():
    read = read_replay(replay([rc(8, command(units=[4], action="u_c_tank", stance=STANCES.index("holdFire"),
                                             cancel=True))]))
    decoded = read.commands[0]
    assert STANCES[decoded.stance] == "holdFire" and decoded.cancel and decoded.produces == "c_tank"


def test_a_movement_order_names_where_it_sends_units_only_when_it_names_a_place():
    read = read_replay(replay([rc(1, command(units=[1], waypoint=order("move", x=10, y=20))),
                               rc(2, command(units=[1], waypoint=order("attack", target=77)))]))
    assert read.commands[0].destination == (10.0, 20.0)
    assert read.commands[1].destination is None and read.commands[1].order.target == 77


# ---- the clock -------------------------------------------------------------------------------

def test_a_step_is_the_rate_in_sixtieths_truncated_to_whole_milliseconds():
    assert step_ms(1.0) == 16 and step_ms(2.0) == 33


def test_a_change_of_rate_governs_the_steps_after_the_one_following_its_frame():
    clock = Clock.of([(170, 2.0)])
    assert clock.time_ms(171) == 171 * 16
    assert clock.time_ms(172) == 171 * 16 + 33
    assert clock.time_ms(26588) == 874497


@pytest.mark.parametrize("changes", [(), ((170, 2.0),), ((50, 2.0), (400, 1.0))])
def test_frame_at_is_the_first_frame_to_reach_a_time(changes):
    clock = Clock.of(changes)
    for time_ms in (0, 1, 15, 16, 17, 2736, 2737, 10000, 60000):
        frame = clock.frame_at(time_ms)
        assert clock.time_ms(frame) >= time_ms
        assert frame == 0 or clock.time_ms(frame - 1) < time_ms


def test_the_clock_of_a_replay_comes_from_its_system_commands():
    read = read_replay(replay([rc(170, command(player=SYSTEM, step_rate=2.0)), rc(300, command(units=[1]))]))
    assert read.clock.time_ms(300) == Clock.of([(170, 2.0)]).time_ms(300)


# ---- the summary -----------------------------------------------------------------------------

def _match() -> bytes:
    return replay([
        chat(0, "<All players ready>"),
        rc(10, command(player=0, units=[1], action="u_builder")),
        rc(20, command(player=1, units=[2], waypoint=order("build", unit_type="extractorT1"))),
        rc(30, command(player=1, units=[2, 3], waypoint=order("move"))),
        rc(40, command(player=1, units=[4], action="u_c_tank")),
        rc(41, command(player=1, units=[4], action="u_c_tank")),
        rc(42, command(player=1, units=[4], action="u_c_tank", cancel=True)),
        rc(50, command(player=1, units=[5], action="extractorT2_0")),
        credits_record(300, 4000, 3500),
        chat(400, "host-0 was defeated (Team: A)"),
        rc(500, command(player=1, units=[4], action="u_c_tank")),
        credits_record(600, 100, 9000),
    ])


def test_the_match_ends_at_the_first_recorded_defeat_and_nothing_after_it_counts():
    summary = summarise(read_replay(_match()), {"c_tank": 350})
    assert (summary.end_frame, summary.end_reason, summary.defeated) == (400, "defeat", ["host-0"])
    human = summary.players[1]
    assert human.commands == 6
    assert [p.type for p in human.placements] == ["extractorT1", "c_tank", "c_tank"]
    assert human.cancels == 1
    assert human.other_actions == {"extractorT2_0": 1}
    assert human.requested_value() == 700
    assert [p.type for p in human.first_seen()] == ["extractorT1", "c_tank"]
    assert human.credits == [(Clock().time_ms(300) / 1000.0, 3500)]
    assert "match ends" in render(summary)


def test_without_a_recorded_defeat_the_match_ends_at_the_last_record():
    frame, reason, defeated = match_end(read_replay(replay([rc(10, command(units=[1])), ("wait", Out().int(90))])))
    assert (frame, reason, defeated) == (90, "recording", [])


# ---- checking a decode against the engine's log -----------------------------------------------

_LOG = """2026-09-30 06:14:07.460: Replay: updateGameFrame: Command: host-0 (0) count:1 id:1
2026-09-30 06:14:07.461: Replay: updateGameFrame: Waypoint: build
2026-09-30 06:14:07.461: Replay: updateGameFrame: Build Type: extractorT1
2026-09-30 06:14:07.461: Replay: updateGameFrame: SpecialAction: b_extractorT1
2026-09-30 06:14:07.461: Replay: updateGameFrame: ------
2026-09-30 06:14:07.473: Replay: updateGameFrame: Command: h4ribote (1) count:2 id:2
2026-09-30 06:14:07.473: Replay: updateGameFrame: Waypoint: move
2026-09-30 06:14:07.473: Replay: updateGameFrame: SetAttackMode: holdFire
2026-09-30 06:14:07.473: Replay: updateGameFrame: ------
""".splitlines()


def test_a_decode_that_reads_what_the_engine_ran_agrees_with_its_log():
    read = read_replay(replay([rc(20, command(player=0, units=[1], waypoint=order("build", unit_type="extractorT1"),
                                              action="b_extractorT1")),
                               rc(30, command(player=1, units=[2, 3], waypoint=order("move"),
                                              stance=STANCES.index("holdFire")))]))
    logged = read_log(_LOG)
    assert [(entry.slot, entry.kind, entry.count) for entry in logged] == [(0, "build", 1), (1, "move", 2)]
    assert compare(read.commands, logged).agrees


def test_a_decode_that_differs_from_the_engine_in_any_field_is_reported():
    read = read_replay(replay([rc(20, command(player=0, units=[1], waypoint=order("build", unit_type="extractorT1"),
                                              action="b_extractorT1")),
                               rc(30, command(player=1, units=[2], waypoint=order("attackMove")))]))
    comparison = compare(read.commands, read_log(_LOG))
    assert not comparison.agrees
    assert any("order decoded 'attackMove', engine 'move'" in d for d in comparison.differences)
    assert any("stance" in d for d in comparison.differences)
    assert any("moved 2 unit(s)" in d for d in comparison.differences)


# ---- the board a playback is read from --------------------------------------------------------

_TYPES = [
    UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", range=130.0, hits_air=False, hits_land=True),
    UnitType(index=1, name="extractor", lookup="extractor", price=800, tech=1, building=True, builder=False,
             movement="BUILDING", extractor=True),
    UnitType(index=2, name="hovercraft", lookup="hovercraft", price=600, tech=1, building=False, builder=False,
             movement="HOVER", speed=1.0, capacity=4, carries=(0, 3)),
    UnitType(index=3, name="builder", lookup="builder", price=200, tech=1, building=False, builder=True,
             movement="LAND", speed=1.0),
]
_TRANSPORT, _BUILDER = 2, 3


def _Catalogue(types):
    return Catalogue.of_types(types)


def _unit(unit_id, x, y, type_index=0, hostile=0):
    return UnitState(id=unit_id, squad=0xFFFF, type_index=type_index, x=x, y=y, health=100.0, max_health=100.0,
                     built=255, order=255, queued=0, target=0, stance=5, hostile=hostile, since_hit_ms=9999)


def _region(region_id, x, y, ours=0.0, theirs=0.0, held=0, distance=0.0):
    return RegionState(id=region_id, resources=2, held_by_us=held, held_by_enemy=0, x=x, y=y, our_value=ours,
                       enemy_value=theirs, enemy_seen_at_ms=0, distance_from_home=distance)


def _board(units, frame=1000):
    regions = [_region(0, 0, 0, ours=1150, held=1, distance=0.0), _region(1, 1000, 0, theirs=700, distance=1000.0),
               _region(2, 0, 1000, distance=1000.0)]
    observation = Observation(frame=frame, game_time_ms=30000, episode=1,
                              blocks=BLOCK_REGIONS | BLOCK_SQUADS | BLOCK_UNITS, slot=1, credits=1000.0,
                              income=20.0, units=len(units), unit_cap=100, under_construction=0, killed_units=0,
                              killed_buildings=0, lost_units=0, lost_buildings=0, regions=regions,
                              unit_states=list(units))
    return build_view(observation, _Catalogue(_TYPES), 0)


def _moved(frame, units, x, y, player=1, kind="move", target=NO_UNIT):
    return Command(frame=frame, player=player, order=Order(kind=kind, unit_type=None, x=x, y=y, target=target),
                   action=None, stance=None, units=tuple(units))


def _squad(members, doctrine=Doctrine.VANGUARD, x=0.0, y=0.0, status=Status.ACTIVE):
    return SquadRecord(id=0, doctrine=doctrine, members=list(members), value=350.0 * len(members),
                       formed_value=350.0 * len(members), x=x, y=y, status=status)


# ---- reading a person's decisions ---------------------------------------------------------------

def _book(commands, player):
    return OrderBook.from_commands(commands, player)


def test_the_last_order_given_by_a_frame_is_the_one_a_unit_is_carrying_out():
    orders = _book([_moved(100, [1], 10, 10), _moved(200, [1, 2], 20, 20), _moved(300, [1], 30, 30, player=0),
                          _moved(250, [2], 0, 0, kind="build")], player=1)
    assert orders.last(1, 99) is None
    assert (orders.last(1, 150).x, orders.last(1, 250).x, orders.last(1, 400).x) == (10.0, 20.0, 20.0)
    assert orders.last(2, 1000).x == 20.0


def test_a_squad_goes_where_most_of_its_worth_was_sent():
    view = _board([_unit(1, 0, 0), _unit(2, 0, 0), _unit(3, 0, 0)])
    orders = _book([_moved(1200, [1, 2], 990, 10), _moved(1300, [3], 10, 990)], player=1)
    vote = infer_region(view, _squad([1, 2, 3]), orders, start=1000, end=2000)
    assert vote.region.id == 1 and vote.basis == ORDERED and abs(vote.agreement - 2.0 / 3.0) < 1e-9


def test_orders_given_before_the_decision_still_say_where_units_are_bound():
    view = _board([_unit(1, 0, 0), _unit(2, 0, 0)])
    vote = infer_region(view, _squad([1, 2]), _book([_moved(500, [1, 2], 990, 10)], player=1), 1000, 2000)
    assert (vote.region.id, vote.basis, vote.agreement) == (1, STANDING, 1.0)


def test_units_never_sent_anywhere_vote_for_where_they_stand():
    view = _board([_unit(1, 5, 995), _unit(2, 5, 995)])
    vote = infer_region(view, _squad([1, 2]), _book([], player=1), 1000, 2000)
    assert (vote.region.id, vote.basis) == (2, IDLE)


def test_an_order_to_attack_a_unit_goes_where_that_unit_is():
    view = _board([_unit(1, 0, 0), _unit(9, 995, 5, hostile=1)])
    vote = infer_region(view, _squad([1]), _book([_moved(1100, [1], 0, 0, kind="attack", target=9)], player=1),
                        1000, 2000)
    assert vote.region.id == 1


def test_the_task_follows_the_ground_and_stays_inside_the_doctrine():
    view = _board([_unit(1, 0, 0)])
    home, contested, empty = view.region(0), view.region(1), view.region(2)
    assert infer_task(Doctrine.GARRISON, home, _squad([1], Doctrine.GARRISON), view) == Task.DEFEND
    assert infer_task(Doctrine.VANGUARD, home, _squad([1]), view) == Task.ATTACK
    assert infer_task(Doctrine.VANGUARD, contested, _squad([1]), view) == Task.ATTACK
    assert infer_task(Doctrine.RAID, empty, _squad([1], Doctrine.RAID), view) == Task.RAID
    losing = _squad([1], Doctrine.RAID, x=1000.0, y=0.0, status=Status.LOSING)
    assert infer_task(Doctrine.RAID, home, losing, view) == Task.WITHDRAW
    assert infer_task(Doctrine.ENGINEER, home, _squad([1], Doctrine.ENGINEER), view) is None
    for doctrine in (Doctrine.VANGUARD, Doctrine.GARRISON, Doctrine.RAID):
        for region in (home, contested, empty):
            assert infer_task(doctrine, region, _squad([1], doctrine), view) in DOCTRINES[doctrine].tasks


def test_a_person_decision_is_recorded_with_how_sure_it_is_and_refused_when_the_orders_disagree():
    view = _board([_unit(1, 0, 0), _unit(2, 0, 0), _unit(3, 0, 0), _unit(4, 0, 0)])
    regions = [1.0] * 3 + [0.0] * 21
    tasks = [[1.0] * OPERATIONAL_PLANS for _ in regions]
    agreed = HumanOperations(_book([_moved(1100, [1, 2, 3, 4], 990, 0)], player=1), Clock(), 30000, weight=0.5)
    choice = agreed.choose_on_board(view, _squad([1, 2, 3, 4]), [], regions, tasks, 30000)
    # Four tanks against half their worth are the odds the script vanguard surrounds at, and they walk there.
    assert (choice.action, choice.second) == (1, plan_of(Task.ENCIRCLE, -1))
    assert choice.meta == {"source": "human", "weight": 0.5, "agreement": 1.0, "basis": ORDERED}
    split = HumanOperations(_book([_moved(1100, [1], 990, 0), _moved(1100, [2], 0, 990),
                                         _moved(1100, [3], 0, 0)], player=1), Clock(), 30000)
    assert 1.0 / 3.0 < AGREEMENT_FLOOR
    assert split.choose_on_board(view, _squad([1, 2, 3]), [], regions, tasks, 30000) is None
    assert split.counts["refused"] == 1
    idle = HumanOperations(_book([], player=1), Clock(), 30000)
    assert idle.choose_on_board(view, _squad([1]), [], regions, tasks, 30000).meta["weight"] == IDLE_WEIGHT
    masked = HumanOperations(_book([_moved(1100, [1], 990, 0)], player=1), Clock(), 30000)
    assert masked.choose_on_board(view, _squad([1]), [], [1.0, 0.0, 1.0] + [0.0] * 21, tasks, 30000) is None
    # Ground the squad cannot walk to is not taken from a move order alone.
    carried = [list(row) for row in tasks]
    carried[1] = [0.0 if plan % MEANS == 0 else 1.0 for plan in range(OPERATIONAL_PLANS)]
    across = HumanOperations(_book([_moved(1100, [1], 990, 0)], player=1), Clock(), 30000)
    assert across.choose_on_board(view, _squad([1]), [], regions, carried, 30000) is None


# ---- reading a lift --------------------------------------------------------------------------

def _special(frame, units, action, player=1):
    return Command(frame=frame, player=player, order=None, action=action, stance=None, units=tuple(units))


def _lift_board(transports, extra=()):
    """Two tanks at home, the given transports, and a Logistics updated over them."""
    units = [_unit(1, 0, 0), _unit(2, 0, 0)] + [_unit(t, 10, 10, type_index=_TRANSPORT) for t in transports] + list(extra)
    view = _board(units)
    logistics = Logistics(_Catalogue(_TYPES))
    logistics.update(view)
    return view, logistics


_OPEN_REGIONS = [1.0] * 3 + [0.0] * 21
_OPEN_PLANS = [[1.0] * OPERATIONAL_PLANS for _ in _OPEN_REGIONS]


def _carried_to_region_two(transport, loading):
    return [loading, _moved(1300, [transport], 5, 990), _special(1400, [transport], UNLOAD_ACTION)]


def test_a_squad_loaded_moved_and_unloaded_is_a_lift_by_that_transports_slot():
    view, logistics = _lift_board([50])
    into = _book(_carried_to_region_two(50, _moved(1100, [1, 2], 0, 0, kind="loadInto", target=50)), player=1)
    decision = infer_decision(view, _squad([1, 2]), into, 1000, 2000, _OPEN_REGIONS, _OPEN_PLANS, logistics)
    task = infer_task(Doctrine.VANGUARD, view.region(2), _squad([1, 2]), view)
    assert (decision.region, decision.plan, decision.basis, decision.agreement) == (2, plan_of(task, 0), LIFT, 1.0)
    up = [_moved(1100, [50], 0, 0, kind="loadUp", target=1), _moved(1150, [50], 0, 0, kind="loadUp", target=2)]
    taken = _book(up + _carried_to_region_two(50, up[0])[1:], player=1)
    assert infer_decision(view, _squad([1, 2]), taken, 1000, 2000, _OPEN_REGIONS, _OPEN_PLANS, logistics) == decision


def test_the_slot_is_the_one_the_lift_layer_holds_the_transport_in():
    view, logistics = _lift_board([50, 51, 52])
    book = _book(_carried_to_region_two(52, _moved(1100, [1, 2], 0, 0, kind="loadInto", target=52)), player=1)
    assert infer_lift(view, _squad([1, 2]), book, 1000, 2000, logistics).slot == 2
    # A transport keeps its slot when one with a lower id appears later.
    sticky = Logistics(_Catalogue(_TYPES))
    sticky.update(_lift_board([60])[0])
    later = _board([_unit(1, 0, 0), _unit(2, 0, 0), _unit(60, 10, 10, type_index=_TRANSPORT),
                    _unit(55, 10, 10, type_index=_TRANSPORT)])
    sticky.update(later)
    book = _book(_carried_to_region_two(60, _moved(1100, [1, 2], 0, 0, kind="loadInto", target=60)), player=1)
    assert infer_lift(later, _squad([1, 2]), book, 1000, 2000, sticky).slot == 0


def test_a_lift_is_refused_by_a_transport_in_no_slot_or_onto_a_closed_plan():
    view, _ = _lift_board([50])
    book = _book(_carried_to_region_two(50, _moved(1100, [1, 2], 0, 0, kind="loadInto", target=50)), player=1)
    empty = Logistics(_Catalogue(_TYPES))
    decision = infer_decision(view, _squad([1, 2]), book, 1000, 2000, _OPEN_REGIONS, _OPEN_PLANS, empty)
    assert decision.refused and decision.basis == LIFT_UNSLOTTED
    view, logistics = _lift_board([50])
    closed = [list(row) for row in _OPEN_PLANS]
    closed[2] = [0.0 if plan % MEANS == 1 else 1.0 for plan in range(OPERATIONAL_PLANS)]
    decision = infer_decision(view, _squad([1, 2]), book, 1000, 2000, _OPEN_REGIONS, closed, logistics)
    assert decision.refused and decision.basis == LIFT_MASKED
    person = HumanOperations(book, Clock(), 30000)
    assert person.choose_on_board(view, _squad([1, 2]), [], _OPEN_REGIONS, closed, 30000, logistics=logistics) is None
    assert person.counts[LIFT_MASKED] == 1


def test_a_cancelled_unload_leaves_the_drop_at_the_later_one():
    view, logistics = _lift_board([50])
    book = _book([_moved(1100, [1, 2], 0, 0, kind="loadInto", target=50), _moved(1200, [50], 990, 5),
                  _special(1250, [50], UNLOAD_ACTION), _special(1260, [50], CANCEL_UNLOAD_ACTION),
                  _moved(1300, [50], 5, 990), _special(1400, [50], UNLOAD_ACTION)], player=1)
    assert infer_lift(view, _squad([1, 2]), book, 1000, 2000, logistics).region.id == 2


def test_without_an_unload_or_a_move_the_squad_walks():
    view, logistics = _lift_board([50])
    book = _book([_moved(1100, [1, 2], 0, 0, kind="loadInto", target=50)], player=1)
    assert infer_lift(view, _squad([1, 2]), book, 1000, 2000, logistics) is None
    decision = infer_decision(view, _squad([1, 2]), book, 1000, 2000, _OPEN_REGIONS, _OPEN_PLANS, logistics)
    assert decision.basis == IDLE and decision.plan % MEANS == 0
    # Moved but never unloaded, the transport's last destination after the load is the drop.
    moved = _book([_moved(1100, [1, 2], 0, 0, kind="loadInto", target=50), _moved(1300, [50], 990, 5)], player=1)
    assert infer_lift(view, _squad([1, 2]), moved, 1000, 2000, logistics).region.id == 1


def test_a_builder_aboard_the_same_transport_is_not_part_of_the_squad():
    view, logistics = _lift_board([50], extra=[_unit(70, 0, 0, type_index=_BUILDER)])
    book = _book([_moved(1100, [70], 0, 0, kind="loadInto", target=50), _moved(1300, [50], 5, 990),
                  _special(1400, [50], UNLOAD_ACTION)], player=1)
    assert infer_lift(view, _squad([1, 2]), book, 1000, 2000, logistics) is None
    # Half the squad aboard is enough agreement; the builder neither adds to it nor dilutes it.
    book = _book([_moved(1100, [1, 70], 0, 0, kind="loadInto", target=50), _moved(1300, [50], 5, 990),
                  _special(1400, [50], UNLOAD_ACTION)], player=1)
    assert infer_lift(view, _squad([1, 2]), book, 1000, 2000, logistics).agreement == 0.5


def _ai_orders(slot, orders):
    from types import SimpleNamespace

    return SimpleNamespace(observation=SimpleNamespace(slot=slot, ai_orders=orders))


def test_the_builtin_teacher_relabels_a_decision_from_the_orders_that_followed_it():
    from rwintel.learn.builtin import BuiltinOperations
    from rwintel.learn.rollout import Step
    from rwintel.wire import AI_ORDER_NO_UNIT, AiOrder, AiOrderKind

    def ai(time_ms, kind, units, x=float("nan"), y=float("nan"), target=AI_ORDER_NO_UNIT, issuer=1):
        return AiOrder(time_ms=time_ms, issuer=issuer, kind=kind, append=False, x=x, y=y, target=target, units=tuple(units))

    view, logistics = _lift_board([50])
    teacher = BuiltinOperations(review_ms=30000)
    choice = teacher.choose_on_board(view, _squad([1, 2]), [], _OPEN_REGIONS, _OPEN_PLANS, 30000, logistics=logistics)
    # Nothing ordered before the decision: the provisional answer is where the members stand.
    assert (choice.action, choice.meta["basis"]) == (0, IDLE)
    teacher.observe(_ai_orders(1, [ai(31000, AiOrderKind.LOAD_INTO, [1, 2], target=50),
                                   ai(32000, AiOrderKind.MOVE, [1, 2], 990, 5, issuer=2),
                                   ai(33000, AiOrderKind.MOVE, [50], 5, 990),
                                   ai(35000, AiOrderKind.UNLOAD, [50])]))
    step = Step(state=[], action=choice.action, mask=list(_OPEN_REGIONS), second=choice.second, squad=0, at_ms=30000,
                meta=dict(choice.meta), label=choice.action, second_label=choice.second)
    assert teacher.revise(step, 70000)
    expected = infer_decision(view, _squad([1, 2]), teacher.book, 30000, 60000, _OPEN_REGIONS, _OPEN_PLANS, logistics)
    assert (step.action, step.second, step.label, step.second_label) == (2, expected.plan, 2, expected.plan)
    assert step.meta["basis"] == LIFT and step.meta["source"] == "builtin" and teacher.counts[LIFT] == 1
    # An order from the other AI player was not filed.
    assert teacher.book.last(1, 99999) is None
    assert not teacher.revise(step, 70000)
    # A decision the later orders say nothing recordable about keeps its step with both labels at -1.
    split = BuiltinOperations(review_ms=30000)
    choice = split.choose_on_board(view, _squad([1, 2]), [], _OPEN_REGIONS, _OPEN_PLANS, 30000, logistics=logistics)
    split.observe(_ai_orders(1, [ai(31000, AiOrderKind.LOAD_INTO, [1, 2], target=99), ai(33000, AiOrderKind.MOVE, [99], 5, 990),
                                 ai(35000, AiOrderKind.UNLOAD, [99])]))
    step = Step(state=[], action=choice.action, mask=list(_OPEN_REGIONS), second=choice.second, squad=0, at_ms=30000,
                meta=dict(choice.meta), label=choice.action, second_label=choice.second)
    assert split.revise(step, 70000)
    assert (step.label, step.second_label, step.meta["basis"], step.action) == (-1, -1, "refused", choice.action)
    assert step.meta["reason"] == LIFT_UNSLOTTED and split.counts[LIFT_UNSLOTTED] == 1


def test_collect_takes_the_builtin_teacher_only_for_the_operational_layer_and_alone():
    from rwintel.learn.__main__ import _collect_checks, build_parser

    def checked(*extra):
        return _collect_checks(build_parser().parse_args(["collect", "--teacher", "builtin", *extra]))

    assert checked("--layer", "operations") is None
    for wrong in (("--layer", "tactics"), ("--layer", "operations", "--rule", "random"),
                  ("--layer", "operations", "--student", "x.pt"), ("--layer", "operations", "--explore", "0.1")):
        with pytest.raises(SystemExit):
            checked(*wrong)
    with pytest.raises(SystemExit):
        _collect_checks(build_parser().parse_args(["collect", "--layer", "operations", "--teacher", "person"]))


def test_an_unload_is_the_special_action_over_the_transports():
    read = read_replay(replay([rc(5, command(units=[50], action=UNLOAD_ACTION)),
                               rc(6, command(units=[50], action=CANCEL_UNLOAD_ACTION))]))
    unload, cancel = read.commands
    assert unload.unloads and not unload.cancels_unload and unload.units == (50,)
    assert cancel.cancels_unload and not cancel.unloads


# ---- what a playback records ------------------------------------------------------------------

def _observed(units, frame):
    return Observation(frame=frame, game_time_ms=frame * 33, episode=1, blocks=BLOCK_UNITS, slot=1, credits=0.0,
                       income=0.0, units=len(units), unit_cap=100, under_construction=0, killed_units=0,
                       killed_buildings=0, lost_units=0, lost_buildings=0, regions=[], unit_states=list(units))


def test_what_leaves_the_board_is_charged_to_the_region_it_was_last_seen_in():
    regions = [Region(id=0, x=0, y=0, radius=100, resources=2, spawn=True),
               Region(id=1, x=1000, y=0, radius=100, resources=2, spawn=False)]
    timeline = Timeline(regions, [350.0, 800.0], omniscient=True)
    timeline.observe(_observed([_unit(1, 10, 0), _unit(2, 990, 0, type_index=1), _unit(9, 980, 0, hostile=1)], 10))
    timeline.observe(_observed([_unit(1, 10, 0)], 20))
    exchange = {e.region: e for e in timeline.exchange()}
    assert (exchange[1].ours, exchange[1].ours_units, exchange[1].theirs, exchange[1].theirs_units) == (800.0, 1, 350.0, 1)
    assert 0 not in exchange
    fogged = Timeline(regions, [350.0, 800.0], omniscient=False)
    fogged.observe(_observed([_unit(9, 980, 0, hostile=1)], 10))
    fogged.observe(_observed([], 20))
    assert fogged.exchange() == []


def test_progress_is_kept_by_period_and_each_checksum_once():
    timeline = Timeline([], [])
    # Team 3 is a slot nobody plays from: nought on everything but the starting credits every slot is handed.
    standing = [{"team": 0, "units": 2, "value": 3500, "income": 18, "killed": 0, "lost": 0, "credits": 4000},
                {"team": 3, "units": 0, "value": 0, "income": 0, "killed": 0, "lost": 0, "credits": 4000}]
    for time_ms, frame in ((0, 0), (200, 6), (400, 12)):
        timeline.progress({"timeMs": time_ms, "frame": frame, "standing": standing, "checksumFrame": 0, "checksum": 5})
    timeline.progress({"checksumFrame": 300, "checksum": 7}, row=False)
    assert len(timeline.rows) == 3 and timeline.checksums == [(0, 5), (300, 7)]
    summary = timeline.summary()
    assert summary["curves"]["0"][0] == [0.0, 2, 3500, 18, 0, 0, 4000]
    assert list(summary["curves"]) == ["0"]


def _job(reference):
    return Job(path="match.replay", replay=read_replay(replay([rc(1, command(units=[1]))])), until_ms=875000,
               reference=reference)


def test_a_playback_agrees_with_its_record_when_the_checksum_and_a_standing_after_the_end_second_match():
    standing = [{"team": 0, "units": 6, "value": 17100, "income": 64, "killed": 1, "lost": 89}]
    reference = {"seconds": 874, "synchronisation": {"frame": 26488, "checksum": 199640548}, "standing": standing}
    timeline = Timeline([], [])
    earlier = [{"team": 0, "units": 7, "value": 17500, "income": 64, "killed": 1, "lost": 88}]
    timeline.progress({"timeMs": 873900, "frame": 1, "standing": standing})
    timeline.progress({"timeMs": 874200, "frame": 2, "standing": earlier, "checksumFrame": 26488, "checksum": 199640548})
    timeline.progress({"timeMs": 874500, "frame": 3, "standing": standing})
    verdict = check(_job(reference), {"mismatches": 0, "standing": standing}, timeline)
    assert verdict["checksum"]["agrees"] and verdict["standing"]["agrees"] and verdict["standing"]["at_ms"] == 874500
    wrong = Timeline([], [])
    wrong.progress({"timeMs": 874500, "frame": 3, "standing": earlier, "checksumFrame": 26488, "checksum": 1})
    verdict = check(_job(reference), {"mismatches": 0, "standing": earlier}, wrong)
    assert not verdict["checksum"]["agrees"] and not verdict["standing"]["agrees"]
    assert check(_job(None), {"mismatches": 3}, wrong) == {"mismatches": 3}


class _Connection:
    """Keeps the frames a session sends, decoded to the control instructions they carry."""

    def __init__(self) -> None:
        self.sent = []

    def sendall(self, data: bytes) -> None:
        import json

        from rwintel.wire.frames import HEADER_SIZE, decode_header

        kind, _, _, length = decode_header(data[:HEADER_SIZE])
        body = data[HEADER_SIZE:HEADER_SIZE + length]
        self.sent.append((kind.name, json.loads(body) if kind.name == "CONTROL" else body))

    def close(self) -> None:
        pass


def test_a_session_plays_its_queue_through_and_records_each_playback_beside_its_record(tmp_path):
    import json

    from rwintel.replay.playback import Jobs, PlaybackOptions, ReplaySession, make_job

    path = tmp_path / "Lake (2p) [v1.15] (30 Sep 2026 05.10.55).replay"
    path.write_bytes(replay([rc(1, command(units=[1])), es(26488, [0]), ("wait", Out().int(26500))]))
    reference = {"seconds": 874, "team": 0, "synchronisation": {"frame": 26488, "checksum": 99, "interval": 300},
                 "standing": [{"team": 0, "units": 6, "value": 17100, "income": 64, "killed": 1, "lost": 89}]}
    job = make_job(str(path), references=[reference])
    assert (job.until_ms, job.other_than_team, job.viewpoint) == (875000, 0, -1)
    connection = _Connection()
    session = ReplaySession(connection, None, Jobs([job]), PlaybackOptions(output=str(tmp_path / "out")),
                            lambda s: None)
    session.directory = str(tmp_path / "instance")
    session.instance = 0

    session.start_episode()
    instruction = connection.sent[-1][1]
    assert instruction["command"] == "replay" and instruction["otherThanTeam"] == 0 and instruction["untilMs"] == 875000
    assert (tmp_path / "instance" / "replays" / instruction["name"]).read_bytes() == path.read_bytes()

    def event(payload):
        session.on_episode(json.dumps(payload).encode("utf-8"))

    event({"event": "started", "episode": 1, "map": "", "players": [], "viewpoint": 1})
    assert session.viewpoint == 1 and session.timeline is not None
    for time_ms in (874000, 874400):
        event({"event": "progress", "timeMs": time_ms, "frame": 1, "standing": reference["standing"],
               "checksumFrame": 26488, "checksum": 99})
    assert session.records == []
    event({"event": "finished", "episode": 1, "seconds": 875, "frames": 26600, "standing": reference["standing"],
           "sync": {"frame": 26488, "checksum": 99}, "replay": instruction["name"], "mismatches": 0,
           "ended": "until", "exhausted": True})
    record = session.records[0]
    assert record.replay == {"file": instruction["name"], "mismatches": 0, "ended": "until", "exhausted": True,
                             "viewpoint": 1}
    verdict = session.results[0]["check"]
    assert verdict["checksum"]["agrees"] and verdict["standing"]["agrees"]
    written = tmp_path / "out" / job.stem
    assert json.loads((written / "summary-slot1.json").read_text())["check"] == verdict
    assert len((written / "timeline-slot1.jsonl").read_text().splitlines()) == 2
    assert session.done


def test_a_replay_that_would_not_load_is_skipped_for_the_next_one(tmp_path):
    import json

    from rwintel.replay.playback import Jobs, PlaybackOptions, ReplaySession, make_job

    paths = []
    for name in ("a.replay", "b.replay"):
        path = tmp_path / name
        path.write_bytes(replay([rc(1, command(units=[1]))]))
        paths.append(str(path))
    connection = _Connection()
    session = ReplaySession(connection, None, Jobs([make_job(p, viewpoint=1) for p in paths]),
                            PlaybackOptions(output=str(tmp_path / "out")), lambda s: None)
    session.directory = str(tmp_path / "instance")
    session.start_episode()
    session.on_episode(json.dumps({"event": "failed", "reason": "did not load"}).encode("utf-8"))
    assert session.failure is None and session.results[0]["failed"] == "did not load"
    assert connection.sent[-1][1]["name"] == "b.replay"


def test_a_replay_with_neither_a_slot_nor_a_record_cannot_say_whose_side_to_watch(tmp_path):
    from rwintel.replay.playback import make_job

    path = tmp_path / "a.replay"
    path.write_bytes(replay([rc(1, command(units=[1]))]))
    with pytest.raises(ValueError):
        make_job(str(path))


def test_a_replay_finds_its_record_by_name_or_by_its_last_checksum_before_the_end():
    read = read_replay(replay([es(300, [0]), es(601, [0]), chat(650, "host-0 was defeated (Team: A)")]))
    named = {"replay": {"file": "match.replay"}, "synchronisation": {"frame": 5}}
    by_frame = {"synchronisation": {"frame": 601, "interval": 300}, "team": 0}
    too_early = {"synchronisation": {"frame": 300, "interval": 300}, "team": 0}
    assert find_reference([by_frame, named], "local/replays/match.replay", read) is named
    assert find_reference([too_early, by_frame], "other.replay", read) is by_frame
    assert find_reference([by_frame, dict(by_frame)], "other.replay", read) is None
