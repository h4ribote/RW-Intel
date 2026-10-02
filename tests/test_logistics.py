"""The lift layer held to: transports kept in fixed slots, lifts planned from the map, a slot busy until its lift ends, a squad carried over in as many trips as it takes, and requests nobody can serve counted as a shortfall.

The map is the two islands of `test_reach`: land, a strait of water, land. A hovercraft can cross it and loads tanks and builders; a gun boat loads nothing.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.logistics import TRANSPORT_SLOTS, UNREPORTED_PERIODS, Logistics
from rwintel.control.policy.reach import Reach
from rwintel.wire import CargoKind, LiftPhase, LiftState, NO_SQUAD

from test_reach import REGIONS, _at, _passage

TANK, BUILDER, HOVERCRAFT, BOAT = 0, 1, 2, 3
KINDS = {
    TANK: SimpleNamespace(index=TANK, movement="LAND", transport=False, carries=()),
    BUILDER: SimpleNamespace(index=BUILDER, movement="LAND", transport=False, carries=()),
    HOVERCRAFT: SimpleNamespace(index=HOVERCRAFT, movement="HOVER", transport=True, carries=(TANK, BUILDER)),
    BOAT: SimpleNamespace(index=BOAT, movement="WATER", transport=True, carries=()),
}


class _Catalogue:
    def kind(self, index):
        return KINDS.get(index)


def _unit(unit_id, type_index, at, aboard=0):
    return SimpleNamespace(id=unit_id, type_index=type_index, x=at[0], y=at[1], built=255, aboard=aboard, carrier=0)


def _view(transports, lifts=()):
    observation = SimpleNamespace(lifts=list(lifts))
    return SimpleNamespace(observation=observation, transports=[SimpleNamespace(unit=u) for u in transports])


def _logistics():
    logistics = Logistics(_Catalogue())
    logistics.reach = Reach(_passage(), REGIONS)
    return logistics


TANKS = [(TANK, "LAND", *_at(1, 4)), (TANK, "LAND", *_at(2, 4))]


def _report(lift, phase, reason=0):
    return LiftState(lift=lift, phase=int(phase), reason=reason, loaded=0, expected=2, transports=1, health=100.0,
                     eta_ms=0, squad=0, drop_region=1)


def test_transports_are_held_in_fixed_slots_by_id_and_keep_them():
    logistics = _logistics()
    logistics.update(_view([_unit(30, HOVERCRAFT, _at(5, 1)), _unit(20, HOVERCRAFT, _at(6, 1))]))
    assert [slot.unit for slot in logistics.slots] == [20, 30] + [None] * (TRANSPORT_SLOTS - 2)
    logistics.update(_view([_unit(30, HOVERCRAFT, _at(5, 1)), _unit(10, HOVERCRAFT, _at(6, 1))]))
    assert [slot.unit for slot in logistics.slots][:2] == [10, 30]


def test_a_squad_is_carried_only_by_a_transport_that_loads_it_and_reaches_both_ends():
    logistics = _logistics()
    logistics.update(_view([_unit(20, HOVERCRAFT, _at(6, 1)), _unit(21, BOAT, _at(5, 1))]))
    assert logistics.candidates(0, TANKS, 1) == [0]
    assert logistics.options(0, TANKS, [1], walk_on=True) == {1: [0]}
    assert logistics.lift_squad(0, TANKS, 0, 1, now=1000)
    row, = logistics.rows()
    assert (row.transports, row.cargo_kind, row.cargo, row.drop_region) == ([20], CargoKind.SQUAD, [0], 1)
    assert row.drop_x >= 8 * 20.0 and row.pickup_x < 4 * 20.0
    # Busy now: no other squad may have it, and asking again for the same errand sends nothing new.
    assert logistics.candidates(1, TANKS, 1) == []
    logistics.update(_view([_unit(20, HOVERCRAFT, _at(6, 1)), _unit(21, BOAT, _at(5, 1))], [_report(row.lift, LiftPhase.LOADING)]))
    assert logistics.lift_squad(0, TANKS, 0, 1, now=2000) and logistics.rows() == []


def test_a_fleet_escorts_only_a_transport_standing_where_its_ships_can_go():
    """An escort's target is the transport itself, so a transport standing inland is escorted by no ship, whatever the region it is bound for."""
    from rwintel.control.policy.contracts import Doctrine, SquadRecord
    from rwintel.control.policy.operations import Operations, _Reach

    from test_learning import _CATALOGUE

    operations = Operations(None, _CATALOGUE)
    operations.logistics = _logistics()
    slot = operations.logistics.slots[0]
    slot.unit, slot.lift, slot.region, slot.drop = 20, 7, 1, _at(10, 3)
    operations.reach = lambda view, squad: _Reach(walk={0, 1}, lift={}, members=[(0, "WATER", *_at(5, 1))], ids=[30])
    view = SimpleNamespace(region=lambda region: SimpleNamespace(id=region))
    fleet = SquadRecord(id=3, doctrine=Doctrine.FLEET)
    slot.x, slot.y = _at(0, 2)
    assert operations._escort(view, fleet, None) is None and operations.escorts == {}
    slot.x, slot.y = _at(6, 2)
    plan = operations._escort(view, fleet, None)
    assert plan is not None and plan.target == 20 and operations.escorts == {0: 3}


def test_a_fleet_escorts_a_transport_by_its_drop_point_not_by_the_centre_of_the_region_it_is_bound_for():
    """A ship cannot walk to a region whose centre stands inland, yet it follows a transport across the water to a drop point on the shore; a drop point farther from any water it reaches than the landing reach is escorted by no ship."""
    from rwintel.control.policy.contracts import Doctrine, SquadRecord
    from rwintel.control.policy.operations import Operations, _Reach
    from rwintel.control.policy.reach import ANCHOR_TILES
    from rwintel.wire.terrain import decode_terrain, encode_terrain

    from test_learning import _CATALOGUE

    import numpy as np

    # Land 0-3, water 4-7, and a far island of land wide enough that its eastern end lies beyond the landing reach of any water.
    width, height = 12 + ANCHOR_TILES, 6
    land = np.full((height, width), -1)
    land[:, 0:4] = 0
    land[:, 8:] = 1
    water = np.full((height, width), -1)
    water[:, 4:8] = 0
    passage = decode_terrain(encode_terrain(width, height, {"LAND": land, "HOVER": np.zeros((height, width), dtype=int),
                                                            "WATER": water}))
    shore, far = _at(12, 3), _at(width - 1, 3)
    regions = [SimpleNamespace(id=0, x=_at(1, 2)[0], y=_at(1, 2)[1]), SimpleNamespace(id=1, x=shore[0], y=shore[1]),
               SimpleNamespace(id=2, x=far[0], y=far[1])]
    terrain = Reach(passage, regions)
    sea = _at(5, 1)
    assert not terrain.walkable("WATER", *sea, 1) and not terrain.walkable("WATER", *sea, 2)

    operations = Operations(None, _CATALOGUE)
    operations.logistics = Logistics(_Catalogue())
    operations.logistics.reach = terrain
    slot = operations.logistics.slots[0]
    slot.unit, slot.type_index, slot.lift = 20, HOVERCRAFT, 7
    slot.x, slot.y = _at(6, 2)
    operations.reach = lambda view, squad: _Reach(walk={0}, lift={}, members=[(0, "WATER", *sea)], ids=[30])
    view = SimpleNamespace(region=lambda region: SimpleNamespace(id=region))
    fleet = SquadRecord(id=3, doctrine=Doctrine.FLEET)

    slot.region, slot.drop = 2, terrain.landing(2, "HOVER")
    assert operations._escort(view, fleet, None) is None and operations.escorts == {}
    slot.region, slot.drop = 1, terrain.landing(1, "HOVER")
    plan = operations._escort(view, fleet, None)
    assert plan is not None and plan.target == 20 and plan.region.id == 1 and operations.escorts == {0: 3}


def test_a_slot_is_freed_when_its_lift_ends_and_the_rest_of_a_squad_goes_as_a_list_of_units():
    logistics = _logistics()
    hovercraft = _unit(20, HOVERCRAFT, _at(6, 1))
    logistics.update(_view([hovercraft]))
    logistics.lift_squad(0, TANKS, 0, 1, now=1000)
    lift = logistics.rows()[0].lift
    logistics.update(_view([hovercraft], [_report(lift, LiftPhase.DONE)]))
    assert logistics.slots[0].free
    assert logistics.lift_squad(0, TANKS[:1], 0, 1, now=5000, units=[7])
    row, = logistics.rows()
    assert (row.cargo_kind, row.cargo) == (CargoKind.UNITS, [7])
    assert (logistics.slots[0].squad, logistics.slots[0].units) == (0, (7,))


def test_a_failed_lift_frees_its_slot():
    logistics = _logistics()
    hovercraft = _unit(20, HOVERCRAFT, _at(6, 1))
    logistics.update(_view([hovercraft]))
    logistics.lift_squad(0, TANKS, 0, 1, now=1000)
    lift = logistics.rows()[0].lift
    logistics.update(_view([hovercraft], [_report(lift, LiftPhase.FAILED, reason=1)]))
    assert logistics.slots[0].free


def test_a_busy_transport_is_not_given_to_another_squad():
    logistics = _logistics()
    logistics.update(_view([_unit(20, HOVERCRAFT, _at(6, 1))]))
    assert logistics.lift_squad(0, TANKS, 0, 1, now=1000)
    assert not logistics.lift_squad(1, TANKS, 0, 1, now=1000)
    assert [row.cargo for row in logistics.rows()] == [[0]]


def test_a_lift_the_game_stops_reporting_gives_its_slot_back():
    logistics = _logistics()
    hovercraft = _unit(20, HOVERCRAFT, _at(6, 1))
    logistics.update(_view([hovercraft]))
    logistics.lift_squad(0, TANKS, 0, 1, now=1000)
    for _ in range(UNREPORTED_PERIODS):
        logistics.update(_view([hovercraft]))
    assert logistics.slots[0].free


def test_a_builder_is_carried_by_the_nearest_free_transport_and_nobody_able_is_a_shortfall():
    logistics = _logistics()
    builder = (5, BUILDER, "LAND", *_at(2, 2))
    logistics.update(_view([_unit(21, BOAT, _at(5, 1))]))
    assert not logistics.lift_units([builder], 1, now=1000)
    assert logistics.shortfall.count == 1 and logistics.shortfall.passengers == {BUILDER}
    logistics.update(_view([_unit(21, BOAT, _at(5, 1)), _unit(20, HOVERCRAFT, _at(6, 1))]))
    assert logistics.shortfall.count == 0
    assert logistics.lift_units([builder], 1, now=2000)
    row, = logistics.rows()
    assert (row.transports, row.cargo_kind, row.cargo) == ([20], CargoKind.UNITS, [5])
    # The boat came first and keeps slot 0; the hovercraft took the next.
    assert logistics.slots[1].squad == NO_SQUAD and logistics.slots[1].units == (5,)


def test_releasing_a_squad_cancels_its_lift():
    logistics = _logistics()
    hovercraft = _unit(20, HOVERCRAFT, _at(6, 1))
    logistics.update(_view([hovercraft]))
    logistics.lift_squad(0, TANKS, 0, 1, now=1000)
    lift = logistics.rows()[0].lift
    logistics.update(_view([hovercraft], [_report(lift, LiftPhase.APPROACH)]))
    logistics.release(0)
    row, = logistics.rows()
    assert row.lift == lift and row.cancel and logistics.slots[0].free
