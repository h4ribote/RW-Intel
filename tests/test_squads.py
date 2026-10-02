"""How reinforcements reach a squad and how a garrison covers the expansion, held to.

A unit bound for a squad on the far side of the board waits for others going the same way, and goes alone only once it has waited long enough. One garrison with nothing to do goes to stand on the next region of the expansion plan, and only when nothing of ours is there or on its way.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.policy.contracts import (
    Doctrine,
    OperationsOrders,
    Posture,
    SquadRecord,
    TaskContract,
)
from rwintel.control.policy.operations import Operations
from rwintel.control.policy.options import Options
from rwintel.control.policy.organisation import CONVOY_DISTANCE, CONVOY_SIZE, CONVOY_WAIT_MS, Organisation
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.wire import BLOCK_REGIONS, BLOCK_UNITS, Observation, RegionState, Stance, Status, Task, UnitState

TYPES = [UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
                  movement="LAND", can_attack=True, range=130.0)]


CATALOGUE = Catalogue.of_types(TYPES)
FAR = CONVOY_DISTANCE + 1000.0


def _tank(unit_id, x, squad=0xFFFF):
    return UnitState(id=unit_id, squad=squad, type_index=0, x=x, y=0.0, health=100.0, max_health=100.0,
                     built=255, order=255, queued=0, target=0, stance=5, hostile=0, since_hit_ms=9999)


def _board(units, t, regions=()):
    observation = Observation(frame=1, game_time_ms=t, episode=1, blocks=BLOCK_UNITS | BLOCK_REGIONS, slot=0,
                              credits=0.0, income=0.0, units=len(units), unit_cap=100, under_construction=0,
                              killed_units=0, killed_buildings=0, lost_units=0, lost_buildings=0,
                              regions=list(regions), unit_states=list(units))
    return build_view(observation, CATALOGUE, None)


def _far_vanguard(options=Options()):
    """An organisation layer holding one vanguard of four tanks, half of its establishment, far out on the board."""
    organisation = Organisation(None, CATALOGUE, options)
    organisation.squads[0] = SquadRecord(id=0, doctrine=Doctrine.VANGUARD, members=[1, 2, 3, 4], value=1400.0,
                                         formed_value=1400.0, x=FAR, y=0.0)
    organisation.free_ids.remove(0)
    squad = [_tank(i, FAR, squad=0) for i in range(1, 5)]
    return organisation, squad


def _joined(organisation):
    return set(organisation.squads[0].members) - {1, 2, 3, 4}


def test_a_lone_reinforcement_for_a_distant_squad_waits_and_then_goes_alone():
    organisation, squad = _far_vanguard()
    organisation.update(_board(squad + [_tank(10, 0.0)], 10000), [])
    assert _joined(organisation) == set()
    organisation.update(_board(squad + [_tank(10, 0.0)], 10000 + CONVOY_WAIT_MS), [])
    assert _joined(organisation) == {10}


def test_reinforcements_for_a_distant_squad_leave_together():
    organisation, squad = _far_vanguard()
    loose = [_tank(10 + i, 0.0) for i in range(CONVOY_SIZE)]
    organisation.update(_board(squad + loose, 10000), [])
    assert _joined(organisation) == {10 + i for i in range(CONVOY_SIZE)}


def test_without_convoys_a_reinforcement_goes_at_once():
    organisation, squad = _far_vanguard(Options(convoy=False))
    organisation.update(_board(squad + [_tank(10, 0.0)], 10000), [])
    assert _joined(organisation) == {10}


def test_the_army_gathers_into_vanguards_once_the_garrisons_are_raised():
    """Tanks arriving two at a time, with the garrisons capped at two: switched to raise vanguards, two garrisons are raised and filled, and the tanks after them wait until four can form a vanguard; not switched, the tanks only ever make garrisons."""
    two = replace(Options().tuning, max_garrisons=2.0)
    def raise_squads(options):
        organisation = Organisation(None, CATALOGUE, options)
        units = []
        for step in range(6):
            units += [_tank(100 + 2 * step, 0.0), _tank(101 + 2 * step, 0.0)]
            organisation.update(_board([_tank(u.id, u.x, squad=_squad_of(organisation, u.id)) for u in units],
                                       10000 * (step + 1)), [])
        return sorted(record.doctrine for record in organisation.squads.values())

    assert raise_squads(Options(tuning=two)) == [Doctrine.VANGUARD, Doctrine.GARRISON, Doctrine.GARRISON]
    assert set(raise_squads(Options(vanguards=False, tuning=two))) == {Doctrine.GARRISON}


def _squad_of(organisation, unit_id):
    """Which squad the organisation has put a unit in, as the game side would report it back."""
    return next((record.id for record in organisation.squads.values() if unit_id in record.members), 0xFFFF)


def _region(region_id, x, held=0, ours=0.0):
    return RegionState(id=region_id, resources=2, held_by_us=held, held_by_enemy=0, x=x, y=0.0, our_value=ours,
                       enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=abs(x))


def _garrison(squad_id, x, target=None, status=Status.COMPLETE):
    squad = SquadRecord(id=squad_id, doctrine=Doctrine.GARRISON, members=[squad_id * 10], value=700.0,
                        formed_value=700.0, x=x, y=0.0, status=status)
    if target is not None:
        squad.contract = TaskContract(squad=squad_id, task=Task.DEFEND, target_region=target,
                                      stance=Stance.GUARD_AREA, cost_budget=500.0, deadline_ms=90000,
                                      issued_at_ms=1000)
    return squad


def _orders(expansion):
    return OperationsOrders(posture=Posture.EXPAND, priorities={0: 1.0, 1: 0.5, 2: 0.5}, offensive=True,
                            loss_allowance=1000.0, expansion=list(expansion))


REGIONS = [_region(0, 0.0, held=1, ours=700.0), _region(1, 900.0, held=1, ours=700.0), _region(2, 2000.0)]


def test_the_nearest_quiet_garrison_covers_the_next_expansion():
    view = _board([], 60000, REGIONS)
    home, near = _garrison(0, 0.0, target=0), _garrison(1, 900.0, target=1)
    contracts, _ = Operations(None, CATALOGUE).decide(view, _orders([2]), [home, near], [], 60000)
    assert [(c.squad, c.target_region, c.task) for c in contracts] == [(1, 2, Task.DEFEND)]


def test_no_garrison_is_sent_when_one_is_already_bound_for_the_expansion_or_cover_is_off():
    view = _board([], 60000, REGIONS)
    bound = [_garrison(0, 0.0, target=0), _garrison(1, 900.0, target=2, status=Status.ACTIVE)]
    contracts, _ = Operations(None, CATALOGUE).decide(view, _orders([2]), bound, [], 60000)
    assert all(c.target_region != 2 or c.squad == 1 for c in contracts)
    assert not any(c.squad == 0 and c.target_region == 2 for c in contracts)

    quiet = [_garrison(0, 0.0, target=0), _garrison(1, 900.0, target=1)]
    contracts, _ = Operations(None, CATALOGUE, Options(cover=False)).decide(view, _orders([2]), quiet, [], 60000)
    assert not any(c.target_region == 2 for c in contracts)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
