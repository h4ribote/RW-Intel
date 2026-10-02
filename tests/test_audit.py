"""Where our losses are booked: by role, in a squad or loose, and under the enemy's defences."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.audit import EVENT_UNIT_LOST, LossAudit
from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.wire import BLOCK_UNITS, NO_SQUAD, EventState, Observation, UnitState

TYPES = [
    UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0),
    UnitType(index=1, name="turret", lookup="turret", price=500, tech=1, building=True, builder=False,
             movement="NONE", can_attack=True, range=200.0),
]


CATALOGUE = Catalogue.of_types(TYPES)


def _unit(unit_id, type_index, x, squad=NO_SQUAD, hostile=0):
    return UnitState(id=unit_id, squad=squad, type_index=type_index, x=x, y=0.0, health=100.0, max_health=100.0,
                     built=255, order=255, queued=0, target=0, stance=5, hostile=hostile, since_hit_ms=9999)


def _view(units, events=()):
    observation = Observation(frame=1, game_time_ms=1000, episode=1, blocks=BLOCK_UNITS, slot=0, credits=0.0,
                              income=0.0, units=len(units), unit_cap=100, under_construction=0, killed_units=0,
                              killed_buildings=0, lost_units=0, lost_buildings=0, unit_states=list(units),
                              events=list(events))
    return build_view(observation, CATALOGUE, None)


def _lost(unit_id, squad):
    return EventState(kind=EVENT_UNIT_LOST, squad=squad, unit=unit_id, type_index=0, value=350.0)


def test_a_loss_is_booked_by_role_by_squad_and_by_whether_the_enemy_defences_reached_it():
    turret = _unit(90, 1, x=1000.0, hostile=1)
    audit = LossAudit()
    audit.update(_view([_unit(1, 0, x=900.0, squad=2), _unit(2, 0, x=0.0), turret]))
    audit.update(_view([turret], [_lost(1, 2), _lost(2, NO_SQUAD)]))
    assert audit.summary() == {"loose": 350, "role_armour": 700, "squad": 350, "under_defences": 350}
