"""What the combat table promises: a square law between forces, damage counted only where it can land, and a type's worth per credit that the build order can choose by."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.policy.combat import MATCHUP_PRIOR, CombatTable
from rwintel.control.policy.contracts import ALLOCATION, EconomyOrders, Posture, Role
from rwintel.control.policy.economy import Economy
from rwintel.control.policy.options import Options
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.wire import BLOCK_MENUS, BLOCK_REGIONS, BLOCK_UNITS, Observation, RegionState, UnitState

TANK, GUN, JET, AA, BUILDER, FACTORY, MECHS, HQ, HEAVY = range(9)

TYPES = [
    UnitType(index=TANK, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0, max_hp=210.0),
    UnitType(index=GUN, name="mechGun", lookup="mechGun", price=600, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0, max_hp=500.0),
    UnitType(index=JET, name="jet", lookup="jet", price=250, tech=1, building=False, builder=False,
             movement="AIR", can_attack=True, range=120.0, max_hp=50.0),
    UnitType(index=AA, name="flak", lookup="flak", price=900, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=190.0, hits_air=True, hits_land=False, max_hp=500.0),
    UnitType(index=BUILDER, name="builder", lookup="builder", price=500, tech=1, building=False, builder=True,
             movement="LAND", menu=(FACTORY, MECHS)),
    UnitType(index=FACTORY, name="landFactory", lookup="landFactory", price=700, tech=1, building=True, builder=False,
             movement="NONE", menu=(BUILDER, TANK, JET)),
    UnitType(index=MECHS, name="mechFactory", lookup="mechFactory", price=1000, tech=1, building=True, builder=False,
             movement="NONE", menu=(GUN, AA)),
    UnitType(index=HQ, name="commandCenter", lookup="commandCenter", price=3000, tech=1, building=True,
             builder=False, movement="NONE", menu=(BUILDER,)),
    UnitType(index=HEAVY, name="heavy", lookup="heavy", price=3900, tech=2, building=False, builder=False,
             movement="LAND", can_attack=True, range=190.0, max_hp=2600.0),
]


def _definition(dps, hp, builds=(), built_from=()):
    return SimpleNamespace(damage_per_second=dps, max_hp=hp, builds=list(builds), built_from=list(built_from))


DEFINITIONS = {
    "tank": _definition(20.0, 210.0),
    "mechGun": _definition(46.0, 500.0, built_from=["mechFactory"]),
    "jet": _definition(15.0, 50.0),
    "flak": _definition(53.0, 500.0, built_from=["mechFactory"]),
    "mechFactory": _definition(0.0, 1800.0, builds=["flak", "mechGun"]),
    "heavy": _definition(156.0, 2600.0, built_from=["landFactory"]),
}


def _Catalogue(definitions=DEFINITIONS):
    """The type table with the definitions a fight's numbers come from."""
    catalogue = Catalogue.of_types(TYPES)
    catalogue.definitions = definitions
    return catalogue


CATALOGUE = _Catalogue()


def _table(matchups=None):
    return CombatTable(CATALOGUE, matchups)


# ---- the model --------------------------------------------------------------------------


def test_equal_forces_draw_and_the_larger_force_keeps_what_the_square_law_says():
    table = _table()
    assert table.outcome([(TANK, 10)], [(TANK, 10)]) == 0.0
    # Twice the tanks: strength twice as large, and the winner keeps the square root of one less a quarter.
    assert abs(table.outcome([(TANK, 20)], [(TANK, 10)]) - (1 - 0.25) ** 0.5) < 1e-9
    assert table.outcome([(TANK, 10)], [(TANK, 20)]) < 0


def test_damage_that_cannot_land_does_not_count():
    """Tanks cannot shoot upward, so aircraft over them lose nothing whatever the numbers; anti-air can, and does."""
    table = _table()
    assert table.outcome([(JET, 5)], [(TANK, 50)]) == 1.0
    assert table.outcome([(AA, 5)], [(JET, 5)]) > 0.9


def test_a_type_is_worth_its_health_times_damage_over_its_price_squared_per_credit():
    table = _table()
    tanks = [(TANK, 10)]
    assert table.efficiency(GUN, tanks) > table.efficiency(TANK, tanks)
    assert abs(table.efficiency(TANK, tanks) - 210.0 * 20.0 / 350.0 ** 2) < 1e-9
    # Per unit it is the heavy that carries the most, which is what matters when the slots run out.
    assert table.efficiency(HEAVY, tanks, per="unit") > table.efficiency(GUN, tanks, per="unit")
    # Against aircraft only what can reach them is worth anything, and what cannot is worth nothing at all.
    air = [(JET, 10)]
    assert table.efficiency(TANK, air) == 0.0 and table.efficiency(AA, air) > 0


def test_measured_matchups_bend_the_model_by_how_many_fights_stand_behind_them():
    """A tank that the arena saw lose every fight against mech guns is worth less against them than the model says, and a single fight moves the figure less than many."""
    model = _table().efficiency(TANK, [(GUN, 5)])
    few = _table({(TANK, GUN): (-0.9, 1.0)}).efficiency(TANK, [(GUN, 5)])
    many = _table({(TANK, GUN): (-0.9, 20 * MATCHUP_PRIOR)}).efficiency(TANK, [(GUN, 5)])
    assert many < few < model


# ---- the build order ---------------------------------------------------------------------


class _Session:
    regions = []
    map_content = None

    def type_by_lookup(self, lookup):
        return next((kind for kind in TYPES if kind.lookup == lookup), None)


def _unit(unit_id, type_index, x=100.0, hostile=0, level=1, upgrade_price=0):
    return UnitState(id=unit_id, squad=0xFFFF, type_index=type_index, x=x, y=100.0, health=100.0, max_health=100.0,
                     built=255, order=255, queued=0, target=0, stance=5, hostile=hostile, since_hit_ms=9999,
                     level=level, upgrade_price=upgrade_price)


def _view(units, credits, menus, income=10.0, t=30000):
    region = RegionState(id=0, resources=2, held_by_us=2, held_by_enemy=0, x=100.0, y=100.0, our_value=0.0,
                         enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=0.0)
    observation = Observation(frame=1, game_time_ms=t, episode=1, blocks=BLOCK_REGIONS | BLOCK_UNITS | BLOCK_MENUS,
                              slot=0, credits=credits, income=income, units=len(units), unit_cap=100,
                              under_construction=0, killed_units=0, killed_buildings=0, lost_units=0,
                              lost_buildings=0, regions=[region], unit_states=list(units), menus=dict(menus))
    return build_view(observation, CATALOGUE, 0)


ORDERS = EconomyOrders(posture=Posture.ARM, allocation=ALLOCATION[Posture.ARM], tech_cap=0.0,
                       target_mix={Role.ARMOUR: 1.0})


def _made(out):
    return ["upgrade" if p.kind == 2 else TYPES[p.type_index].lookup for p in out]


def test_a_factory_makes_what_buys_the_most_against_what_the_enemy_fields():
    """Against tanks the mech gun buys more per credit; against aircraft only anti-air is worth anything."""
    economy = Economy(_Session(), CATALOGUE)
    menus = {10: [GUN, AA, TANK]}
    tanks = [_unit(90 + i, TANK, x=900.0, hostile=1) for i in range(5)]
    assert _made(economy.decide(_view([_unit(1, BUILDER), _unit(10, MECHS)] + tanks, 2000.0, menus), ORDERS, [])) == ["mechGun"]
    jets = [_unit(90 + i, JET, x=900.0, hostile=1) for i in range(5)]
    economy = Economy(_Session(), CATALOGUE)
    assert _made(economy.decide(_view([_unit(1, BUILDER), _unit(10, MECHS)] + jets, 2000.0, menus), ORDERS, [])) == ["flak"]


def test_a_factory_waits_for_the_best_type_and_the_price_is_held_back_from_the_rest():
    """The best type is out of reach for now and the cheap one is worth far less, so the first factory waits; the second may not spend what the first is waiting for."""
    economy = Economy(_Session(), CATALOGUE)
    menus = {10: [GUN, TANK], 11: [GUN, TANK]}
    tanks = [_unit(90 + i, TANK, x=900.0, hostile=1) for i in range(5)]
    units = [_unit(1, BUILDER), _unit(10, FACTORY), _unit(11, FACTORY, x=300.0)] + tanks
    assert _made(economy.decide(_view(units, 500.0, menus, income=20.0), ORDERS, [])) == []
    assert economy.ledger.unordered_factory_periods == 2
    off = Economy(_Session(), CATALOGUE, Options(choose=False))
    assert _made(off.decide(_view(units, 500.0, menus, income=20.0), ORDERS, [])) == ["tank"]


def test_the_second_factory_is_the_kind_whose_units_buy_the_most():
    """The first factory is the land factory, which makes builders; with one standing and credits piling up, the next one is the mech factory, whose mech gun beats the tank per credit."""
    economy = Economy(_Session(), CATALOGUE)
    builder_menu = [FACTORY, MECHS]
    menus = {1: builder_menu, 10: [BUILDER, TANK]}
    tanks = [_unit(90 + i, TANK, x=900.0, hostile=1) for i in range(5)]
    units = [_unit(1, BUILDER), _unit(40, HQ), _unit(10, FACTORY)] + tanks
    made = _made(economy.decide(_view(units, 6000.0, menus), ORDERS, []))
    assert "mechFactory" in made and "landFactory" not in made
    off = Economy(_Session(), CATALOGUE, Options(choose=False))
    assert "landFactory" in _made(off.decide(_view(units, 6000.0, menus), ORDERS, []))


def test_a_factory_tier_is_raised_only_when_what_it_would_make_is_worth_more():
    """The heavy unit of the next tier is worth less per credit than the tank against tanks, so the raise is not bought."""
    orders = replace(ORDERS, tech_cap=100000.0)
    tanks = [_unit(90 + i, TANK, x=900.0, hostile=1) for i in range(5)]
    units = [_unit(1, BUILDER), _unit(10, FACTORY, upgrade_price=1500), _unit(11, FACTORY, x=300.0)] + tanks
    menus = {10: [BUILDER, TANK], 11: [BUILDER, TANK]}
    economy = Economy(_Session(), CATALOGUE, Options(tech="off"))
    assert "upgrade" not in _made(economy.decide(_view(units, 5000.0, menus, income=40.0), orders, []))
