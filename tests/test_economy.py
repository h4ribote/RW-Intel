"""What the build order promises about turning credits into an army, held to.

Three failures are pinned, and all three show up as credits piling in the treasury while nothing is built rather than as an error. A type the definition files make look buildable but that no factory actually delivers must stop being ordered, and must not hold every factory while it is being found out. A dead builder must be replaced ahead of the army, but by one order and not one every period. And credits banking up must raise another factory.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.catalogue import Catalogue, role_of
from rwintel.control.policy.contracts import ALLOCATION, EconomyOrders, Posture, Role
from dataclasses import replace

from rwintel.control.policy.tuning import Tuning
from rwintel.control.policy.economy import (
    PLACEMENT_PENDING_MS,
    BUILDER_ORDER_GRACE_MS,
    FACTORY_BUSY_MS,
    MAX_BUILDERS,
    PLACEMENT_GRACE_MS,
    POINTS_PER_BUILDER,
    STUCK_MS,
    TECH_FUND_LIMIT,
    Economy as _Economy,
    ResourcePoint,
)
from rwintel.control.policy.options import Options
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.wire import BLOCK_MENUS, BLOCK_REGIONS, BLOCK_UNITS, Observation, RegionState, UnitState
from rwintel.wire.action import ProductionKind

TYPES = [
    UnitType(index=0, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0),
    UnitType(index=1, name="builder", lookup="builder", price=500, tech=1, building=False, builder=True,
             movement="LAND", menu=(2, 5)),
    UnitType(index=2, name="landFactory", lookup="landFactory", price=700, tech=1, building=True, builder=False,
             movement="NONE", menu=(1, 0, 4)),
    # Reads as anti-air a factory can make, because its definition names no maker, and is in fact a part of something else.
    UnitType(index=3, name="attachment", lookup="attachment", price=300, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=290.0, hits_air=True, hits_land=False),
    UnitType(index=4, name="aaTank", lookup="aaTank", price=400, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=200.0, hits_air=True, hits_land=False),
    UnitType(index=5, name="extractor", lookup="extractor", price=700, tech=1, building=True, builder=False,
             movement="NONE", extractor=True),
    UnitType(index=6, name="heavyTank", lookup="heavyTank", price=1200, tech=2, building=False, builder=False,
             movement="LAND", can_attack=True, range=150.0),
    # What a raised extractor reports itself as: a building type of a higher level, which says nothing about the factory.
    UnitType(index=7, name="extractorT3", lookup="extractorT3", price=700, tech=3, building=True, builder=False,
             movement="NONE", extractor=True),
    UnitType(index=8, name="commandCenter", lookup="commandCenter", price=3000, tech=1, building=True, builder=False,
             movement="NONE", menu=(1,)),
]


def _Catalogue(types=TYPES):
    """The type table and what each type's menu makes, without the definitions the asset tree provides."""
    return Catalogue.of_types(types)


class _Session:
    regions = []
    map_content = None

    def type_by_lookup(self, lookup):
        return next((kind for kind in TYPES if kind.lookup == lookup), None)


def Economy(session, catalogue, options=Options(), choosing=False):
    """The build order with production filled role by role unless `choosing`, so that the rules tested here are tested on the production rule whose outcome they were written against; the choice by fighting strength has tests of its own."""
    return _Economy(session, catalogue, replace(options, choose=choosing))


def _unit(unit_id, type_index, x=100.0, y=100.0, level=1, upgrade_price=0, hostile=0):
    return UnitState(id=unit_id, squad=0xFFFF, type_index=type_index, x=x, y=y, health=100.0, max_health=100.0,
                     built=255, order=255, queued=0, target=0, stance=5, hostile=hostile, since_hit_ms=9999,
                     level=level, upgrade_price=upgrade_price)


def _view(units, t, credits, income=10.0, unit_cap=100, units_counted=None, enemy_in_region=0.0, regions=(),
          menus=None, catalogue=None):
    region = RegionState(id=0, resources=2, held_by_us=2, held_by_enemy=0, x=100.0, y=100.0, our_value=0.0,
                         enemy_value=enemy_in_region, enemy_seen_at_ms=0, distance_from_home=0.0)
    blocks = BLOCK_REGIONS | BLOCK_UNITS | (BLOCK_MENUS if menus is not None else 0)
    observation = Observation(frame=1, game_time_ms=t, episode=1, blocks=blocks, slot=0,
                              credits=credits, income=income,
                              units=len(units) if units_counted is None else units_counted, unit_cap=unit_cap,
                              under_construction=0, killed_units=0, killed_buildings=0, lost_units=0,
                              lost_buildings=0, regions=[region, *regions], unit_states=list(units),
                              menus=dict(menus or {}))
    return build_view(observation, catalogue or CATALOGUE, 0)


CATALOGUE = _Catalogue()

#: The treasury above which another factory is raised, as the chain is tuned by default.
BANKED_CREDITS = Tuning().banked_credits


def _orders(mix):
    return EconomyOrders(posture=Posture.ARM, allocation=ALLOCATION[Posture.ARM], tech_cap=0.0, target_mix=mix)


def _made(out):
    return ["upgrade" if p.kind == ProductionKind.UPGRADE else TYPES[p.type_index].lookup for p in out]


def test_a_factory_is_asked_only_for_what_its_menu_offers():
    """Anti-air is what the mix wants and the attachment is the cheapest thing that reads as it, but no factory offers it; each factory makes what its own menu holds, and a factory whose menu fills no role at all is left without an order."""
    economy = Economy(_Session(), CATALOGUE)
    mix = _orders({Role.ANTI_AIR: 1.0})
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(11, 2, x=300.0), _unit(12, 2, x=500.0)]
    menus = {10: [1, 0, 4], 11: [1, 0], 12: [1]}
    assert _made(economy.decide(_view(units, 30000, 2000.0, menus=menus), mix, [])) == ["aaTank", "tank"]
    assert economy.ledger.unordered_factory_periods == 1


def test_lost_builders_are_replaced_ahead_of_the_army_and_counted_while_on_their_way():
    """With none standing, a builder is ordered before any tank, one a period, and builders already ordered count towards the target until they could have arrived; the order is not repeated every period while the first is being made."""
    economy = Economy(_Session(), CATALOGUE)
    mix = _orders({Role.ARMOUR: 1.0})
    factories = [_unit(10, 2), _unit(11, 2, x=300.0), _unit(12, 2, x=500.0)]
    t = 30000
    made = _made(economy.decide(_view(factories, t, 2000.0), mix, []))
    assert made[0] == "builder" and made.count("builder") == 1
    second = _made(economy.decide(_view(factories, t + FACTORY_BUSY_MS, 2000.0), mix, []))
    assert second.count("builder") == 1
    third = _made(economy.decide(_view(factories, t + 2 * FACTORY_BUSY_MS, 2000.0), mix, []))
    assert "builder" not in third
    late = _made(economy.decide(_view(factories, t + BUILDER_ORDER_GRACE_MS + 2 * FACTORY_BUSY_MS, 2000.0), mix, []))
    assert late.count("builder") == 1


def test_a_wanted_role_that_cannot_be_paid_for_yet_does_not_stop_the_factories():
    """The only thing on the menu in the wanted role may cost more than the treasury holds; the factories make the next role in the order of need meanwhile rather than standing idle while credits are saved up for one unit."""
    flak = len(TYPES)
    types = TYPES + [UnitType(index=flak, name="flak", lookup="flak", price=12000, tech=1, building=False,
                              builder=False, movement="LAND", can_attack=True, range=200.0, hits_air=True,
                              hits_land=False)]

    catalogue = _Catalogue(types)
    economy = Economy(_Session(), catalogue)
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(11, 2, x=300.0), _unit(30, 0)]
    menus = {10: [0, 1, flak], 11: [0, 1, flak]}
    out = economy.decide(_view(units, 30000, 2000.0, menus=menus, catalogue=catalogue), _orders({Role.ANTI_AIR: 1.0}), [])
    assert [types[p.type_index].lookup for p in out] == ["tank", "tank"]


def _teched(mix):
    return EconomyOrders(posture=Posture.ARM, allocation=ALLOCATION[Posture.ARM], tech_cap=100.0, target_mix=mix)


def test_the_tech_level_is_the_factorys_tier_and_not_another_buildings_type():
    economy = Economy(_Session(), CATALOGUE)
    orders = _teched({Role.ARMOUR: 1.0})
    raised = _view([_unit(10, 2, level=2)], 30000, 0.0)
    assert economy._tech(raised, orders) == 2
    extractor_only = _view([_unit(10, 2, level=1), _unit(20, 7, level=3)], 30000, 0.0)
    assert economy._tech(extractor_only, orders) == 1
    assert economy._tech(raised, _orders({Role.ARMOUR: 1.0})) == 1


def test_an_extractor_is_raised_ahead_of_the_army_and_saved_for_when_it_cannot_be_paid_yet():
    economy = Economy(_Session(), CATALOGUE)
    mix = _orders({Role.ARMOUR: 1.0})
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(30, 0), _unit(20, 5, upgrade_price=1400)]
    out = economy.decide(_view(units, 30000, 2000.0), mix, [])
    assert _made(out) == ["upgrade", "tank"]
    assert out[0].producer == 20

    # Asked again straight away, the extractor is left alone while its raise goes through.
    assert "upgrade" not in _made(economy.decide(_view(units, 30000 + FACTORY_BUSY_MS, 2000.0), mix, []))

    saving = Economy(_Session(), CATALOGUE)
    assert _made(saving.decide(_view(units, 30000, 1000.0), mix, [])) == []


def test_ground_the_enemy_stands_on_is_not_invested_in():
    economy = Economy(_Session(), CATALOGUE)
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(30, 0), _unit(20, 5, upgrade_price=1400)]
    made = _made(economy.decide(_view(units, 30000, 2000.0, enemy_in_region=900.0), _orders({Role.ARMOUR: 1.0}), []))
    assert "upgrade" not in made and "tank" in made


def test_a_factory_is_raised_only_while_another_keeps_producing():
    mix = _orders({Role.ARMOUR: 1.0})
    alone = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(30, 0), _unit(10, 2, upgrade_price=2000)]
    economy = Economy(_Session(), CATALOGUE)
    assert "upgrade" not in _made(economy.decide(_view(alone, 30000, 2500.0, income=40.0), mix, []))

    pair = alone + [_unit(11, 2, x=300.0, upgrade_price=2000)]
    economy = Economy(_Session(), CATALOGUE)
    economy.tech_fund = TECH_FUND_LIMIT
    out = economy.decide(_view(pair, 30000, 2500.0, income=40.0), mix, [])
    assert _made(out).count("upgrade") == 1 and out[0].producer in (10, 11)


def test_a_third_tier_is_not_saved_for_while_the_enemy_army_is_larger():
    mix = _orders({Role.ARMOUR: 1.0})
    ours = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(30, 0), _unit(20, 5, level=2, upgrade_price=4000)]
    enemies = [_unit(90 + i, 0, x=900.0, hostile=1) for i in range(4)]
    behind = Economy(_Session(), CATALOGUE)
    assert _made(behind.decide(_view(ours + enemies, 30000, 1000.0), mix, [])) == ["tank"]
    ahead = Economy(_Session(), CATALOGUE)
    assert _made(ahead.decide(_view(ours, 30000, 1000.0), mix, [])) == []


def test_near_the_unit_cap_each_unit_carries_the_most_worth_it_can():
    economy = Economy(_Session(), CATALOGUE)
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2, level=2), _unit(30, 0)]
    # The factory's second tier adds the heavy tank to what it makes.
    menus = {10: [1, 0, 4, 6]}
    mix = _teched({Role.ARMOUR: 1.0})
    assert _made(economy.decide(_view(units, 30000, 2000.0, menus=menus), mix, [])) == ["tank"]
    crowded = Economy(_Session(), CATALOGUE)
    assert _made(crowded.decide(_view(units, 30000, 2000.0, unit_cap=10, units_counted=8, menus=menus), mix, [])) == ["heavyTank"]
    full = Economy(_Session(), CATALOGUE)
    assert _made(full.decide(_view(units, 30000, 2000.0, unit_cap=10, units_counted=10, menus=menus), mix, [])) == []


def test_credits_banking_up_raise_another_factory():
    economy = Economy(_Session(), CATALOGUE)
    mix = _orders({Role.ARMOUR: 1.0})
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2)]
    rich = _made(economy.decide(_view(units, 30000, BANKED_CREDITS + 2000.0), mix, []))
    assert "landFactory" in rich
    poor = Economy(_Session(), CATALOGUE)
    assert "landFactory" not in _made(poor.decide(_view(units, 30000, 800.0), mix, []))


def test_the_second_factory_goes_ahead_of_the_army_and_is_held_back_for_when_it_cannot_be_paid():
    """Once the income can feed a second factory it is placed before the army takes the credits, and while it cannot be paid for its price is kept from the army; a placement on its way is not repeated."""
    mix = _orders({Role.ARMOUR: 1.0})
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(30, 0)]
    economy = Economy(_Session(), CATALOGUE)
    assert _made(economy.decide(_view(units, 30000, 800.0, income=40.0), mix, [])) == ["landFactory"]
    assert "landFactory" not in _made(economy.decide(_view(units, 30000 + FACTORY_BUSY_MS, 800.0, income=40.0), mix, []))
    saving = Economy(_Session(), CATALOGUE)
    assert _made(saving.decide(_view(units, 30000, 600.0, income=40.0), mix, [])) == []


def _outpost(points):
    """A region beyond home with open resource points, and an economy that knows where they are."""
    region = RegionState(id=1, resources=points, held_by_us=0, held_by_enemy=0, x=1000.0, y=100.0, our_value=0.0,
                         enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=900.0)
    return region, [ResourcePoint(index=i, x=1000.0 + 50.0 * i, y=100.0, region=1) for i in range(points)]


def _expanding(expansion, tech_cap=0.0):
    return EconomyOrders(posture=Posture.EXPAND, allocation=ALLOCATION[Posture.EXPAND], tech_cap=tech_cap,
                         target_mix={Role.ARMOUR: 1.0}, expansion=list(expansion))


def _busy(unit_id, type_index, x=100.0):
    unit = _unit(unit_id, type_index, x=x)
    unit.queued = 1
    return unit


def test_extractors_go_only_into_the_regions_of_the_expansion_plan():
    region, points = _outpost(2)
    units = [_unit(1, 1)]
    planned = Economy(_Session(), CATALOGUE)
    planned.points = points
    out = planned.decide(_view(units, 30000, 2000.0, regions=[region]), _expanding([1]), [])
    assert _made(out) == ["extractor"] and out[0].x >= 1000.0
    unplanned = Economy(_Session(), CATALOGUE)
    unplanned.points = points
    assert _made(unplanned.decide(_view(units, 30000, 2000.0, regions=[region]), _expanding([]), [])) == []


def test_the_builder_target_grows_with_open_ground_and_stops_at_the_most_kept():
    """Two builders busy and a free factory: with plenty of ground open in the plan a third is ordered, and none is when the target is held fixed or when MAX_BUILDERS already stand."""
    region, points = _outpost(POINTS_PER_BUILDER * 2)
    busy = [_busy(1, 1), _busy(2, 1, x=150.0), _unit(10, 2)]
    scaled = Economy(_Session(), CATALOGUE)
    scaled.points = points
    assert "builder" in _made(scaled.decide(_view(busy, 30000, 2000.0, regions=[region]), _expanding([1]), []))

    fixed = Economy(_Session(), CATALOGUE, Options(builders="fixed"))
    fixed.points = points
    assert "builder" not in _made(fixed.decide(_view(busy, 30000, 2000.0, regions=[region]), _expanding([1]), []))

    region, points = _outpost(POINTS_PER_BUILDER * 20)
    crowd = [_busy(i, 1, x=100.0 + i) for i in range(1, MAX_BUILDERS + 1)] + [_unit(10, 2)]
    full = Economy(_Session(), CATALOGUE)
    full.points = points
    assert "builder" not in _made(full.decide(_view(crowd, 30000, 2000.0, regions=[region]), _expanding([1]), []))


def test_the_factory_tier_is_bought_from_the_technology_fund_once_it_holds_the_price():
    """The cap is credits per minute: after one period nothing has been put aside, and a factory raise costing a little over what one minute and some puts aside waits until the fund holds it."""
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2, upgrade_price=900), _unit(11, 2, x=300.0)]
    orders = EconomyOrders(posture=Posture.ARM, allocation=ALLOCATION[Posture.ARM], tech_cap=600.0,
                           target_mix={Role.ARMOUR: 1.0})
    economy = Economy(_Session(), CATALOGUE, Options(tech="factory"))
    assert "upgrade" not in _made(economy.decide(_view(units, 30000, 2500.0, income=40.0), orders, []))
    assert "upgrade" not in _made(economy.decide(_view(units, 90000, 2500.0, income=40.0), orders, []))
    assert "upgrade" in _made(economy.decide(_view(units, 120000, 2500.0, income=40.0), orders, []))
    assert economy.tech_fund < 900.0


def test_the_technology_mode_decides_which_raises_the_fund_pays_for():
    """An extractor's second tier is income, bought by the economy under `factory`; under `all` it waits for the fund like any raise; under `off` the cap is only a ban at zero."""
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(20, 5, upgrade_price=800)]
    orders = _teched({Role.ARMOUR: 1.0})
    for mode, raised in (("factory", True), ("all", False), ("off", True)):
        economy = Economy(_Session(), CATALOGUE, Options(tech=mode))
        assert ("upgrade" in _made(economy.decide(_view(units, 30000, 2000.0), orders, []))) == raised, mode


def test_without_saving_a_raise_holds_nothing_back_from_the_army():
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(20, 5, upgrade_price=1400)]
    saving = Economy(_Session(), CATALOGUE)
    assert _made(saving.decide(_view(units, 30000, 1000.0), _orders({Role.ARMOUR: 1.0}), [])) == []
    spending = Economy(_Session(), CATALOGUE, Options(saving=False))
    assert _made(spending.decide(_view(units, 30000, 1000.0), _orders({Role.ARMOUR: 1.0}), [])) == ["tank"]


def test_a_builder_lost_before_any_factory_is_replaced_from_the_command_centre():
    """With no factory standing and no builder left, only the command centre can make a builder, and does; without it nothing can."""
    base = [_unit(40, 8), _unit(20, 5)]
    economy = Economy(_Session(), CATALOGUE)
    out = economy.decide(_view(base, 60000, 2000.0), _orders({Role.ARMOUR: 1.0}), [])
    assert _made(out) == ["builder"] and out[0].producer == 40
    without = Economy(_Session(), CATALOGUE, Options(hq=False))
    assert _made(without.decide(_view(base, 60000, 2000.0), _orders({Role.ARMOUR: 1.0}), [])) == []


def test_a_builder_is_ordered_from_what_its_producer_makes_and_not_the_cheapest_anywhere():
    """A scenario creature reads as a builder and is cheaper than one, but no menu here makes it; the command centre is asked for the builder its menu holds."""
    types = TYPES + [UnitType(index=len(TYPES), name="spore", lookup="spore", price=100, tech=1, building=False,
                              builder=True, movement="LAND")]
    economy = Economy(_Session(), _Catalogue(types))
    made = [types[p.type_index].lookup for p in economy.decide(_view([_unit(40, 8)], 60000, 2000.0),
                                                              _orders({Role.ARMOUR: 1.0}), [])]
    assert made == ["builder"]


def test_open_ground_in_the_plan_is_claimed_before_the_army_when_switched_so():
    region, points = _outpost(2)
    units = [_unit(1, 1), _busy(2, 1, x=150.0), _unit(10, 2)]
    first = Economy(_Session(), CATALOGUE)
    first.points = points
    assert _made(first.decide(_view(units, 30000, 800.0, regions=[region]), _expanding([1]), [])) == ["extractor"]
    last = Economy(_Session(), CATALOGUE, Options(expand_first=False))
    last.points = points
    assert _made(last.decide(_view(units, 30000, 800.0, regions=[region]), _expanding([1]), [])) == ["tank"]


def test_a_point_whose_placements_never_stand_is_given_up_on():
    """Two placements on a point that never stood and it is not offered a third time; without giving up it is offered for ever."""
    region, points = _outpost(1)
    units = [_unit(1, 1)]
    for retry, third in ((True, []), (False, ["extractor"])):
        economy = Economy(_Session(), CATALOGUE, Options(retry=retry))
        economy.points = points
        made = [_made(economy.decide(_view(units, 30000 + k * PLACEMENT_GRACE_MS, 2000.0, regions=[region]),
                                     _expanding([1]), [])) for k in range(3)]
        assert made[:2] == [["extractor"], ["extractor"]] and made[2] == third, retry


def test_each_factory_placement_tries_fresh_ground():
    economy = Economy(_Session(), CATALOGUE)
    offsets = {tuple(round(v) for v in economy._factory_offset()) for _ in range(6)}
    assert len(offsets) == 6
    fixed = Economy(_Session(), CATALOGUE, Options(retry=False))
    assert {fixed._factory_offset() for _ in range(3)} == {(120.0, 120.0)}


def test_the_ledger_counts_unordered_factories_and_the_openings_milestones():
    units = [_unit(1, 1), _unit(10, 2), _unit(11, 2, x=300.0), _unit(20, 5), _unit(21, 5, x=200.0)]
    economy = Economy(_Session(), CATALOGUE)
    economy.decide(_view(units, 30000, 0.0), _orders({Role.ARMOUR: 1.0}), [])
    ledger = economy.ledger.summary()
    assert ledger["factory_unordered"] == 1.0 and ledger["credits_mean"] == 0.0
    assert ledger["first_factory_s"] == ledger["second_factory_s"] == ledger["second_extractor_s"] == 30.0


def test_a_match_that_lost_every_factory_raises_one_beside_home_before_anything_else():
    """Once a factory has stood, losing all of them makes the next one the first thing done, whatever the extractors say, and it goes up beside home; not switched to recover, the rule that waits for extractors holds it back."""
    units = [_unit(1, 1, x=900.0), _unit(40, 8)]
    economy = Economy(_Session(), CATALOGUE)
    economy.ledger.first_factory_ms = 60000
    out = economy.decide(_view(units, 300000, 2000.0), _orders({Role.ARMOUR: 1.0}), [])
    assert _made(out)[0] == "landFactory"
    assert abs(out[0].x - 100.0) < 400.0 and abs(out[0].y - 100.0) < 400.0
    without = Economy(_Session(), CATALOGUE, Options(recovery=False))
    without.ledger.first_factory_ms = 60000
    assert "landFactory" not in _made(without.decide(_view(units, 300000, 2000.0), _orders({Role.ARMOUR: 1.0}), []))


def test_the_factory_rebuild_takes_a_builder_off_its_errand_when_none_is_free():
    economy = Economy(_Session(), CATALOGUE)
    economy.ledger.first_factory_ms = 60000
    out = economy.decide(_view([_busy(1, 1), _busy(2, 1, x=800.0)], 300000, 2000.0), _orders({Role.ARMOUR: 1.0}), [])
    assert _made(out)[0] == "landFactory" and out[0].producer == 1


def test_a_builder_standing_still_away_from_its_site_is_given_something_else():
    """A builder with an errand that has not moved for STUCK_MS is freed when it stands far from where it was sent, and left alone when it stands at the site, where it is raising the building."""
    lost, working = _busy(1, 1), _busy(2, 1, x=400.0)

    def watched(options):
        economy = Economy(_Session(), CATALOGUE, options)
        economy.sites = {1: (2000.0, 100.0), 2: (420.0, 100.0)}
        economy.issued_at[("builder", 1)] = economy.issued_at[("builder", 2)] = 0
        economy._track_builders(_view([lost, working], 1000, 0.0), 1000)
        economy._track_builders(_view([lost, working], 1000 + STUCK_MS, 0.0), 1000 + STUCK_MS)
        return economy

    economy = watched(Options())
    assert economy._available(lost, 1000 + STUCK_MS) and not economy._available(working, 1000 + STUCK_MS)
    assert not watched(Options(recovery=False))._available(lost, 1000 + STUCK_MS)


def test_a_builder_just_given_a_placement_is_not_free_before_its_queue_shows_it():
    """The game takes an order on a later frame than it is sent, so a builder sent to place something reads as having nothing queued for a period; it is not handed another errand meanwhile, and once the window has passed with nothing queued it is free again."""
    economy = Economy(_Session(), CATALOGUE)
    builder = _unit(1, 1)
    economy._send(builder, 400.0, 100.0, 10000)
    assert not economy._available(builder, 12000)
    assert economy._available(builder, 10000 + PLACEMENT_PENDING_MS)
    assert ("builder", 1) not in economy.issued_at


def test_an_extractor_that_never_reached_its_builder_gives_its_point_back():
    """A builder sent to a point that comes free again with nothing standing there releases the point at once, so it is offered again rather than held for the placement's grace; a point something stood on stays with the record of that."""
    economy = Economy(_Session(), CATALOGUE)
    builder, other = _unit(1, 1), _unit(2, 1)
    economy.issued_at[7] = economy.issued_at[8] = 10000
    economy._send(builder, 400.0, 100.0, 10000)
    economy.extracting[1] = 7
    economy._send(other, 500.0, 100.0, 10000)
    economy.extracting[2] = 8
    economy.stood.add(8)
    later = 10000 + PLACEMENT_PENDING_MS
    assert economy._available(builder, later) and economy._available(other, later)
    assert 7 not in economy.issued_at and economy.issued_at[8] == 10000
    assert economy.failures == {7: 1}


def test_the_ledger_says_where_the_credits_went_and_how_long_nothing_could_be_made():
    """What an order commits is booked to what it is for; once a factory has stood, the periods without one are counted, and so are the periods the treasury sat above the banking line."""
    economy = Economy(_Session(), CATALOGUE)
    mix = _orders({Role.ARMOUR: 1.0})
    units = [_unit(1, 1), _unit(10, 2), _unit(20, 5, upgrade_price=1400)]
    economy.decide(_view(units, 30000, 2000.0), mix, [])
    assert economy.ledger.summary()["spent"] == {"builders": 500, "upgrades": 1400}

    economy.decide(_view([_unit(1, 1)], 30000 + FACTORY_BUSY_MS, BANKED_CREDITS + 100.0), mix, [])
    ledger = economy.ledger.summary()
    assert ledger["factoryless"] == 0.5 and ledger["banked"] == 0.5


class _Directed(_Economy):
    """The build order with its first few choices named in advance, by the kind and type of the offer, and the judge answering after them; every board it was asked about is kept."""

    def __init__(self, directions, **kwargs):
        super().__init__(_Session(), CATALOGUE, replace(Options(), choose=False), **kwargs)
        self.directions = list(directions)
        self.asked = []

    def _choose(self, state, mask, slots, board=None):
        self.asked.append((state, mask, list(slots)))
        if self.directions:
            kind, type_index = self.directions.pop(0)
            return next(index for index, offer in enumerate(slots)
                        if offer is not None and offer.kind == kind and offer.type_index == type_index)
        return super()._choose(state, mask, slots, board)


def test_what_is_chosen_is_carried_out_in_the_order_it_was_chosen_and_stopping_ends_the_period():
    """The judge alone would make a tank from each factory; told to take the builder first, the builder goes to the first factory and the tank to the second, and the period ends when the judge stops."""
    from rwintel.control.policy.encoding import Investment

    economy = _Directed([(Investment.BUILDER, 1)])
    units = [_unit(1, 1), _unit(10, 2), _unit(11, 2, x=300.0)]
    out = economy.decide(_view(units, 30000, 2000.0), _orders({Role.ARMOUR: 1.0}), [])
    assert _made(out) == ["builder", "tank"]
    assert out[0].producer == 10 and out[1].producer == 11
    assert len(economy.asked) == 3


def test_choosing_what_the_treasury_cannot_pay_for_holds_its_price_back_from_what_comes_after():
    """A tank chosen with 300 credits is waited for: its factory makes nothing else, its price is held back from every later choice of the period, and the next board says so."""
    from rwintel.control.policy.encoding import ECONOMIC_CONTEXT, TREASURY_SCALE, Investment

    economy = _Directed([(Investment.UNIT, 0)])
    units = [_unit(1, 1), _unit(2, 1, x=150.0), _unit(10, 2), _unit(11, 2, x=300.0), _unit(30, 0)]
    assert economy.decide(_view(units, 30000, 300.0), _orders({Role.ARMOUR: 1.0}), []) == []
    reserve = ECONOMIC_CONTEXT.index("reserve")
    assert economy.asked[0][0][reserve] == 0.0
    assert abs(economy.asked[1][0][reserve] - 350.0 / TREASURY_SCALE) < 1e-9
    assert economy.ledger.unordered_factory_periods == 2


def test_a_board_offers_what_can_be_done_now_and_nothing_else():
    """A factory at the unit cap offers no unit, a builder with an errand places nothing, and stopping is always there; the mask allows exactly the slots that hold an offer."""
    from rwintel.control.policy.encoding import Investment

    economy = _Directed([])
    units = [_busy(1, 1), _unit(10, 2)]
    economy.decide(_view(units, 30000, 2000.0, unit_cap=10, units_counted=10), _orders({Role.ARMOUR: 1.0}), [])
    _, mask, slots = economy.asked[0]
    kinds = {offer.kind for offer in slots if offer is not None}
    assert kinds == {Investment.STOP, Investment.BUILDER}
    assert mask == [1.0 if offer is not None else 0.0 for offer in slots] and mask[0] == 1.0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
