"""Classification by capability, squads of one domain, the operational layer's means, ground nobody can reach, a lift asked for from outside the chain, and the economy carrying a builder across, held to.

The map is the two islands of `test_reach`: land, a strait of water, land. Region 0 is on the left island and region 1 on the right.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.intervention import Interface
from rwintel.control.policy.catalogue import Catalogue
from rwintel.control.policy.contracts import (
    ALLOCATION,
    Doctrine,
    Domain,
    EconomyOrders,
    Function,
    OperationsOrders,
    Posture,
    Role,
    SquadRecord,
    TaskContract,
)
from rwintel.control.policy.economy import Economy, ResourcePoint
from rwintel.control.policy.encoding import (
    GLOBAL_SIZE,
    REGION_FEATURES,
    REGION_SIZE,
    SQUAD_SIZE,
    TRANSPORT_FEATURES,
    TRANSPORT_SIZE,
    means_of,
    task_of,
)
from rwintel.control.policy.judgement import OperationsJudge
from rwintel.control.policy.logistics import Logistics
from rwintel.control.policy.operations import WALK, Operations
from rwintel.control.policy.options import Options
from rwintel.control.policy.organisation import Organisation
from rwintel.control.policy.reach import Reach
from rwintel.control.policy.strategy import Strategy
from rwintel.control.policy.view import build as build_view
from rwintel.control.session import UnitType
from rwintel.wire import (
    BLOCK_REGIONS,
    BLOCK_UNITS,
    REGION_SLOTS,
    SQUAD_SLOTS,
    Action,
    CargoKind,
    Lift,
    Observation,
    RegionState,
    Stance,
    Task,
    UnitState,
)

from test_reach import REGIONS, _at, _passage

TANK, BUILDER, HOVERCRAFT, HOVER_TANK, GUN_BOAT, FACTORY, EXTRACTOR, HQ, HELICOPTER, BUG, GUNSHIP = range(11)

TYPES = [
    UnitType(index=TANK, name="tank", lookup="tank", price=350, tech=1, building=False, builder=False,
             movement="LAND", can_attack=True, range=130.0, speed=66.0),
    UnitType(index=BUILDER, name="builder", lookup="builder", price=500, tech=1, building=False, builder=True,
             movement="LAND", range=30.0, speed=48.0, menu=(FACTORY, EXTRACTOR)),
    UnitType(index=HOVERCRAFT, name="hovercraft", lookup="hovercraft", price=1000, tech=1, building=False,
             builder=False, movement="HOVER", range=30.0, speed=54.0, capacity=4, carries=(TANK, BUILDER)),
    UnitType(index=HOVER_TANK, name="hoverTank", lookup="hoverTank", price=450, tech=1, building=False, builder=False,
             movement="HOVER", can_attack=True, range=140.0, speed=60.0),
    UnitType(index=GUN_BOAT, name="gunBoat", lookup="gunBoat", price=300, tech=1, building=False, builder=False,
             movement="WATER", can_attack=True, range=120.0, speed=90.0),
    UnitType(index=FACTORY, name="landFactory", lookup="landFactory", price=700, tech=1, building=True, builder=False,
             movement="NONE", menu=(BUILDER, TANK, HOVERCRAFT)),
    UnitType(index=EXTRACTOR, name="extractor", lookup="extractor", price=700, tech=1, building=True, builder=False,
             movement="NONE", extractor=True),
    UnitType(index=HQ, name="commandCenter", lookup="commandCenter", price=3000, tech=1, building=True, builder=False,
             movement="NONE", can_attack=True, range=200.0, menu=(BUILDER,)),
    UnitType(index=HELICOPTER, name="helicopter", lookup="helicopter", price=700, tech=1, building=False,
             builder=False, movement="AIR", can_attack=True, range=130.0, speed=108.0),
    UnitType(index=BUG, name="bug", lookup="bug", price=300, tech=1, building=False, builder=False, movement="LAND",
             can_attack=True, range=9.0, speed=90.0),
    UnitType(index=GUNSHIP, name="gunship", lookup="gunship", price=5000, tech=1, building=False, builder=False,
             movement="HOVER", can_attack=True, range=320.0, speed=24.0, capacity=5, carries=(TANK,)),
]

CATALOGUE = Catalogue.of_types(TYPES)
LEFT, RIGHT = REGIONS


def _unit(unit_id, type_index, at, squad=0xFFFF, hostile=0, queued=0):
    return UnitState(id=unit_id, squad=squad, type_index=type_index, x=at[0], y=at[1], health=100.0, max_health=100.0,
                     built=255, order=255, queued=queued, target=0, stance=5, hostile=hostile, since_hit_ms=9999)


def _regions(held=(1, 0)):
    return [RegionState(id=r.id, resources=1, held_by_us=held[r.id], held_by_enemy=0, x=r.x, y=r.y, our_value=0.0,
                        enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=abs(r.x - LEFT.x)) for r in REGIONS]


def _view(units, t=30000, regions=None):
    observation = Observation(frame=1, game_time_ms=t, episode=1, blocks=BLOCK_REGIONS | BLOCK_UNITS, slot=0,
                              credits=2000.0, income=10.0, units=len(units), unit_cap=100, under_construction=0,
                              killed_units=0, killed_buildings=0, lost_units=0, lost_buildings=0,
                              regions=_regions() if regions is None else regions, unit_states=list(units))
    return build_view(observation, CATALOGUE, 0)


def _logistics(view):
    logistics = Logistics(CATALOGUE)
    logistics.reach = Reach(_passage(), REGIONS)
    logistics.update(view)
    return logistics


def test_roles_are_read_off_what_a_type_does():
    """Slower than the line tank is not fast whatever it moves on, a fast warship or aircraft is not a raider, a transport that can fight is counted as what it fights as, and one that cannot is a transport."""
    role = CATALOGUE.role
    assert [role(TANK), role(HOVER_TANK), role(BUG), role(GUN_BOAT), role(HELICOPTER)] == [Role.ARMOUR] * 2 + [Role.FAST] + [Role.ARMOUR] * 2
    assert (role(HOVERCRAFT), role(GUNSHIP), role(BUILDER)) == (Role.TRANSPORT, Role.ARTILLERY, Role.BUILDER)
    assert [CATALOGUE.domain(i) for i in (TANK, HOVER_TANK, GUN_BOAT, HELICOPTER, FACTORY)] == [
        Domain.GROUND, Domain.AMPHIBIOUS, Domain.NAVAL, Domain.AIR, Domain.STATIC]
    assert CATALOGUE.has(GUNSHIP, Function.TRANSPORT) and CATALOGUE.has(FACTORY, Function.PRODUCER)


def test_factories_and_headquarters_are_found_by_their_menus():
    """A factory is what a builder places and makes something that fights; the command centre makes a builder and no builder places it."""
    assert CATALOGUE.factories == {FACTORY}
    assert CATALOGUE.headquarters == {HQ}
    assert [CATALOGUE.doctrine_for(i) for i in (GUN_BOAT, HELICOPTER, BUG, HOVERCRAFT)] == [
        Doctrine.FLEET, Doctrine.AIRWING, Doctrine.RAID, None]


def test_a_squad_is_raised_in_one_domain_and_takes_only_that_domain():
    organisation = Organisation(None, CATALOGUE, Options(vanguards=False))
    units = [_unit(1, TANK, _at(1, 1)), _unit(2, TANK, _at(1, 2)),
             _unit(3, HOVER_TANK, _at(2, 1)), _unit(4, HOVER_TANK, _at(2, 2)), _unit(5, HOVER_TANK, _at(2, 3))]
    organisation.update(_view(units), [])
    squads = sorted(organisation.squads.values(), key=lambda s: s.id)
    assert [(s.doctrine, s.domain) for s in squads] == [(Doctrine.GARRISON, Domain.AMPHIBIOUS),
                                                        (Doctrine.GARRISON, Domain.GROUND)]
    assert sorted(squads[0].members) == [3, 4, 5] and sorted(squads[1].members) == [1, 2]


def _operations(view, logistics):
    operations = Operations(None, CATALOGUE, logistics=logistics)
    operations._view = view
    operations._orders = OperationsOrders(posture=Posture.EXPAND, priorities={0: 0.2, 1: 1.0}, offensive=True,
                                          loss_allowance=1000.0)
    operations._reach, operations._claimed = {}, {}
    return operations


def test_a_squad_may_be_sent_where_it_can_be_carried_and_two_squads_do_not_share_a_transport():
    units = [_unit(1, TANK, _at(1, 4), squad=0), _unit(2, TANK, _at(2, 4), squad=1), _unit(20, HOVERCRAFT, _at(6, 1))]
    view = _view(units)
    operations = _operations(view, _logistics(view))
    first, second = SquadRecord(id=0, doctrine=Doctrine.VANGUARD, members=[1]), SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[2])
    assert operations.region_mask(view, first)[:2] == [1.0, 1.0]
    assert operations.means(view, first, LEFT.id) == WALK
    assert operations.means(view, first, RIGHT.id) == 0
    assert operations.means(view, second, RIGHT.id) == WALK
    # Without a transport the far island is off the table, and wanting it is a shortfall.
    stranded = _view(units[:2])
    lonely = _operations(stranded, _logistics(stranded))
    assert lonely.region_mask(stranded, first)[:2] == [1.0, 0.0]
    assert lonely.logistics.shortfall.passengers == {TANK}


def test_a_fleet_kept_from_a_wanted_region_asks_for_no_transport():
    """No transport type loads a gun boat, so a fleet that cannot get to ground the strategic layer wants is not a transport to build, and leaves the shortfall to the squads one could carry."""
    inland = SimpleNamespace(id=2, x=_at(0, 2)[0], y=_at(0, 2)[1])
    units = [_unit(1, GUN_BOAT, _at(5, 1), squad=0)]
    regions = _regions() + [RegionState(id=2, resources=1, held_by_us=0, held_by_enemy=0, x=inland.x, y=inland.y,
                                        our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=0.0)]
    view = _view(units, regions=regions)
    assert view.region(2) is not None
    logistics = Logistics(CATALOGUE)
    logistics.reach = Reach(_passage(), REGIONS + [inland])
    logistics.update(view)
    operations = _operations(view, logistics)
    operations._orders.priorities = {0: 0.2, 1: 0.2, 2: 1.0}
    fleet = SquadRecord(id=0, doctrine=Doctrine.FLEET, members=[1])
    reachable = operations.reach(view, fleet)
    assert 2 not in reachable.walk and 2 not in reachable.lift
    assert logistics.shortfall.count == 0 and not logistics.could_carry([GUN_BOAT])
    assert logistics.could_carry([TANK]) and not logistics.could_carry([TANK, GUN_BOAT])


def test_the_plans_open_in_a_region_say_how_the_squad_gets_there_and_the_judge_takes_the_transport_over_water():
    """Walking is the only means to the near island and the hovercraft in slot 0 the only one to the far one; the board shows the transport and what can be reached, a squad sent across is given the slot, and a second squad finds the slot taken."""
    units = [_unit(1, TANK, _at(1, 4), squad=0), _unit(2, TANK, _at(2, 4), squad=1), _unit(20, HOVERCRAFT, _at(6, 1))]
    view = _view(units)
    operations = _operations(view, _logistics(view))
    first, second = SquadRecord(id=0, doctrine=Doctrine.VANGUARD, members=[1]), SquadRecord(id=1, doctrine=Doctrine.VANGUARD, members=[2])
    masks = operations.plan_masks(view, first)
    open_near = {(task_of(p), means_of(p)) for p, allowed in enumerate(masks[LEFT.id]) if allowed}
    open_far = {(task_of(p), means_of(p)) for p, allowed in enumerate(masks[RIGHT.id]) if allowed}
    assert open_near == {(int(Task.ATTACK), WALK), (int(Task.ENCIRCLE), WALK)}
    assert open_far == {(int(Task.ATTACK), 0), (int(Task.ENCIRCLE), 0)}
    state = operations.state(first)
    start = GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + SQUAD_SLOTS * SQUAD_SIZE
    transport = dict(zip(TRANSPORT_FEATURES, state[start:start + TRANSPORT_SIZE]))
    assert (transport["valid"], transport["free"], transport["hover"], transport["carries"]) == (1.0, 1.0, 1.0, 1.0)
    far = dict(zip(REGION_FEATURES, state[GLOBAL_SIZE + RIGHT.id * REGION_SIZE:GLOBAL_SIZE + (RIGHT.id + 1) * REGION_SIZE]))
    assert (far["walkable"], far["carriable"]) == (0.0, 1.0)
    # With the far island the only region left open, the plan the rule takes there is the transport.
    judge = OperationsJudge()
    only_far = [1.0 if slot == RIGHT.id else 0.0 for slot in range(REGION_SLOTS)]
    operations.judge = SimpleNamespace(choose=lambda state, slot, regions, masks: judge.choose(state, slot, only_far, masks))
    task, region, means = operations._target(view, first)
    assert (region.id, means) == (RIGHT.id, 0) and task in (Task.ATTACK, Task.ENCIRCLE)
    assert not any(operations.plan_masks(view, second)[RIGHT.id])


def test_a_squad_already_across_walks_on_rather_than_being_lifted_again():
    units = [_unit(1, TANK, _at(10, 4), squad=0), _unit(20, HOVERCRAFT, _at(6, 1))]
    view = _view(units)
    operations = _operations(view, _logistics(view))
    squad = SquadRecord(id=0, doctrine=Doctrine.VANGUARD, members=[1])
    squad.contract = TaskContract(squad=0, task=Task.ATTACK, target_region=RIGHT.id, stance=Stance.AGGRESSIVE,
                                  cost_budget=500.0, deadline_ms=0, issued_at_ms=0, means=0)
    assert operations.means(view, squad, RIGHT.id) == WALK


def test_ground_no_army_could_reach_is_neither_planned_onto_nor_given_a_priority():
    board = [RegionState(id=r, resources=2, held_by_us=1 if r == 0 else 0, held_by_enemy=0, x=100.0 * r, y=0.0,
                         our_value=0.0, enemy_value=0.0, enemy_seen_at_ms=0, distance_from_home=100.0 * r)
             for r in range(3)]
    session = SimpleNamespace(regions=[])
    strategy = Strategy(session, CATALOGUE)
    strategy.reachable = {0, 1}
    report = SimpleNamespace(income=10.0, credits=0.0, military_value=0.0, enemy_value=0.0, enemy_military=0.0,
                             held=1, enemy_held=0, lost_regions=0, enemy_bases=0)
    economy, operations = strategy.decide(report, board, 0, home=board[0])
    assert 2 not in economy.expansion and 1 in economy.expansion
    assert operations.priorities.get(2, 0.0) == 0.0


def test_a_lift_asked_for_from_outside_goes_out_once_as_that_commanders():
    units = [_unit(1, TANK, _at(1, 4), squad=0), _unit(20, HOVERCRAFT, _at(6, 1))]
    view = _view(units)
    logistics = _logistics(view)
    interface = Interface(organisation=SimpleNamespace(logistics=logistics, release=lambda slot: None))
    squads = [SquadRecord(id=0, doctrine=Doctrine.VANGUARD, members=[1])]
    interface.take(0)
    chain = Action(lifts=[Lift(lift=99, transports=[20], cargo_kind=CargoKind.SQUAD, cargo=[0])])
    interface.intervene(chain, view, squads, view.observation)
    assert chain.lifts == []
    interface.lift(0, 0, RIGHT.id)
    action = Action()
    interface.intervene(action, view, squads, view.observation)
    assert [(row.cargo, row.override) for row in action.lifts] == [([0], True)]
    interface.intervene(Action(), view, squads, view.observation)
    assert len(logistics.rows()) == 1


def _economy(view, logistics):
    session = SimpleNamespace(regions=[], map_content=None)
    economy = Economy(session, CATALOGUE, Options())
    economy.logistics = logistics
    economy.points = [ResourcePoint(index=0, x=_at(10, 2)[0], y=_at(10, 2)[1], region=RIGHT.id)]
    return economy


def _expanding():
    return EconomyOrders(posture=Posture.EXPAND, allocation=ALLOCATION[Posture.EXPAND], tech_cap=0.0,
                         target_mix={Role.ARMOUR: 1.0}, expansion=[RIGHT.id])


def test_ground_only_reachable_by_water_is_taken_by_carrying_a_builder_across_first():
    units = [_unit(5, BUILDER, _at(2, 2)), _unit(20, HOVERCRAFT, _at(6, 1))]
    view = _view(units)
    logistics = _logistics(view)
    economy = _economy(view, logistics)
    made = economy.decide(view, _expanding(), [])
    assert [p for p in made if p.type_index == EXTRACTOR] == []
    row, = logistics.rows()
    assert (row.cargo_kind, row.cargo, row.drop_region) == (CargoKind.UNITS, [5], RIGHT.id)
    # The builder is being carried, so nothing else is asked of it.
    assert not economy._available(view.builders[0].unit, 31000)


def test_a_transport_is_offered_only_while_one_is_wanted():
    units = [_unit(5, BUILDER, _at(2, 2), queued=1), _unit(6, BUILDER, _at(2, 3), queued=1), _unit(30, FACTORY, _at(1, 1))]
    view = _view(units)
    logistics = _logistics(view)
    economy = _economy(view, logistics)
    economy.points = []
    orders = EconomyOrders(posture=Posture.ARM, allocation=ALLOCATION[Posture.ARM], tech_cap=0.0,
                           target_mix={Role.ARMOUR: 1.0})
    assert "hovercraft" not in [TYPES[p.type_index].lookup for p in economy.decide(view, orders, [])]
    logistics.want([TANK])
    later = _view(units, t=40000)
    assert "hovercraft" in [TYPES[p.type_index].lookup for p in _economy(later, logistics).decide(later, orders, [])]


def test_warships_are_built_once_an_enemy_warship_has_been_seen():
    quiet = _view([_unit(30, FACTORY, _at(1, 1))])
    economy = _economy(quiet, _logistics(quiet))
    assert not economy._water_front(quiet)
    assert economy._water_front(_view([_unit(30, FACTORY, _at(1, 1)), _unit(90, GUN_BOAT, _at(6, 2), hostile=1)]))
    assert economy._water_front(quiet)
