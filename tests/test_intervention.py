"""What the intervention interface promises, held to.

Two things are being pinned. The ownership rule, which is the design's answer to every question about a human and a machine commanding the same units: exactly one commander per squad, transfers explicit and one-way, and the command chain silent about anything it does not hold. And the squad cap, which is not an efficiency limit but the width of the observation's squad block — a ninth squad is not surplus, it is invisible, so a commander that wants one of its own has to borrow a slot rather than pick a number.

The tests build the pieces directly rather than through a session, because what is under test is the amendment of one action and none of it needs a game.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.intervention import Interface, Intervention, Kind
from rwintel.control.intruder import Intruder
from rwintel.control.policy.contracts import Doctrine, SQUAD_CAP, SquadRecord, TaskContract
from rwintel.control.policy.organisation import Organisation
from rwintel.wire import (
    Action,
    BLOCK_EVENTS,
    BLOCK_SQUADS,
    BLOCK_UNITS,
    Commander,
    Contract,
    Deviation,
    Observation,
    SquadDeviation,
    Stance,
    Task,
    UnitState,
)


#: A frame carrying every block but the region one. The intruder decides whether to interfere on the operational period, which is the frame the region block rides, so this is how a test drains what it has already asked for without drawing anything new.
QUIET = BLOCK_SQUADS | BLOCK_UNITS | BLOCK_EVENTS


def _observation(**overrides) -> Observation:
    fields = dict(frame=1, game_time_ms=10000, episode=1, blocks=0xF, slot=0, credits=1000.0,
                  income=10.0, units=4, unit_cap=100, under_construction=0, killed_units=0,
                  killed_buildings=0, lost_units=0, lost_buildings=0)
    fields.update(overrides)
    return Observation(**fields)


def _unit(unit_id: int, squad: int = 0xFFFF) -> UnitState:
    return UnitState(id=unit_id, squad=squad, type_index=1, x=0.0, y=0.0, health=100.0,
                     max_health=100.0, built=255, order=255, queued=0, target=0, stance=5,
                     hostile=0, since_hit_ms=9999)


def _squad(squad_id: int = 0, members=(1, 2, 3)) -> SquadRecord:
    return SquadRecord(id=squad_id, doctrine=Doctrine.VANGUARD, members=list(members), value=1000.0,
                       formed_value=1000.0)


def _organisation(chain_squads=(0,)) -> Organisation:
    """An organisation layer that has already raised the squads the chain is running, so that a slot it lends out is one of the remaining ones. Constructed without a session because nothing under test here reaches for one."""
    organisation = Organisation(None, None)
    for squad_id in chain_squads:
        organisation.squads[squad_id] = _squad(squad_id)
        organisation.free_ids.remove(squad_id)
    return organisation


def _chain_action(squad_id: int = 0) -> Action:
    """An action of the sort the command chain produces on its own, so that what an intervention removes from one can be seen."""
    return Action(
        contracts=[Contract(squad=squad_id, task=Task.ATTACK, target_region=4)],
        deviations=[SquadDeviation(squad=squad_id, deviation=Deviation.KITE)],
    )


def test_taking_a_squad_silences_the_chain_about_it_and_states_who_holds_it():
    interface = Interface()
    interface.take(0, int(Commander.OPERATIONS | Commander.TACTICS))
    action = _chain_action()
    applied = interface.intervene(action, None, [_squad()], _observation())

    assert [i.kind for i in applied] == [Kind.TAKE]
    assert action.contracts == [] and action.deviations == []
    assert [(row.squad, int(row.commander), row.units) for row in action.squads] == [
        (0, int(Commander.OPERATIONS | Commander.TACTICS), [1, 2, 3])]


def test_taking_only_the_operational_command_leaves_the_tactical_layer_fighting():
    """The design calls this the most useful arrangement of the two: a person writes the errand and the machine carries it out. It only exists if taking the operational command does not also take the departures away."""
    interface = Interface()
    interface.take(0, int(Commander.OPERATIONS))
    interface.intervene(_chain_action(), None, [_squad()], _observation())

    action = _chain_action()
    interface.intervene(action, None, [_squad()], _observation(game_time_ms=12000))
    assert action.contracts == []
    assert [row.deviation for row in action.deviations] == [Deviation.KITE]


def test_a_holders_contract_is_marked_so_the_game_will_accept_it():
    interface = Interface()
    interface.take(0)
    interface.write(0, Task.DEFEND, target_region=7, stance=Stance.GUARD_AREA, cost_budget=1500.0)
    action = _chain_action()
    interface.intervene(action, None, [_squad()], _observation())

    written = [row for row in action.contracts if row.override]
    assert len(written) == 1
    assert (written[0].squad, written[0].task, written[0].target_region) == (0, Task.DEFEND, 7)
    # The clock the losses are measured from is the moment of the decision, not whatever the chain last stamped.
    assert written[0].issued_at_ms == 10000


def test_an_errand_may_be_rewritten_without_taking_the_squad():
    """The lightest of the interventions and the one a person actually makes most: the squad stays under the tactical layer and only where it is going changes."""
    interface = Interface()
    interface.write(0, Task.RAID, target_region=2)
    action = _chain_action()
    interface.intervene(action, None, [_squad()], _observation())

    assert [(row.target_region, row.override) for row in action.contracts] == [(2, True)]


def test_giving_a_squad_back_hands_it_over_whole_and_to_the_machine():
    interface = Interface()
    interface.take(0)
    interface.intervene(Action(), None, [_squad()], _observation())
    interface.give_back(0)
    action = Action()
    applied = interface.intervene(action, None, [_squad()], _observation(game_time_ms=20000))

    assert [i.kind for i in applied] == [Kind.RETURN]
    assert [(row.squad, int(row.commander), row.units) for row in action.squads] == [
        (0, int(Commander.MACHINE), [1, 2, 3])]
    assert 0 not in interface.held


def test_detached_units_go_into_a_squad_borrowed_from_the_organisation_layer():
    organisation = _organisation()
    interface = Interface(organisation=organisation)
    interface.reassign(0, [2, 3])
    action = Action()
    applied = interface.intervene(action, None, [_squad()], _observation())

    assert [i.kind for i in applied] == [Kind.REASSIGN]
    slot = applied[0].into
    assert slot >= 0 and slot in organisation.reserved
    rosters = {row.squad: row.units for row in action.squads}
    assert rosters[0] == [1]
    assert sorted(rosters[slot]) == [2, 3]


def test_the_borrowed_slot_comes_back_when_the_squad_is_given_up():
    organisation = _organisation()
    interface = Interface(organisation=organisation)
    interface.reassign(0, [3])
    applied = interface.intervene(Action(), None, [_squad()], _observation())
    slot = applied[0].into

    interface.give_back(slot)
    action = Action()
    interface.intervene(action, None, [_squad(members=[1, 2])],
                        _observation(unit_states=[_unit(3, slot)]))
    # An empty roster is how a squad is retired, which releases its units to the reinforcement rule.
    assert [(row.squad, row.units) for row in action.squads] == [(slot, [])]
    assert slot not in organisation.reserved
    assert slot in organisation.free_ids


def test_the_cap_of_eight_is_kept_across_both_kinds_of_commander():
    """The chain and an outside commander draw from one set of slots. Counting them separately would be how a ninth squad comes to exist, and a ninth squad is not reported at all."""
    organisation = _organisation(chain_squads=())
    interface = Interface(organisation=organisation)
    borrowed = [organisation.reserve() for _ in range(SQUAD_CAP)]
    assert None not in borrowed
    assert organisation.reserve() is None

    interface.reassign(0, [2])
    applied = interface.intervene(Action(), None, [_squad()], _observation())
    assert applied == []


def test_the_intruder_gives_back_what_it_took_and_says_which_squads_it_touched():
    """What the log is for: the design requires the results of squads that were interfered with to be kept out of the learning signal, and a squad that was seized is one whose errand was not the one the layer chose."""
    intruder = Intruder(seed=1)
    squad = _squad()
    intruder._regions = [0, 1, 2]
    intruder._seize(squad, _observation())
    # A frame without the region block, so that draining what has been asked for does not also draw fresh interference and make the test about the random number generator.
    intruder.intervene(Action(), None, [squad], _observation(blocks=QUIET))

    assert 0 in intruder.holdings
    assert 0 in intruder.log.touched
    held_until = intruder.holdings[0].until_ms

    intruder._release(held_until + 1)
    action = Action()
    applied = intruder.intervene(action, None, [squad],
                                 _observation(blocks=QUIET, game_time_ms=held_until + 1))
    assert [i.kind for i in applied] == [Kind.RETURN]
    assert intruder.holdings == {}


def test_units_moved_into_a_squad_taint_it_as_surely_as_the_one_they_left():
    """A squad fighting an errand with a composition the operational layer did not give it says nothing about that layer either way."""
    organisation = _organisation()
    intruder = Intruder(organisation=organisation, seed=2)
    squad = _squad()
    intruder.interface.reassign(0, [3])
    intruder.intervene(Action(), None, [squad], _observation(blocks=QUIET))

    assert 0 in intruder.log.touched
    assert len(intruder.log.touched) == 2


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
