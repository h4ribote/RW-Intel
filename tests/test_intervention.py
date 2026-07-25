"""What the intervention interface promises, held to.

Two things are being pinned. The ownership rule, which is the design's answer to every question about a human and a machine commanding the same units: exactly one commander per squad, transfers explicit and one-way, and the command chain silent about anything it does not hold. And the squad cap, which is not an efficiency limit but the width of the observation's squad block — a ninth squad is not surplus, it is invisible, so a commander that wants one of its own has to borrow a slot rather than pick a number.

The tests build the pieces directly rather than through a session, because what is under test is the amendment of one action and none of it needs a game.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.intervention import Interface, Intervention, Kind
from rwintel.control.intruder import Intruder, Log
from rwintel.control.session import EpisodeSettings, Session
from rwintel.control.policy.contracts import Doctrine, Role, SQUAD_CAP, SquadRecord, TaskContract
from rwintel.control.policy.organisation import Organisation
from rwintel.control.policy.view import Sighting, WorldView
from rwintel.wire import (
    Action,
    BLOCK_EVENTS,
    BLOCK_SQUADS,
    BLOCK_UNITS,
    Commander,
    Contract,
    Deviation,
    NO_SQUAD,
    Observation,
    SquadAssignment,
    SquadDeviation,
    SquadState,
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


def _state(squad_id: int, commander: int = 0) -> SquadState:
    """One row of the observation's squad block: the game saying this number names a squad. Only the identity is filled in, since what the rest of it says about a squad's worth and position is nothing the rosters here depend on."""
    return SquadState(id=squad_id, commander=commander, units=0, value=0.0, formed_value=0.0,
                      x=0.0, y=0.0, spread=0.0, task_type=0, stance=0, target_region=0, status=0,
                      cost_budget=0.0, budget_share=0.0, deadline_ms=0, issued_at_ms=0, losses=0.0)


class _Catalogue:
    """What a unit is for, in as much as the organisation layer asks of it. Every unit in these fixtures is armour and every squad a vanguard, which is the one doctrine armour is taken for, so both questions have a constant answer and no unit table has to be built to give it."""

    def accepts(self, doctrine: Doctrine, type_index: int) -> bool:
        return doctrine is Doctrine.VANGUARD

    def doctrine_for(self, type_index: int) -> Doctrine:
        return Doctrine.VANGUARD


def _organisation(chain_squads=(0,)) -> Organisation:
    """An organisation layer that has already raised the squads the chain is running, so that a slot it lends out is one of the remaining ones. Constructed without a session because nothing under test here reaches for one."""
    organisation = Organisation(None, _Catalogue())
    for squad_id in chain_squads:
        organisation.squads[squad_id] = _squad(squad_id)
        organisation.free_ids.remove(squad_id)
    return organisation


def _view(observation: Observation) -> WorldView:
    """The cut of one frame the organisation layer reads. Built by hand rather than through the view builder because that one wants the game's own unit table to say what each unit is for, and every unit here is armour, which is what the vanguard the fixtures use is made of."""
    ours = [Sighting(unit=unit, kind=None, role=Role.ARMOUR) for unit in observation.unit_states]
    return WorldView(observation=observation, catalogue=_Catalogue(), ours=ours,
                     unassigned=[s for s in ours if s.unit.squad == NO_SQUAD])


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


def test_a_merge_moves_every_unit_and_retires_the_squad_it_emptied():
    """The one operation that ends a squad. Both halves have to be in the same action: the units into the squad that survives, and an empty roster for the one they left, which is how a retirement is said."""
    interface = Interface()
    interface.merge(0, 1)
    action = Action()
    applied = interface.intervene(action, None, [_squad(0, members=[1, 2, 3]), _squad(1, members=[4, 5])],
                                  _observation())

    assert [i.kind for i in applied] == [Kind.REASSIGN]
    # In this order, so the game has moved the units out before it is told the squad they were in is finished.
    assert [(row.squad, int(row.commander), list(row.units)) for row in action.squads] == [
        (1, int(Commander.MACHINE), [4, 5, 1, 2, 3]),
        (0, int(Commander.MACHINE), []),
    ]


def test_a_merge_takes_the_roster_as_it_stands_when_it_is_drained():
    """Why a merge is an intention about a squad and not a list of units: a unit finished production and was reinforced in between the moment it was asked for and the period that carries it out, and a list captured at the first of those would leave that unit behind in a squad that was supposed to cease existing."""
    interface = Interface()
    interface.merge(0, 1)
    action = Action()
    interface.intervene(action, None, [_squad(0, members=[1, 2, 3, 9]), _squad(1, members=[4])],
                        _observation())

    rosters = {row.squad: list(row.units) for row in action.squads}
    assert rosters[1] == [4, 1, 2, 3, 9]
    assert rosters[0] == []


def test_merging_a_squad_of_your_own_hands_its_slot_straight_back():
    organisation = _organisation()
    interface = Interface(organisation=organisation)
    interface.reassign(0, [3])
    slot = interface.intervene(Action(), None, [_squad()], _observation())[0].into
    assert slot in organisation.reserved

    interface.merge(slot, 0)
    action = Action()
    interface.intervene(action, None, [_squad(members=[1, 2])],
                        _observation(unit_states=[_unit(1, 0), _unit(2, 0), _unit(3, slot)]))

    rosters = {row.squad: list(row.units) for row in action.squads}
    assert rosters[0] == [1, 2, 3] and rosters[slot] == []
    assert slot not in interface.own and slot not in interface.held
    # Handed back in the same period rather than at the next sweep of spent squads, which is skipped altogether on a frame that carries no unit block.
    assert slot not in organisation.reserved and slot in organisation.free_ids


def test_a_merge_into_a_squad_that_does_not_exist_is_refused():
    """Stating a roster for a number nobody has raised does not fail; it creates a squad in a slot the organisation layer still counts as free, which that layer then hands to a second squad answering to the same number."""
    interface = Interface()
    interface.merge(0, 7)
    action = Action()

    assert interface.intervene(action, None, [_squad()], _observation()) == []
    assert action.squads == []


def test_a_squad_a_merge_has_just_dissolved_is_not_moved_into():
    """The same refusal, for a squad that stopped existing a moment ago rather than one that never existed. Both numbers are ones the organisation layer counts as free, so a roster stated for either raises a squad in a slot that layer will hand out again; the only difference is that this one was handed back by the row two lines above, and nothing but the action being built says so."""
    interface = Interface()
    interface.merge(0, 1)
    interface.reassign(2, [7], into=0)
    action = Action()
    applied = interface.intervene(
        action, None, [_squad(0, members=[1, 2]), _squad(1, members=[3]), _squad(2, members=[7, 8])],
        _observation())

    assert [i.kind for i in applied] == [Kind.REASSIGN]
    # The refused move states nothing at all: unit 7 stays in the squad it is in, and the last word about squad 0 is still the row that ended it.
    assert [(row.squad, list(row.units)) for row in action.squads] == [(1, [3, 1, 2]), (0, [])]


def test_a_squad_the_chain_has_just_disbanded_is_not_merged_into():
    """The organisation layer retires a squad by putting an empty roster for it on this same action, and by the time interventions are drained its record is already gone from the chain's list — while the observation, taken before any of that, still describes the squad. Reading the action is what tells a squad that is about to stop existing from one that is merely not in the records yet."""
    interface = Interface()
    interface.merge(3, 5)
    action = Action()
    action.squads.append(SquadAssignment(squad=5, commander=Commander.MACHINE, units=[]))

    assert interface.intervene(action, None, [_squad(3, members=[1, 2])],
                               _observation(squads=[_state(5)])) == []
    assert [(row.squad, list(row.units)) for row in action.squads] == [(5, [])]


def test_a_squad_is_not_merged_into_itself():
    interface = Interface()
    interface.merge(0, 0)
    action = Action()

    assert interface.intervene(action, None, [_squad()], _observation()) == []
    assert action.squads == []


def test_a_merge_of_a_squad_another_commander_holds_is_refused():
    """One commander per squad, on both ends of the move: taking units into a squad changes it as surely as taking them out of one does."""
    source = _squad(0)
    source.commander = int(Commander.OPERATIONS)
    interface = Interface()
    interface.merge(0, 1)
    action = Action()
    assert interface.intervene(action, None, [source, _squad(1, members=[4])], _observation()) == []
    assert action.squads == []

    destination = _squad(1, members=[4])
    destination.commander = int(Commander.TACTICS)
    interface = Interface()
    interface.merge(0, 1)
    action = Action()
    assert interface.intervene(action, None, [_squad(0), destination], _observation()) == []
    assert action.squads == []


def test_a_merge_with_nothing_left_to_move_is_refused():
    """Nothing is stated at all, rather than an empty roster for the source. Retiring a squad that is already empty is the organisation layer's own rule, and a disband row for a number that layer may be raising a squad into this very period would retire theirs."""
    interface = Interface()
    interface.merge(0, 1)
    action = Action()

    assert interface.intervene(action, None, [_squad(0, members=[]), _squad(1, members=[4])],
                               _observation()) == []
    assert action.squads == []


def test_a_merge_without_a_destination_is_refused():
    """Dissolving a squad into a freshly raised squad of one's own is a rename: the same units under a new number, at the cost of a slot out of a cap of eight."""
    organisation = _organisation()
    interface = Interface(organisation=organisation)
    interface.request(Intervention(kind=Kind.REASSIGN, squad=0, into=-1, whole=True))
    action = Action()

    assert interface.intervene(action, None, [_squad()], _observation()) == []
    assert action.squads == [] and organisation.reserved == set()


def test_the_chains_rows_about_a_dissolved_squad_come_off_the_action():
    """The chain decided this period without knowing the squad was about to be dissolved. The game refuses a contract for a squad that does not exist, so leaving the row on would misfire nothing — but an action is the record of what was commanded, and one that orders a dissolved squad about is a record that lies."""
    interface = Interface()
    interface.merge(0, 1)
    action = _chain_action(0)
    action.contracts.append(Contract(squad=1, task=Task.DEFEND, target_region=2))
    action.deviations.append(SquadDeviation(squad=1, deviation=Deviation.FOCUS))
    interface.intervene(action, None, [_squad(0), _squad(1, members=[4])], _observation())

    assert [row.squad for row in action.contracts] == [1]
    assert [row.squad for row in action.deviations] == [1]


def test_two_merges_in_one_period_read_the_rosters_the_first_of_them_stated():
    """The game applies these rows in order and the later one wins outright, so a second merge built from the roster the period opened with would not merely ignore the first — it would undo it, handing back the units the first had moved."""
    interface = Interface()
    interface.merge(0, 1)
    interface.merge(1, 2)
    action = Action()
    applied = interface.intervene(
        action, None, [_squad(0, members=[1, 2]), _squad(1, members=[3]), _squad(2, members=[4])],
        _observation())

    assert len(applied) == 2
    # Read as the game reads them: the last row about a squad is what the squad becomes.
    rosters = {}
    for row in action.squads:
        rosters[row.squad] = list(row.units)
    assert rosters[2] == [4, 3, 1, 2]
    assert rosters[0] == [] and rosters[1] == []


def test_a_squad_taken_over_after_a_merge_carries_what_the_merge_put_into_it():
    """The same ordering rule, on the other pair of operations that state a roster. A take built from the roster the period opened with would hand the squad over as it was before the merge, and the units the merge had just moved into it would belong to nothing at all."""
    interface = Interface()
    interface.merge(0, 1)
    interface.take(1, int(Commander.OPERATIONS | Commander.TACTICS))
    action = Action()
    applied = interface.intervene(action, None, [_squad(0, members=[1, 2]), _squad(1, members=[4])],
                                  _observation())

    assert [i.kind for i in applied] == [Kind.REASSIGN, Kind.TAKE]
    rosters = {row.squad: (int(row.commander), list(row.units)) for row in action.squads}
    assert rosters[1] == (int(Commander.OPERATIONS | Commander.TACTICS), [4, 1, 2])
    assert rosters[0] == (int(Commander.MACHINE), [])


def test_a_squad_a_merge_has_just_dissolved_cannot_be_taken_over():
    """A squad that ceased to exist earlier in the same period is not there to be taken, and stating a roster for it would raise it again with the units it has just given away."""
    interface = Interface()
    interface.merge(0, 1)
    interface.take(0)
    action = Action()
    applied = interface.intervene(action, None, [_squad(0, members=[1, 2]), _squad(1, members=[4])],
                                  _observation())

    assert [i.kind for i in applied] == [Kind.REASSIGN]
    assert 0 not in interface.held
    assert {row.squad: list(row.units) for row in action.squads}[0] == []


def test_a_merge_is_recorded_as_the_units_that_actually_moved():
    """What is written down is the pair of a board and the decision taken from it, so the decision has to be the one that happened. 'The whole squad' is not something anything can be learnt from without the roster it resolved to."""
    interface = Interface()
    interface.merge(0, 1)
    applied = interface.intervene(Action(), None, [_squad(0, members=[1, 2, 3]), _squad(1, members=[4])],
                                  _observation())

    row = applied[0].as_dict()
    assert row["whole"] is True
    assert row["units"] == [1, 2, 3] and row["into"] == 1


def test_a_squad_takes_onto_its_roster_the_units_the_game_reports_in_it():
    """The rule that the game is right about where a unit is, read in the direction the layer used to ignore. Without it a squad merged into from outside is short on this layer's books by however many arrived, for the rest of the match: it asks for reinforcements it does not need and draws the next loose units towards a squad that is already over strength."""
    organisation = _organisation()
    organisation.squads[0].members = [1, 2]
    observation = _observation(unit_states=[_unit(unit, 0) for unit in (1, 2, 3, 4)])

    _, records, _ = organisation.update(_view(observation), [])
    assert records[0].members == [1, 2, 3, 4]


def test_a_squad_that_took_in_units_from_outside_holds_its_stance_for_a_period():
    organisation = _organisation()
    organisation.squads[0].members = [1, 2]
    observation = _observation(unit_states=[_unit(unit, 0) for unit in (1, 2, 3, 4)])

    _, records, _ = organisation.update(_view(observation), [])
    assert records[0].settling
    # And no longer, once a period has passed with the roster agreed: settling is what the operational layer reads to leave a squad alone until it has seen what the squad now consists of, not a lasting property of it.
    _, records, _ = organisation.update(_view(observation), [])
    assert not records[0].settling


def test_the_number_of_a_retired_squad_is_not_raised_into_again_in_the_same_period():
    """A number has to name nothing for a period between one squad and the next, because that absence is the only announcement a squad's end gets: everything that follows a squad through time follows its number, so a number retired and raised again inside one period would read as one squad carrying on rather than as two, and what the second squad went on to earn would be paid to decisions taken about the first.

    Merging is what makes it reachable by hand. The period that reports a merged squad's units under the squad they were folded into is the same period that finds the squad they left empty and retires it, and the loose units of that period are enough to form a squad the moment a number is free.
    """
    organisation = _organisation(chain_squads=(0, 1))
    organisation.squads[0].members = [1, 2, 3]
    organisation.squads[1].members = [4, 5, 6, 7, 8, 9]
    # The merge landed last period: the game files squad 0's units under squad 1 and no longer reports squad 0 at all. Squad 1 is over strength already, so the four loose units cannot be reinforced into it and are enough to raise a squad of their own.
    merged = _observation(
        unit_states=[_unit(unit, 1) for unit in (1, 2, 3, 4, 5, 6, 7, 8, 9)]
                    + [_unit(unit) for unit in (10, 11, 12, 13)],
        squads=[_state(1)])

    assignments, records, _ = organisation.update(_view(merged), [])
    assert [record.id for record in records] == [1, 2]
    assert records[1].members == [10, 11, 12, 13]
    assert 0 not in {row.squad for row in assignments}
    assert 0 not in organisation.free_ids

    # And it is free again at the top of the next period, having spent a whole one naming nothing.
    settled = _observation(
        unit_states=[_unit(unit, 1) for unit in (1, 2, 3, 4, 5, 6, 7, 8, 9)]
                    + [_unit(unit, 2) for unit in (10, 11, 12, 13)],
        squads=[_state(1), _state(2)])
    organisation.update(_view(settled), [])
    assert 0 in organisation.free_ids


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


def test_a_finished_episode_records_the_interference_before_the_policy_is_put_down():
    """The intruder's log lives on the policy's own outside commanders, and the session puts the policy down the moment the match ends. If the record reads the interference after that put-down it reads an empty list every time, and every episode an intruder disturbed is journalled as undisturbed — the one confusion this field exists to prevent, and the tainting a learning run depends on would have nothing in the record to answer to. So the record has to be gathered while the policy is still standing, exactly as the statistics are."""

    class _Stats:
        def as_dict(self):
            return {"interventions": 7}

    class _Commander:
        def __init__(self, log):
            self.log = log

    class _Policy:
        def __init__(self, commander):
            self.statistics = _Stats()
            self.outside = [commander]
            self.closed = False

        def close(self):
            # The put-down the real chain does at episode end; the reference is cleared by the session right after.
            self.closed = True

    log = Log(events=[{"kind": "contract", "squad": 5}], touched={5})
    policy = _Policy(_Commander(log))

    session = Session.__new__(Session)
    session.policy = policy
    session.outside = list(policy.outside)
    session.sync = {}
    session.settings = EpisodeSettings()
    session.arm = "operations"
    session.instance = 0
    session.episode_started_at = 0.0
    session.records = []
    session.episodes_wanted = 1
    session.journal = None

    session.on_episode(json.dumps({
        "seconds": 300, "winner": -1, "aliveTeams": 2, "timeout": True,
        "team": 0, "episode": 1, "standing": [],
    }).encode("utf-8"))

    assert policy.closed and session.policy is None
    record = session.records[-1]
    assert record.interference.get("touched") == [5]
    assert len(record.interference.get("events", [])) == 1


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
