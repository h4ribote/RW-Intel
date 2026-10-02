"""The one route by which anything outside the command chain commands.

A human takes a squad over through this, and so does the script intruder the design puts into the training environment. There is deliberately no second channel for either: what an outside commander emits is a squad roster, a contract and a departure, in exactly the fields the layers emit them in, and it is amended onto the action after the chain has finished with it. That is what makes an intervention a state and an action of the same kind the layers produce, which is the whole reason the design says a record of interventions reopens imitation learning: no re-simulation is needed to recover what the board looked like when the decision was taken, because the decision was taken from an observation this process was holding.

Ownership is one-way and one-holder. Taking a squad sets the bits the game side reads, and from that moment the layer whose bits are set writes nothing about that squad; the holder's own rows carry an override that says the decision came from the holder, which is what stops taking a squad over from being a way of silencing it. Giving it back clears the bits, and the organisation layer then empties the squad into the unassigned pool and lets the ordinary reinforcement rule place its units, because what a holder did to a squad's composition and position is not knowable from here and a contract written for the old squad would be built on a guess.

Squads an outside commander raises for itself come out of the same cap of eight, borrowed from the organisation layer. The cap is the size of the observation's squad block, so a ninth squad is not surplus but invisible.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..wire import (
    Action,
    CargoKind,
    Commander,
    Contract,
    Deviation,
    Observation,
    SquadAssignment,
    SquadDeviation,
    Stance,
    Task,
)

log = logging.getLogger(__name__)


class Kind(enum.IntEnum):
    """What an outside commander did, in the four operations the design's interface offers."""

    #: Authority: take some layers of a squad's command.
    TAKE = 0
    #: Authority: give them back.
    RETURN = 1
    #: Contract editing: write one for a squad this commander holds.
    CONTRACT = 2
    #: Reorganisation: move units out of a squad, into another or into one of this commander's own.
    REASSIGN = 3
    #: Direct tactical command: choose the departure for a squad whose tactical command this commander holds.
    DEPART = 4
    #: Contract editing: have the transport in a slot of the lift layer carry a squad to a region.
    LIFT = 5


@dataclass
class Intervention:
    """One thing an outside commander did, with the game time it was done at. Recorded whole, because the pair worth keeping is this and the board it was decided from."""

    kind: Kind
    squad: int
    at_ms: int = 0
    #: Commander bits, for taking and returning.
    layers: int = 0
    task: int = int(Task.ATTACK)
    target_region: int = 0
    stance: int = int(Stance.AGGRESSIVE)
    cost_budget: float = 0.0
    deadline_ms: int = 0
    units: Sequence[int] = ()
    #: Where reassigned units go: a squad id, or -1 to leave them loose for the organisation layer to place.
    into: int = -1
    deviation: int = int(Deviation.HOLD)
    #: The transport slot a lift is to go by.
    slot: int = -1
    #: What raised this. "human" for the interface, or the intruder's own name, so a record can be read back knowing who wrote it.
    by: str = "human"

    def as_dict(self) -> dict:
        row = {"kind": self.kind.name.lower(), "squad": self.squad, "at_ms": self.at_ms, "by": self.by}
        if self.kind in (Kind.TAKE, Kind.RETURN):
            row["layers"] = int(self.layers)
        elif self.kind is Kind.CONTRACT:
            row.update(task=int(self.task), target_region=int(self.target_region),
                       stance=int(self.stance), cost_budget=round(float(self.cost_budget), 2),
                       deadline_ms=int(self.deadline_ms))
        elif self.kind is Kind.REASSIGN:
            row.update(units=list(self.units), into=int(self.into))
        elif self.kind is Kind.DEPART:
            row["deviation"] = int(self.deviation)
        elif self.kind is Kind.LIFT:
            row.update(slot=int(self.slot), target_region=int(self.target_region))
        return row


class Recorder:
    """Writes each intervention beside the board it was decided from, one JSON object per line.

    This is human play in the form the chain itself decides in. A replay can be played back to recover the state behind a human's command, but the command is a unit order and the contract it served has to be inferred; here the decision is a contract and the state is in hand at the moment it is taken, so both are simply written down together. The quantity is small -a human playing one match at ordinary speed produces tens to a couple of hundred of these -so what it is for is initialisation, regularisation and preference, mixed with the far larger volume of the same shape that the script chain emits.

    The encoder is supplied rather than imported so that the recording path does not oblige a control process to carry the learning package. Without one the board is kept as the few figures that summarise it, which is enough to read the log by eye and not enough to learn from.
    """

    def __init__(self, path: str, encoder: Optional[Callable[[Observation], Sequence[float]]] = None) -> None:
        self.path = path
        self.encoder = encoder
        self._lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def write(self, intervention: Intervention, observation: Observation, instance: int = -1) -> None:
        row = {
            "instance": instance,
            "episode": observation.episode,
            "game_time_ms": observation.game_time_ms,
            "intervention": intervention.as_dict(),
            "board": _summary(observation),
        }
        if self.encoder is not None:
            row["state"] = [round(float(value), 5) for value in self.encoder(observation)]
        line = json.dumps(row, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


def _summary(observation: Observation) -> dict:
    return {
        "credits": round(observation.credits, 1),
        "income": round(observation.income, 2),
        "units": observation.units,
        "squads": len(observation.squads),
        "enemies": sum(1 for unit in observation.unit_states if unit.hostile),
    }


class Interface:
    """The intervention interface: a queue of requests, drained onto the action once a period.

    Requests are queued rather than applied where they are made because they are made from another thread -a console, a user interface, a rule -while the action they amend is built on the session's own thread when an observation arrives. Draining them at one point keeps the ordering of an action's sections meaningful and means an outside commander never has to know what a period is.
    """

    def __init__(self, organisation=None, recorder: Optional[Recorder] = None,
                 name: str = "human", instance: int = -1) -> None:
        #: The organisation layer, asked for a squad slot when this commander wants a squad of its own. Optional so that an interface can be built before a policy exists.
        self.organisation = organisation
        self.recorder = recorder
        self.name = name
        self.instance = instance
        self._lock = threading.Lock()
        self._pending: List[Intervention] = []
        #: Which layers of which squads this commander is holding, by squad id.
        self.held: Dict[int, int] = {}
        #: Rosters of squads this commander raised itself, which the chain has no record of.
        self.own: Dict[int, List[int]] = {}
        #: The last contract written for each held squad, so that it is restated only when it changes.
        self._written: Dict[int, tuple] = {}

    # ---- what a commander asks for ----------------------------------------------------

    def request(self, intervention: Intervention) -> None:
        with self._lock:
            self._pending.append(intervention)

    def take(self, squad: int, layers: int = int(Commander.OPERATIONS)) -> None:
        self.request(Intervention(kind=Kind.TAKE, squad=squad, layers=int(layers), by=self.name))

    def give_back(self, squad: int) -> None:
        self.request(Intervention(kind=Kind.RETURN, squad=squad, by=self.name))

    def write(self, squad: int, task: Task, target_region: int, stance: Stance = Stance.AGGRESSIVE,
              cost_budget: float = 0.0, deadline_ms: int = 0) -> None:
        self.request(Intervention(kind=Kind.CONTRACT, squad=squad, task=int(task),
                                  target_region=int(target_region), stance=int(stance),
                                  cost_budget=float(cost_budget), deadline_ms=int(deadline_ms),
                                  by=self.name))

    def reassign(self, squad: int, units: Sequence[int], into: int = -1) -> None:
        """Moves units out of a squad. `into` is another squad, or -1 to raise one of this commander's own for them."""
        self.request(Intervention(kind=Kind.REASSIGN, squad=squad, units=tuple(units),
                                  into=int(into), by=self.name))

    def depart(self, squad: int, deviation: Deviation) -> None:
        self.request(Intervention(kind=Kind.DEPART, squad=squad, deviation=int(deviation), by=self.name))

    def lift(self, squad: int, slot: int, target_region: int) -> None:
        """Has the transport in the slot carry the squad to the region."""
        self.request(Intervention(kind=Kind.LIFT, squad=squad, slot=int(slot), target_region=int(target_region),
                                  by=self.name))

    # ---- what happens to it -----------------------------------------------------------

    def intervene(self, action: Action, view, squads: List, observation: Observation) -> List[Intervention]:
        """Amends the action the chain built. Called once a period by the policy, on the session's thread."""
        with self._lock:
            pending, self._pending = self._pending, []

        by_id = {squad.id: squad for squad in squads}
        now = observation.game_time_ms
        # Held squads are cleared of the chain's decisions first, so that anything this commander then puts on the action is not taken off again by the sweep.
        self._restate(action)
        applied: List[Intervention] = []
        for intervention in pending:
            intervention.at_ms = now
            if self._apply(intervention, action, by_id, observation):
                applied.append(intervention)
                if self.recorder is not None:
                    self.recorder.write(intervention, observation, self.instance)
        self._retire_spent(observation)
        return applied

    def _retire_spent(self, observation: Observation) -> None:
        """Gives back the slot of a squad this commander raised once there is nothing left in it.

        The organisation layer has the same rule for its own squads and states the reason: a slot that is not handed back is lost for the rest of the match. A slot lent out and never returned is worse, because the cap it counts against is kept in a layer that cannot see it went, so the chain quietly loses the ability to form a squad and nothing says why.
        """
        alive = {unit.id for unit in observation.unit_states}
        if not alive:
            return
        for slot in [slot for slot, members in self.own.items() if not [m for m in members if m in alive]]:
            del self.own[slot]
            self.held.pop(slot, None)
            self._written.pop(slot, None)
            if self.organisation is not None:
                self.organisation.release(slot)

    def _apply(self, intervention: Intervention, action: Action,
               by_id: Dict[int, object], observation: Observation) -> bool:
        if intervention.kind is Kind.TAKE:
            return self._take(intervention, action, by_id, observation)
        if intervention.kind is Kind.RETURN:
            return self._return(intervention, action, by_id, observation)
        if intervention.kind is Kind.CONTRACT:
            return self._contract(intervention, action)
        if intervention.kind is Kind.REASSIGN:
            return self._reassign(intervention, action, by_id, observation)
        if intervention.kind is Kind.DEPART:
            return self._depart(intervention, action)
        if intervention.kind is Kind.LIFT:
            return self._lift(intervention, action, by_id, observation)
        return False

    def _lift(self, intervention: Intervention, action: Action,
              by_id: Dict[int, object], observation: Observation) -> bool:
        """Has the lift layer plan the lift and sends it as this commander's, which the game side accepts for a squad this commander holds. The lift layer is the chain's, so that the transport is then busy for the chain too."""
        logistics = getattr(self.organisation, "logistics", None)
        if logistics is None or not 0 <= intervention.slot < len(logistics.slots):
            log.info("no transport slot %d to lift with", intervention.slot)
            return False
        if self._taken_by_another(intervention.squad, by_id):
            log.info("squad %d is held by someone else", intervention.squad)
            return False
        kinds = {unit.id: unit for unit in observation.unit_states}
        members = []
        for unit_id in self._roster(intervention.squad, by_id, observation):
            unit = kinds.get(unit_id)
            kind = logistics.catalogue.kind(unit.type_index) if unit is not None else None
            if kind is not None and not unit.carrier:
                members.append((unit.type_index, kind.movement, unit.x, unit.y))
        before = len(logistics.rows())
        if not logistics.lift_squad(intervention.squad, members, intervention.slot, intervention.target_region, intervention.at_ms):
            log.info("the transport in slot %d cannot carry squad %d there", intervention.slot, intervention.squad)
            return False
        for row in logistics.rows()[before:]:
            row.override = True
            action.lifts.append(row)
        return True

    def _take(self, intervention: Intervention, action: Action,
              by_id: Dict[int, object], observation: Observation) -> bool:
        members = self._roster(intervention.squad, by_id, observation)
        if not members:
            log.info("no squad %d to take", intervention.squad)
            return False
        # A squad has one commander. Two of them holding it at once is the shared control the design refuses outright, and it would show up not as a refusal but as two override contracts on the same squad in the same action, the later of which wins for reasons neither commander can see.
        if self._taken_by_another(intervention.squad, by_id):
            log.info("squad %d is already held by someone else", intervention.squad)
            return False
        layers = intervention.layers or int(Commander.OPERATIONS)
        self.held[intervention.squad] = layers
        # The chain decided this period without knowing the squad had changed hands, since it reads that from the observation. Its rows for the squad are taken off before ours go on.
        _strip(action, intervention.squad)
        action.squads.append(SquadAssignment(squad=intervention.squad, commander=Commander(layers),
                                             units=list(members)))
        return True

    def _return(self, intervention: Intervention, action: Action,
                by_id: Dict[int, object], observation: Observation) -> bool:
        if intervention.squad not in self.held and intervention.squad not in self.own:
            return False
        intervention.layers = self.held.pop(intervention.squad, 0)
        self._written.pop(intervention.squad, None)
        _strip(action, intervention.squad)
        if intervention.squad in self.own:
            # A squad this commander raised is not handed back as a squad. Its slot returns to the organisation layer and its units are released loose, which is the same route a squad the chain formed takes when it is retired.
            action.squads.append(SquadAssignment(squad=intervention.squad, commander=Commander.MACHINE, units=[]))
            del self.own[intervention.squad]
            if self.organisation is not None:
                self.organisation.release(intervention.squad)
            return True
        members = self._roster(intervention.squad, by_id, observation)
        action.squads.append(SquadAssignment(squad=intervention.squad, commander=Commander.MACHINE,
                                             units=list(members)))
        return True

    def _contract(self, intervention: Intervention, action: Action) -> bool:
        """Writes a contract, whether or not this commander holds the squad.

        Holding is not a condition because editing the contract of a squad the chain is otherwise running is one of the four things the interface is for, and it is the lightest of the interventions the design wants trained against: the squad stays under the tactical layer and only its errand changes. Where the squad is held the override is what makes the game side accept the row at all; where it is not, the override is harmless and the chain simply finds the errand already set the next time it looks.
        """
        _strip(action, intervention.squad, deviations=False, rosters=False)
        action.contracts.append(Contract(
            squad=intervention.squad, task=Task(intervention.task), stance=Stance(intervention.stance),
            target=intervention.target_region, cost_budget=intervention.cost_budget,
            deadline_ms=intervention.deadline_ms, issued_at_ms=intervention.at_ms, override=True,
        ))
        self._written[intervention.squad] = (intervention.task, intervention.target_region,
                                             intervention.stance, intervention.at_ms)
        return True

    def _reassign(self, intervention: Intervention, action: Action,
                  by_id: Dict[int, object], observation: Observation) -> bool:
        members = self._roster(intervention.squad, by_id, observation)
        taken = [unit for unit in intervention.units if unit in members]
        if not taken:
            return False
        # Moving units from a squad into itself would state two rosters for it in one action, the second of which is the first minus the units, so they would end up belonging to nothing at all.
        if intervention.into == intervention.squad:
            log.info("squad %d is where those units already are", intervention.squad)
            return False
        if self._taken_by_another(intervention.squad, by_id) or (
                intervention.into >= 0 and self._taken_by_another(intervention.into, by_id)):
            log.info("a squad in that move is held by someone else")
            return False
        remaining = [unit for unit in members if unit not in taken]
        destination = intervention.into
        if destination < 0:
            destination = self._raise_squad(taken)
            if destination < 0:
                log.info("no squad slot free to move %d unit(s) into", len(taken))
                return False
            intervention.into = destination
            # The raised squad has to be stated to the game as a roster like any other, or the units are merely missing from the squad they left and belong to nothing.
            action.squads.append(SquadAssignment(
                squad=destination, commander=Commander(self.held[destination]), units=list(taken)))
        else:
            into_members = self._roster(destination, by_id, observation)
            action.squads.append(SquadAssignment(
                squad=destination, commander=Commander(self.held.get(destination, 0)),
                units=list(into_members) + [u for u in taken if u not in into_members]))
            if destination in self.own:
                self.own[destination] = list(into_members) + [u for u in taken if u not in into_members]
        action.squads.append(SquadAssignment(
            squad=intervention.squad, commander=Commander(self.held.get(intervention.squad, 0)),
            units=remaining))
        if intervention.squad in self.own:
            self.own[intervention.squad] = remaining
        return True

    def _depart(self, intervention: Intervention, action: Action) -> bool:
        if not self.held.get(intervention.squad, 0) & int(Commander.TACTICS) and intervention.squad not in self.own:
            return False
        action.deviations = [row for row in action.deviations if row.squad != intervention.squad]
        action.deviations.append(SquadDeviation(squad=intervention.squad,
                                                deviation=Deviation(intervention.deviation), override=True))
        return True

    def _taken_by_another(self, squad: int, by_id: Dict[int, object]) -> bool:
        """Whether somebody other than this commander is holding a squad, read from the byte the game reports rather than from any commander's own bookkeeping, which is the only account all of them share."""
        if squad in self.held or squad in self.own:
            return False
        record = by_id.get(squad)
        return bool(record is not None and record.commander)

    def _raise_squad(self, units: Sequence[int]) -> int:
        """A squad of this commander's own for units taken out of another. The slot is borrowed from the organisation layer, which is where the cap of eight is kept."""
        if self.organisation is None:
            return -1
        slot = self.organisation.reserve()
        if slot is None:
            return -1
        self.own[slot] = list(units)
        self.held[slot] = int(Commander.OPERATIONS | Commander.TACTICS)
        return slot

    def _restate(self, action: Action) -> None:
        """Keeps the chain off the squads this commander holds, on every period rather than only on the one the squad was taken.

        The chain reads who holds a squad out of the observation, so between taking a squad and the next frame that reports the squad block it goes on writing about one it no longer owns. Stripping every period costs a scan of a handful of rows and removes the window entirely.

        What is stripped follows what was taken. Taking only the operational command of a squad leaves the tactical layer to fight it, which the design calls the most useful arrangement of the two, so its departures are left alone; the roster is the holder's in either case, because a squad any of whose command has changed hands is not one to reshuffle.
        """
        for squad in set(self.held) | set(self.own):
            layers = self.held.get(squad, int(Commander.OPERATIONS | Commander.TACTICS))
            _strip(action, squad,
                   contracts=bool(layers & int(Commander.OPERATIONS)),
                   deviations=bool(layers & int(Commander.TACTICS)))

    def _roster(self, squad: int, by_id: Dict[int, object], observation: Observation) -> List[int]:
        """Who is in a squad. The chain's own record where there is one, this commander's where it raised the squad itself, and the observation where neither has caught up."""
        if squad in self.own:
            live = {unit.id for unit in observation.unit_states}
            self.own[squad] = [unit for unit in self.own[squad] if unit in live] if live else self.own[squad]
            return list(self.own[squad])
        record = by_id.get(squad)
        if record is not None and record.members:
            return list(record.members)
        return [unit.id for unit in observation.unit_states if unit.squad == squad]


def _strip(action: Action, squad: int, contracts: bool = True, deviations: bool = True,
           rosters: bool = True) -> None:
    """Takes the chain's own rows about one squad off the action, leaving anything an outside commander put there.

    What marks a row as the chain's differs by section, and each mark is the one that section already carries. A contract or a departure says so outright with its override bit. A roster has no such bit and needs none: the chain only ever states a roster for a squad it commands, so a roster naming a commander is one another outside commander wrote, and taking it off would be one commander undoing another's transfer rather than taking the chain's word back.
    """
    if contracts:
        action.contracts = [row for row in action.contracts if row.squad != squad or row.override]
        # A lift of the squad is part of its errand, so it goes with the contract; the game side would refuse it anyway, and the lift layer frees the slot once the lift goes unreported.
        action.lifts = [row for row in action.lifts
                        if row.override or row.cargo_kind != CargoKind.SQUAD or squad not in row.cargo[:1]]
    if deviations:
        action.deviations = [row for row in action.deviations if row.squad != squad or row.override]
    if rosters:
        action.squads = [row for row in action.squads
                         if row.squad != squad or row.commander != Commander.MACHINE]
