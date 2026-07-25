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

from ..wire import Action, Commander, Contract, Deviation, Observation, SquadAssignment, SquadDeviation, Stance, Task

log = logging.getLogger(__name__)


class Kind(enum.IntEnum):
    """What an outside commander did, in the four operations the design's interface offers."""

    #: Authority: take some layers of a squad's command.
    TAKE = 0
    #: Authority: give them back.
    RETURN = 1
    #: Contract editing: write one for a squad this commander holds.
    CONTRACT = 2
    #: Reorganisation: move units out of a squad, into another or into one of this commander's own. Moving the whole of a squad is a merge, which is this same operation with the squad it empties retired rather than a fifth kind of thing to do.
    REASSIGN = 3
    #: Direct tactical command: choose the departure for a squad whose tactical command this commander holds.
    DEPART = 4


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
    #: True when the reassignment is of the whole squad — a merge — so that which units move is settled from the roster in hand when the request is drained rather than from one named when it was asked for, and the squad it empties is retired rather than left standing with nothing in it.
    whole: bool = False
    deviation: int = int(Deviation.HOLD)
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
            row.update(units=list(self.units), into=int(self.into), whole=bool(self.whole))
        elif self.kind is Kind.DEPART:
            row["deviation"] = int(self.deviation)
        return row


class Recorder:
    """Writes each intervention beside the board it was decided from, one JSON object per line.

    This is the imitation data the design says is otherwise unobtainable. A replay carries commands and no state, and the same settings do not reproduce the same match, so the state a recorded command was conditioned on cannot be recovered by replaying it; here the state is in hand at the moment of the decision and is simply written down with it. The quantity is small — a human playing one match at ordinary speed produces tens to a couple of hundred of these — so what it is for is initialisation, regularisation and preference, mixed with the far larger volume of the same shape that the script chain emits.

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

    Requests are queued rather than applied where they are made because they are made from another thread — a console, a user interface, a rule — while the action they amend is built on the session's own thread when an observation arrives. Draining them at one point keeps the ordering of an action's sections meaningful and means an outside commander never has to know what a period is.
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

    def merge(self, squad: int, into: int) -> None:
        """Folds a squad whole into another. Every unit it has when the request is drained goes, and the emptied squad is retired.

        Stated as an intention rather than as a list of units because the roster moves between the moment a commander decides to merge and the moment the action carrying it is built: a unit finishes production and is reinforced in, or one is destroyed. A list captured at the first of those moments would leave a straggler behind and the squad standing, which is the one outcome a merge must not have — the point of the operation is that one of the two squads stops existing.
        """
        self.request(Intervention(kind=Kind.REASSIGN, squad=squad, into=int(into),
                                  whole=True, by=self.name))

    def depart(self, squad: int, deviation: Deviation) -> None:
        self.request(Intervention(kind=Kind.DEPART, squad=squad, deviation=int(deviation), by=self.name))

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
            self._give_up(slot)

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
        return False

    def _take(self, intervention: Intervention, action: Action,
              by_id: Dict[int, object], observation: Observation) -> bool:
        members = self._membership(action, intervention.squad, by_id, observation)
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
        # Read before the chain's rows come off, because stripping cannot tell the chain's roster for this squad from one this commander wrote earlier in the same drain, and handing a squad back is not a reason to undo a reorganisation asked for a moment before it.
        members = self._membership(action, intervention.squad, by_id, observation)
        _strip(action, intervention.squad)
        if intervention.squad in self.own:
            # A squad this commander raised is not handed back as a squad. Its slot returns to the organisation layer and its units are released loose, which is the same route a squad the chain formed takes when it is retired.
            action.squads.append(SquadAssignment(squad=intervention.squad, commander=Commander.MACHINE, units=[]))
            self._give_up(intervention.squad)
            return True
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
            target_region=intervention.target_region, cost_budget=intervention.cost_budget,
            deadline_ms=intervention.deadline_ms, issued_at_ms=intervention.at_ms, override=True,
        ))
        self._written[intervention.squad] = (intervention.task, intervention.target_region,
                                             intervention.stance, intervention.at_ms)
        return True

    def _reassign(self, intervention: Intervention, action: Action,
                  by_id: Dict[int, object], observation: Observation) -> bool:
        members = self._membership(action, intervention.squad, by_id, observation)
        taken = list(members) if intervention.whole else [u for u in intervention.units if u in members]
        if not taken:
            if intervention.whole:
                log.info("squad %d has nothing left to merge", intervention.squad)
            return False
        # Moving units from a squad into itself would state two rosters for it in one action, the second of which is the first minus the units, so they would end up belonging to nothing at all.
        if intervention.into == intervention.squad:
            log.info("squad %d is where those units already are", intervention.squad)
            return False
        # A merge has to say where the squad is going. Left unsaid it would fall through to raising a squad of this commander's own, which is a rename rather than a merge: the same units under a new number, and a slot spent out of the cap of eight by the one operation that is supposed to give a slot back.
        if intervention.whole and intervention.into < 0:
            log.info("a merge has to say which squad to merge into")
            return False
        if self._taken_by_another(intervention.squad, by_id) or (
                intervention.into >= 0 and self._taken_by_another(intervention.into, by_id)):
            log.info("a squad in that move is held by someone else")
            return False
        if intervention.into >= 0 and not self._exists(action, intervention.into, by_id, observation):
            log.info("there is no squad %d to move units into", intervention.into)
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
            into_members = self._membership(action, destination, by_id, observation)
            joined = list(into_members) + [u for u in taken if u not in into_members]
            action.squads.append(SquadAssignment(
                squad=destination, commander=Commander(self.held.get(destination, 0)), units=joined))
            if destination in self.own:
                self.own[destination] = list(joined)
        if remaining:
            action.squads.append(SquadAssignment(
                squad=intervention.squad, commander=Commander(self.held.get(intervention.squad, 0)),
                units=remaining))
            if intervention.squad in self.own:
                self.own[intervention.squad] = remaining
        else:
            # Stated after the destination's roster, so the game moves the units out before it is told the squad they were in is finished, rather than dropping the squad with them still in it.
            self._dissolve(intervention.squad, action)
        if intervention.whole:
            # Recorded as the units that actually moved rather than as the intention, exactly as the raised slot above is written back: what the record is for is the pair of a board and the decision taken from it, and "the whole squad" is not a decision anything can be learnt from without the roster it resolved to.
            intervention.units = tuple(taken)
        return True

    def _dissolve(self, squad: int, action: Action) -> None:
        """Retires a squad every one of whose units has just been moved out.

        An empty roster is how a squad is retired: the game side drops a squad it is sent one, and the organisation layer expresses its own disbands the same way, so ending a squad needs no second mechanism, only a statement of the emptiness. The row says the squad is nobody's as it goes, because a disband carrying a holder's command bits would hand a squad over in the same breath that removes it.

        The chain's own rows about the squad come off first, for the reason taking a squad over takes them off: the chain decided this period without knowing the squad was about to be dissolved, and a contract or a departure left on the action names a squad that will not exist by the time it is read. The game refuses both in that case, so this is tidiness rather than safety — but an action is also the record of what was commanded, and one that orders a dissolved squad about is a record that lies.
        """
        # Before the row is appended rather than after, because stripping takes off the chain's own rosters for the squad and the disband is one of those.
        _strip(action, squad)
        action.squads.append(SquadAssignment(squad=squad, commander=Commander.MACHINE, units=[]))
        self._give_up(squad)

    def _give_up(self, squad: int) -> None:
        """Forgets everything this commander held about a squad and hands its slot back where the slot was borrowed.

        Both halves matter and they are separate. A holding left behind goes on stripping the chain's rows about a number that now belongs to nothing, or to somebody else's squad once the number is handed out again, which silences a squad this commander has no relation to. And a slot lent by the organisation layer and never returned is worse than one merely lost, because the cap it counts against is kept in a layer that cannot see it went: the chain quietly loses the ability to form a squad and nothing says why.

        The slot is handed back only for a squad this commander raised. A number belonging to a squad the chain formed was never ours to return, and offering it would read as though the chain's slots were in this commander's gift.
        """
        raised = squad in self.own
        self.own.pop(squad, None)
        self.held.pop(squad, None)
        self._written.pop(squad, None)
        if raised and self.organisation is not None:
            self.organisation.release(squad)

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

    def _exists(self, action: Action, squad: int, by_id: Dict[int, object],
                observation: Observation) -> bool:
        """Whether a squad number will name a squad once this action has landed: what the action already says about it where anything does, and where nothing does, whichever side of the process knows — the chain's records, this commander's own, or the game's squad block for one somebody else raised.

        Asked before units are moved into a squad because a roster stated for a number nobody has raised does not fail. It creates a squad in a slot the organisation layer still believes is free, and that layer hands the slot out again the next time it forms a squad; two squads would then answer to one number and the observation would describe whichever was written last, which is the collision the borrowed-slot rule exists to prevent.

        The action is read first, and an empty roster already on it is an answer rather than a silence, exactly as it is when a roster is read: it says the squad has just been retired, whether the chain's own disband row retired it or a merge drained a moment earlier in this same period did. Everything the other sources know describes the board as the period opened, so they still carry a squad this action ends, and moving units into it would append a live roster after the row that retired it — resurrecting a squad whose number the organisation layer has already taken back, which is the same collision arrived at from the other direction.
        """
        stated = self._stated(action, squad)
        if stated is not None:
            return bool(stated)
        return (squad in self.own or squad in by_id
                or any(state.id == squad for state in observation.squads))

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

    def _membership(self, action: Action, squad: int, by_id: Dict[int, object],
                    observation: Observation) -> List[int]:
        """Who a squad will consist of once this action has landed: what the action already says about it where anything does, and who is in it now where nothing does.

        Every row about a squad restates its whole roster and the game takes the last such row as the answer, so a row built from the roster the period opened with does not merely ignore an earlier row in the same action — it undoes it. That is what lets two things be asked for in one period and both mean what they said: a second reorganisation reads the first one's result, and taking a squad over or handing it back carries whatever was just moved into or out of it instead of restoring the composition the period began with.
        """
        stated = self._stated(action, squad)
        return list(stated) if stated is not None else self._roster(squad, by_id, observation)

    def _stated(self, action: Action, squad: int) -> Optional[List[int]]:
        """The roster this action already carries for a squad, read off the last row about it because that is the row the game settles on, or nothing where the action says nothing about it.

        An empty roster already stated is an answer rather than a silence: it says the squad has just been dissolved, and reading past it to the roster the squad had before would raise the squad again with the units it no longer has.
        """
        for row in reversed(action.squads):
            if row.squad == squad:
                return list(row.units)
        return None


def _strip(action: Action, squad: int, contracts: bool = True, deviations: bool = True,
           rosters: bool = True) -> None:
    """Takes the chain's own rows about one squad off the action, leaving anything an outside commander put there.

    What marks a row as the chain's differs by section, and each mark is the one that section already carries. A contract or a departure says so outright with its override bit. A roster has no such bit and needs none: the chain only ever states a roster for a squad it commands, so a roster naming a commander is one another outside commander wrote, and taking it off would be one commander undoing another's transfer rather than taking the chain's word back.
    """
    if contracts:
        action.contracts = [row for row in action.contracts if row.squad != squad or row.override]
    if deviations:
        action.deviations = [row for row in action.deviations if row.squad != squad or row.override]
    if rosters:
        action.squads = [row for row in action.squads
                         if row.squad != squad or row.commander != Commander.MACHINE]
