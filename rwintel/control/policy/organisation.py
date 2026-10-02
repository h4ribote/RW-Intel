"""Squad lifetime: who exists, who is in it, and what has to be built to keep it whole.

A squad is a first class entity with a name that outlives any one mission, which is what lets a contract be about a squad instead of about a set of unit ids, and what makes "hand this squad to the human" a single operation. The price of that is exactly this layer: something has to form squads, feed reinforcements into them, fold the worn ones together and retire the ones that are finished, or the count grows without limit and every squad ends up a remnant.

The cap of eight is kept here and nowhere else. It is what fixes the operational layer's action space and what the observation's squad block is sized for, so a ninth squad would have no slot to be reported in. At the cap this layer forms nothing at all and puts every loose unit into an existing squad, which is the deliberate failure mode: an over strength squad is merely inefficient, whereas a squad with no slot is invisible.

It runs on events rather than on a clock. Production completing, a unit dying and a squad falling through its doctrine's strength are the three things that change what the squads are, and all three arrive as records in the observation rather than having to be inferred by diffing rosters -which matters because a roster diff cannot tell a unit that died from a unit that was handed to another commander, and the two want opposite responses.

Membership is decided here and stated in whole rosters, never as deltas: the game side replaces a squad's roster with what it is sent and retires a squad sent an empty one, so a full roster is also how a disband is expressed. Everything else about a squad -its worth, its centre, how scattered it is, whether a human holds it -is computed game side and folded back in each period, because those are questions about the world rather than about the plan.
"""

from __future__ import annotations

import bisect
import math
from typing import Dict, List, Optional, Set, Tuple

from ...wire import (
    BLOCK_SQUADS,
    BLOCK_UNITS,
    Commander,
    EventState,
    NO_SQUAD,
    Observation,
    PASSAGE_CLASSES,
    SquadAssignment,
    Status,
)
from .catalogue import Catalogue
from .contracts import DOCTRINES, Doctrine, Domain, Replacement, Role, SQUAD_CAP, Shortfall, SquadRecord
from .options import Options
from .view import Sighting, WorldView

#: Event kinds, mirroring the ones the in-process agent emits.
EVENT_UNIT_COMPLETED = 1
EVENT_UNIT_LOST = 2
EVENT_SQUAD_DEPLETED = 3

#: Opening value: the share of its own high water worth below which a squad is no longer a squad in its own right and folds into a neighbour.
MERGE_HEALTH = 0.40

#: Opening value: how far apart two squads of the same doctrine may be, in world units, and still be merged. Far enough that a worn squad ordinarily has somewhere to go, near enough that the merged squad does not spend the next minute walking to itself.
MERGE_DISTANCE = 1200.0

#: Opening value: below this share of its high water worth a squad with nowhere to merge into is retired and its survivors returned to the pool.
DISBAND_HEALTH = 0.15

#: Opening value: how far a loose unit may be from the squad it is meant for and still join it alone. Further than this it waits for others bound the same way.
CONVOY_DISTANCE = 1500.0

#: Opening value: how many units bound for the same distant squad travel together.
CONVOY_SIZE = 3

#: Opening value: the longest a unit waits for a convoy before it goes alone, so that a trickle of production does not keep reinforcements at home for ever.
CONVOY_WAIT_MS = 30000

#: Opening value: which doctrine gets first refusal on the loose units when more than one of them could be formed. Fighting formations first, so that the front is manned before the rear is comfortable.
FORMATION_ORDER: Tuple[Doctrine, ...] = (
    Doctrine.VANGUARD,
    Doctrine.GARRISON,
    Doctrine.RAID,
    Doctrine.FLEET,
    Doctrine.AIRWING,
    Doctrine.ENGINEER,
)


class Organisation:
    """The layer that owns the squads. Constructed once per episode and carried across periods, since squad identity is the state that has to survive between decisions."""

    def __init__(self, session, catalogue: Catalogue, options: Options = Options()) -> None:
        self.session = session
        self.catalogue = catalogue
        self.options = options
        #: Loose units held back to travel with others to a distant squad, with when each started waiting.
        self.waiting_since: Dict[int, int] = {}
        self.squads: Dict[int, SquadRecord] = {}
        #: Ids not in use. They double as the observation's squad slots, so they are drawn from a fixed set and handed back on a disband rather than counted upward.
        self.free_ids: List[int] = list(range(SQUAD_CAP))
        #: Units the game has announced finished but which have not yet been seen loose in a unit block. A completion is the economy handing a unit over, and it should not be forgotten because the frame that carried it had no roster on it.
        self.awaiting_orders: Set[int] = set()
        #: Squads a human held as of last period, which is how a return of command is recognised: the byte going back to zero.
        self.human_held: Set[int] = set()
        #: Squads that took in units a human had been moving. The operational layer is meant to leave these alone for one period and let them hold their stance, because what a human did with them is not knowable from here and a contract issued now would be built on a guess.
        self.settling: Set[int] = set()
        #: Squads the game reported this period, which is what says whether a disband has to be sent at all or the game has already dropped an empty squad on its own.
        self.known_to_game: Set[int] = set()
        #: The lift layer, which holds the transports this layer leaves out of every squad; handed over by the policy, and read by a commander outside the chain that wants a squad carried.
        self.logistics = None
        #: Slots handed to someone outside the chain. The cap belongs to this layer and to nowhere else, so a human or an intruder that wants a squad of its own asks for the slot here rather than picking a number; otherwise two commanders would eventually name the same one and the observation would describe whichever wrote last.
        self.reserved: Set[int] = set()

    def update(self, view: WorldView, shortfalls: List[Shortfall]) -> Tuple[List[SquadAssignment], List[SquadRecord], List[Replacement]]:
        """One period of organisation: fold in what the game reports, act on what happened, and say what the squads are and what they need."""
        observation = view.observation
        by_id: Dict[int, Sighting] = {s.unit.id: s for s in view.ours}

        self._fold(observation)
        lost, depleted = self._read_events(observation.events)
        self._reconcile(observation, by_id, lost)

        changed: Set[int] = set()
        disbanded: Set[int] = set()
        pool: List[Sighting] = []
        self.settling = set()

        returned = self._take_back(by_id, pool, changed)
        # A squad the operational layer says is too worn for a mission is considered for merging whatever its worth says, since it is the layer giving the missions that knows the squad cannot do one.
        worn = depleted | {s.squad for s in shortfalls if s.worn_out}
        self._merge(worn, changed, disbanded)
        self._disband(by_id, pool, disbanded)

        for sighting in view.unassigned:
            # A unit still under construction is in nobody's squad because it does not exist yet, so completion -announced, or evident from the build byte -is what makes it available.
            if sighting.unit.built >= 255 or sighting.unit.id in self.awaiting_orders:
                pool.append(sighting)
        pool.sort(key=lambda s: s.unit.id)
        self.awaiting_orders -= {s.unit.id for s in pool}

        counts = {record.id: _roles_in(record, by_id) for record in self.squads.values()}
        self._distribute(pool, counts, by_id, changed, returned, observation.game_time_ms)
        loose = {s.unit.id for s in pool}
        self.waiting_since = {unit: since for unit, since in self.waiting_since.items() if unit in loose}

        assignments = [SquadAssignment(squad=squad_id, commander=Commander.MACHINE, units=[])
                       for squad_id in sorted(disbanded) if squad_id in self.known_to_game]
        assignments.extend(SquadAssignment(squad=squad_id, commander=Commander.MACHINE,
                                           units=list(self.squads[squad_id].members))
                           for squad_id in sorted(changed) if squad_id in self.squads)
        records = [self.squads[squad_id] for squad_id in sorted(self.squads)]
        # Carried on the record rather than kept here, because the layer that has to act on it is the one that hands out missions and it reads nothing of this layer but the records.
        for record in records:
            record.settling = record.id in self.settling
        return assignments, records, self._replacements(counts)

    def reserve(self) -> Optional[int]:
        """A squad slot for a commander outside the chain, or nothing when the cap leaves none. Handing one out is what keeps the cap true: the observation has eight slots whoever fills them, and a ninth squad would be invisible rather than merely surplus."""
        if not self.free_ids or len(self.squads) + len(self.reserved) >= SQUAD_CAP:
            return None
        slot = self.free_ids.pop(0)
        self.reserved.add(slot)
        return slot

    def release(self, squad_id: int) -> None:
        """Takes a reserved slot back once its holder is finished with it."""
        if squad_id in self.reserved:
            self.reserved.discard(squad_id)
            bisect.insort(self.free_ids, squad_id)

    # ---- what the game says ------------------------------------------------------------

    def _fold(self, observation: Observation) -> None:
        """Takes back the fields the game computes. Worth, centre, spread, status and who holds the command are all questions about the world, and the game process is the only thing positioned to answer them."""
        if not observation.blocks & BLOCK_SQUADS:
            return
        self.known_to_game = {state.id for state in observation.squads}
        for state in observation.squads:
            # A slot lent to someone outside the chain describes a squad this layer neither formed nor may touch.
            if state.id in self.reserved:
                continue
            record = self.squads.get(state.id)
            if record is None:
                record = self._adopt(state, observation)
                if record is None:
                    continue
            record.value = state.value
            record.formed_value = state.formed_value
            record.x = state.x
            record.y = state.y
            record.spread = state.spread
            # A report about a contract older than the one this side has since issued is about the errand that contract replaced: the game has not applied the new one yet. Its status and losses are that old errand's, so the new one reads as just begun.
            superseded = record.contract is not None and state.issued_at_ms < record.contract.issued_at_ms
            record.losses = 0.0 if superseded else state.losses
            record.status = Status.ACTIVE if superseded else Status(state.status)
            record.commander = state.commander
            record.aboard = state.aboard
            record.passage = PASSAGE_CLASSES[state.passage] if 0 <= state.passage < len(PASSAGE_CLASSES) else ""
            record.lift = state.lift

    def _adopt(self, state, observation: Observation) -> Optional[SquadRecord]:
        """Takes over a squad the game is holding that this layer has no record of.

        That happens after a reconnection: the game keeps its squads and goes on advancing them on their last contract, which is exactly what it is supposed to do, while this side has been rebuilt from nothing. Without adopting them their units would report a squad this layer has never heard of, so they would never be seen as loose either, and the army would answer to nobody at all.
        """
        members = [u.id for u in observation.unit_states if u.squad == state.id]
        if not members:
            return None
        doctrine = self._doctrine_of(members, observation)
        if doctrine is None:
            return None
        if state.id in self.free_ids:
            self.free_ids.remove(state.id)
        by_id = {u.id: u for u in observation.unit_states}
        domains = [self.catalogue.domain(by_id[m].type_index) for m in members if m in by_id]
        domain = max(set(domains), key=domains.count) if domains else Domain.GROUND
        record = SquadRecord(id=state.id, doctrine=doctrine, members=members, domain=domain)
        self.squads[state.id] = record
        return record

    def _doctrine_of(self, members: List[int], observation: Observation) -> Optional[Doctrine]:
        """The doctrine a squad of these units would have been formed under, which is the one most of them belong to."""
        tally: Dict[Doctrine, int] = {}
        by_id = {u.id: u for u in observation.unit_states}
        for member in members:
            unit = by_id.get(member)
            if unit is None:
                continue
            doctrine = self.catalogue.doctrine_for(unit.type_index)
            if doctrine is not None:
                tally[doctrine] = tally.get(doctrine, 0) + 1
        return max(tally, key=lambda d: (tally[d], -int(d))) if tally else None

    def _read_events(self, events: List[EventState]) -> Tuple[Set[int], Set[int]]:
        """Sorts the period's events into the two questions this layer asks of them: which units are gone, and which squads have fallen through their strength."""
        lost: Set[int] = set()
        depleted: Set[int] = set()
        for event in events:
            if event.kind == EVENT_UNIT_COMPLETED:
                self.awaiting_orders.add(event.unit)
            elif event.kind == EVENT_UNIT_LOST:
                lost.add(event.unit)
                self.awaiting_orders.discard(event.unit)
            elif event.kind == EVENT_SQUAD_DEPLETED:
                depleted.add(event.squad)
        return lost, depleted

    def _reconcile(self, observation: Observation, by_id: Dict[int, Sighting], lost: Set[int]) -> None:
        """Brings the rosters back in line with what is actually on the field. A unit reported lost goes immediately; the roster is otherwise trusted only when a unit block is present, since an absent block is silence rather than an empty world."""
        for record in self.squads.values():
            record.members = [m for m in record.members if m not in lost]
        if not observation.blocks & BLOCK_UNITS:
            return
        for record in self.squads.values():
            # A member the game now places in another squad has been moved by something outside this layer, and the game is right about where it is.
            record.members = [m for m in record.members
                              if m in by_id and by_id[m].unit.squad in (record.id, NO_SQUAD)]

    # ---- lifetime ----------------------------------------------------------------------

    def _take_back(self, by_id: Dict[int, Sighting], pool: List[Sighting], changed: Set[int]) -> Set[int]:
        """Empties a squad a human has just given back. The units keep nothing of the old squad and go through the ordinary reinforcement rule, because what a human did to a squad's composition and position is unknown here, and restoring the old squad would put a contract written for one thing onto another."""
        returned: Set[int] = set()
        for record in self.squads.values():
            if record.id not in self.human_held or not record.machine:
                continue
            for member in record.members:
                if member in by_id:
                    pool.append(by_id[member])
                    returned.add(member)
            record.members = []
            changed.add(record.id)
        self.human_held = {r.id for r in self.squads.values() if not r.machine}
        return returned

    def _merge(self, worn: Set[int], changed: Set[int], disbanded: Set[int]) -> None:
        """Folds a squad that is no longer worth commanding into a nearby one of its own kind. Weakest first, and a squad that has already taken someone in this period does not itself go anywhere, so the merges of one period cannot chain into a single unplanned mass."""
        absorbed: Set[int] = set()
        candidates = sorted((r for r in self.squads.values() if r.machine),
                            key=lambda r: (r.health, r.id))
        for record in candidates:
            if record.id in absorbed or record.id in disbanded or not record.members:
                continue
            if record.health >= MERGE_HEALTH and record.id not in worn:
                continue
            target = self._merge_target(record, disbanded)
            if target is None:
                continue
            target.members.extend(m for m in record.members if m not in target.members)
            record.members = []
            absorbed.add(target.id)
            changed.add(target.id)
            self._retire(record, disbanded)

    def _merge_target(self, record: SquadRecord, disbanded: Set[int]) -> Optional[SquadRecord]:
        """The nearest squad of the same doctrine and domain within reach that is fit to take this one in."""
        best: Optional[SquadRecord] = None
        best_distance = MERGE_DISTANCE
        for other in self.squads.values():
            if other.id == record.id or not other.machine or not other.members:
                continue
            if other.doctrine != record.doctrine or other.domain != record.domain or other.id in disbanded:
                continue
            if other.health < record.health:
                continue
            distance = math.hypot(other.x - record.x, other.y - record.y)
            if distance <= best_distance:
                best, best_distance = other, distance
        return best

    def _disband(self, by_id: Dict[int, Sighting], pool: List[Sighting], disbanded: Set[int]) -> None:
        """Retires the squads that are finished: the empty ones, and the ones too far gone with nowhere to merge into. Survivors go back into the pool, where they are worth more as reinforcements than as the last two members of a squad that keeps drawing contracts."""
        for record in list(self.squads.values()):
            if record.id in disbanded:
                continue
            # A squad a human holds is not reorganised, but one that is empty and no longer reported has been spent under his command, and its slot has to come back or it is lost for the rest of the match.
            if not record.machine and (record.members or record.id in self.known_to_game):
                continue
            if record.members:
                if record.health >= DISBAND_HEALTH or self._merge_target(record, disbanded) is not None:
                    continue
                pool.extend(by_id[m] for m in record.members if m in by_id)
                record.members = []
            self._retire(record, disbanded)

    def _retire(self, record: SquadRecord, disbanded: Set[int]) -> None:
        del self.squads[record.id]
        bisect.insort(self.free_ids, record.id)
        disbanded.add(record.id)

    # ---- filling the squads ------------------------------------------------------------

    def _distribute(self, pool: List[Sighting], counts: Dict[int, Dict[Role, int]],
                    by_id: Dict[int, Sighting], changed: Set[int], returned: Set[int], now: int) -> None:
        """Reinforcement and formation, alternating until neither has anything left to do.

        Reinforcement runs first so that an under strength squad is made whole before a second one is raised beside it, and it runs again after each formation so the new squad takes the spares that are standing next to it rather than waiting a period for them. Whatever is left when nothing more can be formed is pushed into a squad anyway: a unit standing in the pool contributes nothing, and at the cap the pool is the only place a new unit could otherwise go. The one exception is a unit waiting for a convoy to a distant squad, which waits in the pool until the convoy leaves.
        """
        while True:
            self._reinforce(pool, counts, by_id, changed, returned, strict=True, now=now)
            if not self._form(pool, counts, by_id, changed, returned):
                break
        self._reinforce(pool, counts, by_id, changed, returned, strict=False, now=now)

    def _reinforce(self, pool: List[Sighting], counts: Dict[int, Dict[Role, int]],
                   by_id: Dict[int, Sighting], changed: Set[int], returned: Set[int],
                   strict: bool, now: int) -> None:
        """Places loose units into the squads that want them. A unit bound for a squad further than CONVOY_DISTANCE away waits in the pool until CONVOY_SIZE are bound for the same squad, or until the first of them has waited CONVOY_WAIT_MS, and then they all go at once: one tank crossing the board to its squad is one tank met alone on the way."""
        placed: Set[int] = set()
        bound: Dict[int, List[Sighting]] = {}
        targets: Dict[int, SquadRecord] = {}

        def place(sighting: Sighting, target: SquadRecord) -> None:
            placed.add(sighting.unit.id)
            target.members.append(sighting.unit.id)
            counts[target.id][sighting.role] = counts[target.id].get(sighting.role, 0) + 1
            changed.add(target.id)
            if sighting.unit.id in returned:
                self.settling.add(target.id)

        for sighting in pool:
            target = self._best_squad(sighting, counts, by_id, strict)
            if target is None:
                continue
            x, y = self._centre(target, by_id)
            if self.options.convoy and math.hypot(sighting.unit.x - x, sighting.unit.y - y) > CONVOY_DISTANCE:
                bound.setdefault(target.id, []).append(sighting)
                targets[target.id] = target
                self.waiting_since.setdefault(sighting.unit.id, now)
                continue
            place(sighting, target)
        for squad_id, convoy in bound.items():
            waited = max(now - self.waiting_since[s.unit.id] for s in convoy)
            if len(convoy) >= CONVOY_SIZE or waited >= CONVOY_WAIT_MS:
                for sighting in convoy:
                    place(sighting, targets[squad_id])
        pool[:] = [s for s in pool if s.unit.id not in placed]

    def _best_squad(self, sighting: Sighting, counts: Dict[int, Dict[Role, int]],
                    by_id: Dict[int, Sighting], strict: bool) -> Optional[SquadRecord]:
        """Where a loose unit belongs: among the squads whose doctrine takes its role and its movement type, the one furthest below establishment, and of those the nearest.

        The shortfall that decides it is the one in this unit's own role. A squad four tanks short is not made whole by an anti air unit, so ranking on the squad's total gap would send reinforcements where they are counted rather than where they are wanted; the total breaks the tie between two squads equally short of this particular thing.
        """
        best: Optional[SquadRecord] = None
        best_key: Optional[Tuple[int, int, float, int]] = None
        # When switched to raise vanguards, a unit a vanguard would take is not pushed over strength into another kind of squad while a slot is free for one: it waits loose until enough have gathered to form one, or the garrisons swallow every tank and nothing is ever left to attack with.
        gathering = (self.options.vanguards and not strict and self.catalogue.accepts(Doctrine.VANGUARD, sighting.unit.type_index)
                     and len(self.squads) + len(self.reserved) < SQUAD_CAP)
        for record in self.squads.values():
            if not record.machine or not self.catalogue.accepts(record.doctrine, sighting.unit.type_index, record.domain):
                continue
            spec = DOCTRINES[record.doctrine]
            held = counts[record.id]
            deficit = spec.establishment.get(sighting.role, 0) - held.get(sighting.role, 0)
            if strict and deficit <= 0:
                continue
            if gathering and deficit <= 0 and record.doctrine != Doctrine.VANGUARD:
                continue
            total = sum(max(0, want - held.get(role, 0)) for role, want in spec.establishment.items())
            x, y = self._centre(record, by_id)
            key = (-deficit, -total, math.hypot(sighting.unit.x - x, sighting.unit.y - y), record.id)
            if best_key is None or key < best_key:
                best, best_key = record, key
        return best

    def _form(self, pool: List[Sighting], counts: Dict[int, Dict[Role, int]],
              by_id: Dict[int, Sighting], changed: Set[int], returned: Set[int]) -> bool:
        """Raises one squad if the pool holds a doctrine's minimum and there is a slot for it. Returns whether anything was formed, since the caller alternates this with reinforcement."""
        if len(self.squads) + len(self.reserved) >= SQUAD_CAP or not self.free_ids:
            return False
        garrisons = sum(1 for record in self.squads.values() if record.doctrine == Doctrine.GARRISON)
        for doctrine in FORMATION_ORDER:
            if (doctrine == Doctrine.GARRISON and self.options.vanguards
                    and garrisons >= self.options.tuning.max_garrisons):
                continue
            mustered = _muster(doctrine, pool, self.catalogue)
            if mustered is None:
                continue
            domain, picked = mustered
            squad_id = self.free_ids.pop(0)
            value = sum(s.value for s in picked)
            record = SquadRecord(
                id=squad_id,
                doctrine=doctrine,
                members=[s.unit.id for s in picked],
                domain=domain,
                value=value,
                # Stated here only so that the merge and disband rules mean something before the game has reported the squad back for the first time.
                formed_value=value,
                x=sum(s.unit.x for s in picked) / len(picked),
                y=sum(s.unit.y for s in picked) / len(picked),
            )
            self.squads[squad_id] = record
            counts[squad_id] = _roles_in(record, by_id)
            changed.add(squad_id)
            if any(s.unit.id in returned for s in picked):
                self.settling.add(squad_id)
            taken = {s.unit.id for s in picked}
            pool[:] = [s for s in pool if s.unit.id not in taken]
            return True
        return False

    def _centre(self, record: SquadRecord, by_id: Dict[int, Sighting]) -> Tuple[float, float]:
        """Where the squad is. Taken from the members currently visible rather than from the reported centre, so that a squad formed or reorganised this period is somewhere rather than at the origin."""
        seen = [by_id[m] for m in record.members if m in by_id]
        if not seen:
            return record.x, record.y
        return (sum(s.unit.x for s in seen) / len(seen), sum(s.unit.y for s in seen) / len(seen))

    # ---- what has to be built ----------------------------------------------------------

    def _replacements(self, counts: Dict[int, Dict[Role, int]]) -> List[Replacement]:
        """What the squads are short of, by role, once this period's reinforcements have been placed.

        Counting after the distribution rather than before is what keeps this from asking for units it has just been given. Squads a human holds are left out: their gaps cannot be filled from here without touching a roster that is not this layer's to touch.
        """
        wanted: Dict[Role, int] = {}
        for record in self.squads.values():
            if not record.machine:
                continue
            held = counts.get(record.id, {})
            for role, want in DOCTRINES[record.doctrine].establishment.items():
                short = want - held.get(role, 0)
                if short > 0:
                    wanted[role] = wanted.get(role, 0) + short
        return [Replacement(role=role, count=count)
                for role, count in sorted(wanted.items(), key=lambda item: (-item[1], item[0]))]


def _roles_in(record: SquadRecord, by_id: Dict[int, Sighting]) -> Dict[Role, int]:
    found: Dict[Role, int] = {}
    for member in record.members:
        sighting = by_id.get(member)
        if sighting is None:
            continue
        found[sighting.role] = found.get(sighting.role, 0) + 1
    return found


def _muster(doctrine: Doctrine, pool: List[Sighting], catalogue: Catalogue) -> Optional[Tuple[Domain, List[Sighting]]]:
    """The domain a squad of this doctrine would be raised in and the units it would be raised from, or nothing if the pool cannot meet its minimum in any one domain.

    Only the minimum is taken. The rest of the establishment is filled by the ordinary reinforcement rule, which is the same rule that will keep filling it for the squad's whole life, so there is no second policy deciding what a new squad looks like. A squad is raised in one domain so that every member can go wherever the others can; of the domains that can raise one, the one with the most candidates is taken.

    Which units are taken is decided by how close together they are, since a squad is a thing that moves as one: units are drawn from around whichever candidate sits nearest the middle of them all.
    """
    spec = DOCTRINES[doctrine]
    by_domain: Dict[Domain, List[Sighting]] = {}
    for sighting in pool:
        if catalogue.accepts(doctrine, sighting.unit.type_index):
            by_domain.setdefault(catalogue.domain(sighting.unit.type_index), []).append(sighting)
    for domain, candidates in sorted(by_domain.items(), key=lambda item: (-len(item[1]), int(item[0]))):
        by_role: Dict[Role, List[Sighting]] = {}
        for sighting in candidates:
            by_role.setdefault(sighting.role, []).append(sighting)
        if any(len(by_role.get(role, [])) < need for role, need in spec.minimum.items()):
            continue
        x = sum(s.unit.x for s in candidates) / len(candidates)
        y = sum(s.unit.y for s in candidates) / len(candidates)
        seed = min(candidates, key=lambda s: math.hypot(s.unit.x - x, s.unit.y - y))
        picked: List[Sighting] = []
        for role, need in spec.minimum.items():
            near = sorted(by_role[role], key=lambda s: (math.hypot(s.unit.x - seed.unit.x, s.unit.y - seed.unit.y), s.unit.id))
            picked.extend(near[:need])
        return domain, picked
    return None
