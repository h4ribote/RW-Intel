"""The lift layer: transports held in four fixed slots, and the lifts that carry squads and builders where they cannot walk.

Transports are not formed into squads. They are held here, in TRANSPORT_SLOTS fixed slots so that the operational layer can name one, and lent out a lift at a time. The operational layer decides whether a squad is carried and by which slot; the economy asks for a builder to be carried to ground it cannot walk to. What is worked out here rather than decided is the rest: where the passengers are picked up, where the transport sets them down, and which requests have no transport able to serve them, which is the shortfall the economy builds transports against.

A lift is carried out on the game side (`agent/Lift.java`), which reports its phase every period, and it takes what fits aboard. A slot is busy from the lift being sent until the game reports it done or failed. A squad of which only part is across is lifted again for the rest, as a list of units, which the operational layer asks for (`Operations._carry`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ...wire import CargoKind, Lift, LiftPhase, NO_SQUAD
from .reach import Reach

#: The transports the layer holds, which is how many slots the operational layer chooses a transport among.
TRANSPORT_SLOTS = 4

#: The longest a lift is given from being sent to being done before the game side abandons it. Opening value: a crossing, a fetch and a second crossing of a large map.
LIFT_DEADLINE_MS = 120000

#: Periods a lift may go unreported before its slot is taken back. The game side reports every lift under way every period, so a lift missing for this long is one the game has forgotten, after a reconnection.
UNREPORTED_PERIODS = 10

#: Lift ids wrap within the wire's u16.
_LIFT_IDS = 0xFFFF


@dataclass
class TransportSlot:
    """One transport slot and what it is lent to."""

    slot: int
    unit: Optional[int] = None
    type_index: int = -1
    x: float = 0.0
    y: float = 0.0
    #: Slots of the transport filled now.
    aboard: int = 0
    #: The squad it carries, or NO_SQUAD for builders, and the units of a builder lift.
    squad: int = NO_SQUAD
    units: Tuple[int, ...] = ()
    region: int = -1
    #: Where the lift under way sets its passengers down, or None.
    drop: Optional[Tuple[float, float]] = None
    lift: int = -1
    phase: int = -1
    unreported: int = 0

    @property
    def free(self) -> bool:
        return self.unit is not None and self.lift < 0


@dataclass
class Shortfall:
    """Requests no transport could serve this period: how many, and the passenger types they were for."""

    count: int = 0
    passengers: Set[int] = field(default_factory=set)


class Logistics:
    def __init__(self, catalogue) -> None:
        self.catalogue = catalogue
        self.reach: Optional[Reach] = None
        self.slots: List[TransportSlot] = [TransportSlot(slot=k) for k in range(TRANSPORT_SLOTS)]
        self.shortfall = Shortfall()
        self._rows: List[Lift] = []
        self._next_id = 1

    # The period.

    def update(self, view) -> None:
        """Takes in the transports on the board and the lifts the game reports, and frees the slots whose lift has ended or whose transport is gone."""
        self.shortfall = Shortfall()
        self._rows = []
        transports = sorted((s for s in view.transports if s.unit.built >= 255), key=lambda s: s.unit.id)
        present = {s.unit.id: s for s in transports}
        for slot in self.slots:
            if slot.unit is not None and slot.unit not in present:
                self._clear(slot)
                slot.unit = None
        held = {slot.unit for slot in self.slots if slot.unit is not None}
        for sighting in transports:
            if sighting.unit.id in held:
                continue
            empty = next((slot for slot in self.slots if slot.unit is None), None)
            if empty is None:
                break
            empty.unit = sighting.unit.id
            held.add(sighting.unit.id)
        for slot in self.slots:
            if slot.unit is None:
                continue
            unit = present[slot.unit].unit
            slot.type_index, slot.x, slot.y, slot.aboard = unit.type_index, unit.x, unit.y, unit.aboard
        self._read_lifts(view)

    def _read_lifts(self, view) -> None:
        reported = {lift.lift: lift for lift in view.observation.lifts} if view.observation.lifts is not None else {}
        for slot in self.slots:
            if slot.lift < 0:
                continue
            state = reported.get(slot.lift)
            if state is None:
                slot.unreported += 1
                if slot.unreported >= UNREPORTED_PERIODS:
                    self._clear(slot)
                continue
            slot.unreported = 0
            slot.phase = state.phase
            if state.phase in (LiftPhase.DONE, LiftPhase.FAILED):
                self._clear(slot)

    def rows(self) -> List[Lift]:
        """The lift rows to send this period."""
        return list(self._rows)

    # What a squad can be carried by.

    def carriable(self, slot: TransportSlot, passengers: Sequence[int]) -> bool:
        """Whether the transport in the slot loads every one of these types."""
        kind = self.catalogue.kind(slot.type_index)
        if kind is None or not kind.transport:
            return False
        return all(index in kind.carries for index in passengers)

    def could_carry(self, passengers: Sequence[int]) -> bool:
        """Whether some transport type of the catalogue loads every one of these types, which a request for a transport to be built has to meet: no transport is built for a ship."""
        return any(getattr(kind, "transport", False) and all(index in kind.carries for index in passengers)
                   for kind in getattr(self.catalogue, "types", ()))

    def candidates(self, squad: int, members: Sequence[Tuple[int, str, float, float]], region: int) -> List[int]:
        """The slots whose transport could carry these members, given as type index, movement and position, to the region: it loads all of them, it can get to them and to the region's landing point, and it is free or already carrying this squad there."""
        if self.reach is None or not members:
            return []
        found = []
        for slot in self.slots:
            if slot.unit is None:
                continue
            if not (slot.free or (slot.squad == squad and slot.region == region)):
                continue
            if not self.carriable(slot, [m[0] for m in members]):
                continue
            if self._plan(slot, members, region) is not None:
                found.append(slot.slot)
        return found

    def options(self, squad: int, members: Sequence[Tuple[int, str, float, float]],
                regions: Sequence[int], walk_on: bool = False) -> Dict[int, List[int]]:
        """For each of these regions, the slots whose transport could carry all these members there, by the same test as `candidates`, with each slot's pick-up worked out once. With `walk_on`, only where every member set down at the drop can walk on to the region's centre, which is where a contract sends a squad."""
        found: Dict[int, List[int]] = {}
        if self.reach is None or not members:
            return found
        for slot in self.slots:
            if slot.unit is None or not (slot.free or slot.squad == squad):
                continue
            if not self.carriable(slot, [m[0] for m in members]) or self._pickup(slot, members) is None:
                continue
            for region in regions:
                if not slot.free and slot.region != region:
                    continue
                drop = self._drop(slot, region)
                if drop is None:
                    continue
                if walk_on and not all(self.reach.walkable(m[1], drop[0], drop[1], region) for m in members):
                    continue
                found.setdefault(region, []).append(slot.slot)
        return found

    def _plan(self, slot: TransportSlot, members: Sequence[Tuple[int, str, float, float]],
              region: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The pick-up and the drop point a lift by this slot would use, or None when there are none."""
        drop = self._drop(slot, region)
        if drop is None:
            return None
        pickup = self._pickup(slot, members)
        if pickup is None:
            return None
        return pickup, drop

    def _drop(self, slot: TransportSlot, region: int) -> Optional[Tuple[float, float]]:
        """The region's landing point for the slot's transport, when the transport can get to it."""
        kind = self.catalogue.kind(slot.type_index)
        if kind is None:
            return None
        drop = self.reach.landing(region, kind.movement)
        if drop is None or not self.reach.reachable(kind.movement, (slot.x, slot.y), drop):
            return None
        return drop

    def _pickup(self, slot: TransportSlot, members: Sequence[Tuple[int, str, float, float]]) -> Optional[Tuple[float, float]]:
        """Where the transport would collect the members: the place nearest the member at their middle that it and they can both reach."""
        kind = self.catalogue.kind(slot.type_index)
        if kind is None or not members:
            return None
        movement = kind.movement
        x = sum(m[2] for m in members) / len(members)
        y = sum(m[3] for m in members) / len(members)
        lead = min(members, key=lambda m: math.hypot(m[2] - x, m[3] - y))
        return self.reach.pickup(lead[1], (lead[2], lead[3]), movement, (slot.x, slot.y))

    def nearest(self, slots: Sequence[int], x: float, y: float) -> Optional[int]:
        """Of these slots, the one whose transport stands nearest the point."""
        if not slots:
            return None
        return min(slots, key=lambda k: (math.hypot(self.slots[k].x - x, self.slots[k].y - y), k))

    # Sending lifts.

    def lift_squad(self, squad: int, members: Sequence[Tuple[int, str, float, float]], slot: int, region: int,
                   now: int, units: Optional[Sequence[int]] = None) -> bool:
        """Lends the slot to carry the squad's members to the region, unless it is already doing so: the whole squad, or only `units` of it when some of it is across already. A slot busy with anything else is refused. Returns whether the squad is being carried."""
        held = self.slots[slot]
        if held.squad == squad and held.region == region and held.lift >= 0:
            return True
        if not held.free or not self.carriable(held, [m[0] for m in members]):
            return False
        plan = self._plan(held, members, region)
        if plan is None:
            return False
        pickup, drop = plan
        if units is None:
            self._send(held, CargoKind.SQUAD, [squad], pickup, region, drop, now)
        else:
            self._send(held, CargoKind.UNITS, list(units), pickup, region, drop, now)
            held.units = tuple(units)
        held.squad = squad
        return True

    def lift_units(self, units: Sequence[Tuple[int, int, str, float, float]], region: int, now: int) -> bool:
        """Carries these units, given as unit id, type index, movement and position, to the region by the nearest free slot able to; records a shortfall when there is none. Returns whether a lift went."""
        if self.reach is None or not units:
            return False
        members = [(u[1], u[2], u[3], u[4]) for u in units]
        able = [slot.slot for slot in self.slots if slot.free and self.carriable(slot, [m[0] for m in members])
                and self._plan(slot, members, region) is not None]
        x = sum(u[3] for u in units) / len(units)
        y = sum(u[4] for u in units) / len(units)
        chosen = self.nearest(able, x, y)
        if chosen is None:
            self.want([u[1] for u in units])
            return False
        held = self.slots[chosen]
        pickup, drop = self._plan(held, members, region)
        self._send(held, CargoKind.UNITS, [u[0] for u in units], pickup, region, drop, now)
        held.units = tuple(u[0] for u in units)
        return True

    def want(self, passengers: Sequence[int]) -> None:
        """Notes a request no transport could serve, for the economy to build against."""
        self.shortfall.count += 1
        self.shortfall.passengers.update(passengers)

    def release(self, squad: int) -> None:
        """Cancels whatever the squad is being carried by: its contract no longer needs the crossing."""
        for slot in self.slots:
            if slot.squad == squad and slot.lift >= 0:
                self._rows.append(Lift(lift=slot.lift, cancel=True))
                self._clear(slot)

    def lifting(self, squad: int) -> Optional[TransportSlot]:
        return next((slot for slot in self.slots if slot.squad == squad and slot.lift >= 0), None)

    def busy(self) -> List[TransportSlot]:
        """The slots carrying something, which is what an escort is sent to cover."""
        return [slot for slot in self.slots if slot.lift >= 0 and slot.unit is not None]

    def escortable(self, slot: TransportSlot, members: Sequence[Tuple[int, str, float, float]]) -> bool:
        """Whether every member, given as type index, movement and position, can follow the slot's transport: get to where it stands now and to within the landing reach of its drop point, which for a ship is the water off the shore it lands on, wherever the region's centre lies."""
        if self.reach is None:
            return True
        if slot.drop is None:
            return False
        return all(self.reach.reachable(m[1], (m[2], m[3]), (slot.x, slot.y))
                   and self.reach.approaches(m[1], (m[2], m[3]), slot.drop) for m in members)

    def _send(self, slot: TransportSlot, cargo_kind: CargoKind, cargo: List[int], pickup, region: int, drop,
              now: int) -> None:
        lift_id = self._next_id
        self._next_id = self._next_id % (_LIFT_IDS - 1) + 1
        self._rows.append(Lift(lift=lift_id, transports=[slot.unit], cargo_kind=cargo_kind, cargo=list(cargo),
                               pickup_x=pickup[0], pickup_y=pickup[1], drop_region=region,
                               drop_x=drop[0], drop_y=drop[1], deadline_ms=now + LIFT_DEADLINE_MS))
        slot.lift = lift_id
        slot.region = region
        slot.drop = (float(drop[0]), float(drop[1]))
        slot.phase = int(LiftPhase.APPROACH)
        slot.unreported = 0

    @staticmethod
    def _clear(slot: TransportSlot) -> None:
        slot.squad = NO_SQUAD
        slot.units = ()
        slot.region = -1
        slot.drop = None
        slot.lift = -1
        slot.phase = -1
        slot.unreported = 0
