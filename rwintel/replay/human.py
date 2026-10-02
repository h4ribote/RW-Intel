"""Turning a player's unit orders into the operational layer's decisions.

A person, or the built-in AI, does not issue contracts. They select units and send them somewhere, and the operational layer's decision -this squad, to that region, for this task, walking or carried- has to be read back out of that. The reading is done from the chain's own side: the script chain runs over the player's units as it would over its own, the organisation layer forms squads from them, and the operational layer asks for a decision exactly when it would ask for one in a match. What answers is not a network or a rule but what the player did next.

When the squad's members were loaded aboard a transport, the decision is a lift: the transport carrying the most of the squad's worth is the means, by the slot the chain's lift layer holds it in, and the region is where that transport set them down. Otherwise each member's destination is the last place the player sent it by the end of the window. Members vote for the region nearest their destination with the worth they carry, and the region with the most worth behind it is the decision; the share of worth behind it is how much the player's orders agreed. A member never sent anywhere votes for where it stands. The task is then read off the doctrine and the ground: a move back towards home out of a losing position is a withdrawal, a move onto ground that is ours and uncontested is holding it, and anything else is taking it, in whichever of those forms the doctrine allows.

Because the orders and the board describe the same units by the same identifiers, the two can be joined directly.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..control.policy.contracts import DOCTRINES, Doctrine, SquadRecord
from ..control.policy.encoding import plan_of
from ..control.policy.operations import WALK
from ..control.policy.judgement import ENCIRCLE_SHARE
from ..control.policy.view import WorldView
from ..learn.deciders import Choice
from ..wire import RegionState, Status, Task
from .commands import MOVEMENT_KINDS, NO_UNIT, Command
from .timing import Clock

#: Below this share of the squad's worth agreeing on one region, or on one transport, the player's orders say nothing coherent about the squad.
AGREEMENT_FLOOR = 0.5

#: Weight of a decision drawn from members the player had never sent anywhere. Standing where it was made is a choice too, but a weaker statement than an order.
IDLE_WEIGHT = 0.5

#: How the region of a decision was arrived at.
ORDERED = "orders"
STANDING = "standing"
IDLE = "idle"
LIFT = "lift"

#: Why a decision was not recorded: no coherent region or plan, a lift by a transport the lift layer holds in no slot, or a lift whose region or plan is closed.
REFUSED = "refused"
LIFT_UNSLOTTED = "lift_unslotted"
LIFT_MASKED = "lift_masked"

#: The kinds an order book takes beside the movement kinds: boarding a named transport, a transport taking a named unit aboard, an order to unload at a point, and the special actions that set down a transport's load and call that off.
LOAD_INTO = "loadInto"
LOAD_UP = "loadUp"
UNLOAD_AT = "unloadAt"
UNLOAD = "unload"
CANCEL_UNLOAD = "cancelUnload"
#: A movement order that names no particular kind, as the built-in AI's patrol, guard and follow orders arrive.
OTHER_MOVEMENT = "movement"


@dataclass(frozen=True)
class Destination:
    #: When the order was given, in the book's unit of time.
    time: int
    x: float
    y: float
    #: The unit followed or attacked, whose position at the time of the decision stands in for the place, or NO_UNIT.
    target: int


@dataclass(frozen=True)
class Load:
    time: int
    transport: int


class OrderBook:
    """Per unit, the movement orders one player gave it in the order given; per passenger, when it was loaded and into what; per transport, when it was told to unload and when that was called off.

    Time is in the caller's unit, frames for a replay and game milliseconds for the built-in AI, and is only ever compared with itself.
    """

    def __init__(self) -> None:
        self._moves: Dict[int, List[Destination]] = {}
        self._move_times: Dict[int, List[int]] = {}
        self._loads: Dict[int, List[Load]] = {}
        self._load_times: Dict[int, List[int]] = {}
        #: Per transport, (time, True for an unload or False for a cancel).
        self._unloads: Dict[int, List[Tuple[int, bool]]] = {}

    @classmethod
    def from_commands(cls, commands: Sequence[Command], player: int) -> "OrderBook":
        """The book of one player's recorded commands. An unload is the special action UNLOAD_ACTION over the transports in `units`; a command with the stopOrUndo switch set is not an order."""
        book = cls()
        for command in commands:
            if command.player != player or command.cancel:
                continue
            if command.unloads:
                book.add(command.frame, UNLOAD, command.units)
            elif command.cancels_unload:
                book.add(command.frame, CANCEL_UNLOAD, command.units)
            if command.order is not None:
                order = command.order
                book.add(command.frame, order.kind, command.units, order.x, order.y, order.target)
        return book

    def add(self, time: int, kind: Optional[str], units: Sequence[int], x: float = math.nan, y: float = math.nan,
            target: int = NO_UNIT) -> None:
        """Files one order. Kinds other than the movement kinds and those named in this module are ignored."""
        if kind in MOVEMENT_KINDS or kind == OTHER_MOVEMENT:
            if target == NO_UNIT and (math.isnan(x) or math.isnan(y)):
                return
            destination = Destination(time=time, x=x, y=y, target=target)
            for unit in units:
                self._insert(self._moves, self._move_times, unit, time, destination)
        elif kind == LOAD_INTO and target != NO_UNIT:
            for unit in units:
                self._insert(self._loads, self._load_times, unit, time, Load(time=time, transport=target))
        elif kind == LOAD_UP and target != NO_UNIT and units:
            # The passenger is the target; of several transports told to take it aboard, the first is taken to be the one that does.
            self._insert(self._loads, self._load_times, target, time, Load(time=time, transport=units[0]))
        elif kind == UNLOAD_AT:
            destination = Destination(time=time, x=x, y=y, target=NO_UNIT)
            for unit in units:
                self._insert(self._moves, self._move_times, unit, time, destination)
                self._unloads.setdefault(unit, []).append((time, True))
        elif kind in (UNLOAD, CANCEL_UNLOAD):
            for unit in units:
                self._unloads.setdefault(unit, []).append((time, kind == UNLOAD))

    @staticmethod
    def _insert(entries: Dict[int, list], times: Dict[int, List[int]], unit: int, time: int, entry) -> None:
        keys = times.setdefault(unit, [])
        index = bisect.bisect_right(keys, time)
        keys.insert(index, time)
        entries.setdefault(unit, []).insert(index, entry)

    def last(self, unit: int, by: int) -> Optional[Destination]:
        """The last movement order the unit had been given by `by`."""
        times = self._move_times.get(unit)
        if not times:
            return None
        index = bisect.bisect_right(times, by) - 1
        return self._moves[unit][index] if index >= 0 else None

    def last_load(self, unit: int, by: int) -> Optional[Load]:
        """The last time the unit was loaded aboard a transport by `by`."""
        times = self._load_times.get(unit)
        if not times:
            return None
        index = bisect.bisect_right(times, by) - 1
        return self._loads[unit][index] if index >= 0 else None

    def unloaded(self, transport: int, after: int, by: int) -> Optional[int]:
        """When the transport set down its load after `after`, by `by`: its first unload in that span that no cancel followed before the next unload, or None."""
        pending: Optional[int] = None
        for time, unload in sorted(self._unloads.get(transport, ()), key=lambda event: event[0]):
            if time <= after or time > by:
                continue
            if unload:
                if pending is None:
                    pending = time
            else:
                pending = None
        return pending

    def destinations(self, unit: int, after: int, by: int) -> List[Destination]:
        """The movement orders given to the unit from `after` to `by`, both included."""
        times = self._move_times.get(unit, [])
        return self._moves.get(unit, [])[bisect.bisect_left(times, after):bisect.bisect_right(times, by)]

    def __len__(self) -> int:
        return sum(len(orders) for orders in self._moves.values())


@dataclass(frozen=True)
class RegionVote:
    region: RegionState
    agreement: float
    #: ORDERED when any member was sent somewhere inside the interval, STANDING when members are still bound for places sent to earlier, IDLE when none was ever sent.
    basis: str


@dataclass(frozen=True)
class LiftVote:
    region: RegionState
    #: The share of the squad's worth aboard the transport.
    agreement: float
    transport: int
    #: The slot of the chain's lift layer holding the transport, or -1 when none does.
    slot: int


@dataclass(frozen=True)
class Decision:
    """What the player's orders say about a squad: the region and plan, or a plan of -1 with the reason in `basis` when they say nothing that can be recorded."""

    region: int
    plan: int
    agreement: float
    basis: str

    @property
    def refused(self) -> bool:
        return self.plan < 0

    @property
    def weight(self) -> float:
        """What the decision's weight is multiplied by: how much the orders agreed, or IDLE_WEIGHT for members never sent anywhere."""
        return IDLE_WEIGHT if self.basis == IDLE else self.agreement


def nearest_region(regions: Sequence[RegionState], x: float, y: float) -> Optional[RegionState]:
    best, best_distance = None, math.inf
    for region in regions:
        distance = (region.x - x) ** 2 + (region.y - y) ** 2
        if distance < best_distance:
            best, best_distance = region, distance
    return best


def _place(destination: Destination, positions: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    if destination.target != NO_UNIT:
        return positions.get(destination.target, (destination.x, destination.y))
    return destination.x, destination.y


def infer_region(view: WorldView, squad: SquadRecord, orders: OrderBook, start: int, end: int) -> Optional[RegionVote]:
    """Where the player sent this squad's members, as a worth-weighted vote over regions; an order given at or after `start` is fresh."""
    positions: Dict[int, Tuple[float, float]] = {u.id: (u.x, u.y) for u in view.observation.unit_states}
    worth: Dict[int, float] = {s.unit.id: s.value for s in view.ours}
    votes: Dict[int, float] = {}
    total = 0.0
    fresh = ordered = False
    for member in squad.members:
        if member not in positions:
            continue
        value = max(worth.get(member, 0.0), 1.0)
        order = orders.last(member, end)
        if order is None:
            place = positions[member]
        else:
            ordered = True
            fresh = fresh or order.time >= start
            place = _place(order, positions)
        region = nearest_region(view.regions, *place)
        if region is None:
            continue
        votes[region.id] = votes.get(region.id, 0.0) + value
        total += value
    if not votes:
        return None
    winner = max(votes, key=lambda region_id: (votes[region_id], -region_id))
    return RegionVote(region=view.region(winner), agreement=votes[winner] / total,
                      basis=ORDERED if fresh else STANDING if ordered else IDLE)


def infer_lift(view: WorldView, squad: SquadRecord, book: OrderBook, start: int, end: int,
               logistics=None) -> Optional[LiftVote]:
    """Which transport carried this squad and where it set the squad down, when members were loaded in (start, end] or were still aboard at `start`."""
    positions: Dict[int, Tuple[float, float]] = {u.id: (u.x, u.y) for u in view.observation.unit_states}
    worth: Dict[int, float] = {s.unit.id: s.value for s in view.ours}
    carried: Dict[int, float] = {}
    loaded_at: Dict[int, int] = {}
    total = 0.0
    for member in squad.members:
        if member not in positions:
            continue
        value = max(worth.get(member, 0.0), 1.0)
        total += value
        load = book.last_load(member, end)
        if load is None:
            continue
        if load.time <= start and book.unloaded(load.transport, load.time, start) is not None:
            continue
        carried[load.transport] = carried.get(load.transport, 0.0) + value
        loaded_at[load.transport] = max(loaded_at.get(load.transport, load.time), load.time)
    if not carried or total <= 0:
        return None
    transport = max(carried, key=lambda unit: (carried[unit], -unit))
    agreement = carried[transport] / total
    if agreement < AGREEMENT_FLOOR:
        return None
    after = loaded_at[transport]
    unload = book.unloaded(transport, after, end)
    if unload is not None:
        drop = book.last(transport, unload)
    else:
        moves = book.destinations(transport, after, end)
        drop = moves[-1] if moves else None
    if drop is None:
        return None
    region = nearest_region(view.regions, *_place(drop, positions))
    if region is None:
        return None
    slots = logistics.slots if logistics is not None else ()
    slot = next((held.slot for held in slots if held.unit == transport), -1)
    return LiftVote(region=region, agreement=agreement, transport=transport, slot=slot)


def infer_task(doctrine: Doctrine, target: RegionState, squad: SquadRecord, view: WorldView) -> Optional[Task]:
    """The task a move to `target` amounts to, among those the doctrine allows."""
    allowed = DOCTRINES[doctrine].tasks
    if not allowed:
        return None
    here = nearest_region(view.regions, squad.x, squad.y)
    losing_here = here is not None and (here.enemy_value > here.our_value or squad.status == Status.LOSING)
    if (Task.WITHDRAW in allowed and here is not None and target.id != here.id and losing_here
            and target.distance_from_home < here.distance_from_home):
        return Task.WITHDRAW
    home = view.home
    holding = target.enemy_value <= 0 and (target.held_by_us > 0 or target.our_value > 0
                                           or (home is not None and target.id == home.id))
    if holding:
        preference: Tuple[Task, ...] = (Task.DEFEND, Task.ESCORT, Task.ATTACK, Task.RAID, Task.ENCIRCLE)
    else:
        # The test the script vanguard applies when it is not concentrating: surrounding is what a squad does with the odds to spare.
        strength = squad.value + target.our_value
        encircle = target.enemy_value > 0 and strength >= ENCIRCLE_SHARE * (strength + target.enemy_value)
        pressing = (Task.ENCIRCLE, Task.ATTACK) if encircle else (Task.ATTACK, Task.ENCIRCLE)
        preference = pressing + (Task.RAID, Task.DEFEND, Task.ESCORT)
    return next((task for task in preference if task in allowed), allowed[0])


def _open(region_mask: Sequence[float], region: int) -> bool:
    return 0 <= region < len(region_mask) and region_mask[region] > 0


def infer_decision(view: WorldView, squad: SquadRecord, book: OrderBook, start: int, end: int,
                   region_mask: Sequence[float], plan_masks: Sequence[Sequence[float]], logistics=None) -> Decision:
    """The operational decision the player's orders over (start, end] amount to for this squad: a lift when one carried most of its worth, otherwise where its members walked."""
    lift = infer_lift(view, squad, book, start, end, logistics)
    if lift is not None:
        if lift.slot < 0:
            return Decision(region=lift.region.id, plan=-1, agreement=lift.agreement, basis=LIFT_UNSLOTTED)
        task = infer_task(squad.doctrine, lift.region, squad, view)
        plan = plan_of(task, lift.slot) if task is not None else -1
        if task is None:
            return Decision(region=lift.region.id, plan=-1, agreement=lift.agreement, basis=REFUSED)
        if not _open(region_mask, lift.region.id) or plan_masks[lift.region.id][plan] <= 0:
            return Decision(region=lift.region.id, plan=-1, agreement=lift.agreement, basis=LIFT_MASKED)
        return Decision(region=lift.region.id, plan=plan, agreement=lift.agreement, basis=LIFT)
    vote = infer_region(view, squad, book, start, end)
    if vote is None or vote.region is None or vote.agreement < AGREEMENT_FLOOR:
        return Decision(region=-1, plan=-1, agreement=vote.agreement if vote is not None else 0.0, basis=REFUSED)
    if not _open(region_mask, vote.region.id):
        return Decision(region=vote.region.id, plan=-1, agreement=vote.agreement, basis=REFUSED)
    task = infer_task(squad.doctrine, vote.region, squad, view)
    plan = plan_of(task, WALK) if task is not None else -1
    if plan < 0 or plan_masks[vote.region.id][plan] <= 0:
        return Decision(region=vote.region.id, plan=-1, agreement=vote.agreement, basis=REFUSED)
    return Decision(region=vote.region.id, plan=plan, agreement=vote.agreement, basis=vote.basis)


def new_counts() -> Dict[str, int]:
    """Decisions by basis and refusals by reason, for a run's report."""
    return {ORDERED: 0, STANDING: 0, IDLE: 0, LIFT: 0, REFUSED: 0, LIFT_UNSLOTTED: 0, LIFT_MASKED: 0}


class HumanOperations:
    """Answers the operational layer with what the person in a replay did with the squad in question."""

    #: What this decider plays is the answer to learn, since the person is the teacher; a recording layer writes it as the label instead of asking its own judge.
    teaches = True

    def __init__(self, orders: OrderBook, clock: Clock, review_ms: int, weight: float = 1.0) -> None:
        self.orders = orders
        self.clock = clock
        self.review_ms = review_ms
        #: Multiplies the weight every decision is written with.
        self.weight = weight
        self.counts: Dict[str, int] = new_counts()

    def choose(self, state, slot, region_mask, plan_masks) -> Optional[Choice]:
        raise TypeError("a person's decision is read off the board, so it is asked for with choose_on_board")

    def choose_on_board(self, view: WorldView, squad: SquadRecord, state, region_mask, plan_masks,
                        now_ms: int, logistics=None) -> Optional[Choice]:
        """The decision the person's orders over the review interval that starts now amount to, or None when they say nothing that can be recorded."""
        now_frame = view.observation.frame
        end_frame = max(now_frame, self.clock.frame_at(now_ms + self.review_ms))
        decision = infer_decision(view, squad, self.orders, now_frame, end_frame, region_mask, plan_masks, logistics)
        self.counts[decision.basis] += 1
        if decision.refused:
            return None
        return Choice(action=decision.region, second=decision.plan,
                      meta={"source": "human", "weight": round(self.weight * decision.weight, 4),
                            "agreement": round(decision.agreement, 4), "basis": decision.basis})
