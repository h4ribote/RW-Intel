"""Which squad goes where, under what task, by what means, at what price, by when.

This is the layer the design intends to learn first against a frozen tactical layer, so what is written here is not merely a placeholder: it is the baseline a learnt operational policy has to beat, and the teacher it is first fitted to. The shape of its decision is the action space that policy will be given -one squad, one region, one task, one means, one stance, one budget, one deadline- and where and what are decided by the operational judge (`judgement.OperationsJudge`) from the encoded board alone, the same numbers a network reads. A rule that read the board differently would be a teacher the network could only partly imitate.

Where a squad may be sent is what it can get to: a region every member can walk to, or one a transport in a slot of the lift layer (`logistics.Logistics`) can carry the whole squad to. Of a region the squad can walk to it walks; otherwise it goes by the free transport able to carry it that stands nearest, and the lift layer works out the pick-up, the drop and as many trips as the squad needs. A transport carrying a squad is covered by a fleet or an air wing with nothing better to do, sent to escort it.

A squad is decided about when something calls for it: it has no contract, its mission report says the errand stalled, is going badly, is finished (other than a garrison holding its ground), is out of time or cannot be reached, or REVIEW_MS has passed since it was last decided about. In between it keeps the errand it holds, and a squad being carried keeps it until it is set down. Two regions of nearly equal worth trade places whenever a shot lands in either of them, and a squad re-tasked on that difference spends the match walking between them and arriving at neither; and on the game side the losses a mission is judged by are cut from the moment of issue, so restating a contract every period would keep resetting the measurement that decides whether the mission is going badly.

A contract is re-issued only when it actually differs from the one the squad already holds, for the same reason. An expired one is re-issued precisely so that its clock and its losses start again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ...wire import RegionState
from ...wire.action import Stance, Status, TargetKind, Task
from .catalogue import Catalogue
from .combat import CombatTable
from .contracts import DOCTRINES, Doctrine, MissionReport, OperationsOrders, Shortfall, SquadRecord, TaskContract
from .encoding import (
    Access,
    Forces,
    TransportView,
    forces,
    means_of,
    operational_state,
    plan_masks,
    regions_of,
    task_of,
    transport_distance,
)
from .judgement import HOLDING_TASKS, OperationsJudge
from .options import Options
from .reach import members_reach
from .view import WorldView

#: Health below which a squad is given no mission at all. Set at the merge threshold so that a squad reported worn out is one the organisation layer is already entitled to fold into another. Opening value.
WORN_OUT_HEALTH = 0.4

#: The longest a squad keeps an errand nothing has called into question before it is decided about again, in game time. Opening value.
REVIEW_MS = 30000

#: Assumed marching pace in world units per second, used only to turn a distance into a deadline. Speed is not in the type catalogue, so this is an estimate over the mixed pace of a squad rather than a reading. Opening value.
MARCH_SPEED = 40.0

#: Multiplier on the straight line distance, standing in for terrain and for a squad moving no faster than its slowest member. Opening value.
MARCH_SLACK = 1.5

#: Time allowed on top of the march for the mission itself, since arriving is not finishing. Opening value.
ENGAGEMENT_MS = 60000

#: Time allowed on top of that for being carried: the transport coming, the loading and the setting down. Opening value.
LIFT_MS = 45000

#: The least a mission may be funded with. A squad handed nothing to spend is a squad the tactical layer will pull out of the first exchange. Opening value.
MINIMUM_BUDGET = 200.0

#: How far a budget may drift from the one already issued before it is worth re-issuing, given that re-issuing resets the losses the mission is judged by. Opening value.
BUDGET_TOLERANCE = 0.25

#: Doctrines whose squads cover a transport carrying a squad when they have nothing better to do.
ESCORTS = (Doctrine.FLEET, Doctrine.AIRWING)

#: The means of a squad that goes on its own.
WALK = -1

#: Priority at or above which a region a squad can reach by no means makes the squad ask for a transport. Opening value.
WANT_PRIORITY = 0.5


def running(squad: SquadRecord) -> bool:
    """Whether the squad's errand is still in progress: active, being carried, or complete on a task whose completion is holding."""
    held = squad.contract
    if squad.status in (Status.ACTIVE, Status.AWAITING_LIFT, Status.LIFTING):
        return True
    return squad.status is Status.COMPLETE and held is not None and held.task in HOLDING_TASKS


#: Stance implied by each task. Attacking, surrounding and raiding are all pressing, so they go in aggressive; defending guards the ground rather than chasing off it; withdrawing returns fire only, so that a squad moving away answers what shoots at it without turning to chase anything.
STANCE_FOR: Dict[Task, Stance] = {
    Task.ATTACK: Stance.AGGRESSIVE,
    Task.ENCIRCLE: Stance.AGGRESSIVE,
    Task.RAID: Stance.AGGRESSIVE,
    Task.DEFEND: Stance.GUARD_AREA,
    Task.ESCORT: Stance.GUARD_AREA,
    Task.WITHDRAW: Stance.RETURN_FIRE,
}


@dataclass
class _Plan:
    """A mission decided but not yet priced. Budgets divide the strategic allowance among the missions actually being run, so none of them can be costed until all of them are known."""

    squad: SquadRecord
    task: Task
    region: RegionState
    #: True when the contract has to go out even if it reads the same as the one held, because its clock or its loss count needs restarting.
    forced: bool = False
    #: WALK, or the transport slot that carries the squad.
    means: int = WALK
    target_kind: TargetKind = TargetKind.REGION
    target: int = 0


@dataclass
class _Reach:
    """Where one squad can be sent: the regions all of it can walk to, and for each other region the transport slots that could carry all of it there."""

    walk: Set[int]
    lift: Dict[int, List[int]]
    #: The members that are not aboard anything, as type index, movement and position, and their unit ids in the same order.
    members: List[Tuple[int, str, float, float]]
    ids: List[int]


class Operations:
    """The operational layer. Reads the board, the strategic orders, the squads the organisation layer has formed and what the tactical layer reports back, and answers with contracts and with what it could not do for want of strength."""

    def __init__(self, session, catalogue: Catalogue, options: Options = Options(), review_ms: int = REVIEW_MS,
                 logistics=None) -> None:
        self.session = session
        self.catalogue = catalogue
        self.options = options
        self.review_ms = review_ms
        #: The lift layer, which holds the transports a squad may be carried by. Without one every squad walks.
        self.logistics = logistics
        self.judge = OperationsJudge(concentrate=options.concentrate, cover=options.cover, tuning=options.tuning)
        #: What a fight in each region is expected to come to, which the encoding carries for the judge to read.
        self.combat = CombatTable.load(catalogue, tuning=options.tuning)
        #: Squads a human held as of the last decision. A squad coming back is left alone for one period, because its composition and its position are both unknown to the command chain until it has been seen once under machine command again.
        self.human_held: set = set()
        #: When each squad was last decided about.
        self.decided_at: Dict[int, int] = {}
        #: The last choice made for each squad, as decision time, region slot, plan and whether its plan mask had that plan open, which the chain's debug log reads when a contract comes back unreachable.
        self.picked: Dict[int, Tuple[int, int, int, bool]] = {}
        #: Which squad escorts each busy transport slot.
        self.escorts: Dict[int, int] = {}
        self._spawns = tuple(region.id for region in getattr(session, "regions", ()) or () if region.spawn)
        self._now = 0
        self._view: Optional[WorldView] = None
        self._orders: Optional[OperationsOrders] = None
        self._squads: List[SquadRecord] = []
        self._forces: Optional[Forces] = None
        self._reach: Dict[int, _Reach] = {}
        #: Transport slots a squad has been given this period, so that two squads decided about in one period do not pick the same free transport.
        self._claimed: Dict[int, int] = {}

    def decide(self, view: WorldView, orders: OperationsOrders, squads: List[SquadRecord],
               reports: List[MissionReport], game_time_ms: int) -> Tuple[List[TaskContract], List[Shortfall]]:
        self._now = game_time_ms
        self._view, self._orders, self._squads = view, orders, squads
        self._forces = forces(view, squads)
        self._reach = {}
        by_squad: Dict[int, MissionReport] = {report.squad: report for report in reports}
        shortfalls: List[Shortfall] = []
        plans: List[_Plan] = []
        held_now: set = set()
        present = {squad.id for squad in squads}
        self.decided_at = {squad_id: at for squad_id, at in self.decided_at.items() if squad_id in present}
        self.picked = {squad_id: pick for squad_id, pick in self.picked.items() if squad_id in present}
        busy = {slot.slot for slot in self.logistics.busy()} if self.logistics is not None else set()
        able = {squad.id for squad in squads if squad.ours_to_task and squad.health >= WORN_OUT_HEALTH}
        self.escorts = {slot: squad for slot, squad in self.escorts.items() if slot in busy and squad in able}
        self._claimed = {}

        for squad in squads:
            # A doctrine with no task of its own is one this layer does not command. That is the engineers: the economy drives them, and a contract here would override the placement a builder is walking to.
            if not DOCTRINES[squad.doctrine].tasks:
                continue
            if not squad.ours_to_task:
                held_now.add(squad.id)
                continue
            missing = self._missing(squad)
            worn = squad.health < WORN_OUT_HEALTH
            if missing > 0 or worn:
                shortfalls.append(Shortfall(squad=squad.id, missing=missing, worn_out=worn))
            if worn:
                # No contract at all rather than a withdrawal: half the doctrines may not be told to withdraw, and a squad this far gone is one the organisation layer is about to merge or disband. Its standing contract keeps it fighting on the tactical layer's own judgement until then.
                continue
            if squad.id in self.human_held:
                continue
            # A squad that has just taken in units somebody else was moving waits a period. What was done with them is unknown here, and a contract written now would be written about a composition and a position this layer has not seen; the squad holds the stance it has until the next period, by which time the organisation layer has reported what it actually consists of.
            if squad.settling:
                continue
            plan = self._escort(view, squad, by_squad.get(squad.id)) or self._plan(view, squad, by_squad.get(squad.id))
            if plan is not None:
                plans.append(plan)

        self.human_held = held_now
        contracts = self._price(plans, orders, game_time_ms)
        self._carry(plans, game_time_ms)
        return contracts, shortfalls

    # What a squad is sent to do.

    def _plan(self, view: WorldView, squad: SquadRecord, report: Optional[MissionReport]) -> Optional[_Plan]:
        status = report.status if report is not None else Status.ACTIVE
        chosen = self._target(view, squad)
        if chosen is None:
            return None
        task, region, means = chosen
        if task not in DOCTRINES[squad.doctrine].tasks:
            return None
        # An expired contract goes out again even when it reads identically: what it needs is a fresh deadline and a fresh baseline for the losses it is measured against. So does a withdrawal from a mission going badly.
        forced = status == Status.EXPIRED or (status == Status.LOSING and task == Task.WITHDRAW)
        plan = _Plan(squad=squad, task=task, region=region, forced=forced, means=means)
        held = squad.contract
        if held is not None and held.task == task and held.target_region == region.id:
            # An errand kept is kept whole, including what it targets.
            plan.target_kind, plan.target = held.target_kind, held.target
        return plan

    def keeps(self, squad: SquadRecord, view: WorldView) -> Optional[RegionState]:
        """The region of the errand this squad goes on with, or None when it is to be decided about. A squad this layer has never decided about is decided about whatever it holds: an errand somebody else handed it is not one this layer chose. A squad being carried goes on with its errand whatever the clock says, since deciding about it again would strand it half way."""
        held = squad.contract
        if held is None or not running(squad):
            return None
        decided = self.decided_at.get(squad.id)
        if decided is None:
            return None
        # An escort of a transport ends when the transport has nothing more to carry.
        if held.target_kind == TargetKind.UNIT and squad.id not in self.escorts.values():
            return None
        carried = squad.status in (Status.AWAITING_LIFT, Status.LIFTING)
        if not carried and self._now - decided >= self.review_ms:
            return None
        return view.region(held.target_region)

    def state(self, squad: SquadRecord) -> List[float]:
        """The board as the encoding writes it for this squad, which is everything where it goes, what it does and how it gets there are decided from."""
        return operational_state(self._view, self._orders, self._squads, self._now, self._spawns,
                                 squad=squad, combat=self.combat, present=self._forces,
                                 access=self.access(self._view, squad))

    def access(self, view: WorldView, squad: SquadRecord) -> Access:
        """Where the squad can get to and by which transport, as the encoding reads it: a transport another squad has been given this period is open to nobody else."""
        reachable = self.reach(view, squad)
        claimed = {slot for slot, holder in self._claimed.items() if holder != squad.id}
        lift = {region: [k for k in slots if k not in claimed] for region, slots in reachable.lift.items()}
        lift = {region: slots for region, slots in lift.items() if slots}
        transports: List[TransportView] = []
        coastal: set = set()
        if self.logistics is not None:
            terrain = self.logistics.reach
            if terrain is not None:
                coastal = {region for region, row in terrain.regions.items() if row.coastal}
            types = [m[0] for m in reachable.members]
            for slot in self.logistics.slots:
                kind = self.catalogue.kind(slot.type_index) if slot.unit is not None else None
                if kind is None:
                    transports.append(TransportView(slot=slot.slot))
                    continue
                transports.append(TransportView(
                    slot=slot.slot, valid=True, x=slot.x, y=slot.y, aboard=slot.aboard, capacity=max(0, kind.capacity),
                    movement=kind.movement, busy=(not slot.free and slot.squad != squad.id) or slot.slot in claimed,
                    mine=slot.squad == squad.id,
                    carries=bool(types) and self.logistics.carriable(slot, types)))
        return Access(walk=set(reachable.walk), lift=lift, coastal=coastal, transports=transports)

    def reach(self, view: WorldView, squad: SquadRecord) -> _Reach:
        """Where the squad can be sent, worked out once a period."""
        found = self._reach.get(squad.id)
        if found is not None:
            return found
        by_id = {s.unit.id: s for s in view.ours}
        standing = [s for s in (by_id.get(m) for m in squad.members)
                    if s is not None and s.kind is not None and not s.unit.carrier]
        members = [(s.unit.type_index, s.kind.movement, s.unit.x, s.unit.y) for s in standing]
        ids = [s.unit.id for s in standing]
        terrain = self.logistics.reach if self.logistics is not None else None
        live = [region.id for region in view.regions]
        if terrain is None or not members:
            found = _Reach(walk=set(live), lift={}, members=members, ids=ids)
        else:
            walking = [(m[1], m[2], m[3]) for m in members]
            walk = {r for r in live if members_reach(terrain, walking, r)}
            lift = self.logistics.options(squad.id, members, [r for r in live if r not in walk], walk_on=True)
            found = _Reach(walk=walk, lift=lift, members=members, ids=ids)
            # Ground the strategic layer wants that this squad can neither walk to nor be carried to is a transport the economy should build, when some transport type would load the squad.
            priorities = self._orders.priorities if self._orders is not None else {}
            types = [m[0] for m in members]
            stranded = any(priorities.get(r, 0.0) >= WANT_PRIORITY for r in live if r not in walk and r not in lift)
            if stranded and self.logistics.could_carry(types):
                self.logistics.want(types)
        self._reach[squad.id] = found
        return found

    def plan_masks(self, view: WorldView, squad: SquadRecord) -> List[List[float]]:
        """For every region slot, the plans this squad may be given there: a task its doctrine allows, by walking or by a transport that can carry it there and that no other squad has been given this period."""
        return plan_masks(view, squad.doctrine, self.access(view, squad))

    def region_mask(self, view: WorldView, squad: SquadRecord) -> List[float]:
        """Which region slots this squad may be sent to: those on the map that it can walk to or be carried to."""
        return regions_of(self.plan_masks(view, squad))

    def means(self, view: WorldView, squad: SquadRecord, region: int) -> int:
        """How the squad gets to the region: WALK where it can, otherwise the slot already carrying it there, otherwise the transport able to carry it that stands nearest it and that no other squad has been given this period."""
        reachable = self.reach(view, squad)
        if region in reachable.walk:
            return WALK
        slots = reachable.lift.get(region, [])
        carrying = self.logistics.lifting(squad.id) if self.logistics is not None else None
        if carrying is not None and carrying.slot in slots and carrying.region == region:
            return carrying.slot
        open_slots = [slot for slot in slots if self._claimed.get(slot, squad.id) == squad.id]
        if self.logistics is None or not open_slots:
            return WALK
        # Nearest as the encoding reads it, which is the distance the judge compares.
        held = self.logistics.slots
        chosen = min(open_slots, key=lambda k: (transport_distance(held[k].x - squad.x, held[k].y - squad.y), k))
        self._claimed[chosen] = squad.id
        return chosen

    def _target(self, view: WorldView, squad: SquadRecord) -> Optional[Tuple[Task, RegionState, int]]:
        kept = self.keeps(squad, view)
        if kept is not None:
            # The means is worked out afresh: a squad set down across the water walks from there, and one still waiting keeps the slot that is coming for it.
            return squad.contract.task, kept, self.means(view, squad, kept.id)
        masks = self.plan_masks(view, squad)
        regions = regions_of(masks)
        if not any(regions):
            return None
        chosen = self._choose(self.state(squad), squad, regions, masks)
        if chosen is None:
            return None
        region = view.region(chosen[0])
        if region is None:
            return None
        self.decided_at[squad.id] = self._now
        row = masks[chosen[0]]
        self.picked[squad.id] = (self._now, chosen[0], chosen[1], 0 <= chosen[1] < len(row) and row[chosen[1]] > 0)
        means = means_of(chosen[1])
        if means != WALK:
            self._claimed[means] = squad.id
        return Task(task_of(chosen[1])), region, means

    def _choose(self, state: List[float], squad: SquadRecord, regions: List[float],
                masks: List[List[float]]) -> Optional[Tuple[int, int]]:
        """Where and which plan, as region slot and plan index: the judge's answer. The one decision a learnt layer replaces."""
        return self.judge.choose(state, squad.id, regions, masks)

    def _escort(self, view: WorldView, squad: SquadRecord, report: Optional[MissionReport]) -> Optional[_Plan]:
        """An escort for a busy transport from a fleet or air wing whose errand is not running, or the escort it already holds while the transport stays busy."""
        if self.logistics is None or squad.doctrine not in ESCORTS or Task.ESCORT not in DOCTRINES[squad.doctrine].tasks:
            return None
        held = next((slot for slot, escort in self.escorts.items() if escort == squad.id), None)
        if held is None:
            if squad.contract is not None and running(squad):
                return None
            open_slots = [slot for slot in self.logistics.busy() if slot.slot not in self.escorts]
            reachable = self.reach(view, squad)
            # The escort's target is the transport itself, so what counts is where it stands and where it sets down, not the centre of the region it is bound for.
            open_slots = [slot for slot in open_slots if self.logistics.escortable(slot, reachable.members)]
            if not open_slots:
                return None
            chosen = min(open_slots, key=lambda slot: (math.hypot(slot.x - squad.x, slot.y - squad.y), slot.slot))
            self.escorts[chosen.slot] = squad.id
            held = chosen.slot
        slot = self.logistics.slots[held]
        region = view.region(slot.region)
        if region is None or slot.unit is None:
            return None
        self.decided_at[squad.id] = self._now
        return _Plan(squad=squad, task=Task.ESCORT, region=region, target_kind=TargetKind.UNIT, target=slot.unit)

    # Carrying the squads that go by transport.

    def _carry(self, plans: List[_Plan], now: int) -> None:
        """Has the lift layer carry the members of every squad whose means is a transport that cannot walk to its region, and cancels the lift of one that no longer needs it. The whole squad is lifted as a squad; once some of it is across, the rest goes as a list of units, trip by trip, while the members across carry out the contract."""
        if self.logistics is None:
            return
        terrain = self.logistics.reach
        for plan in plans:
            if plan.means == WALK:
                if self.logistics.lifting(plan.squad.id) is not None and plan.squad.status not in (Status.AWAITING_LIFT, Status.LIFTING):
                    self.logistics.release(plan.squad.id)
                continue
            reachable = self.reach(self._view, plan.squad)
            left = [(unit, member) for unit, member in zip(reachable.ids, reachable.members)
                    if terrain is None or not terrain.walkable(member[1], member[2], member[3], plan.region.id)]
            units = None if len(left) == len(reachable.ids) else [unit for unit, _ in left]
            if not left or not self.logistics.lift_squad(plan.squad.id, [member for _, member in left], plan.means,
                                                         plan.region.id, now, units=units):
                plan.squad.contract.means = WALK

    # What a mission may cost and how long it has.

    def _price(self, plans: List[_Plan], orders: OperationsOrders, game_time_ms: int) -> List[TaskContract]:
        """Cuts the strategic loss allowance among the missions being run, in proportion to what each squad is worth, and issues only the contracts that differ from the ones already held."""
        total = sum(plan.squad.value for plan in plans)
        contracts: List[TaskContract] = []
        for plan in plans:
            portion = plan.squad.value / total if total > 0 else (1.0 / len(plans) if plans else 0.0)
            budget = max(MINIMUM_BUDGET, orders.loss_allowance * portion)
            stance = STANCE_FOR[plan.task]
            contract = TaskContract(
                squad=plan.squad.id,
                task=plan.task,
                target_region=plan.region.id,
                stance=stance,
                cost_budget=budget,
                deadline_ms=game_time_ms + self._march_ms(plan),
                issued_at_ms=game_time_ms,
                target_kind=plan.target_kind,
                target=plan.target,
                means=plan.means,
            )
            if not plan.forced and not self._changed(plan.squad.contract, contract):
                plan.squad.contract.means = plan.means
                continue
            # The record carries what the squad is under, so that the next period compares against what was actually sent rather than against what was last thought.
            plan.squad.contract = contract
            contracts.append(contract)
        return contracts

    @staticmethod
    def _march_ms(plan: _Plan) -> int:
        distance = math.hypot(plan.region.x - plan.squad.x, plan.region.y - plan.squad.y)
        carried = LIFT_MS if plan.means != WALK else 0
        return int(distance * MARCH_SLACK / MARCH_SPEED * 1000) + ENGAGEMENT_MS + carried

    @staticmethod
    def _changed(held: Optional[TaskContract], fresh: TaskContract) -> bool:
        if held is None:
            return True
        if (held.task, held.target_region, held.stance, held.target_kind, held.target) != \
                (fresh.task, fresh.target_region, fresh.stance, fresh.target_kind, fresh.target):
            return True
        # The deadline is deliberately not compared: it moves every period by construction, and re-issuing on that alone would reset the losses on every mission for ever.
        reference = max(held.cost_budget, MINIMUM_BUDGET)
        return abs(fresh.cost_budget - held.cost_budget) / reference > BUDGET_TOLERANCE

    # What the organisation layer is told.

    def _missing(self, squad: SquadRecord) -> float:
        """Credits of strength between what the squad is worth and what a full one of its doctrine would cost, priced at the cheapest type filling each role."""
        establishment = 0.0
        for role, count in DOCTRINES[squad.doctrine].establishment.items():
            kind = self.catalogue.cheapest(role)
            if kind is not None:
                establishment += count * float(kind.price)
        return max(0.0, establishment - squad.value)
