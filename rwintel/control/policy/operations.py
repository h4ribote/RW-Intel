"""Which squad goes where, under what task, at what price, by when.

This is the layer the design intends to learn first against a frozen tactical layer, so what is written here is not merely a placeholder: it is the baseline a learnt operational policy has to beat, and the shape of its decision is the action space that policy will be given. That shape is fixed — one squad, one region, one task, one stance, one budget, one deadline — and everything below is only the rule that fills it in. A rule that read the board differently, or that emitted a target the contract has no field for, would be measuring the learnt policy against something the learnt policy cannot express.

Targets are scored rather than matched. The strategic layer says what a region is worth in the abstract, but worth is not reachability: a region twice as valuable and four times as far away is the worse errand, and a region already full of our own strength does not need a second squad. So the priority the strategic layer hands down is discounted by what the march costs and by what is already standing there, and each doctrine then reads the result through what it is for. The weights of those discounts are opening values; which side of them a region falls on is exactly the sort of thing the design says to settle by measurement.

A contract is re-issued only when it actually differs from the one the squad already holds. On the game side the losses a mission is judged by are cut from the moment of issue, so restating an unchanged contract every period would keep resetting the very measurement that decides whether the mission is going badly, and no squad would ever be reported as losing. That makes re-issue a real decision rather than bookkeeping, and it is why the mission reports coming back up are the trigger for change: a stalled or losing mission is a reason to send the squad elsewhere, a complete one frees it for the next target, and an expired one is re-issued precisely so that its clock and its losses start again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ...wire import RegionState
from ...wire.action import Stance, Status, Task
from .catalogue import Catalogue
from .contracts import DOCTRINES, Doctrine, MissionReport, OperationsOrders, Shortfall, SquadRecord, TaskContract
from .view import WorldView

#: How much of a region's priority a march of REACH_SCALE is worth giving up, which is what stops the layer from sending every squad at the far corner of the map because the strategic layer called it valuable. Opening value.
REACH_COST = 0.4

#: The distance REACH_COST is quoted against, in world units. Regions are cut at a merge distance of 400, so this is a march of several regions rather than a step to the next one. Opening value.
REACH_SCALE = 2000.0

#: How much our own strength already standing in a region discounts it as a target, so that squads spread across the board instead of piling onto the same place. Opening value.
CROWDING_COST = 0.3

#: What one resource point in a region adds to holding or taking it, on the same scale as a strategic priority. Opening value.
RESOURCE_WORTH = 0.15

#: How much enemy strength in a region we hold raises it as something to garrison. Opening value.
THREAT_WEIGHT = 0.3

#: How much enemy strength in a region puts a raid off it, raiders being the one doctrine that is meant to avoid a fight rather than win one. Opening value.
RAID_RISK = 0.8

#: The odds of our own worth against the enemy's at which a vanguard surrounds rather than pushes straight in. Opening value.
ENCIRCLE_ODDS = 2.0

#: Health below which a squad is given no mission at all. Set at the merge threshold so that a squad reported worn out is one the organisation layer is already entitled to fold into another. Opening value.
WORN_OUT_HEALTH = 0.4

#: Assumed marching pace in world units per second, used only to turn a distance into a deadline. Speed is not in the type catalogue, so this is an estimate over the mixed pace of a squad rather than a reading. Opening value.
MARCH_SPEED = 40.0

#: Multiplier on the straight line distance, standing in for terrain and for a squad moving no faster than its slowest member. Opening value.
MARCH_SLACK = 1.5

#: Time allowed on top of the march for the mission itself, since arriving is not finishing. Opening value.
ENGAGEMENT_MS = 60000

#: The least a mission may be funded with. A squad handed nothing to spend is a squad the tactical layer will pull out of the first exchange. Opening value.
MINIMUM_BUDGET = 200.0

#: How far a budget may drift from the one already issued before it is worth re-issuing, given that re-issuing resets the losses the mission is judged by. Opening value.
BUDGET_TOLERANCE = 0.25

#: Stance implied by each task. Attacking, surrounding and raiding are all pressing, so they go in aggressive; defending guards the ground rather than chasing off it; withdrawing holds fire so that a squad breaking off does not stop to trade on the way out.
STANCE_FOR: Dict[Task, Stance] = {
    Task.ATTACK: Stance.AGGRESSIVE,
    Task.ENCIRCLE: Stance.AGGRESSIVE,
    Task.RAID: Stance.AGGRESSIVE,
    Task.DEFEND: Stance.GUARD_AREA,
    Task.ESCORT: Stance.GUARD_AREA,
    Task.WITHDRAW: Stance.HOLD_FIRE,
}


@dataclass
class _Plan:
    """A mission decided but not yet priced. Budgets divide the strategic allowance among the missions actually being run, so none of them can be costed until all of them are known."""

    squad: SquadRecord
    task: Task
    region: RegionState
    #: True when the contract has to go out even if it reads the same as the one held, because its clock or its loss count needs restarting.
    forced: bool = False


def _share(part: float, against: float) -> float:
    """One worth against another, bounded to 0 to 1 so that a term added to a priority stays on the same scale as a priority however large the armies get."""
    total = part + against
    return part / total if total > 0 else 0.0


class Operations:
    """The operational layer. Reads the board, the strategic orders, the squads the organisation layer has formed and what the tactical layer reports back, and answers with contracts and with what it could not do for want of strength."""

    def __init__(self, session, catalogue: Catalogue, crowding: float = CROWDING_COST) -> None:
        self.session = session
        self.catalogue = catalogue
        #: How much our own strength already standing in a region discounts it as a target. Held on the instance rather than read from the constant so that the one term which decides whether the layer spreads or masses can be turned off and measured, which is what the operations arena's massed arm does: with it at nought the same ladder sends every squad at the single best region, and the difference between the two runs on the same boards is what the spreading rule is worth. Nothing in a match changes it; it is a knob for the instrument.
        self.crowding = crowding
        #: Squads a human held as of the last decision. A squad coming back is left alone for one period, because its composition and its position are both unknown to the command chain until it has been seen once under machine command again.
        self.human_held: set = set()

    def decide(self, view: WorldView, orders: OperationsOrders, squads: List[SquadRecord],
               reports: List[MissionReport], game_time_ms: int) -> Tuple[List[TaskContract], List[Shortfall]]:
        by_squad: Dict[int, MissionReport] = {report.squad: report for report in reports}
        shortfalls: List[Shortfall] = []
        plans: List[_Plan] = []
        held_now: set = set()

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
            plan = self._plan(view, orders, squad, by_squad.get(squad.id))
            if plan is not None:
                plans.append(plan)

        self.human_held = held_now
        return self._price(plans, orders, game_time_ms), shortfalls

    # ---- what a squad is sent to do --------------------------------------------------

    def _plan(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
              report: Optional[MissionReport]) -> Optional[_Plan]:
        allowed = DOCTRINES[squad.doctrine].tasks
        status = report.status if report is not None else Status.ACTIVE
        held = squad.contract

        # A mission that has stopped moving, or that is being lost, is a reason to look somewhere else; one that is finished frees the squad for the next target. In all three the region it is at now is taken off the table, which is what makes the search find a different answer rather than the same one.
        avoid = held.target_region if held is not None and status in (Status.STALLED, Status.LOSING, Status.COMPLETE) else None

        if status == Status.LOSING and Task.WITHDRAW in allowed:
            home = view.home
            if home is not None:
                return _Plan(squad=squad, task=Task.WITHDRAW, region=home, forced=True)

        chosen = self._target(view, orders, squad, allowed, avoid)
        if chosen is None:
            return None
        task, region = chosen
        if task not in allowed:
            return None
        # An expired contract goes out again even when it reads identically: what it needs is a fresh deadline and a fresh baseline for the losses it is measured against.
        return _Plan(squad=squad, task=task, region=region, forced=status == Status.EXPIRED)

    def _target(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
                allowed: tuple, avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        chosen = self._pick(view, orders, squad, avoid)
        return self._settled(view, orders, squad, chosen, avoid)

    def _settled(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
                 chosen: Optional[Tuple[Task, RegionState]], avoid: Optional[int]):
        """Keeps a squad on the mission it is already running unless something has said to stop.

        Two regions of nearly equal worth trade places whenever a shot lands in either of them, and a squad re-tasked on that difference spends the match walking between them and arriving at neither. So the score is not what re-tasks a squad; a reason is. The reasons are the ones the mission reports carry — a stall, a mission going badly, one finished, one out of time — and while a mission is simply running it is left to run. That is the same rule that governs re-issuing a contract at all, applied one level up: a decision restated is not a decision.
        """
        held = squad.contract
        if chosen is None or held is None or squad.status is not Status.ACTIVE:
            return chosen
        if held.target_region == avoid:
            return chosen
        current = view.region(held.target_region)
        if current is None:
            return chosen
        return held.task, current

    def _pick(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
              avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        if squad.doctrine == Doctrine.VANGUARD:
            return self._vanguard(view, orders, squad, avoid)
        if squad.doctrine == Doctrine.GARRISON:
            return self._garrison(view, orders, squad, avoid)
        if squad.doctrine == Doctrine.RAID:
            return self._raid(view, orders, squad, avoid)
        # Engineers escort by staying where the building is being done, which is home until something says otherwise. Sending them out is what loses them.
        return (Task.ESCORT, view.home) if view.home is not None else None

    def _vanguard(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
                  avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        """The highest priority contested region, less what it costs to march there and less what we already have standing in it."""
        contested = [r for r in view.regions if r.id != avoid and (r.enemy_value > 0 or r.held_by_enemy)]
        # On the defensive the vanguard fights only over ground we hold, which is what the strategic layer means by not pressing.
        if not orders.offensive:
            contested = [r for r in contested if r.held_by_us]
        if not contested:
            # Nothing in contact: press towards whatever the strategic layer wants that we are not already on.
            contested = [r for r in view.regions if r.id != avoid and not r.held_by_us]
        if not contested:
            return None

        def score(region: RegionState) -> float:
            return (self._priority(orders, region)
                    - self._reach(region)
                    - self.crowding * _share(region.our_value, squad.value))

        region = max(contested, key=lambda r: (score(r), -r.distance_from_home))
        # Surrounding is what you do when you can afford to spend the time; against odds that are merely even it splits a squad that needs to arrive as one.
        strength = squad.value + region.our_value
        encircle = region.enemy_value > 0 and strength >= ENCIRCLE_ODDS * region.enemy_value
        task = Task.ENCIRCLE if encircle and Task.ENCIRCLE in DOCTRINES[Doctrine.VANGUARD].tasks else Task.ATTACK
        return task, region

    def _garrison(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
                  avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        """The most valuable ground we already draw an income from. A garrison that is not standing on something that pays is a garrison spent on nothing."""
        ours = [r for r in view.regions if r.id != avoid and r.held_by_us and r.resources > 0]
        if not ours:
            ours = [r for r in view.regions if r.held_by_us] or ([view.home] if view.home is not None else [])
        if not ours:
            return None

        def score(region: RegionState) -> float:
            return (self._priority(orders, region)
                    + RESOURCE_WORTH * region.resources
                    + THREAT_WEIGHT * _share(region.enemy_value, squad.value)
                    - self.crowding * _share(region.our_value, squad.value)
                    - self._reach(region))

        return Task.DEFEND, max(ours, key=lambda r: (score(r), -r.distance_from_home))

    def _raid(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
              avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        """Something that pays and that nothing is guarding. A raid that has to fight for its ground is a vanguard action being done by the wrong squad."""
        candidates = [r for r in view.regions if r.id != avoid and r.resources > 0 and not r.held_by_us]
        if not candidates:
            candidates = [r for r in view.regions if r.id != avoid and not r.held_by_us]
        if not candidates:
            return None

        def score(region: RegionState) -> float:
            return (self._priority(orders, region)
                    + RESOURCE_WORTH * region.resources
                    - RAID_RISK * _share(region.enemy_value, squad.value)
                    - self._reach(region))

        return Task.RAID, max(candidates, key=lambda r: (score(r), -r.distance_from_home))

    @staticmethod
    def _priority(orders: OperationsOrders, region: RegionState) -> float:
        return orders.priorities.get(region.id, 0.0)

    @staticmethod
    def _reach(region: RegionState) -> float:
        return REACH_COST * region.distance_from_home / REACH_SCALE

    # ---- what a mission may cost and how long it has ---------------------------------

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
            )
            if not plan.forced and not self._changed(plan.squad.contract, contract):
                continue
            # The record carries what the squad is under, so that the next period compares against what was actually sent rather than against what was last thought.
            plan.squad.contract = contract
            contracts.append(contract)
        return contracts

    @staticmethod
    def _march_ms(plan: _Plan) -> int:
        distance = math.hypot(plan.region.x - plan.squad.x, plan.region.y - plan.squad.y)
        return int(distance * MARCH_SLACK / MARCH_SPEED * 1000) + ENGAGEMENT_MS

    @staticmethod
    def _changed(held: Optional[TaskContract], fresh: TaskContract) -> bool:
        if held is None:
            return True
        if (held.task, held.target_region, held.stance) != (fresh.task, fresh.target_region, fresh.stance):
            return True
        # The deadline is deliberately not compared: it moves every period by construction, and re-issuing on that alone would reset the losses on every mission for ever.
        reference = max(held.cost_budget, MINIMUM_BUDGET)
        return abs(fresh.cost_budget - held.cost_budget) / reference > BUDGET_TOLERANCE

    # ---- what the organisation layer is told -----------------------------------------

    def _missing(self, squad: SquadRecord) -> float:
        """Credits of strength between what the squad is worth and what a full one of its doctrine would cost, priced at the cheapest type filling each role because that is what the economy would actually build."""
        establishment = 0.0
        for role, count in DOCTRINES[squad.doctrine].establishment.items():
            kind = self.catalogue.cheapest(role)
            if kind is not None:
                establishment += count * float(kind.price)
        return max(0.0, establishment - squad.value)
