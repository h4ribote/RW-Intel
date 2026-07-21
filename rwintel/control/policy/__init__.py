"""The script command chain: five layers, each deciding one thing, on its own period.

Writing the whole chain as a script comes first, and it does four jobs at once. It is the skeleton, so that something always plays. It is the teacher, since what it emits is a state and an action in the same form a learnt layer would emit. It is the measuring stick a learnt layer has to beat over many episodes. And it is the proof that the contracts between the layers carry enough: a layer that cannot be written from what it is handed is a contract that is too thin, and that shows up here rather than in an argument.

The layers are wired to periods, not to each other. Each runs when its period comes round, reads the contract its superior left, and leaves a contract for its subordinate; nothing calls down the chain. That is what makes it possible to freeze four layers and replace the fifth, which is the whole plan for learning, and it is the same seam a human takes over through.

Periods are counted in game time, never in steps or in wall clock. A step carries a different amount of game time at a different speed multiplier, and wall clock would make the policy depend on how busy the machine is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...wire import Action, BLOCK_REGIONS, Contract, Observation, encode_action
from ...wire.action import Status
from .catalogue import Catalogue
from .contracts import (
    EconomyOrders,
    FrontReport,
    OperationsOrders,
    Replacement,
    Shortfall,
    SquadRecord,
)
from .economy import Economy
from .operations import Operations
from .organisation import Organisation
from .strategy import Strategy
from .tactics import Tactics
from .view import build as build_view, home_region_id

log = logging.getLogger(__name__)

#: Game time between strategic decisions. The tactical and operational periods are the agent's to set, and are visible in the frames themselves; this one has no block of its own, so it is counted here.
STRATEGIC_MS = 10000

#: How long a region slipping out of our hands goes on counting as being pushed back. A running total that never decayed would pin the posture to defending for the rest of the match on the strength of three losses in the opening.
LOST_REGION_MEMORY_MS = 60000


@dataclass
class Statistics:
    """What the chain did over an episode, which is what says which layer a result came from.

    A score on its own tells you that an episode went badly and nothing about where. These counts separate the two questions the design keeps apart: how often each layer got to decide, which is a property of the periods, and how its decisions turned out, which is a property of the layer. A run where the operational layer issued forty contracts and completed two is a different failure from one where it issued two.
    """

    strategic: int = 0
    operational: int = 0
    tactical: int = 0
    contracts: int = 0
    completed: int = 0
    stalled: int = 0
    losing: int = 0
    expired: int = 0
    production: int = 0
    squads_formed: int = 0

    @property
    def fulfilment(self) -> float:
        """The share of contracts that ended in the target being taken and held, which is the operational layer's own measure of itself."""
        return self.completed / self.contracts if self.contracts else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {**vars(self), "fulfilment": round(self.fulfilment, 4)}


class ScriptPolicy:
    def __init__(self, session) -> None:
        self.session = session
        self.catalogue = Catalogue(session.types, session.assets)

        self.strategy = Strategy(session, self.catalogue)
        self.economy = Economy(session, self.catalogue)
        self.organisation = Organisation(session, self.catalogue)
        self.operations = Operations(session, self.catalogue)
        self.tactics = Tactics(session, self.catalogue)

        self.home_id: Optional[int] = None
        self.last_strategic_ms: Optional[int] = None

        self.economy_orders: Optional[EconomyOrders] = None
        self.operations_orders: Optional[OperationsOrders] = None
        self.squads: List[SquadRecord] = []
        self.shortfalls: List[Shortfall] = []
        self.replacements: List[Replacement] = []
        self.reports: List = []

        self.statistics = Statistics()
        #: The status each squad's mission was last seen in, so that reaching a new one is counted once rather than every period.
        self._status: Dict[int, Status] = {}
        #: How many resource points each region was last seen held with, so that losing one can be noticed.
        self._held: Dict[int, int] = {}
        #: When each region was last taken from us, so that being pushed back is something that stops being true.
        self._lost_at: List[int] = []

    def decide(self, observation: Observation) -> Optional[bytes]:
        if self.home_id is None:
            self.home_id = home_region_id(observation)
        view = build_view(observation, self.catalogue, self.home_id)
        now = observation.game_time_ms
        action = Action()

        # The organisation layer is event driven, and the design puts its events on the operational frame so that they are consumed before the operational decision is taken.
        operational = bool(observation.blocks & BLOCK_REGIONS)
        if operational or observation.events:
            assignments, self.squads, self.replacements = self.organisation.update(view, self.shortfalls)
            action.squads.extend(assignments)
            self.statistics.squads_formed += sum(1 for a in assignments if a.units)

        if self.last_strategic_ms is None or now - self.last_strategic_ms >= STRATEGIC_MS:
            self.last_strategic_ms = now
            self.statistics.strategic += 1
            self._note_lost_regions(view, now)
            self.economy_orders, self.operations_orders = self.strategy.decide(
                self._front_report(view), view.regions, now)

        if operational and self.operations_orders is not None:
            contracts, self.shortfalls = self.operations.decide(
                view, self.operations_orders, self.squads, self.reports, now)
            # The layers speak in the design's own contract; the wire carries the same fields in the order both halves agreed on. Keeping the two apart is what lets the layer be replaced without touching the protocol.
            action.contracts.extend(
                Contract(squad=c.squad, task=c.task, stance=c.stance, target_region=c.target_region,
                         cost_budget=c.cost_budget, deadline_ms=c.deadline_ms, issued_at_ms=c.issued_at_ms)
                for c in contracts)
            self.statistics.operational += 1
            self.statistics.contracts += len(contracts)

        if operational and self.economy_orders is not None:
            orders = self.economy.decide(view, self.economy_orders, self.replacements)
            action.production.extend(orders)
            self.statistics.production += len(orders)

        deviations, self.reports = self.tactics.decide(view, self.squads, now)
        action.deviations.extend(deviations)
        self.statistics.tactical += 1
        self._count_outcomes()

        if operational:
            log.debug("t=%5ds %s squads=%d units=%d(%d enemy) credits=%.0f contracts=%d production=%s",
                      now // 1000,
                      self.operations_orders.posture.name if self.operations_orders else "-",
                      len(self.squads), len(view.ours), len(view.enemies), observation.credits,
                      len(action.contracts),
                      [self.catalogue.kind(p.type_index).lookup for p in action.production])

        if not (action.squads or action.contracts or action.deviations or action.production):
            return None
        return encode_action(action)

    def _count_outcomes(self) -> None:
        """Counts a mission's outcome once, when it first reaches it. A status is a state and not an event, so counting it every period would report the length of a stall rather than the number of them."""
        live = set()
        for squad in self.squads:
            live.add(squad.id)
            if self._status.get(squad.id) is squad.status:
                continue
            self._status[squad.id] = squad.status
            if squad.status is Status.COMPLETE:
                self.statistics.completed += 1
            elif squad.status is Status.STALLED:
                self.statistics.stalled += 1
            elif squad.status is Status.LOSING:
                self.statistics.losing += 1
            elif squad.status is Status.EXPIRED:
                self.statistics.expired += 1
        for squad_id in [k for k in self._status if k not in live]:
            del self._status[squad_id]

    def _note_lost_regions(self, view, now: int) -> None:
        """Counts regions that used to have one of our extractors on them and no longer do. A posture change to defending is the answer to being pushed off ground, and there is nothing else in the observation that says it is happening."""
        for region in view.regions:
            was = self._held.get(region.id, 0)
            if region.held_by_us < was:
                self._lost_at.append(now)
            self._held[region.id] = region.held_by_us
        self._lost_at = [when for when in self._lost_at if now - when <= LOST_REGION_MEMORY_MS]

    def _front_report(self, view) -> FrontReport:
        observation = view.observation
        # A base is a foothold the enemy draws from, not a starting position on the map. Counting starting positions would make "down to one base" true from the opening whistle of every match between two, which is the opposite of what it is meant to detect; counting footholds makes it rise as they expand and fall as they are pushed off, which is the thing worth deciding on.
        spawns = {region.id for region in self.session.regions if region.spawn}
        enemy_bases = sum(1 for r in view.regions
                          if r.held_by_enemy > 0
                          or (r.id in spawns and r.id != self.home_id and r.enemy_value > 0))
        return FrontReport(
            income=observation.income,
            credits=observation.credits,
            military_value=sum(s.value for s in view.fighters),
            enemy_value=sum(s.value for s in view.enemies),
            held=sum(r.held_by_us for r in view.regions),
            enemy_held=sum(r.held_by_enemy for r in view.regions),
            lost_regions=len(self._lost_at),
            enemy_bases=enemy_bases,
        )


def script_policy(session) -> ScriptPolicy:
    return ScriptPolicy(session)
