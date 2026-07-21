"""The script command chain: five layers, each deciding one thing, on its own period.

Writing the whole chain as a script comes first, and it does four jobs at once. It is the skeleton, so that something always plays. It is the teacher, since what it emits is a state and an action in the same form a learnt layer would emit. It is the measuring stick a learnt layer has to beat over many episodes. And it is the proof that the contracts between the layers carry enough: a layer that cannot be written from what it is handed is a contract that is too thin, and that shows up here rather than in an argument.

The layers are wired to periods, not to each other. Each runs when its period comes round, reads the contract its superior left, and leaves a contract for its subordinate; nothing calls down the chain. That is what makes it possible to freeze four layers and replace the fifth, which is the whole plan for learning, and it is the same seam a human takes over through.

Periods are counted in game time, never in steps or in wall clock. A step carries a different amount of game time at a different speed multiplier, and wall clock would make the policy depend on how busy the machine is.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ...wire import Action, BLOCK_REGIONS, Contract, Observation, encode_action
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

        if self.last_strategic_ms is None or now - self.last_strategic_ms >= STRATEGIC_MS:
            self.last_strategic_ms = now
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

        if operational and self.economy_orders is not None:
            action.production.extend(self.economy.decide(view, self.economy_orders, self.replacements))

        deviations, self.reports = self.tactics.decide(view, self.squads, now)
        action.deviations.extend(deviations)

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
