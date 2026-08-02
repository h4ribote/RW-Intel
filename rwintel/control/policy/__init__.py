"""The script command chain: five layers, each deciding one thing, on its own period.

Writing the whole chain as a script comes first, and it does four jobs at once. It is the skeleton, so that something always plays. It is the teacher, since what it emits is a state and an action in the same form a learnt layer would emit. It is the measuring stick a learnt layer has to beat over many episodes. And it is the proof that the contracts between the layers carry enough: a layer that cannot be written from what it is handed is a contract that is too thin, and that shows up here rather than in an argument.

The layers are wired to periods, not to each other. Each runs when its period comes round, reads the contract its superior left, and leaves a contract for its subordinate; nothing calls down the chain. That is what makes it possible to freeze four layers and replace the fifth, which is the whole plan for learning, and it is the same seam a human takes over through.

Periods are counted in game time, never in steps or in wall clock. A step carries a different amount of game time at a different speed multiplier, and wall clock would make the policy depend on how busy the machine is.

Anything that is not one of the five layers and still wants to command reaches the game through `outside`. A human at the intervention interface is one such thing and the script intruder that trains against interruption is another, and neither is given a private channel: both amend the action the chain has just built, in the chain's own contract form, after every layer has had its say. That ordering is the ownership rule made mechanical — the last word about a squad belongs to whoever holds it — and it is why a squad taken over stops being written about here without any layer having to know that anyone was taken over from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...wire import Action, BLOCK_REGIONS, BUILT, Contract, Observation, encode_action
from ...wire.action import Status
from .catalogue import Catalogue
from .contracts import (
    EconomyOrders,
    FrontReport,
    OperationsOrders,
    Replacement,
    Role,
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
    #: Decisions taken by someone other than the chain — a human, or the script intruder. Recorded because the design says the results of squads that were interfered with are to be kept out of the learning signal, and a count is the first thing that says whether there were any.
    interventions: int = 0

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
        #: The region table as it last arrived, carried between the operational frames that bring it so that the layers running in between are not handed a board with no places on it.
        self.last_regions: List = []

        self.economy_orders: Optional[EconomyOrders] = None
        self.operations_orders: Optional[OperationsOrders] = None
        self.squads: List[SquadRecord] = []
        self.shortfalls: List[Shortfall] = []
        self.replacements: List[Replacement] = []
        self.reports: List = []
        #: Commanders outside the chain, consulted in order once the chain has decided. A human's interface and the script intruder are both of these; nothing here knows which.
        self.outside: List = []

        self.statistics = Statistics()
        #: The status each squad's mission was last seen in, so that reaching a new one is counted once rather than every period.
        self._status: Dict[int, Status] = {}
        #: How many resource points each region was last seen held with, so that losing one can be noticed.
        self._held: Dict[int, int] = {}
        #: When each region was last taken from us, so that being pushed back is something that stops being true.
        self._lost_at: List[int] = []

    def decide(self, observation: Observation) -> Optional[bytes]:
        action, view = self.plan(observation)
        for commander in self.outside:
            self.statistics.interventions += len(commander.intervene(action, view, self.squads, observation) or ())
        if not (action.squads or action.contracts or action.deviations or action.production):
            return None
        return encode_action(action)

    def plan(self, observation: Observation):
        """What the chain alone decides, before anyone outside it has amended anything. Separate from `decide` so that a layer can be swapped for a learnt one, or the action inspected, without the amendment step having to be repeated in each caller."""
        if self.home_id is None:
            self.home_id = home_region_id(observation)
        view = build_view(observation, self.catalogue, self.home_id, self.last_regions)
        self.last_regions = view.regions
        now = observation.game_time_ms
        action = Action()

        # The organisation layer is event driven, and the design puts its events on the operational frame so that they are consumed before the operational decision is taken.
        operational = bool(observation.blocks & BLOCK_REGIONS)
        if operational or observation.events:
            assignments, self.squads, self.replacements = self.organisation.update(view, self.shortfalls)
            action.squads.extend(assignments)
            self.statistics.squads_formed += sum(1 for a in assignments if a.units)
            # A squad destroyed is retired in the same period its last unit died, so the layer that fought it is never handed a board with an empty squad on it and cannot see the ending for itself. Handed down here, because this is the one place that holds both layers. A layer with nothing to do with it — the handwritten one — does not define this and nothing is called.
            spent = getattr(self.organisation, "wiped", None)
            if spent:
                destroyed = getattr(self.tactics, "wiped", None)
                if destroyed is not None:
                    destroyed(sorted(spent))

        if self.last_strategic_ms is None or now - self.last_strategic_ms >= STRATEGIC_MS:
            self.last_strategic_ms = now
            self.statistics.strategic += 1
            self._note_lost_regions(view, now)
            # What the enemy is fielding, over the whole board. Without it the strategic layer's seven contact features are nought in every frame of every match — a quarter of its cut, structurally dead — and the rule's own answer to an air-heavy enemy can never fire, since it is written as a condition on exactly this reading.
            self.economy_orders, self.operations_orders = self.strategy.decide(
                self._front_report(view), view.regions, now, view.contacted())

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

        return action, view

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
            # Two pairs, each comparing like with like: the mobile armed force on both sides, and everything standing on both sides. The enemy's roles come off the same catalogue ours do, so what is left out of their fighting strength is what is left out of ours.
            #
            # The total pair counts FINISHED units only, which is what the match's own scoring counts: the game side sums a unit's price only once it is built. Counting a half-raised factory at its full price here would make the strategic layer's potential — which is the running form of that very score — read a board the terminal will not agree with, and the disagreement would be largest exactly where the layer is deciding whether to build one.
            our_value=sum(s.value for s in view.ours if s.unit.built >= BUILT),
            enemy_value=sum(s.value for s in view.enemies if s.unit.built >= BUILT),
            enemy_military_value=sum(s.value for s in view.enemies
                                     if s.role not in (Role.STRUCTURE, Role.BUILDER)),
            held=sum(r.held_by_us for r in view.regions),
            enemy_held=sum(r.held_by_enemy for r in view.regions),
            lost_regions=len(self._lost_at),
            enemy_bases=enemy_bases,
            units=observation.units,
            unit_cap=observation.unit_cap,
            under_construction=observation.under_construction,
        )


def script_policy(session) -> ScriptPolicy:
    return ScriptPolicy(session)
