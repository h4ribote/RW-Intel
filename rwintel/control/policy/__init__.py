"""The script command chain: five layers, each deciding one thing, on its own period.

Writing the whole chain as a script comes first, and it does four jobs at once. It is the skeleton, so that something always plays. It is the teacher, since what it emits is a state and an action in the same form a learnt layer would emit. It is the measuring stick a learnt layer has to beat over many episodes. And it is the proof that the contracts between the layers carry enough: a layer that cannot be written from what it is handed is a contract that is too thin, and that shows up here rather than in an argument.

The layers are wired to periods, not to each other. Each runs when its period comes round, reads the contract its superior left, and leaves a contract for its subordinate; nothing calls down the chain. That is what makes it possible to freeze four layers and replace the fifth, which is the whole plan for learning, and it is the same seam a human takes over through.

Periods are counted in game time, never in steps or in wall clock. A step carries a different amount of game time at a different speed multiplier, and wall clock would make the policy depend on how busy the machine is.

Anything that is not one of the five layers and still wants to command reaches the game through `outside`. A human at the intervention interface is one such thing and the script intruder that trains against interruption is another, and neither is given a private channel: both amend the action the chain has just built, in the chain's own contract form, after every layer has had its say. That ordering is the ownership rule made mechanical -the last word about a squad belongs to whoever holds it -and it is why a squad taken over stops being written about here without any layer having to know that anyone was taken over from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...wire import Action, BLOCK_REGIONS, Contract, NO_LIFT, Observation, encode_action
from ...wire.action import Status, Task
from ...wire.observation import LiftFailure, LiftPhase
from .audit import LossAudit
from .catalogue import Catalogue
from .contracts import (
    DOMAIN_OF_MOVEMENT,
    Domain,
    EconomyOrders,
    FrontReport,
    OperationsOrders,
    Replacement,
    Shortfall,
    SquadRecord,
)
from .economy import Economy
from .encoding import means_of, task_of
from .logistics import Logistics
from .operations import Operations
from .reach import Reach, members_reach
from .options import Options
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
    #: Contracts the game side found no member could reach.
    unreachable: int = 0
    #: Lifts sent, for squads and for builders.
    lifts: int = 0
    #: The most extractors, and the most worth of fighting units, of ours ever standing at once on land that cannot be walked to from home: how far the chain has spread across water.
    overseas_extractors: int = 0
    overseas_value: float = 0.0
    production: int = 0
    squads_formed: int = 0
    #: Decisions taken by someone other than the chain -a human, or the script intruder. Recorded because the design says the results of squads that were interfered with are to be kept out of the learning signal, and a count is the first thing that says whether there were any.
    interventions: int = 0
    #: Game seconds spent in each posture, by name, and how often the posture changed. Says whether the chain spent the match arming or kept flipping between postures.
    postures: Dict[str, float] = field(default_factory=dict)
    posture_changes: int = 0
    #: The economy's ledger: credits left in the treasury, factories left without an order and builders without a job, and when the opening's milestones stood.
    ledger: Dict[str, float] = field(default_factory=dict)
    #: Credits of our own lost, by role, by whether in a squad or loose, and under enemy defences (`audit.LossAudit`).
    losses: Dict[str, float] = field(default_factory=dict)

    @property
    def fulfilment(self) -> float:
        """The share of contracts that ended in the target being taken and held, which is the operational layer's own measure of itself."""
        return self.completed / self.contracts if self.contracts else 0.0

    def as_dict(self) -> Dict[str, float]:
        fields = {key: value for key, value in vars(self).items() if key != "ledger"}
        return {**fields, **self.ledger, "fulfilment": round(self.fulfilment, 4)}


class ScriptPolicy:
    def __init__(self, session, options: Options = Options()) -> None:
        self.session = session
        self.options = options
        self.catalogue = Catalogue(session.types, session.assets)

        self.strategy = Strategy(session, self.catalogue, options)
        self.economy = Economy(session, self.catalogue, options)
        self.organisation = Organisation(session, self.catalogue, options)
        self.operations = Operations(session, self.catalogue, options)
        self.tactics = Tactics(session, self.catalogue, options)
        self.audit = LossAudit()
        #: The transports and the lifts they are lent to, shared by the operational layer, which sends squads across, and the economy, which sends builders.
        self.logistics = Logistics(self.catalogue)

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
        #: With debug logging on, the last lift the game reported carrying each squad, and the last phase and failure reported for each lift, which say what a contract coming back unreachable was preceded by.
        self._lift_of: Dict[int, int] = {}
        self._lift_ends: Dict[int, tuple] = {}

    def decide(self, observation: Observation) -> Optional[bytes]:
        action, view = self.plan(observation)
        for commander in self.outside:
            self.statistics.interventions += len(commander.intervene(action, view, self.squads, observation) or ())
        if action.empty():
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
        self.audit.update(view)
        # The lifts are read every period, since a lift's end is reported in the one frame after it ends.
        self._attach_logistics()
        self.logistics.update(view)
        self._note_reachable(view)

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
                self._front_report(view), view.regions, now,
                contact=self._contact() if self.options.contact else None, home=view.home,
                airborne=sum(report.airborne for report in self.reports))
            self.statistics.postures = {posture.name.lower(): ms / 1000 for posture, ms in self.strategy.time_in.items()}
            self.statistics.posture_changes = self.strategy.changes

        if operational and self.operations_orders is not None:
            contracts, self.shortfalls = self.operations.decide(
                view, self.operations_orders, self.squads, self.reports, now)
            # The layers speak in the design's own contract; the wire carries the same fields in the order both halves agreed on. Keeping the two apart is what lets the layer be replaced without touching the protocol.
            action.contracts.extend(
                Contract(squad=c.squad, task=c.task, stance=c.stance, target_kind=c.target_kind, target=c.wire_target,
                         cost_budget=c.cost_budget, deadline_ms=c.deadline_ms, issued_at_ms=c.issued_at_ms)
                for c in contracts)
            self.statistics.operational += 1
            self.statistics.contracts += len(contracts)

        if operational and self.economy_orders is not None:
            orders = self.economy.decide(view, self.economy_orders, self.replacements)
            action.production.extend(orders)
            self.statistics.production += len(orders)
            self.statistics.ledger = self.economy.ledger.summary()
            self.statistics.losses = self.audit.summary()

        # Lifts go out after both layers that ask for them, and after the contracts: a squad's new contract is taken on first, and the lift that carries it then holds it until it is set down.
        lifts = self.logistics.rows()
        action.lifts.extend(lifts)
        self.statistics.lifts += sum(1 for lift in lifts if not lift.cancel)

        deviations, self.reports = self.tactics.decide(view, self.squads, now)
        action.deviations.extend(deviations)
        self.statistics.tactical += 1
        self._count_outcomes(view, now)

        if operational:
            self._count_overseas(view)
            log.debug("t=%5ds %s squads=%d units=%d(%d enemy) credits=%.0f contracts=%d production=%s",
                      now // 1000,
                      self.operations_orders.posture.name if self.operations_orders else "-",
                      len(self.squads), len(view.ours), len(view.enemies), observation.credits,
                      len(action.contracts),
                      [self.catalogue.kind(p.type_index).lookup for p in action.production])
            log.debug("transports %s shortfall=%d %s lifts=%s statuses=%s",
                      [(s.slot, s.unit, s.squad, s.units, s.region, s.lift, s.phase) for s in self.logistics.slots if s.unit],
                      self.logistics.shortfall.count, sorted(self.logistics.shortfall.passengers),
                      [(l.lift, l.phase, l.reason, l.loaded, l.expected) for l in observation.lifts],
                      [(s.id, s.doctrine.name, s.status.name, s.contract.target_region if s.contract else None,
                        s.contract.means if s.contract else None) for s in self.squads])

        return action, view

    def _attach_logistics(self) -> None:
        """Hands the lift layer to the layers that ask it for lifts, whichever of them a learnt one has replaced, and builds its map of the episode once the game has sent the terrain."""
        self.operations.logistics = self.logistics
        self.economy.logistics = self.logistics
        self.organisation.logistics = self.logistics
        passage = getattr(self.session, "passage", None)
        regions = getattr(self.session, "regions", None)
        if self.logistics.reach is None and passage is not None and regions:
            self.logistics.reach = Reach(passage, regions)

    def _note_reachable(self, view) -> None:
        """Tells the strategic layer, once, which regions an army on the ground could get to from home: under its own power over land or land and water, or set down by a kind of transport the catalogue has."""
        reach = self.logistics.reach
        if reach is None or view.home is None or self.strategy.reachable is not None or view.home.id not in reach.regions:
            return
        home = reach.regions[view.home.id]
        movements = [m for m, domain in DOMAIN_OF_MOVEMENT.items() if domain in (Domain.GROUND, Domain.AMPHIBIOUS)]
        found = set()
        for region in reach.regions.values():
            if any(home.components.get(m, -1) >= 0 and region.components.get(m, -2) == home.components.get(m, -1)
                   for m in movements):
                found.add(region.region)
        for kind in self.catalogue.types:
            if not kind.transport or not kind.carries:
                continue
            for region in reach.regions.values():
                landing = reach.landing(region.region, kind.movement)
                if landing is not None and reach.component(kind.movement, *landing) == home.components.get(kind.movement, -1):
                    found.add(region.region)
        self.strategy.reachable = found

    def _count_overseas(self, view) -> None:
        """Notes how much of ours stands on land that cannot be walked to from home."""
        reach = self.logistics.reach
        if reach is None or view.home is None or view.home.id not in reach.regions:
            return
        home = reach.regions[view.home.id].components.get("LAND", -1)
        if home < 0:
            return

        def overseas(sighting) -> bool:
            if sighting.unit.carrier:
                return False
            component = reach.component("LAND", sighting.unit.x, sighting.unit.y)
            return component >= 0 and component != home

        extractors = sum(1 for s in view.buildings if s.kind is not None and s.kind.extractor and overseas(s))
        value = sum(s.value for s in view.fighters if overseas(s))
        self.statistics.overseas_extractors = max(self.statistics.overseas_extractors, extractors)
        self.statistics.overseas_value = max(self.statistics.overseas_value, value)

    def _count_outcomes(self, view, now: int) -> None:
        """Counts a mission's outcome once, when it first reaches it. A status is a state and not an event, so counting it every period would report the length of a stall rather than the number of them."""
        explain = log.isEnabledFor(logging.DEBUG)
        if explain:
            self._note_lifts(view)
        live = set()
        for squad in self.squads:
            live.add(squad.id)
            before = self._status.get(squad.id)
            if before is squad.status:
                continue
            self._status[squad.id] = squad.status
            if explain and squad.status is Status.UNREACHABLE:
                self._explain_unreachable(squad, before, view, now)
            if squad.status is Status.COMPLETE:
                self.statistics.completed += 1
            elif squad.status is Status.STALLED:
                self.statistics.stalled += 1
            elif squad.status is Status.LOSING:
                self.statistics.losing += 1
            elif squad.status is Status.EXPIRED:
                self.statistics.expired += 1
            elif squad.status is Status.UNREACHABLE:
                self.statistics.unreachable += 1
        for squad_id in [k for k in self._status if k not in live]:
            del self._status[squad_id]

    def _note_lifts(self, view) -> None:
        for lift in view.observation.lifts or ():
            self._lift_ends[lift.lift] = (lift.phase, lift.reason)
        for squad in self.squads:
            if squad.lift != NO_LIFT:
                self._lift_of[squad.id] = squad.lift

    def _explain_unreachable(self, squad, before: Optional[Status], view, now: int) -> None:
        """Logs at debug level what a squad whose contract has just come back unreachable was doing: what it was sent to do and by what means, how the operational layer chose it, how long ago it was issued, whether its members can walk there now, and what its last lift came to."""
        def name(kind, value):
            return kind(value).name if value in kind._value2member_map_ else value

        contract = squad.contract
        terrain = self.logistics.reach
        by_id = {s.unit.id: s for s in view.ours}
        sightings = [by_id.get(m) for m in squad.members]
        standing = [s for s in sightings if s is not None and s.kind is not None and not s.unit.carrier]
        carried = sum(1 for s in sightings if s is not None and s.unit.carrier)
        region = contract.target_region if contract is not None else -1
        walk_all = walk_some = None
        if terrain is not None and standing and region >= 0:
            walking = [(s.kind.movement, s.unit.x, s.unit.y) for s in standing]
            walk_all = members_reach(terrain, walking, region)
            walk_some = sum(1 for m in walking if terrain.walkable(m[0], m[1], m[2], region))
        pick = self.operations.picked.get(squad.id)
        picked = None
        if pick is not None:
            at, slot, plan, opened = pick
            picked = dict(ago_ms=now - at, region=slot, task=name(Task, task_of(plan)), means=means_of(plan), open=opened)
        lift = self._lift_of.get(squad.id)
        ended = self._lift_ends.get(lift) if lift is not None else None
        carrying = self.logistics.lifting(squad.id)
        log.debug("unreachable layer=%s squad=%d doctrine=%s domain=%s passage=%s task=%s region=%d kind=%s means=%s "
                  "age_ms=%s before=%s aboard=%d carried=%d standing=%d walk_all=%s walk_some=%s slot_lifting=%s "
                  "last_lift=%s lift_end=%s picked=%s",
                  type(self.operations).__name__, squad.id, squad.doctrine.name, squad.domain.name, squad.passage,
                  contract.task.name if contract is not None else None, region,
                  contract.target_kind.name if contract is not None else None,
                  contract.means if contract is not None else None,
                  now - contract.issued_at_ms if contract is not None else None,
                  before.name if before is not None else None, squad.aboard, carried, len(standing), walk_all,
                  walk_some, carrying.slot if carrying is not None else None, lift,
                  (name(LiftPhase, ended[0]), name(LiftFailure, ended[1])) if ended is not None else None, picked)

    def _contact(self) -> Dict:
        """What every squad has run into, by role and by worth, from the mission reports: the only account of the enemy's army that reaches the strategic layer once the fog is on. Squads standing near one another report the same enemy twice, which leaves the shares by role the strategic layer reads unchanged."""
        found: Dict = {}
        for report in self.reports:
            for role, worth in report.contact.items():
                found[role] = found.get(role, 0.0) + worth
        return found

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
            enemy_military=sum(s.value for s in view.enemy_fighters),
            held=sum(r.held_by_us for r in view.regions),
            enemy_held=sum(r.held_by_enemy for r in view.regions),
            lost_regions=len(self._lost_at),
            enemy_bases=enemy_bases,
        )


def script_policy(session, options: Options = Options()) -> ScriptPolicy:
    return ScriptPolicy(session, options)
