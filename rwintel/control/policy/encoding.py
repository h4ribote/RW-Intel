"""Turning a period's board into numbers, and saying what a layer is allowed to answer.

Each layer gets its own cut, at its own abstraction, because that is how the design divides them: the tactical layer sees around one squad, the operational layer sees the regions and the squads and nothing below them, and the economy sees its own treasury and the investments open to it. Two rules govern everything here. Every feature is a ratio or a length divided by a stated scale, so that nothing depends on how rich the match has become or how large the map is -a policy trained on one map has to be readable on another, and a raw credit total or a raw world coordinate would make that false. And every block is a fixed width with a validity flag, never a packed list, so that a slot means the same thing from one decision to the next; a packed list renumbers everything the moment a squad dies.

The action spaces are exactly the ones the script layers already emit, which is what makes a learnt layer a replacement rather than a parallel system: the departures for the tactical layer, one region and one plan -a task and a means, walking or one of the lift layer's transports- for the operational layer, and the next investment for the economy. The tactical departures were five and are now seven, the two added ones being a withdrawal that commits the whole way out and a concentration that goes onto the longest-ranged enemy rather than the weakest -kinds of move the script already made, with a parameter a rule used to fix handed to the layer. Neither space is a free choice -a squad may only be sent where a region exists and it can get to, only given a task its doctrine allows, and only carried by a transport that can take it there -so both come with a mask, and the mask is computed here from the same tables the script reads rather than being learnt as a soft preference. The tactical action count follows the departure enum, so widening the enum widens the head and the mask with it.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ...wire import REGION_SLOTS, SQUAD_SLOTS, Deviation, RegionState, Stance, Status, Task
from .contracts import DOCTRINES, Doctrine, Domain, Posture, Role, SquadRecord
from .logistics import TRANSPORT_SLOTS
from .view import Sighting, WorldView

#: World units a distance is quoted against. Regions are agglomerated at 400 and a match is fought over a few thousand, so this puts an ordinary march near one.
DISTANCE_SCALE = 2000.0

#: World units a weapon range is quoted against. The longest built-in reach measured is 400.
RANGE_SCALE = 400.0

#: World units a squad's scatter is quoted against, which is the radius the fight around it is cut at.
#:
#: Not the distance a scattering squad is pushed out to, which is 140. Scatter is measured as the root mean square distance of the members from their centre and the feature is clipped at one, so quoting it against 140 would put a squad that had merely done as it was told at the top of the range and leave nothing above it to say that a squad has come apart.
SPREAD_SCALE = 400.0

#: Members a squad is quoted against. The largest doctrine establishes at ten, so a full squad sits a little below one and an over strength one above it.
SQUAD_SIZE_SCALE = 12.0

#: Game milliseconds a deadline or an elapsed mission is quoted against. The tactical reward horizon the design states is ten to sixty seconds and an operational errand is a couple of minutes, so a minute is the natural unit for both.
TIME_SCALE = 60000.0

#: Game milliseconds a whole match is quoted against, for the one feature that is about how late it is.
MATCH_SCALE = 1200000.0

#: Credits the treasury is quoted against. An opening holds four thousand and an economy that is working never sits far above it, because credits standing still are an army that was not built.
CREDIT_SCALE = 5000.0

#: Income per second the economy is quoted against, roughly what a well expanded side reaches.
INCOME_SCALE = 100.0

#: A hit no older than this means the unit is still under fire rather than merely damaged, matching the figure the script tactical layer reads.
RECENT_HIT_MS = 2000

#: Credits a mission's losses are quoted against: a few tanks' worth, so that the few hundred credits the tactical layer's judgement of an exchange starts at sit well inside the range.
LOSS_SCALE = 2000.0

ROLES: Tuple[Role, ...] = tuple(Role)
TASKS: Tuple[Task, ...] = tuple(Task)
STANCES: Tuple[Stance, ...] = tuple(Stance)
STATUSES: Tuple[Status, ...] = tuple(Status)
POSTURES: Tuple[Posture, ...] = tuple(Posture)
DOCTRINE_LIST: Tuple[Doctrine, ...] = tuple(Doctrine)
DEVIATIONS: Tuple[Deviation, ...] = tuple(Deviation)
DOMAINS: Tuple[Domain, ...] = tuple(Domain)

#: How far from a region's centre enemy warships and aircraft count as a threat to a landing there.
SEA_THREAT_RADIUS = 600.0


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else (high if value > high else value)


def _share(part: float, against: float) -> float:
    """One worth against another, on nought to one. Every comparison of two forces in this module goes through it, so that a feature says who is ahead and never by how many credits."""
    total = part + against
    return part / total if total > 0 else 0.5


def _one_hot(value: int, options: Sequence) -> List[float]:
    return [1.0 if int(option) == int(value) else 0.0 for option in options]


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


# ---- the tactical cut ------------------------------------------------------------------

#: What the tactical features are, in order. Named rather than merely counted because a feature vector that has silently shifted by one is the failure this file exists to prevent, and because a trained policy is only readable beside the names of what it was reading.
TACTICAL_FEATURES: Tuple[str, ...] = (
    "members", "health", "spread", "mean_health", "share_under_fire",
    "budget_spent", "budget_share", "deadline_left", "mission_age",
    *tuple(f"status_{status.name.lower()}" for status in STATUSES),
    *tuple(f"task_{task.name.lower()}" for task in TASKS),
    *tuple(f"stance_{stance.name.lower()}" for stance in STANCES),
    "target_distance", "target_dx", "target_dy", "target_ours", "target_theirs",
    "enemies_near", "enemy_weight",
    *tuple(f"enemy_{role.name.lower()}" for role in ROLES),
    *tuple(f"ours_{role.name.lower()}" for role in ROLES),
    *tuple(f"enemy_domain_{domain.name.lower()}" for domain in DOMAINS),
    *tuple(f"ours_domain_{domain.name.lower()}" for domain in DOMAINS),
    "our_reach", "their_reach", "reach_advantage",
    "enemies_in_reach", "ours_in_their_reach", "artillery_present",
    "exchange", "under_fire", "engaged",
    "losses", "armed_members", "target_in_reach", "long_in_reach", "predicted",
    "bias",
)

TACTICAL_SIZE = len(TACTICAL_FEATURES)

#: The departures, which are the whole tactical action space whether a script or a network is choosing. Seven since the space was widened from five; the count follows the enum so that the network head and the mask grow with it.
TACTICAL_ACTIONS = len(DEVIATIONS)


def tactical_state(squad: SquadRecord, members: Sequence[Sighting], threats: Sequence[Sighting],
                   losses: float, killed: float, view: WorldView, game_time_ms: int,
                   combat=None) -> List[float]:
    """What one squad's fight looks like, as the tactical layer is allowed to see it.

    The script layer chooses its departure from these numbers and nothing else, which is deliberate: a learnt layer imitating it sees everything its teacher decided on, and a learnt layer that saw more would not be a replacement for it but a different layer. The comparison the design rests on -does the learnt one beat the script over many episodes -is then between two things doing the same job.

    `predicted` is what the combat table (`combat.CombatTable`) expects of a fight between the squad and what threatens it, each unit counted at the share of health it has left; nought without a table.
    """
    contract = squad.contract
    budget = contract.cost_budget if contract is not None else 0.0
    target = view.region(contract.target_region) if contract is not None else None

    live = [m for m in members if m.unit.max_health > 0]
    mean_health = sum(m.unit.health / m.unit.max_health for m in live) / len(live) if live else 0.0
    hit = sum(1 for m in members if m.unit.since_hit_ms <= RECENT_HIT_MS)
    squad_value = sum(m.value for m in members)
    threat_value = sum(t.value for t in threats)

    dx = dy = 0.0
    distance = 0.0
    if target is not None and members:
        dx, dy = target.x - squad.x, target.y - squad.y
        distance = math.hypot(dx, dy)
        if distance > 1.0:
            dx, dy = dx / distance, dy / distance

    reaches = [m.kind.range for m in members if m.kind is not None and m.kind.armed]
    enemy_reaches = [t.kind.range for t in threats if t.kind is not None and t.kind.armed]
    our_reach = min(reaches) if reaches else 0.0
    their_reach = max(enemy_reaches) if enemy_reaches else 0.0

    in_our_reach = sum(1 for t in threats if math.hypot(t.unit.x - squad.x, t.unit.y - squad.y) <= our_reach)
    in_their_reach = sum(1 for m in members if math.hypot(m.unit.x - squad.x, m.unit.y - squad.y) <= their_reach) if their_reach > 0 else 0

    # Whether something threatening is within the shortest reach of the armed members, measured from their own centre, and whether that something is a gun: the two things a concentration of fire needs to know.
    shooters = [m for m in members if m.kind is not None and m.kind.armed]
    target_in_reach = long_in_reach = False
    if shooters:
        reach = min(m.kind.range for m in shooters)
        cx = sum(m.unit.x for m in shooters) / len(shooters)
        cy = sum(m.unit.y for m in shooters) / len(shooters)
        near = [t for t in threats if math.hypot(t.unit.x - cx, t.unit.y - cy) <= reach]
        target_in_reach = bool(near)
        long_in_reach = any(t.role == Role.ARTILLERY for t in near)
    predicted = combat.outcome(_standing(members), _standing(threats)) if combat is not None else 0.0

    features: List[float] = [
        _clip(len(members) / SQUAD_SIZE_SCALE),
        _clip(squad.health),
        _clip(squad.spread / SPREAD_SCALE),
        _clip(mean_health),
        _clip(hit / len(members)) if members else 0.0,
        _clip(losses / budget) if budget > 0 else 0.0,
        _clip(contract.cost_budget / (squad_value + 1.0)) if contract is not None else 0.0,
        _clip((contract.deadline_ms - game_time_ms) / TIME_SCALE, -1.0, 1.0) if contract is not None and contract.deadline_ms else 0.0,
        _clip((game_time_ms - contract.issued_at_ms) / TIME_SCALE) if contract is not None else 0.0,
    ]
    features.extend(_one_hot(squad.status, STATUSES))
    features.extend(_one_hot(contract.task if contract is not None else Task.DEFEND, TASKS))
    features.extend(_one_hot(contract.stance if contract is not None else Stance.AGGRESSIVE, STANCES))
    features.extend([
        _clip(distance / DISTANCE_SCALE),
        _clip(dx, -1.0, 1.0),
        _clip(dy, -1.0, 1.0),
        _share(target.our_value, target.enemy_value) if target is not None else 0.5,
        _clip(target.enemy_value / (squad_value + 1.0)) if target is not None else 0.0,
        _clip(len(threats) / SQUAD_SIZE_SCALE),
        _share(threat_value, squad_value),
    ])
    features.extend(_role_shares(threats))
    features.extend(_role_shares(members))
    features.extend(_domain_shares(threats, view.catalogue))
    features.extend(_domain_shares(members, view.catalogue))
    features.extend([
        _clip(our_reach / RANGE_SCALE),
        _clip(their_reach / RANGE_SCALE),
        _clip((our_reach - their_reach) / RANGE_SCALE, -1.0, 1.0),
        _clip(in_our_reach / len(threats)) if threats else 0.0,
        _clip(in_their_reach / len(members)) if members else 0.0,
        1.0 if any(t.role == Role.ARTILLERY for t in threats) else 0.0,
        _share(killed, losses),
        1.0 if hit else 0.0,
        1.0 if threats else 0.0,
        _clip(losses / LOSS_SCALE),
        _clip(len(shooters) / SQUAD_SIZE_SCALE),
        1.0 if target_in_reach else 0.0,
        1.0 if long_in_reach else 0.0,
        _clip(predicted, -1.0, 1.0),
        1.0,
    ])
    return [_finite(value) for value in features]


def _standing(sightings: Sequence[Sighting]) -> List[Tuple[int, float]]:
    """A force as the combat table counts it: each unit's type at the share of its health it still has."""
    return [(s.unit.type_index, s.unit.health / s.unit.max_health if s.unit.max_health > 0 else 1.0) for s in sightings]


def _role_shares(sightings: Sequence[Sighting]) -> List[float]:
    total = sum(s.value for s in sightings)
    if total <= 0:
        return [0.0] * len(ROLES)
    by_role: Dict[Role, float] = {}
    for sighting in sightings:
        by_role[sighting.role] = by_role.get(sighting.role, 0.0) + sighting.value
    return [by_role.get(role, 0.0) / total for role in ROLES]


def _domain_shares(sightings: Sequence[Sighting], catalogue) -> List[float]:
    """The share of a force's worth on each domain: what it moves on, and so what can follow it and what can shoot at it. All nought without a type table."""
    total = sum(s.value for s in sightings)
    if total <= 0 or catalogue is None:
        return [0.0] * len(DOMAINS)
    by_domain: Dict[Domain, float] = {}
    for sighting in sightings:
        domain = catalogue.domain(sighting.unit.type_index)
        by_domain[domain] = by_domain.get(domain, 0.0) + sighting.value
    return [by_domain.get(domain, 0.0) / total for domain in DOMAINS]


# ---- the operational cut ---------------------------------------------------------------

GLOBAL_FEATURES: Tuple[str, ...] = (
    "credits", "income", "supply", "under_construction",
    *tuple(f"posture_{posture.name.lower()}" for posture in POSTURES),
    "offensive", "allowance", "military_edge", "region_edge", "elapsed", "squads", "bias",
)

REGION_FEATURES: Tuple[str, ...] = (
    "valid", "resources", "held_by_us", "held_by_enemy", "force_edge",
    "enemy_present", "distance", "seen_recently", "priority", "spawn",
    "ours_present", "from_squad", "current_target", "committed", "predicted", "defences", "plan",
    "walkable", "carriable", "coastal", "sea_threat",
)

SQUAD_FEATURES: Tuple[str, ...] = (
    "valid",
    *tuple(f"doctrine_{doctrine.name.lower()}" for doctrine in DOCTRINE_LIST),
    "weight", "health", "spread", "distance",
    *tuple(f"status_{status.name.lower()}" for status in STATUSES),
    "held", "taskable", "budget_spent", "mission_age",
    *tuple(f"task_{task.name.lower()}" for task in TASKS),
    "plan_distance",
    *tuple(f"domain_{domain.name.lower()}" for domain in DOMAINS),
    "aboard",
)

#: One row per transport slot of the lift layer: whether a transport stands in it, whether it is free, whether it is carrying the squad decided about, how far it is from that squad, how full it is, whether it moves over land and water or flies, and whether it loads every member of the squad.
TRANSPORT_FEATURES: Tuple[str, ...] = (
    "valid", "free", "mine", "distance", "load", "hover", "air", "carries",
)

GLOBAL_SIZE = len(GLOBAL_FEATURES)
REGION_SIZE = len(REGION_FEATURES)
SQUAD_SIZE = len(SQUAD_FEATURES)
TRANSPORT_SIZE = len(TRANSPORT_FEATURES)
OPERATIONAL_SIZE = GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + SQUAD_SLOTS * SQUAD_SIZE + TRANSPORT_SLOTS * TRANSPORT_SIZE

#: One decision is a region and a plan, which is a task and a means: walking, or being carried by the transport in one of the lift layer's slots. Kept factorised rather than flattened because the two are chosen for different reasons -where is worth going, and what to do there and how to get there- and because a mask over the flattened space would be mostly dead outputs where one row of plans per region is small. The plan's mask depends on the region, since which transports can reach a region differs from one region to the next.
OPERATIONAL_REGIONS = REGION_SLOTS
OPERATIONAL_TASKS = len(TASKS)
#: The means a squad goes by: on its own, or carried by the transport in one of the lift layer's slots.
MEANS = 1 + TRANSPORT_SLOTS
OPERATIONAL_PLANS = OPERATIONAL_TASKS * MEANS


def plan_of(task: int, means: int) -> int:
    """The plan of a task and a means, where a means of -1 is walking and k is the transport slot k."""
    return int(task) * MEANS + (int(means) + 1)


def task_of(plan: int) -> int:
    return int(plan) // MEANS


def means_of(plan: int) -> int:
    """The transport slot of a plan, or -1 for walking."""
    return int(plan) % MEANS - 1


@dataclass
class TransportView:
    """One transport slot as the operational layer sees it."""

    slot: int
    valid: bool = False
    x: float = 0.0
    y: float = 0.0
    aboard: int = 0
    capacity: int = 0
    #: The transport's movement type.
    movement: str = ""
    #: Lent to another squad's lift or given to another squad this period, and whether it is lent to the squad decided about.
    busy: bool = False
    mine: bool = False
    #: Whether it loads every member of the squad decided about.
    carries: bool = False


@dataclass
class Access:
    """Where the squad decided about can get to, and by which means: the regions it can walk to, for each other region the transport slots that could carry it there, the coastal regions, and the transport slots themselves."""

    walk: Set[int] = field(default_factory=set)
    lift: Dict[int, List[int]] = field(default_factory=dict)
    coastal: Set[int] = field(default_factory=set)
    transports: List[TransportView] = field(default_factory=list)

#: How recently the enemy must have been seen in a region for the feature to read as contact rather than as memory.
CONTACT_WINDOW_MS = 60000


@dataclass
class Forces:
    """Who stands in each region, worked out once a period and shared by every squad decided about in it: the enemy's fighters and armed buildings and our own fighters, by region, as the combat table counts forces, and each squad's fighting members."""

    enemy: Dict[int, List[Tuple[int, float]]]
    defences: Dict[int, List[Tuple[int, float]]]
    defence_value: Dict[int, float]
    ours: Dict[int, List[Tuple[int, float]]]
    members: Dict[int, List[Tuple[int, float]]]


def forces(view: WorldView, squads: Sequence[SquadRecord]) -> Forces:
    """Places every fighter and every armed enemy building in the region whose centre is nearest it, which is how the game side places them too."""
    enemy: Dict[int, List[Tuple[int, float]]] = {}
    defences: Dict[int, List[Tuple[int, float]]] = {}
    defence_value: Dict[int, float] = {}
    ours: Dict[int, List[Tuple[int, float]]] = {}
    if view.regions:
        def nearest(unit) -> int:
            return min(view.regions, key=lambda r: (r.x - unit.x) ** 2 + (r.y - unit.y) ** 2).id

        for sighting in view.enemies:
            if sighting.role == Role.STRUCTURE:
                if sighting.kind is not None and sighting.kind.armed:
                    region = nearest(sighting.unit)
                    defences.setdefault(region, []).extend(_standing([sighting]))
                    defence_value[region] = defence_value.get(region, 0.0) + sighting.value
            elif sighting.role != Role.BUILDER:
                enemy.setdefault(nearest(sighting.unit), []).extend(_standing([sighting]))
        for sighting in view.fighters:
            ours.setdefault(nearest(sighting.unit), []).extend(_standing([sighting]))
    by_id = {s.unit.id: s for s in view.fighters}
    members = {squad.id: _standing([by_id[m] for m in squad.members if m in by_id]) for squad in squads}
    return Forces(enemy=enemy, defences=defences, defence_value=defence_value, ours=ours, members=members)


def operational_state(view: WorldView, orders, squads: Sequence[SquadRecord],
                      game_time_ms: int, spawns: Sequence[int] = (), squad: Optional[SquadRecord] = None,
                      combat=None, present: Optional[Forces] = None, access: Optional[Access] = None) -> List[float]:
    """The whole board as the operational layer sees it when deciding about one squad: aggregates, twenty-four region slots, eight squad slots and the lift layer's transport slots, always in that order and always that long.

    `access` says where the squad can get to and by which transport; without it every region reads as walkable and no transport stands anywhere.

    Starting positions are passed in rather than read off the observation because the wire's region row does not carry the flag: it comes from the map file, which is read in this process when the region table is built. Which regions are starting positions is the difference between a place with resources on it and the place the enemy came from, and no other feature says it.

    Part of each region's row is about the squad being decided: how far the region is from it, whether it is where the squad is already bound, how much of the rest of our army is bound there too, and how the squad would fare there -the squad, what is bound there and what of ours already stands there, against the enemy's fighters and defences there, as the combat table (`combat.CombatTable`) predicts it. Without a squad those read nought, and without a table so does the prediction.
    """
    observation = view.observation
    ours = sum(s.value for s in view.fighters)
    theirs = sum(s.value for s in view.enemies)
    held = sum(region.held_by_us for region in view.regions)
    enemy_held = sum(region.held_by_enemy for region in view.regions)

    state: List[float] = [
        _clip(observation.credits / CREDIT_SCALE),
        _clip(observation.income / INCOME_SCALE),
        _clip(observation.units / observation.unit_cap) if observation.unit_cap else 0.0,
        _clip(observation.under_construction / 8.0),
    ]
    state.extend(_one_hot(orders.posture if orders is not None else Posture.EXPAND, POSTURES))
    state.extend([
        1.0 if orders is not None and orders.offensive else 0.0,
        _clip(orders.loss_allowance / (ours + 1.0)) if orders is not None else 0.0,
        _share(ours, theirs),
        _share(held, enemy_held),
        _clip(game_time_ms / MATCH_SCALE),
        _clip(len(squads) / SQUAD_SLOTS),
        1.0,
    ])

    priorities = orders.priorities if orders is not None else {}
    plan = list(orders.expansion) if orders is not None else []
    by_slot = {region.id: region for region in view.regions}
    starts = frozenset(spawns)
    if squad is not None and present is None:
        present = forces(view, squads)
    bound: Dict[int, List[SquadRecord]] = {}
    for other in squads:
        if squad is not None and other.id == squad.id:
            continue
        if other.contract is not None and DOCTRINES[other.doctrine].tasks:
            bound.setdefault(other.contract.target_region, []).append(other)
    for slot in range(REGION_SLOTS):
        region = by_slot.get(slot)
        if region is None:
            state.extend([0.0] * REGION_SIZE)
            continue
        # The first region of the expansion plan reads one and the rest fall away in the plan's order; a region not in the plan reads nought.
        rank = 1.0 - plan.index(region.id) / len(plan) if region.id in plan else 0.0
        state.extend(_region_row(region, priorities, ours, game_time_ms, starts))
        state.extend(_squad_view(region, squad, bound.get(region.id, []), ours, combat, present, rank))
        state.extend(_access_row(region, access, view, ours))

    by_id = {record.id: record for record in squads}
    head = by_slot.get(plan[0]) if plan else None
    for slot in range(SQUAD_SLOTS):
        state.extend(_squad_row(by_id.get(slot), ours, game_time_ms, view, head))

    transports = {t.slot: t for t in access.transports} if access is not None else {}
    for slot in range(TRANSPORT_SLOTS):
        state.extend(_transport_row(transports.get(slot), squad))

    return [_finite(value) for value in state]


def _access_row(region: RegionState, access: Optional[Access], view: WorldView, ours: float) -> List[float]:
    """How the squad decided about gets to a region: whether it can walk there, whether a transport could carry it there, whether the region is on the coast, and what of the enemy that floats or flies stands near it, against our whole army."""
    threat = sum(s.value for s in view.enemies_near(region.x, region.y, SEA_THREAT_RADIUS)
                 if view.catalogue is not None and view.catalogue.domain(s.unit.type_index) in (Domain.NAVAL, Domain.AIR))
    if access is None:
        return [1.0, 0.0, 0.0, _clip(threat / (ours + 1.0))]
    return [
        1.0 if region.id in access.walk else 0.0,
        1.0 if access.lift.get(region.id) else 0.0,
        1.0 if region.id in access.coastal else 0.0,
        _clip(threat / (ours + 1.0)),
    ]


def transport_distance(dx: float, dy: float) -> float:
    """How far a transport stands from a squad, as the transport block writes it."""
    return _clip(math.hypot(dx, dy) / (2 * DISTANCE_SCALE))


def _transport_row(transport: Optional[TransportView], squad: Optional[SquadRecord]) -> List[float]:
    if transport is None or not transport.valid:
        return [0.0] * TRANSPORT_SIZE
    return [
        1.0,
        0.0 if transport.busy else 1.0,
        1.0 if transport.mine else 0.0,
        transport_distance(transport.x - squad.x, transport.y - squad.y) if squad is not None else 0.0,
        _clip(transport.aboard / transport.capacity) if transport.capacity > 0 else 0.0,
        1.0 if transport.movement in ("HOVER", "OVER_CLIFF_WATER") else 0.0,
        1.0 if transport.movement == "AIR" else 0.0,
        1.0 if transport.carries else 0.0,
    ]


def _region_row(region: RegionState, priorities: Dict[int, float], ours: float,
                game_time_ms: int, spawns: frozenset) -> List[float]:
    """The part of a region's row that is the same whichever squad is being decided about."""
    return [
        1.0,
        _clip(region.resources / 4.0),
        _clip(region.held_by_us / 4.0),
        _clip(region.held_by_enemy / 4.0),
        _share(region.our_value, region.enemy_value),
        _clip(region.enemy_value / (ours + 1.0)),
        _clip(region.distance_from_home / (2 * DISTANCE_SCALE)),
        1.0 if region.enemy_seen_at_ms and game_time_ms - region.enemy_seen_at_ms <= CONTACT_WINDOW_MS else 0.0,
        _clip(priorities.get(region.id, 0.0)),
        1.0 if region.id in spawns else 0.0,
        _clip(region.our_value / (ours + 1.0)),
    ]


def _squad_view(region: RegionState, squad: Optional[SquadRecord], bound: Sequence[SquadRecord],
                ours: float, combat, present: Optional[Forces], rank: float) -> List[float]:
    """The part of a region's row that is about the squad being decided: distance, whether bound there already, what else is bound there and the predicted fight there, followed by the enemy's defences there and the region's place in the expansion plan, which are worked out from the same forces."""
    committed = sum(other.value for other in bound)
    defences = present.defence_value.get(region.id, 0.0) if present is not None else 0.0
    if squad is None:
        return [0.0, 0.0, _clip(committed / (ours + 1.0)), 0.0, _clip(defences / (ours + 1.0)), rank]
    distance = math.hypot(region.x - squad.x, region.y - squad.y)
    current = squad.contract is not None and squad.contract.target_region == region.id
    predicted = 0.0
    if combat is not None and present is not None:
        attackers = list(present.members.get(squad.id, []))
        for other in bound:
            attackers.extend(present.members.get(other.id, []))
        attackers.extend(present.ours.get(region.id, []))
        defenders = present.enemy.get(region.id, []) + present.defences.get(region.id, [])
        predicted = combat.outcome(attackers, defenders) if defenders else 1.0
    return [
        _clip(distance / (2 * DISTANCE_SCALE)),
        1.0 if current else 0.0,
        _clip(committed / (ours + 1.0)),
        _clip(predicted, -1.0, 1.0),
        _clip(defences / (ours + 1.0)),
        rank,
    ]


def _squad_row(squad: Optional[SquadRecord], ours: float, game_time_ms: int,
               view: WorldView, head: Optional[RegionState] = None) -> List[float]:
    if squad is None:
        return [0.0] * SQUAD_SIZE
    home = view.home
    distance = math.hypot(squad.x - home.x, squad.y - home.y) if home is not None else 0.0
    contract = squad.contract
    row: List[float] = [1.0]
    row.extend(_one_hot(squad.doctrine, DOCTRINE_LIST))
    row.extend([
        _clip(squad.value / (ours + 1.0)),
        _clip(squad.health),
        _clip(squad.spread / SPREAD_SCALE),
        _clip(distance / (2 * DISTANCE_SCALE)),
    ])
    row.extend(_one_hot(squad.status, STATUSES))
    row.extend([
        1.0 if squad.commander else 0.0,
        1.0 if squad.ours_to_task else 0.0,
        _clip(squad.losses / contract.cost_budget) if contract is not None and contract.cost_budget > 0 else 0.0,
        _clip((game_time_ms - contract.issued_at_ms) / TIME_SCALE) if contract is not None else 0.0,
    ])
    # The task of the contract it holds, all noughts for a squad that holds none.
    row.extend(_one_hot(contract.task, TASKS) if contract is not None else [0.0] * len(TASKS))
    # How far it stands from the first region of the expansion plan, which is what says which of several free garrisons is the one to go and cover it; nought with no plan.
    row.append(_clip(math.hypot(head.x - squad.x, head.y - squad.y) / (2 * DISTANCE_SCALE)) if head is not None else 0.0)
    row.extend(_one_hot(squad.domain, DOMAINS))
    row.append(_clip(squad.aboard / max(1, len(squad.members))))
    return row


# ---- what a layer is allowed to answer --------------------------------------------------

def region_mask(view: WorldView) -> List[float]:
    """Which region slots exist on this map. A slot with no region behind it is not a target a policy may choose, and masking is how that is said rather than hoping the policy learns it."""
    live = {region.id for region in view.regions}
    return [1.0 if slot in live else 0.0 for slot in range(REGION_SLOTS)]


def task_mask(doctrine: Doctrine) -> List[float]:
    """Which tasks a squad of this doctrine may be given, read off the same doctrine table the script layer reads. Engineers have no tasks at all, which is how the design keeps a contract from landing on a builder in the middle of a placement and turning a half raised building into a total loss."""
    allowed = set(int(task) for task in DOCTRINES[doctrine].tasks)
    return [1.0 if int(task) in allowed else 0.0 for task in TASKS]


def plan_masks(view: WorldView, doctrine: Doctrine, access: Optional[Access] = None) -> List[List[float]]:
    """For every region slot, which plans the squad may be given there: a task its doctrine allows, by walking where it can walk there and by the transport in slot k where that transport can carry it there. A region it can get to by no means has no plan at all."""
    live = region_mask(view)
    tasks = task_mask(doctrine)
    rows: List[List[float]] = []
    for slot in range(REGION_SLOTS):
        row = [0.0] * OPERATIONAL_PLANS
        if live[slot] > 0:
            means = [access is None or slot in access.walk]
            means.extend(access is not None and k in access.lift.get(slot, ()) for k in range(TRANSPORT_SLOTS))
            for task, allowed in enumerate(tasks):
                if allowed <= 0:
                    continue
                for index, possible in enumerate(means):
                    if possible:
                        row[task * MEANS + index] = 1.0
        rows.append(row)
    return rows


def regions_of(masks: Sequence[Sequence[float]]) -> List[float]:
    """The region mask the plan masks imply: a region is open when some plan is."""
    return [1.0 if any(value > 0 for value in row) else 0.0 for row in masks]


def squad_mask(squads: Sequence[SquadRecord]) -> List[float]:
    """Which squad slots hold a squad this layer may task: one that exists, has anyone left in it, has a doctrine with tasks, and has not been taken over by somebody else."""
    by_id = {squad.id: squad for squad in squads}
    mask: List[float] = []
    for slot in range(SQUAD_SLOTS):
        squad = by_id.get(slot)
        usable = (squad is not None and bool(squad.members) and squad.ours_to_task
                  and bool(DOCTRINES[squad.doctrine].tasks))
        mask.append(1.0 if usable else 0.0)
    return mask


# ---- the economic cut -------------------------------------------------------------------

class Investment(enum.IntEnum):
    """What the next credits of a period can go to. Stopping is an investment too: it is the answer that leaves the rest of the treasury for the next period."""

    STOP = 0
    UNIT = 1
    BUILDER = 2
    EXTRACTOR = 3
    FACTORY = 4
    RAISE = 5
    TURRET = 6


INVESTMENTS: Tuple[Investment, ...] = tuple(Investment)

#: Slots each kind of investment is given in the economic action space, in the order the slots are laid out. Stopping has the first slot to itself; the units a factory could make have the most, because a factory's menu is the widest choice the economy makes; a tier raise has one slot per stage.
INVESTMENT_CAPACITY: Tuple[Tuple[Investment, int], ...] = (
    (Investment.STOP, 1), (Investment.UNIT, 16), (Investment.BUILDER, 1), (Investment.EXTRACTOR, 6),
    (Investment.FACTORY, 4), (Investment.RAISE, 3), (Investment.TURRET, 1),
)

INVESTMENT_SLOTS = sum(count for _, count in INVESTMENT_CAPACITY)

#: Stages of a tier raise: an extractor to its second tier, a factory to its next, an extractor to its third.
RAISE_STAGES = 3

#: Credits the economy's treasury is quoted against. Wider than the operational cut's scale because the economy is the layer that decides what to do with a treasury that has grown, and a feature that read every banked treasury as the same one could not say how far it has grown.
TREASURY_SCALE = 10000.0

#: Credits a price is quoted against: above the dearest thing a factory offers in an ordinary match.
PRICE_SCALE = 20000.0

#: Counts of buildings, builders and resource points are quoted against this.
COUNT_SCALE = 10.0

#: Resource points in a region are quoted against this, as in the operational cut.
POINTS_SCALE = 4.0

#: Technology cap, in credits per minute, is quoted against this.
TECH_CAP_SCALE = 1000.0

#: The highest technology tier a factory reaches.
TIER_SCALE = 3.0

#: The roles a factory is asked to fill, in the order the target mix names them.
FIELDED: Tuple[Role, ...] = (Role.ARMOUR, Role.ARTILLERY, Role.ANTI_AIR, Role.FAST)

ECONOMIC_CONTEXT: Tuple[str, ...] = (
    "credits", "budget", "reserve", "income", "supply", "near_cap", "at_cap",
    "factories", "pending_factory", "free_factories", "factory_raising",
    "builders", "builder_shortfall", "idle_builders", "extractors", "home_open", "plan_open",
    *tuple(f"posture_{posture.name.lower()}" for posture in POSTURES),
    "economy_share", "military_share", "tech_share", "tech_cap", "tech_fund", "tech_level",
    *tuple(f"mix_{role.name.lower()}" for role in FIELDED),
    "value_edge", "army_edge", "ground_edge", "enemy_air", "rebuilding",
    "transports", "transport_wanted", "water_front", "picks", "elapsed", "bias",
)

INVESTMENT_FEATURES: Tuple[str, ...] = (
    "valid",
    *tuple(f"kind_{kind.name.lower()}" for kind in INVESTMENTS),
    "price", "affordable", "reach", "price_share",
    "efficiency", "unit_efficiency", "fighting",
    *tuple(f"role_{role.name.lower()}" for role in ROLES),
    "flying", "hits_air", "hits_land", "range", "tier", "army_share", "asked",
    "role_rank", "cheapest_of_role", "priciest_fitting",
    "home", "contested", "safety", "plan_rank", "distance", "open_points",
    "land_factory", "worth", "count",
    *tuple(f"stage_{stage}" for stage in range(RAISE_STAGES)),
    "gain", "funded", "contact", "transport", "naval", "lift", "ready",
)

ECONOMIC_CONTEXT_SIZE = len(ECONOMIC_CONTEXT)
INVESTMENT_SIZE = len(INVESTMENT_FEATURES)
ECONOMIC_SIZE = ECONOMIC_CONTEXT_SIZE + INVESTMENT_SLOTS * INVESTMENT_SIZE


@dataclass
class Offer:
    """One investment open to the economy this period, as the economy found it: what it is, what it costs, and what the rules and the network judge it by. Which builder, which factory and which ground carry it out is `payload`, which nothing here reads."""

    kind: Investment
    price: float = 0.0
    #: The type made or placed: a unit, a builder, a factory's kind, a turret; -1 for a raise.
    type_index: int = -1
    #: Whether the price falls within what the treasury will hold over the saving horizon.
    in_reach: bool = True
    #: A unit: the fighting strength it buys against what the enemy fields, per credit and per unit, before they are put against the best on offer.
    efficiency: float = 0.0
    unit_efficiency: float = 0.0
    fighting: bool = False
    role: Role = Role.OTHER
    #: Where its role stands in the order the target mix and the squads' shortfalls ask for roles: one for the most wanted, falling to nought for a role the factories are not asked to fill.
    role_rank: float = 0.0
    flying: bool = False
    hits_air: bool = False
    hits_land: bool = False
    range: float = 0.0
    tier: int = 0
    #: Its type's share of the worth our army already fields, and its role's share of what the squads say they are short of.
    army_share: float = 0.0
    asked: float = 0.0
    #: An extractor: the region it goes to, whether that is home, whether the enemy stands in it, and the safety rank of the point it would take, in world units, lower being safer.
    region: int = -1
    home: bool = False
    contested: bool = False
    safety: float = 0.0
    plan_rank: float = 0.0
    distance: float = 0.0
    open_points: int = 0
    #: A factory: whether it is the land factory, what the best its first tier makes is worth against the enemy, and how many of the kind stand.
    land_factory: bool = False
    worth: float = 0.0
    count: int = 0
    #: A raise: its stage, how much more its next tier makes than its current one, less one, and whether the technology fund pays for it.
    stage: int = -1
    gain: float = 0.0
    funded: bool = False
    #: A turret: whether the ground it would stand on has seen the enemy lately.
    contact: bool = False
    #: A unit: whether it is a transport and whether it is a warship. An extractor: whether taking it means carrying a builder across first, and whether an idle builder able to place it is already standing in its region.
    transport: bool = False
    naval: bool = False
    lift: bool = False
    ready: bool = False
    payload: object = None


@dataclass
class EconomicBoard:
    """What the economy's context row is made of, as one pick in a period finds it."""

    credits: float
    reserve: float
    income: float
    units: int
    unit_cap: int
    near_cap: bool
    at_cap: bool
    #: Factories standing, and one ordered and not yet standing when there is none.
    factories: int
    pending_factory: bool
    free_factories: int
    factory_raising: bool
    builders: int
    builder_shortfall: int
    idle_builders: int
    extractors: int
    home_open: int
    plan_open: int
    posture: Posture
    economy_share: float
    military_share: float
    tech_share: float
    tech_cap: float
    tech_fund: float
    tech_fund_limit: float
    tech_level: int
    target_mix: Dict[Role, float] = field(default_factory=dict)
    value_edge: float = 0.5
    army_edge: float = 0.5
    ground_edge: float = 0.5
    enemy_air: float = 0.0
    #: A factory has stood this match and none stands now.
    rebuilding: bool = False
    #: Transports standing or on order, whether the lift layer has a request no transport could serve, and whether there is a fight on the water.
    transports: int = 0
    transport_wanted: bool = False
    water_front: bool = False
    picks: int = 0
    game_time_ms: int = 0

    @property
    def budget(self) -> float:
        return self.credits - self.reserve


def lay_out(offers: Sequence[Offer]) -> List[Optional[Offer]]:
    """The offers placed in the fixed slots of the economic action space, stopping first.

    Each kind is given its slots in `INVESTMENT_CAPACITY` order and fills them in an order that does not depend on how the offers were found: units by type, extractors by region, factories by kind, a raise in the slot of its stage. When a kind has more offers than slots, the ones kept are those a rule would reach for first -a unit within reach and strongest per credit, an extractor at home and then the safest -so that what is cut off is what nothing would have chosen.
    """
    by_kind: Dict[Investment, List[Offer]] = {}
    for offer in offers:
        by_kind.setdefault(offer.kind, []).append(offer)
    slots: List[Optional[Offer]] = []
    for kind, count in INVESTMENT_CAPACITY:
        found = by_kind.get(kind, [])
        if kind == Investment.STOP:
            slots.append(found[0] if found else Offer(kind=Investment.STOP))
            continue
        if kind == Investment.RAISE:
            staged: List[Optional[Offer]] = [None] * count
            for offer in found:
                if 0 <= offer.stage < count and staged[offer.stage] is None:
                    staged[offer.stage] = offer
            slots.extend(staged)
            continue
        if kind == Investment.UNIT:
            kept = sorted(found, key=lambda o: (not o.in_reach, -o.efficiency, o.type_index))[:count]
            kept.sort(key=lambda o: o.type_index)
        elif kind == Investment.EXTRACTOR:
            kept = sorted(found, key=lambda o: (not o.home, o.contested, o.safety, o.region))[:count]
            kept.sort(key=lambda o: o.region)
        else:
            kept = sorted(found, key=lambda o: o.type_index)[:count]
        slots.extend(kept + [None] * (count - len(kept)))
    return slots


def investment_mask(slots: Sequence[Optional[Offer]]) -> List[float]:
    """Which slots hold an investment. Every offer is allowed, including one the treasury cannot pay for yet, which is how saving for it is said; stopping always is."""
    return [1.0 if offer is not None else 0.0 for offer in slots]


def economic_state(board: EconomicBoard, slots: Sequence[Optional[Offer]]) -> List[float]:
    """The economy's board as the build order sees it when choosing its next investment: a context row, then one row per slot of the action space, always in that order and always that long.

    Everything that compares one offer with the others is worked out here from the offers themselves, so that a rule reading the rows and a network reading them see the same comparisons: each unit's strength against the strongest unit on offer, the cheapest of each role and the dearest of each role the budget covers.
    """
    budget = board.budget
    state: List[float] = [
        _clip(board.credits / TREASURY_SCALE),
        _clip(budget / TREASURY_SCALE, -1.0, 1.0),
        _clip(board.reserve / TREASURY_SCALE),
        _clip(board.income / INCOME_SCALE),
        _clip(board.units / board.unit_cap) if board.unit_cap else 0.0,
        1.0 if board.near_cap else 0.0,
        1.0 if board.at_cap else 0.0,
        _clip(board.factories / COUNT_SCALE),
        1.0 if board.pending_factory else 0.0,
        _clip(board.free_factories / COUNT_SCALE),
        1.0 if board.factory_raising else 0.0,
        _clip(board.builders / COUNT_SCALE),
        _clip(board.builder_shortfall / COUNT_SCALE, -1.0, 1.0),
        _clip(board.idle_builders / COUNT_SCALE),
        _clip(board.extractors / (2 * COUNT_SCALE)),
        _clip(board.home_open / COUNT_SCALE),
        _clip(board.plan_open / COUNT_SCALE),
    ]
    state.extend(_one_hot(board.posture, POSTURES))
    state.extend([
        _clip(board.economy_share),
        _clip(board.military_share),
        _clip(board.tech_share),
        _clip(board.tech_cap / TECH_CAP_SCALE),
        _clip(board.tech_fund / board.tech_fund_limit) if board.tech_fund_limit > 0 else 0.0,
        _clip(board.tech_level / TIER_SCALE),
    ])
    state.extend(_clip(board.target_mix.get(role, 0.0)) for role in FIELDED)
    state.extend([
        _clip(board.value_edge),
        _clip(board.army_edge),
        _clip(board.ground_edge),
        _clip(board.enemy_air),
        1.0 if board.rebuilding else 0.0,
        _clip(board.transports / COUNT_SCALE),
        1.0 if board.transport_wanted else 0.0,
        1.0 if board.water_front else 0.0,
        _clip(board.picks / INVESTMENT_SLOTS),
        _clip(board.game_time_ms / MATCH_SCALE),
        1.0,
    ])

    units = [o for o in slots if o is not None and o.kind == Investment.UNIT]
    best = max((o.efficiency for o in units), default=0.0)
    best_unit = max((o.unit_efficiency for o in units), default=0.0)
    cheapest: Dict[Role, Offer] = {}
    priciest: Dict[Role, Offer] = {}
    for offer in units:
        if offer.role not in cheapest or offer.price < cheapest[offer.role].price:
            cheapest[offer.role] = offer
        if offer.price <= budget and (offer.role not in priciest or offer.price > priciest[offer.role].price):
            priciest[offer.role] = offer
    worths = [o.worth for o in slots if o is not None and o.kind == Investment.FACTORY]
    best_worth = max(worths, default=0.0)

    for offer in slots:
        if offer is None:
            state.extend([0.0] * INVESTMENT_SIZE)
            continue
        row: List[float] = [1.0]
        row.extend(_one_hot(offer.kind, INVESTMENTS))
        row.extend([
            _clip(offer.price / PRICE_SCALE),
            1.0 if offer.price <= budget else 0.0,
            1.0 if offer.in_reach else 0.0,
            _share(offer.price, max(0.0, budget)) if offer.price > 0 else 0.0,
            offer.efficiency / best if best > 0 else 0.0,
            offer.unit_efficiency / best_unit if best_unit > 0 else 0.0,
            1.0 if offer.fighting else 0.0,
        ])
        row.extend(_one_hot(offer.role, ROLES) if offer.kind == Investment.UNIT else [0.0] * len(ROLES))
        row.extend([
            1.0 if offer.flying else 0.0,
            1.0 if offer.hits_air else 0.0,
            1.0 if offer.hits_land else 0.0,
            _clip(offer.range / RANGE_SCALE),
            _clip(offer.tier / TIER_SCALE),
            _clip(offer.army_share),
            _clip(offer.asked),
            _clip(offer.role_rank),
            1.0 if cheapest.get(offer.role) is offer and offer.kind == Investment.UNIT else 0.0,
            1.0 if priciest.get(offer.role) is offer and offer.kind == Investment.UNIT else 0.0,
            1.0 if offer.home else 0.0,
            1.0 if offer.contested else 0.0,
            _clip(offer.safety / (4 * DISTANCE_SCALE), -1.0, 1.0),
            _clip(offer.plan_rank),
            _clip(offer.distance / (2 * DISTANCE_SCALE)),
            _clip(offer.open_points / POINTS_SCALE),
            1.0 if offer.land_factory else 0.0,
            offer.worth / best_worth if best_worth > 0 else 0.0,
            _clip(offer.count / COUNT_SCALE),
        ])
        row.extend(1.0 if offer.stage == stage else 0.0 for stage in range(RAISE_STAGES))
        row.extend([
            _clip(offer.gain, -1.0, 1.0),
            1.0 if offer.funded else 0.0,
            1.0 if offer.contact else 0.0,
            1.0 if offer.transport else 0.0,
            1.0 if offer.naval else 0.0,
            1.0 if offer.lift else 0.0,
            1.0 if offer.ready else 0.0,
        ])
        state.extend(row)
    return [_finite(value) for value in state]


# ---- which encoding a recorded state was written by -------------------------------------

#: The revision of each layer's encoding. Raised whenever what a feature means changes, including a change that leaves every name and length as it was; `tests/test_dataset.py` pins the vector each layer's encoding gives a fixed board to its revision.
TACTICAL_REVISION = 4
OPERATIONAL_REVISION = 5
ECONOMIC_REVISION = 2


def describe(layer: str) -> dict:
    """Everything that identifies one layer's encoding: its revision, its feature names in order, its action space and the scales its features are quoted against."""
    scales = {"distance": DISTANCE_SCALE, "range": RANGE_SCALE, "spread": SPREAD_SCALE, "squad_size": SQUAD_SIZE_SCALE,
              "time": TIME_SCALE, "match": MATCH_SCALE, "recent_hit_ms": RECENT_HIT_MS}
    if layer == "tactics":
        scales.update(loss=LOSS_SCALE)
        return {"layer": layer, "revision": TACTICAL_REVISION, "features": list(TACTICAL_FEATURES),
                "actions": TACTICAL_ACTIONS, "scales": scales}
    if layer == "operations":
        scales.update(credit=CREDIT_SCALE, income=INCOME_SCALE, contact_window_ms=CONTACT_WINDOW_MS,
                      sea_threat=SEA_THREAT_RADIUS)
        return {"layer": layer, "revision": OPERATIONAL_REVISION, "global": list(GLOBAL_FEATURES),
                "region": list(REGION_FEATURES), "squad": list(SQUAD_FEATURES),
                "transport": list(TRANSPORT_FEATURES), "regions": REGION_SLOTS, "squads": SQUAD_SLOTS,
                "transports": TRANSPORT_SLOTS, "tasks": OPERATIONAL_TASKS, "plans": OPERATIONAL_PLANS, "scales": scales}
    if layer == "economy":
        scales.update(income=INCOME_SCALE, treasury=TREASURY_SCALE, price=PRICE_SCALE, count=COUNT_SCALE,
                      points=POINTS_SCALE, tech_cap=TECH_CAP_SCALE, tier=TIER_SCALE)
        return {"layer": layer, "revision": ECONOMIC_REVISION, "context": list(ECONOMIC_CONTEXT),
                "offer": list(INVESTMENT_FEATURES),
                "slots": [[kind.name.lower(), count] for kind, count in INVESTMENT_CAPACITY], "scales": scales}
    raise ValueError(f"no encoding for a layer named {layer!r}")


def fingerprint(layer: str) -> str:
    """A digest of `describe(layer)`. Two states with the same fingerprint were written by the same encoding; a dataset whose fingerprint differs from the one in force is refused rather than read."""
    return hashlib.sha256(json.dumps(describe(layer), sort_keys=True).encode("utf-8")).hexdigest()
