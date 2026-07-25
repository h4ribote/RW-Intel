"""Turning a period's board into numbers, and saying what a layer is allowed to answer.

Each layer gets its own cut, at its own abstraction, because that is how the design divides them: the tactical layer sees around one squad, the operational layer sees the regions and the squads and nothing below them. Three rules govern everything here. Every feature is a ratio or a length divided by a stated scale, so that nothing depends on how rich the match has become or how large the map is — a policy trained on one map has to be readable on another, and a raw credit total or a raw world coordinate would make that false. Every block is a fixed width with a validity flag, never a packed list, so that a slot means the same thing from one decision to the next; a packed list renumbers everything the moment a squad dies.

And no feature carries the map's own frame: a direction is always measured between two things standing on the board and never against the world's axes. A policy whose answer depends on which way round the board happens to be numbered is under-specified — it spends half its training experience learning the same thing twice in a different frame — and where one process drives both sides of a board laid out as a point reflection it is not even exchangeable between them, since the two sides then read exactly opposite directions for congruent situations and the same layer becomes two different fighters. The rule is stated here because it cannot be enforced by a scale or a width: it is a property of what a feature is made of, and the only guard on it is the pair of tests that encode a mirrored board from both sides and require the same answer.

The action spaces are exactly the ones the script layers already emit, which is what makes a learnt layer a replacement rather than a parallel system: the departures for the tactical layer, and one region and one task for the operational layer. The tactical departures were five and are now seven, the two added ones being a withdrawal that commits the whole way out and a concentration that goes onto the longest-ranged enemy rather than the weakest — kinds of move the script already made, with a parameter a rule used to fix handed to the layer. Neither space is a free choice — a squad may only be sent where a region exists and only given a task its doctrine allows — so both come with a mask, and the mask is computed here from the same tables the script reads rather than being learnt as a soft preference. The tactical action count follows the departure enum, so widening the enum widens the head and the mask with it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..wire import REGION_SLOTS, SQUAD_SLOTS, Deviation, RegionState, Stance, Status, Task
from ..control.policy.contracts import DOCTRINES, Doctrine, Posture, Role, SquadRecord
from ..control.policy.view import Sighting, WorldView

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

ROLES: Tuple[Role, ...] = tuple(Role)
TASKS: Tuple[Task, ...] = tuple(Task)
STANCES: Tuple[Stance, ...] = tuple(Stance)
STATUSES: Tuple[Status, ...] = tuple(Status)
POSTURES: Tuple[Posture, ...] = tuple(Posture)
DOCTRINE_LIST: Tuple[Doctrine, ...] = tuple(Doctrine)
DEVIATIONS: Tuple[Deviation, ...] = tuple(Deviation)


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
    "target_distance", "target_ahead", "target_abeam", "target_ours", "target_theirs",
    "enemies_near", "enemy_weight",
    *tuple(f"enemy_{role.name.lower()}" for role in ROLES),
    *tuple(f"ours_{role.name.lower()}" for role in ROLES),
    "our_reach", "their_reach", "reach_advantage",
    "enemies_in_reach", "ours_in_their_reach", "artillery_present",
    "exchange", "under_fire", "engaged", "bias",
)

TACTICAL_SIZE = len(TACTICAL_FEATURES)

#: The departures, which are the whole tactical action space whether a script or a network is choosing. Seven since the space was widened from five; the count follows the enum so that the network head and the mask grow with it.
TACTICAL_ACTIONS = len(DEVIATIONS)


def tactical_state(squad: SquadRecord, members: Sequence[Sighting], threats: Sequence[Sighting],
                   losses: float, killed: float, view: WorldView, game_time_ms: int) -> List[float]:
    """What one squad's fight looks like, as the tactical layer is allowed to see it.

    The arguments are exactly what the script layer has in hand when it chooses a departure, which is deliberate: a learnt layer that saw more than the script would not be a replacement for it but a different layer, and the comparison the design rests on — does the learnt one beat the script over many episodes — would not be between two things doing the same job.
    """
    contract = squad.contract
    budget = contract.cost_budget if contract is not None else 0.0
    target = view.region(contract.target_region) if contract is not None else None

    live = [m for m in members if m.unit.max_health > 0]
    mean_health = sum(m.unit.health / m.unit.max_health for m in live) / len(live) if live else 0.0
    hit = sum(1 for m in members if m.unit.since_hit_ms <= RECENT_HIT_MS)
    squad_value = sum(m.value for m in members)
    threat_value = sum(t.value for t in threats)

    ahead = abeam = 0.0
    distance = 0.0
    if target is not None and members:
        distance = math.hypot(target.x - squad.x, target.y - squad.y)
        ahead, abeam = _bearing(squad, target, threats)

    reaches = [m.kind.range for m in members if m.kind is not None and m.kind.armed]
    enemy_reaches = [t.kind.range for t in threats if t.kind is not None and t.kind.armed]
    our_reach = min(reaches) if reaches else 0.0
    their_reach = max(enemy_reaches) if enemy_reaches else 0.0

    in_our_reach = sum(1 for t in threats if math.hypot(t.unit.x - squad.x, t.unit.y - squad.y) <= our_reach)
    in_their_reach = sum(1 for m in members if math.hypot(m.unit.x - squad.x, m.unit.y - squad.y) <= their_reach) if their_reach > 0 else 0

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
        # Redundant on a cosine and a sine, and kept as the same insurance every other slot carries against a float creeping past the range the suite enforces.
        _clip(ahead, -1.0, 1.0),
        _clip(abeam, -1.0, 1.0),
        _share(target.our_value, target.enemy_value) if target is not None else 0.5,
        _clip(target.enemy_value / (squad_value + 1.0)) if target is not None else 0.0,
        _clip(len(threats) / SQUAD_SIZE_SCALE),
        _share(threat_value, squad_value),
    ])
    features.extend(_role_shares(threats))
    features.extend(_role_shares(members))
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
        1.0,
    ])
    return [_finite(value) for value in features]


def _bearing(squad: SquadRecord, target: Optional[RegionState],
             threats: Sequence[Sighting]) -> Tuple[float, float]:
    """Where the errand points, measured against where the fight is rather than against the map's compass: the cosine and the sine of the angle from the direction of what is shooting at the squad to the direction of the region it was sent to.

    Both are read from the squad's own centre and both are products of two vectors that live on the board, so a board turned, moved or turned end for end gives the same pair. The absolute direction this replaces describes the same situation in the map's frame, so a network reading it answers one fight two ways depending on which way round the board happens to be numbered; and where one process drives both sides of a board laid out as a point reflection, the two sides read exactly opposite directions for congruent situations and the layer stops being one layer.

    The lateral term is kept signed rather than folded to its magnitude. A point reflection is a half turn and so preserves which hand is which, which means the sign survives the mirror and carries something: it says which way round the target lies from the fight, and a squad that can go round one way and not the other is in a different position from one that cannot.

    Both are nought where there is nothing shooting or nowhere to be sent, because the angle between a vector and nothing is not a number. Nothing is lost by that: the vector already says which case it is, since the flag for being engaged is nought exactly when there are no threats. Two consequences are behaviour and not tidying, and are written down here rather than found later. A squad whose threats surround it, so that their centre falls on its own centre, reads the same nought pair while still reading as engaged. And a squad marching with nothing shooting at it now carries no direction at all, where before it carried one — which is the frame-dependent part and exactly what is being given up.
    """
    if target is None or not threats:
        return 0.0, 0.0
    tx, ty = target.x - squad.x, target.y - squad.y
    fx = sum(threat.unit.x for threat in threats) / len(threats) - squad.x
    fy = sum(threat.unit.y for threat in threats) / len(threats) - squad.y
    reach, fight = math.hypot(tx, ty), math.hypot(fx, fy)
    # A world unit is far below anything either quantity means, so a separation under one is standing on the spot and its direction is noise rather than a bearing.
    if reach <= 1.0 or fight <= 1.0:
        return 0.0, 0.0
    tx, ty, fx, fy = tx / reach, ty / reach, fx / fight, fy / fight
    return fx * tx + fy * ty, fx * ty - fy * tx


def _role_shares(sightings: Sequence[Sighting]) -> List[float]:
    total = sum(s.value for s in sightings)
    if total <= 0:
        return [0.0] * len(ROLES)
    by_role: Dict[Role, float] = {}
    for sighting in sightings:
        by_role[sighting.role] = by_role.get(sighting.role, 0.0) + sighting.value
    return [by_role.get(role, 0.0) / total for role in ROLES]


# ---- the operational cut ---------------------------------------------------------------

GLOBAL_FEATURES: Tuple[str, ...] = (
    "credits", "income", "supply", "under_construction",
    *tuple(f"posture_{posture.name.lower()}" for posture in POSTURES),
    "offensive", "allowance", "military_edge", "region_edge", "elapsed", "squads", "bias",
)

REGION_FEATURES: Tuple[str, ...] = (
    "valid", "resources", "held_by_us", "held_by_enemy", "force_edge",
    "enemy_present", "distance", "seen_recently", "priority", "spawn",
)

SQUAD_FEATURES: Tuple[str, ...] = (
    "valid",
    *tuple(f"doctrine_{doctrine.name.lower()}" for doctrine in DOCTRINE_LIST),
    "weight", "health", "spread", "distance",
    *tuple(f"status_{status.name.lower()}" for status in STATUSES),
    "held", "taskable", "budget_spent", "mission_age",
)

GLOBAL_SIZE = len(GLOBAL_FEATURES)
REGION_SIZE = len(REGION_FEATURES)
SQUAD_SIZE = len(SQUAD_FEATURES)
OPERATIONAL_SIZE = GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + SQUAD_SLOTS * SQUAD_SIZE

#: What the operational state is made of, as one list, which is the form a set of parameters and a teacher file are stamped with so that neither can be read back under a feature list it was not written under.
#:
#: The three blocks are named once each rather than expanded over their slots, with the slot counts alongside. Everything the stamp has to catch changes this list — a renaming, a reordering, an addition or a removal in any block, and a change to how many slots a block has — while the expansion would be four hundred and thirty-odd names saying the same thing in a file that is read by a machine and printed to a person.
OPERATIONAL_FEATURES: Tuple[str, ...] = (
    *(f"global.{name}" for name in GLOBAL_FEATURES),
    *(f"region.{name}" for name in REGION_FEATURES),
    *(f"squad.{name}" for name in SQUAD_FEATURES),
    f"slots.{REGION_SLOTS}.{SQUAD_SLOTS}",
)

#: One decision is a region and a task, which is the pair the contract carries and the pair the script layer picks. Kept factorised rather than flattened into 144 because the two are chosen for different reasons — where is worth going, and what to do when you arrive — and because a mask over a product space is far sparser than the product of two masks.
OPERATIONAL_REGIONS = REGION_SLOTS
OPERATIONAL_TASKS = len(TASKS)

#: How recently the enemy must have been seen in a region for the feature to read as contact rather than as memory.
CONTACT_WINDOW_MS = 60000


def operational_state(view: WorldView, orders, squads: Sequence[SquadRecord],
                      game_time_ms: int, spawns: Sequence[int] = ()) -> List[float]:
    """The whole board as the operational layer sees it: aggregates, twenty-four region slots and eight squad slots, always in that order and always that long.

    Starting positions are passed in rather than read off the observation because the wire's region row does not carry the flag: it comes from the map file, which is read in this process when the region table is built. Which regions are starting positions is the difference between a place with resources on it and the place the enemy came from, and no other feature says it.
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
    by_slot = {region.id: region for region in view.regions}
    starts = frozenset(spawns)
    for slot in range(REGION_SLOTS):
        state.extend(_region_row(by_slot.get(slot), priorities, ours, game_time_ms, starts))

    by_id = {squad.id: squad for squad in squads}
    for slot in range(SQUAD_SLOTS):
        state.extend(_squad_row(by_id.get(slot), ours, game_time_ms, view))

    return [_finite(value) for value in state]


def _region_row(region: Optional[RegionState], priorities: Dict[int, float], ours: float,
                game_time_ms: int, spawns: frozenset) -> List[float]:
    if region is None:
        return [0.0] * REGION_SIZE
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
    ]


def _squad_row(squad: Optional[SquadRecord], ours: float, game_time_ms: int,
               view: WorldView) -> List[float]:
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
