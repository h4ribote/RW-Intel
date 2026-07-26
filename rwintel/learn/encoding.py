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
from ..control.policy.strategy import INCOME_PLATEAU_FLOOR
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
    "target_distance", "target_ahead_of_home", "target_abeam_of_home", "target_ours", "target_theirs",
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
        ahead, abeam = _bearing(squad, target, view.home_point)

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
             home: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    """Where the errand points, measured against the way the squad has come rather than against the map's compass: the cosine and the sine of the angle from the direction leading out of our own home through the squad to the direction of the region it was sent to.

    Both are read from the squad's own centre and both are products of two vectors that live on the board, so a board turned, moved or turned end for end gives the same pair. The absolute direction this replaces describes the same situation in the map's frame, so a network reading it answers one fight two ways depending on which way round the board happens to be numbered; and where one process drives both sides of a board laid out as a point reflection, the two sides read exactly opposite directions for congruent situations and the layer stops being one layer. A point reflection carries our home onto the other side's, so the outward direction reflects with everything else and the pair survives the mirror — provided the anchor each side reads is its OWN, which is what `WorldView.home_point` is for.

    The ahead term says whether the errand leads further out or back: a squad sent past where it stands, away from home, reads toward +1 and one recalled toward its own ground reads toward -1. That is the sense a policy needs to tell an advance from a withdrawal, and it is the same sense on both sides of a mirrored board.

    The lateral term is kept signed rather than folded to its magnitude. A point reflection is a half turn and so preserves which hand is which, which means the sign survives the mirror and carries something: it says which way round the target lies from the way out, and a squad that can go round one way and not the other is in a different position from one that cannot.

    Reading the angle from home rather than from the centre of what is shooting is what makes the pair defined whether or not anything is shooting. A bearing read from the threats leaves a squad with nothing firing at it — a squad marching on a contest is exactly that — carrying no direction at all, and the two readings taken around that form are a layer refitted under it sitting 0.0398 below the handwritten ladder with its interval off nought, against 0.0042 below and holding nought for the frame-carrying form it replaced. Whether the encoding made that difference is not something those two runs can say, since their intervals overlap; what is not in doubt is that home does not come and go with the fighting, and a squad marching on a contest has a direction here.

    The pair is nought in three cases, and they are behaviour rather than tidying. There is no errand, so there is nowhere to point. There is no anchor — no region table has arrived, or in a match nothing has been built yet, so where home is is not yet known. Or the squad is standing on home, where the way out is not a direction; a world unit is far below anything either quantity means, so a separation under one is standing on the spot and its direction is noise rather than a bearing.
    """
    if target is None or home is None:
        return 0.0, 0.0
    tx, ty = target.x - squad.x, target.y - squad.y
    ox, oy = squad.x - home[0], squad.y - home[1]
    reach, out = math.hypot(tx, ty), math.hypot(ox, oy)
    if reach <= 1.0 or out <= 1.0:
        return 0.0, 0.0
    tx, ty, ox, oy = tx / reach, ty / reach, ox / out, oy / out
    return ox * tx + oy * ty, ox * ty - oy * tx


def _role_shares(sightings: Sequence[Sighting]) -> List[float]:
    total = sum(s.value for s in sightings)
    if total <= 0:
        return [0.0] * len(ROLES)
    by_role: Dict[Role, float] = {}
    for sighting in sightings:
        by_role[sighting.role] = by_role.get(sighting.role, 0.0) + sighting.value
    return [by_role.get(role, 0.0) / total for role in ROLES]


# ---- the strategic cut -----------------------------------------------------------------

#: What the strategic features are, in order. The layer decides one thing every ten seconds — which posture the match is in — and everything else it emits is that posture read through a table, so this is the cut for that one choice.
#:
#: The materials are what the script layer is handed when it makes the same choice: the front report, the region table, the elapsed match time, what has been in contact, and the two histories the layer keeps for itself. The histories are here because the script's own rule is written on them — "income has levelled off" and "ground keeps being taken" are not properties of a number but of a run of them — and a learnt layer denied them would be answering a strictly harder question than the rule it is measured against.
#:
#: Nothing here is a raw credit total or a raw world coordinate, by the same rule the other two cuts follow, and nothing carries the map's frame: every quantity is a share, a ratio against a stated scale, or a flag. There are no directions in this cut at all, so the mirror question the tactical and operational cuts had to answer does not arise here.
STRATEGIC_FEATURES: Tuple[str, ...] = (
    "credits", "income", "income_growth", "income_started",
    "supply", "under_construction",
    "military_edge", "region_edge", "held", "enemy_held", "contested",
    "lost_regions", "losing_periods", "enemy_bases", "driven_back",
    "elapsed",
    *tuple(f"posture_{posture.name.lower()}" for posture in POSTURES),
    *tuple(f"contact_{role.name.lower()}" for role in ROLES),
    "bias",
)

STRATEGIC_SIZE = len(STRATEGIC_FEATURES)

#: The postures, which are the whole strategic action space whether the rule or a network is choosing. All five are always legal and there is no mask: the script's rule never selects TECH, and that is a property of the rule rather than of the game — a mask that forbade it would be the rule written again in the mask's clothing, which is what the tactical space refuses masks for.
STRATEGIC_ACTIONS = len(POSTURES)

#: Fresh losses over the window a strategic decision looks back on, quoted against this. The rule reacts at two, so a couple of periods of ground going reads near the top of the range.
LOSS_SCALE = 3.0

#: Enemy footholds quoted against this. A skirmish map gives a side one to begin with and a few more as it expands.
BASE_SCALE = 4.0


def strategic_state(report, regions: Sequence[RegionState], game_time_ms: int,
                    income_history: Sequence[float] = (), loss_history: Sequence[int] = (),
                    most_enemy_bases: int = 0, contact: Optional[Dict[Role, float]] = None,
                    posture: Posture = Posture.EXPAND) -> List[float]:
    """The whole match as an aggregate, which is the only abstraction this layer is given.

    The two histories arrive as they stand rather than as the rule's verdict on them. `income_growth` is the growth across the whole window as a share of its oldest sample, which is the quantity the rule thresholds, and it is handed over as a number so that a policy can decide for itself where the threshold is — where the rule can only answer the question it was written with. The same goes for the losses: what is here is how many of the last few periods saw ground go, not whether that is enough to defend.

    `driven_back` is the one feature that is a verdict, and it has to be: the rule's condition is that the enemy held more bases at some point and holds one now, so a policy given only the present count could not tell an opponent driven back to their last base from one that started with a single base and has not been touched. It is the peak that carries that, and the peak is meaningless as a bare count.
    """
    income = float(getattr(report, "income", 0.0))
    growth = 0.0
    if len(income_history) >= 2 and income_history[0] > 0.0:
        growth = (income_history[-1] - income_history[0]) / max(income_history[0], 1.0)
    ours = float(getattr(report, "military_value", 0.0))
    theirs = float(getattr(report, "enemy_value", 0.0))
    held = float(getattr(report, "held", 0))
    enemy_held = float(getattr(report, "enemy_held", 0))
    places = max(1, len(regions))
    contested = sum(1 for region in regions if region.our_value > 0.0 and region.enemy_value > 0.0)
    bases = float(getattr(report, "enemy_bases", 0))

    features: List[float] = [
        _clip(float(getattr(report, "credits", 0.0)) / CREDIT_SCALE),
        _clip(income / INCOME_SCALE),
        # Signed, because income falling is a different match from income levelling off, and the rule cannot tell them apart.
        _clip(growth, -1.0, 1.0),
        # Whether there is an economy to read a plateau off at all. The opening is flat and near nothing, and a policy without this would have to learn that a flat nought means "not started" while a flat forty means "levelled off".
        1.0 if income >= INCOME_PLATEAU_FLOOR else 0.0,
        _clip(float(getattr(report, "units", 0)) / getattr(report, "unit_cap", 0)) if getattr(report, "unit_cap", 0) else 0.0,
        _clip(float(getattr(report, "under_construction", 0)) / 8.0),
        _share(ours, theirs),
        _share(held, enemy_held),
        _clip(held / places),
        _clip(enemy_held / places),
        _clip(contested / places),
        _clip(float(getattr(report, "lost_regions", 0)) / LOSS_SCALE),
        _clip(sum(1 for lost in loss_history if lost > 0) / LOSS_SCALE),
        _clip(bases / BASE_SCALE),
        1.0 if bases == 1 and most_enemy_bases > 1 else 0.0,
        _clip(game_time_ms / MATCH_SCALE),
    ]
    features.extend(_one_hot(posture, POSTURES))
    features.extend(_contact_shares(contact))
    features.append(1.0)
    return [_finite(value) for value in features]


def _contact_shares(contact: Optional[Dict[Role, float]]) -> List[float]:
    """What has been run into, by role, as shares of the worth contacted. All nought where nothing has been seen, which is the opening of every match and is not the same statement as an even mix."""
    if not contact:
        return [0.0] * len(ROLES)
    total = sum(contact.values())
    if total <= 0.0:
        return [0.0] * len(ROLES)
    return [contact.get(role, 0.0) / total for role in ROLES]


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
#:
#: The last entry names what a slot IS as well as how many there are, because the widths do not move when the layout does. Ordering the rows from this side's own home rather than by the map's numbering leaves every width where it was and changes what every row of the block means, which is precisely the silent failure the stamp exists to refuse.
OPERATIONAL_FEATURES: Tuple[str, ...] = (
    *(f"global.{name}" for name in GLOBAL_FEATURES),
    *(f"region.{name}" for name in REGION_FEATURES),
    *(f"squad.{name}" for name in SQUAD_FEATURES),
    f"slots.from_home.{REGION_SLOTS}.own_first.{SQUAD_SLOTS}",
)

#: One decision is a region and a task, which is the pair the contract carries and the pair the script layer picks. Kept factorised rather than flattened into 144 because the two are chosen for different reasons — where is worth going, and what to do when you arrive — and because a mask over a product space is far sparser than the product of two masks.
OPERATIONAL_REGIONS = REGION_SLOTS
OPERATIONAL_TASKS = len(TASKS)

#: How recently the enemy must have been seen in a region for the feature to read as contact rather than as memory.
CONTACT_WINDOW_MS = 60000


def operational_slots(view: WorldView) -> List[RegionState]:
    """The regions in the order the operational cut lays them out: outward from this side's own home, nearest first, cut to the number of slots there are.

    Egocentric and not the map's own numbering, and that is the whole of what makes the block readable from either side of a mirrored board. The map numbers a region once for the whole board, so congruent ground sits at a different offset for the two sides and a network reading the block learns the same board twice in two numberings — the frame the tactical cut was freed of, in the layer above it. Ordered from home, the first slot is always this side's own ground and the last always the far side, whichever side is reading and whatever the map.

    The order is the view's own `from_home`, so the layers and the encoder cannot disagree about what a slot means. It breaks an exact tie in distance by the map's number, which is the one place the frame survives: two of a side's own regions at the very same distance from home can be numbered the other way round for the mirror side. What sits in the two rows is then congruent — same distance, and on a mirrored board the same resources and the same standing — so the state a network reads is unchanged by the swap; what can differ is which of two congruent places a chosen slot names.
    """
    return view.from_home()[:REGION_SLOTS]


def squad_slots(squads: Sequence[SquadRecord], base: int = 0) -> Dict[int, int]:
    """Which row each of this side's squads is written into: its own number less the first number this side was ever given.

    A match hands squads out from nought, so the offset is nought and a squad's row is its own number, which is what keeps a row meaning the same thing from one period to the next. A constructed arena is one process driving two sides out of one numbering — this side's squads are the first few numbers and the other side's the next few — so without the offset the two sides' congruent squads would be written into different rows and named to the network by different one-hot slots. The offset is fixed once per side rather than recomputed, so a squad dying does not renumber the ones above it.
    """
    return {squad.id: squad.id - base for squad in squads}


def operational_state(view: WorldView, orders, squads: Sequence[SquadRecord],
                      game_time_ms: int, spawns: Sequence[int] = (), base: int = 0) -> List[float]:
    """The whole board as the operational layer sees it: aggregates, twenty-four region slots and eight squad slots, always in that order and always that long.

    The region slots run outward from this side's own home and the squad slots from this side's own first squad, so that one board read from either side of a mirror puts congruent things in the same rows. See `operational_slots` and `squad_slots`.

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
    ordered = operational_slots(view)
    starts = frozenset(spawns)
    for slot in range(REGION_SLOTS):
        region = ordered[slot] if slot < len(ordered) else None
        state.extend(_region_row(region, priorities, ours, game_time_ms, starts))

    slots = squad_slots(squads, base)
    by_slot = {slots[squad.id]: squad for squad in squads}
    for slot in range(SQUAD_SLOTS):
        state.extend(_squad_row(by_slot.get(slot), ours, game_time_ms, view))

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
    """Which region slots exist on this map. A slot with no region behind it is not a target a policy may choose, and masking is how that is said rather than hoping the policy learns it.

    The slots are the egocentric ones the state is written in, so the live ones are the first however many there are and a policy choosing slot k is choosing the k-th region out from its own home. Whoever decodes a choice must read the same order back — `operational_slots` is the one place it is defined."""
    return [1.0 if slot < len(operational_slots(view)) else 0.0 for slot in range(REGION_SLOTS)]


def task_mask(doctrine: Doctrine) -> List[float]:
    """Which tasks a squad of this doctrine may be given, read off the same doctrine table the script layer reads. Engineers have no tasks at all, which is how the design keeps a contract from landing on a builder in the middle of a placement and turning a half raised building into a total loss."""
    allowed = set(int(task) for task in DOCTRINES[doctrine].tasks)
    return [1.0 if int(task) in allowed else 0.0 for task in TASKS]


def squad_mask(squads: Sequence[SquadRecord], base: int = 0) -> List[float]:
    """Which squad slots hold a squad this layer may task: one that exists, has anyone left in it, has a doctrine with tasks, and has not been taken over by somebody else.

    Read in the same rows the state is written in, so the offset that puts this side's first squad in the first row has to be handed in here too."""
    slots = squad_slots(squads, base)
    by_slot = {slots[squad.id]: squad for squad in squads}
    mask: List[float] = []
    for slot in range(SQUAD_SLOTS):
        squad = by_slot.get(slot)
        usable = (squad is not None and bool(squad.members) and squad.ours_to_task
                  and bool(DOCTRINES[squad.doctrine].tasks))
        mask.append(1.0 if usable else 0.0)
    return mask
