"""The observation frame.

One frame per tactical period. The blocks a frame carries depend on which control periods fall on it, so the operational and strategic layers do not pay for the tactical rate; which blocks are present is stated in a bitmask rather than inferred from the period, so a reader never has to know the schedule.

Everything a layer sees is the strength under its own command: the unit count is what is left after another commander's holdings are taken out, and so is every region's own value. Reading the engine's own aggregate straight would have the command chain planning with units it cannot move. The cumulative kills and losses are not corrected, and cannot be: the engine keeps them per player, and nothing in them says which commander held the unit at the time.

Regions and squads travel as a fixed number of rows with a `valid` byte rather than as a packed list, because a slot number that means the same thing from frame to frame is what lets a policy carry state across decisions; a packed list would renumber every row whenever a squad dies or a region is added. The cost is a constant sized block, which is small at 24 and 8 slots, and the decoder drops the masked rows so nothing above this module ever sees them.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

#: Bits in the observation's block mask.
BLOCK_REGIONS = 1 << 0
BLOCK_SQUADS = 1 << 1
BLOCK_UNITS = 1 << 2
BLOCK_EVENTS = 1 << 3
BLOCK_MENUS = 1 << 4
BLOCK_LIFTS = 1 << 5
#: On every observation: the number of the observation whose answer was applied at the head of this step.
BLOCK_TIMING = 1 << 6
#: Written last, on operational observations of an episode that asked for them: the built-in AI players' orders since the last one.
BLOCK_AI_ORDERS = 1 << 7

#: An AI order's target when it names no unit, and the flag bit of an order appended to those held.
AI_ORDER_NO_UNIT = 0xFFFFFFFF
AI_ORDER_APPEND = 1

#: Region slots the observation and the action agree on, per the interface document.
REGION_SLOTS = 24
#: Squad slots, which is also the cap the organisation layer has to keep to.
SQUAD_SLOTS = 8

#: A unit's order kind when it has no order at all, which is what makes it free to be given one.
NO_ORDER = 255
#: A unit's squad when it belongs to none, and a unit's attack target when it has none.
NO_SQUAD = 0xFFFF
NO_TARGET = 0

#: The movement types a squad's passage byte names, by number; nought is none, or members sharing no one narrowest type.
PASSAGE_CLASSES = ("", "LAND", "OVER_CLIFF", "HOVER", "WATER", "OVER_CLIFF_WATER", "AIR")
#: A squad's lift when it is the cargo of none.
NO_LIFT = 0xFFFF

_COMMON = struct.Struct("<IIIHBBffHHHHHHH2x")
_REGION = struct.Struct("<BBBBffffIf")
_SQUAD = struct.Struct("<BHBBBBBfffffBBBBffIIfIH2x")
_UNIT = struct.Struct("<IHHffffIHBBBBBBHBxI")
_LIFT = struct.Struct("<HBBBBBxfIHBx")
_EVENT = struct.Struct("<BBHIHHf")
_COUNT = struct.Struct("<H")
_MENU = struct.Struct("<IB")
_TIMING = struct.Struct("<i")
_AI_ORDER = struct.Struct("<IBBBxffIH")


class LiftPhase(enum.IntEnum):
    APPROACH = 0
    LOADING = 1
    CARRYING = 2
    UNLOADING = 3
    DONE = 4
    FAILED = 5


class LiftFailure(enum.IntEnum):
    """Why a lift failed. A sunk transport takes everyone aboard down with it."""

    NONE = 0
    SUNK = 1
    UNREACHABLE_PICKUP = 2
    UNREACHABLE_DROP = 3
    REFUSED = 4
    EXPIRED = 5
    CANCELLED = 6
    CARGO_LOST = 7


#: Event kinds.
EVENT_UNIT_COMPLETED = 1
EVENT_UNIT_LOST = 2
EVENT_SQUAD_DEPLETED = 3
EVENT_BOARDED = 4
EVENT_DISEMBARKED = 5
EVENT_LIFT_DONE = 6
EVENT_LIFT_FAILED = 7


@dataclass
class RegionState:
    #: The slot, which is the map's own region numbering. The egocentric order a layer wants is applied where the table is handed to that layer, not on the wire, because home is not known until the first observation arrives.
    id: int
    resources: int
    held_by_us: int
    held_by_enemy: int
    x: float
    y: float
    our_value: float
    enemy_value: float
    enemy_seen_at_ms: int
    distance_from_home: float


@dataclass
class SquadState:
    id: int
    #: Bit mask: 1 = operations held by a human, 2 = tactics held by a human.
    commander: int
    units: int
    value: float
    formed_value: float
    x: float
    y: float
    #: Standard deviation of member distance from the centre, which is how concentrated the squad currently is.
    spread: float
    task_type: int
    stance: int
    #: The region the contract's target lies in: the target itself when it names a region, and the region nearest the squad or unit it names otherwise.
    target_region: int
    status: int
    #: Credits, as a float, so it compares with the value of the losses without a conversion.
    cost_budget: float
    #: The squad's budget over our total commanded value, so a policy can weigh a squad without also being told the total.
    budget_share: float
    deadline_ms: int
    #: When the current contract was issued, which is what turns deadline_ms into a remaining time.
    issued_at_ms: int
    losses: float
    #: What the contract's target names (`action.TargetKind`) and the region, squad or unit it is.
    target_kind: int = 0
    target: int = 0
    #: Members aboard a transport.
    aboard: int = 0
    #: The narrowest movement type among the members, as an index into `PASSAGE_CLASSES`.
    passage: int = 0
    #: The lift the squad is the cargo of, `NO_LIFT` for none.
    lift: int = NO_LIFT


@dataclass
class UnitState:
    id: int
    squad: int
    type_index: int
    x: float
    y: float
    health: float
    max_health: float
    built: int
    #: Ordinal of the current order's kind, 255 when the unit has never had one, so zero is a real order and not "idle".
    order: int
    #: How many orders are queued. This, not the kind, is what says a unit is free: the engine leaves the kind of the last order in place after carrying it out.
    queued: int
    #: The unit this one is shooting at, by id, 0 for nothing.
    target: int
    stance: int
    hostile: int
    #: Game time since this unit was last hit, clamped, which is the cheap trigger for attention.
    since_hit_ms: int
    #: The tier one of our finished buildings has been raised to, and the price of raising it to the next, nought when no further tier is offered. Both are nought for everything that is not one of our finished buildings.
    level: int = 0
    upgrade_price: int = 0
    #: For a transport, how many slots it has filled.
    aboard: int = 0
    #: The transport this unit is aboard, by id, 0 for none. A unit aboard stands where its transport is.
    carrier: int = 0


@dataclass
class LiftState:
    """One lift under way, or one that ended since the last frame, which is reported once."""

    lift: int
    phase: int
    #: Why it failed (`LiftFailure`), nought otherwise.
    reason: int
    loaded: int
    #: How many passengers were assigned a place: the cargo that fits the transports given.
    expected: int
    transports: int
    #: The transports' combined health.
    health: float
    #: Game time the cargo is expected to be down by.
    eta_ms: int
    #: The cargo squad, `NO_SQUAD` for a list of units.
    squad: int
    drop_region: int


@dataclass
class EventState:
    """Something that happened between two frames and would be invisible to a policy that only sees the current state.

    Kinds: 1 a unit completed, 2 a unit was lost, 3 a squad fell below its doctrine's strength, 4 a unit went aboard a transport, 5 a unit came off one, 6 a lift set its cargo down, 7 a lift failed. For the two lift events the unit field carries the lift's id and the value its failure reason. Which of the other fields carry anything depends on the kind.
    """

    kind: int
    squad: int
    unit: int
    type_index: int
    value: float


class AiOrderKind(enum.IntEnum):
    """What a built-in AI player's order does. OTHER_MOVEMENT is a patrol, guard or follow order."""

    MOVE = 0
    ATTACK_MOVE = 1
    ATTACK = 2
    OTHER_MOVEMENT = 3
    LOAD_INTO = 4
    LOAD_UP = 5
    UNLOAD = 6
    CANCEL_UNLOAD = 7
    STOP = 8


@dataclass(frozen=True)
class AiOrder:
    """One order a built-in AI player gave, as its command was issued."""

    time_ms: int
    #: The issuing player's slot.
    issuer: int
    kind: AiOrderKind
    #: Whether the order goes after those the units already hold rather than replacing them.
    append: bool
    #: The point the order names, NaN when it names none.
    x: float
    y: float
    #: The unit the order is aimed at (the transport for LOAD_INTO, the passenger for LOAD_UP), or AI_ORDER_NO_UNIT.
    target: int
    units: Tuple[int, ...]


@dataclass
class Observation:
    frame: int
    game_time_ms: int
    episode: int
    blocks: int
    slot: int
    credits: float
    income: float
    units: int
    unit_cap: int
    under_construction: int
    killed_units: int
    killed_buildings: int
    lost_units: int
    lost_buildings: int
    regions: List[RegionState] = field(default_factory=list)
    squads: List[SquadState] = field(default_factory=list)
    unit_states: List[UnitState] = field(default_factory=list)
    events: List[EventState] = field(default_factory=list)
    #: What each of our finished buildings and builders offers to produce or place at its current tier, by unit id, as type indices. Present only on frames carrying the menu block; a unit that offers nothing is absent.
    menus: Dict[int, List[int]] = field(default_factory=dict)
    lifts: List[LiftState] = field(default_factory=list)
    #: The number (`frame`) of the observation whose answer the agent applied at the head of the step that produced this one; -1 when none was applied, or when the frame carries no timing block.
    answered: int = -1
    #: The built-in AI players' orders since the last operational observation; empty when the frame carries no AI orders block.
    ai_orders: List[AiOrder] = field(default_factory=list)


def decode_observation(body: bytes) -> Observation:
    offset = 0
    (frame, game_time_ms, episode, blocks, slot, _pad, credits, income,
     units, unit_cap, under_construction,
     killed_units, killed_buildings, lost_units, lost_buildings) = _COMMON.unpack_from(body, offset)
    offset += _COMMON.size

    observation = Observation(
        frame=frame, game_time_ms=game_time_ms, episode=episode, blocks=blocks, slot=slot,
        credits=credits, income=income, units=units, unit_cap=unit_cap,
        under_construction=under_construction, killed_units=killed_units,
        killed_buildings=killed_buildings, lost_units=lost_units, lost_buildings=lost_buildings,
    )

    if blocks & BLOCK_REGIONS:
        # The count is the number of rows that follow, which for a fixed length block is the slot count. Reading it rather than assuming it keeps the two halves honest with each other about one live field.
        rows, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for slot_index in range(rows):
            values = _REGION.unpack_from(body, offset)
            offset += _REGION.size
            if not values[0]:
                continue
            observation.regions.append(RegionState(
                id=slot_index, resources=values[1], held_by_us=values[2], held_by_enemy=values[3],
                x=values[4], y=values[5], our_value=values[6], enemy_value=values[7],
                enemy_seen_at_ms=values[8], distance_from_home=values[9],
            ))

    if blocks & BLOCK_SQUADS:
        rows, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(rows):
            values = _SQUAD.unpack_from(body, offset)
            offset += _SQUAD.size
            if not values[0]:
                continue
            (_, squad_id, commander, units, aboard, passage, target_region, value, formed_value, x, y, spread,
             task_type, stance, target_kind, status, cost_budget, budget_share, deadline_ms, issued_at_ms, losses,
             target, lift) = values
            observation.squads.append(SquadState(
                id=squad_id, commander=commander, units=units, value=value,
                formed_value=formed_value, x=x, y=y, spread=spread,
                task_type=task_type, stance=stance, target_region=target_region,
                status=status, cost_budget=cost_budget, budget_share=budget_share,
                deadline_ms=deadline_ms, issued_at_ms=issued_at_ms, losses=losses,
                target_kind=target_kind, target=target, aboard=aboard, passage=passage, lift=lift,
            ))

    if blocks & BLOCK_UNITS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        end = offset + count * _UNIT.size
        # The unit block is most of every frame, so its rows are unpacked in one pass and built positionally, in the dataclass's field order rather than the wire's.
        observation.unit_states = [
            UnitState(unit_id, squad, type_index, x, y, health, max_health, built, order, queued,
                      target, stance, hostile, since_hit_ms, level, upgrade_price, aboard, carrier)
            for (unit_id, squad, type_index, x, y, health, max_health, target, since_hit_ms,
                 built, order, stance, hostile, queued, level, upgrade_price, aboard, carrier)
            in _UNIT.iter_unpack(body[offset:end])]
        offset = end

    if blocks & BLOCK_LIFTS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            observation.lifts.append(LiftState(*_LIFT.unpack_from(body, offset)))
            offset += _LIFT.size

    if blocks & BLOCK_EVENTS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            values = _EVENT.unpack_from(body, offset)
            offset += _EVENT.size
            observation.events.append(EventState(
                kind=values[0], squad=values[2], unit=values[3], type_index=values[4],
                value=values[6],
            ))

    if blocks & BLOCK_MENUS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            unit_id, offered = _MENU.unpack_from(body, offset)
            offset += _MENU.size
            observation.menus[unit_id] = list(struct.unpack_from(f"<{offered}H", body, offset))
            offset += 2 * offered

    if blocks & BLOCK_TIMING:
        observation.answered, = _TIMING.unpack_from(body, offset)
        offset += _TIMING.size

    if blocks & BLOCK_AI_ORDERS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            time_ms, issuer, kind, flags, x, y, target, units = _AI_ORDER.unpack_from(body, offset)
            offset += _AI_ORDER.size
            ids = struct.unpack_from(f"<{units}I", body, offset)
            offset += 4 * units
            observation.ai_orders.append(AiOrder(time_ms=time_ms, issuer=issuer, kind=AiOrderKind(kind),
                                                 append=bool(flags & AI_ORDER_APPEND), x=x, y=y, target=target,
                                                 units=ids))

    return observation
