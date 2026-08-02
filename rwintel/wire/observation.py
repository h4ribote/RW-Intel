"""The observation frame.

One frame per tactical period. The blocks a frame carries depend on which control periods fall on it, so the operational and strategic layers do not pay for the tactical rate; which blocks are present is stated in a bitmask rather than inferred from the period, so a reader never has to know the schedule.

Everything a layer sees is the strength under its own command: the unit count is what is left after another commander's holdings are taken out, and so is every region's own value. Reading the engine's own aggregate straight would have the command chain planning with units it cannot move. The cumulative kills and losses are not corrected, and cannot be: the engine keeps them per player, and nothing in them says which commander held the unit at the time.

Regions and squads travel as a fixed number of rows with a `valid` byte rather than as a packed list, because a slot number that means the same thing from frame to frame is what lets a policy carry state across decisions; a packed list would renumber every row whenever a squad dies or a region is added. The cost is a constant sized block, which is small at 24 and 8 slots, and the decoder drops the masked rows so nothing above this module ever sees them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List

#: Bits in the observation's block mask.
BLOCK_REGIONS = 1 << 0
BLOCK_SQUADS = 1 << 1
BLOCK_UNITS = 1 << 2
BLOCK_EVENTS = 1 << 3

#: Region slots the observation and the action agree on, per the interface document.
REGION_SLOTS = 24
#: Squad slots, which is also the cap the organisation layer has to keep to.
SQUAD_SLOTS = 8

#: A unit's order kind when it has no order at all, which is what makes it free to be given one.
NO_ORDER = 255

#: The value of a unit row's `built` once the unit is finished. The engine reports the progress of a build as a byte and the game's own standing counts a unit only at this value.
BUILT = 255
#: A unit's squad when it belongs to none, and a unit's attack target when it has none.
NO_SQUAD = 0xFFFF
NO_TARGET = 0

_COMMON = struct.Struct("<IIIHBBffHHHHHHH2x")
_REGION = struct.Struct("<BBBBffffIf")
_SQUAD = struct.Struct("<BHBB3xfffffBBBBffIIf")
_UNIT = struct.Struct("<IHHffffIHBBBBB5x")
_EVENT = struct.Struct("<BBHIHHf")
_COUNT = struct.Struct("<H")


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


@dataclass
class UnitState:
    id: int
    squad: int
    type_index: int
    x: float
    y: float
    health: float
    max_health: float
    #: How far along a build is, as the engine's own byte: BUILT is finished. The game's own scoring counts a unit's price only once it is finished, so anything comparing itself with that score has to read this.
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


@dataclass
class EventState:
    """Something that happened between two frames and would be invisible to a policy that only sees the current state.

    Kinds: 1 a unit completed, 2 a unit was lost, 3 a squad fell below its doctrine's strength. Which of the other fields carry anything depends on the kind.
    """

    kind: int
    squad: int
    unit: int
    type_index: int
    value: float


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
            observation.squads.append(SquadState(
                id=values[1], commander=values[2], units=values[3], value=values[4],
                formed_value=values[5], x=values[6], y=values[7], spread=values[8],
                task_type=values[9], stance=values[10], target_region=values[11],
                status=values[12], cost_budget=values[13], budget_share=values[14],
                deadline_ms=values[15], issued_at_ms=values[16], losses=values[17],
            ))

    if blocks & BLOCK_UNITS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            values = _UNIT.unpack_from(body, offset)
            offset += _UNIT.size
            observation.unit_states.append(UnitState(
                id=values[0], squad=values[1], type_index=values[2], x=values[3], y=values[4],
                health=values[5], max_health=values[6], target=values[7], since_hit_ms=values[8],
                built=values[9], order=values[10], stance=values[11], hostile=values[12], queued=values[13],
            ))

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

    return observation
