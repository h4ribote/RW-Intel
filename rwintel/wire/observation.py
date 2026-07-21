"""The observation frame.

One frame per tactical period. The blocks a frame carries depend on which control periods fall on it, so the operational and strategic layers do not pay for the tactical rate; which blocks are present is stated in a bitmask rather than inferred from the period, so a reader never has to know the schedule.

Everything a layer sees is the strength under its own command: the per player aggregates are corrected by subtracting what another commander is holding. Reading `n.T` straight would have the command chain planning with units it cannot move.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List

#: Bits in the observation's block mask.
BLOCK_REGIONS = 1 << 0
BLOCK_SQUADS = 1 << 1
BLOCK_UNITS = 1 << 2

_COMMON = struct.Struct("<IIIHBBffHHHHHHH2x")
_REGION = struct.Struct("<BBBBffffIf")
_SQUAD = struct.Struct("<HBBffffBBHfII")
_UNIT = struct.Struct("<IHHffffBBHBBH")
_COUNT = struct.Struct("<H")


@dataclass
class RegionState:
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
    commander: int
    units: int
    value: float
    formed_value: float
    x: float
    y: float
    task_type: int
    status: int
    target_region: int
    losses: float
    cost_budget: int
    deadline_ms: int


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
    #: Length of the unit's order queue, so zero means it is free to be given something to do.
    orders: int
    #: The squad the unit is shooting at, or 0xFFFF for nothing and for a target in no squad.
    target: int
    stance: int
    hostile: int
    #: Game time since this unit was last hit, clamped, which is the cheap trigger for attention.
    since_hit_ms: int


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
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for index in range(count):
            values = _REGION.unpack_from(body, offset)
            offset += _REGION.size
            observation.regions.append(RegionState(
                id=index, resources=values[1], held_by_us=values[2], held_by_enemy=values[3],
                x=values[4], y=values[5], our_value=values[6], enemy_value=values[7],
                enemy_seen_at_ms=values[8], distance_from_home=values[9],
            ))

    if blocks & BLOCK_SQUADS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            values = _SQUAD.unpack_from(body, offset)
            offset += _SQUAD.size
            observation.squads.append(SquadState(
                id=values[0], commander=values[1], units=values[2], value=values[3],
                formed_value=values[4], x=values[5], y=values[6], task_type=values[7],
                status=values[8], target_region=values[9], losses=values[10],
                cost_budget=values[11], deadline_ms=values[12],
            ))

    if blocks & BLOCK_UNITS:
        count, = _COUNT.unpack_from(body, offset)
        offset += _COUNT.size
        for _ in range(count):
            values = _UNIT.unpack_from(body, offset)
            offset += _UNIT.size
            observation.unit_states.append(UnitState(
                id=values[0], squad=values[1], type_index=values[2], x=values[3], y=values[4],
                health=values[5], max_health=values[6], built=values[7], orders=values[8],
                target=values[9], stance=values[10], hostile=values[11],
                since_hit_ms=values[12],
            ))

    return observation
