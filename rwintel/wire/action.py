"""The action frame.

Four sections, each carrying the decisions of one layer, and each present only on the periods that layer runs on.

Squad membership travels with the actions rather than being decided in the game process. Squads are first class entities with a lifetime of their own, and the layer that manages that lifetime is a policy; the game side only needs to know which units belong to which squad so it can address them and describe them back.

A contract and a deviation are separate sections because they are decided on different clocks. The contract is what the operational layer commits to and is meant to stand until its deadline; the deviation is how the tactical layer carries it out right now, and it is re-decided an order of magnitude more often. Folding the deviation back into the contract row, as the first version did, meant the tactical layer could only speak by restating the whole contract, which loses the distinction between "the plan changed" and "the plan is being executed differently this second".

Both of those rows carry an override bit, which says the decision came from whoever holds the squad rather than from the command chain. The game side refuses a chain decision about a squad someone else has taken over — that is the whole of the ownership rule — and without a way to say "this is that someone" the taking over would silence the squad instead of transferring it. One byte separates the two cases, and it is the only thing on the wire that distinguishes a human at the intervention interface, or the script intruder that trains against it, from the layer whose job the decision ordinarily is.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import Dict, List

_SECTION = struct.Struct("<H")
_SQUAD_HEADER = struct.Struct("<HBBH2x")
#: A squad's owner as it travels: nought is the player this process is, and any other value is that player's slot plus one. Ordinary play only ever sends nought; a constructed engagement drives both sides from one process and needs to say which.
NO_OWNER = 0
_UNIT_ID = struct.Struct("<I")
_CONTRACT = struct.Struct("<HBBBB2xfII")
_DEVIATION = struct.Struct("<HBB")
_PRODUCTION = struct.Struct("<IHBBff")


class Task(enum.IntEnum):
    ATTACK = 0
    DEFEND = 1
    RAID = 2
    WITHDRAW = 3
    ESCORT = 4
    ENCIRCLE = 5


class Stance(enum.IntEnum):
    """Maps straight onto the engine's own engagement stance constants, in their declared order."""

    OUT_OF_RANGE = 0
    ONLY_IN_RANGE = 1
    RETURN_FIRE = 2
    HOLD_FIRE = 3
    GUARD_AREA = 4
    AGGRESSIVE = 5
    MIXED = 6


class Deviation(enum.IntEnum):
    """What the tactical layer does instead of the contract's default advance.

    The first five are the plain departures; a departure names the kind of move and rules on the game side settle where it points. The two after them are the same kinds with the choice a rule used to make handed to the layer instead: how far a withdrawal commits, and which enemy a concentration goes onto. They are appended rather than inserted so that a network trained on the five keeps the meaning of every value it already learnt, and so that the byte the wire carries needs no wider field.
    """

    HOLD = 0
    WITHDRAW = 1
    FOCUS = 2
    SPREAD = 3
    KITE = 4
    #: Withdraw the whole way out of the fight rather than the short step back WITHDRAW takes. The choice of how far, which a rule used to fix, is the layer's here: a squad that is merely repositioning wants the short step and one that is beaten wants the ground between it and the enemy.
    WITHDRAW_FAR = 5
    #: Concentrate on the longest-ranged enemy in reach rather than the weakest FOCUS picks. Which enemy to take out first, which a rule used to fix as the weakest, is the layer's here: the gun that reaches furthest is often worth more dead than the one nearest to dying.
    FOCUS_THREAT = 6


class Status(enum.IntEnum):
    ACTIVE = 0
    STALLED = 1
    LOSING = 2
    COMPLETE = 3
    EXPIRED = 4


class Commander(enum.IntFlag):
    """Which layers of a squad's command a human holds. A flag rather than a choice of two, because the design lets a human take the operational command of a squad while the tactical layer goes on fighting it, and that is the combination it calls the most useful."""

    MACHINE = 0
    OPERATIONS = 1
    TACTICS = 2


class ProductionKind(enum.IntEnum):
    #: A factory producing a unit, addressed by the producing building.
    UNIT = 0
    #: A builder placing a building, which needs a position.
    BUILDING = 1


@dataclass
class SquadAssignment:
    squad: int
    #: Which commander the squad answers to, sent with the membership so handing a squad to a human is one decision rather than a separate channel.
    commander: Commander = Commander.MACHINE
    units: List[int] = field(default_factory=list)
    #: Which player's orders move this squad. -1, the only value ordinary play uses, is the player this process is; a slot number is how a constructed engagement drives the other side of a fight from the same process, which the engine's sandbox flag permits.
    owner: int = -1


@dataclass
class Contract:
    squad: int
    task: Task = Task.DEFEND
    stance: Stance = Stance.AGGRESSIVE
    target_region: int = 0
    #: Credits, as a float, matching the squad block that reports it back.
    cost_budget: float = 0.0
    deadline_ms: int = 0
    #: Sent rather than stamped game side so a contract restated unchanged keeps its original clock instead of silently renewing its deadline.
    issued_at_ms: int = 0
    #: True when this comes from whoever holds the squad rather than from the operational layer, which is what makes it apply to a squad the layer is no longer allowed to task.
    override: bool = False


@dataclass
class SquadDeviation:
    squad: int
    deviation: Deviation = Deviation.HOLD
    #: True when this comes from whoever holds the squad's tactical command rather than from the tactical layer.
    override: bool = False


@dataclass
class Production:
    producer: int
    type_index: int
    kind: ProductionKind = ProductionKind.UNIT
    cancel: bool = False
    x: float = 0.0
    y: float = 0.0


@dataclass
class Action:
    """One period's decisions. Empty sections are legal and mean "nothing changed"."""

    squads: List[SquadAssignment] = field(default_factory=list)
    contracts: List[Contract] = field(default_factory=list)
    deviations: List[SquadDeviation] = field(default_factory=list)
    production: List[Production] = field(default_factory=list)


#: The largest game time the wire's unsigned thirty-two bits can carry, which is about fifty days of match.
_LATEST = 0xFFFFFFFF


def _clock(value) -> int:
    """A game time, brought into the range the wire can carry.

    A deadline is a number a person may type, and a person may type a large one. Left unbounded it reaches the encoder, which raises, and the exception unwinds through the policy and the session and takes the whole link down — an instance lost for the rest of a run because somebody asked for a mission to be finished in a thousand years. Clamping turns an absurd deadline into a distant one, which is what was meant.
    """
    return min(_LATEST, max(0, int(value)))


def encode_action(action: Action) -> bytes:
    parts = [_SECTION.pack(len(action.squads))]
    for assignment in action.squads:
        parts.append(_SQUAD_HEADER.pack(
            assignment.squad, int(assignment.commander),
            NO_OWNER if assignment.owner < 0 else assignment.owner + 1,
            len(assignment.units),
        ))
        for unit in assignment.units:
            parts.append(_UNIT_ID.pack(unit))

    parts.append(_SECTION.pack(len(action.contracts)))
    for contract in action.contracts:
        parts.append(_CONTRACT.pack(
            contract.squad, int(contract.task), int(contract.stance), contract.target_region,
            1 if contract.override else 0,
            float(contract.cost_budget), _clock(contract.deadline_ms), _clock(contract.issued_at_ms),
        ))

    parts.append(_SECTION.pack(len(action.deviations)))
    for deviation in action.deviations:
        parts.append(_DEVIATION.pack(deviation.squad, int(deviation.deviation),
                                     1 if deviation.override else 0))

    parts.append(_SECTION.pack(len(action.production)))
    for item in action.production:
        parts.append(_PRODUCTION.pack(
            item.producer, item.type_index, int(item.kind), 1 if item.cancel else 0,
            item.x, item.y,
        ))

    return b"".join(parts)


def decode_action(body: bytes) -> Action:
    """The mirror of encode_action, used by the tests that hold both sides to the same layout."""
    action = Action()
    offset = 0
    count, = _SECTION.unpack_from(body, offset)
    offset += _SECTION.size
    for _ in range(count):
        squad, commander, owner, units = _SQUAD_HEADER.unpack_from(body, offset)
        offset += _SQUAD_HEADER.size
        members = []
        for _ in range(units):
            member, = _UNIT_ID.unpack_from(body, offset)
            offset += _UNIT_ID.size
            members.append(member)
        action.squads.append(SquadAssignment(
            squad=squad, commander=commander, units=members,
            owner=-1 if owner == NO_OWNER else owner - 1,
        ))

    count, = _SECTION.unpack_from(body, offset)
    offset += _SECTION.size
    for _ in range(count):
        values = _CONTRACT.unpack_from(body, offset)
        offset += _CONTRACT.size
        action.contracts.append(Contract(
            squad=values[0], task=Task(values[1]), stance=Stance(values[2]),
            target_region=values[3], override=bool(values[4]), cost_budget=values[5],
            deadline_ms=values[6], issued_at_ms=values[7],
        ))

    count, = _SECTION.unpack_from(body, offset)
    offset += _SECTION.size
    for _ in range(count):
        squad, deviation, override = _DEVIATION.unpack_from(body, offset)
        offset += _DEVIATION.size
        action.deviations.append(SquadDeviation(squad=squad, deviation=Deviation(deviation),
                                                override=bool(override)))

    count, = _SECTION.unpack_from(body, offset)
    offset += _SECTION.size
    for _ in range(count):
        values = _PRODUCTION.unpack_from(body, offset)
        offset += _PRODUCTION.size
        action.production.append(Production(
            producer=values[0], type_index=values[1], kind=ProductionKind(values[2]),
            cancel=bool(values[3]), x=values[4], y=values[5],
        ))

    return action


_SIZES: Dict[str, int] = {
    "squad_header": _SQUAD_HEADER.size,
    "contract": _CONTRACT.size,
    "deviation": _DEVIATION.size,
    "production": _PRODUCTION.size,
}
