"""The player commands a replay records, read field by field in the order the engine reads them.

A command is the same object a lockstep session carries between processes: who issued it, which of their units it addresses, what order it gives them, and a handful of switches. The reading here follows the engine's own reader for the command class (`gameFramework.e`) and for the order inside it (`game.units.au`), including the stream version each later field is gated on, so a field added after the replay's version is simply absent rather than misread.

Unit and player references are kept as the identifiers the stream carries. Resolving them needs the world the command was played into, which only a playback has; the identifiers are the same ones the observation reports, so a playback can join the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .stream import Reader, ReplayFormatError

#: The engine's order kinds (`game.units.av`) in declaration order, which is what the stream writes as an ordinal.
ORDER_KINDS: Tuple[str, ...] = (
    "move", "attack", "build", "repair", "loadInto", "unloadAt", "reclaim", "attackMove", "loadUp",
    "patrol", "guard", "guardAt", "touchTarget", "follow", "triggerAction", "triggerActionWhenInRange",
    "setPassiveTarget",
)

#: The engagement stances (`game.units.a`) in declaration order, matching `rwintel.wire.Stance`.
STANCES: Tuple[str, ...] = ("outOfRange", "onlyInRange", "returnFire", "holdFire", "guardArea", "aggressive", "mixed")

#: The built-in unit types (`game.units.ar`) in declaration order. A type defined by a file is written by name instead.
BUILT_IN_TYPES: Tuple[str, ...] = (
    "extractor", "landFactory", "airFactory", "seaFactory", "commandCenter", "turret", "antiAirTurret", "builder",
    "tank", "hoverTank", "artillery", "helicopter", "airShip", "gunShip", "missileShip", "gunBoat", "megaTank",
    "laserTank", "hovercraft", "ladybug", "battleShip", "tankDestroyer", "heavyTank", "heavyHoverTank",
    "laserDefence", "dropship", "tree", "repairbay", "NukeLaucher", "AntiNukeLaucher", "mammothTank",
    "experimentalTank", "experimentalLandFactory", "crystalResource", "wall_v", "fabricator", "attackSubmarine",
    "builderShip", "amphibiousJet", "supplyDepot", "experimentalHoverTank", "turret_artillery",
    "turret_flamethrower", "fogRevealer", "spreadingFire", "antiAirTurretT2", "turretT2", "turretT3",
    "damagingBorder", "zoneMarker", "editorOrBuilder", "dummyNonUnitWithTeam",
)

#: The player slot a command carries when the engine itself issued it, as it does for a change of step rate.
SYSTEM = -1

#: The identifier the stream writes where a unit reference is empty.
NO_UNIT = -1

#: The action name the stream writes where a command carries no special action.
NO_ACTION = "-1"

#: Order kinds that send units somewhere, by position or by following a target, which is what an inference of where a player sent a squad reads.
MOVEMENT_KINDS = frozenset({"move", "attack", "attackMove", "patrol", "guard", "guardAt", "follow"})

#: The special action a transport is told to set down what it carries with, and the one that calls that off: the engine's Unload and Cancel actions by identifier.
UNLOAD_ACTION = "109"
CANCEL_UNLOAD_ACTION = "110"


@dataclass(frozen=True)
class Order:
    """The order a command gives, as `game.units.au` holds it."""

    kind: Optional[str]
    #: The type being built, by the name the engine registered it under, for a build order.
    unit_type: Optional[str]
    x: float
    y: float
    #: The unit the order is aimed at, or NO_UNIT.
    target: int
    #: The action a build or trigger order names, when it names one.
    action: Optional[str] = None


@dataclass(frozen=True)
class Command:
    """One recorded command, with the frame the engine executes it on."""

    frame: int
    #: The issuing player's slot, or SYSTEM.
    player: int
    order: Optional[Order]
    #: The special action named, such as `u_<type>` to produce a unit, `b_<type>` to place a building, or a tier raise the unit offers. None when there is none.
    action: Optional[str]
    #: The stance set, as an index into STANCES, or None when the command sets none.
    stance: Optional[int]
    units: Tuple[int, ...]
    #: The engine's stopOrUndo switch (`e.g`), which cancels what the action names.
    cancel: bool = False
    point: Optional[Tuple[float, float]] = None
    #: The acting player (`e.p`), when it differs from nobody.
    acting: Optional[int] = None
    #: The second position and unit reference (`e.l`, `e.m`) that later versions carry.
    second_point: Optional[Tuple[float, float]] = None
    second_unit: int = NO_UNIT
    #: The step rate a system command sets, or None when it sets none.
    step_rate: Optional[float] = None
    system_action: int = 0
    #: How many movement records (`gameFramework.d`) the command carried; their paths are skipped.
    moves: int = 0
    #: The remaining switches, by the engine's own field names, for anyone comparing a decode with the engine's.
    switches: Tuple[Tuple[str, object], ...] = field(default_factory=tuple)

    @property
    def order_kind(self) -> Optional[str]:
        return self.order.kind if self.order is not None else None

    @property
    def produces(self) -> Optional[str]:
        """The type a production command asks for."""
        return self.action[2:] if self.action is not None and self.action.startswith("u_") else None

    @property
    def places(self) -> Optional[str]:
        """The type a building placement asks for. The game's own client places with a build order alone; a `b_` action may accompany it or stand in for it."""
        if self.order is not None and self.order.kind == "build" and self.order.unit_type is not None:
            return self.order.unit_type
        return self.action[2:] if self.action is not None and self.action.startswith("b_") else None

    @property
    def destination(self) -> Optional[Tuple[float, float]]:
        """Where a movement order sends its units, when it names a position rather than a unit."""
        if self.order is None or self.order.kind not in MOVEMENT_KINDS or self.order.target != NO_UNIT:
            return None
        return self.order.x, self.order.y

    @property
    def unloads(self) -> bool:
        """Whether the command tells the transports in `units` to set down what they carry."""
        return self.action == UNLOAD_ACTION

    @property
    def cancels_unload(self) -> bool:
        """Whether the command calls off the transports' unloading."""
        return self.action == CANCEL_UNLOAD_ACTION


def _order_kind(ordinal: Optional[int]) -> Optional[str]:
    if ordinal is None:
        return None
    if not 0 <= ordinal < len(ORDER_KINDS):
        raise ReplayFormatError(f"order kind {ordinal} is outside the {len(ORDER_KINDS)} the engine declares")
    return ORDER_KINDS[ordinal]


def read_unit_type(reader: Reader) -> Optional[str]:
    """A unit type reference: -1 for none, -2 followed by the name of a type defined by a file, or a built-in ordinal."""
    index = reader.int()
    if index == -1:
        return None
    if index == -2:
        return reader.utf()
    if not 0 <= index < len(BUILT_IN_TYPES):
        raise ReplayFormatError(f"built-in unit type {index} is outside the {len(BUILT_IN_TYPES)} the engine declares")
    return BUILT_IN_TYPES[index]


def read_order(reader: Reader) -> Order:
    kind = _order_kind(reader.enum())
    unit_type = read_unit_type(reader)
    x = reader.float()
    y = reader.float()
    target = reader.long()
    if reader.version >= 40:
        reader.byte()
    if reader.version >= 46:
        reader.float()
        reader.float()
    if reader.version >= 58:
        reader.boolean()
    if reader.version >= 65:
        reader.boolean()
    if reader.version >= 79:
        reader.boolean()
    action = reader.optional_utf() if reader.version >= 82 else None
    return Order(kind=kind, unit_type=unit_type, x=x, y=y, target=target, action=action)


def _skip_move(reader: Reader) -> None:
    """One movement record: a unit, four coordinates, a count, a movement type and possibly a compressed path, none of which a command's meaning depends on."""
    reader.long()
    for _ in range(4):
        reader.float()
    reader.int()
    reader.enum()
    if reader.boolean() and reader.boolean():
        reader.expect_block("p")


def read_command(reader: Reader, frame: int) -> Command:
    """Reads the body of a command block `c`, which must be consumed exactly."""
    player = reader.byte()
    order = read_order(reader) if reader.boolean() else None
    flag_e = reader.boolean()
    cancel = reader.boolean()
    reader.int()  # the action as a number, superseded by the name that follows at every version a replay can have
    stance = reader.enum()
    point = (reader.float(), reader.float()) if reader.boolean() else None
    flag_o = reader.boolean()
    units = tuple(reader.long() for _ in range(reader.int()))
    acting = None
    if reader.version >= 16 and reader.boolean():
        acting = reader.byte()
    second_point = None
    second_unit = NO_UNIT
    if reader.version >= 29:
        if reader.boolean():
            second_point = (reader.float(), reader.float())
        second_unit = reader.long()
    action = reader.utf() if reader.version >= 33 else NO_ACTION
    switches: List[Tuple[str, object]] = [("e", flag_e), ("o", flag_o)]
    if reader.version >= 37:
        switches.append(("f", reader.boolean()))
    if reader.version >= 52:
        switches.append(("q", reader.short()))
    step_rate = None
    system_action = 0
    moves = 0
    if reader.version >= 53:
        if reader.boolean():
            reader.byte()
            rate = reader.float()
            switches.append(("t", reader.float()))
            system_action = reader.int()
            step_rate = rate if rate != 0.0 else None
        moves = reader.int()
        for _ in range(moves):
            _skip_move(reader)
    if reader.version >= 80:
        switches.append(("h", reader.boolean()))
    if reader.remaining:
        raise ReplayFormatError(f"command at frame {frame} left {reader.remaining} byte(s) unread")
    return Command(frame=frame, player=player, order=order, action=None if action == NO_ACTION else action,
                   stance=stance, units=units, cancel=cancel, point=point, acting=acting,
                   second_point=second_point, second_unit=second_unit, step_rate=step_rate,
                   system_action=system_action, moves=moves, switches=tuple(switches))
