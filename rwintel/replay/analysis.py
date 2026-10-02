"""What a replay says about how each side played, read without starting the game.

Everything here comes from the command stream and the checksum records: what each player ordered and when, what they asked to have produced and built and in which order, how often they acted, and how many credits each slot held at every checksum. What it cannot say is what came of any of it -whether a unit was finished, where it went, what it destroyed -because that is the simulation's to decide and only a playback runs it.

The match is taken to end at the first system message announcing a defeat, when the recording carries one, and otherwise at the last recorded frame. A recording can run on past the end of the match, and what is issued after the end is not part of it.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from .commands import SYSTEM, Command
from .container import Replay
from .timing import credits_by_slot

#: The system message the recording side writes when a player is defeated.
_DEFEATED = re.compile(r"^(?P<name>.*) was defeated")


@dataclass
class Placement:
    """One production or building request, as the time it was issued and the type it asked for."""

    second: float
    type: str
    #: `unit` for production, `building` for a placement.
    kind: str
    #: What the definition says the type costs, or None when the catalogue does not know it.
    price: Optional[int] = None


@dataclass
class PlayerSummary:
    slot: int
    commands: int = 0
    #: Commands issued in each game minute of the match.
    per_minute: List[int] = field(default_factory=list)
    orders: Dict[str, int] = field(default_factory=dict)
    #: Requests to produce or to place, cancellations excluded, in the order they were issued.
    placements: List[Placement] = field(default_factory=list)
    #: Tier raises and any other offered action by name, with how often each was issued.
    other_actions: Dict[str, int] = field(default_factory=dict)
    cancels: int = 0
    stance_changes: int = 0
    #: Mean units a command with any selected addressed.
    mean_selection: float = 0.0
    #: Credits at every extra checksum record inside the match, as game seconds and credits.
    credits: List[Tuple[float, int]] = field(default_factory=list)

    def first_seen(self) -> List[Placement]:
        """Each type at the first time it was asked for, which is the build order."""
        seen = set()
        order = []
        for placement in self.placements:
            if placement.type not in seen:
                seen.add(placement.type)
                order.append(placement)
        return order

    def requested_value(self) -> int:
        """Credits the requests would cost at definition prices, over the types whose price is known."""
        return sum(p.price for p in self.placements if p.price is not None)

    def as_dict(self) -> dict:
        return {
            "slot": self.slot, "commands": self.commands, "per_minute": self.per_minute,
            "orders": self.orders, "cancels": self.cancels, "stance_changes": self.stance_changes,
            "mean_selection": round(self.mean_selection, 2),
            "build_order": [{"second": round(p.second, 1), "type": p.type, "kind": p.kind}
                            for p in self.first_seen()],
            "requested": dict(collections.Counter(p.type for p in self.placements)),
            "requested_value": self.requested_value(),
            "other_actions": self.other_actions,
            "credits": [[round(second, 1), value] for second, value in self.credits],
        }


@dataclass
class ReplaySummary:
    map_path: str
    version: str
    #: The frame and game second the match ended at, as this module judges it.
    end_frame: int
    end_second: float
    #: Why the match is judged to end where it does: `defeat` for a recorded defeat, `recording` for the last record.
    end_reason: str
    defeated: List[str]
    closed: bool
    checksums: int
    players: List[PlayerSummary]
    #: The system messages inside the recording, as game second and text.
    messages: List[Tuple[float, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "map": self.map_path, "version": self.version, "end_frame": self.end_frame,
            "end_second": round(self.end_second, 3), "end_reason": self.end_reason,
            "defeated": self.defeated, "closed": self.closed, "checksums": self.checksums,
            "messages": [[round(second, 1), text] for second, text in self.messages],
            "players": [player.as_dict() for player in self.players],
        }


def match_end(replay: Replay) -> Tuple[int, str, List[str]]:
    """The frame the match ended at, why that frame, and who was recorded as defeated by then."""
    defeats = [(chat.frame, _DEFEATED.match(chat.message or "")) for chat in replay.chats if chat.slot == SYSTEM]
    defeats = [(frame, found.group("name")) for frame, found in defeats if found]
    if defeats:
        frame = min(frame for frame, _ in defeats)
        return frame, "defeat", [name for at, name in defeats if at == frame]
    return replay.last_frame, "recording", []


def summarise(replay: Replay, prices: Optional[Mapping[str, int]] = None) -> ReplaySummary:
    """Per player what the replay records up to the end of the match. `prices` maps a type's reported name to its cost."""
    prices = prices or {}
    clock = replay.clock
    end_frame, reason, defeated = match_end(replay)
    minutes = int(clock.time_ms(end_frame) // 60000) + 1

    players: Dict[int, PlayerSummary] = {}
    selections: Dict[int, List[int]] = collections.defaultdict(list)
    for command in replay.commands:
        if command.player == SYSTEM or command.frame > end_frame:
            continue
        summary = players.setdefault(command.player, PlayerSummary(slot=command.player, per_minute=[0] * minutes))
        _count(summary, command, clock.time_ms(command.frame) / 1000.0, prices)
        if command.units:
            selections[command.player].append(len(command.units))
    for slot, sizes in selections.items():
        players[slot].mean_selection = sum(sizes) / len(sizes)

    for record in replay.extra_checksums:
        if record.frame > end_frame:
            continue
        second = clock.time_ms(record.frame) / 1000.0
        for slot, credits in credits_by_slot(record.values):
            if slot in players:
                players[slot].credits.append((second, credits))

    messages = [(clock.time_ms(chat.frame) / 1000.0, chat.message or "")
                for chat in replay.chats if chat.slot == SYSTEM and chat.message]
    return ReplaySummary(
        map_path=replay.map_path, version=replay.header.version_name, end_frame=end_frame,
        end_second=clock.time_ms(end_frame) / 1000.0, end_reason=reason, defeated=defeated,
        closed=replay.ended, checksums=sum(1 for frame, _ in replay.checksums if frame <= end_frame),
        players=[players[slot] for slot in sorted(players)], messages=messages)


def _count(summary: PlayerSummary, command: Command, second: float, prices: Mapping[str, int]) -> None:
    summary.commands += 1
    minute = int(second // 60)
    if minute < len(summary.per_minute):
        summary.per_minute[minute] += 1
    if command.order_kind is not None:
        summary.orders[command.order_kind] = summary.orders.get(command.order_kind, 0) + 1
    if command.stance is not None:
        summary.stance_changes += 1
    produced, placed = command.produces, command.places
    if command.action is None and placed is None:
        return
    if command.cancel:
        summary.cancels += 1
        return
    if produced is not None or placed is not None:
        kind_name = produced if produced is not None else placed
        summary.placements.append(Placement(second=second, type=kind_name,
                                            kind="unit" if produced is not None else "building",
                                            price=prices.get(kind_name)))
    else:
        summary.other_actions[command.action] = summary.other_actions.get(command.action, 0) + 1


def catalogue_prices(assets=None) -> Dict[str, int]:
    """Definition prices by every name a type may be reported under, or nothing when the game's assets are not there to read."""
    from ..data.units import read_unit_catalog

    try:
        catalogue = read_unit_catalog(assets)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    prices: Dict[str, int] = {}
    for name, definition in catalogue.items():
        prices.setdefault(definition.definition_name, definition.price)
        prices.setdefault(name, definition.price)
    return prices


def render(summary: ReplaySummary) -> str:
    """The summary as lines a person reads."""
    lines = [f"map {summary.map_path}, version {summary.version}",
             f"match ends at {summary.end_second:.1f}s (frame {summary.end_frame}, {summary.end_reason}"
             + (f": {', '.join(summary.defeated)} defeated" if summary.defeated else "") + ")",
             f"recording {'closed' if summary.closed else 'not closed'}, {summary.checksums} checksum(s) inside the match"]
    for second, text in summary.messages:
        lines.append(f"  {second:7.1f}s  {text}")
    for player in summary.players:
        per_minute = player.commands / max(1, len(player.per_minute))
        lines.append(f"slot {player.slot}: {player.commands} command(s), {per_minute:.1f} a minute, "
                     f"mean selection {player.mean_selection:.1f}, {player.cancels} cancel(s)")
        if player.orders:
            lines.append("  orders: " + ", ".join(f"{k} {v}" for k, v in sorted(player.orders.items(), key=lambda kv: -kv[1])))
        requested = collections.Counter(p.type for p in player.placements)
        if requested:
            lines.append("  requested: " + ", ".join(f"{k} {v}" for k, v in requested.most_common()))
        order = player.first_seen()
        if order:
            lines.append("  build order: " + ", ".join(f"{p.type}@{p.second:.0f}s" for p in order))
        if player.other_actions:
            lines.append("  other actions: " + ", ".join(f"{k} {v}" for k, v in sorted(player.other_actions.items())))
        if player.credits:
            last = player.credits[-1]
            peak = max(player.credits, key=lambda c: c[1])
            lines.append(f"  credits: {last[1]} at {last[0]:.0f}s, peak {peak[1]} at {peak[0]:.0f}s")
    return "\n".join(lines)
