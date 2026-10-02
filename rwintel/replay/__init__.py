"""Replays: reading the recorded command stream, playing a match back under observation, and turning a person's play into decisions to imitate.

A replay played back reproduces the match it recorded, so the state behind every recorded command can be recovered. `container` and `commands` read the file, `analysis` summarises it without the game, and `playback` drives a game instance through it.
"""

from .commands import Command, Order
from .container import Replay, read_replay
from .stream import ReplayFormatError

__all__ = ["Command", "Order", "Replay", "ReplayFormatError", "read_replay"]
