"""Game time at each frame of a recorded match, and what the engine's extra checksums are called.

The engine advances its clock once per simulation step by the step rate in sixtieths of a second, truncated to whole milliseconds, so a step at rate 1 is 16 ms and a step at rate 2 is 33 ms. A match starts at rate 1 and the host changes the rate with a system command; the new rate governs the steps after the one following the frame the command is recorded at. Converting a frame to a time therefore needs the rate changes, which the command stream carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

#: Milliseconds in one sixtieth of a second, as the engine holds it.
SIXTIETH_MS = 16.666666

#: The step rate a match starts at.
INITIAL_RATE = 1.0

#: Steps after the recorded frame before a change of rate takes effect.
RATE_LAG = 1

#: What each value of an extra checksum record (`es`) measures, in the order the engine registers them (`gameFramework.j.ak`). The three credit entries are the credits of the players in slots 0, 1 and 2.
EXTRA_CHECKSUMS: Tuple[str, ...] = (
    "unit_positions", "unit_directions", "unit_health", "unit_ids", "waypoints", "waypoint_positions",
    "credits", "unit_paths", "unit_count", "team_info", "slot0_credits", "slot1_credits", "slot2_credits",
    "command_center2", "command_center3",
)

#: The slots whose credits an extra checksum record carries, and the entry each is under.
CREDIT_ENTRIES: Tuple[Tuple[int, str], ...] = ((0, "slot0_credits"), (1, "slot1_credits"), (2, "slot2_credits"))


def step_ms(rate: float) -> int:
    """Game milliseconds one step advances at a rate."""
    return int(rate * SIXTIETH_MS)


@dataclass(frozen=True)
class Clock:
    """Frame to game time over a list of rate changes, each the frame it was recorded at and the new rate."""

    changes: Tuple[Tuple[int, float], ...] = ()

    @staticmethod
    def of(changes: Iterable[Tuple[int, float]]) -> "Clock":
        return Clock(tuple(sorted(changes)))

    def _segments(self) -> List[Tuple[int, float]]:
        """Each rate with the first step it governs, steps counted from 1."""
        segments = [(1, INITIAL_RATE)]
        for frame, rate in self.changes:
            first = frame + RATE_LAG + 1
            if first <= segments[-1][0]:
                segments[-1] = (segments[-1][0], rate)
            else:
                segments.append((first, rate))
        return segments

    def time_ms(self, frame: int) -> int:
        """Game time after `frame` steps."""
        total = 0
        segments = self._segments()
        for index, (first, rate) in enumerate(segments):
            if frame < first:
                break
            last = segments[index + 1][0] - 1 if index + 1 < len(segments) else frame
            total += (min(last, frame) - first + 1) * step_ms(rate)
        return total

    def frame_at(self, time_ms: int) -> int:
        """The first frame whose game time has reached `time_ms`."""
        frame = 0
        elapsed = 0
        segments = self._segments()
        for index, (first, rate) in enumerate(segments):
            width = step_ms(rate)
            last = segments[index + 1][0] - 1 if index + 1 < len(segments) else None
            steps_here = None if last is None else last - first + 1
            needed = -(-(time_ms - elapsed) // width) if time_ms > elapsed else 0
            if steps_here is None or needed <= steps_here:
                return frame + needed
            frame += steps_here
            elapsed += steps_here * width
        return frame


def credits_by_slot(values: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """The per slot credits out of one extra checksum record, as slot and credits."""
    named = dict(zip(EXTRA_CHECKSUMS, values))
    return tuple((slot, int(named[name])) for slot, name in CREDIT_ENTRIES if name in named)
