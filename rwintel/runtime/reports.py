"""Reading what the probe agent writes to its logs, and reducing it and a control side's journal to the figures a measurement run reports.

The probe agent reports two kinds of line. A rate line, `fps=<n> speed=<n>x step=<n>ms ...`, every reporting interval; and a result line, `[rw-probe] result: seconds=... winner=...`, when an episode ends. A rate line that spans an episode boundary carries no plain positive rate and does not match.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from ..eval.sampling import Summary, episodes_for

RATE = re.compile(r"fps=([\d.]+) speed=([\d.]+)x step=([\d.]+)ms(?:.*?objects=(\d+))?")
RESULT = re.compile(r"^\[rw-probe\] result:")
RESULT_FIELDS = re.compile(r"seconds=(\d+) frames=(\d+) winner=(-?\d+) aliveTeams=(\d+) timeout=(\w+) units=(\d+)")
TEAM_VALUE = re.compile(r"team(-?\d+)Value=(\d+)")

#: Differences in win rate and in surviving value share that an outcome report sizes a comparison for.
WIN_RATE_DIFFERENCES = (0.10, 0.20, 0.30)
EDGE_DIFFERENCES = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class RateSample:
    fps: float
    speed: float
    step_ms: float
    objects: Optional[int] = None


def rate_samples(lines: Iterable[str]) -> List[RateSample]:
    samples = []
    for line in lines:
        match = RATE.search(line)
        if match:
            samples.append(RateSample(float(match.group(1)), float(match.group(2)), float(match.group(3)),
                                      int(match.group(4)) if match.group(4) else None))
    return samples


@dataclass(frozen=True)
class RateSummary:
    instances: int
    requested_speed: float
    speed_average: float
    speed_minimum: float
    speed_aggregate: float
    fps_average: float
    fps_total: float
    step_average_ms: float
    step_maximum_ms: float

    def lines(self) -> List[str]:
        return [
            f"instances          {self.instances}",
            f"requested speed    {f'{self.requested_speed:g}x' if self.requested_speed > 0 else 'unlimited'}",
            f"speed average      {self.speed_average:.2f}x",
            f"speed minimum      {self.speed_minimum:.2f}x",
            f"speed aggregate    {self.speed_aggregate:.1f}x",
            f"fps average        {self.fps_average:.0f}",
            f"fps total          {self.fps_total:.0f}",
            f"step average       {self.step_average_ms:.1f} ms",
            f"step maximum       {self.step_maximum_ms:.1f} ms",
        ]


def summarise_rates(last_samples: Sequence[RateSample], requested_speed: float) -> Optional[RateSummary]:
    """One figure per instance, from the last sample each reported: the early samples are distorted by the JIT warming up, and the last is the furthest from that."""
    if not last_samples:
        return None
    speeds = [s.speed for s in last_samples]
    rates = [s.fps for s in last_samples]
    steps = [s.step_ms for s in last_samples]
    return RateSummary(
        instances=len(last_samples), requested_speed=requested_speed,
        speed_average=sum(speeds) / len(speeds), speed_minimum=min(speeds), speed_aggregate=sum(speeds),
        fps_average=sum(rates) / len(rates), fps_total=sum(rates),
        step_average_ms=sum(steps) / len(steps), step_maximum_ms=max(steps),
    )


@dataclass(frozen=True)
class Outcome:
    seconds: int
    frames: int
    winner: int
    timeout: bool
    units: int
    value0: int = 0
    value1: int = 0

    @property
    def decided(self) -> bool:
        return not self.timeout and self.winner >= 0

    @property
    def edge(self) -> Optional[float]:
        """The share of the surviving value team 0 holds over team 1, which is what a score of an undecided position has to look like."""
        total = self.value0 + self.value1
        return (self.value0 - self.value1) / total if total > 0 else None


def outcomes(lines: Iterable[str]) -> List[Outcome]:
    found = []
    for line in lines:
        if not RESULT.search(line):
            continue
        match = RESULT_FIELDS.search(line)
        if not match:
            continue
        values = {int(team): int(value) for team, value in TEAM_VALUE.findall(line)}
        found.append(Outcome(seconds=int(match.group(1)), frames=int(match.group(2)), winner=int(match.group(3)),
                             timeout=match.group(5).lower() == "true", units=int(match.group(6)),
                             value0=values.get(0, 0), value1=values.get(1, 0)))
    return found


def count_results(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if RESULT.search(line))


@dataclass(frozen=True)
class OutcomeSummary:
    episodes: int
    decided: int
    timeouts: int
    #: Team 0's share of the decided episodes. With the local player watching, team 0 against team 1 is the whole matchup.
    win_rate: Optional[float]
    win_rate_error: Optional[float]
    length: Summary
    length_minimum: int
    length_maximum: int
    units_mean: float
    edge: Summary


def summarise_outcomes(found: Sequence[Outcome]) -> OutcomeSummary:
    decided = [o for o in found if o.decided]
    wins = sum(1 for o in decided if o.winner == 0)
    rate = wins / len(decided) if decided else None
    # A win rate is a Bernoulli mean, so its standard error is sqrt(p(1-p)/n).
    error = math.sqrt(rate * (1 - rate) / len(decided)) if rate is not None else None
    lengths = [o.seconds for o in found]
    edges = [o.edge for o in found if o.edge is not None]
    return OutcomeSummary(
        episodes=len(found), decided=len(decided), timeouts=sum(1 for o in found if o.timeout),
        win_rate=rate, win_rate_error=error, length=Summary.of(lengths),
        length_minimum=min(lengths) if lengths else 0, length_maximum=max(lengths) if lengths else 0,
        units_mean=sum(o.units for o in found) / len(found) if found else 0.0, edge=Summary.of(edges),
    )


def game_seconds_per_second(found: Sequence[Outcome], wall_seconds: float) -> float:
    """Game seconds the whole run played per second of wall clock, episode changes included, which is what an estimate of a longer run has to divide by."""
    if wall_seconds <= 0:
        return 0.0
    return sum(o.seconds for o in found) / wall_seconds


@dataclass(frozen=True)
class Throughput:
    """How much game a control-side run played for the wall clock it took, from the episodes it journalled."""

    episodes: int
    game_seconds: int
    wall_seconds: float

    @property
    def rate(self) -> float:
        return self.game_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def line(self) -> str:
        return (f"throughput: {self.episodes} episode(s), {self.game_seconds} game seconds in {self.wall_seconds:.0f}s, "
                f"{self.rate:.1f} game seconds per second")


def throughput(entries: Iterable[dict], wall_seconds: float) -> Throughput:
    """The episodes journalled and the game seconds they played, over the wall clock from the games' start to the end of the last control process, episode changes and every instance's idle tail included."""
    episodes = seconds = 0
    for entry in entries:
        episodes += 1
        seconds += int(entry.get("seconds", 0))
    return Throughput(episodes, seconds, wall_seconds)


def hours_for_both_arms(per_arm: int, mean_seconds: float, rate: float) -> float:
    """Wall-clock hours two arms of `per_arm` episodes each take at `rate` game seconds per wall second over all instances."""
    if rate <= 0:
        return math.inf
    return 2 * per_arm * mean_seconds / rate / 3600.0


def outcome_lines(summary: OutcomeSummary, rate: float, instances: int) -> List[str]:
    def fixed(value: Optional[float], digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        f"episodes           {summary.episodes}",
        f"decided            {summary.decided}",
        f"timeouts           {summary.timeouts}",
        f"win rate team 0    {fixed(summary.win_rate)}  (standard error {fixed(summary.win_rate_error)})",
        f"length             mean {summary.length.mean:.0f}s  sd {summary.length.sd:.0f}s  range {summary.length_minimum}s to {summary.length_maximum}s",
        f"units at the end   mean {summary.units_mean:.0f}",
        f"value edge         mean {summary.edge.mean:+.3f}  sd {summary.edge.sd:.3f}  over {summary.edge.n}",
        f"throughput         {rate:.0f} game seconds per second over {instances} instances",
        "",
        "episodes per arm needed, at the 5% level with 80% power",
        "  on the win rate, if matches are decided at all:",
    ]
    p = summary.win_rate if summary.win_rate is not None else 0.5
    for delta in WIN_RATE_DIFFERENCES:
        n = episodes_for(math.sqrt(p * (1 - p)), delta)
        lines.append(f"    {delta:4.0%} difference: {n:6d} per arm, "
                     f"{hours_for_both_arms(n, summary.length.mean, rate):.2f} hours for both arms at {instances} instances")
    if summary.edge.n > 1:
        lines.append("  on the surviving value share, which every episode yields:")
        for delta in EDGE_DIFFERENCES:
            n = episodes_for(summary.edge.sd, delta)
            lines.append(f"    {delta:5.2f} difference: {n:6d} per arm, "
                         f"{hours_for_both_arms(n, summary.length.mean, rate):.2f} hours for both arms at {instances} instances")
    return lines
