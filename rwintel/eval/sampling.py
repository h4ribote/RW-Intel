"""How many episodes a difference needs, and whether the episodes run establish it.

The same match run twice does not reproduce, so two policies can never be compared on one game; they are compared on the distribution of a few hundred. Two things follow. Before a run, the scatter of the score says how many episodes a difference of a given size would take (`episodes_for`). After it, the difference found is judged by its interval and its p-value (`Difference`), and when several arms are held against one reference the p-values are corrected for there being several (`holm`).

The sizing is the standard two-sample arithmetic, and its point is a practical one. A win rate needs roughly sixteen hundred episodes a side to resolve five percentage points, and it needs matches that end with a winner, which most do not. A score read off the board is produced by every episode, which puts a ten percent difference under a hundred episodes a side at the scatter the military value edge has.

Everything here is on the sample, not on the population: the standard deviation divides by n-1.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Sequence


#: (z(1 - alpha/2) + z(power))^2 for five per cent significance at eighty per cent power, which is (1.96 + 0.84)^2. This is a choice of how sure a comparison has to be before it counts, not a derivation; it is stated as a constant so that raising the bar is one edit rather than a rederivation everywhere a sample size is quoted.
SIGNIFICANCE_POWER_FACTOR = 7.85

#: The significance level a difference is judged at, the same five per cent the sizing is done at.
SIGNIFICANCE_LEVEL = 0.05

#: z(1 - alpha/2) at that level: the half-width of a 95 per cent interval in standard errors.
SIGNIFICANCE_Z = 1.96

#: The variance of a coin at p = 0.5, which is where a win rate comparison between two comparable policies sits and where the variance is at its largest. Sizing at the worst case means the answer never has to be revised upward once the rate is known.
WIN_RATE_VARIANCE = 0.25

#: What `episodes_for` returns when the difference to demonstrate is zero: no finite number of episodes establishes a difference that is not there.
UNBOUNDED_EPISODES = sys.maxsize


@dataclass(frozen=True)
class Summary:
    """A set of episode scores reduced to what a comparison needs of it."""

    n: int
    mean: float
    #: Sample standard deviation, zero for fewer than two values because scatter is not defined on one observation and reporting anything else would let a single episode size a comparison.
    sd: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "Summary":
        n = len(values)
        if n == 0:
            return cls(0, 0.0, 0.0)
        mean = sum(values) / n
        if n < 2:
            return cls(n, mean, 0.0)
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        return cls(n, mean, math.sqrt(variance))


def episodes_for(sigma: float, delta: float) -> int:
    """Episodes per side needed to resolve a difference of `delta` in a quantity that scatters by `sigma`.

    Rounded up, because a fractional episode is not a thing that can be run and rounding down would understate the requirement.
    """
    if delta == 0.0:
        return UNBOUNDED_EPISODES
    return math.ceil(SIGNIFICANCE_POWER_FACTOR * 2.0 * sigma ** 2 / delta ** 2)


def episodes_for_win_rate(delta: float) -> int:
    """Episodes per side needed to resolve a win rate difference of `delta`. The same sizing with the coin's own variance in place of a measured one."""
    if delta == 0.0:
        return UNBOUNDED_EPISODES
    return math.ceil(SIGNIFICANCE_POWER_FACTOR * 2.0 * WIN_RATE_VARIANCE / delta ** 2)


def standard_error(first: Summary, second: Summary) -> float:
    """The standard error of the difference of two means, each side on its own scatter (Welch). Nought when either side has too few episodes to have a scatter."""
    if first.n < 2 or second.n < 2:
        return 0.0
    return math.sqrt(first.sd ** 2 / first.n + second.sd ** 2 / second.n)


@dataclass(frozen=True)
class Difference:
    """Two sets of episodes held against each other: the difference of their means, the interval around it and how surprising it would be if there were none.

    The p-value is two-sided on the normal approximation, which the episode counts a comparison runs at make adequate. A standard error of nought is a scatter that was never measured rather than one measured to be small, so it gives a p-value of one: nothing has been established.
    """

    first: Summary
    second: Summary
    #: First mean less second mean, so a positive difference means the first side scored higher.
    difference: float
    standard_error: float
    #: Half-width of the interval, in the units of the score.
    interval: float
    p_value: float

    @classmethod
    def of(cls, first: Summary, second: Summary, z: float = SIGNIFICANCE_Z) -> "Difference":
        difference = first.mean - second.mean
        error = standard_error(first, second)
        p_value = math.erfc(abs(difference) / error / math.sqrt(2.0)) if error > 0.0 else 1.0
        return cls(first, second, difference, error, z * error, p_value)

    @property
    def low(self) -> float:
        return self.difference - self.interval

    @property
    def high(self) -> float:
        return self.difference + self.interval


def holm(p_values: Sequence[float], alpha: float = SIGNIFICANCE_LEVEL) -> List[bool]:
    """Which of several p-values stay significant once there are several of them, by Holm's step-down procedure. It holds the chance of any false claim among them to `alpha`."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(p_values) - rank):
            break
        rejected[index] = True
    return rejected
