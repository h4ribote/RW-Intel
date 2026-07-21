"""How many episodes a claimed difference needs before it is worth claiming.

The same match run twice does not reproduce, so two policies can never be compared on one game; they are compared on the distribution of a few hundred. That makes the sample size a first-class part of the design rather than an afterthought — every statement of the form "this policy is better" is really a statement about two means and their scatter, and the honest form of it names the number of episodes it took.

The arithmetic is the standard two-sample sizing, and its point is a practical one. A win rate needs roughly sixteen hundred episodes a side to resolve five percentage points, and worse, it needs matches that end with a winner, which the measured ones do not. The military value edge is produced by every episode without exception and has a measured scatter of about 0.228, which puts a ten percent difference within about eighty episodes a side. That difference in cost is the entire argument for scoring the board instead of counting wins.

Everything here is on the sample, not on the population: the standard deviation divides by n-1, which is what the measurement tool reports and what the recorded figures were computed with.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Sequence


#: (z(1 - alpha/2) + z(power))^2 for five per cent significance at eighty per cent power, which is (1.96 + 0.84)^2. This is a choice of how sure a comparison has to be before it counts, not a derivation; it is stated as a constant so that raising the bar is one edit rather than a rederivation everywhere a sample size is quoted.
SIGNIFICANCE_POWER_FACTOR = 7.85

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


@dataclass(frozen=True)
class Comparison:
    """Two sets of episodes held against each other, with the sample size the difference between them would need.

    `sufficient` is deliberately the weakest possible claim: it says only that both sides have already run at least as many episodes as the observed difference requires. It is not a test result and does not become one — a comparison that is not sufficient means run more, and one that is means the difference is worth reporting with its episode count beside it.
    """

    first: Summary
    second: Summary
    #: First mean less second mean, so a positive difference means the first side scored higher.
    difference: float
    pooled_sd: float
    #: Episodes per side that `difference` needs at `pooled_sd`.
    needed: int
    sufficient: bool

    @classmethod
    def of(cls, first: Summary, second: Summary) -> "Comparison":
        difference = first.mean - second.mean
        pooled = _pooled_sd(first, second)
        needed = episodes_for(pooled, abs(difference))
        return cls(first, second, difference, pooled, needed,
                   first.n >= needed and second.n >= needed)


def _pooled_sd(first: Summary, second: Summary) -> float:
    """The two scatters combined, weighted by how many episodes each rests on. Zero when there are not enough episodes between them to have any scatter at all."""
    degrees = first.n + second.n - 2
    if degrees <= 0:
        return 0.0
    total = (first.n - 1) * first.sd ** 2 + (second.n - 1) * second.sd ** 2
    return math.sqrt(total / degrees)
