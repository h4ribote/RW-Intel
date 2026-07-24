"""How many episodes a claimed difference needs before it is worth claiming.

The same match run twice does not reproduce, so two policies can never be compared on one game; they are compared on the distribution of a few hundred. That makes the sample size a first-class part of the design rather than an afterthought — every statement of the form "this policy is better" is really a statement about two means and their scatter, and the honest form of it names the number of episodes it took.

The arithmetic is the standard two-sample sizing, and its point is a practical one. A win rate needs roughly sixteen hundred episodes a side to resolve five percentage points, and worse, it needs matches that end with a winner, which the measured ones do not. The military value edge is produced by every episode without exception and has a measured scatter of about 0.228, which puts a ten percent difference within about eighty episodes a side. That difference in cost is the entire argument for scoring the board instead of counting wins.

Everything here is on the sample, not on the population: the standard deviation divides by n-1, which is what the measurement tool reports and what the recorded figures were computed with.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple


#: (z(1 - alpha/2) + z(power))^2 for five per cent significance at eighty per cent power, which is (1.96 + 0.84)^2. This is a choice of how sure a comparison has to be before it counts, not a derivation; it is stated as a constant so that raising the bar is one edit rather than a rederivation everywhere a sample size is quoted.
SIGNIFICANCE_POWER_FACTOR = 7.85

#: The variance of a coin at p = 0.5, which is where a win rate comparison between two comparable policies sits and where the variance is at its largest. Sizing at the worst case means the answer never has to be revised upward once the rate is known.
WIN_RATE_VARIANCE = 0.25

#: What `episodes_for` returns when the difference to demonstrate is zero: no finite number of episodes establishes a difference that is not there.
UNBOUNDED_EPISODES = sys.maxsize

#: Below this, relative to the mean it sits beside, a scatter is not a measurement of one. Episodes that all scored identically leave a sample standard deviation that is not nought but the rounding of nought — summing and squaring identical values does not cancel exactly in binary, and twenty differences of 0.4 leave about 6e-17 — and reading that as scatter sizes the comparison off the last bits of the arithmetic and declares it settled on a single pair. Stated as a ratio so it is a claim about significant figures and not about the units of whatever is being scored.
NEGLIGIBLE_SCATTER = 1e-9


def _no_scatter(sd: float, mean: float) -> bool:
    """Whether a reported scatter is the absence of a measurement of scatter rather than a measurement of no scatter."""
    return sd <= NEGLIGIBLE_SCATTER * max(1.0, abs(mean))


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


def pairs_for(sigma: float, delta: float) -> int:
    """Pairs needed to resolve a difference of `delta` when the two arms are run on the same boards and subtracted pair by pair, and `sigma` is the scatter of that difference rather than of either arm.

    Half of `episodes_for`, and for a reason that is the whole argument for pairing: a two-sample comparison pays for the scatter of both arms and needs that many episodes on each side, whereas a paired one has one sample — the differences — and the board draw, which is the largest term in the scatter of either arm, has already cancelled inside each difference. So the count here is per pair, and one pair is one board played by both arms.
    """
    if delta == 0.0:
        return UNBOUNDED_EPISODES
    return math.ceil(SIGNIFICANCE_POWER_FACTOR * sigma ** 2 / delta ** 2)


@dataclass(frozen=True)
class PairedComparison:
    """Two arms run on the same boards, subtracted board by board.

    This is a different instrument from `Comparison` and answers a different question. `Comparison` holds two independent samples against each other and must carry the scatter of both, most of which is the boards they happened to draw. A paired comparison subtracts the two arms on one board first, so whatever the board did to both of them is gone from the difference before any averaging happens, and what is left is what the two arms did differently on it.

    The reported interval is two standard errors of the mean difference, which is the same two-standard-error reading every pooled figure in this project is quoted with, and `resolves` says only that the interval excludes nought. That is the weakest honest claim and it stays weak on purpose: it is not a test, and a comparison that does not resolve means run more pairs rather than that the two arms are equal.
    """

    #: The per-board differences, first arm less second, in the order the boards were paired.
    differences: Tuple[float, ...]
    #: The two arms' own unpaired means, kept beside the difference because a paired difference says nothing about where either arm stood.
    first: Summary
    second: Summary
    #: The differences read as one sample: count of pairs, mean difference and its scatter.
    paired: Summary
    #: Two standard errors of the mean difference. Nought for fewer than two pairs, where scatter is not defined.
    interval: float
    #: Pairs the observed difference needs at the observed scatter of the differences.
    needed: int
    #: Whether the interval excludes nought. False when there are too few pairs to have scatter at all, since a difference with no measured scatter is unresolved rather than certain.
    resolves: bool

    @classmethod
    def of(cls, first_scores: Sequence[float], second_scores: Sequence[float]) -> "PairedComparison":
        """The two arms' scores in matched order — `first_scores[i]` and `second_scores[i]` must be the same board — reduced to the difference between them."""
        if len(first_scores) != len(second_scores):
            raise ValueError("a paired comparison needs the same number of scores on both sides")
        differences = tuple(a - b for a, b in zip(first_scores, second_scores))
        paired = Summary.of(differences)
        interval = 2.0 * paired.sd / math.sqrt(paired.n) if paired.n > 1 else 0.0
        if paired.n < 2 or _no_scatter(paired.sd, paired.mean):
            # No scatter is the absence of a measurement of scatter here, not the certainty that there is none: one pair has no spread by definition, and a handful that happen to agree exactly would size the comparison at nought pairs and declare it settled off a coincidence. So the count is unbounded and the comparison does not resolve, whatever the mean difference came to.
            return cls(differences, Summary.of(first_scores), Summary.of(second_scores),
                       paired, interval, UNBOUNDED_EPISODES, False)
        needed = pairs_for(paired.sd, abs(paired.mean))
        return cls(differences, Summary.of(first_scores), Summary.of(second_scores),
                   paired, interval, needed, abs(paired.mean) > interval)


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
        if difference != 0.0 and _no_scatter(pooled, difference):
            # A pooled scatter of nought here is not the certainty that no scatter would be; it is its absence. It comes from too few episodes between the two arms to have any within-group scatter at all (one a side leaves _pooled_sd's degrees at nought), or from a handful that happen to be identical. Sizing against it would return nought episodes and declare the difference established off a single or coincidental pair, which is the one thing a sample size exists to refuse. So a nonzero difference with no measured scatter is unresolved and never sufficient, not settled by nought.
            return cls(first, second, difference, pooled, UNBOUNDED_EPISODES, False)
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
