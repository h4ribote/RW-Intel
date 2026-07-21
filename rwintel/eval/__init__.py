"""Turning finished episodes into a number, and saying how many of them a claim needs.

Two policies cannot be compared on one match: the same seed and the same settings do not reproduce the same game, so there is no such thing as running the same match with one side swapped out. Everything here follows from that. A comparison is between two distributions, the quantity being distributed has to be produced by every episode rather than only by the ones that end in a win, and the number of episodes is part of the claim.

`scoring` is what an episode was worth. `sampling` is how many of them are enough.
"""

from .sampling import (
    Comparison,
    SIGNIFICANCE_POWER_FACTOR,
    Summary,
    UNBOUNDED_EPISODES,
    WIN_RATE_VARIANCE,
    episodes_for,
    episodes_for_win_rate,
)
from .scoring import (
    Components,
    DECIDED_SCORE,
    Episode,
    OPENING_WEIGHTS,
    SCORE_RANGE,
    WEIGHT_GRID_STEP,
    Weights,
    components,
    decided,
    fit_weights,
    score,
    viewpoint,
)

__all__ = [
    "Comparison",
    "Components",
    "DECIDED_SCORE",
    "Episode",
    "OPENING_WEIGHTS",
    "SCORE_RANGE",
    "SIGNIFICANCE_POWER_FACTOR",
    "Summary",
    "UNBOUNDED_EPISODES",
    "WEIGHT_GRID_STEP",
    "WIN_RATE_VARIANCE",
    "Weights",
    "components",
    "decided",
    "episodes_for",
    "episodes_for_win_rate",
    "fit_weights",
    "score",
    "viewpoint",
]
