"""The match score expected from a board, which the operational layer and the economy are shaped by.

A board is read as the time left in the match, our share of the worth standing on it and our share of the resource points held. The prediction is linear in the two shares, each written from -1 to +1, with coefficients that depend on the time left: they are given at a few knots and interpolated between them. There is no intercept, so an even board reads nought whoever the opponent is, as the match score itself does. With no fitted coefficients the prediction is the worth edge alone, which is what a match cut off by the clock is scored on.

The coefficients are fitted on recorded matches (`fit`), from the boards of their periods to the score each match finally got, and adopted by writing them to `PREDICTOR_FILE`.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Remaining game seconds the fitted coefficients are given at.
KNOTS = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 420.0, 540.0, 660.0, 780.0, 900.0)

#: Strength of the penalty on the squared coefficients of a fit, per sample.
RIDGE = 1e-3

#: Folds of the cross-validation a fit reports its held-out agreement from, split by match.
FOLDS = 5

#: Boards a knot needs before it is fitted; a knot with fewer takes the opening coefficients.
MIN_SAMPLES = 50

#: The adopted coefficients, which are the default whenever the file is present.
PREDICTOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor.json")


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class Predictor:
    """Coefficients on the worth edge and the ground edge at each knot of remaining game seconds, ascending."""

    knots: Tuple[float, ...] = (0.0,)
    value: Tuple[float, ...] = (1.0,)
    ground: Tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        for name in ("knots", "value", "ground"):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        if not self.knots or not len(self.knots) == len(self.value) == len(self.ground):
            raise ValueError("a predictor needs one value and one ground coefficient per knot, and at least one knot")
        if any(b <= a for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError(f"the knots of a predictor have to ascend: {self.knots}")

    def coefficients(self, remaining: float) -> Tuple[float, float]:
        """The value and ground coefficients at this many remaining seconds, interpolated between knots and held beyond the ends. A negative figure means the time left is not known and reads as the last knot."""
        knots = self.knots
        if remaining < 0 or remaining >= knots[-1]:
            return self.value[-1], self.ground[-1]
        if remaining <= knots[0]:
            return self.value[0], self.ground[0]
        index = next(i for i in range(len(knots) - 1) if remaining <= knots[i + 1])
        share = (remaining - knots[index]) / (knots[index + 1] - knots[index])
        return (self.value[index] + share * (self.value[index + 1] - self.value[index]),
                self.ground[index] + share * (self.ground[index + 1] - self.ground[index]))

    def potential(self, remaining: float, value_share: float, ground_share: float) -> float:
        """The score expected from a board, from -1 to +1."""
        a, b = self.coefficients(remaining)
        return _clamp(a * (2.0 * value_share - 1.0) + b * (2.0 * ground_share - 1.0))

    def as_dict(self) -> dict:
        return {"knots": list(self.knots), "value": list(self.value), "ground": list(self.ground)}

    @classmethod
    def from_dict(cls, stored: dict) -> "Predictor":
        return cls(knots=stored["knots"], value=stored["value"], ground=stored["ground"])


#: The prediction before any coefficients are fitted: the worth edge.
OPENING = Predictor()


def load(path: str) -> Predictor:
    with open(path, "r", encoding="utf-8") as handle:
        return Predictor.from_dict(json.load(handle))


def save(predictor: Predictor, path: str) -> None:
    from .. import paths

    with paths.replacing(path) as handle:
        json.dump({name: [round(x, 6) for x in values] for name, values in predictor.as_dict().items()}, handle, indent=1)
        handle.write("\n")


@functools.lru_cache(maxsize=1)
def default_predictor() -> Predictor:
    """The adopted coefficients when `PREDICTOR_FILE` is present, and the opening ones otherwise."""
    return load(PREDICTOR_FILE) if os.path.exists(PREDICTOR_FILE) else OPENING


# ---- fitting ---------------------------------------------------------------------------------

@dataclass
class Samples:
    """Boards of recorded matches with the score each match finally got: per board, the remaining seconds, the two shares, the score, the match it was taken in and the group of matches played under one setting."""

    remaining: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    ground: List[float] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    match: List[int] = field(default_factory=list)
    group: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.score)


def samples_of(datasets: Sequence) -> Samples:
    """Every period board of every match that ended with a score, from recorded runs of the operational layer or the economy. A board an operational period paid to several standing decisions is taken once. A match stopped before it ended carries no score and gives nothing, and neither does a board whose remaining time is not known."""
    import numpy as np

    from .dataset import SIGNALS
    from .reward import CLOSED, ELAPSED

    out = Samples()
    match_base = 0
    for dataset in datasets:
        names = SIGNALS[dataset.layer]
        column = {name: names.index(name) for name in ("kind", "remaining_after", "value_after", "ground_after", "score")}
        starts = dataset.signal_starts
        counts = np.diff(starts)
        row_episode = np.repeat(dataset.episode_index(), counts)
        signals = dataset.signals
        kinds = signals[:, column["kind"]].astype(int)
        scores: Dict[int, float] = {}
        for index in np.flatnonzero(kinds == CLOSED):
            scores[int(row_episode[index])] = float(signals[index, column["score"]])
        seen = set()
        for index in np.flatnonzero(kinds == ELAPSED):
            episode = int(row_episode[index])
            row = signals[index]
            if episode not in scores or row[column["remaining_after"]] < 0:
                continue
            key = (episode, float(row[column["remaining_after"]]), float(row[column["value_after"]]),
                   float(row[column["ground_after"]]))
            if key in seen:
                continue
            seen.add(key)
            record = dataset.episodes[episode]
            out.remaining.append(key[1])
            out.value.append(key[2])
            out.ground.append(key[3])
            out.score.append(scores[episode])
            out.match.append(match_base + episode)
            out.group.append(f"{record.get('run_name', '')}|{record.get('map', '')}")
        match_base += len(dataset.episodes)
    return out


@dataclass
class KnotFit:
    knot: float
    samples: int
    matches: int
    value: float
    ground: float
    #: Share of the variance of the score that the held-out prediction explains, over every board of the knot and with each group's mean taken out of both sides first.
    r2: Optional[float]
    r2_within: Optional[float]


def _solve(x, y, ridge: float):
    import numpy as np

    return np.linalg.solve(x.T @ x + ridge * len(y) * np.eye(x.shape[1]), x.T @ y)


def _r2(y, predicted) -> Optional[float]:
    import numpy as np

    spread = float(((y - y.mean()) ** 2).sum())
    return None if spread <= 0 else 1.0 - float(((y - predicted) ** 2).sum()) / spread


def fit(samples: Samples, knots: Sequence[float] = KNOTS, ridge: float = RIDGE, folds: int = FOLDS,
        min_samples: int = MIN_SAMPLES) -> Tuple[Predictor, List[KnotFit]]:
    """Coefficients fitted at each knot on the boards nearest to it in remaining time, with the held-out agreement of each knot. A board whose remaining time is not known is left out."""
    import numpy as np

    remaining = np.asarray(samples.remaining, dtype=float)
    x_all = np.stack([2.0 * np.asarray(samples.value, dtype=float) - 1.0,
                      2.0 * np.asarray(samples.ground, dtype=float) - 1.0], axis=1) if len(samples) else np.zeros((0, 2))
    y_all = np.asarray(samples.score, dtype=float)
    matches = np.asarray(samples.match, dtype=np.int64)
    groups = np.asarray(samples.group, dtype=object)
    knots = [float(k) for k in knots]
    nearest = np.argmin(np.abs(remaining[:, None] - np.asarray(knots)[None, :]), axis=1) if len(samples) else np.zeros(0, int)
    known = remaining >= 0
    value: List[float] = []
    ground: List[float] = []
    reports: List[KnotFit] = []
    for index, knot in enumerate(knots):
        chosen = known & (nearest == index)
        x, y, m, g = x_all[chosen], y_all[chosen], matches[chosen], groups[chosen]
        if len(y) < min_samples:
            value.append(OPENING.value[0])
            ground.append(OPENING.ground[0])
            reports.append(KnotFit(knot, int(len(y)), int(len(set(m.tolist()))), value[-1], ground[-1], None, None))
            continue
        beta = _solve(x, y, ridge)
        value.append(float(beta[0]))
        ground.append(float(beta[1]))
        predicted = np.zeros(len(y))
        fold_of = m % folds
        for fold in range(folds):
            test = fold_of == fold
            if test.all() or not test.any():
                predicted[test] = np.clip(x[test] @ beta, -1.0, 1.0)
                continue
            fold_beta = _solve(x[~test], y[~test], ridge)
            predicted[test] = np.clip(x[test] @ fold_beta, -1.0, 1.0)
        centred_y = y.copy()
        centred_p = predicted.copy()
        for group in set(g.tolist()):
            members = g == group
            centred_y[members] -= y[members].mean()
            centred_p[members] -= predicted[members].mean()
        reports.append(KnotFit(knot, int(len(y)), int(len(set(m.tolist()))), value[-1], ground[-1],
                               _r2(y, predicted), _r2(centred_y, centred_p)))
    return Predictor(knots=knots, value=value, ground=ground), reports
