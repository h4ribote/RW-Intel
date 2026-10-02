"""What one episode was worth, as a single number from -1 to +1.

A decided match scores what it came to: +1 won, -1 lost. A match cut off by the clock is scored on its board as a prediction of that, so that the two kinds of episode are on one scale. With fitted weights the prediction is the expected outcome under a logistic model, 2 P(win) - 1, whose weights are fitted on boards taken some time before decided matches ended; until such weights are adopted it is the military value edge alone.

Every component is a ratio from -1 to +1, which is what keeps a score comparable across maps and match lengths. Where a component compares us with the others, "theirs" is the strongest opponent in that quantity, taken per quantity rather than by nominating one opponent overall, so that a free-for-all is scored as being ahead of all of them rather than ahead of an average that a weak third party would flatter.

Nothing here imports the control package. The score is a function of what an episode looked like, not of how it was run, so a recorded episode can be rescored from its journal by anything that can produce the same shape of record.
"""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


class Episode(Protocol):
    """What scoring needs an episode to know. `EpisodeRecord` satisfies it structurally, and so does anything reconstructed from a journal."""

    #: The team that won, or negative when the match was cut off with nobody beaten.
    winner: int
    #: The team the observations were taken from, or negative when this side only watched.
    team: int
    timeout: bool
    #: Game seconds the episode lasted.
    seconds: int
    #: One entry per team, carrying `team`, `units`, `value`, `income`, `killed`, `lost` and, from the game builds that send it, `credits`.
    standing: Sequence[Mapping[str, Any]]
    #: Standings taken while the episode ran, each as `second` and `standing`, in the order they were taken. Empty for episodes recorded before they were kept.
    history: Sequence[Mapping[str, Any]]


#: A blend clamped to the score range.
LINEAR = "linear"
#: The expected outcome of a logistic model, tanh(blend / 2), which is 2 P(win) - 1.
LOGISTIC = "logistic"

#: The components in the order a weight vector lists them.
COMPONENTS = ("military", "economy", "exchange", "treasury")


@dataclass(frozen=True)
class Weights:
    """How the board of a cut-off match is turned into a score: a coefficient per component and the link that maps their blend onto the score range."""

    military: float
    economy: float = 0.0
    exchange: float = 0.0
    treasury: float = 0.0
    link: str = LINEAR
    #: How long before the end of a decided match the boards these weights were fitted on were taken, or 0 for weights that were chosen rather than fitted.
    lead_seconds: float = 0.0

    def vector(self) -> Tuple[float, ...]:
        return tuple(getattr(self, name) for name in COMPONENTS)

    def describe(self) -> str:
        coefficients = " ".join(f"{name} {value:+.3f}" for name, value in zip(COMPONENTS, self.vector()))
        fitted = f", fitted {self.lead_seconds:.0f}s before the end" if self.lead_seconds else ""
        return f"{self.link}: {coefficients}{fitted}"


#: The score before any weights are fitted: the military value edge alone, which is measured every episode.
OPENING_WEIGHTS = Weights(military=1.0)

#: The adopted weights, which are the default whenever the file is present.
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")

#: What a decided episode scores, whatever the board looked like.
DECIDED_SCORE = 1.0

#: The range a score is defined on.
SCORE_RANGE = (-1.0, 1.0)

#: How long before the end of a decided match the board a fit is taken on lies. A decided match is over some time before the last defeat is registered, and the board a score is used on is one where both sides are still standing.
DEFAULT_LEAD_SECONDS = 60.0

#: Strength of the penalty on the squared coefficients of a fit. It keeps a set of boards that separates the outcomes perfectly from sending the coefficients to infinity, and leaves a component that never varies at nought.
RIDGE = 1.0

#: Folds of the cross-validation a fit reports its held-out agreement from.
FOLDS = 5

#: The fields that say a team took part in the match. Credits are not among them, because the game hands starting credits to the slots nobody plays from as well.
PRESENCE = ("units", "value", "income", "killed", "lost")


@dataclass(frozen=True)
class Components:
    """The readings a score is blended from, each a ratio from -1 to +1."""

    #: Value of completed units and buildings still standing, ours against the strongest opponent's.
    military: float
    #: Income, ours against the strongest opponent's.
    economy: float
    #: Our own exchange: kills less losses over kills and losses, counting units and buildings alike, and nought before anything has been destroyed.
    exchange: float
    #: Credits held unspent, ours against the strongest opponent's. Nought on records from before the game sent credits.
    treasury: float

    def vector(self) -> Tuple[float, ...]:
        return tuple(getattr(self, name) for name in COMPONENTS)

    def weighted(self, weights: Weights) -> float:
        """The blend before the link is applied."""
        return sum(w * x for w, x in zip(weights.vector(), self.vector()))


def contestants(standing: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The entries for teams that took part in the match.

    The game lists every non-spectator team, including those of slots nobody plays from, and those read nought on everything that says a side was present from the first frame to the last; credits are not among those, since records written before the game stopped counting an absent player's credits carry the starting credits there. A side that took part has at least lost something by the time it has nothing left, so it never reads that way once it has been on the board.
    """
    return [entry for entry in standing
            if int(entry.get("team", -1)) >= 0 and any(float(entry.get(key, 0) or 0) for key in PRESENCE)]


def viewpoint(standing: Sequence[Mapping[str, Any]], team: int) -> int:
    """Which team the score is taken from the perspective of: the team the observations came from, or, when this side only watched, the first contestant. Returns -1 for a watched episode with fewer than two contestants, which is not a scorable one."""
    if team >= 0:
        return team
    playing = contestants(standing)
    if len(playing) < 2:
        return -1
    return int(playing[0].get("team", -1))


def components(standing: Sequence[Mapping[str, Any]], team: int) -> Components:
    """The four ratios, read off one standing from the side of `team`.

    A zero-summing denominator gives 0.0 for that component: a side that has been reduced to nothing scores through the other terms, and a match where nobody has any income yet has no economic edge to report. Fewer than two contestants is no comparison at all and reads 0.0 throughout.
    """
    playing = contestants(standing)
    if len(playing) < 2:
        return Components(0.0, 0.0, 0.0, 0.0)
    if team >= 0:
        ours = next((entry for entry in playing if int(entry.get("team", -1)) == team), {"team": team})
    else:
        ours = playing[0]
    rest = [entry for entry in playing if entry is not ours]
    if not rest:
        return Components(0.0, 0.0, 0.0, 0.0)
    return Components(
        military=_edge(_field(ours, "value"), max(_field(e, "value") for e in rest)),
        economy=_edge(_field(ours, "income"), max(_field(e, "income") for e in rest)),
        exchange=_edge(_field(ours, "killed"), _field(ours, "lost")),
        treasury=_edge(_field(ours, "credits"), max(_field(e, "credits") for e in rest)),
    )


def decided(episode: Episode) -> bool:
    """Whether the match ended with somebody beaten rather than with the clock."""
    return episode.winner >= 0


def outcome(episode: Episode) -> float:
    """+1 when the side the score is taken from won a decided match, -1 when it lost."""
    return DECIDED_SCORE if episode.winner == viewpoint(episode.standing, episode.team) else -DECIDED_SCORE


def board_score(parts: Components, weights: Weights) -> float:
    """What a board is worth under `weights`, from -1 to +1."""
    blended = parts.weighted(weights)
    if weights.link == LOGISTIC:
        return math.tanh(blended / 2.0)
    low, high = SCORE_RANGE
    return min(high, max(low, blended))


def score(episode: Episode, weights: Optional[Weights] = None) -> float:
    """The episode's worth from our side, from -1 to +1: the outcome of a decided match, the board of a cut-off one."""
    if decided(episode):
        return outcome(episode)
    return board_score(components(episode.standing, episode.team), weights or default_weights())


def board_before(episode: Episode, lead_seconds: float) -> Optional[Sequence[Mapping[str, Any]]]:
    """The last standing taken at least `lead_seconds` before the episode ended, or None when none was."""
    limit = episode.seconds - lead_seconds
    found = None
    for entry in getattr(episode, "history", None) or ():
        if float(entry.get("second", 0)) <= limit:
            found = entry.get("standing", [])
    return found


# ---- weights on disk -------------------------------------------------------------------------


def load_weights(path: str) -> Weights:
    with open(path, "r", encoding="utf-8") as handle:
        stored = json.load(handle)
    return Weights(**{name: float(stored.get(name, 0.0)) for name in COMPONENTS},
                   link=str(stored.get("link", LINEAR)), lead_seconds=float(stored.get("lead_seconds", 0.0)))


def save_weights(weights: Weights, path: str) -> None:
    from .. import paths

    stored = {"link": weights.link, "lead_seconds": weights.lead_seconds,
              **{name: round(value, 6) for name, value in zip(COMPONENTS, weights.vector())}}
    with paths.replacing(path) as handle:
        json.dump(stored, handle, indent=1)
        handle.write("\n")


@functools.lru_cache(maxsize=1)
def default_weights() -> Weights:
    """The adopted weights when `WEIGHTS_FILE` is present, and the opening weights otherwise."""
    return load_weights(WEIGHTS_FILE) if os.path.exists(WEIGHTS_FILE) else OPENING_WEIGHTS


# ---- fitting ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    """How well a set of weights calls the winners of decided matches from their boards some time before the end."""

    n: int
    #: Share of the boards whose score has the sign of the outcome. A board scored exactly nought counts as a miss.
    accuracy: float
    #: Mean negative log likelihood of the outcomes under the logistic reading of the blend, or None for linear weights, which make no probability statement.
    log_loss: Optional[float]


@dataclass(frozen=True)
class Fit:
    weights: Weights
    #: Agreement on the boards the weights were fitted on.
    fitted: Agreement
    #: Agreement of weights fitted without each fold on that fold, or None when there are too few boards to fold.
    held_out: Optional[Agreement]

    @property
    def separated(self) -> bool:
        """Whether the weights call every board right. The boards then fix the direction of the weights and not their size: any larger multiple calls them right more confidently, and the size is the penalty's. Such weights do not state a probability of winning."""
        return self.fitted.n > 0 and self.fitted.accuracy == 1.0


def graded(records: Iterable[Episode], lead_seconds: float) -> List[Tuple[Tuple[float, ...], float]]:
    """The decided episodes that carry a board from `lead_seconds` before their end, as that board's components and the outcome, both from the scored side."""
    samples = []
    for record in records:
        if not decided(record):
            continue
        board = board_before(record, lead_seconds)
        side = viewpoint(record.standing, record.team)
        if board is None or side < 0:
            continue
        samples.append((components(board, side).vector(), outcome(record)))
    return samples


def agreement(records: Iterable[Episode], weights: Weights, lead_seconds: float) -> Agreement:
    return _agreement(graded(records, lead_seconds), weights.vector(), weights.link)


def fit_weights(records: Iterable[Episode], lead_seconds: float = DEFAULT_LEAD_SECONDS, folds: int = FOLDS) -> Optional[Fit]:
    """Logistic weights fitted so that the board `lead_seconds` before the end predicts who won, or None when no decided episode carries such a board.

    The model has no intercept, so an even board reads nought whoever the opponent was, and it is odd in the board, so a loss is as informative as a win: a lost match is a won one seen from the other side.
    """
    samples = graded(records, lead_seconds)
    if not samples:
        return None
    beta = _logistic(samples)
    weights = Weights(*beta, link=LOGISTIC, lead_seconds=lead_seconds)
    held_out = None
    if len(samples) >= 2 * folds:
        predictions: List[Tuple[Tuple[float, ...], float, Tuple[float, ...]]] = []
        for fold in range(folds):
            training = [sample for index, sample in enumerate(samples) if index % folds != fold]
            beta_fold = _logistic(training)
            predictions.extend((x, y, beta_fold) for index, (x, y) in enumerate(samples) if index % folds == fold)
        held_out = _agreement_each(predictions, LOGISTIC)
    return Fit(weights=weights, fitted=_agreement(samples, beta, LOGISTIC), held_out=held_out)


def _agreement(samples: Sequence[Tuple[Tuple[float, ...], float]], beta: Sequence[float], link: str) -> Agreement:
    return _agreement_each([(x, y, tuple(beta)) for x, y in samples], link)


def _agreement_each(predictions: Sequence[Tuple[Tuple[float, ...], float, Tuple[float, ...]]], link: str) -> Agreement:
    if not predictions:
        return Agreement(0, 0.0, None)
    hits = 0
    loss = 0.0
    for x, y, beta in predictions:
        margin = y * _dot(beta, x)
        if margin > 0.0:
            hits += 1
        loss += _softplus(-margin)
    n = len(predictions)
    return Agreement(n, hits / n, loss / n if link == LOGISTIC else None)


def _logistic(samples: Sequence[Tuple[Tuple[float, ...], float]]) -> Tuple[float, ...]:
    """Coefficients maximising the penalised likelihood of the outcomes, by Newton's method. The penalised objective is strictly convex, so the iteration converges from nought."""
    size = len(COMPONENTS)
    beta = [0.0] * size
    for _ in range(100):
        gradient = [RIDGE * b for b in beta]
        hessian = [[RIDGE if i == j else 0.0 for j in range(size)] for i in range(size)]
        for x, y in samples:
            p = _sigmoid(_dot(beta, x))
            target = 1.0 if y > 0 else 0.0
            for i in range(size):
                gradient[i] += (p - target) * x[i]
                for j in range(size):
                    hessian[i][j] += p * (1.0 - p) * x[i] * x[j]
        step = _solve(hessian, gradient)
        beta = [b - s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return tuple(beta)


def _solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting. The penalty keeps the matrix positive definite."""
    size = len(vector)
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(column + 1, size):
            factor = rows[r][column] / rows[column][column]
            for c in range(column, size + 1):
                rows[r][c] -= factor * rows[column][c]
    solution = [0.0] * size
    for r in range(size - 1, -1, -1):
        solution[r] = (rows[r][size] - sum(rows[r][c] * solution[c] for c in range(r + 1, size))) / rows[r][r]
    return solution


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    """log(1 + e^z) without overflow."""
    return z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))


def _edge(mine: float, theirs: float) -> float:
    total = mine + theirs
    return (mine - theirs) / total if total else 0.0


def _field(entry: Mapping[str, Any], key: str) -> float:
    return float(entry.get(key, 0) or 0)
