"""What one episode was worth, as a single number from -1 to +1.

A win or a loss would be the natural thing to score on, and it is unusable: built-in AI against built-in AI produced no decision at all across the measured episodes, so a win rate has no observations to be estimated from. The score here is therefore a continuous reading of the board at the cut-off, and a decision, when one does happen, saturates it rather than being blended into it — a match that was actually won is not more or less won depending on how much armour was left standing.

Every component is a ratio of the form (ours - theirs) / (ours + theirs), which is what keeps a score comparable across maps and match lengths: a long match on a large map produces larger absolute figures on both sides and the same ratio. "Theirs" is the strongest opponent in that quantity, taken per quantity rather than by nominating one opponent overall, so that a free-for-all is scored as being ahead of all of them rather than ahead of an average that a weak third party would flatter.

The weights are deliberately not settled here. The design fixes how they are to be decided — fitted so that the score's sign agrees with the winner on episodes that were decided — and there are no decided episodes yet, so the opening choice is the military term alone. `fit_weights` implements the fitting for the day the episodes exist; today it returns None on every set it is handed, which is the honest answer rather than a fabricated fit.

Nothing here imports the control package. The score is a function of what an episode ended up looking like, not of how it was run, and keeping the dependency out means a recorded episode can be rescored from a log by anything that can produce the same shape of standing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


class Episode(Protocol):
    """What scoring needs an episode to know. `EpisodeRecord` satisfies it structurally, and so does anything reconstructed from a log."""

    #: The team that won, or negative when the match was cut off with nobody beaten.
    winner: int
    #: The team the observations were taken from, or negative when this side only watched.
    team: int
    timeout: bool
    #: One entry per team, carrying at least `team`, `value`, `income`, `killed` and `lost`.
    standing: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class Weights:
    """How much of each component the score is made of. Non-negative and summing to one, so that the blended score stays inside the range each component already lives in."""

    military: float
    economy: float
    record: float


#: The opening choice, and only that. The design says the weights are to be fitted against episodes that were decided, and no episode has yet been decided — built-in AI against built-in AI times out even with five difficulty steps between the two sides. Until a population of script policies produces decisions to fit against, the score is the military value edge alone, which is the one component that is measured every episode and whose scatter is known.
OPENING_WEIGHTS = Weights(military=1.0, economy=0.0, record=0.0)

#: Resolution of the search `fit_weights` runs. Coarse on purpose: the fit is judged on how many decided episodes get the right sign out of a few hundred at most, and a grid finer than this splits candidates that the data cannot tell apart.
WEIGHT_GRID_STEP = 0.05

#: What a decided episode scores, whatever the board looked like. The score is meant to predict the winner, so an observed winner overrides the prediction outright.
DECIDED_SCORE = 1.0

#: The range a score is defined on. The components are ratios that normally sit inside it, but a blend of several, or a ratio whose denominator nearly cancelled, can leave it.
SCORE_RANGE = (-1.0, 1.0)


@dataclass(frozen=True)
class Components:
    """The three readings the score is blended from, each already a ratio from -1 to +1 in the ordinary case."""

    #: Value of completed units still standing, ours against the strongest opponent's.
    military: float
    #: Income, ours against the strongest opponent's.
    economy: float
    #: Kills less losses, ours against the strongest opponent's. In a two-sided match our kills are their losses exactly, so this term's denominator cancels to zero and the term reads 0.0; it only carries information with a third party in the match, or once losses are valued rather than counted.
    record: float

    def weighted(self, weights: Weights) -> float:
        """The blend before clamping. Exposed because the fit compares candidate weights by how far on the right side of zero they put an episode, which is a quantity the clamp would flatten."""
        return weights.military * self.military + weights.economy * self.economy + weights.record * self.record


def viewpoint(standing: Sequence[Mapping[str, Any]], team: int) -> int:
    """Which team the score is taken from the perspective of.

    Normally that is the team the observations came from. When this side only watched, there is no "ours" in the standing at all, and the sensible reading is the first contestant against the rest — the same choice `EpisodeRecord.value_edge` makes, so that a watched episode and a played one are scored on the same convention. Returns -1 when fewer than two teams were playing, which is not a scorable episode.
    """
    playing = _playing(standing)
    if len(playing) < 2:
        return -1
    ours = _ours(playing, team)
    return int(ours.get("team", -1))


def components(standing: Sequence[Mapping[str, Any]], team: int) -> Components:
    """The three ratios, read off the standing the episode ended with.

    A missing or zero-summing denominator gives 0.0 for that component rather than an error: a side that has been reduced to nothing scores through the military term, and a match where neither side has any income yet genuinely has no economic edge to report.
    """
    playing = _playing(standing)
    if len(playing) < 2:
        return Components(0.0, 0.0, 0.0)
    ours = _ours(playing, team)
    rest = [entry for entry in playing if entry is not ours]
    return Components(
        military=_edge(_value(ours), max(_value(e) for e in rest)),
        economy=_edge(_income(ours), max(_income(e) for e in rest)),
        record=_edge(_record(ours), max(_record(e) for e in rest)),
    )


def decided(episode: Episode) -> bool:
    """Whether the match ended with somebody beaten rather than with the clock. `timeout` and a winner are meant to be exclusive, and a winner is the stronger statement, so it is the one that decides."""
    return episode.winner >= 0


def score(episode: Episode, weights: Weights = OPENING_WEIGHTS) -> float:
    """The episode's worth from our side, from -1 to +1.

    A decision saturates it: +1 if the winner is the side the score is taken from, -1 otherwise. Only a cut-off match is scored on the board, and then the blend is clamped, because the components are ratios that a nearly cancelling denominator can throw outside the range.
    """
    if decided(episode):
        return DECIDED_SCORE if episode.winner == viewpoint(episode.standing, episode.team) else -DECIDED_SCORE
    low, high = SCORE_RANGE
    return min(high, max(low, components(episode.standing, episode.team).weighted(weights)))


def fit_weights(records: Iterable[Episode]) -> Optional[Weights]:
    """Weights fitted so that the board score predicts the winner, or None when there is nothing to fit against.

    None is what this returns today, and will keep returning until a match is actually won by somebody: every episode measured so far timed out. That is the whole reason the opening weights are a choice rather than a result, and returning None rather than a plausible-looking triple is what keeps that visible.

    The fit is a search over a coarse grid of non-negative weights summing to one. A candidate is judged on how many decided episodes it puts on the right side of zero, ties going to the candidate that does it by the wider margin — between two candidates that are right equally often, the one that is right less narrowly is the one more likely to stay right on the next episode.

    One limit is worth knowing when reading a fitted result. The design asks for the score *just before* the decision, and what is recorded is the standing the episode ended with, which for a decided match is taken after the loser has already been destroyed. The fit is therefore against a board that is easier to call than the one the score is meant to be used on, and the weights it produces should be checked against cut-off episodes rather than trusted from the fit alone.
    """
    decided_records = [record for record in records if decided(record)]
    if not decided_records:
        return None

    graded: List[Tuple[Components, float]] = []
    for record in decided_records:
        outcome = 1.0 if record.winner == viewpoint(record.standing, record.team) else -1.0
        graded.append((components(record.standing, record.team), outcome))

    best: Optional[Weights] = None
    best_key: Tuple[int, float] = (-1, 0.0)
    for candidate in _grid():
        correct = 0
        margin = 0.0
        for parts, outcome in graded:
            blended = parts.weighted(candidate)
            if blended * outcome > 0.0:
                correct += 1
            margin += blended * outcome
        key = (correct, margin)
        if key > best_key:
            best, best_key = candidate, key
    return best


# ---- internals -----------------------------------------------------------------------


def _grid() -> Iterable[Weights]:
    """Every non-negative triple on the grid that sums to one. Generated in a fixed order so that a tie between two genuinely equivalent candidates resolves the same way on every run."""
    steps = int(round(1.0 / WEIGHT_GRID_STEP))
    for military, economy in itertools.product(range(steps + 1), repeat=2):
        if military + economy > steps:
            continue
        record = steps - military - economy
        yield Weights(military * WEIGHT_GRID_STEP, economy * WEIGHT_GRID_STEP, record * WEIGHT_GRID_STEP)


def _playing(standing: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The entries for teams that were actually playing. Watchers and unused slots carry a negative team number and are not part of anybody's comparison."""
    return [entry for entry in standing if int(entry.get("team", -1)) >= 0]


def _ours(playing: List[Mapping[str, Any]], team: int) -> Mapping[str, Any]:
    """The entry the score is taken from, falling back to the first contestant when this side only watched."""
    return next((entry for entry in playing if int(entry.get("team", -1)) == team), playing[0])


def _edge(mine: float, theirs: float) -> float:
    total = mine + theirs
    return (mine - theirs) / total if total else 0.0


def _value(entry: Mapping[str, Any]) -> float:
    return float(entry.get("value", 0))


def _income(entry: Mapping[str, Any]) -> float:
    return float(entry.get("income", 0))


def _record(entry: Mapping[str, Any]) -> float:
    """Kills less losses, counting units and buildings alike. Counts rather than credits, because that is what the game keeps per player."""
    return float(entry.get("killed", 0)) - float(entry.get("lost", 0))
