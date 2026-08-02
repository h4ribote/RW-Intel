"""Pins the scoring and the sample sizing to the figures the evaluation design was written around.

The sizing tests are against the tables in `docs/system/05-evaluation.md` rather than against a rederivation, because the point of those tables is that the budget of a whole comparison is read off them. If an edit here changes what a difference costs to demonstrate, that is a change of plan and it should show up as a broken test.

The scoring tests use a stand-in episode rather than the real `EpisodeRecord`, so that they exercise the shape of the standing the game emits and nothing else. Scoring takes an episode as data, not as an object of a particular class, and the test holds it to that.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.eval.sampling import (
    Comparison,
    SIGNIFICANCE_POWER_FACTOR,
    Summary,
    UNBOUNDED_EPISODES,
    episodes_for,
    episodes_for_win_rate,
)
from rwintel.eval.scoring import (
    Components,
    OPENING_WEIGHTS,
    Weights,
    components,
    decided,
    fit_weights,
    fitted_on_the_end,
    score,
    viewpoint,
)

#: The standard deviation of the military value edge measured over 24 symmetric episodes.
MEASURED_SIGMA = 0.228


def _team(team: int, value: int = 0, income: int = 0, killed: int = 0, lost: int = 0) -> Dict[str, Any]:
    return {"team": team, "units": 0, "value": value, "income": income, "killed": killed, "lost": lost}


@dataclass
class FakeEpisode:
    """The four things scoring asks an episode for, in the shape the agent reports them."""

    winner: int = -1
    team: int = 0
    timeout: bool = True
    standing: List[Dict[str, Any]] = field(default_factory=list)
    #: Where the sides stood before the end, which is the board the weights are fitted on wherever there is one.
    before: List[Dict[str, Any]] = field(default_factory=list)


# ---- components ----------------------------------------------------------------------


def test_symmetric_standing_scores_zero():
    standing = [_team(0, value=10350, income=42, killed=31, lost=18),
                _team(1, value=10350, income=42, killed=18, lost=31)]
    parts = components(standing, 0)
    assert parts.military == 0.0
    assert parts.economy == 0.0
    # Kills and losses are exactly mirrored between two sides, so the record term's denominator cancels and the ratio is defined to be zero rather than left undefined.
    assert parts.record == 0.0


def test_components_are_ratios_of_the_difference_to_the_sum():
    standing = [_team(0, value=3000, income=60), _team(1, value=1000, income=20)]
    parts = components(standing, 0)
    assert parts.military == 0.5
    assert parts.economy == 0.5
    # From the other side the same board reads as the same size of edge with the opposite sign.
    assert components(standing, 1).military == -0.5


def test_the_strongest_opponent_is_taken_per_quantity():
    """With three teams playing, we are compared against whichever opponent is best at each thing, not against one nominated opponent."""
    standing = [_team(0, value=1000, income=10), _team(1, value=3000, income=10), _team(2, value=1000, income=30)]
    parts = components(standing, 0)
    assert parts.military == (1000 - 3000) / 4000
    assert parts.economy == (10 - 30) / 40


def test_zero_denominators_read_as_no_edge():
    standing = [_team(0), _team(1)]
    parts = components(standing, 0)
    assert (parts.military, parts.economy, parts.record) == (0.0, 0.0, 0.0)
    # Fewer than two teams playing is not a comparison at all, whatever the watchers hold.
    assert components([_team(0, value=5000), _team(-3, value=99)], 0) == Components(0.0, 0.0, 0.0)


def test_a_watching_side_is_scored_as_the_first_contestant():
    """The observations came from nobody, so the standing is read as the first playing team against the rest, which is the convention the episode record's value edge already uses."""
    standing = [_team(0, value=3000), _team(1, value=1000)]
    watching = FakeEpisode(team=-3, standing=standing)
    assert viewpoint(standing, -3) == 0
    assert components(standing, -3).military == 0.5
    assert score(watching) == 0.5
    # And a decision in a watched episode is a win for the side the score was taken from.
    assert score(FakeEpisode(winner=0, timeout=False, team=-3, standing=standing)) == 1.0
    assert score(FakeEpisode(winner=1, timeout=False, team=-3, standing=standing)) == -1.0


# ---- score ---------------------------------------------------------------------------


def test_a_decision_saturates_the_score():
    """Whatever was left standing, a match that was won scores +1 and one that was lost scores -1."""
    hopeless = [_team(0, value=1, income=0, killed=0, lost=99), _team(1, value=50000, income=500, killed=99, lost=0)]
    assert score(FakeEpisode(winner=0, timeout=False, team=0, standing=hopeless)) == 1.0
    assert score(FakeEpisode(winner=1, timeout=False, team=0, standing=hopeless)) == -1.0
    commanding = [_team(0, value=50000, income=500), _team(1, value=1, income=0)]
    assert score(FakeEpisode(winner=1, timeout=False, team=0, standing=commanding)) == -1.0
    assert decided(FakeEpisode(winner=1, timeout=False))
    assert not decided(FakeEpisode(winner=-1))


def test_the_opening_weights_score_on_military_value_alone():
    standing = [_team(0, value=3000, income=10), _team(1, value=1000, income=90)]
    assert (OPENING_WEIGHTS.military, OPENING_WEIGHTS.economy, OPENING_WEIGHTS.record) == (1.0, 0.0, 0.0)
    assert score(FakeEpisode(standing=standing)) == 0.5


def test_the_blend_is_clamped_to_the_range():
    """A record term whose denominator nearly cancels runs far outside -1 to +1, and the clamp is what keeps a score comparable with every other score."""
    standing = [_team(0, killed=10, lost=0), _team(1, killed=0, lost=9)]
    parts = components(standing, 0)
    assert parts.record == 19.0
    on_record = Weights(military=0.0, economy=0.0, record=1.0)
    assert score(FakeEpisode(standing=standing), on_record) == 1.0
    assert score(FakeEpisode(team=1, standing=standing), on_record) == -1.0


# ---- fitting -------------------------------------------------------------------------


def test_fit_weights_is_none_without_a_decided_episode():
    """This is the state the project is actually in: every measured episode timed out, so there is nothing to fit the weights against and the opening choice stands."""
    timeouts = [FakeEpisode(standing=[_team(0, value=v), _team(1, value=10000)]) for v in (8000, 9000, 12000)]
    assert fit_weights(timeouts) is None
    assert fit_weights([]) is None


def test_the_weights_are_fitted_on_the_board_before_the_end():
    """A decided match ends with the loser destroyed, so the board it ends ON is one every weighting calls correctly — fitting there is fitting on a question nobody has to ask. The board taken half a minute earlier is the one the score is actually used on, and it is the one the fit reads wherever an episode carries it.

    Written as the case that separates them: at the end the military edge points at the winner in every episode, while on the earlier board only the economy does. A fit that read the final board would answer military.
    """
    episodes = []
    for _ in range(6):
        episodes.append(FakeEpisode(
            winner=0, timeout=False, team=0,
            # At the end: our side is all that is left, so military points at us whatever else is true.
            standing=[_team(0, value=9000, income=10), _team(1, value=0, income=90)],
            # Before the end: we were behind on the army and ahead on the economy, and the economy is what came true.
            before=[_team(0, value=4000, income=90), _team(1, value=9000, income=10)]))
    fitted = fit_weights(episodes)
    assert fitted is not None
    assert fitted.economy > fitted.military, "the fit read the board the episode ended on, where the answer is already visible"

    # And the report can say which board it read, because the two are not the same evidence.
    with_before, at_the_end = fitted_on_the_end(episodes)
    assert (with_before, at_the_end) == (6, 0)
    assert fitted_on_the_end([FakeEpisode(winner=0, timeout=False, team=0,
                                          standing=[_team(0, value=9000), _team(1, value=0)])]) == (0, 1)


def test_fit_weights_prefers_the_component_that_agrees_with_the_winner():
    """Economy points at the winner in every episode and military points away from it, so the fit puts its weight on economy."""
    records = []
    for income, value, winner in ((90, 1000, 0), (80, 2000, 0), (10, 9000, 1), (20, 8000, 1)):
        standing = [_team(0, value=value, income=income), _team(1, value=5000, income=50)]
        records.append(FakeEpisode(winner=winner, timeout=False, team=0, standing=standing))
    fitted = fit_weights(records)
    assert fitted is not None
    assert fitted.economy == 1.0
    assert (fitted.military, fitted.record) == (0.0, 0.0)


# ---- sample sizing -------------------------------------------------------------------


def test_win_rate_table():
    """The episode counts the design quotes for detecting a win rate difference at five per cent significance and eighty per cent power."""
    assert SIGNIFICANCE_POWER_FACTOR == 7.85
    assert episodes_for_win_rate(0.05) == 1570
    assert episodes_for_win_rate(0.10) == 393
    assert episodes_for_win_rate(0.15) == 175
    assert episodes_for_win_rate(0.20) == 99
    assert episodes_for_win_rate(0.30) == 44


def test_military_edge_table():
    """The same sizing on the measured scatter of the military value edge, which is what makes scoring the board affordable where counting wins is not."""
    assert episodes_for(MEASURED_SIGMA, 0.10) == 82
    assert episodes_for(MEASURED_SIGMA, 0.20) == 21
    # The design's 328 for a difference of 0.05 comes from the standard deviation before it was rounded for print: anything from about 0.2282 upward still rounds to 0.228 and still needs 328, while the printed 0.228 exactly needs 327. The two rows above are unaffected either way.
    assert episodes_for(MEASURED_SIGMA, 0.05) == 327
    assert episodes_for(0.2284, 0.05) == 328
    assert episodes_for(0.2284, 0.10) == 82
    assert episodes_for(0.2284, 0.20) == 21


def test_sizing_rounds_up_and_handles_the_degenerate_cases():
    # 7.85 * 2 * 1 / 1 is 15.7 episodes, which is 16 runs and not 15.
    assert episodes_for(1.0, 1.0) == 16
    # No scatter means one episode a side already settles it; no difference is never settled by any number of them.
    assert episodes_for(0.0, 0.1) == 0
    assert episodes_for(0.228, 0.0) == UNBOUNDED_EPISODES
    assert episodes_for_win_rate(0.0) == UNBOUNDED_EPISODES


# ---- summaries and comparisons -------------------------------------------------------


def test_summary_of_a_sample():
    summary = Summary.of([1.0, 2.0, 3.0, 4.0])
    assert (summary.n, summary.mean) == (4, 2.5)
    # Sample standard deviation, dividing by n-1, which is what the measurement tool reports.
    assert abs(summary.sd - 1.2909944487358056) < 1e-12
    # One episode has a mean but no scatter, and none has neither.
    assert Summary.of([7.0]) == Summary(1, 7.0, 0.0)
    assert Summary.of([]) == Summary(0, 0.0, 0.0)


def test_comparison_sizes_the_difference_it_found():
    first = Summary(n=100, mean=0.20, sd=0.228)
    second = Summary(n=100, mean=0.10, sd=0.228)
    comparison = Comparison.of(first, second)
    assert abs(comparison.difference - 0.10) < 1e-12
    assert abs(comparison.pooled_sd - 0.228) < 1e-12
    # A tenth of a difference at this scatter wants 82 episodes a side, and 100 is more than that.
    assert comparison.needed == 82
    assert comparison.sufficient
    # The same difference on twenty episodes a side is not yet evidence of anything.
    thin = Comparison.of(Summary(20, 0.20, 0.228), Summary(20, 0.10, 0.228))
    assert thin.needed == 82
    assert not thin.sufficient


def test_comparison_of_two_identical_means_is_never_sufficient():
    same = Comparison.of(Summary(500, 0.0, 0.228), Summary(500, 0.0, 0.228))
    assert same.difference == 0.0
    assert same.needed == UNBOUNDED_EPISODES
    assert not same.sufficient


def test_a_difference_with_no_measured_scatter_is_unresolved_not_certain():
    """A pooled scatter of nought is the absence of a measurement, not certainty. One episode a side leaves no within-group scatter to size against, and a handful that happen to be identical leave none either; sizing off that would call a difference established on a single or coincidental pair, which is what a sample size exists to refuse."""
    # One episode each: _pooled_sd has no degrees of freedom and returns nought.
    lone = Comparison.of(Summary(1, 0.5, 0.0), Summary(1, -0.5, 0.0))
    assert lone.difference == 1.0 and lone.pooled_sd == 0.0
    assert lone.needed == UNBOUNDED_EPISODES and not lone.sufficient
    # Several episodes each but each arm internally identical: still no scatter to size against.
    flat = Comparison.of(Summary(5, 0.3, 0.0), Summary(5, -0.3, 0.0))
    assert flat.needed == UNBOUNDED_EPISODES and not flat.sufficient


def test_played_carries_the_settings_a_score_was_produced_under():
    """A score is only the same quantity as another when it was produced the same way. The default journal name distinguishes only the arm and whether an intruder was present, so the settings have to travel with each episode for the report to see when a re-read file has pooled two runs under one arm. The seed is deliberately not among the settings that change what a score means; the map and the difficulty are."""
    from rwintel.eval.__main__ import Played, _signature

    lake = {"map": "Lake", "difficulty": 1, "max_seconds": 300, "seed": 1}
    lake_other_seed = {"map": "Lake", "difficulty": 1, "max_seconds": 300, "seed": 2}
    desert = {"map": "Desert", "difficulty": 1, "max_seconds": 300, "seed": 1}
    assert Played.from_dict({"arm": "script", "settings": lake}).settings == lake
    # Fresh seeds are the point of running many episodes, so they pool; a different map or difficulty or cutoff is a different quantity, so it does not.
    assert _signature(lake) == _signature(lake_other_seed)
    assert _signature(lake) != _signature(desert)
    assert _signature({"map": "Lake", "difficulty": 1}) != _signature({"map": "Lake", "difficulty": 2})
    assert _signature({"map": "Lake", "max_seconds": 300}) != _signature({"map": "Lake", "max_seconds": 900})


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
