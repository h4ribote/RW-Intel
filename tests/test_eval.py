"""Pins the scoring, the fitting of its weights, the comparison statistics and the sample sizing.

The sizing tests are against the tables in `docs/project/07-evaluation.md` rather than against a rederivation, because the point of those tables is that the budget of a whole comparison is read off them. If an edit here changes what a difference costs to demonstrate, that is a change of plan and it should show up as a broken test.

The scoring tests use a stand-in episode rather than the real `EpisodeRecord`, so that they exercise the shape of the standing the game emits and nothing else. Scoring takes an episode as data, not as an object of a particular class, and the test holds it to that.
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.eval.sampling import (
    Difference,
    SIGNIFICANCE_POWER_FACTOR,
    Summary,
    UNBOUNDED_EPISODES,
    episodes_for,
    episodes_for_win_rate,
    holm,
)
from rwintel.eval.scoring import (
    Components,
    LOGISTIC,
    OPENING_WEIGHTS,
    Weights,
    agreement,
    board_before,
    components,
    contestants,
    decided,
    default_weights,
    fit_weights,
    load_weights,
    save_weights,
    score,
    viewpoint,
)

#: The standard deviation of the military value edge in symmetric matches, rounded as the design prints it.
MEASURED_SIGMA = 0.23


def _team(team: int, value: int = 0, income: int = 0, killed: int = 0, lost: int = 0, credits: int = 0,
          units: int = -1) -> Dict[str, Any]:
    units = (1 if value else 0) if units < 0 else units
    return {"team": team, "units": units, "value": value, "income": income, "killed": killed, "lost": lost,
            "credits": credits}


def _empty(team: int) -> Dict[str, Any]:
    """A slot nobody plays from, as the game reports it: nought on everything but its starting credits."""
    return _team(team, credits=4000)


@dataclass
class FakeEpisode:
    """What scoring asks an episode for, in the shape the agent reports it."""

    winner: int = -1
    team: int = 0
    timeout: bool = True
    standing: List[Dict[str, Any]] = field(default_factory=list)
    seconds: int = 900
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---- contestants and components --------------------------------------------------------------


def test_slots_nobody_plays_from_are_not_contestants():
    """The game lists the team of every slot. A slot nobody plays from reads nought on everything that says a side was present, whatever its credits, and takes no part in any comparison; a side that was destroyed has lost something and still does."""
    standing = [_team(0, value=3000, lost=2), _team(1, lost=5), _empty(2), _empty(3), _team(-3, value=99)]
    assert [entry["team"] for entry in contestants(standing)] == [0, 1]


def test_an_empty_slot_does_not_move_any_component():
    """Kills and losses mirrored between two sides, with the empty slots of a larger map listed beside them, read exactly as they would without the slots."""
    two = [_team(0, value=10350, income=42, killed=31, lost=18), _team(1, value=10350, income=42, killed=18, lost=31)]
    with_slots = two + [_empty(team) for team in range(2, 8)]
    assert components(with_slots, 0) == components(two, 0)
    assert components(with_slots, 1) == components(two, 1)


def test_symmetric_standing_scores_zero_on_the_edges():
    standing = [_team(0, value=10350, income=42, killed=31, lost=31, credits=500),
                _team(1, value=10350, income=42, killed=31, lost=31, credits=500)]
    assert components(standing, 0) == Components(0.0, 0.0, 0.0, 0.0)


def test_components_are_ratios_of_the_difference_to_the_sum():
    standing = [_team(0, value=3000, income=60, credits=300), _team(1, value=1000, income=20, credits=100)]
    parts = components(standing, 0)
    assert parts.military == 0.5
    assert parts.economy == 0.5
    assert parts.treasury == 0.5
    # From the other side the same board reads as the same size of edge with the opposite sign.
    assert components(standing, 1).military == -0.5


def test_the_exchange_is_our_own_kills_against_our_own_losses():
    """In a two-sided match our kills are their losses, so the exchange is read off our own figures, and it means something in a match between two sides."""
    standing = [_team(0, value=5000, killed=30, lost=10), _team(1, value=5000, killed=10, lost=30)]
    assert components(standing, 0).exchange == 0.5
    assert components(standing, 1).exchange == -0.5
    # Nothing destroyed yet is no exchange, not an undefined one.
    assert components([_team(0, value=1), _team(1, value=1)], 0).exchange == 0.0
    # Bounded whatever the counts: losses larger than kills cannot take it past -1.
    assert components([_team(0, value=1, killed=0, lost=9), _team(1, value=1, killed=9)], 0).exchange == -1.0


def test_the_strongest_opponent_is_taken_per_quantity():
    """With three teams playing, we are compared against whichever opponent is best at each thing, not against one nominated opponent."""
    standing = [_team(0, value=1000, income=10), _team(1, value=3000, income=10), _team(2, value=1000, income=30)]
    parts = components(standing, 0)
    assert parts.military == (1000 - 3000) / 4000
    assert parts.economy == (10 - 30) / 40


def test_records_without_credits_read_no_treasury_edge():
    old = [{"team": 0, "units": 1, "value": 3000, "income": 5, "killed": 0, "lost": 0},
           {"team": 1, "units": 1, "value": 1000, "income": 5, "killed": 0, "lost": 0}]
    assert components(old, 0).treasury == 0.0


def test_fewer_than_two_contestants_is_no_comparison():
    assert components([_team(0), _team(1)], 0) == Components(0.0, 0.0, 0.0, 0.0)
    assert components([_team(0, value=5000), _team(-3, value=99), _empty(1)], 0) == Components(0.0, 0.0, 0.0, 0.0)


def test_our_side_is_never_mistaken_for_an_opponent():
    """A side of ours that had nothing and lost nothing is not among the contestants, and is then compared as nothing rather than swapped for the first contestant."""
    standing = [_team(0), _team(1, value=3000), _team(2, value=1000)]
    assert components(standing, 0).military == -1.0


def test_a_watching_side_is_scored_as_the_first_contestant():
    standing = [_empty(0), _team(1, value=3000), _team(2, value=1000)]
    watching = FakeEpisode(team=-3, standing=standing)
    assert viewpoint(standing, -3) == 1
    assert components(standing, -3).military == 0.5
    assert score(watching, OPENING_WEIGHTS) == 0.5
    # And a decision in a watched episode is a win for the side the score was taken from.
    assert score(FakeEpisode(winner=1, timeout=False, team=-3, standing=standing)) == 1.0
    assert score(FakeEpisode(winner=2, timeout=False, team=-3, standing=standing)) == -1.0


# ---- score -------------------------------------------------------------------------------------


def test_a_decision_saturates_the_score():
    """Whatever was left standing, a match that was won scores +1 and one that was lost scores -1."""
    hopeless = [_team(0, value=1, lost=99), _team(1, value=50000, income=500, killed=99)]
    assert score(FakeEpisode(winner=0, timeout=False, team=0, standing=hopeless)) == 1.0
    assert score(FakeEpisode(winner=1, timeout=False, team=0, standing=hopeless)) == -1.0
    assert decided(FakeEpisode(winner=1, timeout=False))
    assert not decided(FakeEpisode(winner=-1))


def test_the_opening_weights_score_on_military_value_alone():
    standing = [_team(0, value=3000, income=10), _team(1, value=1000, income=90)]
    assert OPENING_WEIGHTS.vector() == (1.0, 0.0, 0.0, 0.0)
    assert score(FakeEpisode(standing=standing), OPENING_WEIGHTS) == 0.5


def test_a_linear_blend_is_clamped_and_a_logistic_one_is_the_expected_outcome():
    standing = [_team(0, value=3000, income=60), _team(1, value=1000, income=20)]
    heavy = Weights(military=2.0, economy=1.0)
    assert score(FakeEpisode(standing=standing), heavy) == 1.0
    logistic = Weights(military=2.0, economy=1.0, link=LOGISTIC)
    # 2 P(win) - 1 at a blend of 1.5 is tanh(0.75).
    assert abs(score(FakeEpisode(standing=standing), logistic) - math.tanh(0.75)) < 1e-12
    # An even board is nought under either link.
    even = [_team(0, value=1000), _team(1, value=1000)]
    assert score(FakeEpisode(standing=even), logistic) == 0.0


def test_the_default_weights_are_the_adopted_file_or_the_opening():
    from rwintel.eval import scoring

    expected = load_weights(scoring.WEIGHTS_FILE) if os.path.exists(scoring.WEIGHTS_FILE) else OPENING_WEIGHTS
    assert default_weights() == expected


def test_weights_survive_a_round_trip_through_a_file():
    weights = Weights(military=1.25, economy=-0.5, exchange=0.75, treasury=0.125, link=LOGISTIC, lead_seconds=60.0)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "weights.json")
        save_weights(weights, path)
        assert load_weights(path) == weights


# ---- boards before the end and fitting ------------------------------------------------------------


def _decided(edge: float, won: bool, seconds: int = 400) -> FakeEpisode:
    """A decided match whose board a minute and more before the end had `edge` of the value, ending with the loser wiped out."""
    ours, theirs = 1000 * (1 + edge), 1000 * (1 - edge)
    history = [{"second": second, "standing": [_team(0, value=int(ours), killed=1, lost=1),
                                               _team(1, value=int(theirs), killed=1, lost=1)]}
               for second in range(30, seconds, 30)]
    end = [_team(0, value=5000, killed=9, lost=1), _team(1, lost=9)] if won else \
        [_team(0, lost=9), _team(1, value=5000, killed=9, lost=1)]
    return FakeEpisode(winner=0 if won else 1, timeout=False, team=0, standing=end, seconds=seconds, history=history)


def test_the_board_before_the_end_is_the_last_one_far_enough_from_it():
    episode = _decided(0.2, True, seconds=400)
    at_330 = next(entry for entry in episode.history if entry["second"] == 330)
    assert board_before(episode, 60) is at_330["standing"]
    assert board_before(episode, 10) is episode.history[-1]["standing"]
    assert board_before(episode, 400) is None
    assert board_before(FakeEpisode(winner=0, timeout=False), 60) is None


def test_fit_weights_is_none_without_a_decided_board_before_the_end():
    """The board a decided match ended on is not what the weights are fitted on: the loser has already gone and every component points at the winner. With no earlier board there is nothing to fit."""
    timeouts = [FakeEpisode(standing=[_team(0, value=v), _team(1, value=10000)]) for v in (8000, 9000, 12000)]
    assert fit_weights(timeouts) is None
    assert fit_weights([]) is None
    ended_only = _decided(0.3, True)
    ended_only.history = []
    assert fit_weights([ended_only]) is None


def test_fit_weights_learns_the_component_that_predicts_the_winner():
    """The winner is drawn from the military edge a minute before the end; the fit puts its weight there, calls most matches right, and says so on the matches it was not fitted on."""
    rng = random.Random(7)
    records = []
    for _ in range(300):
        edge = rng.uniform(-0.8, 0.8)
        won = rng.random() < 1.0 / (1.0 + math.exp(-6.0 * edge))
        records.append(_decided(edge, won))
    fit = fit_weights(records, lead_seconds=60)
    assert fit is not None
    assert fit.weights.link == LOGISTIC and fit.weights.lead_seconds == 60
    assert fit.weights.military > 2.0
    # Every board of these matches has the exchange at nought, so the penalty leaves that coefficient at nought.
    assert abs(fit.weights.exchange) < 1e-9
    assert fit.fitted.n == 300 and fit.fitted.accuracy > 0.7
    assert fit.held_out is not None and fit.held_out.n == 300 and fit.held_out.accuracy > 0.7
    # The same boards read back through `agreement` give the fit's own figure.
    assert agreement(records, fit.weights, 60) == fit.fitted


def test_a_loss_is_as_informative_as_a_win():
    """The model is odd in the board, so a set of matches that were all lost still fits, and fits the same direction as the same matches seen from the winners' side."""
    lost = [_decided(-edge, False) for edge in (0.1, 0.3, 0.5, 0.2, 0.4)]
    won = [_decided(edge, True) for edge in (0.1, 0.3, 0.5, 0.2, 0.4)]
    from_losers, from_winners = fit_weights(lost), fit_weights(won)
    assert from_losers is not None and from_winners is not None
    assert abs(from_losers.weights.military - from_winners.weights.military) < 1e-9
    assert from_losers.weights.military > 0.0
    # Every one of these boards already says who wins, so the fit knows the direction and not the size.
    assert from_losers.separated


def test_a_fit_on_boards_that_did_not_already_say_who_would_win_is_not_separated():
    records = [_decided(0.3, True), _decided(0.1, False), _decided(-0.2, False), _decided(-0.1, True), _decided(0.4, True)]
    fit = fit_weights(records)
    assert fit is not None and not fit.separated
    assert fit.weights.military > 0.0


# ---- sample sizing -------------------------------------------------------------------------------


def test_win_rate_table():
    """The episode counts the design quotes for detecting a win rate difference at five per cent significance and eighty per cent power."""
    assert SIGNIFICANCE_POWER_FACTOR == 7.85
    assert episodes_for_win_rate(0.05) == 1570
    assert episodes_for_win_rate(0.10) == 393
    assert episodes_for_win_rate(0.15) == 175
    assert episodes_for_win_rate(0.20) == 99
    assert episodes_for_win_rate(0.30) == 44


def test_military_edge_table():
    """The same sizing on the scatter of the military value edge the design quotes, which is what makes scoring the board affordable where counting wins is not."""
    assert episodes_for(MEASURED_SIGMA, 0.05) == 333
    assert episodes_for(MEASURED_SIGMA, 0.10) == 84
    assert episodes_for(MEASURED_SIGMA, 0.20) == 21


def test_sizing_rounds_up_and_handles_the_degenerate_cases():
    # 7.85 * 2 * 1 / 1 is 15.7 episodes, which is 16 runs and not 15.
    assert episodes_for(1.0, 1.0) == 16
    # No scatter means one episode a side already settles it; no difference is never settled by any number of them.
    assert episodes_for(0.0, 0.1) == 0
    assert episodes_for(0.228, 0.0) == UNBOUNDED_EPISODES
    assert episodes_for_win_rate(0.0) == UNBOUNDED_EPISODES


# ---- summaries and differences ---------------------------------------------------------------------


def test_summary_of_a_sample():
    summary = Summary.of([1.0, 2.0, 3.0, 4.0])
    assert (summary.n, summary.mean) == (4, 2.5)
    # Sample standard deviation, dividing by n-1, which is what the measurement tool reports.
    assert abs(summary.sd - 1.2909944487358056) < 1e-12
    # One episode has a mean but no scatter, and none has neither.
    assert Summary.of([7.0]) == Summary(1, 7.0, 0.0)
    assert Summary.of([]) == Summary(0, 0.0, 0.0)


def test_a_difference_carries_its_interval_and_p_value():
    difference = Difference.of(Summary(n=100, mean=0.20, sd=0.23), Summary(n=100, mean=0.10, sd=0.23))
    assert abs(difference.difference - 0.10) < 1e-12
    assert abs(difference.standard_error - 0.23 * math.sqrt(2 / 100)) < 1e-12
    assert abs(difference.interval - 1.96 * difference.standard_error) < 1e-12
    assert difference.low < 0.10 < difference.high
    # Three standard errors out, the two-sided p-value is about 0.002.
    assert abs(difference.p_value - math.erfc(0.10 / difference.standard_error / math.sqrt(2))) < 1e-12
    assert difference.p_value < 0.01
    # The same difference on twenty episodes a side is not established.
    assert Difference.of(Summary(20, 0.20, 0.23), Summary(20, 0.10, 0.23)).p_value > 0.05


def test_a_difference_with_no_measured_scatter_establishes_nothing():
    """A scatter of nought is one that was never measured rather than one measured to be small."""
    assert Difference.of(Summary(1, 0.5, 0.0), Summary(50, 0.0, 0.2)).p_value == 1.0
    assert Difference.of(Summary(500, 0.0, 0.23), Summary(500, 0.0, 0.23)).p_value == 1.0


def test_holm_steps_down_through_the_smallest_p_values():
    # 0.01 <= 0.05/4, 0.012 <= 0.05/3, 0.03 > 0.05/2 so it and everything larger stand.
    assert holm([0.03, 0.01, 0.2, 0.012]) == [False, True, False, True]
    # One comparison is judged at the level itself.
    assert holm([0.04]) == [True]
    assert holm([]) == []


# ---- the report's grouping -------------------------------------------------------------------------


def _played(arm: str, seed: int = 1, difficulty: int = 0, intruded: bool = False, map_name: str = "Lake"):
    from rwintel.eval.__main__ import Played

    return Played(arm=arm, winner=-1, team=0, timeout=True, standing=[_team(0, value=1), _team(1, value=1)],
                  seconds=900, statistics={}, interference={"intruders": 1, "events": [], "touched": []} if intruded else {},
                  settings={"map": map_name, "difficulty": difficulty, "seed": seed})


def test_maps_are_reported_apart_and_pooled_only_where_more_than_one_was_played():
    from rwintel.eval.__main__ import groups, pooled_over_maps

    played = [_played("script", map_name="Lake"), _played("script", map_name="Beach"),
              _played("script", difficulty=1)]
    assert len(groups(played)) == 3
    pooled = pooled_over_maps(played)
    assert len(pooled) == 1
    (by_arm,) = pooled.values()
    assert {e.settings["map"] for e in by_arm["script"]} == {"Lake", "Beach"}


def test_the_arms_of_a_comparison_meet_the_maps_in_step():
    """Each map is played for one round of the arms before the next, so the arms alternate on the same map."""
    from rwintel.control.session import EpisodeSettings, Session

    session = Session(None, None, EpisodeSettings(maps=["Lake", "Beach"]), arms=[("a", None), ("b", None)], episodes=4)
    played = []
    for _ in range(6):
        played.append((session.episode_map(), session.episode_settings()["map"]))
        session.records.append(None)
    assert [first for first, _ in played] == ["Lake", "Lake", "Beach", "Beach", "Lake", "Lake"]
    assert all(first == second for first, second in played)
    assert "maps" not in session.episode_settings()


class _Wire:
    def sendall(self, data):
        pass

    def close(self):
        pass


class _Policy:
    pass


def test_a_record_says_where_the_match_stood_in_its_game_and_on_which_map():
    """The order comes from the game's own count of matches, so it starts again at 0 when the game is restarted, and the report reads it back from a journal."""
    import json

    from rwintel.control.session import EpisodeSettings, Session
    from rwintel.eval.__main__ import Played

    session = Session(_Wire(), None, EpisodeSettings(map="Beach"), [("script", lambda s: _Policy())], episodes=3)
    session.on_hello(json.dumps({"instance": 4, "unitTypes": []}).encode("utf-8"))
    finished = json.dumps({"event": "finished", "seconds": 60, "standing": []}).encode("utf-8")
    beach = "maps/skirmish/[p2]Beach landing (2p) [by hxyy].tmx"
    for episode in (1, 2, 1):
        session.on_episode(json.dumps({"event": "started", "episode": episode, "map": beach}).encode("utf-8"))
        session.on_episode(finished)
    assert [record.order for record in session.records] == [0, 1, 0]
    entry = session.records[1].as_dict()
    assert entry["order"] == 1 and entry["map"] == "[p2]Beach landing (2p) [by hxyy].tmx"
    assert Played.from_dict(entry).order == 1 and Played.of(session.records[0]).order == 0
    assert Played.from_dict(entry).map == entry["map"]

    older = Played.from_dict({"arm": "script", "settings": {"map": "Lake"}})
    assert (older.order, older.map) == (-1, "Lake")

    from rwintel.learn.layers import episode_of

    # A recorded dataset keeps the same two facts in each episode's metadata.
    assert episode_of(session)["order"] == 0 and episode_of(session)["map"] == beach


def test_the_report_splits_each_map_into_first_and_later_matches():
    from rwintel.eval.__main__ import openings
    from rwintel.eval.scoring import default_weights

    played = []
    for order, map_name in ((0, "Lake"), (1, "Lake"), (2, "Lake"), (0, "Beach"), (-1, "Beach")):
        episode = _played("script", map_name=map_name)
        episode.order, episode.map = order, map_name
        played.append(episode)
    rows = {(arm, name, place): count for arm, name, place, count, _ in openings(played, default_weights())}
    assert rows == {("script", "Lake", "first"): 1, ("script", "Lake", "later"): 2,
                    ("script", "Beach", "first"): 1, ("script", "Beach", "unknown"): 1}


def test_episodes_differing_only_in_seed_are_one_measurement():
    from rwintel.eval.__main__ import groups

    grouped = groups([_played("script", seed=1), _played("script", seed=2), _played("arm", seed=2)])
    assert len(grouped) == 1
    (by_arm,) = grouped.values()
    assert list(by_arm) == ["script", "arm"] and len(by_arm["script"]) == 2


def test_episodes_under_other_settings_or_interference_are_never_pooled():
    from rwintel.eval.__main__ import describe_conditions, groups

    grouped = groups([_played("script", difficulty=0), _played("script", difficulty=1),
                      _played("script", difficulty=0, intruded=True)])
    assert len(grouped) == 3
    labels = sorted(describe_conditions(list(grouped)).values())
    assert labels == ["[difficulty=0 intruded]", "[difficulty=0 undisturbed]", "[difficulty=1 undisturbed]"]


def test_a_match_that_fell_out_of_step_is_left_out():
    from rwintel.eval.__main__ import out_of_step

    episode = _played("script")
    assert not out_of_step(episode)
    episode.synchronisation = {"networked": True, "peers": [{"desynced": True}]}
    assert out_of_step(episode)
    episode.synchronisation = {"networked": True, "peers": [{"matched": 4}]}
    assert not out_of_step(episode)


# ---- arms ------------------------------------------------------------------------------


def test_arms_name_what_they_run():
    """A learnt operational arm is named after its parameter file, so that the journal and the report say which network played; a file that is not there is refused before any game is started."""
    import torch

    from rwintel.eval.arms import parse, parse_all
    from rwintel.learn.net import OperationalNet

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "ops-bc.pt")
        torch.save(OperationalNet().state_dict(), path)
        assert parse(f"operations:{path}")[0] == "operations-ops-bc"
        assert parse(f"operations-greedy:{path}")[0] == "operations-greedy-ops-bc"
        refused = ""
        try:
            parse(f"operations:{os.path.join(folder, 'missing.pt')}")
        except ValueError as error:
            refused = str(error)
        assert "no parameters" in refused
    assert [name for name, _ in parse_all(["script", "ops-home", "ops-nearest", "ops-random", "arm"])] == [
        "script", "ops-home", "ops-nearest", "ops-random", "arm"]


def test_a_learnt_economy_arm_is_named_after_its_file_and_its_judge_has_an_arm_of_its_own():
    import torch

    from rwintel.eval.arms import economy_script, parse
    from rwintel.learn.net import EconomicNet

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "eco-bc.pt")
        torch.save(EconomicNet().state_dict(), path)
        assert parse(f"economy:{path}")[0] == "economy-eco-bc"
        assert parse(f"economy-greedy:{path}")[0] == "economy-greedy-eco-bc"
        refused = ""
        try:
            parse(f"economy:{os.path.join(folder, 'missing.pt')}")
        except ValueError as error:
            refused = str(error)
        assert "no parameters" in refused
    assert parse("eco-script") == ("eco-script", economy_script)


def test_a_script_arm_with_switched_rules_is_named_after_its_switches():
    from rwintel.control.policy.options import Options
    from rwintel.eval.arms import parse

    name, build = parse("script:frontier=off,tech=all")
    assert name == "script:frontier=off,tech=all"
    captured = {}

    def _record(session, options):
        captured["options"] = options

    import rwintel.eval.arms as arms
    original = arms.script_policy
    arms.script_policy = _record
    try:
        build(None)
    finally:
        arms.script_policy = original
    assert captured["options"] == Options(frontier=False, tech="all")
    for bad in ("script:nothing=on", "script:tech=maybe"):
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_an_arm_is_named_without_being_built_as_it_is_journalled():
    from rwintel.eval.arms import name_of, parse

    assert name_of("operations:local/models/teacher-ops-iql-j1.pt") == "operations-teacher-ops-iql-j1"
    assert name_of("economy-greedy:m/eco.pt") == "economy-greedy-eco"
    for written in ("script", "script:baseline", "ARM", "ops-home", "eco-script"):
        assert name_of(written) == parse(written)[0]


def test_a_report_from_journals_compares_against_the_first_arm_the_run_was_started_with():
    import json

    from rwintel.eval import __main__ as runner

    seen = []
    saved_report, saved_local = runner.report, os.environ.get("RWINTEL_LOCAL")
    with tempfile.TemporaryDirectory() as area:
        os.environ["RWINTEL_LOCAL"] = area
        journal = os.path.join(area, "j.jsonl")
        with open(journal, "w") as handle:
            for arm in ("script", "operations-a"):
                handle.write(json.dumps({"arm": arm, "settings": {"map": "Lake"}}) + "\n")
        runner.report = lambda played, weights, reference, *rest: seen.append(reference)
        try:
            assert runner.main(["--from", journal, "--arm", "operations:m/a.pt", "--arm", "script"]) == 0
            assert runner.main(["--from", journal, "--arm", "script", "--reference", "operations-a"]) == 0
            assert runner.main(["--from", journal]) == 0
        finally:
            runner.report = saved_report
            if saved_local is None:
                os.environ.pop("RWINTEL_LOCAL", None)
            else:
                os.environ["RWINTEL_LOCAL"] = saved_local
    assert seen == ["operations-a", "operations-a", None]


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
