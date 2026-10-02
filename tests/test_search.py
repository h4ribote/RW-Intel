"""The search over opening values: candidates cover the ranges, halving keeps the best and the chain as it stands, and the winner is measured again on its own."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy.options import parse
from rwintel.control.policy.tuning import NAMES, RANGES, Tuning
from rwintel.eval.search import REFERENCE, arm_of, confirm, draw, halve


def test_the_candidates_are_the_chain_and_draws_that_cover_every_range():
    candidates = draw(8, seed=1)
    assert candidates[0] == Tuning() and len(candidates) == 9
    for name in NAMES:
        low, high = RANGES[name]
        values = sorted(getattr(c, name) for c in candidates[1:])
        assert all(low <= v <= high for v in values)
        # One draw per eighth of the range: the lowest sits in the first eighth and the highest in the last.
        assert values[0] <= low + (high - low) / 8 and values[-1] >= high - (high - low) / 8


def test_an_arm_written_for_a_candidate_reads_back_as_that_candidate():
    for candidate in draw(4, seed=2):
        written = arm_of(candidate)
        assert written == REFERENCE or parse(written[len("script:"):]).tuning == candidate


def test_halving_keeps_the_better_half_with_the_chain_and_doubles_the_episodes():
    candidates = draw(7, seed=3)
    runs = []

    def runner(arms, episodes, seed):
        runs.append((len(arms), episodes, seed))
        # Scores rise with the attack edge, so the candidate with the highest one wins.
        return {arm: [parse(arm[len("script:"):]).tuning.attack_edge if arm != REFERENCE else Tuning().attack_edge]
                for arm in arms}

    survivors, history = halve(candidates, runner, rounds=3, episodes=3, seed=100)
    assert [e for _, e, _ in runs] == [3, 6, 12] and [s for _, _, s in runs] == [100, 101, 102]
    # Each round keeps half, rounded up, and the chain as it stands if it was not among them.
    for (before, _, _), (after, _, _) in zip(runs, runs[1:]):
        assert (before + 1) // 2 <= after <= (before + 1) // 2 + 1
    best = max(candidates, key=lambda c: c.attack_edge)
    assert survivors[0] == best and Tuning() in survivors
    assert history[0].ranked[0]["arm"] == arm_of(best)


def test_the_winner_is_measured_against_the_chain_and_the_chain_is_not_measured_against_itself():
    winner = parse("tune.attack_edge=0.4").tuning

    def runner(arms, episodes, seed):
        assert set(arms) == {REFERENCE, arm_of(winner)} and seed == 7
        return {REFERENCE: [0.0, 0.1, -0.1, 0.0], arm_of(winner): [0.3, 0.4, 0.2, 0.3]}

    difference = confirm(winner, runner, episodes=4, seed=7)
    assert difference is not None and abs(difference.difference - 0.3) < 1e-9
    assert confirm(Tuning(), runner, episodes=4, seed=7) is None
