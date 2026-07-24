"""What holding two operations-arena runs against each other board by board promises, without launching a game.

The paired comparison exists because the arena's board draw moves the side score by more than the arms it is asked to tell apart. Its correctness is entirely arithmetic and bookkeeping — which episode is which board, which episodes may be subtracted from which, and what the subtracted sample says — so all of it is pinned here on synthetic journals. The one thing not checkable here is whether two runs at one seed really do draw the same construction; that is `_arena_seed`'s promise, and the board key is checked against `_arena_seed` itself rather than against a restatement of it.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.eval.sampling import UNBOUNDED_EPISODES, PairedComparison, episodes_for, pairs_for
from rwintel.learn.ops_compare import board_of, compare, signature
from rwintel.learn.ops_run import _arena_seed


class _Session:
    """Only what `_arena_seed` reads of a session: which instance it is and how many episodes it has finished."""

    def __init__(self, instance, finished):
        self.instance = instance
        self.records = [None] * finished


def _entry(instance=0, episode=1, seed=60001, score=0.0, scored=True, arm="ops-learnt",
           map_name="Hills", horizon_ms=300000, radius=400.0, squads=4, pairs=2, board=0):
    return {
        "arm": arm, "instance": instance, "episode": episode,
        "settings": {"map": map_name, "opponents": 1, "difficulty": 1, "credits": 0,
                     "starting_units": 0, "income": 1.0, "fog": 0, "arena": True, "seed": seed},
        "statistics": {"scored": scored, "side_score": score, "board": board, "horizon_ms": horizon_ms,
                       "radius": radius, "squads": squads, "pairs": pairs},
    }


def _journal(tmp_path, name, entries):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def test_the_board_key_is_the_seed_the_runner_would_have_drawn():
    """A journalled episode names its board as exactly the arena seed the run drew it from, so two runs at one base seed pair on the same constructions and two runs at different ones pair on nothing. A journal from before the board was written down has it derived, and that derivation has to keep agreeing with the runner it was copied from."""
    for instance in (0, 1, 7):
        for episode in (1, 2, 10):
            drawn = _arena_seed(60001, _Session(instance, episode - 1))
            assert board_of(_entry(instance=instance, episode=episode, seed=60001)) == drawn

    # Where the board is written down it is read rather than derived, whatever the instance and episode say.
    assert board_of(_entry(instance=3, episode=7, board=4242)) == 4242

    # Different instances and different episodes are always different boards, which is what lets the pairing be a dictionary rather than an order.
    keys = {board_of(_entry(instance=i, episode=e)) for i in range(8) for e in range(1, 21)}
    assert len(keys) == 8 * 20


def test_several_arms_in_one_run_meet_the_same_boards():
    """A run of several arms holds the board still until every arm has played it, which is what makes the run its own paired comparison. With four arms the first four episodes of an instance are one board and the fifth begins the next."""
    boards = [_arena_seed(60001, _Session(0, finished), arms=4) for finished in range(12)]
    assert boards[0:4] == [boards[0]] * 4
    assert boards[4:8] == [boards[4]] * 4
    assert boards[0] != boards[4] != boards[8]
    # One arm is the plain case and still advances every episode.
    assert len({_arena_seed(60001, _Session(0, finished), arms=1) for finished in range(12)}) == 12
    # And two instances never meet: the stride is what keeps their streams apart.
    assert not ({_arena_seed(60001, _Session(0, f), arms=4) for f in range(80)}
                & {_arena_seed(60001, _Session(1, f), arms=4) for f in range(80)})


def test_two_arms_of_one_run_are_paired_out_of_the_one_journal(tmp_path):
    """A multi-arm run writes every arm to one file, and the board it played is written down rather than derived, so the comparison reads two arms out of one journal by name."""
    entries = []
    for board, (first, second) in enumerate([(0.30, 0.20), (0.10, 0.05), (-0.20, -0.30)]):
        entries.append(_entry(instance=0, episode=2 * board + 1, score=first, arm="ops-learnt", board=9000 + board))
        entries.append(_entry(instance=0, episode=2 * board + 2, score=second, arm="ops-pin", board=9000 + board))
    path = _journal(tmp_path, "arms.jsonl", entries)

    comparison = compare(path, first_arm="ops-learnt", second_arm="ops-pin")
    assert comparison is not None and comparison.paired.n == 3
    assert all(abs(seen - wanted) < 1e-12
               for seen, wanted in zip(sorted(comparison.differences), [0.05, 0.10, 0.10]))

    # Read without naming the arms, the same file is one arm that played every board twice, which cannot be paired and must not be guessed at.
    assert compare(path, path) is None
    # And an arm cannot be compared with itself.
    assert compare(path, first_arm="ops-pin", second_arm="ops-pin") is None


def test_an_episode_without_its_instance_or_episode_names_no_board():
    assert board_of({"settings": {"seed": 1}, "instance": 0}) is None
    assert board_of({"settings": {}, "instance": 0, "episode": 1}) is None
    assert board_of(_entry(episode=0)) is None


def test_the_pairing_subtracts_the_two_arms_on_the_boards_they_share(tmp_path):
    """The same boards under two arms are subtracted board by board, and boards only one arm played take no part."""
    first = _journal(tmp_path, "first.jsonl", [
        _entry(instance=0, episode=1, score=0.30),
        _entry(instance=0, episode=2, score=0.10),
        _entry(instance=1, episode=1, score=-0.20),
        _entry(instance=1, episode=2, score=0.00),   # no partner below
    ])
    second = _journal(tmp_path, "second.jsonl", [
        _entry(instance=0, episode=1, score=0.20, arm="ops-pin"),
        _entry(instance=0, episode=2, score=0.05, arm="ops-pin"),
        _entry(instance=1, episode=1, score=-0.30, arm="ops-pin"),
        _entry(instance=2, episode=1, score=0.90, arm="ops-pin"),   # no partner above
    ])
    comparison = compare(first, second)
    assert comparison is not None
    assert comparison.paired.n == 3
    assert all(abs(seen - wanted) < 1e-12
               for seen, wanted in zip(sorted(comparison.differences), [0.05, 0.10, 0.10]))
    # The unpaired means are of the paired boards only: a board the other arm never played cannot be in either mean without putting a board into one side of a difference and not the other.
    assert comparison.first.n == comparison.second.n == 3
    assert abs(comparison.first.mean - (0.30 + 0.10 - 0.20) / 3.0) < 1e-12


def test_an_unscored_episode_is_not_a_board(tmp_path):
    """An episode cut off before its horizon carries no score, so it is not a board either arm played and cannot pair with one."""
    first = _journal(tmp_path, "first.jsonl", [
        _entry(instance=0, episode=1, score=0.30),
        _entry(instance=0, episode=2, score=0.0, scored=False),
    ])
    second = _journal(tmp_path, "second.jsonl", [
        _entry(instance=0, episode=1, score=0.10, arm="ops-pin"),
        _entry(instance=0, episode=2, score=0.40, arm="ops-pin"),
    ])
    comparison = compare(first, second)
    assert comparison is not None and comparison.paired.n == 1
    assert abs(comparison.paired.mean - 0.20) < 1e-12


def test_a_board_journalled_twice_is_dropped_rather_than_picked_from(tmp_path):
    """A journal is appended to, so a second run written to one file puts two episodes on one board. They cannot be told apart afterwards, so the board takes no part rather than one of them being chosen."""
    first = _journal(tmp_path, "first.jsonl", [
        _entry(instance=0, episode=1, score=0.30),
        _entry(instance=0, episode=1, score=-0.90),
        _entry(instance=0, episode=2, score=0.10),
    ])
    second = _journal(tmp_path, "second.jsonl", [
        _entry(instance=0, episode=1, score=0.10, arm="ops-pin"),
        _entry(instance=0, episode=2, score=0.05, arm="ops-pin"),
    ])
    comparison = compare(first, second)
    assert comparison is not None
    assert comparison.paired.n == 1
    assert abs(comparison.paired.mean - 0.05) < 1e-12


def test_two_runs_under_different_instruments_are_not_compared(tmp_path):
    """The catchment radius, the horizon, the staged squads, the contest pairs and the episode settings all change what is being measured. Two runs that disagree on any of them are two arenas, and their difference is not a difference between arms."""
    for changed in ({"radius": 250.0}, {"horizon_ms": 120000}, {"squads": 6}, {"pairs": 3}, {"map_name": "Lake"}):
        first = _journal(tmp_path, "first.jsonl", [_entry(instance=0, episode=e, score=0.1 * e) for e in (1, 2, 3)])
        second = _journal(tmp_path, "second.jsonl",
                          [_entry(instance=0, episode=e, score=0.0, arm="ops-pin", **changed) for e in (1, 2, 3)])
        assert compare(first, second) is None, changed


def test_two_runs_at_different_seeds_share_no_board(tmp_path):
    first = _journal(tmp_path, "first.jsonl", [_entry(instance=0, episode=e, seed=60001) for e in (1, 2, 3)])
    second = _journal(tmp_path, "second.jsonl", [_entry(instance=0, episode=e, seed=70001, arm="ops-pin") for e in (1, 2, 3)])
    assert compare(first, second) is None


def test_the_signature_ignores_the_seed_and_the_arm():
    """Which arm played and which board it drew are what a comparison is made of; they must not be what stops it happening."""
    assert signature(_entry(seed=1, arm="ops-pin")) == signature(_entry(seed=999, arm="ops-learnt"))
    assert signature(_entry(radius=400.0)) != signature(_entry(radius=250.0))


def test_a_paired_difference_resolves_only_when_its_interval_excludes_nought():
    steady = PairedComparison.of([0.5] * 20, [0.1] * 20)
    # No scatter at all is the absence of a measurement of the scatter, not the certainty that there is none — and twenty identical differences do not subtract to exactly nought in binary, so the reading has to survive the rounding rather than test against zero.
    assert abs(steady.paired.mean - 0.4) < 1e-12 and not steady.resolves and steady.needed == UNBOUNDED_EPISODES

    noisy = PairedComparison.of([0.4, -0.4, 0.4, -0.4], [0.0, 0.0, 0.0, 0.0])
    assert noisy.paired.mean == 0.0 and not noisy.resolves

    clear = PairedComparison.of([0.11, 0.09, 0.10, 0.12, 0.08], [0.0] * 5)
    assert clear.resolves and clear.interval < abs(clear.paired.mean)


def test_a_paired_comparison_needs_the_same_boards_on_both_sides():
    try:
        PairedComparison.of([0.1, 0.2], [0.1])
    except ValueError:
        return
    raise AssertionError("a paired comparison of unequal samples is not a pairing and must be refused")


def test_pairing_halves_the_episodes_a_difference_needs():
    """The two-sample sizing pays for both arms' scatter and states its answer per side; the paired one has a single sample of differences. At equal scatter that is half as many episodes, which is the arithmetic argument for running the two arms on the same boards."""
    assert pairs_for(0.12, 0.04) * 2 == episodes_for(0.12, 0.04)
    assert pairs_for(0.12, 0.0) == UNBOUNDED_EPISODES
