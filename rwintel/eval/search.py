"""Searches the script chain's opening values for the setting that scores best, and measures what it found again on episodes the search did not use.

    python -m rwintel.eval.search --count 13 --candidates 16 --rounds 3 --episodes 3 --map Lake --map Beach --map Big --difficulty 0 --intrude
    python -m rwintel.eval.search --count 13 --confirm tune.attack_edge=0.1,tune.push_odds=1.1 --confirm-episodes 12 --map Lake --map Beach --map Big

Candidates are drawn from the ranges each searched value is given (`rwintel.control.policy.tuning`), and the chain as it stands is always among them. They are narrowed by successive halving: every round runs all the survivors as arms of one comparison, interleaved on the same instances, keeps the better half by mean score and doubles the episodes for the next. What wins has been chosen on those episodes and is biased upward by the choosing, so it is then measured against the chain as it stands on a seed no round used, and only that second measurement is a difference to report.

Each round is a run of the evaluation runner under the launcher, as `python -m rwintel.runtime run ... -- eval ...`, writing its episodes to a journal of its own, and the result of the whole search goes to local/reports/search-<stamp>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import paths
from ..control.policy.tuning import NAMES, RANGES, Tuning, describe
from .journal import read
from .sampling import Difference, Summary
from .scoring import score

log = logging.getLogger(__name__)

#: The arm that is the chain as it stands, which every round and the confirmation are measured against.
REFERENCE = "script"

#: Runs a set of arms and answers each arm's episode scores: arms, episodes per arm per instance, and the seed.
Runner = Callable[[Sequence[str], int, int], Dict[str, List[float]]]


def arm_of(tuning: Tuning) -> str:
    """The arm that runs the chain at this setting: the chain itself when nothing differs."""
    written = describe(tuning)
    return f"script:{written}" if written else REFERENCE


def draw(count: int, seed: int) -> List[Tuning]:
    """The chain as it stands and `count` settings drawn from the ranges. Each value is drawn by strata across the candidates, a Latin hypercube, so that every part of every range is covered by some candidate however few there are."""
    rng = random.Random(seed)
    columns: Dict[str, List[float]] = {}
    for name in NAMES:
        low, high = RANGES[name]
        strata = [(index + rng.random()) / count for index in range(count)]
        rng.shuffle(strata)
        columns[name] = [low + share * (high - low) for share in strata]
    drawn = [replace(Tuning(), **{name: round(columns[name][index], 4) for name in NAMES}) for index in range(count)]
    return [Tuning()] + drawn


@dataclass
class Round:
    episodes: int
    seed: int
    #: Arm name to mean score and episode count, in the order the arms ranked.
    ranked: List[Dict[str, object]] = field(default_factory=list)


def halve(candidates: List[Tuning], runner: Runner, rounds: int, episodes: int,
          seed: int) -> Tuple[List[Tuning], List[Round]]:
    """Successive halving: run the survivors, keep the better half by mean score, double the episodes. The chain as it stands is carried to every round whatever it scores, since everything is measured against it."""
    survivors = list(candidates)
    history: List[Round] = []
    for number in range(rounds):
        arms = {arm_of(t): t for t in survivors}
        scores = runner(list(arms), episodes, seed + number)
        ranked = sorted(arms, key=lambda arm: -_mean(scores.get(arm, [])))
        history.append(Round(episodes=episodes, seed=seed + number,
                             ranked=[{"arm": arm, "mean": round(_mean(scores.get(arm, [])), 4),
                                      "n": len(scores.get(arm, []))} for arm in ranked]))
        log.info("round %d, %d episode(s) per arm per instance: %s", number + 1, episodes,
                 "; ".join(f"{arm} {_mean(scores.get(arm, [])):+.3f}" for arm in ranked))
        keep = max(1, math.ceil(len(ranked) / 2))
        kept = ranked[:keep]
        if REFERENCE not in kept:
            kept.append(REFERENCE)
        survivors = [arms[arm] for arm in kept]
        episodes *= 2
    return survivors, history


def confirm(best: Tuning, runner: Runner, episodes: int, seed: int) -> Optional[Difference]:
    """The best setting measured against the chain as it stands on a seed of its own. None when the best is the chain itself, which has nothing to be measured against."""
    arm = arm_of(best)
    if arm == REFERENCE:
        return None
    scores = runner([REFERENCE, arm], episodes, seed)
    return Difference.of(Summary.of(scores.get(arm, [])), Summary.of(scores.get(REFERENCE, [])))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else -math.inf


# ---- running the games -----------------------------------------------------------------------

def launcher(arguments) -> Runner:
    """A runner that plays the arms under the launcher and reads their scores back from the journal it wrote."""
    def run(arms: Sequence[str], episodes: int, seed: int) -> Dict[str, List[float]]:
        journal = os.path.join(paths.episodes(), f"search-{paths.stamp()}-s{seed}.jsonl")
        command = [sys.executable, "-m", "rwintel.runtime", "run", "--count", str(arguments.count),
                   "--offset", str(arguments.offset), "--",
                   "eval", "--port", str(arguments.port), "--episodes", str(episodes), "--seed", str(seed),
                   "--difficulty", str(arguments.difficulty), "--max-seconds", str(arguments.max_seconds),
                   "--record", journal]
        for arm in arms:
            command += ["--arm", arm]
        for name in arguments.map or ["Lake"]:
            command += ["--map", name]
        if arguments.intrude:
            command.append("--intrude")
        log.info("running %d arm(s), %d episode(s) each per instance, seed %d, into %s", len(arms), episodes, seed, journal)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from .__main__ import Played

        scores: Dict[str, List[float]] = {}
        for entry in read(journal):
            played = Played.from_dict(entry)
            scores.setdefault(played.arm, []).append(score(played))
        return scores

    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=13, help="game instances each round runs on")
    parser.add_argument("--offset", type=int, default=0,
                        help="instance directory the rounds start numbering at, for a search beside another run")
    parser.add_argument("--port", type=int, default=8650)
    parser.add_argument("--candidates", type=int, default=16, help="settings drawn beside the chain as it stands")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=3,
                        help="episodes per arm per instance in the first round; a multiple of the number of maps keeps the maps even")
    parser.add_argument("--confirm-episodes", type=int, default=12)
    parser.add_argument("--confirm", default=None,
                        help="skip the search and confirm this setting, written as an arm writes it")
    parser.add_argument("--map", action="append", default=None)
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=8400, help="the search's seed; the confirmation runs at a seed a thousand past it")
    parser.add_argument("--intrude", action="store_true")
    arguments = parser.parse_args(argv)

    log_file = paths.configure_logging("search", fmt="%(asctime)s %(levelname)-7s %(message)s")
    log.info("logging to %s", log_file)
    runner = launcher(arguments)
    history: List[Round] = []
    if arguments.confirm:
        from ..control.policy.options import parse

        best = parse(arguments.confirm).tuning
    else:
        survivors, history = halve(draw(arguments.candidates, arguments.seed), runner, arguments.rounds,
                                   arguments.episodes, arguments.seed)
        best = survivors[0]
    log.info("chosen: %s", arm_of(best))
    difference = confirm(best, runner, arguments.confirm_episodes, arguments.seed + 1000)
    if difference is not None:
        log.info("confirmed on seed %d: %s less %s %+.3f, 95%% interval %+.3f to %+.3f, p %.4f",
                 arguments.seed + 1000, arm_of(best), REFERENCE, difference.difference, difference.low,
                 difference.high, difference.p_value)
    report = {"chosen": arm_of(best), "tuning": {name: getattr(best, name) for name in NAMES},
              "rounds": [vars(r) for r in history],
              "confirmation": None if difference is None else {
                  "seed": arguments.seed + 1000, "difference": round(difference.difference, 4),
                  "low": round(difference.low, 4), "high": round(difference.high, 4),
                  "p_value": round(difference.p_value, 4), "n": difference.first.n}}
    out = os.path.join(paths.reports(), f"search-{paths.stamp()}.json")
    with paths.replacing(out) as handle:
        json.dump(report, handle, indent=1)
    log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
