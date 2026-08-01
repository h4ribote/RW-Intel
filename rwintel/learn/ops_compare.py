"""Holds two operations-arena runs against each other board by board.

    python -m rwintel.learn.ops_compare local/ops-eval-learnt.jsonl local/ops-eval-pin.jsonl

This reads journals; it starts nothing and needs no game. `_arena_seed` is the base seed advanced by the instance and by the board, and a multi-arm run holds the board still until every arm has played it, so a board is named by the seed it was drawn from and every episode writes that name down. Two runs share boards when they were made at the same base seed; two arms of one run share all of them. Subtracting two episodes that name the same board removes the board before any averaging happens.

That subtraction is worth doing but it is not free of the board. One arm's scored episodes scatter by about 0.11 while the differences the arms are being compared for are around 0.04, so an unpaired difference carries about 0.15 of spread; paired, the same two runs carried 0.12, which is the two arms' scores correlating at about 0.36 across the shared boards. **The board is a third of the scatter, not all of it**, because the same board played twice is the same staging, the same garrisons and the same regions but not the same fight — the engine is delta-driven and does not reproduce. So a pair is two plays of one construction rather than two plays of one match, the residual is the fight's own scatter, and that residual is what the reported interval measures. The report says all three spreads and the correlation between them every time, so how much the pairing actually bought is never assumed.

The two runs must be the same instrument. Map, horizon, catchment radius, staged squads, contest pairs, the credits a defender is drawn out of, and which tactical layer did the fighting beneath both sides all change what is being measured, and a run drawn under different ones is a different arena; this refuses to pair across them rather than quietly reporting the change in the instrument as a difference between the arms.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..eval.journal import read
from ..eval.sampling import PairedComparison, UNBOUNDED_EPISODES
from .ops_arena import SCRIPT_TACTICS
from .ops_run import INSTANCE_STRIDE

log = logging.getLogger(__name__)

#: The episode settings that have to agree before two runs are the same instrument, and the arena draw settings that have to agree beside them. The seed is deliberately not among them: two runs at different base seeds simply share no board and pair on nothing, which the pair count says by itself. `tactics` is the last of the draw settings and the least obvious: the fighting beneath an operational choice is what turns a deployment into a share of a disc, so a run made under trained tactical parameters and one made under the handwritten ladder are two arenas whatever else they agree on.
EPISODE_KEYS = ("map", "opponents", "difficulty", "credits", "starting_units", "fog", "income", "arena")
DRAW_KEYS = ("horizon_ms", "radius", "squads", "pairs", "garrison", "tactics")

#: What a draw setting a journal does not carry reads as, where it can only have been one thing. This is a deliberate exception to the rule that a missing field reads as None on both sides, and it is stateable exactly: a record carrying no radius could have been drawn at any radius its runner's flag allowed, whereas a record naming no tactical layer was written by a runner that had no way to put anything but the handwritten one under the arena. Without the exception every journal already written would stop pairing with every new one, which would be refusing over a difference that does not exist.
DRAW_DEFAULTS = {"tactics": SCRIPT_TACTICS}


def board_of(entry: dict) -> Optional[int]:
    """Which board an episode was played on, as the arena seed that drew it.

    An episode written by an arena that records its draw says so outright, and that is the answer. Older journals do not carry it, so it is derived instead the way `ops_run._arena_seed` builds it: the run's base seed, advanced by the instance stride and by however many episodes that instance had already finished. The seed a record carries is the run's, not the episode's — the per-episode advance is sent to the game and never written back into the settings — and the episode number is one-based, so the count of finished episodes at the draw was one less than it. That derivation is only right for a run of a single arm, which is why the field exists and why a multi-arm journal without it is refused rather than derived.
    """
    statistics = entry.get("statistics") or {}
    # Present rather than truthy: the field is the seed the board was drawn from, and nought is as good a seed as any. Asked the truthy way, an episode drawn at seed nought was read as an episode from a journal too old to carry the field at all, and fell through to a derivation that is only right for a single-arm run — so one board of a multi-arm run was paired by a rule the other boards were refused under.
    if statistics.get("board") is not None:
        return int(statistics["board"])
    settings = entry.get("settings") or {}
    if "seed" not in settings or "instance" not in entry or "episode" not in entry:
        return None
    episode = int(entry["episode"])
    if episode < 1:
        return None
    return int(settings["seed"]) + INSTANCE_STRIDE * int(entry["instance"]) + (episode - 1)


def signature(entry: dict) -> Tuple:
    """What has to agree between two runs before their episodes are the same measurement. Anything a journal does not carry reads as None on both sides and so cannot make two runs disagree; it also cannot make them agree, which is why the draw settings were added to the arena statistics rather than left to be assumed. The one exception is a draw setting that could only ever have had one value before it was written down, which reads as that value instead of as nothing."""
    settings = entry.get("settings") or {}
    statistics = entry.get("statistics") or {}
    return (tuple(settings.get(key) for key in EPISODE_KEYS),
            tuple(statistics.get(key, DRAW_DEFAULTS.get(key)) for key in DRAW_KEYS))


def _scored(entries: Sequence[dict]) -> List[dict]:
    return [entry for entry in entries
            if (entry.get("statistics") or {}).get("scored")]


def _by_board(entries: Sequence[dict], name: str) -> Dict[int, dict]:
    """One episode per board. A journal is opened for appending, so a second run written to the same file lands beside the first and a board can appear twice; two episodes claiming one board cannot be told apart afterwards, so both are dropped and said aloud rather than one being picked."""
    counts = Counter(board_of(entry) for entry in entries)
    unkeyed = counts.pop(None, 0)
    if unkeyed:
        log.warning("%s: %d episode(s) carry no instance, episode or seed and cannot be placed on a board", name, unkeyed)
    repeated = sorted(board for board, count in counts.items() if count > 1)
    if repeated:
        log.warning("%s: %d board(s) appear more than once, which is two runs appended to one journal; every repeated board is dropped rather than one of its episodes picked", name, len(repeated))
    dropped = set(repeated)
    return {board_of(entry): entry for entry in entries
            if board_of(entry) is not None and board_of(entry) not in dropped}


def compare(first_path: str, second_path: Optional[str] = None, *, first_arm: Optional[str] = None,
            second_arm: Optional[str] = None, first_name: Optional[str] = None,
            second_name: Optional[str] = None, reading: str = "side_score") -> Optional[PairedComparison]:
    """Two runs, or two arms of one run, paired board by board. None when they share no board or were not the same instrument.

    A run of several arms writes them all to one journal, so the two sides may be one file read twice under two arm names. Naming an arm is what makes that unambiguous: without it a journal of several arms would be read as one arm that played every board several times, which is exactly the repeated board the pairing refuses.

    Which reading the difference is taken on is the caller's, and the two ask different questions of one set of boards: `side_score` is the discs as they stood at the horizon, `side_tenure` the mean of the same discs weighted by how long each reading stood. An episode written before the tenure reading existed carries no such field and reads as nought, which would be a difference of two arms' noughts rather than of the arms; so a comparison asked for a reading the episodes do not carry is refused rather than reported.
    """
    second_path = second_path or first_path
    first_entries = _of_arm(read(first_path), first_arm)
    second_entries = _of_arm(read(second_path), second_arm)
    first_name = first_name or first_arm or _arm_of(first_entries) or first_path
    second_name = second_name or second_arm or _arm_of(second_entries) or second_path
    if first_name == second_name:
        log.error("both sides of the comparison name the same arm of the same run, which would pair every board with itself")
        return None
    log.info("%s: %d episode(s) journalled, %d scored", first_name, len(first_entries), len(_scored(first_entries)))
    log.info("%s: %d episode(s) journalled, %d scored", second_name, len(second_entries), len(_scored(second_entries)))

    first_boards = _by_board(_scored(first_entries), first_name)
    second_boards = _by_board(_scored(second_entries), second_name)
    shared = sorted(set(first_boards) & set(second_boards))
    if not shared:
        log.warning("the two runs share no board: they were run at different base seeds, or on different instances, "
                    "and a paired comparison of them is not a comparison of the arms at all")
        return None

    for name, boards in ((first_name, first_boards), (second_name, second_boards)):
        if not _one_policy(name, boards, shared):
            return None
    signatures = {signature(first_boards[board]) for board in shared} | {signature(second_boards[board]) for board in shared}
    if len(signatures) > 1:
        for entry in sorted(signatures):
            log.error("instrument: episode %s, draw %s (%s)", entry[0], entry[1], ", ".join(DRAW_KEYS))
        log.error("the paired episodes were not run under one instrument — the episode settings, the arena's draw or "
                  "the tactical layer that fought beneath both sides differ — so the difference between them is a "
                  "difference between two arenas and not between two arms; it is not reported")
        return None

    missing = [name for name, boards in ((first_name, first_boards), (second_name, second_boards))
               if any((boards[board]["statistics"] or {}).get(reading) is None for board in shared)]
    if missing:
        log.error("%s: at least one paired episode carries no %s, which was written by an arena that did not "
                  "have that reading; a difference taken over it would be a difference of noughts and is not reported",
                  ", ".join(missing), reading)
        return None
    firsts = [float((first_boards[board]["statistics"] or {}).get(reading, 0.0)) for board in shared]
    seconds = [float((second_boards[board]["statistics"] or {}).get(reading, 0.0)) for board in shared]
    comparison = PairedComparison.of(firsts, seconds)
    report(comparison, first_name, second_name,
           unpaired=(len(first_boards) - len(shared), len(second_boards) - len(shared)))
    return comparison


def _one_policy(name: str, boards: Dict[int, dict], shared: Sequence[int]) -> bool:
    """Whether one side of a comparison was one operational policy throughout.

    The instrument check above asks whether two arms met the same arena; this asks whether each arm was one thing. They are separate questions and cannot be one test: the arms of a run differ in exactly this field and must still pair, so it cannot join the draw settings. What it catches is the other way round — one arm name over two policies. An arm is named by the rule it runs, and a learnt arm by the file its parameters were read from, which is a nickname that changes underneath itself: a training run overwrites whatever its save names, so two runs a week apart write `ops-learnt` into two journals and mean two networks. Pairing those reports the change of policy as a difference between two arms that are the same arm.

    A record that carries no operations field says nothing rather than disagreeing, so journals written before the field existed go on pairing exactly as they did.
    """
    policies = {(boards[board].get("statistics") or {}).get("operations") for board in shared}
    policies.discard(None)
    policies.discard("")
    if len(policies) <= 1:
        return True
    log.error("%s: the paired episodes were played by %d different operational policies (%s), so this is one arm name "
              "over more than one policy and the difference would be the change of policy rather than a difference "
              "between the arms", name, len(policies), ", ".join(sorted(policies)))
    return False


def _of_arm(entries: Sequence[dict], arm: Optional[str]) -> List[dict]:
    """The episodes one arm of a run played, or all of them when no arm was named."""
    return list(entries) if arm is None else [entry for entry in entries if entry.get("arm") == arm]


def _arm_of(entries: Sequence[dict]) -> Optional[str]:
    """The arm a journal's episodes were played as, when they agree on one. A journal that holds several is named by its path instead, since no one name describes it."""
    names = {entry.get("arm") for entry in entries if entry.get("arm")}
    return names.pop() if len(names) == 1 else None


def report(comparison: PairedComparison, first_name: str, second_name: str,
           unpaired: Tuple[int, int] = (0, 0)) -> None:
    """The paired difference, what each arm stood at on its own, and how many pairs the difference would need."""
    log.info("%s alone: %d board(s), %+.4f, sd %.4f", first_name, comparison.first.n, comparison.first.mean, comparison.first.sd)
    log.info("%s alone: %d board(s), %+.4f, sd %.4f", second_name, comparison.second.n, comparison.second.mean, comparison.second.sd)
    if any(unpaired):
        log.info("%d and %d scored board(s) had no partner in the other run and take no part in the difference",
                 unpaired[0], unpaired[1])
    log.info("%s less %s over %d paired board(s): %+.4f, 2 standard errors %.4f, interval %+.4f to %+.4f",
             first_name, second_name, comparison.paired.n, comparison.paired.mean, comparison.interval,
             comparison.paired.mean - comparison.interval, comparison.paired.mean + comparison.interval)
    unpaired_sd = math.sqrt(comparison.first.sd ** 2 + comparison.second.sd ** 2)
    if unpaired_sd <= 0.0:
        return
    moved = 100.0 * (1.0 - comparison.paired.sd / unpaired_sd)
    log.info("the differences scatter by %.4f, against %.4f the same two arms would have scattered by unpaired "
             "(%.4f and %.4f): the pairing %s %.0f per cent of the difference's spread",
             comparison.paired.sd, unpaired_sd, comparison.first.sd, comparison.second.sd,
             "took out" if moved >= 0.0 else "added", abs(moved))
    # How much of the two arms' scatter is shared, read off the three spreads. The pairing can only remove what the two arms have in common, so this says how much of the arena's scatter is the board they both played. It is not bounded below by nought on a finite sample: two arms whose scores happen to run opposite on the boards drawn correlate negatively, the pairing then adds spread rather than removing it, and that is a fact about this sample and not a fault to be hidden by clamping it away.
    product = 2.0 * comparison.first.sd * comparison.second.sd
    if product <= 0.0:
        return
    correlation = max(-1.0, min(1.0, (unpaired_sd ** 2 - comparison.paired.sd ** 2) / product))
    if correlation > 0.0:
        log.info("the two arms' scores correlate at %+.2f across the shared boards, so that much of the scatter is the "
                 "board they shared and the rest is the fight, which one board does not reproduce", correlation)
    else:
        log.info("the two arms' scores correlate at %+.2f across the shared boards, so these boards gave the two arms "
                 "nothing in common for the pairing to remove: on this sample it is costing spread rather than saving "
                 "it, which on few pairs is as likely to be the sample as the arms", correlation)
    if comparison.paired.n < 2:
        log.info("one pair has no spread, so it says nothing about whether the two arms differ")
    elif comparison.resolves:
        log.info("the interval excludes nought, so %s and %s differ on these boards by more than the pairs resolve to",
                 first_name, second_name)
    elif comparison.needed == UNBOUNDED_EPISODES:
        log.info("the differences have no scatter to size against, so nothing is resolved however large the mean looks")
    else:
        log.info("the interval holds nought: %s and %s are not told apart by %d pair(s), and the difference observed "
                 "would need about %d", first_name, second_name, comparison.paired.n, comparison.needed)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", help="a journal written by rwintel.learn.ops_run or ops_train")
    parser.add_argument("second", nargs="?", default=None,
                        help="another one, run at the same seed so the two share boards; left out when both arms are in the first journal")
    parser.add_argument("--first-arm", default=None, help="which arm of the first journal, for a run that holds several")
    parser.add_argument("--second-arm", default=None, help="which arm of the second")
    parser.add_argument("--name-first", default=None, help="what to call the first run in the report")
    parser.add_argument("--name-second", default=None, help="what to call the second")
    parser.add_argument("--reading", choices=("side_score", "side_tenure"), default="side_score",
                        help="which reading of an episode the difference is taken on: the discs as they stood at "
                             "the horizon, or the mean of the same discs weighted by how long each reading stood")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    comparison = compare(arguments.first, arguments.second,
                         first_arm=arguments.first_arm, second_arm=arguments.second_arm,
                         first_name=arguments.name_first, second_name=arguments.name_second,
                         reading=arguments.reading)
    return 0 if comparison is not None else 1


if __name__ == "__main__":
    sys.exit(main())
