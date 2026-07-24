"""Runs a comparison, or reports one that has already been run.

    python -m rwintel.eval --instances 4 --episodes 3 --arm script --arm arm --map Lake --max-seconds 300
    python -m rwintel.eval --instances 4 --episodes 3 --arm script --intrude
    python -m rwintel.eval --from local/episodes/run.jsonl

Start this first and then the game instances, as with the plain runner. The arms alternate within each instance rather than one being run to completion before the other, because two arms measured in sequence differ by whatever else changed about the machine in between, and the whole point of the comparison is that nothing else changed.

What it prints is the score, its scatter, and how many episodes the difference it found would need to stand up. That last number is the point of the tool. The same settings do not reproduce the same match, so no difference here is ever read off a single episode; the question is only ever whether enough of them have been run, and a run that answers "not yet" has done its job.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..control.intruder import interference
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from . import arms as arm_names
from .journal import Journal, default_path, read
from .sampling import Comparison, Summary, episodes_for, episodes_for_win_rate
from .scoring import OPENING_WEIGHTS, Weights, components, decided, fit_weights, score

#: Differences worth quoting a sample size for. The design's own table, which is what makes a run's report comparable with it.
REPORTED_DIFFERENCES = (0.05, 0.10, 0.20)


@dataclass
class Played:
    """One finished episode, however it reached us: from a session that just ran it or from a journal written earlier."""

    arm: str
    winner: int
    team: int
    timeout: bool
    standing: Sequence[dict]
    seconds: int
    statistics: dict
    #: What anyone outside the chain did to this episode. Carried through the report because a score taken under interference is not the same quantity as one taken without it, and two of them must never be compared as though they were.
    interference: dict = field(default_factory=dict)

    @classmethod
    def of(cls, record) -> "Played":
        return cls(arm=record.arm, winner=record.winner, team=record.team, timeout=record.timeout,
                   standing=record.standing, seconds=record.seconds, statistics=record.statistics,
                   interference=record.interference)

    @classmethod
    def from_dict(cls, entry: dict) -> "Played":
        return cls(arm=entry.get("arm", ""), winner=int(entry.get("winner", -1)),
                   team=int(entry.get("team", -1)), timeout=bool(entry.get("timeout", False)),
                   standing=entry.get("standing", []), seconds=int(entry.get("seconds", 0)),
                   statistics=entry.get("statistics", {}),
                   interference=entry.get("interference", {}))


def report(played: Sequence[Played], weights: Weights) -> None:
    by_arm: Dict[str, List[Played]] = {}
    for episode in played:
        by_arm.setdefault(episode.arm or "script", []).append(episode)

    logging.info("%d episode(s) over %d arm(s), %d decided",
                 len(played), len(by_arm), sum(1 for e in played if decided(e)))
    logging.info("weights: military %.2f economy %.2f record %.2f",
                 weights.military, weights.economy, weights.record)

    summaries: Dict[str, Summary] = {}
    for name, episodes in sorted(by_arm.items()):
        scores = [score(episode, weights) for episode in episodes]
        summary = Summary.of(scores)
        summaries[name] = summary
        parts = [components(e.standing, e.team) for e in episodes]
        logging.info("%-10s n=%-3d score %+.3f sd %.3f  military %+.3f economy %+.3f  decided %d  seconds %d",
                     name, summary.n, summary.mean, summary.sd,
                     _mean(p.military for p in parts), _mean(p.economy for p in parts),
                     sum(1 for e in episodes if decided(e)), _mean(e.seconds for e in episodes))
        _report_layers(episodes)
        _report_interference(episodes)
        for difference in REPORTED_DIFFERENCES:
            logging.info("%-10s   to resolve %.2f: %d episode(s) per side",
                         "", difference, episodes_for(summary.sd, difference))

    names = sorted(summaries)
    for first, second in zip(names, names[1:]):
        comparison = Comparison.of(summaries[first], summaries[second])
        logging.info("%s against %s: %+.3f, pooled sd %.3f, needs %d per side, %s",
                     first, second, comparison.difference, comparison.pooled_sd, comparison.needed,
                     "sufficient" if comparison.sufficient else "NOT yet sufficient")

    if any(decided(e) for e in played):
        fitted = fit_weights(played)
        if fitted is not None:
            logging.info("weights fitted on the decided episodes: military %.2f economy %.2f record %.2f",
                         fitted.military, fitted.economy, fitted.record)
    else:
        logging.info("no episode was decided, so the weights cannot be fitted and stay at the opening choice")
        logging.info("a win rate difference of 0.10 would need %d episode(s) per side, if one were ever observed",
                     episodes_for_win_rate(0.10))


def _report_layers(episodes: Sequence[Played]) -> None:
    """What the layers did, averaged. A score says an arm went badly; this says which layer it went badly in."""
    keys = ("strategic", "operational", "tactical", "contracts", "completed", "stalled", "losing", "expired", "production")
    if not any(episode.statistics for episode in episodes):
        return
    parts = [f"{key} {_mean(e.statistics.get(key, 0) for e in episodes):.0f}" for key in keys]
    logging.info("%-10s   layers: %s, fulfilment %.2f", "", " ".join(parts),
                 _mean(e.statistics.get("fulfilment", 0.0) for e in episodes))


def _report_interference(episodes: Sequence[Played]) -> None:
    """Whether the arm was measured under interference, and how much of it there was.

    The design says evaluation is run with interference too, because a number measured without it is not the number the system will be operated at. That makes the presence of an intruder part of what a score means rather than a detail of how it was produced, so it is printed with the score and not left to be recovered from the settings of the run. It also says which squads a learning run must leave out, and a report that never mentions them invites a comparison between an arm that was interfered with and one that was not.
    """
    disturbed = [episode for episode in episodes if episode.interference.get("touched")]
    if not disturbed:
        logging.info("%-10s   no interference: this is the undisturbed number", "")
        return
    logging.info("%-10s   interference in %d of %d episode(s): %.1f squad(s) and %.1f "
                 "intervention(s) each, on average over those",
                 "", len(disturbed), len(episodes),
                 _mean(len(e.interference["touched"]) for e in disturbed),
                 _mean(len(e.interference.get("events", [])) for e in disturbed))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", default=None,
                        help="report a journal written by an earlier run instead of running anything")
    parser.add_argument("--arm", action="append", default=None,
                        help="an arm of the comparison: 'script', a posture name to pin the strategic layer to, "
                             "or 'ops:<path>' to load a learnt operational layer with the rest of the chain left "
                             "script. Repeatable")
    parser.add_argument("--device", default=None,
                        help="where a learnt arm's network runs. The default is the processor, which at these "
                             "sizes beats the card")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1, help="episodes each arm runs on each instance")
    parser.add_argument("--map", default="Lake")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1, help="-2 very easy to 3 impossible")
    parser.add_argument("--credits", type=int, default=0, help="starting credits by index, 0 is 4000")
    parser.add_argument("--fog", type=int, default=2, help="0 none, 1 basic, 2 line of sight")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-seconds", type=int, default=300, help="game time an episode is cut off at")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None, help="where the episodes are written, one JSON object per line")
    parser.add_argument("--intrude", action="store_true",
                        help="measure with the script intruder present, which is how the design says "
                             "evaluation is to be run: a number taken without interruption is not the "
                             "number the system will be operated at")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if arguments.source:
        played = [Played.from_dict(entry) for entry in read(arguments.source)]
        if not played:
            logging.error("no episodes in %s", arguments.source)
            return 1
        report(played, OPENING_WEIGHTS)
        return 0

    try:
        arms, batchers = arm_names.build_all(arguments.arm or ["script"], device=arguments.device)
    except ValueError as refusal:
        # A named arm that cannot be built — a posture that is not one, a learnt file that is not there, two arms sharing a name — is refused before any game connects, with the reason rather than a traceback. A comparison that quietly measured the wrong thing is worse than one that never started.
        logging.error("%s", refusal)
        return 1
    # The intruder is in the default file name because an interfered-with run and an undisturbed one measure different quantities, and the likeliest way to confuse them is to have written them to the same place.
    run = "-".join(name for name, _ in arms) + ("-intruded" if arguments.intrude else "")
    journal = Journal(arguments.record or default_path(run))
    logging.info("recording to %s", journal.path)

    # One intruder per episode, seeded from the run's seed and the instance, so that every arm of a comparison meets interference drawn the same way. The arms alternate within an instance and the episode index moves the seed on, so no two episodes of a comparison are disturbed identically either.
    outside = [interference(seed=arguments.seed)] if arguments.intrude else []

    settings = ServerSettings(
        host=arguments.host, port=arguments.port, instances=arguments.instances,
        episodes=arguments.episodes, arms=arms, journal=journal, outside=outside,
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        episode=EpisodeSettings(map=arguments.map, opponents=arguments.opponents,
                                difficulty=arguments.difficulty, credits=arguments.credits,
                                fog=arguments.fog, seed=arguments.seed,
                                max_seconds=arguments.max_seconds),
    )

    server = Server(settings)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        return 1
    finally:
        # Stopped whichever way the run ends. A learnt arm's batching server is a background thread holding the network; a run that returned without stopping it would leave it waiting on requests that never come.
        for batcher in batchers:
            batcher.stop()
        journal.close()

    for batcher in batchers:
        logging.info("batched inference averaged %.1f per call", batcher.batch_size)

    played = [Played.of(record) for session in sessions for record in session.records]
    if not played:
        logging.error("no episodes were played")
        return 1
    report(played, OPENING_WEIGHTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
