"""Reads replays, and plays them back under observation.

    python -m rwintel.replay inspect "local/replays/Lake (2p) [v1.15] (30 Sep 2026 05.10.55).replay"
    python -m rwintel.replay inspect <file> --json summary.json
    python -m rwintel.replay verify <file> --log local/logs/run/<stamp>/00.out

`inspect` reads a replay without starting the game and prints what each side ordered, requested and held, up to the end of the match. `verify` sets the decoded commands beside the ones a game logged while playing the same replay back, and exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .. import paths
from .analysis import catalogue_prices, render, summarise
from .container import read_replay
from .playback import DEFAULT_STEPS
from .stream import ReplayFormatError
from .verify import compare, read_log


def inspect(arguments) -> int:
    try:
        replay = read_replay(arguments.replay)
    except (OSError, ReplayFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = summarise(replay, catalogue_prices())
    print(render(summary))
    if replay.unknown:
        print("unknown block(s): " + ", ".join(f"{name} {count}" for name, count in sorted(replay.unknown.items())))
    if arguments.json:
        with paths.replacing(arguments.json) as out:
            json.dump(summary.as_dict(), out, indent=1)
        print(f"wrote {arguments.json}")
    return 0


def verify(arguments) -> int:
    try:
        replay = read_replay(arguments.replay)
        with open(arguments.log, encoding="utf-8", errors="replace") as log:
            comparison = compare(replay.commands, read_log(log))
    except (OSError, ReplayFormatError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for difference in comparison.differences[:arguments.show]:
        print(difference)
    if len(comparison.differences) > arguments.show:
        print(f"... and {len(comparison.differences) - arguments.show} more")
    print(f"{comparison.compared} command(s) compared ({comparison.decoded} decoded, {comparison.logged} logged by the engine), "
          f"{len(comparison.differences)} difference(s)")
    return 0 if comparison.agrees else 1


def play(arguments) -> int:
    from ..control.server import ServerSettings
    from ..data import AssetPaths
    from ..eval.journal import Journal, default_path
    from .playback import Jobs, PlaybackOptions, ReplayServer, describe, load_references, make_job

    log_file = paths.configure_logging("replay", arguments.verbose)
    logging.info("logging to %s", log_file)
    references = load_references(arguments.journal)
    jobs = []
    for path in arguments.replays:
        try:
            jobs.append(make_job(path, viewpoint=arguments.viewpoint, until_seconds=arguments.until,
                                 references=references))
        except (OSError, ValueError) as error:
            logging.error("cannot play %s: %s", path, error)
            return 1
    for job in jobs:
        logging.info("%s: map %s, playing to %.1fs%s", job.stem, job.replay.map_path, job.until_ms / 1000.0,
                     ", set beside its journal record" if job.reference else "")
    options = PlaybackOptions(steps=arguments.steps, omniscient=not arguments.fog,
                              output=arguments.output or paths.replays())
    journal = Journal(arguments.record or default_path("replay"))
    settings = ServerSettings(host=arguments.host, port=arguments.port, instances=min(arguments.instances, len(jobs)),
                              assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
                              journal=journal)
    imitation = None
    if arguments.imitate:
        from ..learn.dataset import run_directory

        imitation = Imitation(arguments.weight, arguments.dataset or run_directory("operations", f"replay-{paths.stamp()}"),
                              journal.path)
    server = ReplayServer(settings, Jobs(jobs), options, imitation.policy if imitation else _watching)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        return 1
    finally:
        journal.close()
        if imitation is not None:
            imitation.write()
    results = [result for session in sessions for result in session.results]
    failed = 0
    for result in results:
        if "failed" in result:
            failed += 1
            logging.error("%s: not played: %s", result["replay"], result["failed"])
            continue
        verdict = result["check"]
        bad = verdict.get("mismatches", 0) != 0 or not verdict.get("checksum", {}).get("agrees", True)
        failed += bad
        logging.info("%s from slot %d: %s; written to %s", result["replay"], result["viewpoint"], describe(verdict),
                     result["directory"])
    logging.info("%d of %d replay(s) played back%s", len(results) - failed, len(jobs),
                 "" if not failed else f", {failed} with a problem")
    return 1 if failed or len(results) < len(jobs) else 0


def _watching(session):
    """The policy of a playback that only records what happened: none, so every observation is answered with nothing."""
    return None


class Imitation:
    """The policy of a playback that turns a person's play into operational decisions: the script chain over the person's side, with the operational layer answered by what the person did. The decisions are recorded as a dataset of the operational layer, each episode as its playback ends."""

    def __init__(self, weight: float, directory: str, journal: str = "") -> None:
        from ..learn.dataset import Recorder
        from ..learn.reward import MATCH_TRACE, OperationalTerms
        from ..learn.rollout import Rollout

        self.weight = weight
        terms = OperationalTerms()
        self.recorder = Recorder(directory, "operations",
                                 {"what": "replay", "argv": sys.argv[1:], "journal": journal,
                                  "behaviour": {"name": "human", "kind": "unknown", "weight": weight},
                                  "terms": terms.as_dict()})
        self.rollout = Rollout(discount=terms.discount, trace=MATCH_TRACE, sink=self.recorder.accept, retain=False)
        self.deciders = []

    def policy(self, session):
        from ..control.policy.operations import REVIEW_MS
        from ..learn.policy import OPERATIONAL, LearningPolicy
        from .human import HumanOperations, OrderBook

        replay = session.job.replay
        decider = HumanOperations(OrderBook.from_commands(replay.commands, session.viewpoint), replay.clock, REVIEW_MS,
                                  weight=self.weight)
        self.deciders.append(decider)
        return LearningPolicy(session, OPERATIONAL, decider, self.rollout, session.instance)

    def write(self) -> None:
        """Seals whatever is still open as stopped and finishes the dataset."""
        self.rollout.cut_all()
        self.rollout.seal_all({"instance": -1, "attempt": 0, "episode": -1, "arm": "replay", "map": "", "seed": -1,
                               "ending": "stopped"})
        self.recorder.close()
        counts: dict = {}
        for decider in self.deciders:
            for basis, count in decider.counts.items():
                counts[basis] = counts.get(basis, 0) + count
        logging.info("inferred %d decision(s) from the person's play: %s", self.recorder.header["decisions"],
                     ", ".join(f"{basis} {count}" for basis, count in counts.items()))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs = command.add_subparsers(dest="what", required=True)
    p = runs.add_parser("inspect", help="summarise a replay without starting the game")
    p.add_argument("replay")
    p.add_argument("--json", default=None, help="also write the summary to this file")
    p = runs.add_parser("verify", help="compare the decoded commands with those a game logged while playing the replay back")
    p.add_argument("replay")
    p.add_argument("--log", required=True, help="the output of a game that played this replay back")
    p.add_argument("--show", type=int, default=20, help="differences to print at most")
    p = runs.add_parser("play", help="play replays back through game instances under observation (the control side of a run)")
    p.add_argument("replays", nargs="+")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8642)
    p.add_argument("--instances", type=int, default=1, help="how many agents to wait for; each plays the next replay in turn")
    p.add_argument("--viewpoint", type=int, default=-1,
                   help="the player slot to observe from; by default the one person in the match")
    p.add_argument("--until", type=float, default=None,
                   help="game seconds to stop at; by default just past the end of the episode the journal records, "
                        "else the end of the match as the replay records it")
    p.add_argument("--journal", action="append", default=[],
                   help="an episode journal holding the recorded matches, to set each playback beside its record; may be given more than once")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"world steps a frame runs per step of elapsed time, default {DEFAULT_STEPS}")
    p.add_argument("--fog", action="store_true", help="observe what the viewpoint's player could see, rather than everything")
    p.add_argument("--imitate", action="store_true",
                   help="run the script chain over the observed side and record the operational decisions the person's orders amount to")
    p.add_argument("--dataset", default=None,
                   help="where --imitate records the decisions, default local/datasets/operations/replay-<stamp>")
    p.add_argument("--weight", type=float, default=1.0, help="multiplies the weight every inferred decision is written with")
    p.add_argument("--output", default=None, help="where each replay's timeline and summary go, default local/replays")
    p.add_argument("--record", default=None, help="journal of the playbacks, default a new file under local/episodes")
    p.add_argument("--assets", default=None)
    p.add_argument("--verbose", action="store_true")
    return command


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if arguments.what == "inspect":
        return inspect(arguments)
    if arguments.what == "verify":
        return verify(arguments)
    if arguments.what == "play":
        return play(arguments)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
