"""Runs the control process.

    python -m rwintel.control --instances 1 --episodes 1 --map Lake --max-seconds 300
    python -m rwintel.control --console --interventions local/interventions.jsonl
    python -m rwintel.control --intrude --episodes 20

Start this first: the agents dial in, and they wait until it is listening.

The console and the intruder are the same thing to everything below them. Both are commanders outside the command chain, both amend the action the chain has just built, and both speak in the chain's own contracts; what separates them is only that one is fed from a keyboard and the other from a random number generator at the rates the design states. Running with `--intrude` is how a measurement is taken under the interference the system will actually be operated under, and running with `--console` and `--interventions` is how a person's play is turned into pairs of a board and a decision taken from it.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys

from ..data import AssetPaths
from ..eval.journal import Journal
from .console import Console
from .intervention import Recorder
from .intruder import interference
from .policy import script_policy
from .pairing import Pairing, probing
from .server import Server, ServerSettings
from .session import EpisodeSettings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--instances", type=int, default=1, help="how many agents to wait for")
    parser.add_argument("--episodes", type=int, default=1, help="episodes each instance runs")
    parser.add_argument("--map", default="Lake", help="substring of a built-in skirmish map's file name")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1, help="-2 very easy to 3 impossible")
    parser.add_argument("--credits", type=int, default=0, help="starting credits by index, 0 is 4000")
    parser.add_argument("--fog", type=int, default=2, help="0 none, 1 basic, 2 line of sight")
    parser.add_argument("--starting-units", type=int, default=1,
                        help="which of the room's starting-unit sets each player begins with, 1 being a single builder")
    parser.add_argument("--arena", action="store_true",
                        help="hold the episode open when only one side has anything standing, which is what a board used to construct engagements on needs")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-seconds", type=int, default=900, help="game time an episode is cut off at")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None,
                        help="write every episode to this file as it finishes, one JSON object per line")
    parser.add_argument("--console", action="store_true",
                        help="read intervention commands from standard input while the run proceeds")
    parser.add_argument("--intrude", action="store_true",
                        help="put the script intruder in, which interferes at the design's stated rates")
    parser.add_argument("--interventions", default=None,
                        help="write every intervention to this file beside the board it was decided from, "
                             "which is the imitation data a human's play produces")
    parser.add_argument("--paired", action="store_true",
                        help="put every connected instance into one shared lockstep match instead of giving each its own, which is the only arrangement in which the engine has a second world to compare its own against")
    parser.add_argument("--match-port", type=int, default=5123,
                        help="the port the hosting instance of a paired match binds")
    parser.add_argument("--spawn-probe", type=int, default=0,
                        help="create units this many times during a paired match, which is what the run exists to test the safety of")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    journal = Journal(arguments.record) if arguments.record else None
    recorder = Recorder(arguments.interventions) if arguments.interventions else None
    # The intruder goes in first and the console after it, because a squad's last word belongs to whoever is consulted last and a person must be able to override the interference, never the other way about.
    outside = [interference(seed=arguments.seed, recorder=recorder)] if arguments.intrude else []

    pairing = Pairing(port=arguments.match_port) if arguments.paired else None
    if arguments.spawn_probe > 0:
        outside.append(probing(arguments.spawn_probe, pairing))

    settings = ServerSettings(
        pairing=pairing,
        host=arguments.host,
        port=arguments.port,
        instances=arguments.instances,
        episodes=arguments.episodes,
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        journal=journal,
        outside=outside,
        episode=EpisodeSettings(
            map=arguments.map,
            opponents=arguments.opponents,
            difficulty=arguments.difficulty,
            credits=arguments.credits,
            fog=arguments.fog,
            starting_units=arguments.starting_units,
            arena=arguments.arena,
            seed=arguments.seed,
            max_seconds=arguments.max_seconds,
        ),
    )

    server = Server(settings, script_policy)
    if arguments.console:
        # The console needs the server to find the session of an instance, and the server reads its list of outside commanders only when an instance dials in, so the console can be appended to that list after the server exists without any of it being missed.
        console = Console(server, recorder=recorder)
        outside.append(console.commander)
        console.start()

    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        return 1
    finally:
        if journal is not None:
            journal.close()
        if recorder is not None:
            recorder.close()

    records = [record for session in sessions for record in session.records]
    logging.info("%d episode(s) over %d instance(s)", len(records), len(sessions))
    if records:
        edges = [record.value_edge for record in records]
        decided = [r for r in records if not r.timeout and r.winner >= 0]
        logging.info("decided %d, timed out %d", len(decided), len(records) - len(decided))
        logging.info("value edge mean %+.3f sd %.3f",
                     statistics.mean(edges), statistics.stdev(edges) if len(edges) > 1 else 0.0)
        interfered = [r for r in records if r.interference.get("touched")]
        if interfered:
            # Said out loud because a score measured under interference is not the same quantity as one measured without it, and because the squads named here are the ones a learning run has to leave out.
            logging.info("interference in %d of %d episode(s), %d squad-episode(s) touched",
                         len(interfered), len(records),
                         sum(len(r.interference["touched"]) for r in interfered))
    for record in records:
        sync = record.synchronisation
        # Only a match shared with another process has the question. One process playing by itself has nothing to compare its world against, so its report would say "no comparison was ever made" about every episode and mean nothing by it.
        if not sync or not sync.get("networked"):
            continue
        peers = sync.get("peers", [])
        broken = [peer for peer in peers if peer.get("desynced") or peer.get("broken")]
        matched = sum(int(peer.get("matched", 0)) for peer in peers) or int(sync.get("matched", 0))
        # The whole point of a paired run. A session that drifted apart does not stop, it simply stops being one match, so an episode that shows a disagreement describes nothing and an episode that shows no comparisons at all proves nothing either.
        logging.info("instance %d episode %d: %s, %d checksum comparison(s) agreed, %s",
                     record.instance, record.episode,
                     "host" if sync.get("host") else "client", matched,
                     "IN STEP" if matched and not broken else
                     ("DESYNCED: " + str(broken) if broken else "no comparison was ever made"))
    for session in sessions:
        logging.info("instance %d handled %d observations", session.instance, session.observations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
