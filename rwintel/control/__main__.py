"""Runs the control process.

    python -m rwintel.control --instances 1 --episodes 1 --map Lake --max-seconds 300

Start this first: the agents dial in, and they wait until it is listening.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys

from ..data import AssetPaths
from ..eval.journal import Journal
from .policy import script_policy
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
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-seconds", type=int, default=900, help="game time an episode is cut off at")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None,
                        help="write every episode to this file as it finishes, one JSON object per line")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    journal = Journal(arguments.record) if arguments.record else None
    settings = ServerSettings(
        host=arguments.host,
        port=arguments.port,
        instances=arguments.instances,
        episodes=arguments.episodes,
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        journal=journal,
        episode=EpisodeSettings(
            map=arguments.map,
            opponents=arguments.opponents,
            difficulty=arguments.difficulty,
            credits=arguments.credits,
            fog=arguments.fog,
            seed=arguments.seed,
            max_seconds=arguments.max_seconds,
        ),
    )

    server = Server(settings, script_policy)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        return 1
    finally:
        if journal is not None:
            journal.close()

    records = [record for session in sessions for record in session.records]
    logging.info("%d episode(s) over %d instance(s)", len(records), len(sessions))
    if records:
        edges = [record.value_edge for record in records]
        decided = [r for r in records if not r.timeout and r.winner >= 0]
        logging.info("decided %d, timed out %d", len(decided), len(records) - len(decided))
        logging.info("value edge mean %+.3f sd %.3f",
                     statistics.mean(edges), statistics.stdev(edges) if len(edges) > 1 else 0.0)
    for session in sessions:
        logging.info("instance %d handled %d observations", session.instance, session.observations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
