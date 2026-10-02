"""Runs the control process.

    python -m rwintel.control --instances 1 --episodes 1 --map Lake --max-seconds 300
    python -m rwintel.control --console --interventions
    python -m rwintel.control --intrude --episodes 20 --record
    python -m rwintel.control --versus --policy operations:local/models/ops-rl.pt --record

Start this first: the agents dial in, and they wait until it is listening. The log goes to standard error and to local/logs/control, and `--record` and `--interventions` given without a path write under local/episodes and local/interventions.

`--versus` has the one instance host a networked match on `--match-port` for a person to join from their own game client, and `--policy` names what plays against them with any arm name `rwintel.eval` accepts. Each match is recorded to a replay, which the run copies to local/replays when it ends.

The console and the intruder are the same thing to everything below them. Both are commanders outside the command chain, both amend the action the chain has just built, and both speak in the chain's own contracts; what separates them is only that one is fed from a keyboard and the other from a random number generator at the rates the design states. Running with `--intrude` is how a measurement is taken under the interference the system will actually be operated under, and running with `--console` and `--interventions` is how a person's play is turned into pairs of a board and a decision taken from it.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import statistics
from typing import List, Optional, Tuple

from .. import paths
from ..data import AssetPaths
from ..eval import arms
from ..eval.journal import Journal, default_path
from .console import Console
from .intervention import Recorder
from .intruder import interference
from .pairing import PEER_WAIT_SECONDS, Pairing, probing
from .server import Server, ServerSettings
from .session import EpisodeRecord, EpisodeSettings

#: Game time an episode is cut off at when none is given. A match against a person has no cutoff unless one is given.
DEFAULT_MAX_SECONDS = 900


def parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--max-seconds", type=int, default=None,
                        help=f"game time an episode is cut off at, 0 for none; {DEFAULT_MAX_SECONDS} by default, none under --versus")
    parser.add_argument("--policy", default="script",
                        help="what plays, as any arm name rwintel.eval accepts: script, script:<name>=<value>[,...], a posture, "
                             "operations:<path>, operations-greedy:<path>, economy:<path>, economy-greedy:<path>, "
                             "ops-home|ops-nearest|ops-random or eco-script")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", nargs="?", const="", default=None,
                        help="write every episode to this file as it finishes, one JSON object per line; without a path, to a new file under local/episodes")
    parser.add_argument("--console", action="store_true",
                        help="read intervention commands from standard input while the run proceeds")
    parser.add_argument("--intrude", action="store_true",
                        help="put the script intruder in, which interferes at the design's stated rates")
    parser.add_argument("--interventions", nargs="?", const="", default=None,
                        help="write every intervention to this file beside the board it was decided from, "
                             "which is the imitation data a human's play produces; without a path, to a new file under local/interventions")
    parser.add_argument("--paired", action="store_true",
                        help="put every connected instance into one shared lockstep match instead of giving each its own, which is the only arrangement in which the engine has a second world to compare its own against")
    parser.add_argument("--versus", action="store_true",
                        help="have the one instance host a networked match for a person to join from their own game client, on --match-port")
    parser.add_argument("--match-port", type=int, default=5123,
                        help="the port the hosting instance of a paired or versus match binds")
    parser.add_argument("--peer-wait", type=int, default=None,
                        help=f"seconds the host holds its room open for somebody to join, 0 until somebody does; "
                             f"{PEER_WAIT_SECONDS} for --paired, 0 for --versus. A room nobody joined in time stops the run with an error")
    parser.add_argument("--spawn-probe", type=int, default=0,
                        help="create units this many times during a paired match, which is what the run exists to test the safety of")
    parser.add_argument("--verbose", action="store_true")
    return parser


def settle(argv=None) -> Tuple[argparse.Namespace, arms.Arm, Optional[Pairing]]:
    """Parses the command line and settles what depends on the kind of match: the arm that plays, the networked match if there is one, and the cutoff. Exits with the parser's message on a combination that cannot be run."""
    command = parser()
    arguments = command.parse_args(argv)
    if arguments.versus and arguments.paired:
        command.error("--versus and --paired are different matches; give one of them")
    if arguments.versus and arguments.instances != 1:
        command.error(f"a match against a person is hosted by one instance, not {arguments.instances}")
    if arguments.peer_wait is not None and arguments.peer_wait < 0:
        command.error("--peer-wait cannot be negative")
    if arguments.max_seconds is None:
        arguments.max_seconds = 0 if arguments.versus else DEFAULT_MAX_SECONDS
    try:
        arm = arms.parse(arguments.policy)
    except ValueError as error:
        command.error(str(error))
    pairing = None
    if arguments.paired or arguments.versus:
        wait = arguments.peer_wait
        if wait is None:
            wait = 0 if arguments.versus else PEER_WAIT_SECONDS
        pairing = Pairing(port=arguments.match_port, peer_wait=wait)
    return arguments, arm, pairing


def against_a_person(record: EpisodeRecord) -> str:
    """How a match against a person ended, from the side that played it."""
    if record.peer_left:
        return "the player left"
    if record.timeout or record.winner < 0:
        return "cut off undecided"
    return "the policy won" if record.winner == record.team else "the player won"


def keep_replays(sessions, into: Optional[str] = None) -> List[str]:
    """Copies the replay each episode was recorded to out of its instance's directory, where the next preparation of the instances would remove it, and returns where each went.

    A match against a person is the scarce material imitation is made from, and its replay is the one record of it that can be played back. The episode record names the file, so the copy and the record can always be put back together.
    """
    target = into or paths.replays()
    kept: List[str] = []
    for session in sessions:
        directory = session.directory or paths.instance(max(0, session.instance))
        for record in session.records:
            name = (record.replay or {}).get("file")
            source = os.path.join(directory, "replays", name) if name else ""
            if not name or not os.path.exists(source):
                continue
            destination = os.path.join(target, name)
            os.makedirs(target, exist_ok=True)
            shutil.copyfile(source, destination)
            kept.append(destination)
    return kept


def main(argv=None) -> int:
    arguments, arm, pairing = settle(argv)

    log_file = paths.configure_logging("control", arguments.verbose)
    logging.info("logging to %s", log_file)

    journal = None
    if arguments.record is not None:
        journal = Journal(arguments.record or default_path("control"))
        logging.info("recording episodes to %s", journal.path)
    recorder = None
    if arguments.interventions is not None:
        recorder = Recorder(arguments.interventions or os.path.join(paths.interventions(), f"{paths.stamp()}.jsonl"))
        logging.info("recording interventions to %s", recorder.path)
    # The intruder goes in first and the console after it, because a squad's last word belongs to whoever is consulted last and a person must be able to override the interference, never the other way about.
    outside = [interference(seed=arguments.seed, recorder=recorder)] if arguments.intrude else []

    if arguments.spawn_probe > 0:
        outside.append(probing(arguments.spawn_probe, pairing))
    if arguments.versus:
        logging.info("%s plays whoever joins port %d%s", arm[0], arguments.match_port,
                     "" if pairing.peer_wait == 0 else f" within {pairing.peer_wait}s")

    settings = ServerSettings(
        pairing=pairing,
        arms=[arm],
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

    server = Server(settings)
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
    if arguments.versus:
        for record in records:
            logging.info("episode %d against a person: %s after %ds, %s", record.episode,
                         against_a_person(record), record.seconds, record.arm)
        for kept in keep_replays(sessions):
            logging.info("kept the replay %s", kept)
    for session in sessions:
        logging.info("instance %d handled %d observations", session.instance, session.observations)
    failures = server.failures
    for instance, reason in failures:
        logging.error("instance %d could not play: %s", instance, reason)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
