"""Runs a training pass, or collects what the script chain does as data to start one from.

    python -m rwintel.learn tactics    --instances 4 --max-seconds 1800 --save local/tactics.pt
    python -m rwintel.learn operations --instances 4 --episodes 6 --map Lake --max-seconds 300 --intruder
    python -m rwintel.learn collect    --layer tactics --instances 4 --record local/teacher.jsonl

Start this first and then the game instances, as with every other runner here.

The order the two layers are trained in is not a preference. The tactical layer goes first because it can be trained without playing matches at all — engagements are constructed on an empty board and fought in a minute apiece — while the operational layer needs whole matches and is therefore an order of magnitude more expensive per decision. Settling the cheap layer while the thing it will be frozen against is still cheap is the right way round.

Neither run pauses to update. The games do not stop, so a batch is collected while the parameters that collected it are already moving; that is what the clipped ratio in the optimiser is for, and it is why the batch is small.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from ..control.intruder import Intruder
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from ..eval.journal import Journal, default_path
from .arena import Arena
from .deciders import NetworkOperations, NetworkTactics, operational_batcher, tactical_batcher
from .encoding import OPERATIONAL_SIZE, TACTICAL_SIZE
from .layers import LearntOperations, LearntTactics
from .policy import OPERATIONAL, TACTICAL, LearningPolicy, learning_arm
from .rollout import Rollout
from .train import Optimiser, Trainer

log = logging.getLogger(__name__)

#: Game seconds an arena episode runs for by default. Long, because the cost the arena exists to avoid is the cost of starting a match, and one arena episode holds dozens of engagements.
ARENA_SECONDS = 1800


def _device(name: Optional[str]):
    """Where the networks run, which is the processor unless told otherwise.

    The card is the wrong device at these sizes and measurement says so plainly: the tactical policy is eight thousand parameters and the operational one is a hundred and twenty thousand, so every call is dominated by the cost of dispatching it rather than by the arithmetic. Measured on this machine, a batch of sixty-four tactical decisions takes 2.4 milliseconds on the processor against 8.1 on the card, a single decision 1.0 against 7.0, and an update over a thousand steps 186 milliseconds against 352. The design derived a requirement of four hundred decisions a second and expected the card to be the constraint; at these sizes the constraint turned out to be the other way round, and the card only becomes worth its overhead if the networks grow by orders of magnitude.
    """
    import torch

    return torch.device(name) if name else torch.device("cpu")


def _load(net, path: Optional[str], device) -> None:
    if not path:
        return
    import os

    import torch

    if not os.path.exists(path):
        log.info("no parameters at %s yet, starting from a fresh policy", path)
        return
    net.load_state_dict(torch.load(path, map_location=device))
    log.info("loaded parameters from %s", path)


def _save(net, path: Optional[str]) -> None:
    if not path:
        return
    import os

    import torch

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(net.state_dict(), path)
    log.info("saved parameters to %s", path)


def _episode(arguments, arena: bool) -> EpisodeSettings:
    return EpisodeSettings(
        map=arguments.map, opponents=arguments.opponents, difficulty=arguments.difficulty,
        credits=arguments.credits, fog=0 if arena else arguments.fog, seed=arguments.seed,
        max_seconds=arguments.max_seconds,
        # An arena episode starts with nothing on the board: there is no command that removes a unit, so the only way to have a clean board to build engagements on is never to have put anything on it.
        starting_units=0 if arena else 1, arena=arena,
    )


def _serve(arguments, arms, episode: EpisodeSettings, journal) -> list:
    settings = ServerSettings(
        host=arguments.host, port=arguments.port, instances=arguments.instances,
        episodes=arguments.episodes, arms=arms, journal=journal,
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        episode=episode,
    )
    server = Server(settings)
    try:
        return server.serve()
    except KeyboardInterrupt:
        server.stop()
        return []


# ---- the tactical run ---------------------------------------------------------------------

def train_tactics(arguments) -> int:
    from .net import TacticalNet

    device = _device(arguments.device)
    net = TacticalNet().to(device)
    _load(net, arguments.load, device)
    log.info("tactical policy on %s: %d features, %d parameters",
             device, TACTICAL_SIZE, sum(p.numel() for p in net.parameters()))

    rollout = Rollout()
    optimiser = Optimiser(net, device=device)
    batcher = tactical_batcher(net, device=device)
    trainer = Trainer(rollout, optimiser, batch=arguments.batch)
    trainer.start()

    def learnt(session, catalogue):
        return LearntTactics(session, catalogue, NetworkTactics(net, device, batcher), rollout,
                             session.instance)

    def arm(session):
        return Arena(session, tactics=learnt, seed=arguments.seed + session.instance,
                     opponent=None if arguments.script_opponent else learnt)

    journal = Journal(arguments.record or default_path("tactics"))
    try:
        sessions = _serve(arguments, [("tactics", arm)], _episode(arguments, arena=True), journal)
    finally:
        journal.close()
    report = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)

    # Counted from the episode records rather than from the policies, which are put down as each episode ends: what the arena did is a fact about the episodes it did it in, and the record is where that is kept.
    fights = sum(int(record.statistics.get("engagements", 0))
                 for session in sessions for record in session.records)
    log.info("%d engagement(s) over %d instance(s); batched inference averaged %.1f per call",
             fights, len(sessions), batcher.batch_size)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    return 0


# ---- the operational run ------------------------------------------------------------------

def train_operations(arguments) -> int:
    from .net import OperationalNet

    device = _device(arguments.device)
    net = OperationalNet().to(device)
    _load(net, arguments.load, device)
    log.info("operational policy on %s: %d features, %d parameters",
             device, OPERATIONAL_SIZE, sum(p.numel() for p in net.parameters()))

    rollout = Rollout()
    optimiser = Optimiser(net, device=device, two_headed=True)
    batcher = operational_batcher(net, device=device)
    trainer = Trainer(rollout, optimiser, batch=arguments.batch)
    trainer.start()

    arm = learning_arm(
        OPERATIONAL,
        lambda session: NetworkOperations(net, device, batcher),
        rollout,
        intruders=(lambda session: Intruder(seed=arguments.seed + session.instance,
                                            instance=session.instance)) if arguments.intruder else None,
    )
    journal = Journal(arguments.record or default_path("operations"))
    try:
        _serve(arguments, [("operations", arm)], _episode(arguments, arena=False), journal)
    finally:
        journal.close()
    report = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)
    log.info("batched inference averaged %.1f per call", batcher.batch_size)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    return 0


# ---- collecting what the script does -------------------------------------------------------

def collect(arguments) -> int:
    """Runs the ordinary chain and writes down every decision the layer under study made, in the form a learnt layer emits.

    This is the design's answer to having no way to start from human play: a replay carries commands without state and the same settings do not reproduce the same match, so the state a human command was conditioned on cannot be recovered. A script's can, because it is decided here from an observation this process is holding. What comes out is not as good as human data would have been and there is an unlimited amount of it.
    """
    rollout = Rollout()

    if arguments.layer == TACTICAL:
        # A learnt layer with no decider falls through to the rule it inherited, so what plays is the script and what is written down is the script's own decisions in the form a learnt layer emits.
        def arm(session):
            return Arena(session,
                         tactics=lambda s, catalogue: LearntTactics(s, catalogue, None, rollout, s.instance),
                         seed=arguments.seed + session.instance)

        episode = _episode(arguments, arena=True)
    else:
        def arm(session):
            return LearningPolicy(session, OPERATIONAL, None, rollout, session.instance)

        episode = _episode(arguments, arena=False)

    journal = Journal(arguments.record_episodes or default_path("collect"))
    try:
        _serve(arguments, [("script", arm)], episode, journal)
    finally:
        journal.close()
    rollout.cut_all()
    steps = rollout.drain(keep_tainted=True)
    log.info("collected %d decision(s)", len(steps))
    _write_steps(steps, arguments.record or "local/teacher.jsonl")
    return 0


def _write_steps(steps, path: Optional[str]) -> None:
    if not path or not steps:
        return
    import json
    import os

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for step in steps:
            out.write(json.dumps({"state": [round(v, 5) for v in step.state], "action": step.action,
                                  "second": step.second, "squad": step.squad, "at_ms": step.at_ms,
                                  "reward": round(step.reward, 5), "tainted": step.tainted},
                                 separators=(",", ":")) + "\n")
    log.info("wrote %d decision(s) to %s", len(steps), path)


def main(argv=None) -> int:
    # The console this is developed against is not UTF-8, and the help text below is prose rather than ASCII. Printing the help must not be the thing that raises.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["tactics", "operations", "collect"])
    parser.add_argument("--layer", default=TACTICAL, choices=[TACTICAL, OPERATIONAL],
                        help="which layer to collect, when collecting")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--map", default="Lake")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--credits", type=int, default=0)
    parser.add_argument("--fog", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-seconds", type=int, default=0,
                        help="game time an episode is cut off at, defaulting to a long one for the arena")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--device", default=None, help="torch device, defaulting to the card when there is one")
    parser.add_argument("--load", default=None, help="parameters to start from")
    parser.add_argument("--save", default=None, help="where to write the parameters afterwards")
    parser.add_argument("--batch", type=int, default=1024, help="steps that make an update")
    parser.add_argument("--intruder", action="store_true",
                        help="inject the script intruder, which the design requires for the operational layer")
    parser.add_argument("--script-opponent", action="store_true",
                        help="fight the script tactical layer rather than the policy being trained")
    parser.add_argument("--record", default=None, help="where decisions or episodes are written")
    parser.add_argument("--record-episodes", default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if arguments.max_seconds <= 0:
        arguments.max_seconds = ARENA_SECONDS if arguments.what != "operations" else 300

    if arguments.what == "tactics":
        return train_tactics(arguments)
    if arguments.what == "operations":
        return train_operations(arguments)
    return collect(arguments)


if __name__ == "__main__":
    sys.exit(main())
