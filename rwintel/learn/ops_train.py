"""Trains a learnt operational layer on the constructed operations arena.

    python -m rwintel.learn.ops_train --instances 8 --episodes 40 --map Hills --load local/operations-bc.pt --warmup 5 --save local/ops-arena.pt

Start this first and then the game instances, as with every other runner here. This is the operational counterpart of `rwintel.learn tactics`: it puts a learnt operational layer on our side of the mirror board and the script chain on the other, runs many bounded contests, and reinforces the layer against the per-decision region-domination terminal the arena pays at each horizon. The economy is frozen by construction, so unlike a full match the region-and-task choice is the only lever and the ~0.04 it is worth is resolvable rather than buried — which is the whole reason the arena exists (docs/project/08-learning.md, "アリーナは選択を分解した").

The reward is the per-squad `OperationalReward` shaping plus the terminal `OpsArena._finish_side` hands to `LearntOperations.finish` at the horizon. An arena contest is one bounded errand, so it is discounted at nothing (FIGHT_DISCOUNT / FIGHT_TRACE), and the advantage of an operational decision becomes how much better the contest went from there than the critic expected.

Measure what comes out with `rwintel.learn.ops_run` against the pinned and script arms on fresh, unused seeds — the arena is the instrument, and the discipline of separating the seed a candidate was chosen on from the seed it is confirmed on holds here as everywhere.
"""

from __future__ import annotations

import argparse
import logging
import sys

from ..control.session import EpisodeSettings
from ..eval.journal import Journal, default_path
from .__main__ import _device, _given, _load, _save, _serve, _warmup
from .deciders import NetworkOperations, operational_batcher
from .layers import LearntOperations
from .net import OperationalNet
from .ops_arena import CATCHMENT_RADIUS, CREDIT, CREDITS, GARRISON_SCALE, HORIZON_MS, OpsArena
from .ops_run import _arena_seed, pool, report
from .rollout import FIGHT_DISCOUNT, FIGHT_TRACE, Rollout
from .train import Trainer
from .train import Optimiser

log = logging.getLogger(__name__)


def train(arguments) -> int:
    device = _device(arguments.device)
    net = OperationalNet().to(device)
    _load(net, arguments.load, device)
    log.info("operational policy on %s: %d parameters", device, sum(p.numel() for p in net.parameters()))

    # One contest is a whole bounded errand, so it is discounted at nothing: the terminal — the region domination the contest came to — reaches every operational decision in it, and the advantage is how much better it went than the critic expected. The shaping telescopes at the same one.
    rollout = Rollout(discount=FIGHT_DISCOUNT, trace=FIGHT_TRACE)
    optimiser = Optimiser(net, device=device, two_headed=True, warmup=_warmup(arguments),
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate))
    # The batcher reads the very parameters this run's optimiser writes, so it is handed the lock the optimiser takes around a minibatch step.
    batcher = operational_batcher(net, device=device, guard=optimiser.lock)
    trainer = Trainer(rollout, optimiser, **_given(batch=arguments.batch))
    trainer.start()

    def learnt(session, catalogue):
        # A learnt operational layer with the network decider and this run's rollout, discounted as the arena errand is. The rest of the chain below it is the script, frozen, exactly as the engagement arena freezes everything but the tactical decision.
        return LearntOperations(session, catalogue, NetworkOperations(net, device, batcher),
                                rollout, session.instance, discount=FIGHT_DISCOUNT)

    def arm(session) -> OpsArena:
        # One arm trains, so the board advances with every episode.
        return OpsArena(session, operations=learnt, seed=_arena_seed(arguments.seed, session, 1),
                        horizon_ms=arguments.horizon * 1000, our_squads=arguments.squads,
                        catchment_radius=arguments.radius, contest_pairs=arguments.pairs,
                        credit=arguments.credit, garrison_scale=arguments.garrison)

    episode = EpisodeSettings(
        map=arguments.map, opponents=arguments.opponents, difficulty=arguments.difficulty,
        credits=arguments.credits, fog=0, seed=arguments.seed, max_seconds=arguments.max_seconds,
        starting_units=0, arena=True,
    )
    journal = Journal(arguments.record or default_path("ops-train"))
    log.info("training the operational layer on the arena over %d episode(s) each on %d instance(s), horizon %ds",
             arguments.episodes, arguments.instances, arguments.horizon)
    try:
        sessions = _serve(arguments, [("ops-learn", arm)], episode, journal)
    finally:
        journal.close()
    report_ = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)
    if report_ is not None:
        log.info("last update: %s", report_.as_dict())
    # The learnt side's own domination over the horizon, as a run of it — not a duel, which ops_run does against the pin and the script. A rising figure over a run is the layer learning to dominate; the honest comparison is the separate paired duel, which has to be given a base seed this run did not use — both runners build a board from the same seed arithmetic, so a duel at the training seed replays the very boards the policy was fitted on.
    report(pool(sessions), "learnt")
    return 0


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--map", default="Hills")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--credits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--horizon", type=int, default=HORIZON_MS // 1000)
    parser.add_argument("--squads", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--radius", type=float, default=CATCHMENT_RADIUS)
    parser.add_argument("--garrison", type=float, default=GARRISON_SCALE,
                        help="credits a contested region's defender is drawn out of, which is what decides whether taking ground pays at all")
    parser.add_argument("--credit", choices=CREDITS, default=CREDIT,
                        help="what a squad's terminal is: the whole domination of the region its contract named, which every squad sent there takes in full, or only the part its own surviving units account for")
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--load", default=None, help="parameters to start from, an imitation of the script or an earlier run")
    parser.add_argument("--save", default=None, help="where the trained parameters are written")
    parser.add_argument("--warmup", type=int, default=None, help="updates at the start that fit the value head and nothing else")
    parser.add_argument("--entropy", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if arguments.max_seconds <= 0:
        arguments.max_seconds = arguments.horizon + 60
    return train(arguments)


if __name__ == "__main__":
    sys.exit(main())
