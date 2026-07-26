"""Trains a learnt operational layer on the constructed operations arena.

    python -m rwintel.learn.ops_train --instances 8 --episodes 40 --map Hills --load local/operations-bc.pt --warmup 5 --save local/ops-arena.pt

Start this first and then the game instances, as with every other runner here. This is the operational counterpart of `rwintel.learn tactics`: it puts a learnt operational layer on our side of the mirror board and the script chain on the other, runs many bounded contests, and reinforces the layer against the per-decision region-domination terminal the arena pays at each horizon. The economy is frozen by construction, so unlike a full match the region-and-task choice is the only lever and the ~0.04 it is worth is resolvable rather than buried — which is the whole reason the arena exists (docs/record/04-operations.md).

The reward is the arena's own reading of its scored discs, paid every operational period: a squad is paid the movement of `priority(region) * (the share of that region's catchment now − the share it opened at)`, each board read under the contract in force at that board. Those payments telescope, so an episode returns exactly what the horizon terminal always was and only its density has changed. An arena contest is one bounded errand, so it is discounted at nothing (FIGHT_DISCOUNT / FIGHT_TRACE), and the advantage of an operational decision becomes how much better the contest went from there than the critic expected.

That density is the whole of what this runner was missing. A trajectory used to be cut every time a fresh contract replaced an errand, because the region block a match is shaped by is re-based against fresh ground at every contract; a cut trajectory is never paid a terminal, its last decision is bootstrapped from its own value estimate, and the period that does the replacing pays nought — so with a layer re-drawing its region every period, nearly every sampled decision carried an advantage of exactly nought and normalisation turned that mass into one identical push on whatever the policy happened to draw. Paid off one quantity from the first decision to the last, there is no errand boundary left to cut at. The run reports what it collected as well as what it scored: the buffer's census, per update, says how many of a batch's decisions lay in a trajectory a payment ever reached, and the runner's signal line says how long an errand ran, which is now a description of how decisive the layer is rather than a bound on the signal.

This is also the runner in which the project's learning order reaches its second half. That order is settle the tactical layer first, then move the operational layer alone against a tactical layer held still, and `--tactics` is what holds it still: trained tactical parameters are read once, frozen under BOTH sides of the mirror board, and record nothing, while the operational layer above them is the only thing this run changes. Without the option the handwritten layer is the default on both sides, which is what every measurement taken on this arena so far was made under, and every one of them is unchanged.

Measure what comes out with `rwintel.learn.ops_run` against the pinned and script arms on fresh, unused seeds — the arena is the instrument, and the discipline of separating the seed a candidate was chosen on from the seed it is confirmed on holds here as everywhere. A run made under trained tactical parameters is measured under the same ones: the tactical layer is part of the instrument, the episode records say which one was beneath the board, and the comparison refuses to pair two runs that disagree about it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ..control.policy.operations import Concentrated, Operations
from ..control.session import EpisodeSettings
from ..eval.journal import Journal, default_path
from .__main__ import _device, _given, _load, _save, _serve, _warmup
from .deciders import NetworkOperations, operational_batcher
from .layers import LearntOperations
from .net import OperationalNet
from .ops_arena import (CATCHMENT_RADIUS, CREDIT, CREDITS, GARRISON_SCALE, HORIZON_MS,
                        OPENING_BASELINE, OpsArena)
from .ops_run import _arena_seed, frozen_tactics, pool, report, signal
from .rollout import FIGHT_DISCOUNT, FIGHT_TRACE, Rollout
from .train import Trainer
from .train import Optimiser

#: The layers the other side of the board may be given, and what each teaches by being there.
#:
#: `script` is the handwritten chain and is what every measurement on this arena was taken against; it is the default because a run trained against something else is still MEASURED against it, and changing both at once would leave nothing to read the change by. `concentrate` sends every enemy squad at the one region the board wants most, which is the arm that scores best here — against it, ground left unheld is ground lost, where against the script it is very nearly ground kept.
#:
#: Which opponent trains the better layer is not something this project has measured. What it is here for is stated plainly: the per-squad reward pays exactly nought for a squad sent at ground the board puts no priority on, less than nought for one sent to hold ground already owned, and something positive only for one that takes. Against an enemy that takes nothing, doing nothing is a stable answer to that arithmetic, and the arena had no way to ask a harder question.
OPPONENTS = ("script", "concentrate")

log = logging.getLogger(__name__)


def train(arguments) -> int:
    # The frozen tactical layer before anything else this run builds, so that a mistyped path is refused while there is nothing to tear down: no optimiser, no buffer, no trainer thread and no inference server exists yet, so the refusal costs nothing and leaves nothing running.
    frozen = frozen_tactics(arguments.tactics, arguments.device)
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
        # A learnt operational layer with the network decider and this run's rollout, discounted as the arena errand is. The rest of the chain below it is frozen, exactly as the engagement arena freezes everything but the tactical decision: the handwritten fighter, or the trained one this run was given, and either way it is handed no rollout and records nothing. This buffer is the operational layer's alone, and it has to be: a trajectory is keyed by the instance and the squad, and the tactical layer under the board decides about those very same squads.
        return LearntOperations(session, catalogue, NetworkOperations(net, device, batcher),
                                rollout, session.instance, discount=FIGHT_DISCOUNT)

    def enemy(session, catalogue):
        # The other side of the mirror. Built here rather than left to the arena's default so that a run can be trained against something that punishes abandoning ground; it is handed no rollout either way, since only one layer moves in a training run.
        return Concentrated(session, catalogue) if arguments.opponent == "concentrate" else Operations(session, catalogue)

    def arm(session) -> OpsArena:
        # One arm trains, so the board advances with every episode.
        return OpsArena(session, operations=learnt, opponent=enemy,
                        tactics=frozen.build, tactics_name=frozen.name,
                        # Not a digest, because the parameters this side plays under change with every update: what a training episode was played by is a moving policy and no file names it. Written all the same so that a training journal can never be paired against a measuring run's arm as though it were a fixed one.
                        operations_name="learning",
                        seed=_arena_seed(arguments.seed, session, 1),
                        horizon_ms=arguments.horizon * 1000, our_squads=arguments.squads,
                        catchment_radius=arguments.radius, contest_pairs=arguments.pairs,
                        credit=arguments.credit, opening_baseline=arguments.opening,
                        garrison_scale=arguments.garrison)

    episode = EpisodeSettings(
        map=arguments.map, opponents=arguments.opponents, difficulty=arguments.difficulty,
        credits=arguments.credits, fog=0, seed=arguments.seed, max_seconds=arguments.max_seconds,
        starting_units=0, arena=True,
    )
    # A run made under trained tactical parameters writes to a journal of its own, because it is not the same instrument as a run made under the handwritten layer: a journal is opened for appending, and two runs in one file put two episodes on one board, which a later comparison can only drop.
    under = "-under-" + os.path.splitext(os.path.basename(arguments.tactics))[0] if arguments.tactics else ""
    journal = Journal(arguments.record or default_path("ops-train" + under))
    log.info("training the operational layer on the arena over %d episode(s) each on %d instance(s), horizon %ds, "
             "against the %s chain, both sides fighting under the %s tactical layer",
             arguments.episodes, arguments.instances, arguments.horizon, arguments.opponent, frozen.name)
    try:
        sessions = _serve(arguments, [("ops-learn", arm)], episode, journal)
    finally:
        journal.close()
    report_ = trainer.finish()
    batcher.stop()
    if frozen.batcher is not None:
        # Under a frozen tactical layer the fighting is the dominant inference load of the run — every squad of both sides asks it for a departure every tactical frame, against one operational request a squad a period — and nothing else here reports it. How it batches is the first thing to read the run's speed against.
        log.info("the frozen tactical layer's batched inference averaged %.1f per call over %d call(s)",
                 frozen.batcher.batch_size, frozen.batcher.calls)
    frozen.stop()
    _save(net, arguments.save)
    if report_ is not None:
        log.info("last update: %s", report_.as_dict())
    # The learnt side's own domination over the horizon, as a run of it — not a duel, which ops_run does against the pin and the script. A rising figure over a run is the layer learning to dominate; the honest comparison is the separate paired duel, which has to be given a base seed this run did not use — both runners build a board from the same seed arithmetic, so a duel at the training seed replays the very boards the policy was fitted on.
    report(pool(sessions), "learnt")
    # And how decisive the layer was: how many decisions it took and how few errands it divided them into, which a rising return cannot say and which is what separates a layer that settled on a deployment from one that changed its mind every period. Whether the payments reached those decisions is the census's question and is answered per update.
    signal(sessions)
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
    parser.add_argument("--opening", type=float, default=OPENING_BASELINE,
                        help="what point a squad's terminal is read from: one measures the errand against the opening ownership of the disc it was sent to, which is what the side score is a mean of, and nought measures it against the neutral half, which pays a squad for how the ground stands rather than for what it did to the ground and is the reading the layers that learnt to attack nothing were trained under")
    parser.add_argument("--credit", choices=CREDITS, default=CREDIT,
                        help="what a squad's terminal is: the whole domination of the region its contract named, which every squad sent there takes in full, or only the part its own surviving units account for")
    parser.add_argument("--max-seconds", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--load", default=None, help="parameters to start from, an imitation of the script or an earlier run")
    parser.add_argument("--opponent", choices=OPPONENTS, default=OPPONENTS[0],
                        help="what the other side of the mirror plays. The script chain is the default and is what "
                             "every measurement on this arena is taken against; the concentrating arm is the "
                             "hardest thing here, and against it ground left unheld is ground lost, where against "
                             "the script it is nearly ground kept. The measuring runner always faces the script, so "
                             "moving this changes the training and not the instrument")
    parser.add_argument("--tactics", default=None,
                        help="parameters for a trained tactical layer to be put, frozen, under BOTH sides of the board, read at its likeliest action and recording nothing; this is the half of the learning order in which the settled layer is held still and the operational layer alone moves. Left out, both sides fight the handwritten layer, which is what every measurement on this arena so far was made under; given, the run is a different instrument and its journal says so")
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
