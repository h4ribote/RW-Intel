"""Runs a training pass, collects what the script chain does as data to start one from, imitates it, or measures a policy against it.

    python -m rwintel.learn tactics    --instances 4 --save local/tactics.pt
    python -m rwintel.learn operations --instances 4 --episodes 6 --map Lake --max-seconds 300 --intruder
    python -m rwintel.learn collect    --layer tactics --instances 4 --record local/teacher.jsonl
    python -m rwintel.learn clone      --layer tactics --teacher local/teacher.jsonl --save local/tactics-bc.pt
    python -m rwintel.learn duel       --load local/tactics.pt --instances 8 --max-seconds 600

Start this first and then the game instances, as with every other runner here. The cloning run is the exception: it reads a file and touches no game at all.

The four make one order of work. The collecting run turns the handwritten layer into a file of decisions, the cloning run fits a network to them, the training run improves that network against the arena while warming its value head first, and the duelling run measures what came out against the handwritten layer it started from. None of the four is required by the others — a policy can be trained from noise and measured without ever having been cloned — but skipping the first two spends the early part of a training run rediscovering a rule ladder that was already written down.

The order the two layers are trained in is not a preference. The tactical layer goes first because it can be trained without playing matches at all — engagements are constructed on an empty board and fought in a minute apiece — while the operational layer needs whole matches and is therefore an order of magnitude more expensive per decision. Settling the cheap layer while the thing it will be frozen against is still cheap is the right way round.

Neither training run pauses to update. The games do not stop, so a batch is collected while the parameters that collected it are already moving; that is what the clipped ratio in the optimiser is for, and it is why the batch is small.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from typing import Optional

from ..control.intruder import Intruder
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from ..eval.journal import Journal, default_path
from ..eval.sampling import UNBOUNDED_EPISODES, episodes_for
from .arena import Arena
from .deciders import NetworkOperations, NetworkTactics, operational_batcher, tactical_batcher
from .encoding import OPERATIONAL_SIZE, TACTICAL_SIZE
from .layers import LearntOperations, LearntTactics
from .policy import OPERATIONAL, TACTICAL, LearningPolicy, learning_arm
from .rollout import Rollout
from .train import Optimiser, Trainer

log = logging.getLogger(__name__)

#: When the process started, so that a run can report what it produced against the wall clock it took, which is the only thing an optimisation of the arena can be judged against.
_started_at = time.time()

#: Game seconds an arena episode runs for by default, which is about six fights.
#:
#: Short, and measured rather than chosen. The obvious setting is long, because the cost the arena exists to avoid is the cost of starting a match and one long episode holds dozens of fights instead of a handful. What that misses is that a fight cannot be cleared away: there is no instruction that removes a unit, so the survivors of every fight stay on the board, and the next fight is built on a board with more and more of them standing about on it. They are not idle. Measured with the handwritten layer on both sides, where the score of a fight has to average to nought because the score of one side is the score of the other negated, the first five fights of an episode averaged minus a sixth, fights ten to twenty plus a seventh, and everything past the fortieth plus four tenths. A long episode is therefore not a cheap way of getting many fights; it is a way of getting many fights on a board that is no longer even, and no comparison of two policies run on it means anything.
#:
#: At this length the same measurement comes out at two hundredths over three hundred and thirty eight fights, which is inside the noise. Restarting the match more often costs about a fifth of the throughput and is what buys the arena back.
ARENA_SECONDS = 240

#: Threads the tensor library is allowed. Bounded rather than left at the default, which is one per core, because the games this process is learning from are on the same cores.
TORCH_THREADS = 2


def _device(name: Optional[str]):
    """Where the networks run, which is the processor unless told otherwise.

    The card is the wrong device at these sizes and measurement says so plainly: the tactical policy is eight thousand parameters and the operational one is a hundred and twenty thousand, so every call is dominated by the cost of dispatching it rather than by the arithmetic. Measured on this machine, a batch of sixty-four tactical decisions takes 2.4 milliseconds on the processor against 8.1 on the card, a single decision 1.0 against 7.0, and an update over a thousand steps 186 milliseconds against 352. The design derived a requirement of four hundred decisions a second and expected the card to be the constraint; at these sizes the constraint turned out to be the other way round, and the card only becomes worth its overhead if the networks grow by orders of magnitude.
    """
    import torch

    # Two threads, because the networks are small enough that one is nearly as fast and the machine is not idle. A dozen game instances are running beside this process and each wants a core; a tensor library that helps itself to all of them turns every update into a fight with the simulation it is learning from, and the simulation is the part that cannot be made faster.
    torch.set_num_threads(TORCH_THREADS)
    return torch.device(name) if name else torch.device("cpu")


def _given(**asked) -> dict:
    """Only the options somebody actually asked for, as keywords.

    What is left out then stands at the figure stated by the module that owns it rather than at a copy of that figure kept here. The two batch sizes are the case that makes this worth a helper: the steps in a reinforcement update and the teacher's decisions in one gradient step are different numbers living in different files, and one option carries both.
    """
    return {name: value for name, value in asked.items() if value is not None}


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
    net = TacticalNet(**_given(width=arguments.width)).to(device)
    _load(net, arguments.load, device)
    log.info("tactical policy on %s: %d features, %d parameters",
             device, TACTICAL_SIZE, sum(p.numel() for p in net.parameters()))

    rollout = Rollout()
    optimiser = Optimiser(net, device=device, warmup=arguments.warmup,
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate))
    batcher = tactical_batcher(net, device=device)
    trainer = Trainer(rollout, optimiser, **_given(batch=arguments.batch))
    trainer.start()

    def learnt(session, catalogue):
        # An arena fight carries one contract from the moment it is joined to the moment it is called, and nothing reissues it. What ends the errand is therefore what ends the fight, and the conditions written into the contract are left as something for the layer to read and act on rather than as something that stops it being paid.
        return LearntTactics(session, catalogue, NetworkTactics(net, device, batcher), rollout,
                             session.instance, status_terminals=False)

    def arm(session):
        # Both sides script is how the arena itself is measured rather than a policy: it is the baseline a learnt layer has to beat, and it is the only setting in which what the arena produces says something about the arena rather than about whatever the policy currently happens to do.
        ours = None if arguments.script else learnt
        return Arena(session, tactics=ours, seed=arguments.seed + session.instance,
                     **_given(outcome_weight=arguments.outcome_weight,
                              stall_ms=arguments.stall_seconds * 1000 if arguments.stall_seconds else None),
                     opponent=None if (arguments.script or arguments.script_opponent) else learnt)

    journal = Journal(arguments.record or default_path("tactics"))
    try:
        sessions = _serve(arguments, [("tactics", arm)], _episode(arguments, arena=True), journal)
    finally:
        journal.close()
    report = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)

    # Counted from the episode records rather than from the policies, which are put down as each episode ends: what the arena did is a fact about the episodes it did it in, and the record is where that is kept.
    _report_arena(sessions, batcher)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    return 0


def _report_arena(sessions, batcher=None) -> None:
    """What an arena run produced, in the terms it is optimised against: how much of the time went into fights that happened, how those fights ended, and how much the inference actually batched.

    There is no batching to report when both sides were the handwritten layer, since nothing was asked of a network at all.
    """
    records = [record for session in sessions for record in session.records]
    if not records:
        return
    total = {key: sum(int(r.statistics.get(key, 0)) for r in records)
             for key in ("engagements", "stillborn", "fought", "won", "lost", "drawn", "stalled",
                         "expired", "mutual", "decisions")}
    seconds = sum(r.seconds for r in records)
    wall = max(1.0, sum(r.wall_seconds for r in records) / max(1, len(sessions)))
    # The drawn fights are broken out because they are the ones that say what kind of arena this was: two forces that stopped hurting each other, two that never reached each other before the clock, and two that destroyed each other are three different results and only the last of them is a fight.
    log.info("%d engagement(s) built, %d never appeared (%.0f%% wasted), %d fought: %d won %d lost %d drawn (%.0f%% drawn: %d stalled, %d out of time, %d mutual)",
             total["engagements"], total["stillborn"],
             100.0 * total["stillborn"] / max(1, total["engagements"]), total["fought"],
             total["won"], total["lost"], total["drawn"],
             100.0 * total["drawn"] / max(1, total["fought"]),
             total["stalled"], total["expired"], total["mutual"])
    log.info("errand(s) closed by reason: %s", {reason: sum(int(record.statistics.get("terminals", {}).get(reason, 0)) for record in records) for reason in sorted({reason for record in records for reason in record.statistics.get("terminals", {})})})
    speeds = [r.speed for r in records if r.speed > 0]
    per_instance = sum(speeds) / len(speeds) if speeds else 0.0
    log.info("%d game second(s) over %d episode(s), %.1fx per instance and %.0fx over %d of them: %.1f fight(s) and %d decision(s) per game minute, %.0f decision(s) per wall second",
             seconds, len(records), per_instance, per_instance * len(sessions), len(sessions),
             60.0 * total["fought"] / max(1, seconds), int(60.0 * total["decisions"] / max(1, seconds)),
             total["decisions"] / max(1.0, wall))
    if batcher is not None:
        log.info("batched inference averaged %.1f per call over %d call(s)", batcher.batch_size, batcher.calls)


# ---- the measurement run -------------------------------------------------------------------

def duel(arguments) -> int:
    """Runs the arena for measurement rather than for learning: the loaded policy on one side, the handwritten tactical layer on the other, and nothing kept but the score.

    Neither a buffer nor a trainer is built here, and the layer is handed no rollout at all, so no decision is written down anywhere. That is not thrift. A buffer nobody drains grows for the length of the run, and a trainer would move the parameters being measured while they were being measured, which would make the number that came out a number about no policy in particular.

    Leaving the policy out entirely is the baseline, and it is worth running before every comparison. Both sides are then the same handwritten layer, so the average result has to be nought by the antisymmetry of the score itself; anything else is the arena favouring one side of the board, and on an arena that favours one side no comparison between two policies means anything.
    """
    learnt = batcher = None
    if arguments.load:
        import os

        from .net import TacticalNet

        if not os.path.exists(arguments.load):
            # Refused rather than started from nothing, which is what a training run does with a missing file. A measurement that quietly scored a freshly initialised policy would produce a perfectly plausible number about a policy nobody asked about.
            log.error("there are no parameters at %s to measure", arguments.load)
            return 1
        device = _device(arguments.device)
        net = TacticalNet(**_given(width=arguments.width)).to(device)
        _load(net, arguments.load, device)
        batcher = tactical_batcher(net, device=device, greedy=arguments.greedy)

        def learnt(session, catalogue):
            # No rollout: this layer is being read from and not learnt from, and with nowhere to record a decision it records none.
            return LearntTactics(session, catalogue,
                                 NetworkTactics(net, device, batcher, arguments.greedy),
                                 None, session.instance, status_terminals=False)

    def arm(session):
        # The opponent is left unnamed, which is what puts the handwritten layer on the other side of every fight. That is the thing being measured against, so it is not something this run offers a choice about.
        return Arena(session, tactics=learnt, seed=arguments.seed + session.instance,
                     **_given(stall_ms=arguments.stall_seconds * 1000 if arguments.stall_seconds else None))

    # The baseline is written down as the baseline. It is a different quantity from a policy's score rather than a run of it that happens to have scored nought, and the likeliest way to confuse the two is to have journalled them under one name.
    name = "duel" if arguments.load else "duel-baseline"
    journal = Journal(arguments.record or default_path(name))
    try:
        sessions = _serve(arguments, [(name, arm)], _episode(arguments, arena=True), journal)
    finally:
        journal.close()
    if batcher is not None:
        batcher.stop()

    _report_arena(sessions, batcher)
    _report_duel(sessions, loaded=bool(arguments.load))
    return 0


def _report_duel(sessions, loaded: bool) -> None:
    """The score of a measurement run: how the fights went on average, how widely that scattered, and how many fights an assertion of that average would take.

    Recombined from the episode records rather than kept as one long list of fights. A record carries its episode's mean, its spread and how many fights it was taken over, and those three are enough to reconstitute both figures over the whole run exactly, however many episodes and instances it ran across.
    """
    records = [record for session in sessions for record in session.records]
    counted = [(int(record.statistics.get("fought", 0)),
                float(record.statistics.get("outcome_mean", 0.0)),
                float(record.statistics.get("outcome_sd", 0.0)))
               for record in records]
    counted = [entry for entry in counted if entry[0] > 0]
    total = sum(count for count, _, _ in counted)
    if not total:
        log.info("no fight was called, so there is nothing to score")
        return

    mean = sum(count * value for count, value, _ in counted) / total
    # The spread each record carries is over its own fights and around its own mean, so the two are put back together by pooling the second moments and taking this run's mean off afterwards.
    spread = sum(count * (deviation ** 2 + value ** 2) for count, value, deviation in counted) / total - mean ** 2
    # Quoted on the sample rather than on the population, matching how every other scatter in this project is reported and how the sample sizes were computed.
    deviation = math.sqrt(max(0.0, spread) * total / (total - 1)) if total > 1 else 0.0

    log.info("%d fight(s) over %d episode(s): outcome %+.4f, spread %.4f",
             total, len(counted), mean, deviation)
    needed = episodes_for(deviation, abs(mean))
    # A spread of nought is one that was never measured rather than one measured to be small, and it is what a single fight or a run of identical fights produces. The sizing arithmetic answers nought fights for it, quite correctly given a scatter of nought, so the claim has to be gated on there having been a scatter at all: without that a run of one engagement sizes its own claim at no engagements and declares itself sufficient.
    enough = total > 1 and deviation > 0.0 and needed <= total
    if needed >= UNBOUNDED_EPISODES:
        log.info("the average result is exactly nought, which is not a difference and which no number of fights would establish")
    elif deviation <= 0.0:
        log.info("all %d fight(s) came out at %+.4f, so this run measured no spread at all and there is nothing to size a claim against",
                 total, mean)
    else:
        log.info("claiming an average of %+.4f at that spread takes %d fight(s), and %d were fought: %s",
                 mean, needed, total, "enough" if enough else "not enough yet")

    if not loaded:
        if enough:
            log.warning("both sides were the handwritten layer, so this average has to be nought and it is %+.4f over enough fights to say so: the arena favours one side of the board, and until that is found and fixed a comparison of two policies run on it does not mean anything", mean)
        else:
            log.info("both sides were the handwritten layer and %d fight(s) have not separated their average of %+.4f from nought, which is as much as this run says about whether the arena is even", total, mean)


# ---- the operational run ------------------------------------------------------------------

def train_operations(arguments) -> int:
    from .net import OperationalNet

    device = _device(arguments.device)
    net = OperationalNet().to(device)
    _load(net, arguments.load, device)
    log.info("operational policy on %s: %d features, %d parameters",
             device, OPERATIONAL_SIZE, sum(p.numel() for p in net.parameters()))

    rollout = Rollout()
    optimiser = Optimiser(net, device=device, two_headed=True, warmup=arguments.warmup,
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate))
    batcher = operational_batcher(net, device=device)
    trainer = Trainer(rollout, optimiser, **_given(batch=arguments.batch))
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
                         tactics=lambda s, catalogue: LearntTactics(s, catalogue, None, rollout, s.instance,
                                                                    status_terminals=False),
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
    """One decision per line, in the form anything fitting to them reads.

    What was legal is written down beside what was chosen, as flags rather than as weights. Without it a decision is not reconstructible: the operational layer picks a region out of the few that exist on the board it saw, and a reader that could not tell which those were would be fitting a distribution over twenty-four regions of which most were never on offer. A file that predates this carries none, and a reader is expected to take that as everything having been allowed, which is what the tactical layer's mask always is anyway.
    """
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
                                  "mask": [int(value > 0) for value in step.mask],
                                  "second_mask": [int(value > 0) for value in step.second_mask],
                                  "reward": round(step.reward, 5), "tainted": step.tainted},
                                 separators=(",", ":")) + "\n")
    log.info("wrote %d decision(s) to %s", len(steps), path)


# ---- imitating what the script does ---------------------------------------------------------

def clone(arguments) -> int:
    """Fits a network to the decisions a collecting run wrote down, which is where a training run is meant to start from rather than from noise.

    Nothing is connected to and no game is started: the teacher is a file, and the whole of this is a few minutes of arithmetic on the processor. What comes out has a policy worth measuring and a value head that is still random, which is what the training run's warm-up is for.
    """
    from .imitation import EPOCHS, PATIENCE, SMOOTHING, fit, read_teacher, report
    from .net import OperationalNet, TacticalNet

    device = _device(arguments.device)
    net = (TacticalNet(**_given(width=arguments.width)) if arguments.layer == TACTICAL
           else OperationalNet()).to(device)
    _load(net, arguments.load, device)
    samples = read_teacher(arguments.teacher or "local/teacher.jsonl", arguments.layer,
                           keep_tainted=arguments.keep_tainted)
    # Only what was actually asked for is passed on, so that everything else stands at the figure the cloning module states rather than at a copy of it kept here.
    asked = _given(smoothing=arguments.smoothing, epochs=arguments.epochs,
                   patience=arguments.patience, batch=arguments.batch)
    log.info("cloning the %s layer with smoothing %.3f over at most %d epoch(s), stopping after %d "
             "without improvement", arguments.layer,
             asked.get("smoothing", SMOOTHING), asked.get("epochs", EPOCHS), asked.get("patience", PATIENCE))
    net, cloning = fit(samples, arguments.layer, net=net, device=device, seed=arguments.seed, **asked)
    report(cloning)
    _save(net, arguments.save)
    return 0


def main(argv=None) -> int:
    # The console this is developed against is not UTF-8, and the help text below is prose rather than ASCII. Printing the help must not be the thing that raises.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["tactics", "operations", "collect", "clone", "duel"])
    parser.add_argument("--layer", default=TACTICAL, choices=[TACTICAL, OPERATIONAL],
                        help="which layer to collect or to clone")
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
    parser.add_argument("--device", default=None,
                        help="torch device. The default is the processor, which at these network sizes "
                             "measures three to seven times faster than the card")
    parser.add_argument("--load", default=None, help="parameters to start from")
    parser.add_argument("--save", default=None, help="where to write the parameters afterwards")
    parser.add_argument("--batch", type=int, default=None,
                        help="rows in one gradient step: the steps that make a reinforcement update when "
                             "training, the teacher's decisions in one minibatch when cloning")
    parser.add_argument("--stall-seconds", type=int, default=None,
                        help="game seconds a fight may go without a casualty before the arena calls it, which is how decisive its fights are")
    parser.add_argument("--width", type=int, default=None,
                        help="hidden units per layer in the tactical network, which the measured cost of inference leaves room to raise")
    parser.add_argument("--entropy", type=float, default=None,
                        help="how hard the objective pushes the policy towards choosing evenly, which a run starting from an imitation wants much less of than one starting from noise")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--outcome-weight", type=float, default=None,
                        help="how much of a terminal the score of a fight is worth in the arena")
    parser.add_argument("--warmup", type=int, default=0,
                        help="updates at the start that fit the value head alone, holding the trunk and the "
                             "policy still. Meant for a run started from imitated parameters, whose critic "
                             "did not come with them")
    parser.add_argument("--teacher", default=None,
                        help="decisions to clone from, as written by a collecting run, defaulting to "
                             "local/teacher.jsonl")
    # The three that follow default to nothing here and are filled in from the cloning module when a clone is actually run, because quoting its numbers in this help text would be a second place for them to be stated and a first place for them to go stale. What was used is logged as the run starts.
    parser.add_argument("--smoothing", type=float, default=None,
                        help="how much of each label is shared out over the actions the teacher did not "
                             "choose, when cloning. Nought copies a deterministic script exactly and leaves "
                             "the training run after it nothing to explore with")
    parser.add_argument("--epochs", type=int, default=None,
                        help="passes over the teacher, when cloning")
    parser.add_argument("--patience", type=int, default=None,
                        help="epochs the held-out tenth may fail to improve for before cloning stops")
    parser.add_argument("--keep-tainted", action="store_true",
                        help="clone from decisions about squads somebody outside the chain interfered with too")
    parser.add_argument("--greedy", action="store_true",
                        help="take the likeliest action rather than drawing one, when duelling. The default "
                             "is to draw, which is what the policy does when it is operated")
    parser.add_argument("--intruder", action="store_true",
                        help="inject the script intruder, which the design requires for the operational layer")
    parser.add_argument("--script", action="store_true",
                        help="run the handwritten layer on both sides, which is the baseline and the way to measure the arena itself")
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
    if arguments.what == "clone":
        return clone(arguments)
    if arguments.what == "duel":
        return duel(arguments)
    return collect(arguments)


if __name__ == "__main__":
    sys.exit(main())
