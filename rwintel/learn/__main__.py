"""Runs a training pass, collects what the script chain does as data to start one from, imitates it, or measures a policy against it.

    python -m rwintel.learn tactics    --instances 4 --save local/tactics.pt
    python -m rwintel.learn operations --instances 4 --episodes 6 --map Lake --max-seconds 300 --intruder
    python -m rwintel.learn strategy   --instances 8 --episodes 40 --map Lake --max-seconds 300 --intruder
    python -m rwintel.learn collect    --layer tactics --instances 4 --record local/teacher.jsonl
    python -m rwintel.learn clone      --layer tactics --teacher local/teacher.jsonl --save local/tactics-bc.pt
    python -m rwintel.learn duel       --load local/tactics.pt --instances 8 --max-seconds 600
    python -m rwintel.learn avow       --layer operations --load local/operations.pt --because "why you know"

Start this first and then the game instances, as with every other runner here. The cloning run and the avowal are the exceptions: they read a file and touch no game at all.

The avowal is not part of the order of work below and is run only where a set of parameters cannot say for itself what it was fitted to. Every set written here states the feature list it was fitted to, and every loader refuses one that does not, because nothing else in a file can tell parameters fitted to an older meaning of a slot from ones fitted to this meaning. Parameters recorded before that list was written down state none and are refused with the rest — which is honest, and which throws away work that is perfectly good whenever the layer's encoding has not in fact moved. Avowing is how a person says that into the file itself, where every later reader finds it and every loader says out loud that the list beside those parameters is somebody's word and not a fit's record.

The four make one order of work. The collecting run turns the handwritten layer into a file of decisions, the cloning run fits a network to them, the training run improves that network against the arena while warming its value head first, and the duelling run measures what came out against the handwritten layer it started from. None of the four is required by the others — a policy can be trained from noise and measured without ever having been cloned — but skipping the first two spends the early part of a training run rediscovering a rule ladder that was already written down.

The order the three layers are trained in is not a preference. The tactical layer goes first because it can be trained without playing matches at all — engagements are constructed on an empty board and fought in a minute apiece — while the operational layer needs whole matches, or the constructed board that stands in for them, and is an order of magnitude more expensive per decision. Settling the cheap layer while the thing it will be frozen against is still cheap is the right way round. The strategic layer goes last and has no constructed board of its own, because what it decides between is how a whole match is to be spent: construct that away and there is nothing left to decide. It is therefore trained on matches and paid the match, which is the one place in this design where a layer sees the result.

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
from ..eval.sampling import UNBOUNDED_EPISODES, Summary, episodes_for
from ..wire import Deviation
from .arena import BY_HEALTH, BY_KILLS, DECISION_ORDERS, SCORES, Arena
from .deciders import (
    NetworkOperations,
    NetworkStrategy,
    NetworkTactics,
    PinnedDeparture,
    operational_batcher,
    strategic_batcher,
    tactical_batcher,
)
from .encoding import OPERATIONAL_SIZE, STRATEGIC_SIZE, TACTICAL_SIZE
from .frozen import frozen_layers
from .layers import LearntOperations, LearntTactics
from .policy import LAYERS, OPERATIONAL, STRATEGIC, TACTICAL, LearningPolicy, learning_arm
from .rollout import FIGHT_DISCOUNT, FIGHT_TRACE, Rollout
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


#: How far apart two instances' arena draws are set. It only has to exceed the episodes one instance will ever run, and it is prime so that two runs started at neighbouring seeds do not lay one instance's stream on top of another's.
INSTANCE_STRIDE = 100003


def _arena_seed(arguments, session) -> int:
    """The seed an arena episode draws its fights from, which advances with the episode as well as with the instance.

    An arena is built afresh for every episode, and it used to be built from the instance alone. Every episode of an instance therefore drew the same site, the same two budgets, the same imbalance, the same angle and the same two forces as the one before it, in the same order: a run of fifty episodes on seven instances was not two thousand fights but about fifty distinct fights fought forty times over. Measured on the recorded runs, the first fight of every episode of an instance had one single spawn order across all fifty of them, and the second and third nearly always did too.

    What that costs is the sample size, and it costs it by a factor of thirty to fifty. The scatter of the fight-level mean was quoted as two standard errors over the number of fights, which for two thousand fights is about a fortieth; taken over the distinct draws instead it is about a seventh. Every left-right lean the arena has been charged with — the tenth of a point that came and went with the seed, the seven hundredths a lopsided draw was blamed for — sits comfortably inside that. There was no asymmetry to find. There were fifty fights being reported as two thousand.

    The episode is folded in per arm rather than per record, so that the arms of a comparison run the very same fights as each other while each of them runs different fights from one episode to the next. That is what makes a policy and its baseline a paired measurement: the two meet the same sites, the same budgets and the same forces, and what is left between them is the play.
    """
    arms = max(1, len(getattr(session, "arms", ()) or ()))
    return arguments.seed + INSTANCE_STRIDE * session.instance + len(session.records) // arms


def _arena_options(arguments, order: Optional[str] = None, floor: Optional[float] = None,
                   stall: Optional[int] = None, separation: Optional[float] = None) -> dict:
    """The arena settings a run actually asked for, as keywords, so that everything unasked for stands at the figure the arena states rather than at a copy of it kept here."""
    options = _given(stall_ms=stall * 1000 if stall else None, imbalance_floor=floor,
                     score=getattr(arguments, "score", None), separation=separation)
    if order is not None:
        options["decision_order"] = order
    return options


def _orders(arguments) -> list:
    """Which decision orders a run builds its arenas under, in the order they were named.

    More than one turns the run into a comparison between them, which is what the question they exist for takes: whether the left-right lean the fighting carries is made of the order the two sides are decided in can only be answered by running both orders and seeing whether the lean changes sign, and running them as two arms of one run is what holds everything else — the machine, the seed, the draws, the hour — still between the two.
    """
    named = [name.strip() for name in str(arguments.decision_order).split(",") if name.strip()]
    orders = []
    for name in named:
        if name not in DECISION_ORDERS:
            raise SystemExit(f"no decision order named {name!r}: expected one of {', '.join(DECISION_ORDERS)}")
        if name not in orders:
            orders.append(name)
    return orders or [DECISION_ORDERS[0]]


def _numbers(given, name: str, whole: bool, check=None) -> list:
    """A setting written as one number or as several, comma separated.

    Several make the arms of one run. The two settings written this way — how lopsided a draw may be and how long a quiet spell is tolerated before a fight is called — are the two that trade the same pair of things against each other, how fair the arena is against how decisive its fights are, and neither trade can be settled by argument. Run as arms they are measured against each other on the very same draws, in one run, on one machine, in one hour.
    """
    if given is None:
        return [None]
    values = []
    for written in str(given).split(","):
        if not written.strip():
            continue
        try:
            value = int(written) if whole else float(written)
        except ValueError:
            raise SystemExit(f"{name} has to be a number, not {written.strip()!r}")
        if check is not None and not check(value):
            raise SystemExit(f"{name} cannot be {value:g}")
        if value not in values:
            values.append(value)
    return values or [None]


def _pinned(arguments) -> list:
    """Which departures a run measures a layer pinned to, as arms beside the ordinary ones.

    This is how the band a policy is playing inside gets measured: pin the layer to one departure and the arena reports what giving up the choice costs. Named from the wire's own list so that adding a departure adds an ablation rather than leaving one behind.
    """
    if not arguments.pin:
        return []
    departures = []
    for written in str(arguments.pin).split(","):
        name = written.strip().upper()
        if not name:
            continue
        if name not in Deviation.__members__:
            raise SystemExit(f"no departure named {written.strip()!r}: expected one of "
                             f"{', '.join(member.name.lower() for member in Deviation)}")
        if Deviation[name] not in departures:
            departures.append(Deviation[name])
    return departures


def _floors(arguments) -> list:
    """How lopsided a draw may be, as the weaker side's share of the stronger."""
    return _numbers(arguments.imbalance_floor, "an imbalance floor", whole=False,
                    check=lambda value: 0.0 < value <= 1.0)


def _stalls(arguments) -> list:
    """How long a fight may go without a casualty before it is called, in game seconds."""
    return _numbers(arguments.stall_seconds, "a stall time", whole=True, check=lambda value: value > 0)


def _separations(arguments) -> list:
    """How far apart the two sides are put down when a fight is drawn, in world units.

    Comma separated and run as arms, for the reason the imbalance floor and the stall time are: a setting of the draw is a different instrument, and the only way to read two instruments against each other without also reading the machine, the hour and the seed is to run them as arms of one run. What that costs is that the arms are not paired — two separations draw different fights, so there is no shared draw to difference across — and what it buys is that each separation carries its own baseline, which is the figure any single arm of it is only readable beside.

    This is the sharpest of the three. The arena's own figure is inside the gap the engine halts two converging forces in, which is what makes a fight begin at all and is also what leaves the departures least to decide: two forces already exchanging fire from the first period can only be held or pulled back. Put them down further apart and whether the fight happens is itself a departure's to settle.
    """
    return _numbers(arguments.separation, "a separation", whole=False, check=lambda value: value > 0.0)


def _load(net, path: Optional[str], device) -> None:
    """Parameters to carry on from, where there are any. A path that names nothing yet is a run starting from a fresh policy, which is the ordinary way a first run begins; a path that names a file fitted to a different feature list is not tolerated the same way, because carrying on from it would train a policy that reads the board wrongly and report the run as a continuation of the one before."""
    if not path:
        return
    import os

    import torch

    from .net import EncodingRefused, load_encoded

    if not os.path.exists(path):
        log.info("no parameters at %s yet, starting from a fresh policy", path)
        return
    state = torch.load(path, map_location=device)
    try:
        avowal = load_encoded(net, state)
    except EncodingRefused as refused:
        raise SystemExit("the parameters at %s cannot be carried on from: %s" % (path, refused))
    log.info("loaded parameters from %s", path)
    if avowal:
        # Said every time rather than once when the file was avowed, because what is being carried on from is then a policy accepted on somebody's word about which features it was fitted to, and a run's log is where a later reader looks to find out what it was made of.
        log.warning("the feature list at %s is a person's word and not a fit's record: %s", path, avowal)


def _save(net, path: Optional[str]) -> None:
    """Parameters written out with the feature list they were fitted to inside them, which the network carries as a buffer so that saving is the ordinary call and the two can never be separated."""
    if not path:
        return
    import os

    import torch

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(net.state_dict(), path)
    log.info("saved parameters to %s", path)


def avow(arguments) -> int:
    """Writes into a set of parameters the feature list a person swears it was fitted to, so that parameters recorded before the list existed can be read again.

    Every loader here refuses a file that states no feature list, because nothing in such a file says which encoding produced it, and that refusal is right: the failure it prevents is a network reading two slots as something they no longer are while reporting perfectly ordinary numbers for doing so. But parameters recorded before the list was written down are not thereby wrong, and where a layer's encoding has not moved since they were fitted, a person knows something true that the file does not state. This is where they state it, into the file, and it is the only way: nothing else here opens a state dictionary to add anything to it.

    The layer is named rather than guessed at, and every file named is held to it. What a set of parameters can prove about itself is the width it reads, which says which layer it belongs to and says nothing at all about what the numbers in those slots mean — so naming the layer turns that width into a real refusal, and a run over a directory of files under one layer's name refuses the other layer's files one by one instead of stamping them. That is the accident worth being safe against: the two encodings do not move together, so a tactical file avowed under today's tactical list would afterwards load in silence and be measured under its own trained name, reporting a number about a misreading of the board.

    The file is rewritten beside itself and moved into place, so that a failure part way through leaves the parameters as they were. There is only one copy of most of these and several cost a training run to make again.
    """
    import os

    import torch

    from .net import EncodingRefused, OperationalNet, StrategicNet, TacticalNet, avowed

    paths = [path.strip() for path in str(arguments.load or "").split(",") if path.strip()]
    if not paths:
        raise SystemExit("say which parameters to avow, as in --load local/operations.pt")
    words = str(arguments.because or "").strip()
    net = {TACTICAL: TacticalNet, OPERATIONAL: OperationalNet, STRATEGIC: StrategicNet}[arguments.layer]()
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit("there are no parameters at %s to avow" % path)
        try:
            state = avowed(torch.load(path, map_location="cpu"), net, words)
        except EncodingRefused as refused:
            raise SystemExit("the parameters at %s cannot be avowed as the %s layer's: %s"
                             % (path, arguments.layer, refused))
        beside = path + ".avowing"
        torch.save(state, beside)
        os.replace(beside, path)
        log.info("the parameters at %s now state the %s feature list on your word: %s",
                 path, arguments.layer, words)
    return 0


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

#: Value-head warmup for a run that loaded parameters and was not told how much to warm. A policy loaded from somewhere — an imitation of the handwritten layer, an earlier run — arrives without its critic, so its value head is random and every advantage the first thousands of steps produce is noise the size of the returns; a policy gradient taken against that dismantles the policy before it has been paid for anything. The successful reinforced runs warmed for this many, and forgetting the flag was a way to reinforce from a random critic and lose the run without anything saying so.
WARMUP_FROM_LOAD = 5


def _warmup(arguments) -> int:
    """How many updates fit the value head alone before the policy is let move.

    Obeyed exactly when given, 0 included, since a run that means to reinforce straight from a loaded critic has to be able to say so. Left unset it is the protective default: a warmup when parameters were loaded, because their critic did not come with them, and none when starting from a fresh policy, where the critic is as new as the policy and there is nothing to protect.
    """
    if arguments.warmup is not None:
        return arguments.warmup
    chosen = WARMUP_FROM_LOAD if arguments.load else 0
    if chosen:
        log.info("warming the value head for %d update(s): parameters were loaded and no --warmup was given, "
                 "so the loaded critic is caught up before the policy moves; pass --warmup 0 to reinforce from it straight away",
                 chosen)
    return chosen


def train_tactics(arguments) -> int:
    from .net import TacticalNet

    device = _device(arguments.device)
    net = TacticalNet(**_given(width=arguments.width)).to(device)
    _load(net, arguments.load, device)
    log.info("tactical policy on %s: %d features, %d parameters",
             device, TACTICAL_SIZE, sum(p.numel() for p in net.parameters()))

    # One figure discounts the returns and telescopes the shaping, and it is stated once here so that the two cannot drift apart. An arena errand is a whole fight, which is short and ends properly, so the default is to discount it at nothing and let the score of the fight reach every decision taken in it.
    discount = arguments.discount if arguments.discount is not None else FIGHT_DISCOUNT
    trace = arguments.trace if arguments.trace is not None else FIGHT_TRACE
    log.info("paying fights scored by %s, discounting an errand at %.4f with a trace of %.4f",
             arguments.score, discount, trace)

    rollout = Rollout(discount=discount, trace=trace)
    optimiser = Optimiser(net, device=device, warmup=_warmup(arguments),
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate))
    # The batcher reads the very parameters this run's optimiser writes, so it is handed the lock the optimiser takes around a minibatch step.
    batcher = tactical_batcher(net, device=device, guard=optimiser.lock)
    trainer = Trainer(rollout, optimiser, **_given(batch=arguments.batch))
    trainer.start()

    def learnt(session, catalogue):
        # An arena fight carries one contract from the moment it is joined to the moment it is called, and nothing reissues it. What ends the errand is therefore what ends the fight, and the conditions written into the contract are left as something for the layer to read and act on rather than as something that stops it being paid.
        return LearntTactics(session, catalogue, NetworkTactics(net, device, batcher), rollout,
                             session.instance, status_terminals=False, discount=discount)

    # One arena setting to a training run: a run whose arms differed would be training one policy on two arenas and reporting one number for it.
    order, floor = _orders(arguments)[0], _floors(arguments)[0]
    stall, separation = _stalls(arguments)[0], _separations(arguments)[0]

    def arm(session):
        # Both sides script is how the arena itself is measured rather than a policy: it is the baseline a learnt layer has to beat, and it is the only setting in which what the arena produces says something about the arena rather than about whatever the policy currently happens to do.
        ours = None if arguments.script else learnt
        return Arena(session, tactics=ours, seed=_arena_seed(arguments, session),
                     **_given(outcome_weight=arguments.outcome_weight),
                     **_arena_options(arguments, order, floor, stall, separation),
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
    _report_training(sessions, "script" if (arguments.script or arguments.script_opponent) else "policy")
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
    # The drawn fights are broken out because they are the ones that say what kind of arena this was: two forces that stopped hurting each other after trading about a third of each side away, two that never reached each other before the clock, and two that destroyed each other are three different results, and only the middle one is an arena that failed to produce a fight.
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

    Leaving the policy out is the baseline: both sides are then the same handwritten layer, so the average result has to be nought by the antisymmetry of the score itself, and anything else is the arena favouring one side of the board.

    The baseline is taken alongside the policy rather than left to a separate run, and that is the default because the alternative has already produced a wrong reading twice. How far the arena leans is not a property of the arena but of the seed it was run under - the same handwritten layer against itself came out at -0.02 under one seed and +0.06 under another - so a policy's score is only readable beside the lean of the very seed it was measured on. Run as two arms of one run they alternate within each instance, which holds the seed, the machine and the hour still between them; the difference of the two is then what the policy is worth, and the interval on that difference is what says whether it is worth anything at all.

    Several policies may be named at once, comma separated, and then each is an arm beside the same baseline. That is not thrift either: two policies measured in separate runs are compared through their baselines, and two measured here are compared on the very fights both of them fought.
    """
    policies, batchers = [], []
    if arguments.load:
        import os

        from .net import TacticalNet

        device = _device(arguments.device)
        paths = [path.strip() for path in str(arguments.load).split(",") if path.strip()]
        if len(set(paths)) != len(paths):
            # Two arms of the same name would be journalled as one and reported as one, so the run would silently measure half of what it was asked for.
            raise SystemExit("the same parameters were named twice, and two arms cannot share a name")
        # How a policy plays, as an axis of the comparison rather than a setting of the run. Drawing from the distribution is what the policy does when it is operated and is therefore the figure that counts; taking the likeliest action is the same weights without the exploration, and the difference between them is the tax the run's own randomness charges. Measured as two arms of one run they meet the same fights, which is the only way that difference is separable from how the fights were drawn.
        draws = (False, True) if arguments.greedy else (False,)
        for path in paths:
            if not os.path.exists(path):
                # Refused rather than started from nothing, which is what a training run does with a missing file. A measurement that quietly scored a freshly initialised policy would produce a perfectly plausible number about a policy nobody asked about.
                log.error("there are no parameters at %s to measure", path)
                return 1
            net = TacticalNet(**_given(width=arguments.width)).to(device)
            _load(net, path, device)
            for greedy in draws:
                batcher = tactical_batcher(net, device=device, greedy=greedy)
                batchers.append(batcher)

                def learnt(session, catalogue, net=net, batcher=batcher, greedy=greedy):
                    # No rollout: this layer is being read from and not learnt from, and with nowhere to record a decision it records none.
                    return LearntTactics(session, catalogue,
                                         NetworkTactics(net, device, batcher, greedy),
                                         None, session.instance, status_terminals=False)

                # Named after the file when there are several to tell apart, and simply the duel when there is one, which is the name the journal has always carried.
                policies.append(("duel" + ("" if len(paths) == 1 else
                                           "-" + os.path.splitext(os.path.basename(path))[0])
                                 + ("-greedy" if greedy else ""), learnt))

    for pinned in _pinned(arguments):
        def fixed(session, catalogue, pinned=pinned):
            return LearntTactics(session, catalogue, PinnedDeparture(pinned.value), None,
                                 session.instance, status_terminals=False)

        policies.append((f"duel-always-{pinned.name.lower()}", fixed))

    def build(policy, order: str, floor: Optional[float], stall: Optional[int],
              separation: Optional[float]):
        def arm(session):
            # The opponent is left unnamed, which is what puts the handwritten layer on the other side of every fight. That is the thing being measured against, so it is not something this run offers a choice about.
            return Arena(session, tactics=policy, seed=_arena_seed(arguments, session),
                         **_arena_options(arguments, order, floor, stall, separation))

        return arm

    orders, floors = _orders(arguments), _floors(arguments)
    stalls, separations = _stalls(arguments), _separations(arguments)

    def named(side: str, order: str, floor: Optional[float], stall: Optional[int],
              separation: Optional[float]) -> str:
        # Named after whatever is actually varying across the arms, so that a plain measurement keeps the two names the journal has always used and a comparison of arena settings says which setting each arm was.
        return (side + (f"-{order}" if len(orders) > 1 else "")
                + (f"-floor{floor:g}" if len(floors) > 1 and floor is not None else "")
                + (f"-stall{stall:d}" if len(stalls) > 1 and stall is not None else "")
                + (f"-sep{separation:g}" if len(separations) > 1 and separation is not None else ""))

    # The policies' sides and the baseline's side of the comparison. A run with nothing loaded is the baseline alone, which is how the arena itself is measured; a run with a policy takes both unless the baseline was explicitly declined.
    sides = list(policies) + ([("duel-baseline", None)] if not policies or arguments.baseline else [])
    arms = [(named(side, order, floor, stall, separation),
             build(policy, order, floor, stall, separation))
            for order in orders for floor in floors for stall in stalls
            for separation in separations for side, policy in sides]
    # The baseline is written down as the baseline. It is a different quantity from a policy's score rather than a run of it that happens to have scored nought, and the likeliest way to confuse the two is to have journalled them under one name. Under one journal that is the arm each episode carries; the file is named for what the run was for.
    baselines = {name for name, _ in arms if name.startswith("duel-baseline")}
    journal = Journal(arguments.record or default_path("duel" if arguments.load else "duel-baseline"))
    log.info("measuring %s over %d episode(s) each on %d instance(s)",
             ", ".join(name for name, _ in arms), arguments.episodes, arguments.instances)
    try:
        sessions = _serve(arguments, arms, _episode(arguments, arena=True), journal)
    finally:
        journal.close()
    for batcher in batchers:
        batcher.stop()

    # The batching is reported off the first policy's server, which is the one whose window and queue the run was configured with; a second policy's is the same arrangement over a share of the same calls.
    _report_arena(sessions, batchers[0] if batchers else None)
    _report_duel(sessions, baselines)
    return 0


#: The two readings of a fight a run reports, as (what the journal calls the mean, what it calls the spread, what the history row calls it, what to call it in a log).
#:
#: Both, always, whichever one was paid. The sparse reading is what every ceiling this project has quoted was measured on and dropping it would make a new run unreadable against any of them; the health reading is the one that also counts the damage left standing on the survivors, which is where two thirds of fights end and which the other cannot see until it has killed something. A run costs the same either way — both are computed as a fight is called — so there is no reason to report one.
READINGS = (("outcome_mean", "outcome_sd", "outcome", "scored on bodies"),
            ("health_outcome_mean", "health_outcome_sd", "outcome_health", "scored on health"))


def _summarise(records, mean_key: str = "outcome_mean", sd_key: str = "outcome_sd") -> Summary:
    """The fights of a set of episodes as one count, one mean and one spread.

    Recombined from the episode records rather than kept as one long list of fights. A record carries its episode's mean, its spread and how many fights it was taken over, and those three are enough to reconstitute both figures over the whole set exactly, however many episodes and instances it ran across.
    """
    counted = [(int(record.statistics.get("fought", 0)),
                float(record.statistics.get(mean_key, 0.0)),
                float(record.statistics.get(sd_key, 0.0)))
               for record in records]
    counted = [entry for entry in counted if entry[0] > 0]
    total = sum(count for count, _, _ in counted)
    if not total:
        return Summary(0, 0.0, 0.0)
    mean = sum(count * value for count, value, _ in counted) / total
    # The spread each record carries is over its own fights and around its own mean, so the two are put back together by pooling the second moments and taking the whole set's mean off afterwards.
    spread = sum(count * (deviation ** 2 + value ** 2) for count, value, deviation in counted) / total - mean ** 2
    # Quoted on the sample rather than on the population, matching how every other scatter in this project is reported and how the sample sizes were computed.
    return Summary(total, mean, math.sqrt(max(0.0, spread) * total / (total - 1)) if total > 1 else 0.0)


def _report_score(summary: Summary, name: str, episodes: int) -> bool:
    """One set of fights: how they went on average, how widely that scattered, and how many fights an assertion of that average would take. True when the run has already fought as many as its own average would need."""
    log.info("%s: %d fight(s) over %d episode(s), outcome %+.4f, spread %.4f",
             name, summary.n, episodes, summary.mean, summary.sd)
    needed = episodes_for(summary.sd, abs(summary.mean))
    # A spread of nought is one that was never measured rather than one measured to be small, and it is what a single fight or a run of identical fights produces. The sizing arithmetic answers nought fights for it, quite correctly given a scatter of nought, so the claim has to be gated on there having been a scatter at all: without that a run of one engagement sizes its own claim at no engagements and declares itself sufficient.
    enough = summary.n > 1 and summary.sd > 0.0 and needed <= summary.n
    if needed >= UNBOUNDED_EPISODES:
        log.info("the average result is exactly nought, which is not a difference and which no number of fights would establish")
    elif summary.sd <= 0.0:
        log.info("all %d fight(s) came out at %+.4f, so this run measured no spread at all and there is nothing to size a claim against",
                 summary.n, summary.mean)
    else:
        log.info("claiming an average of %+.4f at that spread takes %d fight(s), and %d were fought: %s",
                 summary.mean, needed, summary.n, "enough" if enough else "not enough yet")
    return enough


def _interval(first: Summary, second: Summary) -> float:
    """Two standard errors on the difference of two means, which is the width the difference has to clear before the interval around it stops holding nought."""
    if first.n < 2 or second.n < 2:
        return 0.0
    return 2.0 * math.sqrt(first.sd ** 2 / first.n + second.sd ** 2 / second.n)


def _report_difference(first_name: str, first: Summary, second_name: str, second: Summary) -> None:
    """What one arm was worth over another, which for a policy against its own seed's baseline is the whole of what a measurement run is for.

    Reported as the difference and the interval around it rather than as two numbers to be read against each other, because the two numbers have been read wrongly twice: a policy that scored above nought was taken for a policy that had beaten the handwritten layer, when the arena it was measured on was itself scoring above nought under that seed.
    """
    if not first.n or not second.n:
        return
    difference = first.mean - second.mean
    interval = _interval(first, second)
    log.info("%s less %s: %+.4f, 2 standard errors %.4f, so the interval is %+.4f to %+.4f and %s nought",
             first_name, second_name, difference, interval,
             difference - interval, difference + interval,
             "excludes" if interval > 0.0 and abs(difference) > interval else "holds")


def _fights_by_draw(sessions, key: str = "outcome") -> dict:
    """Every fight of every arm, keyed by the draw it was fought on: which instance, which round of the arms, and which fight of the episode.

    The arms of a run are handed the same arena seed in the same round, so the fight under one key is the same fight in every arm — the same site, the same two budgets, the same imbalance, the same two forces. Keyed that way the arms can be differenced fight by fight, and the difference is then free of the only thing that makes the score scatter, which is how the fight was drawn rather than how it was fought.

    The round is counted per arm rather than read off the episode number, because the arms alternate within an instance and the episode number counts both.
    """
    fights: dict = {}
    for session in sessions:
        rounds: dict = {}
        for record in session.records:
            index = rounds.get(record.arm, 0)
            rounds[record.arm] = index + 1
            for fight in record.statistics.get("history", ()):
                if key in fight and "index" in fight:
                    drawn = (record.instance, index, int(fight["index"]))
                    fights.setdefault(record.arm, {})[drawn] = float(fight[key])
    return fights


def _report_paired(first_name: str, second_name: str, fights: dict) -> None:
    """What one arm was worth over another on the fights both of them fought.

    This is the measurement the baseline is taken alongside the policy for. Unpaired, the difference of two arms carries the whole scatter of how fights are drawn — a two-to-one draw scores half a point whoever is playing — and that scatter is several times anything a policy has ever moved. Paired on the draw it cancels, and what is left is the play.
    """
    ours, theirs = fights.get(first_name, {}), fights.get(second_name, {})
    shared = sorted(set(ours) & set(theirs))
    if len(shared) < 2:
        return
    differences = Summary.of([ours[key] - theirs[key] for key in shared])
    interval = 2.0 * differences.sd / math.sqrt(differences.n)
    log.info("%s less %s on the %d fight(s) both drew: %+.4f, 2 standard errors %.4f, interval %+.4f to %+.4f, %s nought",
             first_name, second_name, differences.n, differences.mean, interval,
             differences.mean - interval, differences.mean + interval,
             "excludes" if interval > 0.0 and abs(differences.mean) > interval else "holds")
    needed = episodes_for(differences.sd, abs(differences.mean))
    if needed < UNBOUNDED_EPISODES:
        log.info("claiming that difference at that spread takes %d paired fight(s), and %d were fought: %s",
                 needed, differences.n, "enough" if needed <= differences.n else "not enough yet")


def _report_duel(sessions, baselines) -> None:
    """The score of a measurement run, arm by arm, and then arm against arm.

    Every pair is compared rather than only the pair a run was built to compare, because the arms of a run are few and which pair carries the question differs by run: a policy against its seed's baseline for a measurement, one arena setting against another for a question about the arena.
    """
    records = [record for session in sessions for record in session.records]
    if not records:
        return
    # In the order the arms were run rather than sorted, so that a policy's arm is reported before the baseline it is read against.
    names = list(dict.fromkeys(record.arm for record in records))
    for mean_key, sd_key, fight_key, reading in READINGS:
        if not any(mean_key in record.statistics for record in records):
            # A record written before a reading existed carries nothing to report it from, and reporting nought would be reporting a measurement nobody took.
            continue
        log.info("---- %s ----", reading)
        drawn = _fights_by_draw(sessions, fight_key)
        summaries = {}
        for name in names:
            theirs = [record for record in records if record.arm == name]
            summary = _summarise(theirs, mean_key, sd_key)
            summaries[name] = summary
            if not summary.n:
                log.info("%s: no fight was called, so there is nothing to score", name)
                continue
            enough = _report_score(summary, name, len(theirs))
            if name in baselines:
                if enough:
                    log.warning("%s had the handwritten layer on both sides, so this average has to be nought and it is %+.4f over enough fights to say so: the arena favours one side of the board under this seed, and a policy measured on it is only readable as the difference from this figure",
                                name, summary.mean)
                else:
                    log.info("%s had the handwritten layer on both sides and %d fight(s) have not separated its average of %+.4f from nought, which is as much as this run says about whether the arena is even",
                             name, summary.n, summary.mean)
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                _report_difference(first, summaries[first], second, summaries[second])
                _report_paired(first, second, drawn)


def _report_training(sessions, opponent: str) -> None:
    """What the fights of a training run scored, which is a measurement the run has already paid for and used to throw away.

    A training run scores every fight it builds, exactly as a measurement run does and in the same quantity - the run that produced the policy measured last was three times the size of the duel that measured it. What it is not is a measurement of the parameters that were saved: they moved throughout, so the figure is an average over every policy the run passed through rather than over the one it ended at. The later half is reported beside the whole for that reason; on a run that improved, the whole is a lower bound on the end of it.

    It is also only a score at all when the opponent was fixed. With the run's own policy on both sides the score is antisymmetric by construction and its average is a self-check on the arena rather than anything about the policy, so it is reported as that instead.
    """
    records = [record for session in sessions for record in session.records]
    if not records:
        return
    if not _summarise(records).n:
        return
    log.info("the fights this run scored while training, which are a measurement it has already paid for:")
    # Halved within each instance rather than across the run, because the instances run concurrently and finish different numbers of episodes; the second half of every instance is the second half of the run, and the episode numbers an instance reports are its own and need not start at one.
    later = [record for session in sessions for record in session.records[len(session.records) // 2:]]
    for mean_key, sd_key, _, reading in READINGS:
        if not any(mean_key in record.statistics for record in records):
            continue
        _report_score(_summarise(records, mean_key, sd_key), f"training, whole run, {reading}", len(records))
        if later and len(later) < len(records):
            _report_score(_summarise(later, mean_key, sd_key), f"training, later half, {reading}", len(later))
    if opponent == "script":
        log.info("the opposing side was the handwritten layer throughout, so this is the same quantity a duel reports, taken over a moving policy: read it against a baseline run under this same seed, since how far the arena leans is a property of the seed")
    else:
        log.info("the opposing side was this run's own policy, so the score is antisymmetric by construction and its average says nothing about the policy: it is a self-check on the arena, which has to be nought")


# ---- the operational run ------------------------------------------------------------------

def train_operations(arguments) -> int:
    from .net import OperationalNet

    device = _device(arguments.device)
    net = OperationalNet().to(device)
    _load(net, arguments.load, device)
    log.info("operational policy on %s: %d features, %d parameters",
             device, OPERATIONAL_SIZE, sum(p.numel() for p in net.parameters()))

    rollout = Rollout()
    optimiser = Optimiser(net, device=device, two_headed=True, warmup=_warmup(arguments),
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate))
    # The batcher reads the very parameters this run's optimiser writes, so it is handed the lock the optimiser takes around a minibatch step.
    batcher = operational_batcher(net, device=device, guard=optimiser.lock)
    trainer = Trainer(rollout, optimiser, **_given(batch=arguments.batch))
    trainer.start()

    # Trained layers held still beneath this one, which is the half of the learning order that had no wiring in a match: the arena could freeze a trained fighter under its board, and an ordinary match could not.
    frozen = frozen_layers(arguments.frozen, arguments.device, training=OPERATIONAL)
    arm = learning_arm(
        OPERATIONAL,
        lambda session: NetworkOperations(net, device, batcher),
        rollout,
        intruders=(lambda session: Intruder(seed=arguments.seed + session.instance,
                                            instance=session.instance)) if arguments.intruder else None,
        frozen_for=frozen.build if frozen.deciders else None,
    )
    journal = Journal(arguments.record or default_path("operations"))
    try:
        _serve(arguments, [("operations", arm)], _episode(arguments, arena=False), journal)
    finally:
        journal.close()
    report = trainer.finish()
    frozen.stop()
    batcher.stop()
    _save(net, arguments.save)
    log.info("batched inference averaged %.1f per call", batcher.batch_size)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    return 0


# ---- the strategic run --------------------------------------------------------------------

#: Decisions a strategic update is collected from, which is a quarter of what the other two layers use, and the arithmetic says why. This layer decides once per side per ten game seconds, so eight instances running five-minute matches at ten times speed produce about two hundred and forty decisions a minute of wall clock, against the thousands a second the tactical layer produces. At the usual thousand-and-twenty-four a batch would take four minutes to fill and a run of an hour would take fifteen steps; at this figure a batch is about half a minute, which is where the design says to keep it — a batch collected under parameters much older than the ones it updates is what the clipped ratio is a safety valve for, and it is a safety valve rather than a licence.
STRATEGIC_BATCH = 256


def train_strategy(arguments) -> int:
    """Trains the strategic layer in ordinary matches, which is the only board its choice is made on.

    There is no arena here and there is not going to be one. The other two layers got constructed boards because what they decide can be cut out of a match and staged — one engagement, one deployment — and because the thing that buried their choice was the economy. The strategic layer's choice IS the economy, and what it decides between is how a whole match is to be spent: a posture pays off in ground taken twenty periods later or in an army that exists at minute twelve. Construct that and what is left is the match.

    So the match's own result is the terminal, and this is the one layer in the design that is paid it. It arrives from outside through the session, which knows how the episode ended, exactly as an arena's terminal arrives from the runner: the layer sees periods and cannot see an ending.

    The errand is the whole match, so it is discounted at nothing by default, for the arithmetic the constructed fight is discounted at nothing by: it is one bounded errand with a real terminal, and there is no reason to make the terminal reach the opening of a match at less than its own weight. What that costs is variance — every decision of a match shares one terminal and the advantage of one is separated from another's only by the critic — and what a discount below one would buy instead is the shaping reaching the policy at all, since at one it is an action-independent offset. Both are `--discount` and `--trace` so that the choice can be swept rather than argued.
    """
    from .net import StrategicNet

    device = _device(arguments.device)
    net = StrategicNet(**_given(width=arguments.width)).to(device)
    _load(net, arguments.load, device)
    log.info("strategic policy on %s: %d features, %d parameters",
             device, STRATEGIC_SIZE, sum(p.numel() for p in net.parameters()))

    # One figure discounts the returns and telescopes the shaping, stated once so the two cannot drift apart.
    discount = arguments.discount if arguments.discount is not None else FIGHT_DISCOUNT
    trace = arguments.trace if arguments.trace is not None else FIGHT_TRACE
    log.info("paying the match's own score, discounting a match at %.4f with a trace of %.4f", discount, trace)

    rollout = Rollout(discount=discount, trace=trace)
    optimiser = Optimiser(net, device=device, warmup=_warmup(arguments),
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate))
    batcher = strategic_batcher(net, device=device, guard=optimiser.lock)
    trainer = Trainer(rollout, optimiser, batch=arguments.batch or STRATEGIC_BATCH)
    trainer.start()

    # Trained layers held still beneath this one. The design's order puts the strategic layer last precisely so that what is below it can be the layers already improved rather than the handwritten ones; without any named, the layers below are the script, which is also what this layer is measured against.
    frozen = frozen_layers(arguments.frozen, arguments.device, training=STRATEGIC)

    def build(session) -> LearningPolicy:
        policy = LearningPolicy(session, STRATEGIC, NetworkStrategy(net, device, batcher),
                                rollout, session.instance,
                                frozen=frozen.build() if frozen.deciders else None)
        # The layer's shaping has to telescope with the discount the returns are taken at, so the run's one figure reaches it too.
        policy.strategy.reward.discount = discount
        if arguments.intruder:
            intruder = Intruder(seed=arguments.seed + session.instance, instance=session.instance)
            intruder.organisation = policy.organisation
            policy.outside.append(intruder)
        return policy

    journal = Journal(arguments.record or default_path("strategy"))
    try:
        sessions = _serve(arguments, [("strategy", build)], _episode(arguments, arena=False), journal)
    finally:
        journal.close()
    report = trainer.finish()
    frozen.stop()
    batcher.stop()
    _save(net, arguments.save)
    log.info("batched inference averaged %.1f per call over %d call(s)", batcher.batch_size, batcher.calls)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    _report_matches(sessions)
    return 0


def _report_matches(sessions) -> None:
    """What the matches this run played came to, in the quantity the strategic layer is paid: the project's own episode score, pooled over the run and over its later half.

    Halved within each instance rather than across the run, exactly as the arena's training report is halved, because the instances finish different numbers of episodes and the second half of every instance is the second half of the run.

    It is a training figure and not a measurement. The policy moved while it was collected and the opponent is whatever the room put there, so what it says is whether the run went anywhere; what says whether the layer is better than the rule is a separate run of arms against the same opponents (`python -m rwintel.eval`).
    """
    from ..eval.scoring import score as episode_score

    records = [record for session in sessions for record in session.records]
    if not records:
        return
    later = [record for session in sessions for record in session.records[len(session.records) // 2:]]
    for name, subset in (("whole run", records), ("later half", later)):
        if not subset or (name == "later half" and len(subset) >= len(records)):
            continue
        summary = Summary.of([episode_score(record) for record in subset])
        interval = 2.0 * summary.sd / math.sqrt(summary.n) if summary.n > 1 else 0.0
        log.info("match score, %s: %d episode(s), %+.4f, 2 standard errors %.4f",
                 name, summary.n, summary.mean, interval)
    decided = sum(1 for record in records if record.winner >= 0)
    log.info("%d of %d match(es) were decided rather than cut off by the clock", decided, len(records))


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
                         seed=_arena_seed(arguments, session),
                         **_arena_options(arguments, _orders(arguments)[0], _floors(arguments)[0],
                                          _stalls(arguments)[0], _separations(arguments)[0]))

        episode = _episode(arguments, arena=True)
    else:
        def arm(session):
            return LearningPolicy(session, arguments.layer, None, rollout, session.instance)

        episode = _episode(arguments, arena=False)

    journal = Journal(arguments.record_episodes or default_path("collect"))
    try:
        _serve(arguments, [("script", arm)], episode, journal)
    finally:
        journal.close()
    rollout.cut_all()
    steps = rollout.drain(keep_tainted=True)
    log.info("collected %d decision(s)", len(steps))
    _write_steps(steps, arguments.record or "local/teacher.jsonl", arguments.layer)
    return 0


def _write_steps(steps, path: Optional[str], layer: str) -> None:
    """The feature list the decisions were written under, and then one decision per line, in the form anything fitting to them reads.

    The list comes first because a teacher file records an encoding as much as it records a policy, and the decisions themselves cannot say which one: a state of the right length written under an older feature list fits without complaint and yields a network reading two slots as something they are no longer. Stated once at the head rather than on every line, because it is a fact about the file and not about the decision — and the file is opened with truncation here, so the way anybody makes a larger teacher is by joining two of these end to end, which leaves the second run's head in the middle where the reader checks it again and passes over it.

    What was legal is written down beside what was chosen, as flags rather than as weights. Without it a decision is not reconstructible: the operational layer picks a region out of the few that exist on the board it saw, and a reader that could not tell which those were would be fitting a distribution over twenty-four regions of which most were never on offer.
    """
    if not path or not steps:
        return
    import json
    import os

    from .imitation import feature_names, feature_recipe

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(json.dumps({"layer": layer, "encoding": list(feature_names(layer)),
                              "recipe": feature_recipe(layer)},
                             separators=(",", ":")) + "\n")
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
    from .net import OperationalNet, StrategicNet, TacticalNet

    device = _device(arguments.device)
    if arguments.layer == TACTICAL:
        net = TacticalNet(**_given(width=arguments.width))
    elif arguments.layer == STRATEGIC:
        net = StrategicNet(**_given(width=arguments.width))
    else:
        net = OperationalNet()
    net = net.to(device)
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
    parser.add_argument("what", choices=["tactics", "operations", "strategy", "collect", "clone", "duel", "avow"])
    parser.add_argument("--layer", default=None, choices=list(LAYERS),
                        help="which layer to collect, to clone or to avow. Collecting and cloning default to "
                             "the tactical layer; avowing has no default and must be told, because the width "
                             "a file reads is the only thing that can refuse one layer's parameters offered as "
                             "another's, and it can only do that once a layer has been named")
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
    parser.add_argument("--load", default=None,
                        help="parameters to start from. A duel takes several, comma separated, and measures "
                             "each as an arm against the one baseline")
    parser.add_argument("--pin", default=None,
                        help="departures to measure a layer pinned to, comma separated, as arms of a duel. "
                             "This is the ablation the band a policy plays inside is read from")
    parser.add_argument("--frozen", default=None,
                        help="trained layers to hold still beneath the one being trained, written "
                             "layer:path and separated by commas, as in "
                             "tactics:local/tactics.pt,operations:local/operations.pt. They are read at their "
                             "likeliest action and record nothing. Left out, the layers below are the handwritten "
                             "script, which is what every run so far was trained against; given, the run is a "
                             "different environment and pairs only with runs made under the same one")
    parser.add_argument("--save", default=None, help="where to write the parameters afterwards")
    parser.add_argument("--batch", type=int, default=None,
                        help="rows in one gradient step: the steps that make a reinforcement update when "
                             "training, the teacher's decisions in one minibatch when cloning")
    parser.add_argument("--stall-seconds", default=None,
                        help="game seconds a fight may go without a casualty before the arena calls it, "
                             "which is how decisive its fights are. Comma separated, they run as arms")
    parser.add_argument("--separation", default=None,
                        help="world units between the two sides when a fight is put down. The arena's own figure "
                             "is inside the gap the engine halts two converging forces in, which is what makes a "
                             "fight begin at all; a larger one puts the departures rather than the exchange in "
                             "charge of the fight, and is how much of the score the tactical choice can move at "
                             "all is asked. Comma separated, they run as arms, and each carries its own baseline "
                             "because a run that moves it is a different instrument")
    parser.add_argument("--imbalance-floor", default=None,
                        help="the weaker side's smallest share of the stronger when a fight is drawn. Lower "
                             "draws more lopsided fights, which are more decisive. Comma separated, they run "
                             "as arms")
    parser.add_argument("--decision-order", default=DECISION_ORDERS[0],
                        help="which side's departure is decided first in a period: " +
                             ", ".join(DECISION_ORDERS) + ". Naming more than one, comma separated, runs "
                             "them as arms of one comparison, which is how the question of whether the order "
                             "is what a left-right lean is made of gets answered")
    parser.add_argument("--width", type=int, default=None,
                        help="hidden units per layer in the tactical network, which the measured cost of inference leaves room to raise")
    parser.add_argument("--entropy", type=float, default=None,
                        help="how hard the objective pushes the policy towards choosing evenly, which a run starting from an imitation wants much less of than one starting from noise")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--outcome-weight", type=float, default=None,
                        help="how much of a terminal the score of a fight is worth in the arena")
    parser.add_argument("--score", default=BY_HEALTH, choices=list(SCORES),
                        help="how a fight is scored for the terminal it pays: on what is left standing, or on "
                             "what is left standing weighted by the health it has left. Both are computed and "
                             "both are reported whichever is chosen; this settles only which one is paid")
    parser.add_argument("--discount", type=float, default=None,
                        help="how much a decision's future is discounted per period. One discounts nothing, "
                             "which is what an errand that is a whole fight wants: the score of the fight then "
                             "reaches every decision taken in it. The shaping telescopes at whatever this is")
    parser.add_argument("--trace", type=float, default=None,
                        help="how far the advantage estimator trades bias against variance")
    parser.add_argument("--warmup", type=int, default=None,
                        help="updates at the start that fit the value head alone, holding the trunk and the "
                             "policy still. Meant for a run started from imitated parameters, whose critic "
                             "did not come with them. Left unset it defaults to a short warmup when --load "
                             "supplies parameters and to none when starting from a fresh policy; pass it "
                             "explicitly, 0 included, to override that")
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
                        help="measure each policy a second time taking the likeliest action rather than "
                             "drawing one, as an arm beside the drawing one. Drawing is what the policy does "
                             "when it is operated, so it stays; what this adds is the same weights without "
                             "the exploration, on the same fights")
    parser.add_argument("--intruder", action="store_true",
                        help="inject the script intruder, which the design requires for the operational layer")
    parser.add_argument("--script", action="store_true",
                        help="run the handwritten layer on both sides, which is the baseline and the way to measure the arena itself")
    parser.add_argument("--script-opponent", action="store_true",
                        help="fight the script tactical layer rather than the policy being trained")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false",
                        help="duel without taking the handwritten layer against itself alongside. The "
                             "baseline is taken by default because how far the arena leans is a property of "
                             "the seed, so a policy's score is only readable beside the lean of the very "
                             "seed it was measured under")
    parser.add_argument("--because", default=None,
                        help="why a person believes a set of parameters was fitted to the feature list now in "
                             "force, written into the file beside that list when avowing. Required there, and "
                             "kept, because it is the only evidence the file will ever carry for a claim "
                             "nothing in it can check")
    parser.add_argument("--record", default=None, help="where decisions or episodes are written")
    parser.add_argument("--record-episodes", default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if arguments.max_seconds <= 0:
        arguments.max_seconds = 300 if arguments.what in ("operations", "strategy") else ARENA_SECONDS
    if arguments.what == "avow" and arguments.layer is None:
        # No default here, where every other command has one. An avowal writes one layer's feature list into a file on a person's word, and a default would let the wrong layer's list be written by saying nothing at all — which is the one accident this command has to be safe against, since a file avowed under the wrong list afterwards loads in silence.
        parser.error("say which layer's parameters are being avowed, with --layer and one of %s"
                     % ", ".join(LAYERS))
    if arguments.layer is None:
        arguments.layer = TACTICAL

    if arguments.what == "avow":
        return avow(arguments)
    if arguments.what == "tactics":
        return train_tactics(arguments)
    if arguments.what == "operations":
        return train_operations(arguments)
    if arguments.what == "strategy":
        return train_strategy(arguments)
    if arguments.what == "clone":
        return clone(arguments)
    if arguments.what == "duel":
        return duel(arguments)
    return collect(arguments)


if __name__ == "__main__":
    sys.exit(main())
