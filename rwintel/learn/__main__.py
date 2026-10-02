"""Runs a training pass, collects what the script chain does as data to start one from, imitates it, measures a policy against it, or works on what was recorded.

    python -m rwintel.learn tactics    --instances 4 --save local/models/tactics.pt
    python -m rwintel.learn operations --instances 13 --load local/models/ops-bc.pt --save local/models/ops-rl.pt --anchor 0.5 --difficulty -1 --intruder
    python -m rwintel.learn economy    --instances 12 --load local/models/eco-bc2.pt --save local/models/eco-rl.pt --warmup 2 --anchor 0.5 --difficulty 0
    python -m rwintel.learn collect    --layer tactics --instances 4
    python -m rwintel.learn collect    --layer tactics --instances 4 --both-sides --pin withdraw --map Lake --map Beach
    python -m rwintel.learn collect    --layer economy --instances 4 --explore 0.1 --difficulty 0
    python -m rwintel.learn clone      --layer tactics --dataset local/datasets/tactics/collect-<stamp> --save local/models/tactics-bc.pt
    python -m rwintel.learn clone      --layer operations --dataset local/datasets/operations/collect-<stamp> --dataset local/datasets/operations/replay-<stamp>:0.2
    python -m rwintel.learn offline    --layer tactics --dataset local/datasets/tactics/collect-<stamp> --method bc --net set --save local/models/tactics-set.pt
    python -m rwintel.learn offline    --layer tactics --dataset local/datasets/tactics/collect-<stamp> --method distill --teacher local/models/tactics-set.pt --net flat
    python -m rwintel.learn offline    --layer tactics --dataset local/datasets/tactics/collect-<stamp> --method iql --load local/models/tactics-set.pt
    python -m rwintel.learn offline    --layer tactics --method iql --follow --watch local/datasets/tactics/actors --load local/models/tactics-set.pt --save local/models/tactics-actor.pt
    python -m rwintel.learn collect    --layer tactics --student local/models/tactics-actor.pt --reload-seconds 30 --dataset local/datasets/tactics/actors/<run>
    python -m rwintel.learn online     --layer tactics --dataset local/datasets/tactics/collect-<stamp> --load local/models/tactics-set.pt --save local/models/tactics-actor.pt --count 8 --episodes 20
    python -m rwintel.learn bench      --layer tactics --net set --device cuda
    python -m rwintel.learn duel       --load local/models/tactics.pt --instances 8
    python -m rwintel.learn duel       --from local/episodes/duel-<stamp>.jsonl local/episodes/duel-<stamp>.jsonl
    python -m rwintel.learn matchups   --instances 8 --episodes 10
    python -m rwintel.learn dataset    inspect local/datasets/tactics/collect-<stamp> --verify
    python -m rwintel.learn dataset    reencode local/datasets/tactics/collect-<stamp> --out local/datasets/tactics/collect-<stamp>-r2

Start this first and then the game instances, as with every other runner here. Cloning, learning offline, re-reporting a duel, working on a dataset and timing inference (`bench`) are the exceptions: they read files and touch no game at all. `online` is the other exception: it starts the following learner and then the actor games itself, through the runtime launcher (`rwintel.learn.online`). Every run that plays the layer under study records its decisions to local/datasets/<layer>/<what>-<stamp>/ unless told not to (`--no-dataset`); episodes go to local/episodes/<what>-<stamp>.jsonl under the same stamp, and the log to local/logs/learn. A cloning run takes any number of datasets, each with a weight on its decisions, which is how the decisions a replay playback inferred from a person's play are mixed in with the script's.

The four make one order of work. The collecting run records the handwritten layer's decisions, the cloning run fits a network to them, the training run improves that network against the arena while warming its value head first, and the duelling run measures what came out against the handwritten layer it started from. None of the four is required by the others -a policy can be trained from noise and measured without ever having been cloned -but skipping the first two spends the early part of a training run rediscovering a rule ladder that was already written down.

The order the layers are trained in is not a preference. The tactical layer goes first because it can be trained without playing matches at all -engagements are constructed on an empty board and fought in a minute apiece -while the operational layer and the economy need whole matches and are therefore an order of magnitude more expensive per decision. Settling the cheap layer while the thing it will be frozen against is still cheap is the right way round.

Neither training run pauses to update. The games do not stop, so a batch is collected while the parameters that collected it are already moving; that is what the clipped ratio in the optimiser is for, and it is why the batch is small.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Optional

from .. import paths
from ..control.intruder import Intruder
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from ..eval.journal import Journal, read as read_journal
from ..eval.sampling import UNBOUNDED_EPISODES, Summary, episodes_for, standard_error
from ..wire import Deviation
from .arena import DECISION_ORDERS, Arena
from .dataset import SHARD_DECISIONS
from .deciders import (
    Labelled,
    NetworkChoice,
    NetworkOperations,
    PinnedDeparture,
    PinnedOperations,
    ScriptChoice,
    choice_batcher,
    operational_batcher,
)
from ..control.policy.encoding import ECONOMIC_SIZE, OPERATIONAL_SIZE, TACTICAL_SIZE
from .layers import LearntOperations, LearntTactics
from .policy import ECONOMIC, LAYERS, OPERATIONAL, TACTICAL, LearningPolicy, learning_arm
from .reward import BY_HEALTH, REWARDS, SCORES, EconomicTerms, OperationalTerms, TacticalTerms, preset
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

#: Game seconds a match played for the operational layer runs to by default, which is inside the ten to twenty-five minutes the design gives that layer's episodes.
MATCH_SECONDS = 900

#: Threads the tensor library is allowed. Bounded rather than left at the default, which is one per core, because the games this process is learning from are on the same cores.
TORCH_THREADS = 2


def _device(name: Optional[str]):
    """Where the networks run: the processor unless told otherwise, and with `auto` a graphics card when there is one.

    The flat tactical policy is about ten thousand parameters and the operational one about two hundred thousand, so in a game-playing run every call is dominated by the cost of dispatching it rather than by the arithmetic, and the processor answers a batch of decisions faster than a graphics card does. Runs that only read datasets (`clone`, `offline`) default to `auto`.
    """
    import torch

    # Two threads, because the networks are small enough that one is nearly as fast and the machine is not idle. A dozen game instances are running beside this process and each wants a core; a tensor library that helps itself to all of them turns every update into a fight with the simulation it is learning from, and the simulation is the part that cannot be made faster.
    torch.set_num_threads(TORCH_THREADS)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name) if name else torch.device("cpu")


def _serving_device(name: Optional[str], path: Optional[str], share: Optional[float] = None):
    """Where a game-playing run serves the network in the file at `path`: an explicit `--device` wins; otherwise a set network goes to the graphics card when there is one, and every other file stays on the processor. On a graphics card the process is limited to `share` of it (`models.limit_card`)."""
    from . import models

    if name or not path or not os.path.exists(path):
        device = _device(name)
        models.limit_card(device, share)
        return device
    if models.read(path)["kind"] != "set":
        return _device(None)
    import torch

    if torch.cuda.is_available():
        device = _device("cuda")
        models.limit_card(device, share)
        return device
    log.warning("%s holds a set network and there is no graphics card, so it is served on the processor; "
                "the network distilled from it (`offline --method distill --net flat`) is the one meant for the processor", path)
    return _device(None)


def _network(layer: str, path: Optional[str], device, fresh):
    """The network in the model file at `path`, flat or set as the file says, or `fresh()` when there is no file there yet."""
    from . import models

    if path and os.path.exists(path):
        net = models.load(path, layer, device)
        net.train()
        log.info("loaded the %s %s network from %s", models.kind_of(net), layer, path)
        return net
    if path:
        log.info("no parameters at %s yet, starting from a fresh policy", path)
    return fresh().to(device)


def _given(**asked) -> dict:
    """Only the options somebody actually asked for, as keywords.

    What is left out then stands at the figure stated by the module that owns it rather than at a copy of that figure kept here. The two batch sizes are the case that makes this worth a helper: the steps in a reinforcement update and the teacher's decisions in one gradient step are different numbers living in different files, and one option carries both.
    """
    return {name: value for name, value in asked.items() if value is not None}


def match_terms(arguments, layer: str):
    """The terms a match layer's run is paid under and the trace it is collected at: the named reward (`--reward`, the default one unless given) with every figure given on the command line laid over it."""
    from .predictor import load as load_predictor

    named, trace = preset(layer, arguments.reward or REWARDS[0])
    given = _given(discount=arguments.discount, terminal_weight=arguments.terminal_weight,
                   value_flow=arguments.value_flow, ground_flow=arguments.ground_flow)
    if layer == OPERATIONAL:
        given.update(_given(local_exchange=arguments.local_exchange, achievement_weight=arguments.achievement_weight))
    elif arguments.local_exchange is not None or arguments.achievement_weight is not None:
        raise SystemExit("--local-exchange and --achievement-weight pay the operational layer, not the economy")
    if arguments.predictor:
        given["predictor"] = load_predictor(arguments.predictor)
    kind = OperationalTerms if layer == OPERATIONAL else EconomicTerms
    terms = kind(**{**named, **given})
    trace = arguments.trace if arguments.trace is not None else trace
    log.info("paying the %s layer under %s at a trace of %.4f", layer,
             {name: value for name, value in terms.as_dict().items() if name != "predictor"}, trace)
    return terms, trace


def _repriced(arguments) -> Optional[dict]:
    """The terms an offline run prices its recorded rewards at, when any reward option was given; None prices every run at the terms it was paid under."""
    asked = (arguments.reward, arguments.discount, arguments.terminal_weight, arguments.value_flow,
             arguments.ground_flow, arguments.local_exchange, arguments.achievement_weight, arguments.predictor)
    if arguments.layer == TACTICAL or all(value is None for value in asked):
        return None
    return match_terms(arguments, arguments.layer)[0].as_dict()


#: How far apart two instances' arena draws are set. It only has to exceed the episodes one instance will ever run, and it is prime so that two runs started at neighbouring seeds do not lay one instance's stream on top of another's.
INSTANCE_STRIDE = 100003


def _arena_seed(arguments, session) -> int:
    """The seed an arena episode draws its fights from, which advances with the episode as well as with the instance.

    An arena is built afresh for every episode, and it used to be built from the instance alone. Every episode of an instance therefore drew the same site, the same two budgets, the same imbalance, the same angle and the same two forces as the one before it, in the same order: a run of fifty episodes on seven instances was not two thousand fights but about fifty distinct fights fought forty times over. Measured on the recorded runs, the first fight of every episode of an instance had one single spawn order across all fifty of them, and the second and third nearly always did too.

    What that costs is the sample size, and it costs it by a factor of thirty to fifty. The scatter of the fight-level mean was quoted as two standard errors over the number of fights, which for two thousand fights is about a fortieth; taken over the distinct draws instead it is about a seventh. Every left-right lean the arena has been charged with -the tenth of a point that came and went with the seed, the seven hundredths a lopsided draw was blamed for -sits comfortably inside that. There was no asymmetry to find. There were fifty fights being reported as two thousand.

    The episode is folded in per arm rather than per record, so that the arms of a comparison run the very same fights as each other while each of them runs different fights from one episode to the next. That is what makes a policy and its baseline a paired measurement: the two meet the same sites, the same budgets and the same forces, and what is left between them is the play.
    """
    arms = max(1, len(getattr(session, "arms", ()) or ()))
    return arguments.seed + INSTANCE_STRIDE * session.instance + len(session.records) // arms


def _arena_options(arguments, order: Optional[str] = None, floor: Optional[float] = None,
                   stall: Optional[int] = None) -> dict:
    """The arena settings a run actually asked for, as keywords, so that everything unasked for stands at the figure the arena states rather than at a copy of it kept here."""
    options = _given(stall_ms=stall * 1000 if stall else None, imbalance_floor=floor)
    if order is not None:
        options["decision_order"] = order
    return options


def _orders(arguments) -> list:
    """Which decision orders a run builds its arenas under, in the order they were named.

    More than one turns the run into a comparison between them, which is what the question they exist for takes: whether the left-right lean the fighting carries is made of the order the two sides are decided in can only be answered by running both orders and seeing whether the lean changes sign, and running them as two arms of one run is what holds everything else -the machine, the seed, the draws, the hour -still between the two.
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

    Several make the arms of one run. The two settings written this way -how lopsided a draw may be and how long a quiet spell is tolerated before a fight is called -are the two that trade the same pair of things against each other, how fair the arena is against how decisive its fights are, and neither trade can be settled by argument. Run as arms they are measured against each other on the very same draws, in one run, on one machine, in one hour.
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


def _save(net, path: Optional[str], version: int = 0) -> None:
    if not path:
        return
    from . import models

    models.save(net, path, version=version)
    log.info("saved parameters to %s", path)


def _episode(arguments, arena: bool) -> EpisodeSettings:
    maps = arguments.map or ["Lake"]
    return EpisodeSettings(
        map=maps[0], maps=maps if len(maps) > 1 else [], opponents=arguments.opponents, difficulty=arguments.difficulty,
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
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        # The episodes finished before the interruption are still reported; a run stopped from outside is the ordinary end of a long one.
        sessions = list(server.sessions)
    # An episode still under way when the run ends is closed as stopped, so that what it was collecting is recorded as ending there.
    server.close_unfinished()
    return sessions


def _journal_path(arguments, name: str) -> str:
    """Where a run's episodes are written: as asked, or under local/episodes named for the run and its stamp, the same stamp its dataset is named by."""
    return arguments.record or os.path.join(paths.episodes(), f"{name}-{arguments.stamp}.jsonl")


def _sha256(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _recorder(arguments, layer: str, name: str, behaviour: dict, terms, discount: float, trace: float,
              opponent: Optional[dict] = None):
    """The recorder a run writes its layer's decisions with, or None when it was asked not to record. The header says what the run was, so that a dataset read later can be told apart from every other without its log.

    `opponent` is what played the arena's other side when that side is recorded too, under squad number 1 (`Dataset.behaviours`).
    """
    if arguments.no_dataset:
        return None
    from .dataset import Recorder, run_directory

    directory = arguments.dataset or run_directory(layer, f"{name}-{arguments.stamp}")
    header = {"what": arguments.what, "argv": sys.argv[1:], "seed": arguments.seed, "instances": arguments.instances,
              "episodes": arguments.episodes, "behaviour": behaviour, "terms": terms.as_dict(),
              "discount": discount, "trace": trace, "journal": _journal_path(arguments, name)}
    if opponent is not None:
        header["opponent"] = opponent
    recorder = Recorder(directory, layer, header, shard_decisions=arguments.shard_decisions)
    log.info("recording the %s layer's decisions to %s", layer, directory)
    return recorder


def _network_behaviour(path: Optional[str], greedy: bool = False, learning: bool = False, explore: float = 0.0,
                       reloading: bool = False) -> dict:
    """What a network that plays is, for a dataset's header: which parameters it started from, whether it was learning as it played or reloading what a learner published (its `version` column then says which file version played each decision), and how it drew."""
    behaviour = {"name": "network", "parameters": os.path.basename(path) if path else "", "sha256": _sha256(path),
                 "learning": learning, "greedy": greedy, "explore": explore,
                 "kind": "mixture" if explore > 0 else ("deterministic" if greedy else "stochastic")}
    if reloading:
        behaviour.update(reloading=True, versions="vary")
    return behaviour


def _finish_recording(rollout: Optional[Rollout], recorder) -> None:
    """Seals whatever the run still holds, as stopped, and writes the rest of the dataset."""
    if recorder is None:
        return
    rollout.cut_all()
    rollout.seal_all({"instance": -1, "attempt": 0, "episode": -1, "arm": "", "map": "", "order": -1, "seed": -1,
                      "ending": "stopped"})
    recorder.close()


# ---- the tactical run ---------------------------------------------------------------------

def train_tactics(arguments) -> int:
    from .net import TacticalNet

    device = _serving_device(arguments.device, arguments.load, arguments.card_share)
    net = _network(TACTICAL, arguments.load, device, lambda: TacticalNet(**_given(width=arguments.width)))
    from . import models
    from .tokens import SET_SIZE

    log.info("tactical %s policy on %s: %d inputs, %d parameters", models.kind_of(net), device,
             SET_SIZE if models.kind_of(net) == "set" else TACTICAL_SIZE, sum(p.numel() for p in net.parameters()))

    # One figure discounts the returns and telescopes the shaping, and it is stated once here so that the two cannot drift apart. An arena errand is a whole fight, which is short and ends properly, so the default is to discount it at nothing and let the score of the fight reach every decision taken in it.
    discount = arguments.discount if arguments.discount is not None else FIGHT_DISCOUNT
    trace = arguments.trace if arguments.trace is not None else FIGHT_TRACE
    log.info("paying fights scored by %s, discounting an errand at %.4f with a trace of %.4f",
             arguments.score, discount, trace)

    paid = _given(outcome_weight=arguments.outcome_weight, score=arguments.score)
    behaviour = _network_behaviour(arguments.load, learning=True)
    # Self-play records both sides, each its own squad number; against the script only this side is recorded.
    recorder = None if arguments.script else _recorder(
        arguments, TACTICAL, "tactics", behaviour, TacticalTerms(discount=discount, **paid), discount, trace,
        opponent=None if arguments.script_opponent else behaviour)
    rollout = Rollout(discount=discount, trace=trace, sink=recorder.accept if recorder else None)
    optimiser = Optimiser(net, device=device, warmup=arguments.warmup,
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate,
                                   anchor_weight=arguments.anchor))
    batcher = choice_batcher(net, device=device, version=lambda: optimiser.report.updates, lock=optimiser.lock)
    trainer = Trainer(rollout, optimiser, on_update=_checkpointer(net, optimiser, arguments),
                      **_given(batch=arguments.batch))
    trainer.start()

    def learnt(session, catalogue):
        # An arena fight carries one contract from the moment it is joined to the moment it is called, and nothing reissues it. What ends the errand is therefore what ends the fight, and the conditions written into the contract are left as something for the layer to read and act on rather than as something that stops it being paid.
        return LearntTactics(session, catalogue, NetworkChoice(net, device, batcher), rollout,
                             session.instance, status_terminals=False, discount=discount, **paid)

    # One arena setting to a training run: a run whose arms differed would be training one policy on two arenas and reporting one number for it.
    order, floor, stall = _orders(arguments)[0], _floors(arguments)[0], _stalls(arguments)[0]

    def arm(session):
        # Both sides script is how the arena itself is measured rather than a policy: it is the baseline a learnt layer has to beat, and it is the only setting in which what the arena produces says something about the arena rather than about whatever the policy currently happens to do.
        ours = None if arguments.script else learnt
        return Arena(session, tactics=ours, seed=_arena_seed(arguments, session),
                     **_arena_options(arguments, order, floor, stall),
                     opponent=None if (arguments.script or arguments.script_opponent) else learnt)

    journal = Journal(_journal_path(arguments, "tactics"))
    try:
        sessions = _serve(arguments, [("tactics", arm)], _episode(arguments, arena=True), journal)
    finally:
        journal.close()
    report = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)
    _finish_recording(rollout, recorder)

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
    # The drawn fights are broken out because they are the ones that say what kind of arena this was: two forces that stopped hurting each other, two that never reached each other before the clock, and two that destroyed each other are three different results and only the last of them is a fight.
    log.info("%d engagement(s) built, %d never appeared (%.0f%% wasted), %d fought: %d won %d lost %d drawn (%.0f%% drawn: %d stalled, %d out of time, %d mutual)",
             total["engagements"], total["stillborn"],
             100.0 * total["stillborn"] / max(1, total["engagements"]), total["fought"],
             total["won"], total["lost"], total["drawn"],
             100.0 * total["drawn"] / max(1, total["fought"]),
             total["stalled"], total["expired"], total["mutual"])
    by_map: dict = {}
    for record in records:
        by_map.setdefault(str(getattr(record, "settings", {}).get("map", "")), []).append(record)
    if len(by_map) > 1:
        # Engagements are built at region centres and resource points, so a map with much water or cliff wastes more of them, and the share is what says whether a map is worth an arena's time.
        for name, theirs in sorted(by_map.items()):
            built = sum(int(r.statistics.get("engagements", 0)) for r in theirs)
            stillborn = sum(int(r.statistics.get("stillborn", 0)) for r in theirs)
            log.info("on %s: %d episode(s), %d engagement(s) built, %d never appeared (%.0f%% wasted), %d fought",
                     name, len(theirs), built, stillborn, 100.0 * stillborn / max(1, built),
                     sum(int(r.statistics.get("fought", 0)) for r in theirs))
    log.info("errand(s) closed by reason: %s", {reason: sum(int(record.statistics.get("terminals", {}).get(reason, 0)) for record in records) for reason in sorted({reason for record in records for reason in record.statistics.get("terminals", {})})})
    speeds = [r.speed for r in records if r.speed > 0]
    per_instance = sum(speeds) / len(speeds) if speeds else 0.0
    log.info("%d game second(s) over %d episode(s), %.1fx per instance and %.0fx over %d of them: %.1f fight(s) and %d decision(s) per game minute, %.0f decision(s) per wall second",
             seconds, len(records), per_instance, per_instance * len(sessions), len(sessions),
             60.0 * total["fought"] / max(1, seconds), int(60.0 * total["decisions"] / max(1, seconds)),
             total["decisions"] / max(1.0, wall))
    if batcher is not None:
        log.info("batched inference averaged %.1f per call over %d call(s)", batcher.batch_size, batcher.calls)


# ---- the matchup run -------------------------------------------------------------------------

def matchups(arguments) -> int:
    """Measures what each type does to each other type at equal credits, for the combat table the build order chooses by.

    Both sides are the handwritten tactical layer, and each fight is one type against one type, drawn afresh. What comes out is merged into the report the combat table reads (`combat.MATCHUPS_FILE` under the reports directory), so that runs add up rather than replace one another.
    """
    import json

    from ..control.policy.catalogue import Catalogue
    from ..control.policy.combat import MATCHUPS_FILE
    from .arena import matchup_table

    def arm(session):
        return Arena(session, seed=_arena_seed(arguments, session), matchups=True, **_arena_options(arguments))

    journal = Journal(_journal_path(arguments, "matchups"))
    try:
        sessions = _serve(arguments, [("matchups", arm)], _episode(arguments, arena=True), journal)
    finally:
        journal.close()
    _report_arena(sessions)
    records = [record for session in sessions for record in session.records]
    if not sessions or not records:
        log.error("no fights were fought, so no matchups were measured")
        return 1
    path = os.path.join(paths.reports(), MATCHUPS_FILE)
    previous = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
    table = matchup_table(records, Catalogue(sessions[0].types, sessions[0].assets), previous)
    with paths.replacing(path) as handle:
        json.dump(table, handle, indent=1)
    log.info("%d measured pair(s) over %d fight(s) written to %s", len(table["pairs"]),
             sum(entry["fights"] for entry in table["pairs"]) // 2, path)
    return 0


# ---- the measurement run -------------------------------------------------------------------

def duel(arguments) -> int:
    """Runs the arena for measurement rather than for learning: the loaded policy on one side, the handwritten tactical layer on the other, and nothing kept but the score.

    Neither a buffer nor a trainer is built here, and the layer is handed no rollout at all, so no decision is written down anywhere. That is not thrift. A buffer nobody drains grows for the length of the run, and a trainer would move the parameters being measured while they were being measured, which would make the number that came out a number about no policy in particular.

    Leaving the policy out is the baseline: both sides are then the same handwritten layer, so the average result has to be nought by the antisymmetry of the score itself, and anything else is the arena favouring one side of the board.

    The baseline is taken alongside the policy rather than left to a separate run, and that is the default because the alternative has already produced a wrong reading twice. How far the arena leans is not a property of the arena but of the seed it was run under - the same handwritten layer against itself came out at -0.02 under one seed and +0.06 under another - so a policy's score is only readable beside the lean of the very seed it was measured on. Run as two arms of one run they alternate within each instance, which holds the seed, the machine and the hour still between them; the difference of the two is then what the policy is worth, and the interval on that difference is what says whether it is worth anything at all.

    Several policies may be named at once, comma separated, and then each is an arm beside the same baseline. That is not thrift either: two policies measured in separate runs are compared through their baselines, and two measured here are compared on the very fights both of them fought.

    With `--from` nothing is run: the journals of finished duels are read and reported together (`report_duels`).
    """
    if arguments.source:
        return report_duels(arguments.source)
    policies, batchers = [], []
    if arguments.load:
        from . import models

        files = [path.strip() for path in str(arguments.load).split(",") if path.strip()]
        if len(set(files)) != len(files):
            # Two arms of the same name would be journalled as one and reported as one, so the run would silently measure half of what it was asked for.
            raise SystemExit("the same parameters were named twice, and two arms cannot share a name")
        # How a policy plays, as an axis of the comparison rather than a setting of the run. Drawing from the distribution is what the policy does when it is operated and is therefore the figure that counts; taking the likeliest action is the same weights without the exploration, and the difference between them is the tax the run's own randomness charges. Measured as two arms of one run they meet the same fights, which is the only way that difference is separable from how the fights were drawn.
        draws = (False, True) if arguments.greedy else (False,)
        for path in files:
            if not os.path.exists(path):
                # Refused rather than started from nothing, which is what a training run does with a missing file. A measurement that quietly scored a freshly initialised policy would produce a perfectly plausible number about a policy nobody asked about.
                log.error("there are no parameters at %s to measure", path)
                return 1
            device = _serving_device(arguments.device, path, arguments.card_share)
            net = models.load(path, TACTICAL, device)
            lock, version, _ = _following(net, path, arguments.reload_seconds)
            for greedy in draws:
                batcher = choice_batcher(net, device=device, greedy=greedy, version=version, lock=lock)
                batchers.append(batcher)

                def learnt(session, catalogue, net=net, batcher=batcher, greedy=greedy, device=device):
                    # No rollout: this layer is being read from and not learnt from, and with nowhere to record a decision it records none.
                    return LearntTactics(session, catalogue,
                                         NetworkChoice(net, device, batcher, greedy),
                                         None, session.instance, status_terminals=False)

                # Named after the file when there are several to tell apart, and simply the duel when there is one, which is the name the journal has always carried.
                policies.append(("duel" + ("" if len(files) == 1 else
                                           "-" + os.path.splitext(os.path.basename(path))[0])
                                 + ("-greedy" if greedy else ""), learnt))

    for pinned in _pinned(arguments):
        def fixed(session, catalogue, pinned=pinned):
            return LearntTactics(session, catalogue, PinnedDeparture(pinned.value), None,
                                 session.instance, status_terminals=False)

        policies.append((f"duel-always-{pinned.name.lower()}", fixed))

    def build(policy, order: str, floor: Optional[float], stall: Optional[int]):
        def arm(session):
            # The opponent is left unnamed, which is what puts the handwritten layer on the other side of every fight. That is the thing being measured against, so it is not something this run offers a choice about.
            return Arena(session, tactics=policy, seed=_arena_seed(arguments, session),
                         **_arena_options(arguments, order, floor, stall))

        return arm

    orders, floors, stalls = _orders(arguments), _floors(arguments), _stalls(arguments)

    def named(side: str, order: str, floor: Optional[float], stall: Optional[int]) -> str:
        # Named after whatever is actually varying across the arms, so that a plain measurement keeps the two names the journal has always used and a comparison of arena settings says which setting each arm was.
        return (side + (f"-{order}" if len(orders) > 1 else "")
                + (f"-floor{floor:g}" if len(floors) > 1 and floor is not None else "")
                + (f"-stall{stall:d}" if len(stalls) > 1 and stall is not None else ""))

    # The policies' sides and the baseline's side of the comparison. A run with nothing loaded is the baseline alone, which is how the arena itself is measured; a run with a policy takes both unless the baseline was explicitly declined.
    sides = list(policies) + ([("duel-baseline", None)] if not policies or arguments.baseline else [])
    arms = [(named(side, order, floor, stall), build(policy, order, floor, stall))
            for order in orders for floor in floors for stall in stalls for side, policy in sides]
    # The baseline is written down as the baseline. It is a different quantity from a policy's score rather than a run of it that happens to have scored nought, and the likeliest way to confuse the two is to have journalled them under one name. Under one journal that is the arm each episode carries; the file is named for what the run was for.
    baselines = {name for name, _ in arms if name.startswith("duel-baseline")}
    journal = Journal(_journal_path(arguments, "duel" if arguments.load else "duel-baseline"))
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
#: Both, always, whichever one was paid. The sparse reading is what every ceiling this project has quoted was measured on and dropping it would make a new run unreadable against any of them; the health reading is the one that is not nought on the three quarters of fights that end with two damaged forces still standing. A run costs the same either way -both are computed as a fight is called -so there is no reason to report one.
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
    return 2.0 * standard_error(first, second)


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

    The arms of a run are handed the same arena seed in the same round, so the fight under one key is the same fight in every arm -the same site, the same two budgets, the same imbalance, the same two forces. Keyed that way the arms can be differenced fight by fight, and the difference is then free of the only thing that makes the score scatter, which is how the fight was drawn rather than how it was fought.

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

    This is the measurement the baseline is taken alongside the policy for. Unpaired, the difference of two arms carries the whole scatter of how fights are drawn -a two-to-one draw scores half a point whoever is playing -and that scatter is several times anything a policy has ever moved. Paired on the draw it cancels, and what is left is the play.
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


def duel_sessions(sources) -> list:
    """The episodes journalled in these files, grouped as the sessions that ran them, each instance named by its file as well as its number.

    Fights are paired within one file only. Two runs are separate draws even under the same seed, so a fight of one run and the fight with the same numbers in another are not the same fight; naming each instance by its file keeps them apart.
    """
    sessions = []
    for number, source in enumerate(sources):
        by_instance: dict = {}
        for entry in read_journal(source):
            record = SimpleNamespace(arm=entry.get("arm", ""), instance=(number, int(entry.get("instance", -1))),
                                     statistics=entry.get("statistics", {}))
            by_instance.setdefault(record.instance, []).append(record)
        sessions.extend(SimpleNamespace(records=records) for records in by_instance.values())
    return sessions


def report_duels(sources) -> int:
    """Reports the duels journalled in these files as one measurement: every arm over every file, and every pair of arms on the fights both drew within one file (`duel_sessions`)."""
    sessions = duel_sessions(sources)
    if not any(session.records for session in sessions):
        log.error("no episodes in %s", ", ".join(sources))
        return 1
    log.info("reporting %d duel journal(s): %s", len(sources), ", ".join(sources))
    names = {record.arm for session in sessions for record in session.records}
    _report_duel(sessions, {name for name in names if name.startswith("duel-baseline")})
    return 0


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

    device = _serving_device(arguments.device, arguments.load, arguments.card_share)
    net = _network(OPERATIONAL, arguments.load, device, OperationalNet)
    log.info("operational policy on %s: %d features, %d parameters",
             device, OPERATIONAL_SIZE, sum(p.numel() for p in net.parameters()))

    terms, trace = match_terms(arguments, OPERATIONAL)
    recorder = _recorder(arguments, OPERATIONAL, "operations", _network_behaviour(arguments.load, learning=True),
                         terms, terms.discount, trace)
    rollout = Rollout(discount=terms.discount, trace=trace, sink=recorder.accept if recorder else None)
    optimiser = Optimiser(net, device=device, two_headed=True, warmup=arguments.warmup,
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate,
                                   anchor_weight=arguments.anchor))
    batcher = operational_batcher(net, device=device, version=lambda: optimiser.report.updates, lock=optimiser.lock)
    trainer = Trainer(rollout, optimiser, on_update=_checkpointer(net, optimiser, arguments),
                      **_given(batch=arguments.batch))
    trainer.start()

    layer_options = dict(_given(review_ms=int(arguments.review_seconds * 1000) if arguments.review_seconds else None),
                         terms=terms)
    arm = learning_arm(
        OPERATIONAL,
        lambda session: NetworkOperations(net, device, batcher),
        rollout,
        intruders=(lambda session: Intruder(seed=arguments.seed + INSTANCE_STRIDE * session.instance
                                            + len(session.records), instance=session.instance))
        if arguments.intruder else None,
        **layer_options,
    )
    journal = Journal(_journal_path(arguments, "operations"))
    try:
        sessions = _serve(arguments, [("operations", arm)], _episode(arguments, arena=False), journal)
    finally:
        journal.close()
    report = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)
    _finish_recording(rollout, recorder)
    log.info("batched inference averaged %.1f per call", batcher.batch_size)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    _report_matches(sessions)
    return 0


# ---- the economic run ---------------------------------------------------------------------

def train_economy(arguments) -> int:
    """Trains the build order over whole matches, every other layer the script. Each investment is a step; the ones a period makes follow each other undiscounted, and the last of a period collects what the board shows until the next."""
    from .net import EconomicNet

    device = _serving_device(arguments.device, arguments.load, arguments.card_share)
    net = _network(ECONOMIC, arguments.load, device, EconomicNet)
    log.info("economic policy on %s: %d features, %d parameters",
             device, ECONOMIC_SIZE, sum(p.numel() for p in net.parameters()))

    terms, trace = match_terms(arguments, ECONOMIC)
    layer_options = dict(terms=terms)
    recorder = _recorder(arguments, ECONOMIC, "economy", _network_behaviour(arguments.load, learning=True),
                         terms, terms.discount, trace)
    rollout = Rollout(discount=terms.discount, trace=trace, sink=recorder.accept if recorder else None)
    optimiser = Optimiser(net, device=device, warmup=arguments.warmup,
                          **_given(entropy_weight=arguments.entropy, learning_rate=arguments.learning_rate,
                                   anchor_weight=arguments.anchor))
    batcher = choice_batcher(net, device=device, version=lambda: optimiser.report.updates, lock=optimiser.lock)
    trainer = Trainer(rollout, optimiser, on_update=_checkpointer(net, optimiser, arguments),
                      **_given(batch=arguments.batch))
    trainer.start()

    arm = learning_arm(
        ECONOMIC,
        lambda session: NetworkChoice(net, device, batcher),
        rollout,
        intruders=(lambda session: Intruder(seed=arguments.seed + INSTANCE_STRIDE * session.instance
                                            + len(session.records), instance=session.instance))
        if arguments.intruder else None,
        **layer_options,
    )
    journal = Journal(_journal_path(arguments, "economy"))
    try:
        sessions = _serve(arguments, [("economy", arm)], _episode(arguments, arena=False), journal)
    finally:
        journal.close()
    report = trainer.finish()
    batcher.stop()
    _save(net, arguments.save)
    _finish_recording(rollout, recorder)
    log.info("batched inference averaged %.1f per call", batcher.batch_size)
    if report is not None:
        log.info("last update: %s", report.as_dict())
    _report_matches(sessions)
    return 0


def _checkpointer(net, optimiser, arguments):
    """Writes the parameters every --checkpoint-every updates, under the lock the updates take: over --save, so that a run stopped from outside keeps what it had learnt, and with --snapshots also to a file of its own per checkpoint (`snapshot_path`), so that the policies a run passed through can be played again later."""
    every = arguments.checkpoint_every
    if not every or not (arguments.save or arguments.snapshots):
        return None

    def on_update(report) -> None:
        if report.updates % every:
            return
        with optimiser.lock:
            _save(net, arguments.save, version=report.updates)
            if arguments.snapshots:
                _save(net, snapshot_path(arguments, report.updates), version=report.updates)

    return on_update


def snapshot_path(arguments, updates: int) -> str:
    """Where the parameters after this many updates are kept: `<stem of --save, or the run>-u<updates>.pt` under --snapshots."""
    stem = os.path.splitext(os.path.basename(arguments.save))[0] if arguments.save else arguments.what
    return os.path.join(arguments.snapshots, f"{stem}-u{updates:05d}.pt")


def _report_matches(sessions) -> None:
    """The matches a training run played, scored as an evaluation scores them. The parameters moved throughout, so this is a reading of the run and not a measurement of the policy it ended with; the later half is the nearer of the two to that policy."""
    from ..eval.__main__ import Played, report
    from ..eval.scoring import default_weights

    played = [Played.of(record) for session in sessions for record in session.records]
    if not played:
        return
    log.info("the matches this run trained on, scored as an evaluation scores them:")
    report(played, default_weights())
    later = [Played.of(record) for session in sessions for record in session.records[len(session.records) // 2:]]
    if later and len(later) < len(played):
        for episode in later:
            episode.arm = "later-half"
        report(later, default_weights())


# ---- collecting what the script does -------------------------------------------------------

def collect(arguments) -> int:
    """Runs the ordinary chain and writes down every decision the layer under study made, in the form a learnt layer emits.

    This is the design's answer to human play being scarce: a replay played back recovers the state behind each human command, but there are only as many replays as people play, and a human's unit orders have to be turned into contracts by inference. A script's decisions come already in contract form, decided here from an observation this process is holding. What comes out is not as good as human play and there is an unlimited amount of it.

    What plays is the script, a fixed operational rule (`--rule`), one pinned tactical departure (`--pin`) or a saved network (`--student`), with `--explore` of its decisions drawn evenly from the legal ones instead; whichever it is, the layer's judge labels every board it reaches. With `--both-sides` the arena's other side, which is always the script, is recorded as well. Nothing is learnt, so nothing is kept once an episode has been written down.
    """
    pinned = _collect_checks(arguments)
    if arguments.teacher == BUILTIN:
        return _collect_builtin(arguments)
    pupil = _pupil(arguments)
    behaviour = _collect_behaviour(arguments, pinned)
    temperature = _given(temperature=arguments.temperature)
    if arguments.layer == TACTICAL:
        discount = arguments.discount if arguments.discount is not None else FIGHT_DISCOUNT
        trace = arguments.trace if arguments.trace is not None else FIGHT_TRACE
        paid = _given(outcome_weight=arguments.outcome_weight, score=arguments.score)
        terms = TacticalTerms(discount=discount, **paid)
    else:
        terms, trace = match_terms(arguments, arguments.layer)
        discount = terms.discount
    recorder = _recorder(arguments, arguments.layer, "collect", behaviour, terms, discount, trace,
                         opponent=SCRIPT_BEHAVIOUR if arguments.both_sides else None)
    rollout = Rollout(discount=discount, trace=trace, sink=recorder.accept if recorder else None, retain=False)

    def decider_for(session):
        """What plays for one episode, seeded per instance and episode so that the draws differ from one to the next; None for the script."""
        seed = arguments.seed + INSTANCE_STRIDE * session.instance + len(session.records)
        if arguments.rule:
            player = PinnedOperations(arguments.rule, seed=seed)
        elif pinned is not None:
            player = PinnedDeparture(pinned.value)
        else:
            player = pupil
        if arguments.explore <= 0.0:
            return player
        # Without a player of its own the script explores around its own answers, which the layer binds its judge into.
        return Labelled(player if player is not None else ScriptChoice(), explore=arguments.explore,
                        seed=EXPLORE_STREAM + seed)

    if arguments.layer == TACTICAL:
        # A learnt layer with no decider falls through to the rule it inherited, so what plays is the script and what is written down is the script's own decisions in the form a learnt layer emits.
        def recorded(decider):
            return lambda s, catalogue: LearntTactics(s, catalogue, decider, rollout, s.instance, status_terminals=False,
                                                      discount=discount, **paid, **temperature)

        def arm(session):
            return Arena(session, tactics=recorded(decider_for(session)),
                         opponent=recorded(None) if arguments.both_sides else None,
                         seed=_arena_seed(arguments, session),
                         **_arena_options(arguments, _orders(arguments)[0], _floors(arguments)[0],
                                          _stalls(arguments)[0]))

        episode = _episode(arguments, arena=True)
    else:
        options = dict(terms=terms, **temperature)
        if arguments.layer == OPERATIONAL and arguments.review_seconds:
            options["review_ms"] = int(arguments.review_seconds * 1000)
        arm = learning_arm(
            arguments.layer, decider_for, rollout,
            intruders=(lambda session: Intruder(seed=arguments.seed + INSTANCE_STRIDE * session.instance
                                                + len(session.records), instance=session.instance))
            if arguments.intruder else None,
            **options)
        episode = _episode(arguments, arena=False)

    journal = Journal(_journal_path(arguments, "collect"))
    try:
        sessions = _serve(arguments, [(f"ops-{arguments.rule}" if arguments.rule else "script", arm)], episode, journal)
    finally:
        journal.close()
    _finish_recording(rollout, recorder)
    if arguments.layer == TACTICAL:
        _report_arena(sessions)
    return 0


#: The script as a behaviour policy, for a dataset's header.
SCRIPT_BEHAVIOUR = {"name": "script", "kind": "deterministic", "explore": 0.0}

#: Added to an episode's seed for the exploration's own draws, so that they stay apart from the draws of a rule playing under the same seed.
EXPLORE_STREAM = 1 << 40


def _collect_checks(arguments):
    """Refuses a collecting run whose settings do not name one player for its layer, before any game is started, and returns the departure it pins the tactical layer to, or None. One player, because a dataset's header names one policy that played it."""
    departures = _pinned(arguments)
    if departures and arguments.layer != TACTICAL:
        raise SystemExit("--pin pins a tactical departure, and only the tactical layer has them")
    if len(departures) > 1:
        raise SystemExit("a collecting run pins one departure; run one collection per departure")
    pinned = departures[0] if departures else None
    if arguments.rule and arguments.layer != OPERATIONAL:
        raise SystemExit("--rule names an operational rule, and only the operational layer has them")
    if arguments.layer == TACTICAL and arguments.intruder:
        raise SystemExit("--intruder interferes with a match, and the tactical layer is collected in the arena")
    if arguments.both_sides and arguments.layer != TACTICAL:
        raise SystemExit("--both-sides records the arena's other side, and only the tactical layer is collected in the arena")
    if not 0.0 <= arguments.explore <= 1.0:
        raise SystemExit(f"--explore is a share of the decisions, from 0 to 1, not {arguments.explore:g}")
    if sum(1 for given in (arguments.rule, arguments.student, pinned) if given is not None) > 1:
        raise SystemExit("--rule, --pin and --student each name what plays, and a run takes one")
    if arguments.teacher is not None:
        if arguments.teacher != BUILTIN:
            raise SystemExit(f"collect --teacher takes {BUILTIN}, not {arguments.teacher}")
        if arguments.layer != OPERATIONAL:
            raise SystemExit("--teacher builtin reads the built-in AI's operational decisions, so it needs --layer operations")
        if arguments.rule or arguments.student or pinned is not None:
            raise SystemExit("--teacher builtin is what plays, so it takes no --rule, --pin or --student")
        if arguments.explore > 0 or arguments.intruder:
            raise SystemExit("--teacher builtin only watches the built-in AI, so nothing can explore or intrude")
    return pinned


#: The `--teacher` of a collecting run that records the built-in AI.
BUILTIN = "builtin"


def _collect_builtin(arguments) -> int:
    """Records the built-in AI's operational decisions: matches between two AI players, each episode watched from one of them in turn, with the script chain run beside the watched player and the operational layer answered by what that player's orders amount to."""
    from .builtin import BuiltinOperations, merge

    terms, trace = match_terms(arguments, OPERATIONAL)
    discount = terms.discount
    episode = _builtin_episode(arguments)
    behaviour = {"name": BUILTIN, "kind": "deterministic", "teacher": BUILTIN, "difficulty": arguments.difficulty,
                 "watch": "alternate" if episode.contestants else "opponent"}
    recorder = _recorder(arguments, OPERATIONAL, "collect", behaviour, terms, discount, trace)
    rollout = Rollout(discount=discount, trace=trace, sink=recorder.accept if recorder else None, retain=False)
    options = dict(terms=terms, **_given(temperature=arguments.temperature))
    review_ms =int(arguments.review_seconds * 1000) if arguments.review_seconds else None
    if review_ms:
        options["review_ms"] = review_ms
    deciders = []

    def decider_for(session):
        decider = BuiltinOperations(**({"review_ms": review_ms} if review_ms else {}))
        deciders.append(decider)
        return decider

    arm = learning_arm(OPERATIONAL, decider_for, rollout, **options)
    journal = Journal(_journal_path(arguments, "collect"))
    try:
        _serve(arguments, [(BUILTIN, arm)], episode, journal)
    finally:
        journal.close()
    _finish_recording(rollout, recorder)
    log.info("built-in AI teacher: %s", merge(deciders).report())
    return 0


def _builtin_episode(arguments) -> EpisodeSettings:
    """The episodes a built-in AI teacher is recorded in: two AI contestants watched from each side in turn where every map has a start for the local player and both, otherwise the one AI opponent watched against the local player's side, which then nothing commands."""
    from dataclasses import replace

    from ..data.__main__ import find_maps
    from ..data.maps import read_map

    episode = _episode(arguments, arena=False)
    assets = AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default()
    starts = min(read_map(path, assets).players for path in find_maps(arguments.map or [episode.map], assets))
    if starts >= 3:
        return replace(episode, contestants=2, ai_orders=True, watch=[0, 1])
    return replace(episode, opponents=1, contestants=0, ai_orders=True, watch=[0])


def _collect_behaviour(arguments, pinned) -> dict:
    """What played a collecting run, for its dataset's header: the name of the player and how its decisions were drawn."""
    if arguments.student:
        return _network_behaviour(arguments.student, explore=arguments.explore, reloading=bool(arguments.reload_seconds))
    if arguments.rule:
        name, kind = f"rule:{arguments.rule}", "uniform" if arguments.rule == "random" else "deterministic"
    elif pinned is not None:
        name, kind = f"pin:{pinned.name.lower()}", "deterministic"
    else:
        name, kind = SCRIPT_BEHAVIOUR["name"], SCRIPT_BEHAVIOUR["kind"]
    return {"name": name, "kind": "mixture" if arguments.explore > 0 else kind, "explore": arguments.explore}


def _pupil(arguments):
    """The network a collecting run lets play while the judge labels, loaded from `--student`, or None when something else plays."""
    if not arguments.student:
        return None
    from . import models

    if not os.path.exists(arguments.student):
        raise SystemExit(f"no parameters at {arguments.student}")
    device = _serving_device(arguments.device, arguments.student, arguments.card_share)
    net = models.load(arguments.student, arguments.layer, device)
    lock, version, reloader = _following(net, arguments.student, getattr(arguments, "reload_seconds", None))
    if arguments.layer in (TACTICAL, ECONOMIC):
        decider = NetworkChoice(net, device, choice_batcher(net, device=device, version=version, lock=lock))
    else:
        decider = NetworkOperations(net, device, operational_batcher(net, device=device, version=version, lock=lock))
    decider.reloader = reloader
    return decider


def _following(net, path: str, seconds: Optional[float]):
    """The lock a served network's batcher evaluates inside, the version its choices are stamped with, and the reloader that keeps it at the latest version published to `path` when `seconds` is given (None otherwise)."""
    import threading

    from .reload import Reloader

    lock = threading.Lock()
    if not seconds:
        served = int(getattr(net, "version", 0))
        return lock, (lambda: served), None
    reloader = Reloader(net, path, lock, float(seconds)).start()
    log.info("reloading %s every %g s, serving version %d", path, float(seconds), reloader.version)
    return lock, (lambda: reloader.version), reloader


# ---- imitating what the script does ---------------------------------------------------------

def weighted(given) -> list:
    """Each dataset named on the command line as its directory and the weight its decisions are multiplied by: `path:weight`, or a bare path at one."""
    parsed = []
    for text in given:
        path, _, suffix = text.rpartition(":")
        try:
            scale = float(suffix) if path else None
        except ValueError:
            scale = None
        if scale is None:
            parsed.append((text, 1.0))
        elif scale < 0:
            raise SystemExit(f"a dataset's weight cannot be negative: {text}")
        else:
            parsed.append((path, scale))
    return parsed


def clone(arguments) -> int:
    """Fits a network to the teacher's answers in recorded datasets, which is where a training run is meant to start from rather than from noise.

    Nothing is connected to and no game is started: the teacher is on disk, and the whole of this is arithmetic. What comes out has a policy worth measuring and a value head that is still random, which is what the training run's warm-up is for.
    """
    from .dataset import Dataset
    from .imitation import EPOCHS, PATIENCE, SMOOTHING, Teacher, fit, network, report
    from .net import TacticalNet

    if not arguments.dataset_sources:
        raise SystemExit("name at least one recorded run to clone from with --dataset")
    if arguments.net == "set":
        # A set network is cloned by the offline runner, which is the same loss on the graphics card.
        arguments.method = "bc"
        return offline_command(arguments)
    if arguments.init_flat:
        raise SystemExit("--init-flat starts the flat part of a tactical set network, so it needs --net set")
    device = _device(arguments.device or "auto")
    from . import models

    models.limit_card(device, arguments.card_share)
    net = _network(arguments.layer, arguments.load, device,
                   lambda: TacticalNet(**_given(width=arguments.width)) if arguments.layer == TACTICAL
                   else network(arguments.layer))
    sources = []
    for path, scale in weighted(arguments.dataset_sources):
        log.info("cloning from %s at weight %g", path, scale)
        sources.append((Dataset.open([path], layer=arguments.layer), scale))
    if not 0.0 < arguments.fraction <= 1.0:
        raise SystemExit(f"--fraction is a share of the episodes, above 0 and at most 1, not {arguments.fraction:g}")
    teacher = Teacher.of(sources, keep_tainted=arguments.keep_tainted, fraction=arguments.fraction)
    if arguments.fraction < 1.0:
        log.info("fitting to %d decision(s) from %g of the training side's episodes, judged on %d held out",
                 int((~teacher.held_out).sum()), arguments.fraction, int(teacher.held_out.sum()))
    # Only what was actually asked for is passed on, so that everything else stands at the figure the cloning module states rather than at a copy of it kept here.
    asked = _given(smoothing=arguments.smoothing, epochs=arguments.epochs,
                   patience=arguments.patience, batch=arguments.batch)
    log.info("cloning the %s layer with smoothing %.3f over at most %d epoch(s), stopping after %d "
             "without improvement", arguments.layer,
             asked.get("smoothing", SMOOTHING), asked.get("epochs", EPOCHS), asked.get("patience", PATIENCE))
    net, cloning = fit(teacher, net=net, device=device, seed=arguments.seed, **asked)
    report(cloning)
    _save(net, arguments.save)
    return 0


def offline_command(arguments) -> int:
    """Learns from recorded datasets alone with one of the offline methods (`offline.METHODS`); no game is started."""
    from .offline import EPOCHS, PATIENCE, Settings, follow, run

    if not arguments.dataset_sources and not (arguments.follow and arguments.watch):
        raise SystemExit("name at least one recorded run with --dataset")
    _device(arguments.device or "auto")
    settings = Settings(layer=arguments.layer, sources=weighted(arguments.dataset_sources), method=arguments.method,
                        net=arguments.net, depth=arguments.depth, width=arguments.width, load=arguments.load,
                        init_flat=arguments.init_flat, teacher=arguments.teacher, policy=arguments.policy, save=arguments.save,
                        device=arguments.device or "auto", epochs=arguments.epochs or EPOCHS, batch=arguments.batch,
                        learning_rate=arguments.learning_rate, seed=arguments.seed, fraction=arguments.fraction,
                        keep_tainted=arguments.keep_tainted, report=arguments.report, discount=arguments.discount,
                        terms=_repriced(arguments),
                        patience=arguments.patience or PATIENCE, watch=list(arguments.watch),
                        publish_every=arguments.publish_every, rescan_seconds=arguments.rescan_seconds,
                        updates=arguments.updates, buffer=arguments.buffer, card_share=arguments.card_share,
                        **_given(expectile=arguments.expectile, beta=arguments.beta, top=arguments.top,
                                 alpha=arguments.alpha, judge=arguments.judge, critic_epochs=arguments.critic_epochs,
                                 smoothing=arguments.smoothing, decay=arguments.decay))
    if arguments.follow:
        import signal
        import threading

        stop = threading.Event()
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, lambda *_: stop.set())
        try:
            report = follow(settings, stop)
        except (ValueError, FileNotFoundError) as error:
            raise SystemExit(str(error))
        log.info("following learner stopped after %d update(s); published versions %s; report at %s", report["updates"],
                 report["published"], report["report"])
        return 0
    try:
        report = run(settings)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
    log.info("offline %s of the %s layer finished; report at %s", settings.method, settings.layer, report["report"])
    return 0


# ---- working on what was recorded -----------------------------------------------------------

def dataset_command(arguments) -> int:
    """`inspect` reports what recorded runs hold and, with `--verify`, rebuilds every state from its materials and counts those that differ from what was recorded; `reencode` writes a run again with its states encoded by the encoding in force."""
    from .dataset import Dataset, inspect, reencode, verify

    if not arguments.paths:
        raise SystemExit("say what to do: dataset inspect <run>... or dataset reencode <run> --out <run>")
    action, sources = arguments.paths[0], arguments.paths[1:]
    if not sources:
        raise SystemExit(f"dataset {action} needs at least one recorded run")
    if action == "inspect":
        dataset = Dataset.open(sources, check=False, with_materials=arguments.verify)
        log.info("%s", json.dumps(inspect(dataset), indent=1, sort_keys=True))
        if not arguments.verify:
            return 0
        checked, mismatched, unbuildable = verify(dataset)
        log.info("rebuilt %d state(s) from their materials: %d differ from what was recorded, %d could not be rebuilt",
                 checked, mismatched, unbuildable)
        return 1 if mismatched or unbuildable else 0
    if action == "reencode":
        if len(sources) != 1 or not arguments.out:
            raise SystemExit("dataset reencode takes one run and --out for where to write it")
        written = reencode(sources[0], arguments.out)
        log.info("encoded %d decision(s) again into %s", written, arguments.out)
        return 0
    raise SystemExit(f"no dataset action named {action!r}: expected inspect or reencode")


def predictor_command(arguments) -> int:
    """Fits the score expected from a board on the matches recorded in the --dataset runs, of the operational layer or the economy, reports how well each knot predicts the score of matches it was not fitted on, and writes the coefficients to --save when given. The report goes to --report, or under the reports directory."""
    from dataclasses import asdict

    from . import predictor as predictor_module
    from .dataset import Dataset

    sources = [path for path, _ in weighted(arguments.dataset_sources)]
    if not sources:
        raise SystemExit("predictor fits on recorded runs: give them with --dataset")
    datasets = [Dataset.open([source], check=False) for source in sources]
    if any(dataset.layer == TACTICAL for dataset in datasets):
        raise SystemExit("predictor fits on runs of the operational layer or the economy, which play matches")
    samples = predictor_module.samples_of(datasets)
    fitted, knots = predictor_module.fit(samples)
    log.info("fitted on %d board(s) of %d match(es)", len(samples), len(set(samples.match)))
    for knot in knots:
        log.info("  %5.0fs left: %6d board(s) of %4d match(es), value %+.3f ground %+.3f, held-out R2 %s, within groups %s",
                 knot.knot, knot.samples, knot.matches, knot.value, knot.ground,
                 "-" if knot.r2 is None else f"{knot.r2:+.3f}", "-" if knot.r2_within is None else f"{knot.r2_within:+.3f}")
    report_path = arguments.report or os.path.join(paths.reports(), f"predictor-{arguments.stamp}.json")
    with paths.replacing(report_path) as handle:
        json.dump({"sources": sources, "boards": len(samples), "matches": len(set(samples.match)),
                   "predictor": fitted.as_dict(), "knots": [asdict(knot) for knot in knots]}, handle, indent=1)
    log.info("report written to %s", report_path)
    if arguments.save:
        predictor_module.save(fitted, arguments.save)
        log.info("coefficients written to %s", arguments.save)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    arguments = build_parser().parse_args(argv)
    arguments.argv = argv
    return _dispatch(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["tactics", "operations", "economy", "collect", "clone", "offline", "duel",
                                         "matchups", "dataset", "bench", "online", "predictor"])
    parser.add_argument("paths", nargs="*",
                        help="for dataset: what to do (inspect or reencode) and the recorded runs to do it to")
    parser.add_argument("--layer", default=TACTICAL, choices=list(LAYERS),
                        help="which layer to collect, to clone, to learn offline or online, or whose network to bench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--map", action="append", default=None,
                        help="the map to play, Lake unless given; repeated, the maps are played in turn, one round "
                             "of the arms each")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--credits", type=int, default=0)
    parser.add_argument("--fog", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-seconds", type=int, default=0,
                        help="game time an episode is cut off at, defaulting to a long one for the arena")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--device", default=None,
                        help="torch device, for example cuda, or auto for a graphics card when there is one. The "
                             "default is auto for clone, offline and bench; a run that plays games serves a set "
                             "network on a graphics card when there is one and every other network on the processor")
    parser.add_argument("--card-share", type=float, default=None, dest="card_share", metavar="FRACTION",
                        help="share of a graphics card's memory the process may take, above nought and at most one; an "
                             "overrun ends the process with an out-of-memory error. No limit unless given, except "
                             "offline --follow, which takes 0.5; online passes it to the learner only")
    parser.add_argument("--method", default=None, choices=["bc", "distill", "iql", "awr", "topbc", "cql", "fqe"],
                        help="for offline, the method: bc, distill, iql, awr, topbc, cql or fqe, bc unless given; "
                             "offline --follow and online learn by iql, awr or cql only, iql unless given")
    parser.add_argument("--net", default="flat", choices=["flat", "set"],
                        help="for offline and clone, the kind of network to build when --load names none")
    parser.add_argument("--depth", type=int, default=None, help="encoder layers of a set network")
    parser.add_argument("--init-flat", default=None,
                        help="for offline and clone with --net set on the tactical layer, a flat tactical model file whose "
                             "network becomes the starting flat part of the new residual set network")
    parser.add_argument("--teacher", default=None,
                        help="for offline distill, the model file distilled from; for collect, 'builtin' records the "
                             "operational decisions of the built-in AI, read from its orders in matches between two AI "
                             "players watched from each side in turn")
    parser.add_argument("--policy", default=None, help="for offline fqe, the model file of the policy evaluated")
    parser.add_argument("--report", default=None,
                        help="for offline and for online's learner, where the JSON report is written instead of "
                             "local/reports/offline/<stamp>.json")
    parser.add_argument("--expectile", type=float, default=None, help="for offline iql and cql, the value regression's expectile")
    parser.add_argument("--beta", type=float, default=None,
                        help="for offline iql, awr and cql, the inverse temperature of the advantage weights")
    parser.add_argument("--judge", type=float, default=None,
                        help="for offline iql, awr and cql, the weight of the clone loss against the judge's labels "
                             "beside the advantage-weighted loss on the actions played; nought drops the judge")
    parser.add_argument("--critic-epochs", type=int, default=None,
                        help="for offline iql, awr and cql, passes over the training side that fit only the critic "
                             "before the actor moves on its advantages; under --follow, that many passes' worth of "
                             "updates over the decisions read at the start")
    parser.add_argument("--top", type=float, default=None,
                        help="for offline topbc, the share of each behaviour policy's trajectories kept by return; "
                             "for every offline method, the share at either end that return_alignment compares")
    parser.add_argument("--alpha", type=float, default=None, help="for offline cql, the weight of the conservative penalty")
    parser.add_argument("--load", default=None,
                        help="parameters to start from. A duel takes several, comma separated, and measures "
                             "each as an arm against the one baseline")
    parser.add_argument("--pin", default=None,
                        help="departures to measure a layer pinned to, comma separated, as arms of a duel. "
                             "This is the ablation the band a policy plays inside is read from. When collecting "
                             "the tactical layer, the one departure it plays throughout")
    parser.add_argument("--both-sides", action="store_true",
                        help="when collecting the tactical layer, record the arena's other side as well, which "
                             "the script plays")
    parser.add_argument("--save", default=None, help="where to write the parameters afterwards")
    parser.add_argument("--batch", type=int, default=None,
                        help="rows in one gradient step: the steps that make a reinforcement update when "
                             "training, the teacher's decisions in one minibatch when cloning")
    parser.add_argument("--stall-seconds", default=None,
                        help="game seconds a fight may go without a casualty before the arena calls it, "
                             "which is how decisive its fights are. Comma separated, they run as arms")
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
    parser.add_argument("--decay", default=None, choices=["cosine", "none"],
                        help="for offline bc, distill and topbc (and so clone --net set), how the learning rate moves after its warm-up: cosine "
                             "lowers it along a half cosine to a small share of itself by the last epoch, none holds "
                             "it; cosine unless given")
    parser.add_argument("--anchor", type=float, default=None,
                        help="weight on the divergence of the policy from the one it was loaded as, which holds "
                             "a run started from an imitation near its teacher except where the advantages keep "
                             "pointing away from it. Unset is nought, no anchor")
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
    parser.add_argument("--warmup", type=int, default=0,
                        help="updates at the start that fit the value head alone, holding the trunk and the "
                             "policy still. Meant for a run started from imitated parameters, whose critic "
                             "did not come with them")
    parser.add_argument("--dataset", dest="dataset_sources", action="append", default=[],
                        help="for a run that records, the directory to record into instead of "
                             "local/datasets/<layer>/<what>-<stamp>; for clone, offline and online, a recorded run to learn from, as "
                             "path or path:weight, the weight multiplying every decision in it, given once per run")
    parser.add_argument("--no-dataset", action="store_true",
                        help="record nothing, for a run whose decisions are not wanted")
    parser.add_argument("--shard-decisions", type=int, default=SHARD_DECISIONS,
                        help="for a run that records, the decisions gathered before a shard is written; smaller shards "
                             "reach a following learner sooner")
    parser.add_argument("--from", dest="source", nargs="+", default=None,
                        help="for a duel, report the journals of finished duels instead of running one")
    parser.add_argument("--out", default=None, help="for dataset reencode, the directory to write the run to")
    parser.add_argument("--verify", action="store_true",
                        help="for dataset inspect, rebuild every state from its materials and count those that differ")
    # The three that follow default to nothing here and are filled in from the cloning module when a clone is actually run, because quoting its numbers in this help text would be a second place for them to be stated and a first place for them to go stale. What was used is logged as the run starts.
    parser.add_argument("--smoothing", type=float, default=None,
                        help="how much of each label is shared out over the actions the teacher did not "
                             "choose, when cloning. Nought copies a deterministic script exactly and leaves "
                             "the training run after it nothing to explore with")
    parser.add_argument("--epochs", type=int, default=None,
                        help="passes over the teacher, when cloning")
    parser.add_argument("--patience", type=int, default=None,
                        help="epochs the held-out tenth may fail to improve for before cloning stops")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="when cloning, fit to this share of the training side's episodes only, a smaller share "
                             "being a subset of every larger one, while the held-out episodes stay whole; for a "
                             "learning curve over the amount of data")
    parser.add_argument("--keep-tainted", action="store_true",
                        help="clone from decisions about squads somebody outside the chain interfered with too")
    parser.add_argument("--greedy", action="store_true",
                        help="measure each policy a second time taking the likeliest action rather than "
                             "drawing one, as an arm beside the drawing one. Drawing is what the policy does "
                             "when it is operated, so it stays; what this adds is the same weights without "
                             "the exploration, on the same fights")
    parser.add_argument("--intruder", action="store_true",
                        help="inject the script intruder, which the design requires for the operational layer")
    parser.add_argument("--student", default=None,
                        help="when collecting, let the network saved here play while the script's judge labels "
                             "every board it reaches, so that the record covers the boards the network visits")
    parser.add_argument("--reload-seconds", type=float, default=None,
                        help="for collect --student and duel --load, read the learner's sidecar beside the model file "
                             "this often and serve each newer version it publishes; the version column records which "
                             "played")
    parser.add_argument("--follow", action="store_true",
                        help="for offline iql, awr or cql, learn without end from --dataset and the --watch runs, reloading "
                             "them as shards are added and publishing versioned weights to --save with a sidecar")
    parser.add_argument("--watch", action="append", default=[],
                        help="for offline --follow, a directory whose run directories of the layer are read, "
                             "complete or still growing; repeatable")
    parser.add_argument("--publish-every", type=int, default=500,
                        help="for offline --follow, updates between published versions")
    parser.add_argument("--rescan-seconds", type=float, default=30.0,
                        help="for offline --follow, the shortest time between two looks for new shards")
    parser.add_argument("--updates", type=int, default=None,
                        help="for offline --follow, stop after this many updates instead of at SIGINT or SIGTERM")
    parser.add_argument("--buffer", type=int, default=None,
                        help="for offline --follow and online, the most decisions the replay buffer holds; the oldest "
                             "shards of the watched runs are dropped first and the --dataset runs are kept whole")
    parser.add_argument("--count", type=int, default=1, help="for online, the actor games started")
    parser.add_argument("--offset", type=int, default=0,
                        help="for online, the instance directory the actor games start numbering at")
    parser.add_argument("--actors", default=None,
                        help="for online, the directory the actors record a run under and the learner watches, "
                             "local/datasets/<layer>/actors unless given")
    parser.add_argument("--actor-device", default=None,
                        help="for online, the actors' --device; unless given they serve a set network on a graphics "
                             "card when there is one")
    parser.add_argument("--actor-card-share", type=float, default=None, metavar="FRACTION",
                        help="for online, the actors' --card-share, 0.1 unless given")
    parser.add_argument("--sizes", default="1,2,4,8,16,32,64,128,256",
                        help="for bench, the batch sizes timed, comma separated")
    parser.add_argument("--repeats", type=int, default=200, help="for bench, timed calls per batch size")
    parser.add_argument("--explore", type=float, default=0.0,
                        help="when collecting, the share of decisions played evenly at random among the legal ones "
                             "instead of by whatever plays: the network, the rule, the pinned departure or the "
                             "script")
    parser.add_argument("--temperature", type=float, default=None,
                        help="when collecting, the temperature the judge's scores are softened at before they are "
                             "written beside each decision as the teacher's distribution")
    parser.add_argument("--rule", default=None, choices=list(PinnedOperations.RULES),
                        help="when collecting the operational layer, record this fixed rule's decisions instead of "
                             "the script's: home, nearest, weakest, richest, spawn, random, or carry (the nearest "
                             "rule's region, reached by a lift wherever one is open)")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="write the parameters to --save after every this many updates as well as at the end, "
                             "so that a long run stopped from outside keeps what it had learnt")
    parser.add_argument("--snapshots", default=None,
                        help="with --checkpoint-every, also keep every checkpoint in this directory under a name "
                             "of its own carrying the number of updates, so that a policy the run passed through "
                             "can be played later")
    parser.add_argument("--reward", default=None, choices=list(REWARDS),
                        help="the named reward the operational layer or the economy is paid under: predicted (the "
                             "default) pays the match score at its end and shapes by the score expected from the "
                             "board, undiscounted; flows pays shares of the board each period, as the comparison. "
                             "For offline and online, recorded rewards are priced again at it")
    parser.add_argument("--terminal-weight", type=float, default=None,
                        help="how much of the match score the end of a match pays the operational layer or the economy")
    parser.add_argument("--predictor", default=None,
                        help="a predictor file (`learn predictor --save`) the expected score is read from instead of "
                             "the adopted one")
    parser.add_argument("--achievement-weight", type=float, default=None,
                        help="multiplier on the operational layer's pay for meeting the strategic orders, and on "
                             "its charge for overspending the loss allowance. Unset is nought")
    parser.add_argument("--local-exchange", type=float, default=None,
                        help="also pay each operational decision this much of its own squad's exchange, worth "
                             "destroyed less worth lost per thousand credits, while it stands")
    parser.add_argument("--review-seconds", type=float, default=None,
                        help="game seconds a learnt operational layer lets a running errand go before deciding "
                             "about the squad again when nothing else has called for it")
    parser.add_argument("--value-flow", type=float, default=None,
                        help="what the operational layer or the economy is paid each period per unit of our share "
                             "of the worth on the board, from -1 to +1, over the named reward's figure")
    parser.add_argument("--ground-flow", type=float, default=None,
                        help="what the operational layer or the economy is paid each period per unit of our share "
                             "of the resource points held, from -1 to +1, over the named reward's figure")
    parser.add_argument("--script", action="store_true",
                        help="run the handwritten layer on both sides, which is the baseline and the way to measure the arena itself")
    parser.add_argument("--script-opponent", action="store_true",
                        help="fight the script tactical layer rather than the policy being trained")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false",
                        help="duel without taking the handwritten layer against itself alongside. The "
                             "baseline is taken by default because how far the arena leans is a property of "
                             "the seed, so a policy's score is only readable beside the lean of the very "
                             "seed it was measured under")
    parser.add_argument("--record", default=None,
                        help="where episodes are written, defaulting to local/episodes/<what>-<stamp>.jsonl")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_method(arguments) -> None:
    """Fills in --method for the command: a following learner (`offline --follow`, `online`) takes iql unless given and refuses anything but iql, awr or cql, and every other command takes bc unless given."""
    if arguments.what == "online" or (arguments.what == "offline" and arguments.follow):
        from .online import following_method

        arguments.method = following_method(arguments.method, "online" if arguments.what == "online" else "offline --follow")
    elif arguments.method is None:
        arguments.method = "bc"


def _dispatch(arguments) -> int:
    if arguments.what in ("clone", "offline", "dataset", "bench", "online", "predictor"):
        arguments.dataset = None
    elif len(arguments.dataset_sources) > 1:
        raise SystemExit("a run records into one directory, and --dataset was given more than once")
    else:
        arguments.dataset = arguments.dataset_sources[0] if arguments.dataset_sources else None
    if arguments.paths and arguments.what != "dataset":
        raise SystemExit(f"{arguments.what} takes no positional arguments: {' '.join(arguments.paths)}")
    _resolve_method(arguments)
    # One stamp names the run's journal and its dataset, so that the two are found from each other.
    arguments.stamp = paths.stamp()

    log_file = paths.configure_logging("learn", arguments.verbose, fmt="%(asctime)s %(levelname)-7s %(message)s")
    log.info("logging to %s", log_file)
    if arguments.max_seconds <= 0:
        plays_matches = (arguments.what in ("operations", "economy")
                         or (arguments.what == "collect" and arguments.layer in (OPERATIONAL, ECONOMIC)))
        arguments.max_seconds = MATCH_SECONDS if plays_matches else ARENA_SECONDS

    if arguments.what == "tactics":
        return train_tactics(arguments)
    if arguments.what == "operations":
        return train_operations(arguments)
    if arguments.what == "economy":
        return train_economy(arguments)
    if arguments.what == "clone":
        return clone(arguments)
    if arguments.what == "offline":
        return offline_command(arguments)
    if arguments.what == "duel":
        return duel(arguments)
    if arguments.what == "matchups":
        return matchups(arguments)
    if arguments.what == "dataset":
        return dataset_command(arguments)
    if arguments.what == "predictor":
        return predictor_command(arguments)
    if arguments.what == "bench":
        from .bench import bench_command

        return bench_command(arguments)
    if arguments.what == "online":
        from . import online

        return online.run(arguments, build_parser(), getattr(arguments, "argv", []))
    return collect(arguments)


if __name__ == "__main__":
    sys.exit(main())
