"""Runs the constructed operations arena over one or more operational arms and reports what each came to and how they differ board by board.

    python -m rwintel.learn.ops_run --instances 4 --episodes 50 --map Hills
    python -m rwintel.learn.ops_run --instances 8 --episodes 10 --our script --our pin --our massed --our learnt --load local/ops-arena.pt

Start this first and then the game instances, as with every other runner here.

With the script chain alone it measures the self-play zero: the script `Operations` on both sides of the mirror board makes the two sides' side scores exact negatives every episode, so a run of many fresh boards must pool the reported side score to nought. A nonzero mean is a board lean the mirror-symmetric draw was supposed to have removed — the operational analogue of the headquarters-in-a-squad bias the fight baseline once carried — and it is the only instrument that can see the leans the within-episode sign check cannot: all-enemy garrisons, the free base polluting the region block, a non-congruent reflected layout, and empty regions reading a half under asymmetric reach. Every one of those is a break in exchange symmetry, and only this mean sees it. This is the gate the arena must pass before any operational policy measured on it is trusted, exactly as the engagement arena gates on its own script-against-itself baseline. It is a statement about the board UNDER THE TACTICAL LAYER THE RUN WAS MADE WITH and under no other: the zero is a statistical claim about how the fighting on a mirrored board comes out, not a structural guarantee, so a run made under fresh tactical parameters has to re-take the gate before any arm measured beside it is trusted.

Given several arms it runs all of them on the same boards. The arms alternate inside each instance and the board is held still until every one of them has played it, so each board is one paired observation across the arms and the run reports every pair's difference itself. That is the honest way to compare two operational policies here, because a board's draw moves the side score by more than the arms differ: one arm's episodes scatter by about 0.11 while the differences being looked for are around 0.04. Running the arms separately and subtracting the means pays for that scatter twice and also has to assume two runs, made at different moments on a machine doing different things, were otherwise alike. Running them together assumes nothing of the sort.

It trains nothing and keeps no trajectories: no layer here is handed a rollout, so with nowhere to record a decision none is recorded. That covers the trained tactical layer `--tactics` freezes under both sides of the board as well as the operational arms above it — it is read at its likeliest departure rather than drawn from, exactly as a duel reads the policy it is measuring rather than the one it is training.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ..control.policy.operations import Concentrated, Operations
from ..wire.action import Deviation
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from ..eval.journal import Journal, default_path
from ..eval.sampling import Summary
from .__main__ import _device, _load
from .deciders import (
    NetworkOperations,
    NetworkTactics,
    PinnedRegion,
    operational_batcher,
    tactical_batcher,
)
from .encoding import TACTICAL_SIZE
from .inference import Batcher
from .layers import LearntOperations, LearntTactics
from .net import (
    INPUT_WEIGHT,
    EncodingRefused,
    OperationalNet,
    TacticalNet,
    load_encoded,
    reads,
)
from .ops_arena import (CATCHMENT_RADIUS, GARRISON_SCALE, HORIZON_MS, SCRIPT_TACTICS,
                        STANDING_MIRROR, STANDINGS, OpsArena)

log = logging.getLogger(__name__)

#: How far apart two instances' arena draws are set, matching the engagement arena's stride. It only has to exceed the episodes one instance will ever run, and it is prime so two runs at neighbouring seeds do not lay one instance's stream on top of another's.
INSTANCE_STRIDE = 100003


def _arena_seed(base_seed: int, session, arms: int = 1) -> int:
    """The seed an arena episode draws its board from, advancing with the board as well as with the instance, so every board is fresh rather than the first one replayed, and the sample size a pooled mean rests on is the boards fought rather than the instances.

    With several arms in one run the board must advance once every arm has played it, not once per episode: a session takes its arm as `records % arms`, so `records // arms` is which board it is on and every arm meets that board exactly once. That is what makes a multi-arm run a paired one — the same construction under each arm, in the same instances, at the same moment of the same machine — and it is the difference between a comparison that has to trust two separate runs to have been alike and one that does not have to.
    """
    return base_seed + INSTANCE_STRIDE * session.instance + len(session.records) // max(1, arms)


@dataclass(frozen=True)
class FrozenTactics:
    """The tactical layer an arena run does its fighting under, and the name a journal writes it down by.

    Empty when a run named no tactical parameters, and then the arena builds the handwritten `Tactics` on both sides for itself, which is what every measurement taken on this arena so far was made under. Given parameters, `build` is the factory the arena calls once for each side, `batcher` is the single inference server the one network answers through, and `name` identifies the parameters themselves so that a later comparison can refuse to pair two runs made under different fighters.
    """

    build: Optional[Callable] = None
    batcher: Optional[Batcher] = None
    name: str = SCRIPT_TACTICS

    def stop(self) -> None:
        """Ends the inference server, if this run started one. A run fighting under the handwritten layer has nothing to tear down."""
        if self.batcher is not None:
            self.batcher.stop()


def _digest(path: str) -> str:
    """What a set of parameters is called in a journal: the first half of the SHA-256 of the file's own bytes.

    The parameters themselves and not the path they were read from. A path is a nickname that changes underneath itself — a tactical training run overwrites whatever its `--save` names, every time it is run — so two runs a week apart would claim one instrument, would be paired board by board, and would report the change of fighter as a difference between the operational arms, which is exactly the silent misreading the paired comparison exists to refuse. The file is tens of kilobytes and is read once at the start of a run, and the runner logs the path beside the digest so a person can tie the two together.
    """
    with open(path, "rb") as handle:
        return "sha256:" + hashlib.sha256(handle.read()).hexdigest()[:16]


def frozen_tactics(path: Optional[str], device_name: Optional[str] = None) -> FrozenTactics:
    """Trained tactical parameters read off a file and made into the layer that fights, frozen, beneath BOTH sides of the arena. Nothing at all when no path was given, which leaves the arena to build the handwritten ladder for itself.

    Both sides, because the arena builds its two tactical layers from the one factory it is handed and there is deliberately no way to hand it two. Putting a trained fighter on one side only would plainly stop the two sides being exchangeable, and the script arm's pooled self-play mean — the arena's one check on whether the board leans — would no longer have to be nought. It is also what the project's learning order means when it says the operational layer is trained against a frozen tactical layer: the frozen layer is the whole environment's fighting, not our own side's.

    The same layer on both sides is the least a mirror requires. The other side's view turns the ownership flags over without reflecting the coordinates, so the two sides are given one function on inputs congruent in meaning and not in place — and what makes that safe is that no tactical feature is measured against the map's axes any more: the direction a squad's errand points is measured from the way leading out of that side's OWN home, which a half turn of the board carries onto the other side's and so leaves alone. A learnt tactical layer therefore answers a squad and its exact reflection alike, and the encoding suite pins it by reading one mirrored board from both sides. A run made under frozen tactical parameters is a different instrument all the same — it pairs only with runs made under the same parameters, which the journal records so the comparison can refuse rather than be trusted, and its script arm has to re-pass the self-play zero before any arm measured beside it is believed.

    One network and one batching server for the run, and a fresh layer object per side per episode. One network is the point of batching at all: every request from both sides of every instance then meets in the same window, where two networks would halve the batch per call and buy nothing. The layer objects cannot be shared, because a tactical layer keeps per-side state — what each of its squads has destroyed, and the board it last saw — and the two sides are handed different boards.

    Read at its likeliest departure rather than drawn from, on both the batcher and the decider, since one of them answers when there is a batcher and the other when there is not. Drawing is the exploration a run needs of the layer it is training, and this layer is not being trained; a deterministic layer also leaves the self-play zero something it can repeat, where one drawing its departures would put a fresh difference between the two sides into every board.
    """
    if not path:
        return FrozenTactics()
    if not os.path.exists(path):
        # Refused rather than started from nothing, unlike the tolerant load a training run gives the layer it is about to train. Beginning from a fresh policy is meaningful for the layer being learnt and is never meaningful for the instrument beneath it: a mistyped path would leave a randomly initialised fighter under the whole arena, and the run would be journalled as having been made under trained parameters with nothing downstream able to tell.
        raise SystemExit("no tactical parameters at %s, so there is nothing to freeze under the arena" % path)
    device = _device(device_name)
    state = torch.load(path, map_location=device)
    # An operational network's file has a first layer too, so the test that catches parameters of the wrong layer handed to this option is the width of the input that layer reads. A file that carries no first layer at all fails the same test and is refused the same way; the strict load below is the backstop for everything subtler than a different layer.
    features = reads(state)
    if features != TACTICAL_SIZE:
        raise SystemExit("the parameters at %s are not a tactical layer's: their first layer reads %s feature(s) "
                         "where a tactical layer reads %d, which is what another layer's parameters look like here"
                         % (path, "none" if features is None else features, TACTICAL_SIZE))
    # The width is read off the file rather than asked for as a flag. The first layer is one linear map from the tactical features to the width, so the file states its own width, and a flag that had to be kept in step with a file would only ever fail a load that was going to succeed. The duel needs a width because it can also build a fresh network; here there is never a fresh network.
    net = TacticalNet(width=int(state[INPUT_WEIGHT].shape[0])).to(device)
    # And the width having matched says only that the file is a tactical layer's, not that its features mean what they now mean. This is the one place it matters most: the layer frozen here fights for both sides of every board and is never trained, so parameters fitted to an older reading of a slot would fight the whole run on a misreading and nothing downstream could tell — the run would simply be journalled as having been made under trained parameters.
    try:
        avowal = load_encoded(net, state)
    except EncodingRefused as refused:
        raise SystemExit("the parameters at %s cannot be frozen under the arena: %s" % (path, refused))
    if avowal:
        # The instrument beneath the whole run, accepted on somebody's word about what it was fitted to, is worth saying at every run rather than only where the word was written. The name below is a digest of the file, so avowing one changes its name and no run made before the avowal can be confused with one made after it.
        log.warning("the feature list at %s is a person's word and not a fit's record: %s", path, avowal)
    batcher = tactical_batcher(net, device=device, greedy=True)
    name = _digest(path)

    def build(session, catalogue):
        # No rollout: this layer is read and not learnt from, so with nowhere to record a decision it records none. That is the argument that matters here rather than thrift — the arena's operational layer records its own decisions against the very same squad ids, so a tactical layer handed the training run's buffer would splice tactical decisions into the operational trajectories and the trainer would feed them to a network that reads a different state and answers a different question.
        #
        # The layer's `status_terminals` is left at its default, which allows the conditions written into a contract to end an errand, rather than turned off the way the engagement arena's duel turns it off. That flag is inert without a rollout, so nothing is paid either way; it is set truthfully because this board reissues contracts every operational period while a constructed fight is one contract that nothing reissues, and a future reader must not find a lie here.
        return LearntTactics(session, catalogue, NetworkTactics(net, device, batcher, greedy=True), None, -1)

    log.info("both sides of the board will fight under the tactical parameters at %s (%s), read at their likeliest "
             "departure and recording nothing", path, name)
    return FrozenTactics(build=build, batcher=batcher, name=name)


@dataclass(frozen=True)
class Arm:
    """One operational arm of a run: which layer it puts on our side, what the run calls it, and what identifies it in a journal.

    `kind` is the rule — the script chain, the pin, the massed ablation, the concentrating arm, or a learnt network. `label` is what the run and the journal call it, which is the kind itself for every handwritten arm and carries the parameter file's stem for a learnt one, since a run may hold several learnt arms and two of them under one name would pair with each other. `name` is the identity rather than the nickname: a digest of the parameters for a learnt arm, exactly as the frozen tactical layer is named by its content, so that two runs whose arms share a label but not a policy can be refused rather than silently pooled.
    """

    kind: str
    label: str
    name: str
    path: Optional[str] = None
    net: object = None
    device: object = None
    batcher: object = None
    #: What the `wanted` arm puts a strategic priority at, against the terms the ladder reads for itself. Meaningless to every other arm, which leave it where it is.
    weight: float = 0.0


def _arm(arguments, arm: "Arm", arms: int = 1, frozen: FrozenTactics = FrozenTactics()):
    """One `OpsArena` per episode of one arm, seeded so that every arm of the run meets the same boards.

    `script` on our side is the self-play zero: the same chain on both sides of the mirror, whose pooled score must be nought. `pin` is a layer that sends every squad to the lowest-numbered legal region and task, making no operational choice at all; subtracting it from the script arm board by board cancels the enemy and the board lean and leaves how much the script's careful deployment beat making no choice, which is the resolution the arena exists to produce. The enemy is always the script, so every arm is measured against one fixed opponent.

    What the pin is not is a concentration arm. Its region is the lowest live id on the map and the contests are drawn about the board centre, so its squads march off the scored ground: measured on Hills at a catchment of four hundred, no squad of it was inside any catchment in fifty of sixty-eight scored episodes on one draw and in thirty-five of fifty-one on another, and the nearest ended a median of seven hundred and seventy and seven hundred and eighty-three units out. It beat the script's careful deployment there, but not by overwhelming anything — by leaving discs to empty out, and an empty disc reads a half rather than a loss. That is a floor for abandoning the board, and reading it as evidence that concentration wins was wrong. `diagnose` reports the reach of every arm for this reason.

    `massed` was meant to be the concentration arm the reading needed, and measurement says it is not one. It is the script ladder with the discount a region takes for the strength we already have standing in it removed — and on this arena that discount is already nought, because the squads are at the staging point and not in the contested regions when the choice is made. Measured: the two arms score identically on 43 of 51 boards at the default garrison and on 37 of 38 at half of it, and the engine does not reproduce, so identical scores mean identical decisions. It is therefore an ablation of one term of the ladder, worth keeping as that, and it is no evidence about concentration. An arm that actually concentrates has to replace the region choice rather than remove a term from it, which is what `concentrate` does. Against the script it says what the spreading rule costs; against the pin it says whether massing on the right ground beats leaving it.

    `narrow` is the ladder with each doctrine's candidate set left as its own filter builds it, which is the ladder as it stood before the strategic layer's priority could admit a region to that set. Against the present script arm it says what admission is worth. It replaced an earlier arm that tried the opposite rule - narrowing the candidates to the wanted ones - and that rule is not kept, because measurement said it changed one operational decision in 41508: narrowing cannot move a decision whose candidate set holds no wanted region to begin with.

    `wanted` is the ladder with the strategic layer's request re-weighted against the terms the ladder reads for itself, written `wanted:3` for three times its own worth. It exists because admitting the wanted ground to the candidates was implemented, measured, and moved nothing, and the arithmetic that survived that measurement is a matter of scale: a priority never exceeds one, four resource points are worth 0.6 and enemy strength on ground we hold another 0.3, so a contested point that draws no income loses the scoring to a quiet mine even once it is a candidate. This arm asks whether that is what sends the ladder off the scored ground, and unlike the concentrating arm it does not replace the choice -- every term is still read, and only what the request weighs has moved.

    `concentrate` is the arm that does what the massed arm was supposed to do. It keeps the doctrine's own choice of task and overrides only the region, sending every squad at the single region the strategic layer wants most. The contested discs are drawn from a higher band than the unscored ground and the two bands overlap, so that is every squad at one contest on almost every board and not by construction on any — concentration in the plain sense, made by replacing the choice rather than by removing a term from it.

    `learnt` is a trained network read from a file. A run may carry several of them at once, each with its own parameters, which is what puts two generations of one training line on the same boards: the length of a training run is the thing an arm comparison most often has to resolve, and taking the two generations in separate runs pays the arena's board scatter twice over and has to assume two runs on a machine doing different things were otherwise alike.

    Whatever the arm, the tactical layer under both sides is whatever `frozen` carries: the handwritten ladder by default, and trained parameters where the run named some. Every arm of a run fights under the same one, which is what keeps the arms comparable to each other and what makes the run as a whole one instrument.
    """
    if arm.kind == "pin":
        operations = lambda session, catalogue: LearntOperations(session, catalogue, PinnedRegion(), None, -1)
    elif arm.kind == "massed":
        operations = lambda session, catalogue: Operations(session, catalogue, crowding=0.0)
    elif arm.kind == "narrow":
        operations = lambda session, catalogue: Operations(session, catalogue, admit=False)
    elif arm.kind == "concentrate":
        operations = lambda session, catalogue: Concentrated(session, catalogue)
    elif arm.kind == "wanted":
        operations = lambda session, catalogue: Operations(session, catalogue, wanted=arm.weight)
    elif arm.kind == "learnt":
        # The trained layer read greedily — its most probable region and task, not a draw — since this measures the policy rather than trains it, and with no rollout it records nothing. Each learnt arm closes over its own network and its own batching server, so several of them in one run neither share parameters nor queue behind one another's window.
        operations = lambda session, catalogue: LearntOperations(
            session, catalogue, NetworkOperations(arm.net, arm.device, arm.batcher, greedy=True), None, -1)
    else:
        operations = None

    def build(session) -> OpsArena:
        return OpsArena(session, operations=operations, tactics=frozen.build, tactics_name=frozen.name,
                        operations_name=arm.name,
                        seed=_arena_seed(arguments.seed, session, arms),
                        horizon_ms=arguments.horizon * 1000, our_squads=arguments.squads,
                        catchment_radius=arguments.radius, contest_pairs=arguments.pairs,
                        garrison_scale=arguments.garrison, standing=arguments.standing)
    return build


#: The operational arms a run may ask for. A learnt one may be written `learnt:PATH` to give it its own parameters, which is how one run carries several of them.
ARM_KINDS = ("script", "pin", "massed", "narrow", "concentrate", "wanted", "learnt")


def arms_of(arguments) -> List[Arm]:
    """The run's arms, read off the repeated `--our` flags and named, with nothing yet loaded and no thread yet started.

    A learnt arm is written either as the bare word, which takes the run's single `--load`, or as `learnt:PATH`, which carries its own. The second form is what lets one run hold two generations of a training line and pair them board by board; the first is kept because every measurement so far was taken with it and its journals name their arm `ops-learnt`.

    Two arms whose label would be the same are refused rather than run: they would be written into one journal under one name, and a comparison reading that journal cannot tell two arms apart afterwards — it would read them as one arm that played every board twice, which is exactly the repeated board the pairing drops. A missing path is refused here for the reason the duel refuses it: a mistyped path would otherwise leave a freshly initialised network in place, and the run would measure a random policy and journal it under the trained one's name.

    Nothing is loaded here on purpose. This runs before the tactical layer beneath the board is read, so that every refusal this can make is made while the run holds no network, no inference thread and no game.
    """
    arms: List[Arm] = []
    for our in arguments.our:
        kind, _, path = our.partition(":")
        if kind not in ARM_KINDS:
            raise SystemExit("no such operational arm as %s: the arms are %s, and a learnt one may name its own "
                             "parameters as learnt:PATH" % (kind, ", ".join(ARM_KINDS)))
        if path and kind not in ("learnt", "wanted"):
            raise SystemExit("only a learnt arm carries parameters, and %s named some" % kind)
        if kind == "wanted":
            # The weight is the whole of what this arm is, so a bare `wanted` would be the script arm under another name and is refused rather than run as one.
            try:
                weight = float(path)
            except ValueError:
                raise SystemExit("the wanted arm is written wanted:WEIGHT, as in wanted:3, and %r is not a weight"
                                 % (path or ""))
            if weight < 0.0:
                raise SystemExit("a negative weight would send the layer away from the ground the strategic layer asked for, which is not what any arm here is for")
            label = "wanted-" + path
            arms.append(Arm(kind=kind, label=label, name=label, weight=weight))
            continue
        if kind != "learnt":
            arms.append(Arm(kind=kind, label=kind, name=kind))
            continue
        path = path or arguments.load
        if not path:
            raise SystemExit("the learnt arm has no parameters to measure: give --load, or write it as learnt:PATH")
        if not os.path.exists(path):
            raise SystemExit("no parameters at %s, so there is nothing for the learnt arm to measure" % path)
        # Named after the file it was read from where a run carries several, and by the bare word where it carries one, so that the journals every measurement so far was written into go on being read under the name they carry. The identity is the digest of the parameters themselves, since a path is a nickname that changes underneath itself.
        stem = os.path.splitext(os.path.basename(path))[0]
        label = "learnt" if our == "learnt" else "learnt-" + stem
        arms.append(Arm(kind=kind, label=label, name=_digest(path), path=path))
    labels = [arm.label for arm in arms]
    for label in labels:
        if labels.count(label) > 1:
            raise SystemExit("two arms would be journalled as %s, so nothing downstream could tell them apart: give "
                             "each learnt arm its own parameters as learnt:PATH" % label)
    names = [arm.name for arm in arms if arm.kind == "learnt"]
    for name in names:
        if names.count(name) > 1:
            # Two labels over one file, which is the bare word and its own path given together. The two arms would be one policy measured twice and the run would report it as differing from itself by the engine's scatter, under two names that look like two policies.
            raise SystemExit("two learnt arms read the same parameters (%s), so they are one policy under two names" % name)
    return arms


def load_arms(arms: Sequence[Arm], device_name: Optional[str] = None) -> List[Arm]:
    """Every learnt arm with its network read off its file and its own batching server started; the handwritten arms unchanged.

    One network and one server per learnt arm, shared across every instance of the run, for the reason the training runner builds its network once: the network is what is being measured, and one copy batched across the instances is the whole point of batching the inference. Separate servers rather than one, because two arms are two policies and a batch is one forward pass of one network.
    """
    loaded: List[Arm] = []
    for arm in arms:
        if arm.kind != "learnt":
            loaded.append(arm)
            continue
        device = _device(device_name)
        net = OperationalNet().to(device)
        _load(net, arm.path, device)
        log.info("the %s arm reads the operational parameters at %s (%s)", arm.label, arm.path, arm.name)
        loaded.append(replace(arm, net=net, device=device,
                              batcher=operational_batcher(net, device=device, greedy=True)))
    return loaded


def pool(sessions, arm: Optional[str] = None, reading: str = "side_score") -> Summary:
    """Every scored episode's side score as one count, one mean and one spread, over one arm of the run or over all of them. An episode cut off before its horizon carries no score and is skipped, so a run whose match length did not clear the horizon pools nothing rather than pooling a nought that was never measured.

    Which reading is pooled is the caller's: `side_score` is the discs as they stood at the horizon and `side_tenure` the mean of the same discs over the episode. Both are written by every episode, so one run answers both questions and neither has to be taken on its own boards.
    """
    scores: List[float] = [float(record.statistics.get(reading, 0.0))
                           for session in sessions for record in session.records
                           if record.statistics.get("scored") and (arm is None or record.arm == arm)]
    return Summary.of(scores)


def report(summary: Summary, arm: str = "script", reading: str = "side_score") -> None:
    """The run's pooled side score, as a mean and the two standard errors it has to sit inside.

    What a nonzero mean means depends on which arm produced it, and saying the wrong one of these is worse than saying nothing. For the script arm the two sides are the same chain, so the mean is the self-play zero: inside the interval is a board with no lean the mirror draw did not remove, and outside it is a lean to be found and fixed before the arena is trusted. For every other arm our side is deliberately not the enemy's chain, so a mean outside the interval is the arm beating the script — which is the measurement, not a fault — and warning about a lean there would be reporting the instrument working as if it were broken. The lean is read once, on the script arm, and every other arm is then read against it and against the other arms board by board with `ops_compare`.
    """
    if summary.n == 0:
        log.warning("no episode reached its horizon, so there is no side score to pool: is the match length longer than the horizon plus the settle and spawn waits?")
        return
    named = "held over the episode" if reading == "side_tenure" else "held at the horizon"
    interval = 2.0 * summary.sd / math.sqrt(summary.n) if summary.n > 1 else 0.0
    log.info("pooled side score (%s) over %d scored episode(s): %+.4f, 2 standard errors %.4f, interval %+.4f to %+.4f",
             named, summary.n, summary.mean, interval, summary.mean - interval, summary.mean + interval)
    if summary.n < 2:
        log.info("one episode has no spread, so it says nothing about how this arm stands; run several hundred")
        return
    outside = interval > 0.0 and abs(summary.mean) > interval
    if arm != "script":
        log.info("this is the %s arm against the script, so the figure is how far the arm stands from the script's "
                 "side of the mirror and not a lean: the board's own lean is the script arm's figure, and the honest "
                 "comparison against another arm is the paired one over the same boards (rwintel.learn.ops_compare)", arm)
        if not outside:
            log.info("the interval holds nought, so this arm is not told apart from the script by these episodes")
    elif outside:
        log.warning("the pooled side score is outside two standard errors of nought, so the board leans under this draw and a policy measured on it would be reading the lean: find and remove it before trusting the arena")
    elif summary.sd <= 0.0:
        # A gate that cannot fail is not a gate. With no spread at all the interval is nought wide, the mean sits inside it whatever it is, and the run reports a pass — while a board on which every episode scores identically has resolved nothing, which is the one condition under which the self-play zero is met trivially and means nothing. It has been reached before: at a horizon too short for the staged squads to reach their contests, the mirror garrisons cancel and every board comes back at exactly nought.
        log.warning("every scored episode came back at the same figure, so this board resolved nothing and its "
                    "self-play zero is the mirror garrisons cancelling rather than a board with no lean: check the "
                    "horizon, the catchment and whether the squads reached their contests at all")
    else:
        log.info("the pooled side score holds nought within two standard errors, which is the self-play zero the arena has to pass before it is trusted")


def diagnose(sessions, arm: str, radius: float) -> None:
    """Whether this arm's own squads ever reached the ground the episode was scored on.

    An arm can score well without its squads ever entering a catchment, and the figure alone cannot tell that apart from an arm that fought for the ground and won it — the two look identical in the pooled mean. The distinction is the whole difference between measuring a deployment and measuring an abstention, and the arena already writes down what settles it: how far the nearest surviving squad member ended from a contest, and how many of them ended inside one.

    This is the check that was in hand and not pointed at the pinned arm. That arm sends every squad to the lowest-numbered legal region, which is a fixed region id with nothing to do with where the contests were drawn, so its squads finished outside every scored disc in most episodes and what looked like concentration beating a spread was an arm that had left the scored board. An arm whose median reach is outside the catchment is not deploying onto the contests, whatever its score says, and the run says so rather than leaving it to be noticed.

    The other side's figures are reported beside this side's for a different question: whether the two deployments reached their contests alike at all. The board is a point reflection, but the ground under it is not — this side stages from a site the map was searched for and the other from that site's reflection, which is wherever it lands — and the script arm's self-play zero bounds what that asymmetry is worth only under the script. A pair of medians a catchment apart is a physical difference between the sides large enough to decide discs, and it would show up in an arm comparison as an operational difference with nothing to tell the two apart.
    """
    def figures(field: str):
        reaches = sorted(float(record.statistics.get(field + "_reach", -1.0))
                         for session in sessions for record in session.records
                         if record.arm == arm and record.statistics.get("scored")
                         and float(record.statistics.get(field + "_reach", -1.0)) >= 0.0)
        absent = sum(1 for session in sessions for record in session.records
                     if record.arm == arm and record.statistics.get("scored")
                     and not record.statistics.get(field + "_in_catchment"))
        return reaches, absent

    reaches, absent = figures("our")
    scored = sum(1 for session in sessions for record in session.records
                 if record.arm == arm and record.statistics.get("scored"))
    if not reaches or not scored:
        return
    median = reaches[len(reaches) // 2]
    log.info("this arm's nearest squad member ended a median %.0f world units from a contest, against a catchment of "
             "%.0f, and no squad of it was inside any catchment in %d of %d scored episode(s)",
             median, radius, absent, scored)
    if median > radius:
        log.warning("this arm's squads ended outside the catchment in the median episode, so its score is not a "
                    "measure of how it deployed onto the contests but of what happened on ground it never reached: "
                    "read it as a floor for abandoning the scored board, not as a deployment")
    theirs, their_absent = figures("their")
    if theirs:
        their_median = theirs[len(theirs) // 2]
        log.info("the other side of the same boards ended a median %.0f world units out, and was inside no catchment "
                 "in %d of %d", their_median, their_absent, scored)
        if abs(median - their_median) > radius:
            log.warning("the two sides ended a whole catchment apart in how far they were from their contests, so "
                        "the mirror is not congruent in the ground it lays under the two deployments and part of "
                        "this arm's score is that difference rather than its choices")
    _discs(sessions, arm)
    signal(sessions, arm)


#: How far a final share must sit from an even split before the disc is called won or lost rather than neither. A disc both sides emptied reads exactly a half and is no more a defeat than a victory, and a disc decided by a handful of health is not the kind of outcome a deployment should be credited with either.
_DECIDED = 0.05


def _discs(sessions, arm: str) -> None:
    """How the arm's contested discs ended, split by what each one started as.

    The pooled side score cannot tell an arm that took ground from one that abandoned it, because a disc left empty reads a half and so scores better than a disc assaulted and lost. What separates them is the opening: a disc the enemy's garrison stood on is one this side had to TAKE and can only gain on, and a disc its own garrison stood on is one it had to HOLD and can only lose on. Measured on this arena at the default draw, the handwritten ladder takes about seven of every hundred it has to take and a layer that masses every squad on one contest takes about twenty-one, and the two sit within a few hundredths of each other in the pooled mean — so the tally is the statistic that says which skill an arm actually has, and it was being recomputed by hand from the journal every time it was wanted.
    """
    tally: Dict[str, int] = {}
    for session in sessions:
        for record in session.records:
            if record.arm != arm or not record.statistics.get("scored"):
                continue
            held = record.statistics.get("held") or {}
            for region, share in (record.statistics.get("shares") or {}).items():
                start = float(held.get(region, 0.5))
                kind = "take" if start < 0.25 else "hold" if start > 0.75 else "open"
                end = ("won" if float(share) > 0.5 + _DECIDED else
                       "lost" if float(share) < 0.5 - _DECIDED else "neither")
                tally[kind + "/" + end] = tally.get(kind + "/" + end, 0) + 1
                tally[kind] = tally.get(kind, 0) + 1
    if not tally:
        return
    log.info("of the discs an enemy garrison opened on, which are the ones this arm could only gain by taking, it took "
             "%d of %d and left %d neither; of the discs its own garrison opened on, which are the ones it could only "
             "lose, it kept %d of %d and lost %d",
             tally.get("take/won", 0), tally.get("take", 0), tally.get("take/neither", 0),
             tally.get("hold/won", 0), tally.get("hold", 0), tally.get("hold/lost", 0))


def signal(sessions, arm: Optional[str] = None) -> None:
    """How decisive this arm was, and how many of the arena's payments landed on a decision.

    The arena pays a squad every operational period the movement of its own scored figure, so how long an errand ran no longer decides how much of the episode a payment reaches — the payments telescope and reach all of it. What the ratio still says is whether an arm settled on a deployment or changed its mind: an episode in which four contracts stood from the staging point to the horizon and one in which they were re-drawn every period report the same score, the same shares and the same disc tallies, and nothing else here tells them apart. That is a real difference between arms and worth reporting per arm, because the handwritten ladder holds a squad on the errand it is running while a learnt layer re-draws as it likes.

    The three figures are the decisions the squads were given, the errands those decisions were divided into, and how many horizon payments actually landed on a decision. The last is nought for every arm of this runner and that is not a fault: no arm here is handed a rollout, so no decision is recorded and there is nothing for a payment to land on. It is reported all the same, because it is the figure a training run has to be read by and a measuring run is where the errand lengths it is compared against are taken.
    """
    periods = errands = terminals = staged = on_priority = massed = 0
    their_periods = their_errands = their_on_priority = 0
    # Whether the fields are there at all, kept apart from their values: an episode written before one existed carries no count, and a missing count read as nought would report an arm as having named no scored ground, or never massed, when nothing had asked.
    counted = False
    counted_massing = False
    counted_theirs = False
    for session in sessions:
        for record in session.records:
            if (arm is not None and record.arm != arm) or not record.statistics.get("scored"):
                continue
            periods += int(record.statistics.get("periods", 0))
            errands += int(record.statistics.get("errands", 0))
            terminals += int(record.statistics.get("terminals", 0))
            staged += int(record.statistics.get("squads", 0))
            if "on_priority" in record.statistics:
                counted = True
                on_priority += int(record.statistics["on_priority"])
            if "massed" in record.statistics:
                counted_massing = True
                massed += int(record.statistics["massed"])
            if "their_periods" in record.statistics:
                counted_theirs = True
                their_periods += int(record.statistics["their_periods"])
                their_errands += int(record.statistics.get("their_errands", 0))
                their_on_priority += int(record.statistics.get("their_on_priority", 0))
    if not periods or not errands:
        return
    # No share of the decisions is quoted any more. It used to be `min(staged, errands) / errands`, on the ground that the horizon paid each squad's last errand and no other, and that ground is gone: every period is paid its own movement now, so a payment reaches every decision whatever the errands come to, and the old figure would assert the opposite of the truth on every run.
    log.info("this arm's squads took %d operational decision(s) over %d errand(s), so an errand ran %.1f decision(s) "
             "before the arm changed its mind; %d horizon payment(s) landed on a decision out of %d squad(s) staged",
             periods, errands, periods / errands, terminals, staged)
    # Whether those decisions were even pointed at ground the episode is scored on. An arm can end far from every contest because it kept changing its mind and never arrived, or because it was never sent at a contest at all, and the reach cannot tell those apart while this can: the count is taken against the weights the episode is SCORED by, which sit on the contested points alone, so a decision counted here named ground that is scored — whatever the layer was told the rest of the board was worth.
    if counted:
        log.info("%d of them named ground the board put a priority on, which is %.0f per cent of this arm's decisions",
                 on_priority, 100.0 * on_priority / periods)
    # And whether the arm masses at all. The region block carries how many of this side's squads hold a contract on each region, so a layer can see the allocation; this is the only figure that says whether it does anything with it. The concentrating rule reads 100 per cent by construction and the handwritten ladder pushes the other way, discounting a region by the strength already standing in it.
    if counted_massing:
        log.info("%d of them were about a squad sharing its region with another of this side, which is %.0f per cent",
                 massed, 100.0 * massed / periods)
    # And the same two figures for the seat opposite. Under the script arm the two seats are one ladder on two reflections of one board, so the columns have to agree up to the draw; a gap between them is the board handing the two seats different decisions, which is exactly what a leaning self-play mean is made of and what the score alone cannot separate from a fair board played unevenly.
    if counted_theirs and their_periods and their_errands:
        log.info("the seat opposite took %d decision(s) over %d errand(s) (%.1f per errand) and named priority ground "
                 "in %.0f per cent of them, against this arm's %.1f and %.0f per cent",
                 their_periods, their_errands, their_periods / their_errands,
                 100.0 * their_on_priority / their_periods, periods / errands,
                 100.0 * on_priority / periods if counted else float("nan"))
    _report_departures(sessions, arm)


def _report_departures(sessions: Sequence, arm: Optional[str] = None) -> None:
    """What the frozen tactical layer chose beneath each of the two sides, side by side.

    The two sides run one policy on boards that are one reflection of each other, so a fighter that reads only distances and strengths gives two counts that differ by the draw. A trained one need not: the reflection is congruent in what the layer is handed and not in the ground it is handed it about, so a decision boundary can fall between the two seats and make them play differently on every board of a run. Measured, the self-play mean leans by up to about two hundredths under some sets of trained parameters and by nothing under others and under the handwritten ladder, which is exactly what that would look like from above. This is what it looks like from underneath, and nothing else recorded here can separate it from an even board fought unevenly by chance.

    Reported for every arm rather than only the script one because the frozen layer is the same beneath every arm of a run, so each arm is another reading of the same question.
    """
    ours: Dict[int, int] = {}
    theirs: Dict[int, int] = {}
    for session in sessions:
        for record in session.records:
            # No arm named is every arm, which is the training runner's case: it has one arm and calls its signal line
            # without naming it. Compared against None instead, the filter matched nothing and this diagnostic — the
            # one that says whether the frozen fighter plays the two seats alike — was silently empty in every
            # training run, which is exactly the run whose fighter nobody has looked at yet.
            if arm is not None and record.arm != arm:
                continue
            for kind, count in (record.statistics.get("our_departures") or {}).items():
                ours[int(kind)] = ours.get(int(kind), 0) + int(count)
            for kind, count in (record.statistics.get("their_departures") or {}).items():
                theirs[int(kind)] = theirs.get(int(kind), 0) + int(count)
    total, other = sum(ours.values()), sum(theirs.values())
    if not total or not other:
        return
    kinds = sorted(set(ours) | set(theirs))
    shares = ", ".join("%s %.1f%%/%.1f%%" % (Deviation(kind).name.lower(),
                                             100.0 * ours.get(kind, 0) / total,
                                             100.0 * theirs.get(kind, 0) / other)
                       for kind in kinds)
    # The largest gap between the two seats on any one departure, which is the single number that says whether they are playing the same game.
    widest = max(kinds, key=lambda k: abs(ours.get(k, 0) / total - theirs.get(k, 0) / other))
    gap = abs(ours.get(widest, 0) / total - theirs.get(widest, 0) / other)
    log.info("the layer beneath chose, this side against the other: %s", shares)
    log.info("the two seats differ most on %s, by %.1f percentage point(s) of %d and %d departure(s)",
             Deviation(widest).name.lower(), 100.0 * gap, total, other)


def measure(arguments) -> Dict[str, Summary]:
    """Runs every asked arm over the asked instances and episodes and returns each one's pooled side score.

    Several arms in one run is the paired design and the default way to use this: the arms alternate inside each instance and `_arena_seed` holds the board still until all of them have played it, so every arm meets every board. `--episodes` is per arm, as it is everywhere else in this project, so four arms at ten episodes is forty episodes an instance. The plumbing a human runs against live game instances; the pooling and the report are the same arithmetic the game-free tests exercise on synthetic captures.
    """
    episode = EpisodeSettings(
        map=arguments.map, opponents=arguments.opponents, difficulty=arguments.difficulty,
        credits=arguments.credits, fog=0, seed=arguments.seed, max_seconds=arguments.max_seconds,
        # An arena episode starts with nothing on the board: there is no command that removes a unit, so the only clean board to construct on is one nothing was ever put on.
        starting_units=0, arena=True,
    )
    # The arms are named and their files checked first of all, and the tactical layer beneath both sides read next, so that every refusal either can make is made while the run has started nothing: no server, no inference thread and no game connected. Only then is anything loaded.
    arms = arms_of(arguments)
    frozen = frozen_tactics(arguments.tactics, arguments.device)
    arms = load_arms(arms, arguments.device)

    count = len(arms)
    # A run made under trained tactical parameters writes to a file of its own, because it is not the same instrument as a run made under the handwritten layer and the two must not land in one journal: a journal is opened for appending, and a board that appears twice in it cannot be told apart afterwards, so every repeated board would be dropped from every comparison. Named after the file the parameters came from, as the duel names its arms, which is a nickname rather than the identity — the identity is the digest each episode carries.
    under = "-under-" + os.path.splitext(os.path.basename(arguments.tactics))[0] if arguments.tactics else ""
    path = arguments.record or default_path("ops-" + "-".join(arm.label for arm in arms) + under)
    settings = ServerSettings(
        host=arguments.host, port=arguments.port, instances=arguments.instances,
        episodes=arguments.episodes,
        arms=[("ops-" + arm.label, _arm(arguments, arm, count, frozen)) for arm in arms],
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        episode=episode,
        journal=Journal(path),
    )
    server = Server(settings)
    # The seed is in the opening line because the boards are the seed: a measuring run made at the seed a policy was
    # trained on is a run replaying the boards it was fitted to, and the two runners' defaults are the same number.
    log.info("measuring the operations arena %s arm(s) over %d board(s) each on %d instance(s), horizon %ds, seed %d",
             ", ".join(arm.label for arm in arms), arguments.episodes, arguments.instances, arguments.horizon,
             arguments.seed)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        sessions = server.sessions
    finally:
        for arm in arms:
            if arm.batcher is not None:
                arm.batcher.stop()
        frozen.stop()
        if settings.journal is not None:
            settings.journal.close()

    summaries: Dict[str, Summary] = {}
    for arm in arms:
        log.info("---- %s ----", arm.label)
        summaries[arm.label] = pool(sessions, "ops-" + arm.label)
        report(summaries[arm.label], arm.label)
        # The same boards read the other way. An arm that ends the horizon on the discs and an arm that held them the whole way there are the same figure above and differ here, which is the one question a run of this arena could not be asked before.
        report(pool(sessions, "ops-" + arm.label, "side_tenure"), arm.label, "side_tenure")
        diagnose(sessions, "ops-" + arm.label, arguments.radius)
    if count > 1:
        # Every arm met every board, so the run is its own paired comparison and there is no reason to make anyone assemble it by hand from the journal afterwards. Read back off the file that was just written rather than off the sessions, so that what is reported is what was recorded. Imported here rather than at the top because the comparison reads this module for the stride that names a board, and the two would otherwise import each other.
        from .ops_compare import compare

        # Every pair rather than neighbouring ones: the arms are not on a line, and which two of them the run was really asked about is not something the order they were typed in says.
        for index, first in enumerate(arms):
            for second in arms[index + 1:]:
                log.info("---- %s against %s, board by board ----", first.label, second.label)
                compare(path, path, first_arm="ops-" + first.label, second_arm="ops-" + second.label)
                log.info("---- %s against %s, board by board, held over the episode ----", first.label, second.label)
                compare(path, path, first_arm="ops-" + first.label, second_arm="ops-" + second.label,
                        reading="side_tenure")
    if frozen.batcher is not None:
        # Under a frozen tactical layer the fighting is the dominant inference load of the run and nothing else reports it: every squad of both sides asks it for a departure every tactical frame, against one operational request a squad a period. How that batches is the first thing to read a run's speed against, so it is said in the same words the engagement arena's runner says it in.
        log.info("the frozen tactical layer's batched inference averaged %.1f per call over %d call(s)",
                 frozen.batcher.batch_size, frozen.batcher.calls)
    return summaries


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
    parser.add_argument("--map", default="Hills",
                        help="a symmetric, compact map: the point-reflected layout only stays even where the map's terrain is even under the reflection, and Lake's is not (it self-play-leans about +0.06 while Hills holds nought)")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--credits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--horizon", type=int, default=HORIZON_MS // 1000,
                        help="game seconds the two chains run before the board is scored")
    parser.add_argument("--squads", type=int, default=4, help="assorted-doctrine squads staged per side")
    parser.add_argument("--pairs", type=int, default=2, help="contested offset pairs, so twice this many scored regions")
    parser.add_argument("--our", action="append", default=None,
                        help="our side's operational layer, repeatable: the script chain (the self-play zero), a pinned deployment that makes no choice, the script with its crowding discount removed, which on this arena decides the same as the script, the script with its candidate set narrowed, the script with the strategic layer's own weights re-scaled (written wanted:WEIGHT), an arm that sends every squad at the single region the strategic layer wants most, or a learnt network read from --load. A learnt arm may instead be written learnt:PATH and carry its own parameters, so that one run holds two generations of a training line and pairs them board by board. Give it more than once and every arm plays every board and the run reports the paired difference between every pair of arms itself; --episodes is per arm")
    parser.add_argument("--load", default=None, help="parameters for a learnt arm written as the bare word, read greedily")
    parser.add_argument("--tactics", default=None,
                        help="parameters for a trained tactical layer to be put, frozen, under BOTH sides of the board, read at its likeliest action and recording nothing; left out, both sides fight the handwritten layer, which is what every measurement so far was made under; given, the run is a different instrument and its journal says so, so it pairs only with runs made under the same parameters")
    parser.add_argument("--device", default=None)
    parser.add_argument("--radius", type=float, default=CATCHMENT_RADIUS, help="world units a contest's catchment disc reaches; sized to the engagement standoff so an assaulting squad registers")
    parser.add_argument("--garrison", type=float, default=GARRISON_SCALE,
                        help="credits a contested region's defender is drawn out of, which is what decides whether taking ground pays at all; a defender too strong for the squads a side can bring makes holding what one already owns the best play")
    parser.add_argument("--standing", choices=STANDINGS, default=STANDING_MIRROR,
                        help="what is done about the free base a starting position is given, which stands on this side alone: mirror places its reflection for the other side so the opening board is congruent, leave runs the board one base short as this arena did before the reflection was placed")
    parser.add_argument("--max-seconds", type=int, default=0,
                        help="game time an episode is cut off at, defaulting to the horizon plus the settle and spawn waits and a margin")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None, help="where episodes are written")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    # An appending option cannot carry a default without the default staying in front of whatever is given, so the one-arm case is filled in here instead.
    arguments.our = arguments.our or ["script"]
    if len(set(arguments.our)) != len(arguments.our):
        parser.error("an arm given twice would play each board twice under one name and pair with itself")
    if arguments.max_seconds <= 0:
        # The board is scored at the horizon, which the episode has to outlast: the settle and spawn waits come first, and a margin leaves room for the scoring frame to arrive.
        arguments.max_seconds = arguments.horizon + 60
    measure(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
