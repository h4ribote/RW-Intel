"""Runs the constructed operations arena over one or more operational arms and reports what each came to and how they differ board by board.

    python -m rwintel.learn.ops_run --instances 4 --episodes 50 --map Hills
    python -m rwintel.learn.ops_run --instances 8 --episodes 10 --our script --our pin --our massed --our learnt --load local/ops-arena.pt

Start this first and then the game instances, as with every other runner here.

With the script chain alone it measures the self-play zero: the script `Operations` on both sides of the mirror board makes the two sides' side scores exact negatives every episode, so a run of many fresh boards must pool the reported side score to nought. A nonzero mean is a board lean the mirror-symmetric draw was supposed to have removed — the operational analogue of the headquarters-in-a-squad bias the fight baseline once carried — and it is the only instrument that can see the leans the within-episode sign check cannot: all-enemy garrisons, the free base polluting the region block, a non-congruent reflected layout, and empty regions reading a half under asymmetric reach. Every one of those is a break in exchange symmetry, and only this mean sees it. This is the gate the arena must pass before any operational policy measured on it is trusted, exactly as the engagement arena gates on its own script-against-itself baseline.

Given several arms it runs all of them on the same boards. The arms alternate inside each instance and the board is held still until every one of them has played it, so each board is one paired observation across the arms and the run reports every pair's difference itself. That is the honest way to compare two operational policies here, because a board's draw moves the side score by more than the arms differ: one arm's episodes scatter by about 0.11 while the differences being looked for are around 0.04. Running the arms separately and subtracting the means pays for that scatter twice and also has to assume two runs, made at different moments on a machine doing different things, were otherwise alike. Running them together assumes nothing of the sort.

It trains nothing and keeps no trajectories: no layer here is handed a rollout, so with nowhere to record a decision none is recorded.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import Dict, List, Optional

from ..control.policy.operations import Concentrated, Operations
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from ..eval.journal import Journal, default_path
from ..eval.sampling import Summary
from .__main__ import _device, _load
from .deciders import NetworkOperations, PinnedRegion, operational_batcher
from .layers import LearntOperations
from .net import OperationalNet
from .ops_arena import CATCHMENT_RADIUS, GARRISON_SCALE, HORIZON_MS, OpsArena

log = logging.getLogger(__name__)

#: How far apart two instances' arena draws are set, matching the engagement arena's stride. It only has to exceed the episodes one instance will ever run, and it is prime so two runs at neighbouring seeds do not lay one instance's stream on top of another's.
INSTANCE_STRIDE = 100003


def _arena_seed(base_seed: int, session, arms: int = 1) -> int:
    """The seed an arena episode draws its board from, advancing with the board as well as with the instance, so every board is fresh rather than the first one replayed, and the sample size a pooled mean rests on is the boards fought rather than the instances.

    With several arms in one run the board must advance once every arm has played it, not once per episode: a session takes its arm as `records % arms`, so `records // arms` is which board it is on and every arm meets that board exactly once. That is what makes a multi-arm run a paired one — the same construction under each arm, in the same instances, at the same moment of the same machine — and it is the difference between a comparison that has to trust two separate runs to have been alike and one that does not have to.
    """
    return base_seed + INSTANCE_STRIDE * session.instance + len(session.records) // max(1, arms)


def _arm(arguments, our: str, arms: int = 1, net=None, device=None, batcher=None):
    """One `OpsArena` per episode of one arm, seeded so that every arm of the run meets the same boards.

    `script` on our side is the self-play zero: the same chain on both sides of the mirror, whose pooled score must be nought. `pin` is a layer that sends every squad to the lowest-numbered legal region and task, making no operational choice at all; subtracting it from the script arm board by board cancels the enemy and the board lean and leaves how much the script's careful deployment beat making no choice, which is the resolution the arena exists to produce. The enemy is always the script, so every arm is measured against one fixed opponent.

    What the pin is not is a concentration arm. Its region is the lowest live id on the map and the contests are drawn about the board centre, so its squads march off the scored ground: measured on Hills at a catchment of four hundred, no squad of it was inside any catchment in fifty of sixty-eight scored episodes on one draw and in thirty-five of fifty-one on another, and the nearest ended a median of seven hundred and seventy and seven hundred and eighty-three units out. It beat the script's careful deployment there, but not by overwhelming anything — by leaving discs to empty out, and an empty disc reads a half rather than a loss. That is a floor for abandoning the board, and reading it as evidence that concentration wins was wrong. `diagnose` reports the reach of every arm for this reason.

    `massed` was meant to be the concentration arm the reading needed, and measurement says it is not one. It is the script ladder with the discount a region takes for the strength we already have standing in it removed — and on this arena that discount is already nought, because the squads are at the staging point and not in the contested regions when the choice is made. Measured: the two arms score identically on 43 of 51 boards at the default garrison and on 37 of 38 at half of it, and the engine does not reproduce, so identical scores mean identical decisions. It is therefore an ablation of one term of the ladder, worth keeping as that, and it is no evidence about concentration. An arm that actually concentrates has to replace the region choice rather than remove a term from it, which is what `concentrate` does. Against the script it says what the spreading rule costs; against the pin it says whether massing on the right ground beats leaving it.

    `concentrate` is the arm that does what the massed arm was supposed to do. It keeps the doctrine's own choice of task and overrides only the region, sending every squad at the single region the strategic layer wants most. On this arena the priorities sit on the contested regions alone, so that is every squad at one contest — concentration in the plain sense, made by replacing the choice rather than by removing a term from it.

    `learnt` is a trained network read from a file.
    """
    if our == "pin":
        operations = lambda session, catalogue: LearntOperations(session, catalogue, PinnedRegion(), None, -1)
    elif our == "massed":
        operations = lambda session, catalogue: Operations(session, catalogue, crowding=0.0)
    elif our == "concentrate":
        operations = lambda session, catalogue: Concentrated(session, catalogue)
    elif our == "learnt":
        # The trained layer read greedily — its most probable region and task, not a draw — since this measures the policy rather than trains it, and with no rollout it records nothing.
        operations = lambda session, catalogue: LearntOperations(
            session, catalogue, NetworkOperations(net, device, batcher, greedy=True), None, -1)
    else:
        operations = None

    def build(session) -> OpsArena:
        return OpsArena(session, operations=operations, seed=_arena_seed(arguments.seed, session, arms),
                        horizon_ms=arguments.horizon * 1000, our_squads=arguments.squads,
                        catchment_radius=arguments.radius, contest_pairs=arguments.pairs,
                        garrison_scale=arguments.garrison)
    return build


def pool(sessions, arm: Optional[str] = None) -> Summary:
    """Every scored episode's side score as one count, one mean and one spread, over one arm of the run or over all of them. An episode cut off before its horizon carries no score and is skipped, so a run whose match length did not clear the horizon pools nothing rather than pooling a nought that was never measured."""
    scores: List[float] = [float(record.statistics.get("side_score", 0.0))
                           for session in sessions for record in session.records
                           if record.statistics.get("scored") and (arm is None or record.arm == arm)]
    return Summary.of(scores)


def report(summary: Summary, arm: str = "script") -> None:
    """The run's pooled side score, as a mean and the two standard errors it has to sit inside.

    What a nonzero mean means depends on which arm produced it, and saying the wrong one of these is worse than saying nothing. For the script arm the two sides are the same chain, so the mean is the self-play zero: inside the interval is a board with no lean the mirror draw did not remove, and outside it is a lean to be found and fixed before the arena is trusted. For every other arm our side is deliberately not the enemy's chain, so a mean outside the interval is the arm beating the script — which is the measurement, not a fault — and warning about a lean there would be reporting the instrument working as if it were broken. The lean is read once, on the script arm, and every other arm is then read against it and against the other arms board by board with `ops_compare`.
    """
    if summary.n == 0:
        log.warning("no episode reached its horizon, so there is no side score to pool: is the match length longer than the horizon plus the settle and spawn waits?")
        return
    interval = 2.0 * summary.sd / math.sqrt(summary.n) if summary.n > 1 else 0.0
    log.info("pooled side score over %d scored episode(s): %+.4f, 2 standard errors %.4f, interval %+.4f to %+.4f",
             summary.n, summary.mean, interval, summary.mean - interval, summary.mean + interval)
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
    else:
        log.info("the pooled side score holds nought within two standard errors, which is the self-play zero the arena has to pass before it is trusted")


def diagnose(sessions, arm: str, radius: float) -> None:
    """Whether this arm's own squads ever reached the ground the episode was scored on.

    An arm can score well without its squads ever entering a catchment, and the figure alone cannot tell that apart from an arm that fought for the ground and won it — the two look identical in the pooled mean. The distinction is the whole difference between measuring a deployment and measuring an abstention, and the arena already writes down what settles it: how far the nearest surviving squad member ended from a contest, and how many of them ended inside one.

    This is the check that was in hand and not pointed at the pinned arm. That arm sends every squad to the lowest-numbered legal region, which is a fixed region id with nothing to do with where the contests were drawn, so its squads finished outside every scored disc in most episodes and what looked like concentration beating a spread was an arm that had left the scored board. An arm whose median reach is outside the catchment is not deploying onto the contests, whatever its score says, and the run says so rather than leaving it to be noticed.
    """
    reaches = sorted(float(record.statistics.get("our_reach", -1.0))
                     for session in sessions for record in session.records
                     if record.arm == arm and record.statistics.get("scored")
                     and float(record.statistics.get("our_reach", -1.0)) >= 0.0)
    absent = sum(1 for session in sessions for record in session.records
                 if record.arm == arm and record.statistics.get("scored")
                 and not record.statistics.get("our_in_catchment"))
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
    """How far this arm's one payment could reach the decisions it is supposed to teach.

    The arena pays a squad once, at the horizon, and pays it to the errand the squad was on when the board was scored. So what a run has to be able to say is how long an errand was: an episode in which a squad held one contract from the staging point to the horizon is an episode in which the payment reaches every decision taken about it, and an episode in which the contract was re-drawn every period is one in which the payment reaches the last decision and no other, however many hundred were taken. Those two episodes report the same score, the same shares and the same disc tallies, and nothing else here tells them apart.

    The three figures are the decisions the squads were given, the errands those decisions were divided into, and how many payments actually landed on a decision. The last is nought for every arm of this runner and that is not a fault: no arm here is handed a rollout, so no decision is recorded and there is nothing for a payment to land on. It is reported all the same, because it is the figure a training run has to be read by and a measuring run is where the ratio it has to be compared against is taken.
    """
    periods = errands = terminals = staged = 0
    for session in sessions:
        for record in session.records:
            if (arm is not None and record.arm != arm) or not record.statistics.get("scored"):
                continue
            periods += int(record.statistics.get("periods", 0))
            errands += int(record.statistics.get("errands", 0))
            terminals += int(record.statistics.get("terminals", 0))
            staged += int(record.statistics.get("squads", 0))
    if not periods or not errands:
        return
    # Each squad's last errand is the only one the horizon pays, so the share of the decisions a payment can reach is the share of the errands that are somebody's last one — the squads staged, against every errand they were given.
    log.info("this arm's squads took %d operational decision(s) over %d errand(s), so an errand ran %.1f decision(s) "
             "and the payment made at the horizon reaches about %.1f%% of them; %d payment(s) landed on a decision",
             periods, errands, periods / errands, 100.0 * min(staged, errands) / errands, terminals)


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
    # A learnt arm reads one network off a file and shares it across every instance, built once here rather than per session for the same reason the training runner does: the network is what is being measured, and one copy batched across the instances is the whole point of batching the inference. The script, pin and massed arms need none of this.
    net = device = batcher = None
    if "learnt" in arguments.our:
        # Refused rather than loaded blind, for the same reason the duel refuses it: a missing or mistyped path leaves a freshly initialised network in place, and the run then measures a random policy and journals it under the trained one's name. Nothing downstream can tell those apart afterwards, and the figure looks like an ordinary measurement.
        if not arguments.load:
            raise SystemExit("the learnt arm has no parameters to measure: give --load")
        if not os.path.exists(arguments.load):
            raise SystemExit("no parameters at %s, so there is nothing for the learnt arm to measure" % arguments.load)
        device = _device(arguments.device)
        net = OperationalNet().to(device)
        _load(net, arguments.load, device)
        batcher = operational_batcher(net, device=device, greedy=True)

    count = len(arguments.our)
    path = arguments.record or default_path("ops-" + "-".join(arguments.our))
    settings = ServerSettings(
        host=arguments.host, port=arguments.port, instances=arguments.instances,
        episodes=arguments.episodes,
        arms=[("ops-" + our, _arm(arguments, our, count, net, device, batcher)) for our in arguments.our],
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        episode=episode,
        journal=Journal(path),
    )
    server = Server(settings)
    log.info("measuring the operations arena %s arm(s) over %d board(s) each on %d instance(s), horizon %ds",
             ", ".join(arguments.our), arguments.episodes, arguments.instances, arguments.horizon)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        sessions = server.sessions
    finally:
        if batcher is not None:
            batcher.stop()
        if settings.journal is not None:
            settings.journal.close()

    summaries: Dict[str, Summary] = {}
    for our in arguments.our:
        log.info("---- %s ----", our)
        summaries[our] = pool(sessions, "ops-" + our)
        report(summaries[our], our)
        diagnose(sessions, "ops-" + our, arguments.radius)
    if count > 1:
        # Every arm met every board, so the run is its own paired comparison and there is no reason to make anyone assemble it by hand from the journal afterwards. Read back off the file that was just written rather than off the sessions, so that what is reported is what was recorded. Imported here rather than at the top because the comparison reads this module for the stride that names a board, and the two would otherwise import each other.
        from .ops_compare import compare

        # Every pair rather than neighbouring ones: the arms are not on a line, and which two of them the run was really asked about is not something the order they were typed in says.
        for index, first in enumerate(arguments.our):
            for second in arguments.our[index + 1:]:
                log.info("---- %s against %s, board by board ----", first, second)
                compare(path, path, first_arm="ops-" + first, second_arm="ops-" + second)
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
    parser.add_argument("--our", choices=("script", "pin", "massed", "concentrate", "learnt"), action="append", default=None,
                        help="our side's operational layer, repeatable: the script chain (the self-play zero), a pinned deployment that makes no choice, the script with its crowding discount removed, which on this arena decides the same as the script, an arm that sends every squad at the single region the strategic layer wants most, or a learnt network read from --load. Give it more than once and every arm plays every board and the run reports the paired difference between every pair of arms itself; --episodes is per arm")
    parser.add_argument("--load", default=None, help="parameters for the learnt arm, read greedily")
    parser.add_argument("--device", default=None)
    parser.add_argument("--radius", type=float, default=CATCHMENT_RADIUS, help="world units a contest's catchment disc reaches; sized to the engagement standoff so an assaulting squad registers")
    parser.add_argument("--garrison", type=float, default=GARRISON_SCALE,
                        help="credits a contested region's defender is drawn out of, which is what decides whether taking ground pays at all; a defender too strong for the squads a side can bring makes holding what one already owns the best play")
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
