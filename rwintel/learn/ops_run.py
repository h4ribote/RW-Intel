"""Runs the constructed operations arena, the handwritten operational chain on both sides, and reports the pooled side score.

    python -m rwintel.learn.ops_run --instances 4 --episodes 50 --map Lake

Start this first and then the game instances, as with every other runner here. What it measures is the self-play zero: with the script `Operations` on both sides of the mirror board, the side score of the two sides is exact negatives every episode, so a run of many fresh paired episodes must pool the reported side score to nought. A nonzero mean is a board lean the mirror-symmetric draw was supposed to have removed — the operational analogue of the headquarters-in-a-squad bias the fight baseline once carried — and it is the only instrument that can see the leans the within-episode sign check cannot: all-enemy garrisons, the free base polluting the region block, a non-congruent reflected layout, and empty regions reading a half under asymmetric reach. Every one of those is a break in exchange symmetry, and only this mean sees it.

This is the gate the arena must pass before any operational policy measured on it is trusted, exactly as the engagement arena gates on its own script-against-itself baseline. It trains nothing and keeps no trajectories: the script layer is handed no rollout, so with nowhere to record a decision it records none.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from typing import List, Optional

from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from ..eval.journal import Journal, default_path
from ..eval.sampling import Summary
from .deciders import PinnedRegion
from .layers import LearntOperations
from .ops_arena import CATCHMENT_RADIUS, HORIZON_MS, OpsArena

log = logging.getLogger(__name__)

#: How far apart two instances' arena draws are set, matching the engagement arena's stride. It only has to exceed the episodes one instance will ever run, and it is prime so two runs at neighbouring seeds do not lay one instance's stream on top of another's.
INSTANCE_STRIDE = 100003


def _arena_seed(base_seed: int, session) -> int:
    """The seed an arena episode draws its board from, advancing with the episode as well as with the instance, so every episode draws a fresh board rather than replaying the first one, and the sample size a pooled mean rests on is the episodes fought rather than the instances. One arm here, so the episode folds in directly."""
    return base_seed + INSTANCE_STRIDE * session.instance + len(session.records)


def _arm(arguments):
    """One `OpsArena` per episode, seeded so each episode is a fresh board and two runs at the same seed draw the same boards.

    Our side is the script chain by default (the self-play zero) or a pinned deployment when `--our pin` is given: a layer that sends every squad to the lowest-numbered legal region and task, making no operational choice at all. Running the two at the same seed and subtracting the pinned run from the self-play run cancels the enemy and the board lean and leaves how much the script's careful deployment beat making no choice — the resolution the arena exists to produce. The enemy is always the script, so the pinned run is our-pin against their-script on the very board the self-play run drew.
    """
    if arguments.our == "pin":
        operations = lambda session, catalogue: LearntOperations(session, catalogue, PinnedRegion(), None, -1)
    else:
        operations = None

    def build(session) -> OpsArena:
        return OpsArena(session, operations=operations, seed=_arena_seed(arguments.seed, session),
                        horizon_ms=arguments.horizon * 1000, our_squads=arguments.squads,
                        catchment_radius=arguments.radius, contest_pairs=arguments.pairs)
    return build


def pool(sessions) -> Summary:
    """Every scored episode's side score as one count, one mean and one spread. An episode cut off before its horizon carries no score and is skipped, so a run whose match length did not clear the horizon pools nothing rather than pooling a nought that was never measured."""
    scores: List[float] = [float(record.statistics.get("side_score", 0.0))
                           for session in sessions for record in session.records
                           if record.statistics.get("scored")]
    return Summary.of(scores)


def report(summary: Summary) -> None:
    """The self-play zero, as a mean and the two standard errors it has to sit inside. A mean inside the interval is a board with no lean the mirror draw did not remove; a mean outside it is a lean to be found and fixed before the arena is trusted."""
    if summary.n == 0:
        log.warning("no episode reached its horizon, so there is no side score to pool: is the match length longer than the horizon plus the settle and spawn waits?")
        return
    interval = 2.0 * summary.sd / math.sqrt(summary.n) if summary.n > 1 else 0.0
    log.info("pooled side score over %d scored episode(s): %+.4f, 2 standard errors %.4f, interval %+.4f to %+.4f",
             summary.n, summary.mean, interval, summary.mean - interval, summary.mean + interval)
    if summary.n < 2:
        log.info("one episode has no spread, so it says nothing about whether the arena is even; run several hundred")
    elif interval > 0.0 and abs(summary.mean) > interval:
        log.warning("the pooled side score is outside two standard errors of nought, so the board leans under this draw and a policy measured on it would be reading the lean: find and remove it before trusting the arena")
    else:
        log.info("the pooled side score holds nought within two standard errors, which is the self-play zero the arena has to pass before it is trusted")


def self_play(arguments) -> Summary:
    """Runs the self-play arm over the asked instances and episodes and returns the pooled side score. The plumbing a human runs against live game instances; the pooling and the report are the same arithmetic the game-free tests exercise on synthetic captures."""
    episode = EpisodeSettings(
        map=arguments.map, opponents=arguments.opponents, difficulty=arguments.difficulty,
        credits=arguments.credits, fog=0, seed=arguments.seed, max_seconds=arguments.max_seconds,
        # An arena episode starts with nothing on the board: there is no command that removes a unit, so the only clean board to construct on is one nothing was ever put on.
        starting_units=0, arena=True,
    )
    settings = ServerSettings(
        host=arguments.host, port=arguments.port, instances=arguments.instances,
        episodes=arguments.episodes, arms=[("ops-" + arguments.our, _arm(arguments))],
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        episode=episode,
        journal=Journal(arguments.record or default_path("ops-self-play")),
    )
    server = Server(settings)
    log.info("measuring the operations arena self-play zero over %d episode(s) each on %d instance(s), horizon %ds",
             arguments.episodes, arguments.instances, arguments.horizon)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        sessions = server.sessions
    finally:
        if settings.journal is not None:
            settings.journal.close()
    summary = pool(sessions)
    report(summary)
    return summary


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
    parser.add_argument("--our", choices=("script", "pin"), default="script",
                        help="our side's operational layer: the script chain (the self-play zero) or a pinned deployment that makes no choice; run both at one seed and subtract to read the resolution")
    parser.add_argument("--radius", type=float, default=CATCHMENT_RADIUS, help="world units a contest's catchment disc reaches; sized to the engagement standoff so an assaulting squad registers")
    parser.add_argument("--max-seconds", type=int, default=0,
                        help="game time an episode is cut off at, defaulting to the horizon plus the settle and spawn waits and a margin")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None, help="where episodes are written")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    if arguments.max_seconds <= 0:
        # The board is scored at the horizon, which the episode has to outlast: the settle and spawn waits come first, and a margin leaves room for the scoring frame to arrive.
        arguments.max_seconds = arguments.horizon + 60
    self_play(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
