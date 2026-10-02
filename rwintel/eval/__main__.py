"""Runs a comparison, or reports one that has already been run.

    python -m rwintel.eval --instances 4 --episodes 3 --arm script --arm arm --map Lake --max-seconds 300
    python -m rwintel.eval --instances 4 --episodes 3 --arm script --intrude
    python -m rwintel.eval --from local/episodes/script-arm-20260101-120000.jsonl
    python -m rwintel.eval --from run-a.jsonl --fit-out local/weights-a.json

Start this first and then the game instances, as with the plain runner. Episodes go to a new file under local/episodes unless `--record` names one, and the log to standard error and local/logs/eval. The arms alternate within each instance rather than one being run to completion before the other, because two arms measured in sequence differ by whatever else changed about the machine in between.

What it prints is each arm's score, its scatter and the seeds it was taken under, then every arm against the reference arm: the difference, its 95 per cent interval, its p-value and whether it survives the correction for there being several comparisons. Episodes played under different settings, or with and without an intruder, are reported apart and never compared. Last comes how well the board predicts the winners of the decided matches, which is what the weights of the score are fitted on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .. import paths
from ..control.intruder import interference
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings
from ..data import AssetPaths
from . import arms as arm_names
from .journal import Journal, default_path, read
from .sampling import Difference, SIGNIFICANCE_LEVEL, Summary, episodes_for, episodes_for_win_rate, holm
from .scoring import (DEFAULT_LEAD_SECONDS, OPENING_WEIGHTS, Fit, Weights, agreement, components, decided,
                      default_weights, fit_weights, graded, load_weights, save_weights, score)

#: Differences worth quoting a sample size for. The design's own table, which is what makes a run's report comparable with it.
REPORTED_DIFFERENCES = (0.05, 0.10, 0.20)

#: How long before the end of a decided match the boards the report fits on are taken, beside the lead asked for.
FIT_LEADS = (30.0, 60.0, 120.0, 240.0)


@dataclass
class Played:
    """One finished episode, however it reached us: from a session that just ran it or from a journal written earlier."""

    arm: str
    winner: int
    team: int
    timeout: bool
    standing: Sequence[dict]
    seconds: int
    statistics: dict
    #: What anyone outside the chain did to this episode, and whether an intruder was attached at all. A score taken under interference is not the same quantity as one taken without it, and two of them are never compared.
    interference: dict = field(default_factory=dict)
    history: Sequence[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    synchronisation: dict = field(default_factory=dict)
    peer_left: bool = False
    instance: int = -1
    episode: int = 0
    #: Matches the game process had played before this one, 0 for an instance's first match, or -1 from a journal that did not record it.
    order: int = -1
    #: The map the episode was played on: the engine's file name, or the map asked for in a journal that did not record it.
    map: str = ""
    #: The decision-latency meter's summary of the episode (`EpisodeRecord.latency`), empty from a journal that did not record it.
    latency: dict = field(default_factory=dict)

    @classmethod
    def of(cls, record) -> "Played":
        return cls(arm=record.arm, winner=record.winner, team=record.team, timeout=record.timeout,
                   standing=record.standing, seconds=record.seconds, statistics=record.statistics,
                   interference=record.interference, history=record.history, settings=record.settings,
                   synchronisation=record.synchronisation, peer_left=record.peer_left,
                   instance=record.instance, episode=record.episode, order=record.order,
                   map=record.map or str(record.settings.get("map", "")), latency=record.latency)

    @classmethod
    def from_dict(cls, entry: dict) -> "Played":
        return cls(arm=entry.get("arm", ""), winner=int(entry.get("winner", -1)),
                   team=int(entry.get("team", -1)), timeout=bool(entry.get("timeout", False)),
                   standing=entry.get("standing", []), seconds=int(entry.get("seconds", 0)),
                   statistics=entry.get("statistics", {}),
                   interference=entry.get("interference", {}),
                   history=entry.get("history", []), settings=entry.get("settings", {}),
                   synchronisation=entry.get("synchronisation", {}),
                   peer_left=bool(entry.get("peer_left", False)),
                   instance=int(entry.get("instance", -1)), episode=int(entry.get("episode", 0)),
                   order=int(entry.get("order", -1)),
                   map=str(entry.get("map") or entry.get("settings", {}).get("map", "")),
                   latency=entry.get("latency", {}))


def latency_by_arm(played: Sequence[Played]) -> Dict[str, dict]:
    """Per arm, over the episodes that carry the latency meter's summary: the largest of the episodes' tactical 95th percentile lags in milliseconds and in periods, and the answers missed in all."""
    out: Dict[str, dict] = {}
    for episode in played:
        tactical = (episode.latency or {}).get("tactics")
        if not tactical:
            continue
        row = out.setdefault(episode.arm, {"episodes": 0, "max_p95_ms": 0.0, "max_p95_periods": 0.0, "missed": 0})
        row["episodes"] += 1
        row["max_p95_ms"] = max(row["max_p95_ms"], float(tactical.get("p95_ms", 0.0)))
        row["max_p95_periods"] = max(row["max_p95_periods"], float(tactical.get("p95_periods", 0.0)))
        row["missed"] += int(tactical.get("missed", 0))
    return out


def openings(played: Sequence[Played], weights: Weights) -> List[Tuple[str, str, str, int, float]]:
    """The mean score by arm, map and place in the instance's run of matches, as (arm, map, place, n, mean) rows in the order they first appear.

    The place is `first` for an instance's first match, `later` for the rest, and `unknown` for an episode from a journal that did not record it.
    """
    cells: Dict[Tuple[str, str, str], List[float]] = {}
    for episode in played:
        place = "unknown" if episode.order < 0 else ("first" if episode.order == 0 else "later")
        name = episode.map.rsplit("/", 1)[-1] or "?"
        cells.setdefault((episode.arm or "script", name, place), []).append(score(episode, weights))
    return [(arm, name, place, len(scores), sum(scores) / len(scores)) for (arm, name, place), scores in cells.items()]


#: What separates two groups of episodes that are not the same quantity: the settings less the seed, and whether an intruder was attached.
Condition = Tuple[str, bool]


def condition(episode: Played, pool_maps: bool = False) -> Condition:
    """The settings an episode was played under, less the seed, and whether an intruder was attached. Episodes that differ in either are not comparable; episodes that differ only in the seed are the same measurement repeated. A journal written before the record said whether an intruder was attached shows one only where it touched something.

    With `pool_maps` the map is left out as well, which is the condition of a pool over maps: arms that played the same maps in step are compared on all of them together.
    """
    ignored = ("seed", "map") if pool_maps else ("seed",)
    settings = {key: value for key, value in (episode.settings or {}).items() if key not in ignored}
    return json.dumps(settings, sort_keys=True), bool(episode.interference)


def out_of_step(episode: Played) -> bool:
    """Whether a match shared between processes drifted apart, after which its numbers describe nothing."""
    sync = episode.synchronisation or {}
    if not sync.get("networked"):
        return False
    return any(peer.get("desynced") or peer.get("broken") for peer in sync.get("peers", []))


def groups(played: Sequence[Played], pool_maps: bool = False) -> Dict[Condition, Dict[str, List[Played]]]:
    """The episodes by condition and, within one, by arm, both in the order they first appear."""
    grouped: Dict[Condition, Dict[str, List[Played]]] = {}
    for episode in played:
        grouped.setdefault(condition(episode, pool_maps), {}).setdefault(episode.arm or "script", []).append(episode)
    return grouped


def pooled_over_maps(played: Sequence[Played]) -> Dict[Condition, Dict[str, List[Played]]]:
    """The pools over maps worth reporting: conditions that differ only in the map, where more than one map was played."""
    return {key: by_arm for key, by_arm in groups(played, pool_maps=True).items()
            if len({e.settings.get("map") for episodes in by_arm.values() for e in episodes}) > 1}


def describe_conditions(conditions: Sequence[Condition]) -> Dict[Condition, str]:
    """A short name for each condition, naming only what differs between them, or nothing when there is only one."""
    if len(conditions) < 2:
        return {key: "" for key in conditions}
    decoded = [(key, json.loads(key[0])) for key in conditions]
    names = sorted({name for _, settings in decoded for name in settings})
    differing = [name for name in names if len({json.dumps(settings.get(name)) for _, settings in decoded}) > 1]
    intruded = len({key[1] for key in conditions}) > 1
    labels = {}
    for key, settings in decoded:
        parts = [f"{name}={settings.get(name)}" for name in differing]
        if intruded:
            parts.append("intruded" if key[1] else "undisturbed")
        labels[key] = "[" + " ".join(parts) + "]"
    return labels


def report(played: Sequence[Played], weights: Weights, reference: Optional[str] = None,
           lead_seconds: float = DEFAULT_LEAD_SECONDS, fit_out: Optional[str] = None) -> Optional[Fit]:
    """Logs the report and returns the fit at `lead_seconds`, writing its weights to `fit_out` when asked."""
    broken = [episode for episode in played if out_of_step(episode)]
    played = [episode for episode in played if not out_of_step(episode)]
    logging.info("%d episode(s), %d decided%s", len(played), sum(1 for e in played if decided(e)),
                 f"; {len(broken)} left out for having fallen out of step" if broken else "")
    logging.info("weights: %s", weights.describe())
    for arm, row in latency_by_arm(played).items():
        logging.info("%s: decision latency over %d episode(s): largest tactical p95 %.0f ms (%.2f periods), %d answer(s) missed",
                     arm, row["episodes"], row["max_p95_ms"], row["max_p95_periods"], row["missed"])

    grouped = groups(played)
    labels = describe_conditions(list(grouped))
    comparisons: List[Tuple[str, str, Difference]] = []
    for key, by_arm in grouped.items():
        _report_condition(labels[key], by_arm, reference, weights, comparisons)
    pooled = pooled_over_maps(played)
    pooled_labels = describe_conditions(list(pooled))
    for key, by_arm in pooled.items():
        maps = sorted({str(e.settings.get("map")) for episodes in by_arm.values() for e in episodes})
        label = f"[maps pooled: {','.join(maps)}]{pooled_labels[key]}"
        _report_condition(label, by_arm, reference, weights, comparisons)

    if comparisons:
        significant = holm([difference.p_value for _, _, difference in comparisons])
        logging.info("---- against the reference, %d comparison(s), Holm-corrected at %.2f ----",
                     len(comparisons), SIGNIFICANCE_LEVEL)
        for (first, second, difference), claimed in zip(comparisons, significant):
            logging.info("%s less %s: %+.3f, 95%% interval %+.3f to %+.3f, p %.4f, %s",
                         first, second, difference.difference, difference.low, difference.high,
                         difference.p_value, "a difference" if claimed else "not established")

    rows = openings(played, weights)
    if rows:
        logging.info("---- by map, the first match of each instance apart from the later ones ----")
        for arm, name, place, count, mean in rows:
            logging.info("%-10s %-40s %-7s n=%-3d score %+.3f", arm, name, place, count, mean)

    return _report_fit(played, weights, lead_seconds, fit_out)


def _report_condition(label: str, by_arm: Dict[str, List[Played]], reference: Optional[str], weights: Weights,
                      comparisons: List[Tuple[str, str, Difference]]) -> None:
    """Reports every arm under one condition and adds each arm's difference from the reference to `comparisons`."""
    if label:
        logging.info("---- %s ----", label)
    summaries: Dict[str, Summary] = {}
    for name, episodes in by_arm.items():
        summaries[name] = _report_arm(name, episodes, weights)
    base = reference if reference in summaries else next(iter(summaries))
    if reference is not None and reference not in summaries:
        logging.info("no arm named %s under this condition; comparing against %s", reference, base)
    for name, summary in summaries.items():
        if name != base:
            comparisons.append((f"{name} {label}".strip(), f"{base} {label}".strip(),
                                Difference.of(summary, summaries[base])))


def _report_arm(name: str, episodes: Sequence[Played], weights: Weights) -> Summary:
    scores = [score(episode, weights) for episode in episodes]
    summary = Summary.of(scores)
    parts = [components(e.standing, e.team) for e in episodes]
    won = sum(1 for e, s in zip(episodes, scores) if decided(e) and s > 0)
    lost = sum(1 for e, s in zip(episodes, scores) if decided(e) and s < 0)
    seeds = sorted({e.settings.get("seed") for e in episodes if e.settings.get("seed") is not None})
    logging.info("%-10s n=%-3d score %+.3f sd %.3f  military %+.3f economy %+.3f exchange %+.3f treasury %+.3f  "
                 "won %d lost %d  seconds %d  seed(s) %s",
                 name, summary.n, summary.mean, summary.sd,
                 _mean(p.military for p in parts), _mean(p.economy for p in parts),
                 _mean(p.exchange for p in parts), _mean(p.treasury for p in parts),
                 won, lost, _mean(e.seconds for e in episodes), ",".join(str(seed) for seed in seeds) or "?")
    for episode, value, part in zip(episodes, scores, parts):
        logging.debug("%-10s   instance %d episode %d: %s at %ds, score %+.3f (military %+.3f economy %+.3f exchange %+.3f treasury %+.3f)",
                      name, episode.instance, episode.episode,
                      ("won" if value > 0 else "lost") if decided(episode) else "cut off", episode.seconds,
                      value, part.military, part.economy, part.exchange, part.treasury)
    _report_layers(episodes)
    _report_interference(episodes)
    for difference in REPORTED_DIFFERENCES:
        logging.info("%-10s   to resolve %.2f: %d episode(s) per side", "", difference, episodes_for(summary.sd, difference))
    return summary


def _report_fit(played: Sequence[Played], weights: Weights, lead_seconds: float, fit_out: Optional[str]) -> Optional[Fit]:
    """How well the board predicts the winners of decided matches, at several distances before the end, and the weights that do it best at `lead_seconds`."""
    if not any(decided(e) for e in played):
        logging.info("no episode was decided, so there is nothing to fit the weights on")
        logging.info("a win rate difference of 0.10 would need %d episode(s) per side, if one were ever observed",
                     episodes_for_win_rate(0.10))
        return None
    used = agreement(played, weights, weights.lead_seconds or lead_seconds)
    if used.n:
        logging.info("the weights in use call %d decided match(es) from %.0fs before the end: %.3f right%s",
                     used.n, weights.lead_seconds or lead_seconds, used.accuracy,
                     f", log loss {used.log_loss:.3f}" if used.log_loss is not None else "")
    chosen: Optional[Fit] = None
    for lead in sorted(set(FIT_LEADS) | {lead_seconds}):
        fit = fit_weights(played, lead)
        if fit is None:
            continue
        if lead == lead_seconds:
            chosen = fit
        held = fit.held_out
        logging.info("fitted %.0fs before the end on %d match(es) (%d won): %.3f right, log loss %.3f%s; %s%s",
                     lead, fit.fitted.n, sum(1 for _, y in graded(played, lead) if y > 0),
                     fit.fitted.accuracy, fit.fitted.log_loss,
                     f", held out {held.accuracy:.3f} right, log loss {held.log_loss:.3f}" if held else "",
                     fit.weights.describe(), "; SEPARATED" if fit.separated else "")
    if chosen is None:
        logging.info("no decided match carries a board from %.0fs before its end, so the weights cannot be fitted: "
                     "only episodes recorded with their history can be fitted on", lead_seconds)
        return None
    if chosen.separated:
        logging.warning("every board %.0fs before the end calls its match right, so these matches fix the direction of the "
                        "weights but not their size, which is the penalty's; they need matches whose board did not already "
                        "say who would win, and weights fitted on them are not written", lead_seconds)
        return chosen
    if fit_out:
        save_weights(chosen.weights, fit_out)
        logging.info("weights fitted %.0fs before the end written to %s", lead_seconds, fit_out)
    return chosen


def _report_layers(episodes: Sequence[Played]) -> None:
    """What the layers did, averaged. A score says an arm went badly; this says which layer it went badly in."""
    keys = ("strategic", "operational", "tactical", "contracts", "completed", "stalled", "losing", "expired", "production")
    if not any(episode.statistics for episode in episodes):
        return
    parts = [f"{key} {_mean(e.statistics.get(key, 0) for e in episodes):.0f}" for key in keys]
    logging.info("%-10s   layers: %s, fulfilment %.2f", "", " ".join(parts),
                 _mean(e.statistics.get("fulfilment", 0.0) for e in episodes))
    # The economy's ledger says whether credits were turned into things at all, which is what the score mostly turns on; a milestone is averaged over the episodes that reached it, with how many did.
    if any("credits_mean" in e.statistics for e in episodes):
        milestones = []
        for key in ("first_factory_s", "second_extractor_s", "second_factory_s"):
            reached = [e.statistics[key] for e in episodes if e.statistics.get(key, -1) >= 0]
            milestones.append(f"{key[:-2]} {_mean(reached):.0f}s ({len(reached)}/{len(episodes)})" if reached else f"{key[:-2]} never")
        logging.info("%-10s   economy: credits %.0f, factories unordered %.2f, idle builders %.2f, %s", "",
                     _mean(e.statistics.get("credits_mean", 0.0) for e in episodes),
                     _mean(e.statistics.get("factory_unordered", 0.0) for e in episodes),
                     _mean(e.statistics.get("builder_idle", 0.0) for e in episodes),
                     ", ".join(milestones))
    # Where the credits went, as shares of everything committed, beside how much of the match was spent unable to produce at all or with the treasury full.
    if any(e.statistics.get("spent") for e in episodes):
        categories = sorted({name for e in episodes for name in e.statistics.get("spent", {})})
        totals = {name: _mean(e.statistics.get("spent", {}).get(name, 0.0) for e in episodes) for name in categories}
        committed = sum(totals.values())
        logging.info("%-10s   spending: %.0f committed, %s; no factory %.2f, banked %.2f", "", committed,
                     " ".join(f"{name} {totals[name] / committed:.2f}" for name in categories if committed > 0),
                     _mean(e.statistics.get("factoryless", 0.0) for e in episodes),
                     _mean(e.statistics.get("banked", 0.0) for e in episodes))
    # Where our units were lost, as shares of everything lost: by role, loose rather than in a squad, and within reach of the enemy's defences.
    if any(e.statistics.get("losses") for e in episodes):
        keys = sorted({name for e in episodes for name in e.statistics.get("losses", {})})
        lost = {name: _mean(e.statistics.get("losses", {}).get(name, 0.0) for e in episodes) for name in keys}
        total = sum(value for name, value in lost.items() if name.startswith("role_"))
        logging.info("%-10s   losses: %.0f, %s", "", total,
                     " ".join(f"{name.removeprefix('role_')} {lost[name] / total:.2f}" for name in keys if total > 0))
    if any(e.statistics.get("postures") for e in episodes):
        names = sorted({name for e in episodes for name in e.statistics.get("postures", {})})
        logging.info("%-10s   postures: %s, changes %.1f", "",
                     " ".join(f"{name} {_mean(e.statistics.get('postures', {}).get(name, 0.0) for e in episodes):.0f}s" for name in names),
                     _mean(e.statistics.get("posture_changes", 0) for e in episodes))
    # A learnt or pinned operational layer also records how often it was asked and how well it met the strategic orders, which is what it is paid for.
    if any("achievement" in e.statistics for e in episodes):
        logging.info("%-10s   operational decisions %.0f, achievement of the strategic orders %+.3f", "",
                     _mean(e.statistics.get("decisions", 0) for e in episodes),
                     _mean(e.statistics.get("achievement", 0.0) for e in episodes))
    # A learnt economy, or its judge played through the learnt layer, records how many investments it chose.
    elif any("decisions" in e.statistics for e in episodes):
        logging.info("%-10s   economic decisions %.0f", "", _mean(e.statistics.get("decisions", 0) for e in episodes))


def _report_interference(episodes: Sequence[Played]) -> None:
    """Whether the arm was measured under interference, and how much of it there was."""
    attached = [episode for episode in episodes if episode.interference]
    disturbed = [episode for episode in episodes if episode.interference.get("touched")]
    if not attached:
        logging.info("%-10s   no intruder: this is the undisturbed number", "")
        return
    if not disturbed:
        logging.info("%-10s   an intruder was attached in %d of %d episode(s) and touched nothing", "",
                     len(attached), len(episodes))
        return
    logging.info("%-10s   interference in %d of %d episode(s): %.1f squad(s) and %.1f "
                 "intervention(s) each, on average over those",
                 "", len(disturbed), len(episodes),
                 _mean(len(e.interference["touched"]) for e in disturbed),
                 _mean(len(e.interference.get("events", [])) for e in disturbed))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def weights_named(name: str) -> Weights:
    """`default` for the adopted weights, `opening` for the military edge alone, or the path of a weights file."""
    if name == "default":
        return default_weights()
    if name == "opening":
        return OPENING_WEIGHTS
    return load_weights(name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", nargs="+", default=None,
                        help="report journals written by earlier runs instead of running anything; several are "
                             "pooled, arm by arm and condition by condition")
    parser.add_argument("--arm", action="append", default=None,
                        help="an arm of the comparison: 'script', script:<name>=<value>[,...] or script:baseline "
                             "for the chain with some of its rules switched, a posture name to pin the strategic layer to, "
                             "operations:<path> or operations-greedy:<path> for a learnt operational layer, "
                             "economy:<path> or economy-greedy:<path> for a learnt economy, "
                             "ops-<rule> (home, nearest, weakest, richest, spawn, random, carry) to pin the operational "
                             "layer to a rule, "
                             "or eco-script for the economy's judge played through the learnt layer. Repeatable")
    parser.add_argument("--reference", default=None,
                        help="the arm every other arm is compared against; defaults to the first --arm, and with --from "
                             "and no --arm to the first arm each condition's episodes name")
    parser.add_argument("--weights", default="default",
                        help="how a cut-off match is scored: 'default' for the adopted weights, 'opening' for the "
                             "military edge alone, or the path of a weights file")
    parser.add_argument("--lead", type=float, default=DEFAULT_LEAD_SECONDS,
                        help="seconds before the end of a decided match the boards the weights are fitted on are taken")
    parser.add_argument("--fit-out", default=None, help="write the weights fitted at --lead to this file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1, help="episodes each arm runs on each instance")
    parser.add_argument("--map", action="append", default=None,
                        help="the map to play, Lake unless given; repeated, the maps are played in turn, one round of "
                             "the arms each, and reported map by map and pooled")
    parser.add_argument("--opponents", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=1, help="-2 very easy to 3 impossible")
    parser.add_argument("--credits", type=int, default=0, help="starting credits by index, 0 is 4000")
    parser.add_argument("--fog", type=int, default=2, help="0 none, 1 basic, 2 line of sight")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-seconds", type=int, default=300, help="game time an episode is cut off at")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--record", default=None,
                        help="where the episodes are written, one JSON object per line; defaults to a new file under local/episodes")
    parser.add_argument("--intrude", action="store_true",
                        help="measure with the script intruder present, which is how the design says "
                             "evaluation is to be run: a number taken without interruption is not the "
                             "number the system will be operated at")
    parser.add_argument("--device", default=None,
                        help="torch device the learnt arms' networks run on; unset, a set network runs on a graphics "
                             "card when there is one and every other network on the processor")
    parser.add_argument("--verbose", action="store_true", help="also log every episode's score")
    arguments = parser.parse_args(argv)

    log_file = paths.configure_logging("eval", arguments.verbose, fmt="%(asctime)s %(levelname)-7s %(message)s")
    logging.info("logging to %s", log_file)
    weights = weights_named(arguments.weights)

    if arguments.source:
        played = [Played.from_dict(entry) for source in arguments.source for entry in read(source)]
        if not played:
            logging.error("no episodes in %s", ", ".join(arguments.source))
            return 1
        # The arms the run was started with, when given, set the reference as they did when it ran: the first of them.
        reference = arguments.reference or (arm_names.name_of(arguments.arm[0]) if arguments.arm else None)
        report(played, weights, reference, arguments.lead, arguments.fit_out)
        return 0

    arm_names.DEVICE = arguments.device
    arms = arm_names.parse_all(arguments.arm or ["script"])
    maps = arguments.map or ["Lake"]
    # The intruder is in the default file name because an interfered-with run and an undisturbed one measure different quantities, and the likeliest way to confuse them is to have written them to the same place.
    run = "-".join(name for name, _ in arms) + ("-intruded" if arguments.intrude else "")
    journal = Journal(arguments.record or default_path(run))
    logging.info("recording to %s", journal.path)

    # One intruder per episode, seeded from the run's seed and the instance, so that every arm of a comparison meets interference drawn the same way. The arms alternate within an instance and the episode index moves the seed on, so no two episodes of a comparison are disturbed identically either.
    outside = [interference(seed=arguments.seed)] if arguments.intrude else []

    settings = ServerSettings(
        host=arguments.host, port=arguments.port, instances=arguments.instances,
        episodes=arguments.episodes, arms=arms, journal=journal, outside=outside,
        assets=AssetPaths.at(arguments.assets) if arguments.assets else AssetPaths.default(),
        episode=EpisodeSettings(map=maps[0], maps=maps if len(maps) > 1 else [], opponents=arguments.opponents,
                                difficulty=arguments.difficulty, credits=arguments.credits,
                                fog=arguments.fog, seed=arguments.seed,
                                max_seconds=arguments.max_seconds),
    )

    server = Server(settings)
    try:
        sessions = server.serve()
    except KeyboardInterrupt:
        server.stop()
        return 1
    finally:
        journal.close()

    played = [Played.of(record) for session in sessions for record in session.records]
    if not played:
        logging.error("no episodes were played")
        return 1
    report(played, weights, arguments.reference or arms[0][0], arguments.lead, arguments.fit_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
