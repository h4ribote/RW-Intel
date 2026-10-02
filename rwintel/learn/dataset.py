"""Recorded decisions: how a run writes them down and how a learner reads them back.

A run that records writes one directory, `local/datasets/<layer>/<run>/`. Its `run.json` says what wrote it: the layer, the fingerprint of the encoding in force (`encoding.fingerprint`), the run's arguments and reward terms, the commit, the machine and the policy that played. Its shards (`shard-NNNNN.npz`) hold whole episodes' trajectories, each decision with its state, what was legal, what was played and from which distribution, what the layer's judge would have chosen, the value and reward, the signals the reward was priced from, and what the state was encoded from. Its `tables/` hold the type and combat tables those materials are encoded against, once per distinct table.

Trajectories are written when the episode they belong to is sealed (`Rollout.seal`), on a thread of the recorder's own, so that writing never holds up a game. A shard is written under a temporary name and renamed when complete; a run stopped part way leaves complete shards and nothing half written.

Reading refuses a run written by another encoding, whatever its lengths say (`DatasetMismatch`). A learner splits by episode, never by row (`fold`): two decisions of one fight are nearly the same decision, and a split that put one on each side would grade the fit on what it was fitted to.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import queue
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .. import paths
from ..control.policy.encoding import (
    ECONOMIC_SIZE,
    INVESTMENT_SLOTS,
    MEANS,
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    TACTICAL_ACTIONS,
    TACTICAL_SIZE,
    describe,
    fingerprint,
)
from . import materials as materials_module
from .reward import (
    ECONOMIC_SIGNALS,
    OPERATIONAL_SIGNALS,
    TACTICAL_SIGNALS,
    EconomicTerms,
    OperationalTerms,
    TacticalTerms,
)
from .rollout import Trajectory, estimate

log = logging.getLogger(__name__)

#: The version of each layer's layout below, its arrays and its materials tables. Raised for a layer when a field of it changes meaning or a reader of the old layout would misread the new one; a run in another layout is refused, and cannot be encoded again either, since its materials are not the ones the encoding in force reads.
FORMAT: Dict[str, int] = {"tactics": 2, "operations": 3, "economy": 2}

#: Decisions gathered before a shard is written. A shard is the unit a run stopped from outside keeps whole, so this bounds what such a stop can lose.
SHARD_DECISIONS = 20000

#: Share of the episodes a learner holds back to judge its fit by.
VALIDATION_SHARE = 0.1

#: The signal columns of each layer's reward rows.
SIGNALS: Dict[str, Tuple[str, ...]] = {"tactics": TACTICAL_SIGNALS, "operations": OPERATIONAL_SIGNALS,
                                       "economy": ECONOMIC_SIGNALS}

#: The episode endings a session reports (`Session.ending`).
ENDINGS = ("finished", "lost", "rejoined", "stopped")


class DatasetMismatch(ValueError):
    """A recorded run that does not describe the encoding in force, or that is not a run of the layer asked for. Refused rather than worked around: a state written by another feature list loads, fits and yields a network reading every feature in the wrong place, and nothing about its lengths need say so."""


def widths(layer: str) -> Tuple[int, int, int]:
    """How long a state, a first choice and a second choice are for one layer. The second is nought for the tactical layer and the economy, which choose one thing."""
    if layer == "tactics":
        return TACTICAL_SIZE, TACTICAL_ACTIONS, 0
    if layer == "operations":
        return OPERATIONAL_SIZE, OPERATIONAL_REGIONS, OPERATIONAL_PLANS
    if layer == "economy":
        return ECONOMIC_SIZE, INVESTMENT_SLOTS, 0
    raise ValueError(f"no layer named {layer!r}")


def second_rows(second_mask: np.ndarray, first: np.ndarray, layer: str) -> np.ndarray:
    """The row of the second mask that goes with each decision's first choice. The second mask is written one row per first choice laid end to end, since which plans are open depends on the region; a first choice of -1 reads as nothing open."""
    _, regions, second = widths(layer)
    table = second_mask.reshape(len(second_mask), regions, second)
    picked = table[np.arange(len(table)), np.clip(first, 0, regions - 1)]
    return np.where((first >= 0)[:, None], picked, 0).astype(second_mask.dtype)


def terms_for(layer: str, written: Optional[dict] = None):
    """The reward terms a layer's rows are priced at, as written in a run's header or at their defaults."""
    kind = {"tactics": TacticalTerms, "operations": OperationalTerms, "economy": EconomicTerms}[layer]
    return kind(**(written or {}))


def run_directory(layer: str, run: str) -> str:
    return os.path.join(paths.datasets(), layer, run)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _commit() -> Tuple[str, bool]:
    """The commit the code was run from, and whether the working tree differed from it."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPOSITORY, capture_output=True, text=True,
                              timeout=10).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=paths.REPOSITORY,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return head, bool(status)
    except (OSError, subprocess.SubprocessError):
        return "", False


def behaviour_name(behaviour: dict) -> str:
    """A behaviour policy from a run's header as one name: what played, which parameters for a network, whether it was learning as it played, and the share it explored at."""
    name = str(behaviour.get("name", "?"))
    if behaviour.get("parameters"):
        name += f"[{behaviour['parameters']}]"
    if behaviour.get("learning"):
        name += "+learning"
    explore = float(behaviour.get("explore") or 0.0)
    return name + (f"@explore{explore:g}" if explore > 0 else "")


def lineage(header: dict) -> str:
    """The name a run's episodes are known by for the split: the run that first wrote them, which a run encoded again keeps."""
    return str(header.get("lineage") or header.get("run", ""))


def shard_files(run: str) -> List[str]:
    """A run's complete shard files in order; a shard still being written sits under a hidden temporary name (`paths.replacing`) until it is whole, so it is not listed."""
    return sorted(glob.glob(os.path.join(run, "shard-*.npz")))


def fold(run: str, instance: int, attempt: int, share: float = VALIDATION_SHARE, salt: str = "") -> bool:
    """Whether an episode belongs to the held-out side, decided by a digest of what names it. Nothing is written down, so adding data never moves an episode from one side to the other."""
    digest = hashlib.sha256(f"{salt}:{run}:{instance}:{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64 < share


# ---- writing -------------------------------------------------------------------------------

class Recorder:
    """Writes the trajectories a rollout seals into one run's directory, on a thread of its own.

    `accept` is the rollout's sink and only queues what it is handed; a batch is turned into arrays and written once enough decisions have gathered, and `close` writes the rest and finishes the run's header.
    """

    def __init__(self, directory: str, layer: str, header: Optional[dict] = None,
                 shard_decisions: int = SHARD_DECISIONS) -> None:
        widths(layer)
        self.directory = directory
        self.layer = layer
        self.shard_decisions = shard_decisions
        os.makedirs(os.path.join(directory, "tables"), exist_ok=True)
        commit, dirty = _commit()
        self.header = dict(header or {})
        run = os.path.basename(os.path.normpath(directory))
        # The run that first wrote these decisions, which names its episodes for the split however often they are written again.
        self.header.setdefault("lineage", run)
        self.header.update(format=FORMAT[layer], layer=layer, run=run,
                           fingerprint=fingerprint(layer), encoding=describe(layer), signals=list(SIGNALS[layer]),
                           materials={name: list(columns) for name, columns in materials_module.TABLES[layer].items()},
                           commit=commit, dirty=dirty, host=socket.gethostname(), cpus=os.cpu_count(),
                           started=time.strftime("%Y-%m-%dT%H:%M:%S"), decisions=0, trajectories=0, episodes=0,
                           shards=0, complete=False)
        self._write_header()
        self.decisions = 0
        self._pending: List[tuple] = []
        self._pending_count = 0
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[list]]" = queue.Queue()
        self._tables = {os.path.splitext(name)[0] for name in os.listdir(os.path.join(directory, "tables"))}
        self.failure: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def accept(self, trajectories: List[Trajectory], episode: dict, discount: float, trace: float) -> None:
        with self._lock:
            self._pending.append((trajectories, dict(episode), discount, trace))
            self._pending_count += sum(len(t.steps) for t in trajectories)
            if self._pending_count >= self.shard_decisions:
                batch, self._pending, self._pending_count = self._pending, [], 0
                self._queue.put(batch)

    def close(self) -> None:
        """Writes whatever is still gathered, waits for the writing thread and marks the run complete. A failure on the writing thread is raised here, since the run would otherwise end looking as if it had recorded everything."""
        with self._lock:
            batch, self._pending, self._pending_count = self._pending, [], 0
        if batch:
            self._queue.put(batch)
        self._queue.put(None)
        self._thread.join()
        self.header["complete"] = self.failure is None
        self.header["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_header()
        log.info("recorded %d decision(s) in %d shard(s) to %s", self.header["decisions"], self.header["shards"],
                 self.directory)
        if self.failure is not None:
            raise RuntimeError(f"recording to {self.directory} failed") from self.failure

    def _write_header(self) -> None:
        with paths.replacing(os.path.join(self.directory, "run.json")) as handle:
            json.dump(self.header, handle, indent=1, sort_keys=True)

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is None:
                return
            if self.failure is not None:
                continue
            try:
                self._write(batch)
            except BaseException as error:  # a failed shard must surface when the run closes, not vanish with this thread
                log.exception("writing a shard to %s failed", self.directory)
                self.failure = error

    def _context(self, episode: dict) -> str:
        context = episode.pop("context", None)
        if context is None:
            return ""
        name = _digest(context)[:16]
        if name not in self._tables:
            with paths.replacing(os.path.join(self.directory, "tables", f"{name}.json")) as handle:
                json.dump(context, handle, sort_keys=True)
            self._tables.add(name)
        return name

    def _write(self, batch: List[tuple]) -> None:
        arrays = shard_arrays(self.layer, batch, self._context)
        number = self.header["shards"]
        with paths.replacing(os.path.join(self.directory, f"shard-{number:05d}.npz"), "wb") as handle:
            np.savez_compressed(handle, **arrays)
        count = int(arrays["state"].shape[0])
        self.header["shards"] = number + 1
        self.header["decisions"] += count
        self.header["trajectories"] += int(arrays["t_start"].shape[0])
        self.header["episodes"] += len(json.loads(str(arrays["episodes"])))
        self._write_header()


def shard_arrays(layer: str, batch: Sequence[tuple], context=lambda episode: "") -> Dict[str, np.ndarray]:
    """One shard's arrays from sealed episodes, each given as (trajectories, episode, discount, trace)."""
    _, first, second = widths(layer)
    tables = materials_module.TABLES[layer]
    episodes: List[dict] = []
    steps = []
    t_columns: Dict[str, list] = {name: [] for name in ("episode", "squad", "finished", "tail_value", "discount", "trace",
                                                       "start", "length", "engagement")}
    trajectory_of: List[int] = []
    for trajectories, episode, discount, trace in batch:
        episode = dict(episode)
        episode["context"] = context(episode)
        episodes.append(episode)
        for trajectory in trajectories:
            key = trajectory.key[1] if isinstance(trajectory.key, tuple) and len(trajectory.key) > 1 else -1
            t_columns["episode"].append(len(episodes) - 1)
            t_columns["squad"].append(key if isinstance(key, int) else -1)
            t_columns["finished"].append(trajectory.finished)
            t_columns["tail_value"].append(trajectory.tail_value)
            t_columns["discount"].append(discount)
            t_columns["trace"].append(trace)
            t_columns["start"].append(len(steps))
            t_columns["length"].append(len(trajectory.steps))
            t_columns["engagement"].append(max((s.engagement for s in trajectory.steps), default=-1))
            for step in trajectory.steps:
                trajectory_of.append(len(t_columns["start"]) - 1)
                steps.append(step)

    count = len(steps)
    arrays: Dict[str, np.ndarray] = {
        "state": np.asarray([s.state for s in steps], dtype=np.float32).reshape(count, -1),
        "action": np.asarray([s.action for s in steps], dtype=np.int16),
        "second": np.asarray([s.second for s in steps], dtype=np.int16),
        "label": np.asarray([s.label for s in steps], dtype=np.int16),
        "second_label": np.asarray([s.second_label for s in steps], dtype=np.int16),
        "mask": _rows([s.mask for s in steps], first, np.uint8, 1),
        "second_mask": _rows([s.second_mask for s in steps], first * second, np.uint8, 1),
        "soft": _rows([s.soft for s in steps], first, np.float32, math.nan),
        "second_soft": _rows([s.second_soft for s in steps], second, np.float32, math.nan),
        "probabilities": _rows([s.probabilities for s in steps], first, np.float32, math.nan),
        "second_probabilities": _rows([s.second_probabilities for s in steps], second, np.float32, math.nan),
        "log_prob": np.asarray([s.log_prob for s in steps], dtype=np.float32),
        "value": np.asarray([s.value for s in steps], dtype=np.float32),
        "reward": np.asarray([s.reward for s in steps], dtype=np.float32),
        "periods": np.asarray([s.periods for s in steps], dtype=np.int32),
        "at_ms": np.asarray([s.at_ms for s in steps], dtype=np.int64),
        "squad": np.asarray([s.squad for s in steps], dtype=np.int32),
        "tainted": np.asarray([s.tainted for s in steps], dtype=bool),
        "version": np.asarray([s.version for s in steps], dtype=np.int32),
        "engagement": np.asarray([s.engagement for s in steps], dtype=np.int32),
        "weight": np.asarray([float(s.meta.get("weight", 1.0)) for s in steps], dtype=np.float32),
        "trajectory": np.asarray(trajectory_of, dtype=np.int32),
        "meta": np.asarray(json.dumps({str(i): s.meta for i, s in enumerate(steps) if s.meta}, sort_keys=True)),
        "episodes": np.asarray(json.dumps(episodes, sort_keys=True)),
    }
    width = len(SIGNALS[layer])
    arrays["signals"] = np.asarray([row for s in steps for row in s.signals], dtype=np.float64).reshape(-1, width)
    arrays["signal_counts"] = np.asarray([len(s.signals) for s in steps], dtype=np.int32)
    written = [materials_module.tables(s.materials) if s.materials is not None else None for s in steps]
    arrays["has_materials"] = np.asarray([w is not None for w in written], dtype=bool)
    for name, columns in tables.items():
        rows = [row for w in written if w is not None for row in w[name]]
        arrays[f"m_{name}"] = np.asarray(rows, dtype=np.float64).reshape(-1, len(columns))
        arrays[f"m_{name}_counts"] = np.asarray([len(w[name]) if w is not None else 0 for w in written], dtype=np.int32)
    arrays["t_episode"] = np.asarray(t_columns["episode"], dtype=np.int32)
    arrays["t_squad"] = np.asarray(t_columns["squad"], dtype=np.int32)
    arrays["t_finished"] = np.asarray(t_columns["finished"], dtype=bool)
    arrays["t_tail_value"] = np.asarray(t_columns["tail_value"], dtype=np.float32)
    arrays["t_discount"] = np.asarray(t_columns["discount"], dtype=np.float64)
    arrays["t_trace"] = np.asarray(t_columns["trace"], dtype=np.float64)
    arrays["t_start"] = np.asarray(t_columns["start"], dtype=np.int64)
    arrays["t_length"] = np.asarray(t_columns["length"], dtype=np.int64)
    arrays["t_engagement"] = np.asarray(t_columns["engagement"], dtype=np.int32)
    return arrays


def _rows(values: Sequence[Sequence[float]], width: int, dtype, missing) -> np.ndarray:
    """A fixed-width block from per-decision rows, with a row that was not given filled with `missing`."""
    block = np.full((len(values), width), missing, dtype=dtype)
    for index, row in enumerate(values):
        if len(row):
            block[index] = row
    return block


# ---- reading -------------------------------------------------------------------------------

#: Decision arrays a reader concatenates as they are, and those that index into something and are shifted as shards are joined.
_DECISION = ("state", "action", "second", "label", "second_label", "mask", "second_mask", "soft", "second_soft",
             "probabilities", "second_probabilities", "log_prob", "value", "reward", "periods", "at_ms", "squad",
             "tainted", "version", "engagement", "weight", "signal_counts", "has_materials")
_TRAJECTORY = ("t_squad", "t_finished", "t_tail_value", "t_discount", "t_trace", "t_length", "t_engagement")


@dataclass
class Dataset:
    """One or more recorded runs of one layer, read into memory as one set of arrays."""

    layer: str
    runs: List[dict]
    arrays: Dict[str, np.ndarray]
    #: Every episode, in shard order, each carrying the index of the run it came from (`run`) and the run's name (`run_name`).
    episodes: List[dict]
    signals: np.ndarray
    meta: Dict[int, dict] = field(default_factory=dict)
    #: Per table, its rows and where each decision's rows start; empty when the materials were not asked for.
    materials: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    _contexts: Dict[Tuple[int, str], materials_module.Context] = field(default_factory=dict)

    @classmethod
    def open(cls, sources: Sequence[str], layer: Optional[str] = None, check: bool = True,
             with_materials: bool = False, shards: Optional[Sequence[str]] = None) -> "Dataset":
        """Reads every shard of the runs named, which are directories with a `run.json` in them; with `shards`, only the shard files listed, which is how a run still being written is read as of one listing."""
        chosen = {os.path.abspath(shard) for shard in shards} if shards is not None else None
        runs: List[dict] = []
        parts: Dict[str, list] = {name: [] for name in _DECISION + _TRAJECTORY + ("trajectory", "t_episode", "t_start")}
        signals: List[np.ndarray] = []
        material_rows: Dict[str, list] = {}
        material_counts: Dict[str, list] = {}
        episodes: List[dict] = []
        meta: Dict[int, dict] = {}
        decisions = trajectories = 0
        for source in sources:
            with open(os.path.join(source, "run.json"), encoding="utf-8") as handle:
                header = json.load(handle)
            header["path"] = source
            if layer is None:
                layer = header["layer"]
            if header.get("layer") != layer:
                raise DatasetMismatch(f"{source} is a run of the {header.get('layer')} layer, not of the {layer} layer")
            _check_layout(source, header, layer)
            if check and header.get("fingerprint") != fingerprint(layer):
                raise DatasetMismatch(
                    f"{source} was written by another {layer} encoding (fingerprint {str(header.get('fingerprint'))[:12]} "
                    f"against {fingerprint(layer)[:12]} in force, revision {header.get('encoding', {}).get('revision')}); "
                    f"encode it again from its materials with `learn dataset reencode` rather than reading it")
            run_index = len(runs)
            runs.append(header)
            for shard in shard_files(source):
                if chosen is not None and os.path.abspath(shard) not in chosen:
                    continue
                with np.load(shard, allow_pickle=False) as data:
                    count = int(data["state"].shape[0])
                    for name in _DECISION + _TRAJECTORY:
                        parts[name].append(data[name])
                    parts["trajectory"].append(data["trajectory"] + trajectories)
                    parts["t_episode"].append(data["t_episode"] + len(episodes))
                    parts["t_start"].append(data["t_start"] + decisions)
                    signals.append(data["signals"])
                    for index, row_meta in json.loads(str(data["meta"])).items():
                        meta[int(index) + decisions] = row_meta
                    for episode in json.loads(str(data["episodes"])):
                        episodes.append(dict(episode, run=run_index, run_name=lineage(header)))
                    if with_materials:
                        for name in materials_module.TABLES[layer]:
                            material_rows.setdefault(name, []).append(data[f"m_{name}"])
                            material_counts.setdefault(name, []).append(data[f"m_{name}_counts"])
                    decisions += count
                    trajectories += int(data["t_start"].shape[0])
        if layer is None:
            raise ValueError("no run was named")
        state, first, second = widths(layer)
        empty = {"state": np.zeros((0, state), np.float32), "mask": np.zeros((0, first), np.uint8),
                 "second_mask": np.zeros((0, first * second), np.uint8), "soft": np.zeros((0, first), np.float32),
                 "second_soft": np.zeros((0, second), np.float32), "probabilities": np.zeros((0, first), np.float32),
                 "second_probabilities": np.zeros((0, second), np.float32)}
        arrays = {name: (np.concatenate(chunks) if chunks else empty.get(name, np.zeros(0)))
                  for name, chunks in parts.items()}
        width = len(SIGNALS[layer])
        dataset = cls(layer=layer, runs=runs, arrays=arrays, episodes=episodes,
                      signals=np.concatenate(signals) if signals else np.zeros((0, width)), meta=meta)
        for name, columns in materials_module.TABLES[layer].items():
            if name in material_rows:
                counts = np.concatenate(material_counts[name])
                dataset.materials[name] = (np.concatenate(material_rows[name]).reshape(-1, len(columns)),
                                           np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        return dataset

    def __len__(self) -> int:
        return int(self.arrays["state"].shape[0])

    @property
    def signal_starts(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.arrays["signal_counts"])]).astype(np.int64)

    def episode_of(self, decision: int) -> dict:
        return self.episodes[int(self.arrays["t_episode"][self.arrays["trajectory"][decision]])]

    def episode_index(self) -> np.ndarray:
        """Which episode each decision belongs to."""
        return self.arrays["t_episode"][self.arrays["trajectory"]]

    def held_out(self, share: float = VALIDATION_SHARE, salt: str = "") -> np.ndarray:
        """Per decision, whether its episode is on the held-out side of the split."""
        sides = np.asarray([fold(e.get("run_name", ""), int(e.get("instance", -1)), int(e.get("attempt", 0)), share, salt)
                            for e in self.episodes], dtype=bool)
        return sides[self.episode_index()] if len(self) else np.zeros(0, dtype=bool)

    def behaviours(self) -> np.ndarray:
        """Per decision, the name of the policy that played it (`behaviour_name`): the run's `opponent` for the arena's other side, squad number 1, where the run recorded that side, and the run's `behaviour` otherwise."""
        if not len(self):
            return np.zeros(0, dtype=object)
        names = []
        for run in self.runs:
            ours = behaviour_name(run.get("behaviour") or {})
            theirs = behaviour_name(run["opponent"]) if run.get("opponent") else ours
            names.append((ours, theirs))
        runs = np.asarray([int(e["run"]) for e in self.episodes], dtype=np.int64)[self.arrays["t_episode"]]
        per_trajectory = np.asarray([names[run][1 if squad == 1 else 0]
                                     for run, squad in zip(runs.tolist(), self.arrays["t_squad"].tolist())],
                                    dtype=object)
        return per_trajectory[self.arrays["trajectory"]]

    def usable(self, keep_tainted: bool = False, endings: Optional[Iterable[str]] = None) -> np.ndarray:
        """Per decision, whether it is to be learnt from: not about a squad somebody interfered with unless asked for, and from an episode that ended in one of `endings` when they are given."""
        keep = np.ones(len(self), dtype=bool) if keep_tainted else ~self.arrays["tainted"]
        if endings is not None:
            allowed = set(endings)
            ended = np.asarray([e.get("ending") in allowed for e in self.episodes], dtype=bool)
            keep &= ended[self.episode_index()]
        return keep

    def signal_rows(self, decision: int) -> np.ndarray:
        starts = self.signal_starts
        return self.signals[starts[decision]:starts[decision + 1]]

    def rewards(self, terms=None) -> np.ndarray:
        """Every decision's reward priced from its signals, at the terms the run was paid under or at the ones given."""
        if terms is None and len({json.dumps(run.get("terms"), sort_keys=True) for run in self.runs}) > 1:
            raise ValueError("these runs were paid under different terms, so a reward priced again has to say which")
        terms = terms or terms_for(self.layer, self.runs[0].get("terms") if self.runs else None)
        starts = self.signal_starts
        return np.asarray([terms.reward(self.signals[starts[i]:starts[i + 1]]) for i in range(len(self))],
                          dtype=np.float64)

    def estimates(self, rewards: Optional[np.ndarray] = None, discount: Optional[float] = None,
                  trace: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Advantages and returns over every trajectory, from the recorded rewards and values or from the rewards given, at the discount and trace each trajectory was collected under or at the ones given."""
        rewards = self.arrays["reward"].astype(np.float64) if rewards is None else rewards
        values = self.arrays["value"].astype(np.float64)
        periods = self.arrays["periods"]
        advantages = np.zeros(len(self))
        returns = np.zeros(len(self))
        a = self.arrays
        for t in range(len(a["t_start"])):
            start, length = int(a["t_start"][t]), int(a["t_length"][t])
            span = slice(start, start + length)
            adv, ret = estimate(rewards[span], values[span], periods[span], bool(a["t_finished"][t]),
                                float(a["t_tail_value"][t]),
                                float(a["t_discount"][t]) if discount is None else discount,
                                float(a["t_trace"][t]) if trace is None else trace)
            advantages[span], returns[span] = adv, ret
        return advantages, returns

    def material_rows(self, decision: int) -> Dict[str, np.ndarray]:
        if not self.materials:
            raise ValueError("these runs were opened without their materials")
        return {name: rows[starts[decision]:starts[decision + 1]] for name, (rows, starts) in self.materials.items()}

    def context(self, decision: int) -> materials_module.Context:
        episode = self.episode_of(decision)
        key = (int(episode["run"]), str(episode.get("context", "")))
        if key not in self._contexts:
            path = os.path.join(self.runs[key[0]]["path"], "tables", f"{key[1]}.json")
            with open(path, encoding="utf-8") as handle:
                written = json.load(handle)
            self._contexts[key] = materials_module.Context.of(written["types"], written.get("combat"))
        return self._contexts[key]

    def rebuild(self, decision: int) -> List[float]:
        """The state the encoding in force writes for one decision's materials."""
        return materials_module.rebuild(self.layer, self.material_rows(decision), self.context(decision))


# ---- what the tools do ---------------------------------------------------------------------

def verify(dataset: Dataset) -> Tuple[int, int, int]:
    """How many decisions were rebuilt from their materials, how many of those the encoding in force does not rebuild to exactly the state recorded (compared at the single precision the state is kept in), and how many could not be rebuilt at all: no materials, or an episode sealed without the tables they are encoded against."""
    checked = mismatched = unbuildable = 0
    states = dataset.arrays["state"]
    for decision in range(len(dataset)):
        if not dataset.arrays["has_materials"][decision] or not dataset.episode_of(decision).get("context"):
            unbuildable += 1
            continue
        rebuilt = np.asarray(dataset.rebuild(decision), dtype=np.float32)
        checked += 1
        if rebuilt.shape != states[decision].shape or not np.array_equal(rebuilt, states[decision]):
            mismatched += 1
    return checked, mismatched, unbuildable


def inspect(dataset: Dataset) -> dict:
    """What a set of runs holds, in the terms that say whether it is fit to learn from: as a whole, run by run, policy by policy (what each played, how far it agreed with the judge and how widely its distribution was spread), and map by map."""
    a = dataset.arrays
    count = len(dataset)
    endings: Dict[str, int] = {}
    for episode in dataset.episodes:
        endings[episode.get("ending", "?")] = endings.get(episode.get("ending", "?"), 0) + 1
    _, first, second = widths(dataset.layer)

    def shares(values: np.ndarray, width: int) -> List[float]:
        values = values[values >= 0]
        if not len(values):
            return [0.0] * width
        return (np.bincount(values.astype(np.int64), minlength=width)[:width] / len(values)).round(4).tolist()

    labelled = a["label"] >= 0

    def agreement(rows: np.ndarray) -> float:
        rows = rows & labelled
        return round(float((a["action"][rows] == a["label"][rows]).mean()) if rows.any() else 0.0, 4)

    def entropy(probabilities: np.ndarray, rows: np.ndarray) -> Tuple[float, float]:
        """The mean entropy in nats of the distributions the played decisions were drawn from, and the share of decisions whose distribution is not known."""
        chosen = probabilities[rows]
        if not len(chosen) or not chosen.shape[1]:
            return 0.0, 0.0
        known = ~np.isnan(chosen).any(axis=1)
        if not known.any():
            return 0.0, 1.0
        p = chosen[known].astype(np.float64)
        logs = np.log(np.where(p > 0, p, 1.0))
        return round(max(0.0, float(-(p * logs).sum(axis=1).mean())), 4), round(float(1.0 - known.mean()), 4)

    episode_of = dataset.episode_index() if count else np.zeros(0, dtype=np.int64)
    run_of = np.asarray([int(e["run"]) for e in dataset.episodes], dtype=np.int64)[episode_of] if count else episode_of
    map_names = [os.path.splitext(os.path.basename(str(e.get("map", ""))))[0] for e in dataset.episodes]
    map_of = np.asarray(map_names, dtype=object)[episode_of] if count else np.zeros(0, dtype=object)

    by_run = []
    for index, run in enumerate(dataset.runs):
        rows = run_of == index
        mine = [e for e in dataset.episodes if int(e["run"]) == index]
        endings_of: Dict[str, int] = {}
        for episode in mine:
            endings_of[episode.get("ending", "?")] = endings_of.get(episode.get("ending", "?"), 0) + 1
        by_run.append({"run": run.get("run"), "behaviour": behaviour_name(run.get("behaviour") or {}),
                       "opponent": behaviour_name(run["opponent"]) if run.get("opponent") else None,
                       "decisions": int(rows.sum()), "episodes": len(mine), "endings": endings_of,
                       "agreement": agreement(rows),
                       "maps": sorted({name for name, episode in zip(map_names, dataset.episodes)
                                       if int(episode["run"]) == index})})

    names = dataset.behaviours()
    by_behaviour = {}
    for name in sorted(set(names.tolist())):
        rows = names == name
        spread, unknown = entropy(a["probabilities"], rows)
        entry = {"decisions": int(rows.sum()), "played": shares(a["action"][rows], first), "agreement": agreement(rows),
                 "entropy": spread, "unknown_distribution": unknown}
        if second:
            entry["played_second"] = shares(a["second"][rows], second)
            entry["second_entropy"] = entropy(a["second_probabilities"], rows)[0]
        by_behaviour[name] = entry

    by_map: Dict[str, dict] = {}
    for name in sorted(set(map_names)):
        by_map[name] = {"episodes": map_names.count(name), "decisions": int((map_of == name).sum())}

    held = dataset.held_out()
    report = {
        "layer": dataset.layer, "runs": [run.get("run") for run in dataset.runs], "decisions": count,
        "trajectories": int(len(a["t_start"])), "episodes": len(dataset.episodes), "endings": endings,
        "behaviour": sorted({json.dumps(run.get("behaviour", {}), sort_keys=True) for run in dataset.runs}),
        "fingerprint_current": all(run.get("fingerprint") == fingerprint(dataset.layer) for run in dataset.runs),
        "tainted_share": round(float(a["tainted"].mean()) if count else 0.0, 4),
        "finished_share": round(float(a["t_finished"].mean()) if len(a["t_finished"]) else 0.0, 4),
        "played": shares(a["action"], first), "label": shares(a["label"], first),
        "agreement": round(float((a["action"][labelled] == a["label"][labelled]).mean()) if labelled.any() else 0.0, 4),
        "materials_share": round(float(a["has_materials"].mean()) if count else 0.0, 4),
        "reward_mean": round(float(a["reward"].mean()) if count else 0.0, 5),
        "signals_mean": dict(zip(SIGNALS[dataset.layer],
                                 (dataset.signals.mean(axis=0).round(5).tolist() if len(dataset.signals)
                                  else [0.0] * len(SIGNALS[dataset.layer])))),
        "versions": [int(a["version"].min()), int(a["version"].max())] if count else [0, 0],
        "held_out": {"episodes": int(sum(fold(e.get("run_name", ""), int(e.get("instance", -1)), int(e.get("attempt", 0)))
                                         for e in dataset.episodes)),
                     "decisions": int(held.sum())},
        "by_run": by_run, "by_behaviour": by_behaviour, "by_map": by_map,
    }
    if second:
        report["played_second"] = shares(a["second"], second)
        report["label_second"] = shares(a["second_label"], second)
    if dataset.layer == "operations":
        # How often a squad was sent walking (the first share) and by each transport slot.
        report["played_means"] = shares(np.where(a["second"] >= 0, a["second"] % MEANS, -1), MEANS)
        report["label_means"] = shares(np.where(a["second_label"] >= 0, a["second_label"] % MEANS, -1), MEANS)
    return report


def reencode(source: str, target: str) -> int:
    """Writes a run again with every state encoded afresh from its materials by the encoding in force, and says how many decisions were written. A run holding a decision without materials cannot be encoded again and is refused."""
    with open(os.path.join(source, "run.json"), encoding="utf-8") as handle:
        header = json.load(handle)
    layer = header["layer"]
    _check_layout(source, header, layer)
    os.makedirs(os.path.join(target, "tables"), exist_ok=True)
    for table in glob.glob(os.path.join(source, "tables", "*.json")):
        with open(table, encoding="utf-8") as handle, paths.replacing(os.path.join(target, "tables", os.path.basename(table))) as out:
            out.write(handle.read())
    written = shards = 0
    for shard in sorted(glob.glob(os.path.join(source, "shard-*.npz"))):
        if shard.endswith(".tmp.npz"):
            continue
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        part = _shard_dataset(layer, header, source, arrays)
        if not part.arrays["has_materials"].all():
            raise DatasetMismatch(f"{shard} holds decisions without materials, which cannot be encoded again")
        arrays["state"] = np.asarray([part.rebuild(int(i)) for i in range(len(part))], dtype=np.float32).reshape(len(part), -1)
        with paths.replacing(os.path.join(target, f"shard-{shards:05d}.npz"), "wb") as handle:
            np.savez_compressed(handle, **arrays)
        shards += 1
        written += len(part)
    header.update(fingerprint=fingerprint(layer), encoding=describe(layer), run=os.path.basename(os.path.normpath(target)),
                  lineage=lineage(header), reencoded_from=header.get("run"), shards=shards, decisions=written)
    with paths.replacing(os.path.join(target, "run.json")) as handle:
        json.dump(header, handle, indent=1, sort_keys=True)
    return written


def _check_layout(source: str, header: dict, layer: str) -> None:
    if header.get("format") != FORMAT.get(layer):
        raise DatasetMismatch(f"{source} is a {layer} run written in layout {header.get('format')}, and this reads "
                              f"layout {FORMAT.get(layer)}; a run in another layout cannot be read or encoded again")


def _shard_dataset(layer: str, header: dict, source: str, data: Dict[str, np.ndarray]) -> Dataset:
    """One shard's arrays as a dataset, for working through a run a shard at a time."""
    width = len(SIGNALS[layer])
    episodes = [dict(e, run=0, run_name=lineage(header)) for e in json.loads(str(data["episodes"]))]
    arrays = {name: data[name] for name in _DECISION + _TRAJECTORY + ("trajectory", "t_episode", "t_start")}
    dataset = Dataset(layer=layer, runs=[dict(header, path=source)], arrays=arrays, episodes=episodes,
                      signals=data["signals"].reshape(-1, width))
    for name, columns in materials_module.TABLES[layer].items():
        counts = data[f"m_{name}_counts"]
        dataset.materials[name] = (data[f"m_{name}"].reshape(-1, len(columns)),
                                   np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    return dataset
