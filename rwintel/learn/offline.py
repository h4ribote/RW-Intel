"""Learning from recorded datasets alone, on a graphics card when there is one: `python -m rwintel.learn offline`.

Nothing here plays a game or talks to a control process. The decisions of the named runs are read once, moved to the device (states at half precision, upcast per batch), and fitted with one of seven methods:

- `bc`: the clone loss of `imitation` (label smoothing over the legal actions, the teacher's distribution where one was written) against the judge's labels.
- `distill`: the distributions a teacher network (`--teacher`) puts on every decision become the targets, computed once in batches; the report adds agreement with the teacher.
- `iql`: implicit Q-learning on the actions played. Twin action values regress on `r + gamma^periods * V(s')`, the value by expectile regression on the smaller target action value, and the policy is extracted by advantage-weighted regression.
- `awr`: advantage-weighted regression on the actions played, the advantage being the recorded discounted return less a value fitted to it.
- `topbc`: the clone loss against the actions played, on the trajectories whose return is in the top share of their behaviour policy's.
- `cql`: conservative Q-learning on the same heads, the policy extracted as for `iql`.
- `fqe`: fitted Q evaluation of a fixed policy (`--policy`), reporting its estimated value at the start of the held-out trajectories with a bootstrap interval over episodes.

The policy of `iql`, `awr` and `cql` also carries the clone loss against the judge's labels at the weight `judge`, so it leaves the judge only where the advantages keep pointing away from it.
The held-out side is the dataset's episode split. Every run writes a JSON report, and every method reports how far the policy it produced is from the policies that played (`behaviour_distance`) and how much more it agrees with the actions of high-return trajectories than of low-return ones (`return_alignment`).
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import paths
from ..control.policy.encoding import MEANS, OPERATIONAL_PLANS, OPERATIONAL_REGIONS, SQUAD_SLOTS, TACTICAL_SIZE
from . import models
from .dataset import VALIDATION_SHARE, Dataset, shard_files, terms_for, widths
from .imitation import FRACTION_SALT, SCRIPT, SMOOTHING, _cross_entropy
from .net import MASKED, actor_critic_parameters, value_parameters
from .policy import OPERATIONAL, TACTICAL

log = logging.getLogger(__name__)

METHODS = ("bc", "distill", "iql", "awr", "topbc", "cql", "fqe")
PLAYED_METHODS = ("iql", "awr", "topbc", "cql", "fqe")
#: The methods that extract the policy by advantage-weighted regression and can follow growing runs.
ADVANTAGE_METHODS = ("iql", "awr", "cql")

#: Rows per gradient step on a graphics card and on the processor.
GPU_BATCH = 4096
CPU_BATCH = 512

#: Learning rate at `RATE_BATCH` rows per step, per kind of network; it is scaled by the square root of the batch's ratio to that.
BASE_RATE = {"flat": 1e-3, "set": 3e-4}
RATE_BATCH = 256

#: Steps over which the learning rate rises linearly from nought.
WARMUP_STEPS = 100

#: Share of the rate imitation's cosine decay reaches on its last step, and the decays offered (`none` holds the rate after the warm-up).
DECAY_FLOOR = 0.05
DECAYS = ("cosine", "none")

EPOCHS = 20
PATIENCE = 5

#: Polyak rate of the target action values, expectile of the value regression, inverse temperature and cap of the advantage weights.
TAU = 0.005
EXPECTILE = 0.7
BETA = 3.0
WEIGHT_CAP = 100.0

#: Inverse temperature of `awr`, whose advantage carries the noise of each single return rather than the expectation an action value takes.
AWR_BETA = 1.0

#: Weight of the clone loss against the judge's labels beside the advantage-weighted loss of `iql`, `awr` and `cql`.
JUDGE = 1.0

#: Passes over the training side in which `iql`, `awr` and `cql` fit only the critic, before the actor moves on its advantages.
CRITIC_EPOCHS = 5

#: Weight of the conservative penalty in `cql`, and the share of trajectories `topbc` keeps and `return_alignment` compares at either end.
CQL_ALPHA = 1.0
TOP = 0.25

#: A greedy action the behaviour policy played with less than this probability counts as off the data, and a policy choosing such actions on more than `UNTRUSTED_SHARE` of decisions is flagged as untrusted.
OFF_DATA = 1e-3
UNTRUSTED_SHARE = 0.2

#: Resamples of the held-out episodes for the interval on an evaluated policy's value.
BOOTSTRAP = 1000

#: Rows per forward pass when measuring and when a teacher labels the data.
EVAL_BATCH = 8192

#: Share of the card's free memory the set states may take; above it they stay in pinned host memory and each batch is copied over.
SET_SHARE = 0.5


@dataclass
class Settings:
    layer: str
    sources: List[Tuple[str, float]]
    method: str = "bc"
    net: str = "flat"
    depth: Optional[int] = None
    width: Optional[int] = None
    load: Optional[str] = None
    init_flat: Optional[str] = None
    teacher: Optional[str] = None
    policy: Optional[str] = None
    save: Optional[str] = None
    device: str = "auto"
    epochs: int = EPOCHS
    batch: Optional[int] = None
    learning_rate: Optional[float] = None
    decay: str = "cosine"
    seed: int = 0
    fraction: float = 1.0
    keep_tainted: bool = False
    report: Optional[str] = None
    discount: Optional[float] = None
    #: The terms every recorded reward is priced at, as a match layer's terms are written in a run's header; None prices each run at the terms it was paid under. Given, their discount is the one the returns are discounted at.
    terms: Optional[dict] = None
    tau: float = TAU
    expectile: float = EXPECTILE
    #: The advantage weights' inverse temperature; None is `AWR_BETA` for `awr` and `BETA` otherwise.
    beta: Optional[float] = None
    top: float = TOP
    alpha: float = CQL_ALPHA
    judge: float = JUDGE
    critic_epochs: int = CRITIC_EPOCHS
    patience: int = PATIENCE
    smoothing: float = SMOOTHING
    #: Follow mode (`follow`): directories whose runs of the layer are read as they grow, updates between published versions, the shortest time between looks for new shards, the updates to stop after (None runs until stopped), and the most decisions the replay buffer holds (None holds every one).
    watch: List[str] = field(default_factory=list)
    publish_every: int = 500
    rescan_seconds: float = 30.0
    updates: Optional[int] = None
    buffer: Optional[int] = None
    #: Share of a graphics card's memory the process may take (`models.limit_card`); None is no limit, except under `follow`, which takes `FOLLOW_CARD_SHARE`.
    card_share: Optional[float] = None


#: Share of the card a following learner limits itself to when no share is given.
FOLLOW_CARD_SHARE = 0.5


def inverse_temperature(settings: Settings) -> float:
    """The advantage weights' inverse temperature: the one given, or the method's default."""
    if settings.beta is not None:
        return settings.beta
    return AWR_BETA if settings.method == "awr" else BETA


def follow_card_share(settings: Settings) -> float:
    """The share of the card the following learner takes: the one given, or `FOLLOW_CARD_SHARE`."""
    return settings.card_share if settings.card_share is not None else FOLLOW_CARD_SHARE


def device_of(name: Optional[str]) -> torch.device:
    if not name or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# The data on the device.

@dataclass
class Data:
    """Every usable decision of the named runs, as tensors on the device. Index `-1` in `next` means no next decision: the trajectory ended there (`done`) or was cut off (`cut`).

    The discounted return of a decision is `mc + tail_gamma * V(tail)`: `mc` sums the scaled rewards up to the end of its trajectory, or up to the last decision of a cut trajectory, whose value `tail` names (-1 where the trajectory ended).
    """

    layer: str
    flat: torch.Tensor
    sets: Optional[torch.Tensor]
    mask: torch.Tensor
    table: Optional[torch.Tensor]
    label: torch.Tensor
    second_label: torch.Tensor
    action: torch.Tensor
    second: torch.Tensor
    squad: torch.Tensor
    weight: torch.Tensor
    held: torch.Tensor
    soft: torch.Tensor
    second_soft: torch.Tensor
    behaviour: torch.Tensor
    second_behaviour: torch.Tensor
    reward: torch.Tensor
    gamma: torch.Tensor
    next: torch.Tensor
    done: torch.Tensor
    start: torch.Tensor
    episode: torch.Tensor
    mc: torch.Tensor
    tail: torch.Tensor
    tail_gamma: torch.Tensor
    sources: np.ndarray
    behaviours: np.ndarray
    returns: np.ndarray
    trajectory: np.ndarray
    reward_scale: float = 1.0

    def __len__(self) -> int:
        return int(self.label.shape[0])


def _next_indices(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per decision, the index of the next decision of its trajectory (-1 at the last), whether the last ended properly, and whether a decision opens its trajectory."""
    a = dataset.arrays
    count = len(dataset)
    following = np.arange(1, count + 1, dtype=np.int64)
    ends = (a["t_start"] + a["t_length"] - 1).astype(np.int64)
    following[ends[a["t_length"] > 0]] = -1
    done = np.zeros(count, dtype=bool)
    done[ends[(a["t_length"] > 0) & a["t_finished"]]] = True
    start = np.zeros(count, dtype=bool)
    start[a["t_start"][a["t_length"] > 0].astype(np.int64)] = True
    return following, done, start


def _returns(rewards: np.ndarray, gamma: np.ndarray, following: np.ndarray) -> np.ndarray:
    """Discounted returns, each trajectory closed at nought where it ends or is cut."""
    out = np.zeros(len(rewards))
    for index in range(len(rewards) - 1, -1, -1):
        after = following[index]
        out[index] = rewards[index] + (gamma[index] * out[after] if after >= 0 else 0.0)
    return out


def _bootstrapped(rewards: np.ndarray, gamma: np.ndarray, following: np.ndarray,
                  done: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per decision, the discounted rewards up to where its trajectory ends or is cut, the decision whose value stands for the rest of a cut trajectory (-1 where it ended), and the discount on that value.

    The last decision of a cut trajectory has no next decision to close its own reward, so its value replaces that reward and everything after it.
    """
    count = len(rewards)
    summed = np.zeros(count)
    tail = np.full(count, -1, dtype=np.int64)
    factor = np.zeros(count)
    for index in range(count - 1, -1, -1):
        after = following[index]
        if after >= 0:
            summed[index] = rewards[index] + gamma[index] * summed[after]
            tail[index] = tail[after]
            factor[index] = gamma[index] * factor[after]
        elif done[index]:
            summed[index] = rewards[index]
        else:
            tail[index] = index
            factor[index] = 1.0
    return summed, tail, factor


#: The device type of each decision column; None is the state's storage type (half precision on a graphics card).
_COLUMNS: Dict[str, Optional[torch.dtype]] = {
    "flat": None, "mask": torch.uint8, "table": torch.uint8, "label": torch.long, "second_label": torch.long,
    "action": torch.long, "second": torch.long, "squad": torch.long, "weight": torch.float32, "held": torch.bool,
    "soft": torch.float32, "second_soft": torch.float32, "behaviour": torch.float32, "second_behaviour": torch.float32,
    "reward": torch.float32, "gamma": torch.float32, "next": torch.long, "done": torch.bool, "start": torch.bool,
    "episode": torch.long, "mc": torch.float32, "tail": torch.long, "tail_gamma": torch.float32}
_HOST = ("sources", "behaviours", "returns", "trajectory")
#: The columns holding the row of another decision (-1 for none), which appends offset and eviction renumbers.
_LINKS = ("next", "tail")

#: Rows copied at a time when an append writes into the storage and when eviction compacts it.
_SET_CHUNK = 65536

#: Rows beyond the cap that one append may bring before the trim, reserved with a capped buffer's storage.
APPEND_HEADROOM = 65536


def _write(target: torch.Tensor, start: int, values: np.ndarray) -> None:
    """Copies host rows into `target` from row `start`, a chunk at a time, converting to the target's type."""
    for begin in range(0, len(values), _SET_CHUNK):
        part = torch.as_tensor(np.ascontiguousarray(values[begin:begin + _SET_CHUNK]))
        target[start + begin:start + begin + part.shape[0]].copy_(part)


def _compact(storage: torch.Tensor, kept: torch.Tensor) -> None:
    """Moves the rows `kept` (ascending) to the front of `storage` in place, a chunk at a time; each chunk reads rows at or after the ones it writes, so it never overwrites a row a later chunk reads."""
    kept = kept.to(storage.device)
    for begin in range(0, int(kept.shape[0]), _SET_CHUNK):
        part = kept[begin:begin + _SET_CHUNK]
        storage[begin:begin + part.shape[0]] = storage[part]


class Buffer:
    """The usable decisions of recorded runs on the device, grown a listing of shards at a time: what `load_data` reads at once, and the follow learner's replay buffer, which appends only the shards a run gained.

    Each shard is read on its own, since a shard holds whole episodes and their trajectories, so appending shards in order yields the same rows as reading them together. The reward scale is fixed by the first append. With `cap`, the oldest shards of runs not in `kept` are dropped, whole, until at most `cap` decisions remain or none of those is left.
    Every column and the set states live in a storage of `capacity` rows reserved ahead, of which the first `length` are held: an append writes behind them in place and eviction compacts them in place, so a capped buffer reserves its storage once (the cap and `APPEND_HEADROOM`) unless one append brings more rows than that, and otherwise the storage grows by half at a time, returning the old storage to the card.
    """

    def __init__(self, settings: Settings, device: torch.device, with_sets: bool, with_rewards: bool,
                 cap: Optional[int] = None, kept: Sequence[str] = ()) -> None:
        self.settings = settings
        self.device = device
        self.with_sets = with_sets
        self.with_rewards = with_rewards
        self.cap = cap
        self.kept = {os.path.abspath(path) for path in kept}
        self.store = torch.float16 if device.type == "cuda" else torch.float32
        #: Per column, and for the set states, a storage of `capacity` rows whose first `length` are held.
        self.storage: Dict[str, torch.Tensor] = {}
        self.sets: Optional[torch.Tensor] = None
        self.capacity = 0
        self.length = 0
        self.sets_on_host = False
        self.host: Dict[str, np.ndarray] = {}
        #: Per shard appended and still held, in row order: the shard file, its rows, and whether it is kept under the cap.
        self.segments: List[Tuple[str, int, bool]] = []
        self.episodes = 0
        self.trajectories = 0
        self.terms: set = set()
        self.scale: Optional[float] = None
        self.dropped = 0

    def __len__(self) -> int:
        return sum(rows for _, rows, _ in self.segments)

    def _read(self, path: str, weight: float, shard: str) -> dict:
        """One shard's usable decisions as host arrays, every index local to the shard."""
        from . import tokens

        settings = self.settings
        dataset = Dataset.open([path], layer=settings.layer, shards=[shard])
        a = dataset.arrays
        keep = dataset.usable(settings.keep_tainted)
        held = dataset.held_out(VALIDATION_SHARE)
        if settings.fraction < 1.0:
            keep &= held | dataset.held_out(settings.fraction, salt=FRACTION_SALT)
        following, done, start = _next_indices(dataset)
        # A next decision that is not kept leaves the decision before it without a target, as a cut does.
        where = np.full(len(dataset), -1, dtype=np.int64)
        rows = np.flatnonzero(keep)
        where[rows] = np.arange(len(rows))
        following = np.where(following >= 0, where[np.maximum(following, 0)], -1)
        terms = terms_for(settings.layer, settings.terms) if settings.terms is not None else None
        discount = terms.discount if terms is not None else settings.discount
        gamma = a["t_discount"][a["trajectory"]].astype(np.float64)
        if discount is not None:
            gamma = np.full(len(dataset), discount)
        gamma = gamma ** a["periods"].astype(np.float64)
        rewards = dataset.rewards(terms) if self.with_rewards else np.zeros(len(dataset))
        returns = _returns(rewards[rows], gamma[rows], following[rows]) if self.with_rewards else np.zeros(len(rows))
        if self.with_rewards:
            summed, tail, factor = _bootstrapped(rewards[rows], gamma[rows], following[rows], done[rows])
        else:
            summed, tail, factor = np.zeros(len(rows)), np.full(len(rows), -1, dtype=np.int64), np.zeros(len(rows))
        meta = dataset.meta
        host = {"sources": np.asarray([str(meta.get(int(i), {}).get("source", SCRIPT)) for i in rows], dtype=object),
                "behaviours": dataset.behaviours()[rows] if len(rows) else np.zeros(0, dtype=object),
                "returns": returns, "trajectory": a["trajectory"][rows].astype(np.int64)}
        columns = {
            "flat": a["state"][rows], "mask": a["mask"][rows], "label": a["label"][rows].astype(np.int64),
            "second_label": a["second_label"][rows].astype(np.int64), "action": a["action"][rows].astype(np.int64),
            "second": a["second"][rows].astype(np.int64), "squad": a["squad"][rows].astype(np.int64),
            "weight": a["weight"][rows].astype(np.float32) * weight, "held": held[rows],
            "soft": a["soft"][rows], "second_soft": a["second_soft"][rows], "behaviour": a["probabilities"][rows],
            "second_behaviour": a["second_probabilities"][rows], "reward": rewards[rows].astype(np.float32),
            "gamma": gamma[rows].astype(np.float32), "next": following[rows], "done": done[rows], "start": start[rows],
            "episode": dataset.episode_index()[rows].astype(np.int64), "mc": summed.astype(np.float32), "tail": tail,
            "tail_gamma": factor.astype(np.float32)}
        if widths(settings.layer)[2]:
            columns["table"] = a["second_mask"][rows]
        sets = np.asarray(tokens.cached_set_states(path, shards=[shard])[rows]) if self.with_sets else None
        return {"path": path, "shard": shard, "weight": weight, "columns": columns, "host": host, "sets": sets,
                "rows": len(rows), "count": len(dataset), "episodes": len(dataset.episodes),
                "trajectories": int(len(a["t_start"])),
                "terms": json.dumps(settings.terms if settings.terms is not None else dataset.runs[0].get("terms"), sort_keys=True)}

    def append(self, shards: Sequence[Tuple[str, float, str]]) -> int:
        """Reads these shards, given in order as run, weight and shard file, appends their usable decisions after the ones held, drops the oldest beyond the cap, and says how many decisions were appended."""
        pieces = [self._read(path, weight, shard) for path, weight, shard in shards]
        if not pieces:
            return 0
        offset = len(self)
        counted: Dict[str, List[float]] = {}
        for piece in pieces:
            columns = piece["columns"]
            for name in _LINKS:
                columns[name] = np.where(columns[name] >= 0, columns[name] + offset, -1)
            columns["episode"] = columns["episode"] + self.episodes
            piece["host"]["trajectory"] = piece["host"]["trajectory"] + self.trajectories
            offset += piece["rows"]
            self.episodes += piece["episodes"]
            self.trajectories += piece["trajectories"]
            self.segments.append((piece["shard"], piece["rows"], os.path.abspath(piece["path"]) in self.kept))
            self.terms.add(piece["terms"])
            entry = counted.setdefault(piece["path"], [0, 0, piece["weight"]])
            entry[0] += piece["rows"]
            entry[1] += piece["count"]
        for path, (rows, count, weight) in counted.items():
            log.info("%s: %d of %d decision(s) kept at weight %g", path, rows, count, weight)
        added = sum(piece["rows"] for piece in pieces)
        host = {name: np.concatenate([piece["host"][name] for piece in pieces]) for name in _HOST}
        columns = {name: np.concatenate([piece["columns"][name] for piece in pieces]) for name in pieces[0]["columns"]}
        if self.with_rewards and self.scale is None:
            spread = float(np.std(host["returns"])) if len(host["returns"]) else 0.0
            self.scale = 1.0 / spread if spread > 1e-8 else 1.0
            log.info("rewards scaled by %.5g, one over the spread of the returns", self.scale)
        if self.with_rewards and len(self.terms) > 1:
            log.warning("the runs were paid under different terms; each is priced at its own")
        for name in ("reward", "mc"):
            columns[name] = columns[name] * np.float32(self.scale if self.scale is not None else 1.0)
        sets = np.concatenate([piece["sets"] for piece in pieces]) if self.with_sets else None
        self._reserve(self.length + added, columns, sets)
        for name, values in columns.items():
            _write(self.storage[name], self.length, values)
        if sets is not None:
            _write(self.sets, self.length, sets)
        del columns, sets
        self.length += added
        for name, values in host.items():
            self.host[name] = values if name not in self.host else np.concatenate([self.host[name], values])
        self._trim()
        return added

    def _reserve(self, rows: int, columns: Dict[str, np.ndarray], sets: Optional[np.ndarray]) -> None:
        """Makes room for `rows` rows: nothing while they fit, otherwise a storage of half as many rows again (at least the cap and `APPEND_HEADROOM` under a cap) with the held rows copied over and the old storage returned to the card."""
        if self.storage and rows <= self.capacity:
            return
        capacity = max(rows, self.capacity * 3 // 2)
        if self.cap is not None:
            capacity = max(capacity, self.cap + APPEND_HEADROOM)
        for name, values in columns.items():
            held = self.storage.pop(name, None)
            grown = torch.empty((capacity, *values.shape[1:]), dtype=_COLUMNS[name] or self.store, device=self.device)
            if held is not None:
                grown[:self.length].copy_(held[:self.length])
            self.storage[name] = grown
            del held
        self._release()
        if sets is not None:
            self._reserve_sets(capacity, sets.shape[1:])
        self.capacity = capacity

    def _reserve_sets(self, capacity: int, shape: Tuple[int, ...]) -> None:
        """The set states' storage of `capacity` rows, on the card unless it would take more than `SET_SHARE` of the card's free memory, in which case it is pinned host memory from then on."""
        held, self.sets = self.sets, None
        if self.device.type == "cuda" and not self.sets_on_host:
            reserved = capacity * int(np.prod(shape)) * torch.tensor([], dtype=self.store).element_size()
            if reserved > SET_SHARE * torch.cuda.mem_get_info(self.device)[0]:
                self.sets_on_host = True
                log.info("the set states do not fit on the card beside the network, so they stay in pinned host memory")
        if self.sets_on_host:
            grown = torch.empty((capacity, *shape), dtype=self.store, pin_memory=True)
        else:
            grown = torch.empty((capacity, *shape), dtype=self.store, device=self.device)
        if held is not None:
            grown[:self.length].copy_(held[:self.length])
        self.sets = grown
        del held
        self._release()

    def _release(self) -> None:
        """Returns the blocks the caching allocator holds unused to the card."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _trim(self) -> None:
        """Drops the oldest shards not kept until at most `cap` decisions are held, renumbering the links (`_LINKS`) of the rows after them."""
        if self.cap is None or len(self) <= self.cap:
            return
        total = len(self)
        keep = np.ones(total, dtype=bool)
        remaining, position, segments = total, 0, []
        for shard, rows, kept in self.segments:
            if not kept and remaining > self.cap:
                keep[position:position + rows] = False
                remaining -= rows
                self.dropped += 1
            else:
                segments.append((shard, rows, kept))
            position += rows
        if remaining == total:
            return
        kept = torch.as_tensor(np.flatnonzero(keep), device=self.device)
        for storage in self.storage.values():
            _compact(storage, kept)
        if self.sets is not None:
            _compact(self.sets, kept)
        shift = torch.as_tensor(np.cumsum(keep) - 1, device=self.device)
        for name in _LINKS:
            linked = self.storage[name][:remaining]
            linked.copy_(torch.where(linked >= 0, shift[linked.clamp(min=0)], linked))
        for name in _HOST:
            self.host[name] = self.host[name][keep]
        self.length = remaining
        log.info("dropped %d decision(s) of the oldest actor shards to stay within %d", total - remaining, self.cap)
        self.segments = segments

    def data(self) -> Data:
        """The held rows, as views of the storage."""
        if not self.storage:
            raise ValueError("the named runs hold no complete shard to learn from")
        columns = {name: storage[:self.length] for name, storage in self.storage.items()}
        sets = None if self.sets is None else self.sets[:self.length]
        return Data(layer=self.settings.layer, sets=sets, table=columns.pop("table", None),
                    sources=self.host["sources"], behaviours=self.host["behaviours"], returns=self.host["returns"],
                    trajectory=self.host["trajectory"], reward_scale=self.scale if self.scale is not None else 1.0,
                    **columns)


def listing(sources: Sequence[Tuple[str, float]]) -> List[Tuple[str, float, str]]:
    """Every complete shard of the runs, in run and then shard order, with its run and weight."""
    return [(path, weight, shard) for path, weight in sources for shard in shard_files(path)]


def load_data(settings: Settings, device: torch.device, with_sets: bool, with_rewards: bool) -> Data:
    """Reads the runs, keeps the usable decisions (and of the training side's episodes only `fraction`), and moves them to the device."""
    buffer = Buffer(settings, device, with_sets, with_rewards)
    buffer.append(listing(settings.sources))
    return buffer.data()


# Reading a network on a batch.

def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _input(net: nn.Module, data: Data, index: torch.Tensor) -> torch.Tensor:
    """The rows a network reads. A tactical set state takes its 78 flat features from `data.flat`, which holds them at the precision the flat network reads, rather than from the half-precision token cache."""
    if models.kind_of(net) == "set" and data.layer == TACTICAL:
        if data.sets is None:
            raise ValueError("a tactical set network needs the set states, which were not loaded")
        if data.sets.device != index.device:
            sets = data.sets[index.cpu()].to(index.device, non_blocking=True)
        else:
            sets = data.sets[index]
        return torch.cat([data.flat[index].float(), sets[:, TACTICAL_SIZE:].float()], dim=1)
    return data.flat[index].float()


def _slots(data: Data, index: torch.Tensor) -> torch.Tensor:
    squad = data.squad[index]
    inside = (squad >= 0) & (squad < SQUAD_SLOTS)
    return F.one_hot(squad.clamp(0, SQUAD_SLOTS - 1), SQUAD_SLOTS).float() * inside.unsqueeze(-1).float()


def _plan_row(data: Data, index: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    table = data.table[index].reshape(-1, OPERATIONAL_REGIONS, OPERATIONAL_PLANS)
    picked = table[torch.arange(table.shape[0], device=table.device), region.clamp(min=0)]
    return picked * (region >= 0).unsqueeze(-1).to(picked.dtype)


def heads(net: nn.Module, data: Data, index: torch.Tensor, region: Optional[torch.Tensor] = None):
    """The policy's logits in single precision: the first head under the decision's mask, and for the operational layer the plan head at `region` (the likeliest region when none is given) under that region's plan mask."""
    device = data.label.device
    x = _input(net, data, index)
    mask = data.mask[index].float()
    with _autocast(device):
        if data.layer != OPERATIONAL:
            logits, _ = net(x, mask)
            return logits.float(), None
        hidden = net.hidden(x, _slots(data, index))
        regions = net.regions(hidden, mask).float()
        if region is None:
            region = regions.argmax(dim=-1)
        plans = net.plans(hidden, region.clamp(min=0), _plan_row(data, index, region).float()).float()
    return regions, plans


def _log_prob(logits: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1).gather(-1, chosen.clamp(min=0).unsqueeze(-1)).squeeze(-1)


# Measuring a policy.

def measure(net: nn.Module, data: Data, rows: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
            smoothing: float, softs=None, second_softs=None, name: str = "", teacher=None) -> dict:
    """Loss, accuracy (and second, means and carried accuracy for the operational layer), entropy, by source and by behaviour, of the policy against the targets given for `rows`; with `teacher` (its argmax per row), agreement with it."""
    net.eval()
    count = int(rows.shape[0])
    if not count:
        return {"name": name, "count": 0}
    loss_sum = weight_sum = 0.0
    right = torch.zeros(count, dtype=torch.bool, device=rows.device)
    right2 = torch.zeros_like(right)
    means = torch.zeros_like(right)
    agree = torch.zeros_like(right)
    spread = torch.zeros(count, device=rows.device)
    with torch.no_grad():
        for begin in range(0, count, EVAL_BATCH):
            index = rows[begin:begin + EVAL_BATCH]
            span = slice(begin, begin + len(index))
            target = first[index]
            one, two = heads(net, data, index, region=target)
            weights = data.weight[index]
            soft, rows_soft = _distribution(softs, index)
            loss = _cross_entropy(one, target, data.mask[index].float(), smoothing, weights, soft, rows_soft)
            chosen = one.argmax(dim=-1)
            right[span] = chosen == target
            entropy = -(torch.softmax(one, -1) * torch.log_softmax(one, -1)).sum(-1)
            if teacher is not None:
                agree[span] = chosen == teacher[index]
            if two is not None:
                soft2, rows_soft2 = _distribution(second_softs, index)
                loss = loss + _cross_entropy(two, second[index], _plan_row(data, index, target).float(), smoothing,
                                             weights, soft2, rows_soft2)
                picked = two.argmax(dim=-1)
                right2[span] = picked == second[index]
                means[span] = (picked % MEANS) == (second[index] % MEANS)
                entropy = entropy + -(torch.softmax(two, -1) * torch.log_softmax(two, -1)).sum(-1)
            spread[span] = entropy
            loss_sum += float(loss) * float(weights.sum())
            weight_sum += float(weights.sum())
    right_h, right2_h = right.cpu().numpy(), right2.cpu().numpy()
    index_h = rows.cpu().numpy()
    result = {"name": name, "count": count, "loss": round(loss_sum / max(weight_sum, 1e-8), 5),
              "accuracy": round(float(right_h.mean()), 4), "entropy": round(float(spread.mean()), 4)}
    if teacher is not None:
        result["teacher_agreement"] = round(float(agree.float().mean()), 4)
    if data.layer == OPERATIONAL:
        taught = (second[rows] % MEANS).cpu().numpy()
        means_h = means.cpu().numpy()
        carried = taught > 0
        result.update(second_accuracy=round(float(right2_h.mean()), 4), means_accuracy=round(float(means_h.mean()), 4),
                      carried_accuracy=round(float(means_h[carried].mean()), 4) if carried.any() else 0.0)
    for key, labels in (("by_source", data.sources[index_h]), ("by_behaviour", data.behaviours[index_h])):
        groups = {}
        for group in sorted(set(labels.tolist())):
            members = labels == group
            entry = {"count": int(members.sum()), "accuracy": round(float(right_h[members].mean()), 4)}
            if data.layer == OPERATIONAL:
                entry["second_accuracy"] = round(float(right2_h[members].mean()), 4)
            groups[group] = entry
        result[key] = groups
    return result


def _distribution(values: Optional[torch.Tensor], index: torch.Tensor):
    if values is None:
        return None, None
    picked = values[index]
    present = ~torch.isnan(picked).any(dim=-1) if picked.shape[-1] else torch.zeros(len(index), dtype=torch.bool,
                                                                                         device=picked.device)
    if not bool(present.any()):
        return None, None
    return torch.nan_to_num(picked, nan=0.0), present


def behaviour_distance(net: nn.Module, data: Data, rows: torch.Tensor) -> dict:
    """Mean KL(pi || beta) against the recorded behaviour distributions, and the share of decisions whose greedy action the behaviour played with probability below `OFF_DATA`, over the rows that recorded a distribution; by behaviour as well."""
    net.eval()
    kls, offs, kept = [], [], []
    with torch.no_grad():
        for begin in range(0, int(rows.shape[0]), EVAL_BATCH):
            index = rows[begin:begin + EVAL_BATCH]
            beta = data.behaviour[index]
            known = ~torch.isnan(beta).any(dim=-1)
            if data.layer == OPERATIONAL:
                played = data.action[index]
                one, two_played = heads(net, data, index, region=played)
                beta2 = data.second_behaviour[index]
                known = known & ~torch.isnan(beta2).any(dim=-1) & (played >= 0)
            else:
                one, _ = heads(net, data, index)
            legal = data.mask[index] > 0
            kl = _kl(one, beta, legal)
            greedy = one.argmax(dim=-1)
            off = torch.nan_to_num(beta, nan=0.0).gather(-1, greedy.unsqueeze(-1)).squeeze(-1) < OFF_DATA
            if data.layer == OPERATIONAL:
                plan_legal = _plan_row(data, index, played) > 0
                kl = kl + _kl(two_played, beta2, plan_legal)
                _, two_greedy = heads(net, data, index, region=greedy)
                plan = two_greedy.argmax(dim=-1)
                plan_beta = torch.nan_to_num(beta2, nan=0.0).gather(-1, plan.unsqueeze(-1)).squeeze(-1)
                off = off | ((greedy == played) & (plan_beta < OFF_DATA))
            kls.append(kl[known])
            offs.append(off[known])
            kept.append(index[known])
    if not kept or not sum(int(k.shape[0]) for k in kept):
        return {"count": 0, "kl": None, "off_data_share": None, "untrusted": None}
    kl_all, off_all = torch.cat(kls).cpu().numpy(), torch.cat(offs).cpu().numpy()
    names = data.behaviours[torch.cat(kept).cpu().numpy()]
    share = float(off_all.mean())
    by = {name: {"count": int((names == name).sum()), "kl": round(float(kl_all[names == name].mean()), 4),
                 "off_data_share": round(float(off_all[names == name].mean()), 4)} for name in sorted(set(names.tolist()))}
    return {"count": int(len(kl_all)), "kl": round(float(kl_all.mean()), 4), "off_data_share": round(share, 4),
            "untrusted": share > UNTRUSTED_SHARE, "by_behaviour": by}


def return_alignment(net: nn.Module, data: Data, rows: torch.Tensor, top: float = TOP) -> dict:
    """How much more often the policy's greedy action is the one played on the trajectories in the top `top` share of their behaviour policy's returns than on those in the bottom share, over `rows`; per behaviour, and pooled over the behaviours with both ends weighted by their decisions."""
    net.eval()
    rows_h = rows[data.action[rows] >= 0].cpu().numpy()
    ends = {"top": _top_rows(data, rows_h, top), "bottom": _top_rows(data, rows_h, top, bottom=True)}
    agreed = {}
    with torch.no_grad():
        for end, chosen in ends.items():
            index_all = torch.as_tensor(chosen, dtype=torch.long, device=data.label.device)
            parts = []
            for begin in range(0, len(chosen), EVAL_BATCH):
                index = index_all[begin:begin + EVAL_BATCH]
                played = data.action[index]
                one, two = heads(net, data, index, region=played)
                same = one.argmax(dim=-1) == played
                if two is not None:
                    same = same & (two.argmax(dim=-1) == data.second[index])
                parts.append(same)
            agreed[end] = torch.cat(parts).cpu().numpy() if parts else np.zeros(0, dtype=bool)
    by, weighted, weights = {}, 0.0, 0
    for name in sorted(set(data.behaviours[rows_h].tolist())):
        upper = agreed["top"][data.behaviours[ends["top"]] == name]
        lower = agreed["bottom"][data.behaviours[ends["bottom"]] == name]
        if not len(upper) or not len(lower):
            continue
        gap = float(upper.mean()) - float(lower.mean())
        by[name] = {"top": int(len(upper)), "bottom": int(len(lower)), "top_agreement": round(float(upper.mean()), 4),
                    "bottom_agreement": round(float(lower.mean()), 4), "gap": round(gap, 4)}
        weighted += gap * (len(upper) + len(lower))
        weights += len(upper) + len(lower)
    return {"top": top, "count": weights, "gap": round(weighted / weights, 4) if weights else None, "by_behaviour": by}


def _kl(logits: torch.Tensor, beta: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    log_pi = torch.log_softmax(logits, dim=-1)
    pi = log_pi.exp()
    log_beta = torch.log(torch.nan_to_num(beta, nan=0.0).clamp(min=1e-8))
    terms = torch.where(legal & (pi > 0), pi * (log_pi - log_beta), torch.zeros_like(pi))
    return terms.sum(dim=-1)


# The runner.

@dataclass
class Clock:
    device: torch.device
    started: float = field(default_factory=time.time)

    def lap(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.time()
        taken, self.started = now - self.started, now
        return taken


def _peak(device: torch.device) -> float:
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == "cuda" else 0.0


def _host_peak() -> float:
    """The process's peak resident memory in MB, nought where the platform does not report it."""
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except (ImportError, OSError):
        return 0.0


def _fresh(settings: Settings) -> nn.Module:
    """A new network of the settings' layer and kind; with `init_flat`, a residual tactical set network whose flat part is the flat model file's network."""
    config = {name: value for name, value in (("width", settings.width), ("depth", settings.depth)) if value is not None}
    if not settings.init_flat:
        return models.build(settings.layer, settings.net, config)
    if settings.layer != TACTICAL or settings.net != "set":
        raise ValueError("--init-flat starts the flat part of a tactical set network, so it needs --layer tactics and --net set")
    flat = models.load(settings.init_flat, settings.layer, kind="flat")
    net = models.build(settings.layer, "set", {**config, "residual": True, "flat_width": flat.config["width"]})
    net.flat.load_state_dict(flat.state_dict())
    log.info("the residual network's flat part starts from %s", settings.init_flat)
    return net


def _optimiser(net_or_params, settings: Settings, kind: str, batch: int, rows: int, total_steps: Optional[int] = None,
               share: float = 1.0):
    """Adam at `share` of the batch-scaled rate, warmed up linearly over `WARMUP_STEPS` steps or one epoch of `rows`, whichever is shorter.

    With `total_steps` and the cosine decay, the rate then falls along a half cosine from the full rate at the end of the warm-up to `DECAY_FLOOR` of it at `total_steps`, and stays there; without them it holds the full rate.
    """
    rate = share * (settings.learning_rate or BASE_RATE[kind] * math.sqrt(batch / RATE_BATCH))
    params = list(net_or_params.parameters()) if isinstance(net_or_params, nn.Module) else list(net_or_params)
    optimiser = torch.optim.Adam(params, lr=rate)
    warmup = max(1, min(WARMUP_STEPS, math.ceil(rows / batch)))
    decaying = total_steps is not None and settings.decay == "cosine"

    def multiplier(step: int) -> float:
        if step < warmup or not decaying:
            return min(1.0, (step + 1) / warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return DECAY_FLOOR + (1.0 - DECAY_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, multiplier)
    return optimiser, schedule, rate


def _step(optimiser, schedule, loss: torch.Tensor, params) -> None:
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimiser.step()
    schedule.step()


def run(settings: Settings) -> dict:
    """Runs one offline method and returns its report, which is also written to `settings.report` or under the reports directory."""
    if settings.method not in METHODS:
        raise ValueError(f"no offline method named {settings.method!r}: expected one of {', '.join(METHODS)}")
    if not settings.sources:
        raise ValueError("name at least one recorded run with --dataset")
    if not 0.0 < settings.fraction <= 1.0:
        raise ValueError(f"a fraction of the episodes is above nought and at most one, not {settings.fraction}")
    if settings.decay not in DECAYS:
        raise ValueError(f"no learning-rate decay named {settings.decay!r}: expected one of {', '.join(DECAYS)}")
    if settings.load and settings.init_flat:
        raise ValueError("--init-flat starts a new network, so it does not go with --load")
    device = device_of(settings.device)
    models.limit_card(device, settings.card_share)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(settings.seed)
    batch = settings.batch or (GPU_BATCH if device.type == "cuda" else CPU_BATCH)

    teacher = models.load(settings.teacher, settings.layer, device) if settings.method == "distill" else None
    if settings.method == "distill" and teacher is None:
        raise ValueError("distillation needs --teacher")
    evaluated = None
    if settings.method == "fqe":
        if not settings.policy:
            raise ValueError("fqe evaluates the policy named by --policy")
        evaluated = models.load(settings.policy, settings.layer, device)
    if settings.load:
        net = models.load(settings.load, settings.layer, device)
        settings.net = models.kind_of(net)
    else:
        net = _fresh(settings).to(device)
    kinds = {models.kind_of(n) for n in (net, teacher, evaluated) if n is not None}
    with_sets = settings.layer == TACTICAL and "set" in kinds
    clock = Clock(device)
    # Every method reads the rewards, since every report carries `return_alignment`.
    data = load_data(settings, device, with_sets, with_rewards=True)
    loading = clock.lap()
    log.info("%d decision(s) on %s in %.1f s (%d held out)", len(data), device, loading, int(data.held.sum()))

    report = {"layer": settings.layer, "method": settings.method, "net": settings.net, "device": str(device),
              "batch": batch, "datasets": [f"{p}:{w:g}" for p, w in settings.sources], "fraction": settings.fraction,
              "decisions": len(data), "held_out": int(data.held.sum()), "loading_seconds": round(loading, 1),
              "config": dict(net.config), "parameters": sum(p.numel() for p in net.parameters()), "epochs": []}
    if settings.init_flat:
        report["init_flat"] = settings.init_flat
    if settings.method in ("bc", "distill", "topbc"):
        net = _imitate(settings, data, net, teacher, batch, report)
        produced = net
    elif settings.method in ADVANTAGE_METHODS:
        produced = _advantage_learning(settings, data, net, batch, report)
    else:
        produced = _evaluate(settings, data, net, evaluated, batch, report)
    held_rows = torch.nonzero(data.held).squeeze(-1)
    report["behaviour_distance"] = behaviour_distance(produced, data, held_rows)
    report["return_alignment"] = return_alignment(produced, data, held_rows, settings.top)
    report["terms"] = settings.terms if settings.terms is not None else "as recorded"
    report["peak_gpu_mb"] = _peak(device)
    report["peak_host_mb"] = _host_peak()
    if settings.save and settings.method != "fqe":
        models.save(produced, settings.save, extra={"reward_scale": data.reward_scale} if settings.method in PLAYED_METHODS else None)
        report["saved"] = settings.save
        log.info("saved the %s network to %s", models.kind_of(produced), settings.save)
    path = settings.report or os.path.join(paths.reports(), "offline", f"{paths.stamp()}.json")
    with paths.replacing(path) as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    report["report"] = path
    log.info("report written to %s", path)
    return report


def _split(data: Data, rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    held = data.held[rows]
    learning, checking = rows[~held], rows[held]
    if not len(checking) or not len(learning):
        log.warning("only one side of the split has any episode in it, so the held-out figures are taken on the fitted decisions")
        return rows, rows
    return learning, checking


# Imitation: bc, distill, topbc.

def _teacher_targets(teacher: nn.Module, data: Data) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """The teacher's distributions on every decision: over the first head, over the plans at the label's region, and its argmax over the first head."""
    first = torch.zeros_like(data.mask, dtype=torch.float32)
    second = None if data.table is None else torch.zeros(len(data), OPERATIONAL_PLANS, device=data.mask.device)
    every = torch.arange(len(data), device=data.mask.device)
    with torch.no_grad():
        for begin in range(0, len(data), EVAL_BATCH):
            index = every[begin:begin + EVAL_BATCH]
            one, two = heads(teacher, data, index, region=data.label[index].clamp(min=0))
            first[index] = torch.softmax(one, dim=-1)
            if two is not None:
                second[index] = torch.softmax(two, dim=-1)
    return first, second, first.argmax(dim=-1)


def _top_rows(data: Data, rows: np.ndarray, top: float, bottom: bool = False) -> np.ndarray:
    """The rows of trajectories whose return at their first decision is in the top `top` share of their behaviour policy's trajectories, or with `bottom` in the bottom share."""
    starts = data.start.cpu().numpy()
    first_rows = rows[starts[rows]]
    trajectory_return = dict(zip(data.trajectory[first_rows].tolist(), data.returns[first_rows].tolist()))
    trajectory_name = dict(zip(data.trajectory[first_rows].tolist(), data.behaviours[first_rows].tolist()))
    kept_trajectories = set()
    for name in set(trajectory_name.values()):
        mine = [t for t, n in trajectory_name.items() if n == name]
        values = np.asarray([trajectory_return[t] for t in mine])
        if bottom:
            ceiling = np.quantile(values, top)
            kept_trajectories.update(t for t, v in zip(mine, values) if v <= ceiling)
        else:
            floor = np.quantile(values, 1.0 - top)
            kept_trajectories.update(t for t, v in zip(mine, values) if v >= floor)
    keep = np.asarray([t in kept_trajectories for t in data.trajectory[rows].tolist()], dtype=bool)
    return rows[keep]


def _clone_loss(net: nn.Module, data: Data, index: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
                softs: Optional[torch.Tensor], second_softs: Optional[torch.Tensor], smoothing: float) -> torch.Tensor:
    """The clone loss on the decisions `index` against `first` (with the distributions `softs` where written), and for the operational layer against the plans `second` at that region, weighted by each decision's weight."""
    target = first[index]
    one, two = heads(net, data, index, region=target)
    weights = data.weight[index]
    soft, soft_rows = _distribution(softs, index)
    loss = _cross_entropy(one, target, data.mask[index].float(), smoothing, weights, soft, soft_rows)
    if two is not None:
        soft2, soft_rows2 = _distribution(second_softs, index)
        loss = loss + _cross_entropy(two, second[index], _plan_row(data, index, target).float(), smoothing, weights,
                                     soft2, soft_rows2)
    return loss


def _imitate(settings: Settings, data: Data, net: nn.Module, teacher: Optional[nn.Module], batch: int,
             report: dict) -> nn.Module:
    device = data.label.device
    played = settings.method == "topbc"
    first = data.action if played else data.label
    second = data.second if played else data.second_label
    usable = (first >= 0).cpu().numpy()
    rows = np.flatnonzero(usable)
    if played:
        rows = _top_rows(data, rows, settings.top)
        report["top"] = settings.top
    rows_t = torch.as_tensor(rows, device=device)
    learning, checking = _split(data, rows_t)
    softs, second_softs, teacher_choice = (None, None, None) if played else (data.soft, data.second_soft, None)
    if teacher is not None:
        softs, second_softs, teacher_choice = _teacher_targets(teacher, data)
        report["teacher"] = settings.teacher
    log.info("%s: fitting %d decision(s), %d held out", settings.method, len(learning), len(checking))

    # The value and action-value heads take no part in imitation.
    spared = {id(p) for p in value_parameters(net)}
    if settings.init_flat and settings.method == "bc":
        # A flat part copied from a flat model already fits the judge's labels, so bc fits only the tokens' part.
        spared |= {id(p) for p in net.flat.parameters()}
        report["held_flat"] = True
    moving = [p for p in actor_critic_parameters(net) if id(p) not in spared]
    total_steps = settings.epochs * math.ceil(len(learning) / batch)
    optimiser, schedule, rate = _optimiser(moving, settings, models.kind_of(net), batch, len(learning), total_steps)
    report["learning_rate"] = rate
    report["decay"] = settings.decay
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    best, best_state, stale = float("inf"), None, 0
    clock = Clock(device)
    for epoch in range(1, settings.epochs + 1):
        net.train()
        order = learning[torch.randperm(len(learning), generator=generator, device=device)]
        total = steps = 0
        losses = []
        for begin in range(0, len(order), batch):
            index = order[begin:begin + batch]
            loss = _clone_loss(net, data, index, first, second, softs, second_softs, settings.smoothing)
            _step(optimiser, schedule, loss, moving)
            losses.append(loss.detach())
            steps += 1
        total = float(torch.stack(losses).mean()) if losses else 0.0
        seconds = clock.lap()
        check = measure(net, data, checking, first, second, settings.smoothing, softs, second_softs, "held out",
                        teacher_choice)
        clock.lap()
        entry = {"epoch": epoch, "train_loss": round(total, 5), "held_loss": check["loss"],
                 "held_accuracy": check["accuracy"], "seconds": round(seconds, 2),
                 "decisions_per_second": round(len(learning) / max(seconds, 1e-9)), "peak_gpu_mb": _peak(device)}
        for key in ("second_accuracy", "means_accuracy", "carried_accuracy", "teacher_agreement", "entropy"):
            if key in check:
                entry[key] = check[key]
        report["epochs"].append(entry)
        log.info("epoch %2d: fitting %.4f, held out %.4f, accuracy %.4f%s, %.1f s (%.0f decisions/s)%s", epoch, total,
                 check["loss"], check["accuracy"],
                 f" and {check['second_accuracy']:.4f}" if "second_accuracy" in check else "", seconds,
                 entry["decisions_per_second"], "" if check["loss"] < best else "   (no better)")
        if check["loss"] < best:
            best, stale = check["loss"], 0
            best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    report["training"] = measure(net, data, learning, first, second, settings.smoothing, softs, second_softs,
                                 "fitted", teacher_choice)
    report["validation"] = measure(net, data, checking, first, second, settings.smoothing, softs, second_softs,
                                   "held out", teacher_choice)
    if not played:
        report["validation_labels"] = report["validation"]
    else:
        # Judged against the labels on every held-out decision, not only on the top trajectories it was fitted to.
        labelled = torch.nonzero(data.held & (data.label >= 0)).squeeze(-1)
        report["validation_labels"] = measure(net, data, labelled, data.label, data.second_label, settings.smoothing,
                                              name="held out, labels")
    return net


# Action values: iql, cql, fqe.

def _critic(net: nn.Module, data: Data, index: torch.Tensor):
    """The critic's action values over every action and its value, in single precision: (2, rows, actions) and (rows,) for a one-part layer; for the operational layer the region part (2, rows, regions), the plan part for every region (2, rows, regions, plans) and the value."""
    x = _input(net, data, index)
    with _autocast(data.label.device):
        if data.layer != OPERATIONAL:
            q, v = net.critic(x)
            return q.float(), None, v.float()
        hidden = net.hidden(x, _slots(data, index))
        return (net.critic_regions(hidden).float(), net.critic_plans(hidden).float(),
                net.value(hidden).squeeze(-1).float())


def _taken(q_first: torch.Tensor, q_second: Optional[torch.Tensor], first: torch.Tensor,
           second: torch.Tensor) -> torch.Tensor:
    """Both action values of the action taken, (2, rows)."""
    rows = torch.arange(first.shape[0], device=first.device)
    value = q_first[:, rows, first.clamp(min=0)]
    if q_second is not None:
        value = value + q_second[:, rows, first.clamp(min=0), second.clamp(min=0)]
    return value


def _legal(data: Data, index: torch.Tensor):
    """The legal first actions (rows, actions) and for the operational layer the legal plans of every region (rows, regions, plans)."""
    first = data.mask[index] > 0
    second = None if data.table is None else data.table[index].reshape(-1, OPERATIONAL_REGIONS, OPERATIONAL_PLANS) > 0
    return first, second


def _joint(q_first: torch.Tensor, q_second: Optional[torch.Tensor], legal_first, legal_second, reduce) -> torch.Tensor:
    """A reduction (max or log-sum-exp) of the action values over the legal actions, for the operational layer over the legal pairs of region and plan; (2, rows)."""
    if q_second is not None:
        inner = reduce(q_second.masked_fill(~legal_second.unsqueeze(0), MASKED), -1)
        q_first = q_first + torch.where(legal_second.any(-1).unsqueeze(0), inner, torch.zeros_like(inner))
    return reduce(q_first.masked_fill(~legal_first.unsqueeze(0), MASKED), -1)


def _expected(policy: nn.Module, data: Data, index: torch.Tensor, q_first, q_second) -> torch.Tensor:
    """The policy's expectation of each action-value head on these decisions, (2, rows)."""
    x = _input(policy, data, index)
    legal_first, legal_second = _legal(data, index)
    with _autocast(data.label.device):
        if data.layer != OPERATIONAL:
            logits, _ = policy(x, data.mask[index].float())
            pi = torch.softmax(logits.float(), -1)
            return (pi.unsqueeze(0) * q_first).sum(-1)
        hidden = policy.hidden(x, _slots(data, index))
        regions = policy.regions(hidden, data.mask[index].float()).float()
        plans = policy.plans_all(hidden).float().masked_fill(~legal_second, MASKED)
    pi_r = torch.softmax(regions, -1)
    pi_p = torch.softmax(plans, -1)
    inner = (pi_p.unsqueeze(0) * q_second).sum(-1)
    return (pi_r.unsqueeze(0) * (q_first + inner)).sum(-1)


def _targets(data: Data, index: torch.Tensor, after: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`r + gamma^periods * after(s')` and which rows have a target: a decision with a next one, or the last of a trajectory that ended (whose next term is nought); the last of a cut trajectory has none."""
    following = data.next[index]
    has_next = following >= 0
    target = data.reward[index] + data.gamma[index] * torch.where(has_next, after, torch.zeros_like(after))
    return target, has_next | data.done[index]


def _next_values(net: nn.Module, data: Data, index: torch.Tensor, reduce: str, policy: Optional[nn.Module] = None):
    """A value of each decision's next decision, nought where there is none: the critic's value (`v`), the largest legal smaller target action value (`max`), or the policy's expectation of the target action values (`expect`)."""
    following = data.next[index]
    has_next = following >= 0
    out = torch.zeros(index.shape[0], device=index.device)
    if not bool(has_next.any()):
        return out
    nxt = following[has_next]
    with torch.no_grad():
        q_first, q_second, v = _critic(net, data, nxt)
        if reduce == "v":
            value = v
        elif reduce == "max":
            legal_first, legal_second = _legal(data, nxt)
            value = _joint(q_first, q_second, legal_first, legal_second, lambda t, d: t.max(d).values).min(0).values
        else:
            value = _expected(policy, data, nxt, q_first, q_second).mean(0)
    out[has_next] = value
    return out


def _polyak(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for slow, fast in zip(target.parameters(), source.parameters()):
            slow.mul_(1.0 - tau).add_(fast.detach(), alpha=tau)


def _expectile(difference: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(difference > 0, torch.full_like(difference, tau), torch.full_like(difference, 1.0 - tau))
    return weight * difference ** 2


class AdvantageLearner:
    """IQL, AWR or CQL (`settings.method`): a critic of the same kind as the actor (a second instance, its own trunk) with a Polyak-averaged target, and the actor extracted by advantage-weighted regression on the actions played, held to the judge's labels by their clone loss at the weight `settings.judge`.

    IQL and CQL take the advantage from the critic's action values; AWR fits only the critic's value, to the recorded return bootstrapped from the target's value where a trajectory was cut, and takes the return less that value. `data` may be replaced between updates; learning goes on from the same networks and optimiser state.
    """

    def __init__(self, settings: Settings, data: Data, actor: nn.Module, batch: int, rows: int) -> None:
        device = data.label.device
        self.settings = settings
        self.data = data
        self.actor = actor
        self.batch = batch
        self.critic = models.build(settings.layer, models.kind_of(actor), dict(actor.config)).to(device)
        self.target = copy.deepcopy(self.critic)
        for p in self.target.parameters():
            p.requires_grad_(False)
        kind = models.kind_of(actor)
        # An actor read from a file is already fitted, so it moves on at the rate imitation's decay ended at.
        share = DECAY_FLOOR if settings.load else 1.0
        self.actor_optimiser, self.actor_schedule, self.actor_rate = _optimiser(actor_critic_parameters(actor), settings,
                                                                                kind, batch, rows, share=share)
        self.critic_optimiser, self.critic_schedule, self.rate = _optimiser(self.critic, settings, kind, batch, rows)
        self.generator = torch.Generator(device=device).manual_seed(settings.seed)
        self.updates = 0

    def _critic_step(self, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One gradient step of the critic on the decisions `index`; the q and v losses, each decision's advantage, and which decisions the actor's advantage term counts."""
        settings, data, critic, target = self.settings, self.data, self.critic, self.target
        first, second = data.action[index], data.second[index]
        if settings.method == "awr":
            _, _, v = _critic(critic, data, index)
            tail = data.tail[index]
            bootstrapped = torch.zeros_like(v)
            if bool((tail >= 0).any()):
                with torch.no_grad():
                    bootstrapped[tail >= 0] = _critic(target, data, tail[tail >= 0])[2]
            goal = data.mc[index] + data.tail_gamma[index] * bootstrapped
            valid = (data.next[index] >= 0) | data.done[index]
            valid_f = valid.float()
            v_loss = (((goal - v) ** 2) * valid_f).sum() / valid_f.sum().clamp(min=1.0)
            q_loss = torch.zeros_like(v_loss)
            _step(self.critic_optimiser, self.critic_schedule, v_loss, list(critic.parameters()))
            _polyak(target, critic, settings.tau)
            return q_loss, v_loss, goal - v.detach(), valid
        q_first, q_second, v = _critic(critic, data, index)
        with torch.no_grad():
            t_first, t_second, _ = _critic(target, data, index)
            taken_target = _taken(t_first, t_second, first, second).min(0).values
        after = _next_values(critic if settings.method == "iql" else target, data, index,
                             "v" if settings.method == "iql" else "max")
        goal, valid = _targets(data, index, after)
        taken = _taken(q_first, q_second, first, second)
        valid_f = valid.float()
        q_loss = (((taken - goal.unsqueeze(0)) ** 2).sum(0) * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        if settings.method == "cql":
            legal_first, legal_second = _legal(data, index)
            spread = _joint(q_first, q_second, legal_first, legal_second, torch.logsumexp)
            q_loss = q_loss + settings.alpha * (spread - taken).sum(0).mean()
        v_loss = _expectile(taken_target - v, settings.expectile).mean()
        _step(self.critic_optimiser, self.critic_schedule, q_loss + v_loss, list(critic.parameters()))
        _polyak(target, critic, settings.tau)
        return q_loss, v_loss, taken_target - v.detach(), torch.ones_like(valid)

    def update(self, index: torch.Tensor, moving: bool = True) -> torch.Tensor:
        """One gradient step of the critic, and unless `moving` is false of the actor, on the decisions `index`; the q, v, advantage-weighted and judge losses and the mean weight, on the device."""
        settings, data, actor = self.settings, self.data, self.actor
        first, second = data.action[index], data.second[index]
        q_loss, v_loss, advantage, counted = self._critic_step(index)
        weights = torch.exp(inverse_temperature(settings) * advantage).clamp(max=WEIGHT_CAP)
        if not moving:
            self.updates += 1
            nought = q_loss.new_zeros(())
            return torch.stack([q_loss.detach(), v_loss.detach(), nought, nought,
                                weights[counted].mean().detach() if bool(counted.any()) else nought])
        one, two = heads(actor, data, index, region=first)
        log_prob = _log_prob(one, first)
        if two is not None:
            log_prob = log_prob + _log_prob(two, second)
        # The advantage weights are normalised over the batch, so the term weighs as much as a clone loss and `judge` is the ratio of the two.
        mass = weights * data.weight[index] * counted.float()
        weighted_loss = -(mass * log_prob).sum() / mass.sum().clamp(min=1e-8)
        judge_loss = torch.zeros_like(weighted_loss)
        labelled = index[data.label[index] >= 0]
        if settings.judge > 0 and len(labelled):
            judge_loss = _clone_loss(actor, data, labelled, data.label, data.second_label, data.soft, data.second_soft,
                                     settings.smoothing)
        _step(self.actor_optimiser, self.actor_schedule, weighted_loss + settings.judge * judge_loss,
              list(actor.parameters()))
        self.updates += 1
        return torch.stack([q_loss.detach(), v_loss.detach(), weighted_loss.detach(), judge_loss.detach(),
                            weights[counted].mean().detach() if bool(counted.any()) else weights.new_zeros(())])


def _advantage_learning(settings: Settings, data: Data, actor: nn.Module, batch: int, report: dict) -> nn.Module:
    """`AdvantageLearner` over `settings.critic_epochs` passes of the training side that fit only the critic and then `settings.epochs` that fit both, measured on the held-out side after each."""
    device = data.label.device
    rows = torch.nonzero(data.action >= 0).squeeze(-1)
    learning, checking = _split(data, rows)
    learner = AdvantageLearner(settings, data, actor, batch, len(learning))
    report.update(learning_rate=learner.rate, actor_learning_rate=learner.actor_rate, tau=settings.tau,
                  beta=inverse_temperature(settings), weight_cap=WEIGHT_CAP,
                  judge=settings.judge, smoothing=settings.smoothing, critic_epochs=settings.critic_epochs,
                  reward_scale=data.reward_scale)
    if settings.method != "awr":
        report["expectile"] = settings.expectile
    if settings.method == "cql":
        report["alpha"] = settings.alpha
    clock = Clock(device)
    for epoch in range(1, settings.critic_epochs + settings.epochs + 1):
        moving = epoch > settings.critic_epochs
        actor.train()
        learner.critic.train()
        order = learning[torch.randperm(len(learning), generator=learner.generator, device=device)]
        sums = torch.zeros(5, device=device)
        steps = 0
        for begin in range(0, len(order), batch):
            sums += learner.update(order[begin:begin + batch], moving)
            steps += 1
        seconds = clock.lap()
        means = (sums / max(1, steps)).tolist()
        check = measure(actor, data, checking, data.action, data.second, 0.0, name="held out, played")
        labelled = checking[data.label[checking] >= 0]
        labels = measure(actor, data, labelled, data.label, data.second_label, 0.0, name="held out, labels")
        clock.lap()
        entry = {"epoch": epoch, "critic_only": not moving, "q_loss": round(means[0], 5), "v_loss": round(means[1], 5),
                 "actor_loss": round(means[2], 5), "judge_loss": round(means[3], 5), "mean_weight": round(means[4], 4),
                 "held_played_accuracy": check["accuracy"], "held_label_accuracy": labels.get("accuracy"),
                 "held_label_loss": labels.get("loss"), "seconds": round(seconds, 2),
                 "decisions_per_second": round(len(learning) / max(seconds, 1e-9)), "peak_gpu_mb": _peak(device)}
        report["epochs"].append(entry)
        log.info("epoch %2d%s: q %.4f, v %.4f, actor %.4f, judge %.4f, weight %.3f, held out accuracy %.4f on played and %.4f on labels, %.1f s",
                 epoch, "" if moving else " (critic only)", means[0], means[1], means[2], means[3], means[4],
                 check["accuracy"], labels.get("accuracy", 0.0), seconds)
    report["validation"] = measure(actor, data, checking, data.action, data.second, 0.0, name="held out, played")
    report["validation_labels"] = measure(actor, data, checking[data.label[checking] >= 0], data.label,
                                          data.second_label, 0.0, name="held out, labels")
    return actor


def watched_runs(settings: Settings) -> List[Tuple[str, float]]:
    """The named datasets and every run under a watched directory (or the watched directory itself, when it is a run) of the layer and the encoding in force, complete or still growing."""
    from ..control.policy.encoding import fingerprint

    found = list(settings.sources)
    named = {os.path.abspath(path) for path, _ in found}
    for directory in settings.watch:
        candidates = [directory] if os.path.exists(os.path.join(directory, "run.json")) else sorted(
            os.path.dirname(path) for path in glob.glob(os.path.join(directory, "*", "run.json")))
        for run in candidates:
            try:
                with open(os.path.join(run, "run.json"), encoding="utf-8") as handle:
                    header = json.load(handle)
            except (OSError, ValueError):
                continue
            if (header.get("layer") == settings.layer and header.get("fingerprint") == fingerprint(settings.layer)
                    and os.path.abspath(run) not in named):
                named.add(os.path.abspath(run))
                found.append((run, 1.0))
    return found


def publish(actor: nn.Module, settings: Settings, version: int, extra: dict, sidecar: dict) -> None:
    """Writes the actor to `settings.save` as `version` and then its sidecar `<save>.json`, each replaced in one step, so a reader that sees a sidecar's version finds at least that version in the file."""
    from .reload import sidecar_path

    models.save(actor, settings.save, version=version, extra=extra)
    with paths.replacing(sidecar_path(settings.save)) as handle:
        json.dump(dict(sidecar, version=version, time=time.strftime("%Y-%m-%dT%H:%M:%S")), handle, indent=1, sort_keys=True)


def follow(settings: Settings, stop: Optional[threading.Event] = None) -> dict:
    """The learner of an actor/learner split: IQL, AWR or CQL without epochs over the named datasets and the watched runs (the first `critic_epochs` passes' worth of updates fitting only the critic), publishing the actor once before learning and then every `publish_every` updates with a sidecar, and appending to its replay buffer (`Buffer`, capped at `settings.buffer` decisions) only the shards the runs gained. The reward scale is fixed at the first read. Runs until `stop` is set or `updates` have been made, publishes once more on the way out, and writes its report to `settings.report` or under the reports directory."""
    if settings.method not in ADVANTAGE_METHODS:
        raise ValueError("--follow learns with --method iql, awr or cql")
    if not settings.save:
        raise ValueError("--follow publishes to --save, which was not given")
    if settings.load and settings.init_flat:
        raise ValueError("--init-flat starts a new network, so it does not go with --load")
    stop = stop or threading.Event()
    device = device_of(settings.device)
    models.limit_card(device, follow_card_share(settings))
    torch.manual_seed(settings.seed)
    batch = settings.batch or (GPU_BATCH if device.type == "cuda" else CPU_BATCH)
    actor = models.load(settings.load, settings.layer, device) if settings.load else _fresh(settings).to(device)
    with_sets = settings.layer == TACTICAL and models.kind_of(actor) == "set"
    sources = watched_runs(settings)
    fresh = listing(sources)
    while not fresh and not stop.is_set():
        log.info("waiting for a shard in %s", ", ".join(path for path, _ in sources) or "nothing named")
        stop.wait(settings.rescan_seconds)
        sources = watched_runs(settings)
        fresh = listing(sources)

    def training(data: Data) -> torch.Tensor:
        rows = torch.nonzero(data.action >= 0).squeeze(-1)
        return rows[~data.held[rows]]

    buffer = Buffer(settings, device, with_sets, with_rewards=True, cap=settings.buffer,
                    kept=[path for path, _ in settings.sources])
    clock = Clock(device)
    buffer.append(fresh)
    known = {shard for _, _, shard in fresh}
    data = buffer.data()
    loading = clock.lap()
    log.info("read %d decision(s) from %d shard(s) in %.2f s", len(data), len(known), loading)
    scale = data.reward_scale
    rows = training(data)
    learner = AdvantageLearner(settings, data, actor, batch, len(rows))
    # The critic alone takes the first passes' worth of updates over the decisions read at the start.
    warming = settings.critic_epochs * math.ceil(len(rows) / batch)
    extra = {"reward_scale": scale, "method": settings.method}
    report = {"layer": settings.layer, "method": settings.method, "net": models.kind_of(actor), "device": str(device),
              "batch": batch, "reward_scale": scale, "critic_updates": warming, "published": [],
              "loading_seconds": round(loading, 3), "appends": []}
    scanned = time.monotonic()
    version = int(getattr(actor, "version", 0))

    def sidecar() -> dict:
        return {"updates": learner.updates, "decisions": len(learner.data), "runs": [path for path, _ in sources],
                "shards": len(known)}

    def out() -> None:
        nonlocal version
        version += learner.updates - published[0]
        published[0] = learner.updates
        publish(actor, settings, version, extra, sidecar())
        report["published"].append(version)
        log.info("published version %d after %d update(s) over %d decision(s)", version, learner.updates, len(learner.data))

    published = [0]
    # Actors started beside the learner play the published file, so it is there before the first update.
    out()
    actor.train()
    learner.critic.train()
    learning_started = time.monotonic()
    while not stop.is_set() and (settings.updates is None or learner.updates < settings.updates):
        if len(rows):
            pick = torch.randint(len(rows), (min(batch, len(rows)),), generator=learner.generator, device=rows.device)
            learner.update(rows[pick], learner.updates >= warming)
        else:
            stop.wait(min(1.0, settings.rescan_seconds))
        if learner.updates and learner.updates % settings.publish_every == 0 and learner.updates != published[0]:
            out()
        if time.monotonic() - scanned >= settings.rescan_seconds:
            scanned = time.monotonic()
            now = watched_runs(settings)
            fresh = [entry for entry in listing(now) if entry[2] not in known]
            if fresh:
                sources = now
                clock.lap()
                # The learner lets go of its views of the storage, so that an append which regrows the storage returns the old one.
                learner.data = data = rows = None
                added = buffer.append(fresh)
                known.update(shard for _, _, shard in fresh)
                learner.data = data = buffer.data()
                rows = training(data)
                seconds = clock.lap()
                report["appends"].append({"shards": len(fresh), "decisions": added, "seconds": round(seconds, 3)})
                log.info("appended %d decision(s) from %d new shard(s) in %.2f s; the buffer holds %d decision(s) from %d shard(s)",
                         added, len(fresh), seconds, len(data), len(buffer.segments))
    if learner.updates != published[0]:
        out()
    learning = time.monotonic() - learning_started
    # The wall clock per update counts the appends and publishes between updates, since that is what bounds how many updates a run of a given length makes.
    report.update(updates=learner.updates, decisions=len(learner.data), shards=len(known), learning_seconds=round(learning, 3),
                  seconds_per_update=round(learning / learner.updates, 5) if learner.updates else None)
    report["peak_gpu_mb"] = _peak(device)
    report["peak_host_mb"] = _host_peak()
    path = settings.report or os.path.join(paths.reports(), "offline", f"{paths.stamp()}.json")
    with paths.replacing(path) as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    report["report"] = path
    log.info("report written to %s", path)
    return report


def _evaluate(settings: Settings, data: Data, critic: nn.Module, policy: nn.Module, batch: int, report: dict) -> nn.Module:
    """Fitted Q evaluation of `policy`: the critic's action values regress on `r + gamma^periods * E_{a'~pi} Q_target(s', a')`; the policy's value at the start of every held-out trajectory is then reported, with a bootstrap interval over episodes, in the reward's own units."""
    device = data.label.device
    critic = models.build(settings.layer, models.kind_of(critic), dict(critic.config)).to(device)
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    rows = torch.nonzero(data.action >= 0).squeeze(-1)
    learning, checking = _split(data, rows)
    optimiser, schedule, rate = _optimiser(critic, settings, models.kind_of(critic), batch, len(learning))
    report.update(learning_rate=rate, tau=settings.tau, reward_scale=data.reward_scale, policy=settings.policy)
    generator = torch.Generator(device=device).manual_seed(settings.seed)
    clock = Clock(device)
    policy.eval()
    for epoch in range(1, settings.epochs + 1):
        critic.train()
        order = learning[torch.randperm(len(learning), generator=generator, device=device)]
        total, steps = 0.0, 0
        for begin in range(0, len(order), batch):
            index = order[begin:begin + batch]
            q_first, q_second, _ = _critic(critic, data, index)
            after = _next_values(target, data, index, "expect", policy)
            goal, valid = _targets(data, index, after)
            taken = _taken(q_first, q_second, data.action[index], data.second[index])
            valid_f = valid.float()
            loss = (((taken - goal.unsqueeze(0)) ** 2).sum(0) * valid_f).sum() / valid_f.sum().clamp(min=1.0)
            _step(optimiser, schedule, loss, list(critic.parameters()))
            _polyak(target, critic, settings.tau)
            total += float(loss.detach())
            steps += 1
        seconds = clock.lap()
        report["epochs"].append({"epoch": epoch, "q_loss": round(total / max(1, steps), 5), "seconds": round(seconds, 2),
                                 "decisions_per_second": round(len(learning) / max(seconds, 1e-9)),
                                 "peak_gpu_mb": _peak(device)})
        log.info("epoch %2d: q %.5f, %.1f s", epoch, total / max(1, steps), seconds)
    report["fqe"] = start_values(critic, policy, data, checking)
    log.info("estimated value at the held-out starts: %s", report["fqe"])
    return policy


def start_values(critic: nn.Module, policy: nn.Module, data: Data, rows: torch.Tensor, resamples: int = BOOTSTRAP,
                 seed: int = 0) -> dict:
    """The policy's estimated value at the first decision of every trajectory among `rows`, as a mean over episodes with a bootstrap interval, in the reward's own units."""
    starts = rows[data.start[rows]]
    if not len(starts):
        return {"count": 0}
    with torch.no_grad():
        values = []
        for begin in range(0, len(starts), EVAL_BATCH):
            index = starts[begin:begin + EVAL_BATCH]
            q_first, q_second, _ = _critic(critic, data, index)
            values.append(_expected(policy, data, index, q_first, q_second).mean(0))
    estimate = (torch.cat(values) / data.reward_scale).cpu().numpy()
    episodes = data.episode[starts].cpu().numpy()
    names = np.unique(episodes)
    per_episode = np.asarray([estimate[episodes == name].mean() for name in names])
    draw = np.random.default_rng(seed)
    means = [per_episode[draw.integers(0, len(per_episode), len(per_episode))].mean() for _ in range(resamples)]
    return {"count": int(len(estimate)), "episodes": int(len(names)), "mean": round(float(per_episode.mean()), 5),
            "low": round(float(np.percentile(means, 2.5)), 5), "high": round(float(np.percentile(means, 97.5)), 5)}
