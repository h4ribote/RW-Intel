"""Fitting a network to what a teacher does, so that a reinforcement run starts from something rather than from noise.

A freshly built policy spends its first tens of thousands of decisions discovering things the rule ladder has known all along, and it discovers them from a reward that is mostly shaping. The material to skip that with is already being produced: every recorded decision carries what the layer's judge would have chosen on that board (`label`) beside what was played, which is a supervised problem of exactly the ordinary shape. Fitting it is cheap, and what comes out is a policy that is already worth measuring, which is what makes the first reinforcement updates informative instead of merely destructive.

Two details are the whole difficulty of doing it properly.

The first is that the teacher is deterministic. The script answers the same board the same way every time, so a fit allowed to be confident is nearly one-hot within a few epochs, and a policy that reaches reinforcement learning with no entropy left has stopped producing the evidence that would ever argue it out of an opinion. Smoothing the labels is the cure and it is not optional here. The mass held back is shared out over the actions that were legal rather than over the whole row, because most of the twenty-four regions do not exist on a given board and their logits are floored to a large negative number: asking the network to put a fiftieth of the label on one of them asks it to raise a logit that the mask floors again on the next pass, and the loss goes somewhere no learning rate survives.

The second is that the held-out part has to be held out by episode. Two decisions a period apart in one fight are nearly the same decision, so a split by row grades the fit on rows it was in effect fitted to. The split is the dataset's (`dataset.fold`), by episode.

The value head is deliberately left out of this. A label carries an action and no return, so there is nothing here to fit a critic against, and it is warmed up at the start of the reinforcement run instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..wire import Deviation, Task
from ..control.policy.encoding import INVESTMENT_CAPACITY, MEANS, OPERATIONAL_REGIONS, SQUAD_SLOTS
from .dataset import VALIDATION_SHARE, Dataset, second_rows, widths
from .net import EconomicNet, OperationalNet, TacticalNet, actor_critic_parameters, entropy, one_hot_slot, value_parameters
from .policy import ECONOMIC, LAYERS, OPERATIONAL, TACTICAL

log = logging.getLogger(__name__)

#: How much of each label is held back from the action the teacher chose and shared out over the ones it was allowed to choose instead. Small, and the difference between a policy that can still be argued with and one that cannot: the script is a function rather than a distribution, so a fit with nothing held back converges to certainty about boards it has seen a handful of.
SMOOTHING = 0.05

#: Passes over the teacher before the fit is given up on, when the held-out part has not stopped improving first.
EPOCHS = 50

#: Rows in one gradient step.
BATCH = 256

#: Larger than the reinforcement run's rate by a factor of three, because fitting a fixed set of labels is a far better conditioned problem than a policy gradient against a target that moves as the policy does.
LEARNING_RATE = 1e-3

#: Epochs the held-out part is allowed not to improve for before the fit is stopped and the best parameters kept.
PATIENCE = 5

#: The source a decision recorded without one came from.
SCRIPT = "script"

#: The salt the episodes a fit on part of the data keeps are drawn under, apart from the split's own so that the part kept does not lean towards or away from the held-out side.
FRACTION_SALT = "fraction"


@dataclass
class Teacher:
    """The decisions a fit is made against, as arrays: the board, the teacher's answer and distribution, what was legal, how much each counts, where it came from, and which side of the split its episode is on."""

    layer: str
    states: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    weights: np.ndarray
    sources: np.ndarray
    held_out: np.ndarray
    seconds: np.ndarray
    second_masks: np.ndarray
    squads: np.ndarray
    #: The teacher's distributions, NaN in a row that has none.
    softs: np.ndarray
    second_softs: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @classmethod
    def of(cls, sources: Sequence[Tuple[Dataset, float]], keep_tainted: bool = False,
           share: float = VALIDATION_SHARE, fraction: float = 1.0) -> "Teacher":
        """The labelled decisions of several datasets of one layer, each with the weight its decisions are multiplied by.

        Below one, `fraction` keeps that share of the training side's episodes, chosen by a digest of their names under a salt of its own (`FRACTION_SALT`), so that a smaller fraction is a subset of every larger one; the held-out side is kept whole, so fits on every fraction are judged on the same episodes.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"a fraction of the episodes is above nought and at most one, not {fraction}")
        if not sources:
            raise ValueError("there are no decisions to fit to")
        layers = {dataset.layer for dataset, _ in sources}
        if len(layers) != 1:
            raise ValueError(f"a fit is of one layer, and these datasets are of {', '.join(sorted(layers))}")
        parts: Dict[str, list] = {}
        for dataset, scale in sources:
            if scale < 0:
                raise ValueError(f"a teacher's weight cannot be negative, not {scale}")
            a = dataset.arrays
            keep = dataset.usable(keep_tainted) & (a["label"] >= 0)
            if fraction < 1.0:
                keep &= dataset.held_out(share) | dataset.held_out(fraction, salt=FRACTION_SALT)
            rows = np.flatnonzero(keep)
            log.info("%d of %d decision(s) from %s carry a teacher's answer and are kept, at weight %g", len(rows),
                     len(dataset), ", ".join(str(run.get("run")) for run in dataset.runs), scale)
            source = np.asarray([str(dataset.meta.get(int(i), {}).get("source", SCRIPT)) for i in rows], dtype=object)
            # The second answer is the teacher's for the region the teacher chose, so it is judged against that region's row of the second mask.
            second_masks = second_rows(a["second_mask"][rows], a["label"][rows].astype(np.int64), dataset.layer)
            for name, values in (("states", a["state"][rows]), ("actions", a["label"][rows].astype(np.int64)),
                                 ("masks", a["mask"][rows].astype(np.float32)),
                                 ("weights", a["weight"][rows].astype(np.float32) * scale), ("sources", source),
                                 ("held_out", dataset.held_out(share)[rows]),
                                 ("seconds", a["second_label"][rows].astype(np.int64)),
                                 ("second_masks", second_masks.astype(np.float32)),
                                 ("squads", a["squad"][rows].astype(np.int64)), ("softs", a["soft"][rows]),
                                 ("second_softs", a["second_soft"][rows])):
                parts.setdefault(name, []).append(values)
        return cls(layer=layers.pop(), **{name: np.concatenate(chunks) for name, chunks in parts.items()})


@dataclass
class Fit:
    """One half of the split, scored once the fitting has stopped."""

    name: str
    count: int
    loss: float
    #: Share of the teacher's choices the network now makes for itself.
    accuracy: float
    #: The same for the second half of a decision made in two parts, and nought for a layer that chooses one thing.
    second_accuracy: float = 0.0
    #: For the operational layer, the share of decisions whose means -walking or which transport- the network takes as the teacher did, and the same over the decisions the teacher sent by transport alone.
    means_accuracy: float = 0.0
    carried_accuracy: float = 0.0
    #: Mean entropy of the distribution the network puts on these boards, in nats, summed over both heads where there are two. A figure near nought means the fit has become a lookup table and the reinforcement run after it will explore nothing.
    entropy: float = 0.0
    #: How often each action came up, once for the teacher and once for the network. Both, because a fit that only ever answers one thing is a failure and a teacher that only ever asks for one thing is not, and the two cannot be told apart from the network's distribution alone.
    teacher_share: List[float] = field(default_factory=list)
    policy_share: List[float] = field(default_factory=list)
    teacher_second_share: List[float] = field(default_factory=list)
    policy_second_share: List[float] = field(default_factory=list)
    #: The same accuracies over the decisions of each source alone, with how many there were. A fit can match the plentiful teacher and not the scarce one, and only this says which.
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "count": self.count, "loss": round(self.loss, 5),
                "accuracy": round(self.accuracy, 4), "second_accuracy": round(self.second_accuracy, 4),
                "means_accuracy": round(self.means_accuracy, 4), "carried_accuracy": round(self.carried_accuracy, 4),
                "entropy": round(self.entropy, 4), "by_source": self.by_source,
                "teacher_share": [round(share, 4) for share in self.teacher_share],
                "policy_share": [round(share, 4) for share in self.policy_share],
                "teacher_second_share": [round(share, 4) for share in self.teacher_second_share],
                "policy_second_share": [round(share, 4) for share in self.policy_second_share]}


@dataclass
class Cloning:
    """What a fit was made of and how it came out."""

    layer: str
    samples: int
    epochs: int
    #: True when the held-out part stopped improving before the epoch budget ran out, which is the ordinary and the healthy way for this to end.
    stopped_early: bool
    training: Fit
    validation: Fit

    def as_dict(self) -> dict:
        return {"layer": self.layer, "samples": self.samples, "epochs": self.epochs,
                "stopped_early": self.stopped_early,
                "training": self.training.as_dict(), "validation": self.validation.as_dict()}


def network(layer: str) -> nn.Module:
    """A fresh network for a layer."""
    if layer == TACTICAL:
        return TacticalNet()
    if layer == OPERATIONAL:
        return OperationalNet()
    if layer == ECONOMIC:
        return EconomicNet()
    raise ValueError(f"no layer named {layer!r}: expected one of {', '.join(LAYERS)}")


def fit(teacher: Teacher, net: Optional[nn.Module] = None, device=None, smoothing: float = SMOOTHING,
        epochs: int = EPOCHS, batch: int = BATCH, learning_rate: float = LEARNING_RATE, patience: int = PATIENCE,
        seed: int = 0) -> Tuple[nn.Module, Cloning]:
    """Fits a network to the teacher's choices and hands it back with the report of how well it fits, judged on the episodes held out."""
    layer = teacher.layer
    if not len(teacher):
        raise ValueError("there are no decisions to fit to")
    if net is None:
        net = network(layer)
    net = net.to(device)
    log.info("fitting the %s layer to %d decision(s): %d feature(s), %d parameter(s)",
             layer, len(teacher), widths(layer)[0], sum(p.numel() for p in net.parameters()))

    held = teacher.held_out
    learning_rows, checking_rows = np.flatnonzero(~held), np.flatnonzero(held)
    if not len(checking_rows) or not len(learning_rows):
        # Too few episodes to have one on each side: the fit is still the right thing to do, but the figure reported for the held-out side is then taken on the rows that were fitted, which is the optimistic one.
        log.warning("only one side of the split has any episode in it, so the held-out figures are taken on the fitted decisions")
        learning_rows = checking_rows = np.arange(len(teacher))
    learnt = _tensors(teacher, learning_rows, device)
    checked = _tensors(teacher, checking_rows, device)
    log.info("%d decision(s) to fit and %d held out by episode", len(learnt), len(checked))

    # The critic is not in the optimiser at all. The teacher has no returns in it, so there is no target to move the value head towards, and leaving it in would let Adam's own state drift it on gradients that are not there.
    spared = {id(parameter) for parameter in value_parameters(net)}
    moving = [parameter for parameter in actor_critic_parameters(net) if id(parameter) not in spared]
    optimiser = torch.optim.Adam(moving, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)

    best = float("inf")
    best_state = None
    stale = 0
    ran = 0
    for epoch in range(1, epochs + 1):
        ran = epoch
        order = torch.randperm(len(learnt), generator=generator).to(device)
        total, taken = 0.0, 0
        for start in range(0, len(learnt), batch):
            loss = _loss(net, layer, learnt.take(order[start:start + batch]), smoothing)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            taken += 1
        with torch.no_grad():
            against = float(_loss(net, layer, checked, smoothing))
        log.info("epoch %2d: fitting %.4f, held out %.4f%s", epoch, total / max(1, taken), against,
                 "" if against < best else "   (no better)")
        if against < best:
            best, stale = against, 0
            # The parameters kept are the ones that scored best on the held-out episodes, not the ones the last epoch happened to leave behind.
            best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    return net, Cloning(layer=layer, samples=len(teacher), epochs=ran, stopped_early=ran < epochs,
                        training=_score(net, layer, learnt, smoothing, "fitted"),
                        validation=_score(net, layer, checked, smoothing, "held out"))


def report(cloning: Cloning) -> None:
    """Says what the fit came out at, in the terms that tell a bad fit from a lopsided teacher.

    The two distributions are printed side by side on purpose. A network that answers one thing whatever it is shown is a failed fit; a teacher that asks for one thing whatever it saw is a fact about the rule ladder, which holds position far more often than it does anything else. The two look identical in the network's own distribution and are told apart only by the teacher's beside it.
    """
    names, second_names = _names(cloning.layer)
    log.info("%s layer cloned from %d decision(s) over %d epoch(s)%s", cloning.layer, cloning.samples,
             cloning.epochs, ", stopped early" if cloning.stopped_early else "")
    for part in (cloning.training, cloning.validation):
        log.info("%-9s n=%-6d loss %.4f  accuracy %.3f%s  entropy %.3f", part.name, part.count,
                 part.loss, part.accuracy,
                 f" and {part.second_accuracy:.3f}" if part.policy_second_share else "",
                 part.entropy)
        if len(part.by_source) > 1:
            for source, figures in sorted(part.by_source.items()):
                log.info("%-9s   from %-6s n=%-6d accuracy %.3f%s", "", source, figures["count"], figures["accuracy"],
                         f" and {figures['second_accuracy']:.3f}" if part.policy_second_share else "")
        log.info("%-9s   teacher chose: %s", "", _shares(part.teacher_share, names))
        log.info("%-9s   network chose: %s", "", _shares(part.policy_share, names))
        if part.policy_second_share:
            log.info("%-9s   teacher's plan: %s", "", _shares(part.teacher_second_share, second_names))
            log.info("%-9s   network's plan: %s", "", _shares(part.policy_second_share, second_names))
        if cloning.layer == OPERATIONAL:
            log.info("%-9s   means accuracy %.3f, of it carried %.3f", "", part.means_accuracy, part.carried_accuracy)


# ---- the fitting itself ---------------------------------------------------------------------

@dataclass
class _Tensors:
    """One half of the split, on the device, in the form the networks take."""

    states: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    weights: torch.Tensor
    #: Each row's source, kept beside the tensors for the report.
    sources: Tuple[str, ...] = ()
    slots: Optional[torch.Tensor] = None
    seconds: Optional[torch.Tensor] = None
    second_masks: Optional[torch.Tensor] = None
    #: The teachers' distributions, one row per decision, with a flag saying which rows carry one; rows without fall back to the single choice.
    softs: Optional[torch.Tensor] = None
    soft_rows: Optional[torch.Tensor] = None
    second_softs: Optional[torch.Tensor] = None
    second_soft_rows: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def take(self, index: torch.Tensor) -> "_Tensors":
        rows = index.tolist()

        def pick(tensor):
            return None if tensor is None else tensor[index]

        return _Tensors(states=self.states[index], actions=self.actions[index],
                        masks=self.masks[index], weights=self.weights[index],
                        sources=tuple(self.sources[row] for row in rows) if self.sources else (),
                        slots=pick(self.slots), seconds=pick(self.seconds), second_masks=pick(self.second_masks),
                        softs=pick(self.softs), soft_rows=pick(self.soft_rows),
                        second_softs=pick(self.second_softs), second_soft_rows=pick(self.second_soft_rows))


def _tensors(teacher: Teacher, rows: np.ndarray, device) -> _Tensors:
    _, _, second = widths(teacher.layer)

    def put(values: np.ndarray, dtype) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype, device=device)

    def distributions(values: np.ndarray):
        present = ~np.isnan(values).any(axis=1) if values.shape[1] else np.zeros(len(values), dtype=bool)
        if not present.any():
            return None, None
        return put(np.nan_to_num(values, nan=0.0), torch.float32), put(present, torch.bool)

    softs, soft_rows = distributions(teacher.softs[rows])
    tensors = _Tensors(states=put(teacher.states[rows], torch.float32), actions=put(teacher.actions[rows], torch.long),
                       masks=put(teacher.masks[rows], torch.float32), weights=put(teacher.weights[rows], torch.float32),
                       sources=tuple(str(s) for s in teacher.sources[rows]), softs=softs, soft_rows=soft_rows)
    if not second:
        return tensors
    second_softs, second_soft_rows = distributions(teacher.second_softs[rows])
    tensors.slots = torch.stack([one_hot_slot(int(slot), device=device) for slot in teacher.squads[rows]]) \
        if len(rows) else torch.zeros((0, SQUAD_SLOTS), device=device)
    tensors.seconds = put(teacher.seconds[rows], torch.long)
    tensors.second_masks = put(teacher.second_masks[rows], torch.float32)
    tensors.second_softs, tensors.second_soft_rows = second_softs, second_soft_rows
    return tensors


def _heads(net: nn.Module, layer: str, tensors: _Tensors):
    """The logits of both heads, the second being nothing for a layer that chooses one thing. The value the networks also return is discarded here, which is the whole of what it means to say the critic is not cloned."""
    if layer != OPERATIONAL:
        logits, _ = net(tensors.states, tensors.masks)
        return logits, None
    # The plan head is shown the region the teacher chose and that region's plan mask, as it is shown the region actually drawn when deciding.
    regions, plans, _ = net(tensors.states, tensors.slots, tensors.masks, tensors.second_masks, tensors.actions)
    return regions, plans


def _loss(net: nn.Module, layer: str, tensors: _Tensors, smoothing: float) -> torch.Tensor:
    """Cross entropy against what the teacher chose, summed over both heads where a decision has two.

    Summed rather than averaged because the two are one decision: a contract names a region and a plan together, and weighting them against each other would be a claim about which half matters more that nothing here supports.
    """
    first, second = _heads(net, layer, tensors)
    loss = _cross_entropy(first, tensors.actions, tensors.masks, smoothing, tensors.weights,
                          tensors.softs, tensors.soft_rows)
    if second is not None:
        loss = loss + _cross_entropy(second, tensors.seconds, tensors.second_masks, smoothing, tensors.weights,
                                     tensors.second_softs, tensors.second_soft_rows)
    return loss


def _cross_entropy(logits: torch.Tensor, chosen: torch.Tensor, mask: torch.Tensor,
                   smoothing: float, weights: Optional[torch.Tensor] = None,
                   softs: Optional[torch.Tensor] = None, soft_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross entropy against the teacher's distribution where it wrote one, and otherwise against its choice with a little of the mass shared out over the other actions that were legal, averaged with each decision's weight.

    A written distribution replaces the smoothing rather than adding to it: it already says how sure the teacher was and of what else, which is what the smoothing stands in for when all that is known is the choice. Either way the mass sits on the legal actions only. A masked action's logit has been floored to a large negative number, so a target that put mass on it would ask the network for a probability the mask removes again on the next pass, and the loss it charges for the failure is proportional to that floor.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    legal = mask > 0
    share = smoothing / legal.sum(dim=-1, keepdim=True).clamp(min=1)
    target = torch.where(legal, share.expand_as(logits), torch.zeros_like(logits))
    target.scatter_add_(-1, chosen.unsqueeze(-1),
                        torch.full_like(chosen, 1.0 - smoothing, dtype=target.dtype).unsqueeze(-1))
    if softs is not None and soft_rows is not None:
        written = torch.where(legal, softs, torch.zeros_like(softs))
        written = written / written.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = torch.where(soft_rows.unsqueeze(-1), written, target)
    per_row = -(target * log_probs).sum(dim=-1)
    if weights is None:
        return per_row.mean()
    return (per_row * weights).sum() / weights.sum().clamp(min=1e-8)


def _score(net: nn.Module, layer: str, tensors: _Tensors, smoothing: float, name: str) -> Fit:
    _, first, second = widths(layer)
    with torch.no_grad():
        one, two = _heads(net, layer, tensors)
        loss = _cross_entropy(one, tensors.actions, tensors.masks, smoothing, tensors.weights,
                              tensors.softs, tensors.soft_rows)
        chosen = one.argmax(dim=-1)
        spread = entropy(one)
        right = chosen == tensors.actions
        fit = Fit(name=name, count=len(tensors), loss=0.0,
                  accuracy=float(right.float().mean()),
                  teacher_share=_distribution(tensors.actions, first),
                  policy_share=_distribution(chosen, first))
        second_right = None
        if two is not None:
            loss = loss + _cross_entropy(two, tensors.seconds, tensors.second_masks, smoothing, tensors.weights,
                                         tensors.second_softs, tensors.second_soft_rows)
            picked = two.argmax(dim=-1)
            spread = spread + entropy(two)
            second_right = picked == tensors.seconds
            fit.second_accuracy = float(second_right.float().mean())
            taught = tensors.seconds % MEANS
            same = (picked % MEANS) == taught
            fit.means_accuracy = float(same.float().mean())
            carried = taught > 0
            fit.carried_accuracy = float(same[carried].float().mean()) if bool(carried.any()) else 0.0
            fit.teacher_second_share = _distribution(tensors.seconds, second)
            fit.policy_second_share = _distribution(picked, second)
        fit.loss = float(loss)
        fit.entropy = float(spread.mean())
        for source in sorted(set(tensors.sources)):
            rows = torch.tensor([s == source for s in tensors.sources], device=right.device)
            figures = {"count": int(rows.sum()), "accuracy": round(float(right[rows].float().mean()), 4)}
            if second_right is not None:
                figures["second_accuracy"] = round(float(second_right[rows].float().mean()), 4)
            fit.by_source[source] = figures
        return fit


def _distribution(choices: torch.Tensor, width: int) -> List[float]:
    counts = torch.bincount(choices, minlength=width).float()
    return (counts / counts.sum().clamp(min=1.0)).tolist()


def _names(layer: str) -> Tuple[List[str], List[str]]:
    """What to call each action in a line a person reads. Regions are numbered rather than named because a region slot is a place on this map and has no name anywhere else."""
    if layer == TACTICAL:
        return [departure.name.lower() for departure in Deviation], []
    if layer == ECONOMIC:
        # A slot is named after the kind of investment it holds, numbered within the kind where the kind has several, because which unit or which region a slot held differs from one board to the next.
        names = []
        for kind, count in INVESTMENT_CAPACITY:
            names.extend([kind.name.lower()] if count == 1 else [f"{kind.name.lower()}{index}" for index in range(count)])
        return names, []
    means = ["walk"] + [f"lift{slot}" for slot in range(MEANS - 1)]
    return ([str(index) for index in range(OPERATIONAL_REGIONS)],
            [f"{task.name.lower()}/{way}" for task in Task for way in means])


def _shares(shares: Sequence[float], names: Sequence[str], most: int = 5) -> str:
    """The heaviest few of a distribution as names and percentages. Truncated because the operational layer has twenty-four regions and what is being looked for is whether the weight sits on one of them."""
    ranked = sorted(zip(names, shares), key=lambda pair: pair[1], reverse=True)
    shown = [f"{name} {100.0 * share:.0f}%" for name, share in ranked[:most] if share > 0.0]
    return " ".join(shown) if shown else "nothing"
