"""Fitting a network to what the handwritten layer already does, so that a reinforcement run starts from something rather than from noise.

A freshly built policy spends its first tens of thousands of decisions discovering things the rule ladder has known all along, and it discovers them from a reward that is mostly shaping. The material to skip that with is already being produced: a layer built without a decider falls through to the inherited rule and writes down the state it saw beside the action the rule chose, which is a supervised problem of exactly the ordinary shape. Fitting it is cheap — a few million multiply-accumulates over a file — and what comes out is a policy that is already worth measuring, which is what makes the first reinforcement updates informative instead of merely destructive.

Two details are the whole difficulty of doing it properly.

The first is that the teacher is deterministic. The script answers the same board the same way every time, so a fit allowed to be confident is nearly one-hot within a few epochs, and a policy that reaches reinforcement learning with no entropy left has stopped producing the evidence that would ever argue it out of an opinion. Smoothing the labels is the cure and it is not optional here. The mass held back is shared out over the actions that were legal rather than over the whole row, because most of the twenty-four regions do not exist on a given board and their logits are floored to a large negative number: asking the network to put a fiftieth of the label on one of them asks it to raise a logit that the mask floors again on the next pass, and the loss goes somewhere no learning rate survives.

The second is that a teacher file records an encoding as much as it records a policy. A written decision is a row of numbers, and nothing in the numbers says what they were made of, so a file written before a feature was added or renamed would fit without complaint and yield a network reading every feature one place from where it now is — or, worse, in the right place and meaning something else. The file therefore states the feature list it was written under at its head, and a file whose list differs, or which states none at all because it was collected before the list was recorded, ends the run. It states one at its head and nowhere else only for as long as it is one collecting run: the writer opens with truncation and cannot append, so the way a teacher is made larger is by joining two files end to end, and a joined file carries the second run's head in its middle. Every stated list is therefore checked wherever it appears and then passed over, which is what makes a join across an encoding change a refusal instead of a fit. The length of a row is checked too and is the backstop rather than the guard: it catches a truncated file and it cannot catch a renaming, which moves nothing. Either way the run ends rather than the row being skipped, since skipping it would mean fitting to the rest of a file that cannot be trusted either.

The value head is deliberately left out of this. A teacher carries actions and no returns — the script's decision has no estimate of what the errand was worth attached to it — so there is nothing here to fit a critic against, and it is warmed up at the start of the reinforcement run instead.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from ..wire import Deviation, Task
from .encoding import (
    OPERATIONAL_FEATURES,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    OPERATIONAL_TASKS,
    TACTICAL_ACTIONS,
    TACTICAL_FEATURES,
    TACTICAL_SIZE,
)
from .net import OperationalNet, TacticalNet, entropy, one_hot_slot
from .policy import OPERATIONAL, TACTICAL

log = logging.getLogger(__name__)

#: How much of each label is held back from the action the teacher chose and shared out over the ones it was allowed to choose instead. Small, and the difference between a policy that can still be argued with and one that cannot: the script is a function rather than a distribution, so a fit with nothing held back converges to certainty about boards it has seen a handful of.
SMOOTHING = 0.05

#: Passes over the teacher before the fit is given up on, when the held-out tenth has not stopped improving first.
EPOCHS = 50

#: Rows in one gradient step.
BATCH = 256

#: Larger than the reinforcement run's rate by a factor of three, because fitting a fixed set of labels is a far better conditioned problem than a policy gradient against a target that moves as the policy does.
LEARNING_RATE = 1e-3

#: Epochs the held-out tenth is allowed not to improve for before the fit is stopped and the best parameters kept. The teacher is finite and the network is small enough to memorise a lot of it, so what is being watched for is the point where further epochs buy the training half at the expense of everything else.
PATIENCE = 5

#: Share of the teacher held back from the fitting and used only to judge it.
VALIDATION_SHARE = 0.1


class TeacherMismatch(ValueError):
    """A teacher file that does not describe the encoding now in force.

    Raised rather than worked around. A fit that quietly dropped the offending rows would be fitting the rest of a file that is wrong in the same way, and a fit that took a file's word for it because the rows are the right length would be trusting the one piece of evidence that cannot see a feature being renamed.
    """


@dataclass
class Sample:
    """One of the teacher's decisions: the board it read, what it chose, and what it was allowed to choose from."""

    state: List[float]
    action: int
    #: The second half of a decision made in two parts, as the operational layer chooses a region and then a task.
    second: int = -1
    #: Which choices were legal. Empty means everything was, which is what the tactical layer's mask always is and what a file that did not record one cannot contradict.
    mask: Sequence[float] = ()
    second_mask: Sequence[float] = ()
    #: Which squad the decision was about, which the operational network is told because its answer depends on it.
    squad: int = 0


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
    #: Mean entropy of the distribution the network puts on these boards, in nats, summed over both heads where there are two. What it is watched for is the collapse that label smoothing exists to prevent: a figure near nought means the fit has become a lookup table and the reinforcement run after it will explore nothing.
    entropy: float = 0.0
    #: How often each action came up, once for the teacher and once for the network. Both, because a fit that only ever answers one thing is a failure and a teacher that only ever asks for one thing is not, and the two cannot be told apart from the network's distribution alone.
    teacher_share: List[float] = field(default_factory=list)
    policy_share: List[float] = field(default_factory=list)
    teacher_second_share: List[float] = field(default_factory=list)
    policy_second_share: List[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"name": self.name, "count": self.count, "loss": round(self.loss, 5),
                "accuracy": round(self.accuracy, 4), "second_accuracy": round(self.second_accuracy, 4),
                "entropy": round(self.entropy, 4),
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
    #: True when the held-out tenth stopped improving before the epoch budget ran out, which is the ordinary and the healthy way for this to end.
    stopped_early: bool
    training: Fit
    validation: Fit

    def as_dict(self) -> dict:
        return {"layer": self.layer, "samples": self.samples, "epochs": self.epochs,
                "stopped_early": self.stopped_early,
                "training": self.training.as_dict(), "validation": self.validation.as_dict()}


def widths(layer: str) -> Tuple[int, int, int]:
    """How long a state, a first choice and a second choice are for one layer. The second is nought for the tactical layer, which chooses one thing."""
    if layer == TACTICAL:
        return TACTICAL_SIZE, TACTICAL_ACTIONS, 0
    if layer == OPERATIONAL:
        return OPERATIONAL_SIZE, OPERATIONAL_REGIONS, OPERATIONAL_TASKS
    raise ValueError(f"no layer named {layer!r}: expected {TACTICAL!r} or {OPERATIONAL!r}")


def feature_names(layer: str) -> Tuple[str, ...]:
    """What one layer's state is made of, in order, which is what a teacher file states at its head and is checked against when it is read back.

    One name per number of the state for the tactical layer, whose fifty-odd features are fifty-odd separate quantities. NOT one name per number for the operational layer: its state is a handful of aggregates followed by a fixed block repeated over twenty-four region slots and another repeated over eight squad slots, so its list names the aggregates, then each block once, then the slot counts — a few dozen names describing a four-hundred-wide state. Anything that reports the length of this list has to say which of the two it is reporting, or it misdescribes an operational file by an order of magnitude; `feature_entry` is what says it.
    """
    if layer == TACTICAL:
        return TACTICAL_FEATURES
    if layer == OPERATIONAL:
        return OPERATIONAL_FEATURES
    raise ValueError(f"no layer named {layer!r}: expected {TACTICAL!r} or {OPERATIONAL!r}")


def feature_entry(layer: str) -> str:
    """What one name in a layer's feature list stands for, for a refusal that has to quote how many there are or which of them moved.

    The tactical list names one number of the state apiece, so its entries are features and calling them that is exact. The operational list names blocks that the state is built by repeating, so its entries are not features and quoting them as though they were would tell somebody staring at a four-hundred-wide state that their file has forty-five of them.
    """
    if layer == TACTICAL:
        return "feature"
    if layer == OPERATIONAL:
        return "block"
    raise ValueError(f"no layer named {layer!r}: expected {TACTICAL!r} or {OPERATIONAL!r}")


def read_teacher(path: str, layer: str = TACTICAL, keep_tainted: bool = False) -> List[Sample]:
    """Every decision in a file written by the collecting run, one JSON object per line, checked against the encoding now in force as it is read.

    The file states its own feature list at its head and that is checked before a single decision is taken from it, because it is the only thing in the file that can tell a state written under an older list from one written under this one. The lengths agree either way when a feature has merely been renamed or replaced, so the check that follows on every row is a backstop and not the guard.

    Every later line that states a list is checked in the same way and then passed over. A teacher file can only be made larger by joining two collecting runs end to end — the writer opens with truncation and has no way of appending — so a file of two runs carries the second one's head in the middle of it, and that head is the only thing that says whether the two halves were collected under the same encoding. Checked, so that a join across an encoding change is refused rather than fitted; passed over, because it is a statement about the file and not a decision to learn from.

    Decisions about a squad somebody outside the command chain took over are dropped unless they are asked for. What the script did with a squad while a person was moving it is not what the script does, and a fit that learnt those would be learning the person.
    """
    samples: List[Sample] = []
    tainted = 0
    heads = 0
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{path} line {number}"
            row = _row(line, where)
            if "encoding" in row or not heads:
                _stated_encoding(row, layer, where)
                heads += 1
                continue
            if row.get("tainted") and not keep_tainted:
                tainted += 1
                continue
            sample = _sample(row, where)
            _check(sample, layer, where)
            samples.append(sample)
    log.info("read %d decision(s) from %s for the %s layer, %d dropped as interfered with%s",
             len(samples), path, layer, tainted,
             f", across {heads} collecting runs joined end to end" if heads > 1 else "")
    return samples


def fit(samples: Sequence[Sample], layer: str = TACTICAL, net: Optional[nn.Module] = None,
        device=None, smoothing: float = SMOOTHING, epochs: int = EPOCHS, batch: int = BATCH,
        learning_rate: float = LEARNING_RATE, patience: int = PATIENCE,
        seed: int = 0) -> Tuple[nn.Module, Cloning]:
    """Fits a network to the teacher's choices and hands it back with the report of how well it fits.

    The set is shuffled before the tenth is taken off it, because a teacher file is written in the order the decisions were made and the last tenth of one is the last few engagements rather than a sample of all of them.
    """
    if not samples:
        raise ValueError("there are no decisions to fit to")
    state_size = widths(layer)[0]
    for index, sample in enumerate(samples):
        _check(sample, layer, f"decision {index}")

    if net is None:
        net = TacticalNet() if layer == TACTICAL else OperationalNet()
    net = net.to(device)
    log.info("fitting the %s layer to %d decision(s): %d feature(s), %d parameter(s)",
             layer, len(samples), state_size, sum(p.numel() for p in net.parameters()))

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    held_out = max(1, round(len(shuffled) * VALIDATION_SHARE)) if len(shuffled) > 1 else 0
    learning, checking = shuffled[:len(shuffled) - held_out], shuffled[len(shuffled) - held_out:]
    if not checking:
        # A handful of decisions has nothing to hold out of it, so the two halves are the same rows and the figure reported for the second one is an optimistic one. Said here rather than refused, because the fit is still the right thing to do on a set that small; it is only the judgement of it that is weaker.
        checking = learning
    learnt = _tensors(learning, layer, device)
    checked = _tensors(checking, layer, device)

    # The critic is not in the optimiser at all. The teacher has no returns in it, so there is no target to move the value head towards, and leaving it in would let Adam's own state drift it on gradients that are not there.
    spared = {id(parameter) for parameter in net.value.parameters()}
    moving = [parameter for parameter in net.parameters() if id(parameter) not in spared]
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
            # The parameters kept are the ones that scored best on the tenth that was held back, not the ones the last epoch happened to leave behind: every epoch after the best one was buying the fitting half at the expense of everything else, which is the whole reason for holding a tenth back.
            best_state = {name: value.detach().clone() for name, value in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    return net, Cloning(layer=layer, samples=len(samples), epochs=ran, stopped_early=ran < epochs,
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
        log.info("%-9s   teacher chose: %s", "", _shares(part.teacher_share, names))
        log.info("%-9s   network chose: %s", "", _shares(part.policy_share, names))
        if part.policy_second_share:
            log.info("%-9s   teacher's task: %s", "", _shares(part.teacher_second_share, second_names))
            log.info("%-9s   network's task: %s", "", _shares(part.policy_second_share, second_names))


# ---- the fitting itself ---------------------------------------------------------------------

@dataclass
class _Tensors:
    """One half of the split, on the device, in the form the networks take."""

    states: torch.Tensor
    actions: torch.Tensor
    masks: torch.Tensor
    slots: Optional[torch.Tensor] = None
    seconds: Optional[torch.Tensor] = None
    second_masks: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def take(self, index: torch.Tensor) -> "_Tensors":
        return _Tensors(states=self.states[index], actions=self.actions[index],
                        masks=self.masks[index],
                        slots=None if self.slots is None else self.slots[index],
                        seconds=None if self.seconds is None else self.seconds[index],
                        second_masks=None if self.second_masks is None else self.second_masks[index])


def _stated_encoding(row: dict, layer: str, where: str) -> None:
    """Holds a line that states a feature list to be stating the one now in force.

    A file whose head states none is refused rather than assumed to match. It was written before the list was recorded, so the only evidence available about its features is their number — which is exactly the evidence that cannot tell a renamed feature from the one it replaced, and a fit made on that assumption produces a network that reads the board wrongly and reports a perfectly ordinary accuracy for doing so.

    What a refusal quotes is entries of the stated list, and the word for them is the layer's own. The tactical list names one number of the state apiece; the operational list names blocks that the state is built by repeating, so its numbering is a numbering of blocks and its length is a count of blocks, and reporting either as a count of features would misdescribe a four-hundred-wide state as a forty-five-wide one.
    """
    stated = row.get("encoding")
    if not isinstance(stated, list):
        raise TeacherMismatch(
            f"{where}: this file does not state the feature list it was written under, so nothing in it says "
            f"whether its states mean what the {layer} encoding now means; it was collected before the list was "
            f"recorded and has to be collected again")
    named = row.get("layer")
    if named is not None and named != layer:
        raise TeacherMismatch(f"{where}: this is a {named} teacher, and it is being read for the {layer} layer")
    expected = feature_names(layer)
    if tuple(stated) == expected:
        return
    entry = feature_entry(layer)
    for index, (before, after) in enumerate(zip(stated, expected)):
        if before != after:
            raise TeacherMismatch(
                f"{where}: this teacher was written by a different feature list — its {entry} {index} is "
                f"{before!r} where the {layer} encoding now reads {after!r} — so fitting to it would produce a "
                f"network reading what that {entry} covers as something it no longer is")
    raise TeacherMismatch(
        f"{where}: this teacher was written by a feature list of {len(stated)} {entry}s where the {layer} "
        f"encoding now has {len(expected)}, so fitting to it would produce a network reading every feature in "
        f"the wrong place")


def _row(line: str, where: str) -> dict:
    """One line of a teacher file as the object it is meant to be, refused by name and line where it is not one.

    A collecting run killed while it was writing leaves its last line half finished, and a half finished line is not a thing anybody should have to recognise from a JSON decoder's own complaint about a column of a string it cannot name a file for. Everything malformed in this file is refused in the one way and says which line it was on.
    """
    try:
        row = json.loads(line)
    except ValueError as broken:
        raise TeacherMismatch(f"{where}: this line is not the JSON object every line of a teacher file is "
                              f"({broken}); a collecting run that was killed while writing leaves its last "
                              f"line half finished") from broken
    if not isinstance(row, dict):
        raise TeacherMismatch(f"{where}: this line is a {type(row).__name__} where every line of a teacher "
                              f"file is an object")
    return row


def _sample(row: dict, where: str) -> Sample:
    """One written decision in the form the fitting takes, refused by name and line where the line is not a decision at all.

    The two fields it cannot do without are named rather than left to fail on their absence. A line missing them is not a decision, and the likeliest reason for one is that it is a head — a statement of the feature list, which a file carries at its start and again wherever two collecting runs were joined — that something has taken for a decision. Refused as everything else malformed in this file is refused, since a bare missing-key error names neither the file, nor the line, nor what it was expecting.
    """
    for field in ("state", "action"):
        if field not in row:
            raise TeacherMismatch(f"{where}: this line states no {field!r}, so it is not one of the teacher's "
                                  f"decisions; a line that is a feature list rather than a decision states "
                                  f"'encoding' and is read as the head of a collecting run")
    return Sample(state=[float(value) for value in row["state"]],
                  action=int(row["action"]), second=int(row.get("second", -1)),
                  mask=tuple(row.get("mask") or ()),
                  second_mask=tuple(row.get("second_mask") or ()),
                  squad=int(row.get("squad", 0)))


def _check(sample: Sample, layer: str, where: str) -> None:
    state_size, first, second = widths(layer)
    if len(sample.state) != state_size:
        raise TeacherMismatch(
            f"{where}: the state is {len(sample.state)} long where the {layer} encoding is {state_size}, "
            f"so this teacher was written by a different feature list and fitting to it would produce a "
            f"network reading every feature in the wrong place")
    if not 0 <= sample.action < first:
        raise TeacherMismatch(f"{where}: action {sample.action} is outside the {first} the {layer} layer chooses between")
    if sample.mask and len(sample.mask) != first:
        raise TeacherMismatch(f"{where}: the mask is {len(sample.mask)} long where the {layer} layer has {first} actions")
    if not second:
        return
    if not 0 <= sample.second < second:
        raise TeacherMismatch(f"{where}: task {sample.second} is outside the {second} the {layer} layer chooses between")
    if sample.second_mask and len(sample.second_mask) != second:
        raise TeacherMismatch(f"{where}: the task mask is {len(sample.second_mask)} long where there are {second} tasks")


def _tensors(samples: Sequence[Sample], layer: str, device) -> _Tensors:
    _, first, second = widths(layer)
    states = torch.tensor([list(sample.state) for sample in samples], dtype=torch.float32, device=device)
    actions = torch.tensor([sample.action for sample in samples], dtype=torch.long, device=device)
    masks = torch.tensor([list(sample.mask) if sample.mask else [1.0] * first for sample in samples],
                         dtype=torch.float32, device=device)
    if not second:
        return _Tensors(states=states, actions=actions, masks=masks)
    return _Tensors(
        states=states, actions=actions, masks=masks,
        slots=torch.stack([one_hot_slot(sample.squad, device=device) for sample in samples]),
        seconds=torch.tensor([sample.second for sample in samples], dtype=torch.long, device=device),
        second_masks=torch.tensor([list(sample.second_mask) if sample.second_mask else [1.0] * second
                                   for sample in samples], dtype=torch.float32, device=device))


def _heads(net: nn.Module, layer: str, tensors: _Tensors):
    """The logits of both heads, the second being nothing for a layer that chooses one thing. The value the networks also return is discarded here, which is the whole of what it means to say the critic is not cloned."""
    if layer == TACTICAL:
        logits, _ = net(tensors.states, tensors.masks)
        return logits, None
    regions, tasks, _ = net(tensors.states, tensors.slots, tensors.masks, tensors.second_masks)
    return regions, tasks


def _loss(net: nn.Module, layer: str, tensors: _Tensors, smoothing: float) -> torch.Tensor:
    """Cross entropy against what the teacher chose, summed over both heads where a decision has two.

    Summed rather than averaged because the two are one decision: a contract names a region and a task together, and weighting them against each other would be a claim about which half matters more that nothing here supports.
    """
    first, second = _heads(net, layer, tensors)
    loss = _cross_entropy(first, tensors.actions, tensors.masks, smoothing)
    if second is not None:
        loss = loss + _cross_entropy(second, tensors.seconds, tensors.second_masks, smoothing)
    return loss


def _cross_entropy(logits: torch.Tensor, chosen: torch.Tensor, mask: torch.Tensor,
                   smoothing: float) -> torch.Tensor:
    """Cross entropy with a little of each label's mass shared out over the other actions that were legal.

    Over the legal ones only. A masked action's logit has been floored to a large negative number, so a target that put mass on it would ask the network for a probability the mask removes again on the next pass, and the loss it charges for the failure is proportional to that floor.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    legal = mask > 0
    share = smoothing / legal.sum(dim=-1, keepdim=True).clamp(min=1)
    target = torch.where(legal, share.expand_as(logits), torch.zeros_like(logits))
    target.scatter_add_(-1, chosen.unsqueeze(-1),
                        torch.full_like(chosen, 1.0 - smoothing, dtype=target.dtype).unsqueeze(-1))
    return -(target * log_probs).sum(dim=-1).mean()


def _score(net: nn.Module, layer: str, tensors: _Tensors, smoothing: float, name: str) -> Fit:
    _, first, second = widths(layer)
    with torch.no_grad():
        one, two = _heads(net, layer, tensors)
        loss = _cross_entropy(one, tensors.actions, tensors.masks, smoothing)
        chosen = one.argmax(dim=-1)
        spread = entropy(one)
        fit = Fit(name=name, count=len(tensors), loss=0.0,
                  accuracy=float((chosen == tensors.actions).float().mean()),
                  teacher_share=_distribution(tensors.actions, first),
                  policy_share=_distribution(chosen, first))
        if two is not None:
            loss = loss + _cross_entropy(two, tensors.seconds, tensors.second_masks, smoothing)
            picked = two.argmax(dim=-1)
            spread = spread + entropy(two)
            fit.second_accuracy = float((picked == tensors.seconds).float().mean())
            fit.teacher_second_share = _distribution(tensors.seconds, second)
            fit.policy_second_share = _distribution(picked, second)
        fit.loss = float(loss)
        fit.entropy = float(spread.mean())
        return fit


def _distribution(choices: torch.Tensor, width: int) -> List[float]:
    counts = torch.bincount(choices, minlength=width).float()
    return (counts / counts.sum().clamp(min=1.0)).tolist()


def _names(layer: str) -> Tuple[List[str], List[str]]:
    """What to call each action in a line a person reads. Regions are numbered rather than named because a region slot is a place on this map and has no name anywhere else."""
    if layer == TACTICAL:
        return [departure.name.lower() for departure in Deviation], []
    return [str(index) for index in range(OPERATIONAL_REGIONS)], [task.name.lower() for task in Task]


def _shares(shares: Sequence[float], names: Sequence[str], most: int = 5) -> str:
    """The heaviest few of a distribution as names and percentages. Truncated because the operational layer has twenty-four regions and what is being looked for is whether the weight sits on one of them."""
    ranked = sorted(zip(names, shares), key=lambda pair: pair[1], reverse=True)
    shown = [f"{name} {100.0 * share:.0f}%" for name, share in ranked[:most] if share > 0.0]
    return " ".join(shown) if shown else "nothing"
