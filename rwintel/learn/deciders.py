"""What answers when a layer is asked to decide.

A decider is the one thing a learnt layer holds that a script layer does not: everything else about the layer — the reports it sends up, the way a contract is priced, the rules that decide when a departure is re-issued — is the same code. That is deliberate and it is what the comparison the whole project rests on requires. If a learnt layer and a script layer differed in more than the decision, then beating the script would not mean the decision had got better.

There is a third possibility and it is why the interface is shaped this way: no decider at all. A learnt layer built without one falls through to the rule it inherited and writes down what the rule chose, which turns the script into a teacher producing state and action pairs of exactly the form a learnt layer emits. That is the design's plan for starting from something rather than from nothing, and it is the only source of such pairs available, since a replay of a human game cannot be re-simulated to recover the state each command was conditioned on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .inference import Batcher


@dataclass
class Choice:
    """A decision and what the decider knew about it. The log probability and the value are nought from a decider that is not a distribution, which is what makes a script's decision storable in the same buffer without pretending it came from one."""

    action: int
    log_prob: float = 0.0
    value: float = 0.0
    #: The second half of a decision made in two parts, as the operational layer chooses a region and then a task.
    second: int = -1
    second_log_prob: float = 0.0

    @property
    def total_log_prob(self) -> float:
        return self.log_prob + self.second_log_prob


class NetworkTactics:
    """Answers from a network, through the batching server when there is one."""

    def __init__(self, net, device=None, batcher: Optional[Batcher] = None, greedy: bool = False) -> None:
        self.net = net
        self.device = device
        self.batcher = batcher
        self.greedy = greedy

    def choose_many(self, requests: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> List[Choice]:
        """A whole side's departures in one ask.

        The plural is the only form a tactical decider has, because a tactical layer decides about all of its squads from one board read at one instant, and asking one at a time is what makes a side of four cost four batching windows a frame instead of one. It is also why this is not a widened `choose`: a decider written against the older one-squad signature would silently take a list of pairs as its state and answer nonsense, where one that no longer has the method it is asked for fails at once.
        """
        batch = [(list(state), list(mask)) for state, mask in requests]
        if self.batcher is not None:
            return list(self.batcher.submit_many(batch))
        return evaluate_tactical(self.net, batch, self.device, self.greedy)


class NetworkStrategy:
    """Answers the posture from a network, through the batching server when there is one.

    Plural like the tactical decider and for a different reason. One side takes exactly one strategic decision a period, so there is nothing to batch within a side; what meets in a window here is the other instances of the run, which at one decision per ten seconds of game time is the only way this layer's requests are ever more than one at a time. The signature is the tactical one because the question is the same shape — a state, a mask over a categorical head, one choice — and having both go through one form is what lets one batching server implementation serve either.
    """

    def __init__(self, net, device=None, batcher: Optional[Batcher] = None, greedy: bool = False) -> None:
        self.net = net
        self.device = device
        self.batcher = batcher
        self.greedy = greedy

    def choose_many(self, requests: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> List[Choice]:
        batch = [(list(state), list(mask)) for state, mask in requests]
        if self.batcher is not None:
            return list(self.batcher.submit_many(batch))
        return evaluate_strategic(self.net, batch, self.device, self.greedy)


class PinnedPosture:
    """Answers with one posture, whatever board it is shown.

    The strategic analogue of the pinned departure and the pinned region, and it exists for the measurement they exist for: what the posture choice is worth at all is bounded by what happens when there is no choice, and the way to find that bound is to take it away. It is also the arm the evaluation runner already had by another name — a run that pins the strategic layer to `defend` and one that lets the rule transition are two arms of the same question — so a learnt posture has a floor to be read against that is not only the rule.
    """

    def __init__(self, posture: int) -> None:
        self.posture = int(posture)

    def choose_many(self, requests: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> List[Choice]:
        # No log probability and no value: this is not a distribution, and nothing is ever learnt from what it chose.
        return [Choice(action=self.posture) for _ in requests]


class PinnedDeparture:
    """Answers with one departure, whatever it is shown.

    An ablation rather than a policy, and the reason it exists as a decider rather than as a file of parameters is that it is a measurement the arena is read against. What the score of a fight can be moved by at all is bounded below by what the departures are worth, and the way to find that bound is to take them away one at a time: a layer that never departs from its contract leaves the engine's own attack-move to fight the whole fight, and a layer that always breaks off gives up every fight it could have won. Both were once kept as hand-made parameter files, and both stopped loading the day the action space went from five departures to seven — a measurement the document quotes should not be able to rot like that. Written here it is a line of code that cannot go stale, and it needs no tensor library at all.
    """

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def choose_many(self, requests: Sequence[Tuple[Sequence[float], Sequence[float]]]) -> List[Choice]:
        # No log probability and no value: this is not a distribution, and nothing is ever learnt from what it chose.
        return [Choice(action=self.action) for _ in requests]


class PinnedRegion:
    """Answers with a fixed legal region and task, whatever board it is shown.

    The operational analogue of `PinnedDeparture`, and it exists for the same reason: what a match's score can be moved by the operational layer at all is bounded by what the region-and-task choice is worth, and the way to find that bound is to take the choice away. A layer that sends every squad to the same region on the same task is making no choice at all; if the match scores the same with it as with the script that chooses carefully, then the choice was not moving the match. It picks the first legal slot and task rather than a fixed number, because which regions exist depends on the map, and an out-of-mask region would be no deployment at all. No network and no tensor library, like the departure it mirrors.

    WHERE the first slot is has moved, and readings taken before it moved describe a different arm. The slots were the map's own numbering, so this pinned on the lowest live id — ground with no relation to a run's scored contests, and on the constructed operations arena nowhere near them, so its squads marched off the scored board altogether and its own diagnostics said so. The slots run outward from the side's own home now, so the first of them is the side's HOME region: the arm sends every squad to the ground it starts on. Both are floors for making no deployment at all, which is what the arm is for, but they are not the same floor and a figure taken under one does not carry to the other.

    What this is NOT, under either ordering, is a concentration arm, and reading it as one was a real mistake in this project. The arm that actually concentrates is `Concentrated`, which keeps the doctrine's own task and overrides only the region, sending every squad at the one region the strategic layer wants most. Removing the ladder's crowding term instead — the massed arm — concentrates nothing on this arena, because a squad chooses from its staging point and is standing in no contested region, so the term that discount removes is already nought and the ladder returns the same order with and without it.
    """

    def choose(self, state: Sequence[float], slot: int, region_mask: Sequence[float],
               task_mask: Sequence[float]) -> Optional[Choice]:
        region = next((index for index, value in enumerate(region_mask) if value > 0), -1)
        task = next((index for index, value in enumerate(task_mask) if value > 0), -1)
        if region < 0 or task < 0:
            return None
        # No log probability and no value: this is not a distribution, and nothing is ever learnt from what it chose.
        return Choice(action=region, second=task)


class NetworkOperations:
    def __init__(self, net, device=None, batcher: Optional[Batcher] = None, greedy: bool = False) -> None:
        self.net = net
        self.device = device
        self.batcher = batcher
        self.greedy = greedy

    def choose(self, state: Sequence[float], slot: int, region_mask: Sequence[float],
               task_mask: Sequence[float]) -> Optional[Choice]:
        request = (list(state), int(slot), list(region_mask), list(task_mask))
        if self.batcher is not None:
            return self.batcher.submit(request)
        return evaluate_operational(self.net, [request], self.device, self.greedy)[0]


def evaluate_tactical(net, requests: Sequence[tuple], device=None, greedy: bool = False) -> List[Choice]:
    """One forward pass for a whole batch of squads, wherever they came from. Imported lazily so that a control process that is only running scripts never loads the tensor library at all."""
    import torch

    from .net import sample

    states = torch.tensor([request[0] for request in requests], dtype=torch.float32, device=device)
    masks = torch.tensor([request[1] for request in requests], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, values = net(states, masks)
        actions, log_probs = sample(logits, greedy)
        answers = _fetch(actions, log_probs, values)
    return [Choice(action=int(action), log_prob=log_prob, value=value)
            for action, log_prob, value in zip(*answers)]


def evaluate_strategic(net, requests: Sequence[tuple], device=None, greedy: bool = False) -> List[Choice]:
    """One forward pass for a batch of postures.

    The arithmetic is the tactical one — a state, a mask, one categorical head and a value — so it is that function called under this layer's name rather than a second copy of it. Named separately all the same, because the two networks are different objects with different feature lists, and a caller that had to know they happen to share a forward signature would be a caller that stops working quietly the day one of them does not.
    """
    return evaluate_tactical(net, requests, device, greedy)


def evaluate_operational(net, requests: Sequence[tuple], device=None, greedy: bool = False) -> List[Choice]:
    import torch

    from .net import one_hot_slot, sample

    states = torch.tensor([request[0] for request in requests], dtype=torch.float32, device=device)
    slots = torch.stack([one_hot_slot(request[1], device=device) for request in requests])
    regions = torch.tensor([request[2] for request in requests], dtype=torch.float32, device=device)
    tasks = torch.tensor([request[3] for request in requests], dtype=torch.float32, device=device)
    with torch.no_grad():
        region_logits, task_logits, values = net(states, slots, regions, tasks)
        chosen_regions, region_log = sample(region_logits, greedy)
        chosen_tasks, task_log = sample(task_logits, greedy)
        answers = _fetch(chosen_regions, region_log, values, chosen_tasks, task_log)
    return [Choice(action=int(region), log_prob=region_lp, value=value,
                   second=int(task), second_log_prob=task_lp)
            for region, region_lp, value, task, task_lp in zip(*answers)]


def _fetch(*tensors):
    """Brings a batch of answers back from the device in one transfer.

    Reading them one number at a time is what an obvious implementation does and it is ruinous: every scalar taken off a device tensor waits for the device to finish, so a batch of sixty-four costs a couple of hundred round trips and the batching that was supposed to make inference cheap makes it slower per decision than not batching at all. Stacking first means one wait for the whole batch, whatever its size.
    """
    import torch

    return torch.stack([tensor.float() for tensor in tensors]).cpu().tolist()


def tactical_batcher(net, device=None, greedy: bool = False, **kwargs) -> Batcher:
    return Batcher(lambda requests: evaluate_tactical(net, requests, device, greedy), **kwargs)


def operational_batcher(net, device=None, greedy: bool = False, **kwargs) -> Batcher:
    return Batcher(lambda requests: evaluate_operational(net, requests, device, greedy), **kwargs)


def strategic_batcher(net, device=None, greedy: bool = False, **kwargs) -> Batcher:
    return Batcher(lambda requests: evaluate_strategic(net, requests, device, greedy), **kwargs)
