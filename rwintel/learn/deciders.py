"""What answers when a layer is asked to decide.

A decider is the one thing a learnt layer holds that a script layer does not: everything else about the layer — the reports it sends up, the way a contract is priced, the rules that decide when a departure is re-issued — is the same code. That is deliberate and it is what the comparison the whole project rests on requires. If a learnt layer and a script layer differed in more than the decision, then beating the script would not mean the decision had got better.

There is a third possibility and it is why the interface is shaped this way: no decider at all. A learnt layer built without one falls through to the rule it inherited and writes down what the rule chose, which turns the script into a teacher producing state and action pairs of exactly the form a learnt layer emits. That is the design's plan for starting from something rather than from nothing, and it is the only source of such pairs available, since a replay of a human game cannot be re-simulated to recover the state each command was conditioned on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

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

    def choose(self, state: Sequence[float], mask: Sequence[float]) -> Choice:
        if self.batcher is not None:
            return self.batcher.submit((list(state), list(mask)))
        return evaluate_tactical(self.net, [(list(state), list(mask))], self.device, self.greedy)[0]


class PinnedDeparture:
    """Answers with one departure, whatever it is shown.

    An ablation rather than a policy, and the reason it exists as a decider rather than as a file of parameters is that it is a measurement the arena is read against. What the score of a fight can be moved by at all is bounded below by what the departures are worth, and the way to find that bound is to take them away one at a time: a layer that never departs from its contract leaves the engine's own attack-move to fight the whole fight, and a layer that always breaks off gives up every fight it could have won. Both were once kept as hand-made parameter files, and both stopped loading the day the action space went from five departures to seven — a measurement the document quotes should not be able to rot like that. Written here it is a line of code that cannot go stale, and it needs no tensor library at all.
    """

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def choose(self, state: Sequence[float], mask: Sequence[float]) -> Choice:
        # No log probability and no value: this is not a distribution, and nothing is ever learnt from what it chose.
        return Choice(action=self.action)


class PinnedRegion:
    """Answers with a fixed legal region and task, whatever board it is shown.

    The operational analogue of `PinnedDeparture`, and it exists for the same reason: what a match's score can be moved by the operational layer at all is bounded by what the region-and-task choice is worth, and the way to find that bound is to take the choice away. A layer that sends every squad to the same region on the same task is making no choice at all; if the match scores the same with it as with the script that chooses carefully, then the choice was not moving the match. It picks the lowest-numbered legal region and task rather than a fixed number, because which regions exist depends on the map, and an out-of-mask region would be no deployment at all. No network and no tensor library, like the departure it mirrors.

    What this is NOT is a concentration arm, and reading it as one was a real mistake in this project. The region it picks is the lowest live id on the map, which has nothing to do with where a run's scored ground is: on the constructed operations arena the contests are drawn a few hundred units either side of the board centre and the lowest-numbered region is nowhere near them, so the squads march off the scored board altogether and the arm's own diagnostics — how far the nearest of them ended from a contest, and how many were inside one — say so plainly. It is a floor for abandoning the ground, which is a useful floor and a different one. The arm that actually concentrates is `Concentrated`, which keeps the doctrine's own task and overrides only the region, sending every squad at the one region the strategic layer wants most. Removing the ladder's crowding term instead — the massed arm — concentrates nothing on this arena, because a squad chooses from its staging point and is standing in no contested region, so the term that discount removes is already nought and the ladder returns the same order with and without it.
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
