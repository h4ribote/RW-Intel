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
    return [Choice(action=int(action), log_prob=float(log_prob), value=float(value))
            for action, log_prob, value in zip(actions, log_probs, values)]


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
    return [Choice(action=int(region), log_prob=float(region_lp), value=float(value),
                   second=int(task), second_log_prob=float(task_lp))
            for region, region_lp, task, task_lp, value
            in zip(chosen_regions, region_log, chosen_tasks, task_log, values)]


def tactical_batcher(net, device=None, greedy: bool = False, **kwargs) -> Batcher:
    return Batcher(lambda requests: evaluate_tactical(net, requests, device, greedy), **kwargs)


def operational_batcher(net, device=None, greedy: bool = False, **kwargs) -> Batcher:
    return Batcher(lambda requests: evaluate_operational(net, requests, device, greedy), **kwargs)
