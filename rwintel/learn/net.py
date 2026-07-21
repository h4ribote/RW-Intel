"""The networks, and how small they are obliged to be.

The tactical layer is asked for a decision five times a second per squad, and eight game instances at ten times speed put that at four hundred batches a second on one six gigabyte card. That budget, and not any view about what architecture suits a real-time strategy game, is what fixes the size here: two hidden layers of sixty-four for the tactical policy is what fits, and the design says so in advance. The operational layer runs at a tenth of the rate and reads the whole board, so it is allowed to be wider — but not deeper, because it is the same card.

Both are actor-critic in one body with two heads. Sharing the trunk is what makes the value estimate cost nothing extra, which matters at this rate, and the value head exists at all because the advantage estimator needs it; nothing else reads it.

The operational head is factorised into a region and a task rather than emitting the 144 combinations, because the two questions are different — where is worth going, and what to do on arrival — and because the legal set is a product of two small masks rather than a sparse subset of a large one. Masking is applied as an additive floor on the logits rather than by renormalising afterwards, so that an illegal action has no gradient at all rather than a vanishing one.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from .encoding import (
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    OPERATIONAL_TASKS,
    SQUAD_SLOTS,
    TACTICAL_ACTIONS,
    TACTICAL_SIZE,
)

#: What a masked-out action's logit is pushed to. Large enough that it never survives a softmax at single precision, finite so that it never produces a not-a-number when every action in a row happens to be masked.
MASKED = -1e9


def _trunk(inputs: int, width: int, depth: int = 2) -> nn.Sequential:
    layers: list = []
    size = inputs
    for _ in range(depth):
        layers.append(nn.Linear(size, width))
        layers.append(nn.Tanh())
        size = width
    return nn.Sequential(*layers)


def _initialise(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal weights with a small gain on the output heads, which is what stops a freshly built policy from starting out nearly deterministic. A policy that begins confident explores nothing, and with this sample budget there is no room to wait for it to be argued out of its first opinion."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
    return module


class TacticalNet(nn.Module):
    """One squad's fight to one of the five departures, with a value for it."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.body = _initialise(_trunk(TACTICAL_SIZE, width), gain=2.0 ** 0.5)
        self.action = _initialise(nn.Linear(width, TACTICAL_ACTIONS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden = self.body(state)
        logits = self.action(hidden)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, self.value(hidden).squeeze(-1)


class OperationalNet(nn.Module):
    """The whole board plus which squad is being decided about, to a region and a task, with a value for the pair.

    The squad is named to the network as a slot rather than by handing it its own row separately, because the row is already in the board: the layer's decision about squad three depends on where squads one and two have been sent, and a network that could not see them would be deciding eight independent problems that are not independent.
    """

    def __init__(self, width: int = 192) -> None:
        super().__init__()
        self.body = _initialise(_trunk(OPERATIONAL_SIZE + SQUAD_SLOTS, width), gain=2.0 ** 0.5)
        self.region = _initialise(nn.Linear(width, OPERATIONAL_REGIONS), gain=0.01)
        self.task = _initialise(nn.Linear(width, OPERATIONAL_TASKS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)

    def forward(self, state: torch.Tensor, squad: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None,
                task_mask: Optional[torch.Tensor] = None):
        hidden = self.body(torch.cat([state, squad], dim=-1))
        regions = self.region(hidden)
        tasks = self.task(hidden)
        if region_mask is not None:
            regions = regions.masked_fill(region_mask <= 0, MASKED)
        if task_mask is not None:
            tasks = tasks.masked_fill(task_mask <= 0, MASKED)
        return regions, tasks, self.value(hidden).squeeze(-1)


def one_hot_slot(slot: int, device=None) -> torch.Tensor:
    row = torch.zeros(SQUAD_SLOTS, device=device)
    if 0 <= slot < SQUAD_SLOTS:
        row[slot] = 1.0
    return row


def sample(logits: torch.Tensor, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draws from a categorical distribution over the logits, or takes its mode. Returns the choice and its log probability, which is what the optimiser needs to know how likely the behaviour policy thought the behaviour was."""
    distribution = torch.distributions.Categorical(logits=logits)
    choice = logits.argmax(dim=-1) if greedy else distribution.sample()
    return choice, distribution.log_prob(choice)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).entropy()
