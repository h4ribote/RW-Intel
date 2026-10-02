"""The networks, and how small they are obliged to be.

The tactical layer is asked for a decision five times a second per squad in every game instance, which at ten times speed and a dozen instances is hundreds of requests a second. That rate, and not any view about what architecture suits a real-time strategy game, is what fixes the size here: two hidden layers of sixty-four for the tactical policy, small enough that a call costs its dispatch rather than its arithmetic and the processor answers faster than a graphics card. The operational layer runs at a tenth of the rate and reads the whole board, so it is allowed to be wider - but not deeper, because it shares the same budget.

The economic network decides twice a second at most a few times per instance, so its cost is nothing beside the tactical one; what shapes it is that the investments it chooses between are a different set on every board, which it meets by scoring each offer with one shared network rather than giving each slot an output of its own.

All three are actor-critic in one body with two heads. Sharing the trunk is what makes the value estimate cost nothing extra, which matters at this rate, and the value head exists at all because the advantage estimator needs it; nothing else reads it. Each also carries twin action-value heads, which only offline reinforcement learning reads (`offline.py`) and which imitation and the policy gradient leave out of their optimisers (`actor_critic_parameters`).

These sizes bind the networks that answer a game from the processor. The set networks of `setnet.py` are larger, and are trained from recorded data on a graphics card, where no decision waits on them.

The operational head is factorised into a region and a plan -a task and a means of getting there- rather than emitting every combination, because the two questions are different - where is worth going, and what to do there and how to get there - and because the legal plans of each region are a small mask rather than a sparse subset of one large one. Masking is applied as an additive floor on the logits rather than by renormalising afterwards, so that an illegal action has no gradient at all rather than a vanishing one.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from ..control.policy.encoding import (
    ECONOMIC_CONTEXT_SIZE,
    INVESTMENT_SIZE,
    INVESTMENT_SLOTS,
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
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


#: The submodules that hold the action-value heads, in a flat network and in a set network alike.
ACTION_VALUE_HEADS = ("q", "q_region", "q_plan")


def actor_critic_parameters(net: nn.Module) -> list:
    """Every parameter but the action-value heads', including those of a residual network's flat part: what imitation and the policy gradient optimise."""
    return [parameter for name, parameter in net.named_parameters()
            if not any(part in ACTION_VALUE_HEADS for part in name.split("."))]


def value_parameters(net: nn.Module) -> list:
    """The value head's parameters, together with those of a residual network's flat part's value head."""
    return [parameter for name, parameter in net.named_parameters() if "value" in name.split(".")]


def _twin(inputs: int, outputs: int) -> nn.ModuleList:
    """Two action-value heads of the same shape, which offline reinforcement learning takes the smaller of. Neither imitation nor the policy gradient reads them, so they receive no gradient there."""
    return nn.ModuleList([_initialise(nn.Linear(inputs, outputs), gain=1.0) for _ in range(2)])


class TacticalNet(nn.Module):
    """One squad's fight to one of the departures, with a value for it."""

    KIND = "flat"
    LAYER = "tactics"

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.config = {"width": width}
        self.body = _initialise(_trunk(TACTICAL_SIZE, width), gain=2.0 ** 0.5)
        self.action = _initialise(nn.Linear(width, TACTICAL_ACTIONS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        self.q = _twin(width, TACTICAL_ACTIONS)

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden = self.body(state)
        logits = self.action(hidden)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, self.value(hidden).squeeze(-1)

    def critic(self, state: torch.Tensor):
        """Both action-value heads over every departure, stacked as (2, rows, actions), and the value."""
        hidden = self.body(state)
        return torch.stack([head(hidden) for head in self.q]), self.value(hidden).squeeze(-1)


#: Width of the learnt embedding of a region slot that the plan head reads beside the board.
REGION_EMBEDDING = 16


class OperationalNet(nn.Module):
    """The whole board plus which squad is being decided about, to a region and then a plan for that region, with a value for the board.

    The squad is named to the network as a slot rather than by handing it its own row separately, because the row is already in the board: the layer's decision about squad three depends on where squads one and two have been sent, and a network that could not see them would be deciding eight independent problems that are not independent.

    The plan -a task and a means of getting there- is chosen after the region and reads it, because what to do on arrival depends on where one is arriving, and how to get there depends on where it is: a vanguard surrounds a region it outnumbers and pushes straight into one it does not, and walks to a region it can walk to and is carried to one across water. Which plans are open is given per region, as a mask on the plan logits of that region.
    """

    KIND = "flat"
    LAYER = "operations"

    def __init__(self, width: int = 192) -> None:
        super().__init__()
        self.config = {"width": width}
        self.body = _initialise(_trunk(OPERATIONAL_SIZE + SQUAD_SLOTS, width), gain=2.0 ** 0.5)
        self.region = _initialise(nn.Linear(width, OPERATIONAL_REGIONS), gain=0.01)
        self.embedding = nn.Embedding(OPERATIONAL_REGIONS, REGION_EMBEDDING)
        self.plan = _initialise(nn.Linear(width + REGION_EMBEDDING, OPERATIONAL_PLANS), gain=0.01)
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        self.q_region = _twin(width, OPERATIONAL_REGIONS)
        self.q_plan = _twin(width + REGION_EMBEDDING, OPERATIONAL_PLANS)

    def hidden(self, state: torch.Tensor, squad: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([state, squad], dim=-1))

    def regions(self, hidden: torch.Tensor, region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.region(hidden)
        return logits if region_mask is None else logits.masked_fill(region_mask <= 0, MASKED)

    def _plan_input(self, hidden: torch.Tensor, region: Optional[torch.Tensor]) -> torch.Tensor:
        """What the plan head reads: for one region per row, or for every region at once (rows, regions, inputs) when none is given."""
        if region is not None:
            return torch.cat([hidden, self.embedding(region)], dim=-1)
        rows = hidden.shape[0]
        every = self.embedding.weight.unsqueeze(0).expand(rows, -1, -1)
        return torch.cat([hidden.unsqueeze(1).expand(rows, OPERATIONAL_REGIONS, hidden.shape[-1]), every], dim=-1)

    def plans(self, hidden: torch.Tensor, region: torch.Tensor,
              plan_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Plan logits for the region given, masked by that region's row of plans."""
        logits = self.plan(self._plan_input(hidden, region))
        return logits if plan_mask is None else logits.masked_fill(plan_mask <= 0, MASKED)

    def plans_all(self, hidden: torch.Tensor) -> torch.Tensor:
        """Unmasked plan logits for every region, as (rows, regions, plans)."""
        return self.plan(self._plan_input(hidden, None))

    def critic_regions(self, hidden: torch.Tensor) -> torch.Tensor:
        """The region part of both joint action values, as (2, rows, regions)."""
        return torch.stack([head(hidden) for head in self.q_region])

    def critic_plans(self, hidden: torch.Tensor, region: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The plan part of both joint action values for the region given, as (2, rows, plans), or for every region, as (2, rows, regions, plans)."""
        features = self._plan_input(hidden, region)
        return torch.stack([head(features) for head in self.q_plan])

    def forward(self, state: torch.Tensor, squad: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None,
                plan_mask: Optional[torch.Tensor] = None,
                region: Optional[torch.Tensor] = None):
        """Region logits, plan logits for the region given (the most likely one when none is) under that region's plan mask, and the value."""
        hidden = self.hidden(state, squad)
        regions = self.regions(hidden, region_mask)
        if region is None:
            region = regions.argmax(dim=-1)
        return regions, self.plans(hidden, region, plan_mask), self.value(hidden).squeeze(-1)


class EconomicNet(nn.Module):
    """The economy's board and the investments on offer, to which one comes next, with a value for the board.

    What is on offer changes with every pick and every match -the units depend on which factories are idle and what their menus hold- so a slot does not name the same investment from one board to the next the way a region slot names the same place. Each offer is therefore scored by one shared network from its own row and the board, and the slot it sits in carries no meaning of its own. The board the value and the scores read is the context row together with the mean of the offers' encodings, so that what could be bought is part of how good the position is.
    """

    KIND = "flat"
    LAYER = "economy"

    def __init__(self, width: int = 128, offer_width: int = 64) -> None:
        super().__init__()
        self.config = {"width": width, "offer_width": offer_width}
        self.offer = _initialise(_trunk(INVESTMENT_SIZE, offer_width), gain=2.0 ** 0.5)
        self.body = _initialise(_trunk(ECONOMIC_CONTEXT_SIZE + offer_width, width), gain=2.0 ** 0.5)
        self.score = nn.Sequential(_initialise(nn.Linear(width + offer_width, offer_width), gain=2.0 ** 0.5),
                                   nn.Tanh(), _initialise(nn.Linear(offer_width, 1), gain=0.01))
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        self.q = _twin(width + offer_width, 1)

    def _paired(self, state: torch.Tensor):
        context = state[..., :ECONOMIC_CONTEXT_SIZE]
        rows = state[..., ECONOMIC_CONTEXT_SIZE:].reshape(*state.shape[:-1], INVESTMENT_SLOTS, INVESTMENT_SIZE)
        offers = self.offer(rows)
        # The first feature of a row is its validity flag, so the pooled offer reads only the slots that hold one.
        present = rows[..., :1]
        pooled = (offers * present).sum(dim=-2) / present.sum(dim=-2).clamp(min=1.0)
        hidden = self.body(torch.cat([context, pooled], dim=-1))
        paired = torch.cat([hidden.unsqueeze(-2).expand(*offers.shape[:-1], hidden.shape[-1]), offers], dim=-1)
        return hidden, paired

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden, paired = self._paired(state)
        logits = self.score(paired).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, self.value(hidden).squeeze(-1)

    def critic(self, state: torch.Tensor):
        """Both action-value heads over every offer slot, stacked as (2, rows, slots), and the value."""
        hidden, paired = self._paired(state)
        return torch.stack([head(paired).squeeze(-1) for head in self.q]), self.value(hidden).squeeze(-1)


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
