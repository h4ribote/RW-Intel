"""Networks that read a layer's state as a set of tokens through a transformer encoder.

Each keeps the actor-critic interface of its layer's flat network (`net.py`): the same forward signature and outputs, and for the operational layer the same `hidden`, `regions` and `plans` methods, so imitation, the deciders and the policy-gradient optimiser use them unchanged. Each also carries twin action-value heads for offline reinforcement learning, which imitation and the policy gradient do not read.

The encoder is pre-norm (batch first, GELU, no dropout). Every token type has its own input projection and a learnt type embedding; tokens absent from a board are left out of attention by the key padding mask, so what a padded slot holds does not change any output.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from ..control.policy.encoding import (
    ECONOMIC_CONTEXT_SIZE,
    GLOBAL_SIZE,
    INVESTMENT_SIZE,
    INVESTMENT_SLOTS,
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    REGION_SIZE,
    SQUAD_SIZE,
    SQUAD_SLOTS,
    TACTICAL_ACTIONS,
    TACTICAL_SIZE,
    TRANSPORT_SIZE,
)
from ..control.policy.encoding import TRANSPORT_SLOTS
from . import tokens as tokens_module
from .net import MASKED, TacticalNet, _initialise

#: Defaults per layer.
WIDTH = 128
HEADS = 4
DEPTH = {"tactics": 3, "operations": 3, "economy": 2}


def _encoder(width: int, heads: int, depth: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)


def _twin(inputs: int, outputs: int) -> nn.ModuleList:
    return nn.ModuleList([_initialise(nn.Linear(inputs, outputs), gain=1.0) for _ in range(2)])


def _zero(module: nn.Module) -> nn.Module:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    return module


class TacticalSetNet(nn.Module):
    """The squad token, one token per member and one per threat, to the departures and a value read off the squad token. Reads a set state (`tokens.set_state`).

    In residual mode the logits, the value and both action values are those of a flat tactical network (`flat`, a `TacticalNet` of width `flat_width`) on the 78 flat features plus heads on the squad token's output, which start at nought, so a freshly built residual network answers exactly as its flat part does and the tokens only add what the flat features leave out.
    """

    KIND = "set"
    LAYER = "tactics"

    #: What a model file written without these keys was built with.
    FILE_DEFAULTS = {"residual": False}

    def __init__(self, width: int = WIDTH, heads: int = HEADS, depth: int = DEPTH["tactics"],
                 members: int = tokens_module.MEMBER_CAP, threats: int = tokens_module.THREAT_CAP,
                 residual: bool = True, flat_width: int = 64) -> None:
        super().__init__()
        self.config = {"width": width, "heads": heads, "depth": depth, "members": members, "threats": threats,
                       "unit_features": tokens_module.UNIT_SIZE, "residual": bool(residual)}
        self.residual = bool(residual)
        self.members, self.threats = members, threats
        self.input_size = tokens_module.set_size(members, threats)
        self.squad_in = nn.Linear(tokens_module.SQUAD_TOKEN, width)
        self.member_in = nn.Linear(tokens_module.UNIT_SIZE, width)
        self.threat_in = nn.Linear(tokens_module.UNIT_SIZE, width)
        self.kinds = nn.Parameter(torch.zeros(3, width))
        self.encoder = _encoder(width, heads, depth)
        if self.residual:
            self.config["flat_width"] = flat_width
            self.flat = TacticalNet(width=flat_width)
            self.action = _zero(nn.Linear(width, TACTICAL_ACTIONS))
            self.value = _zero(nn.Linear(width, 1))
            self.q = nn.ModuleList([_zero(nn.Linear(width, TACTICAL_ACTIONS)) for _ in range(2)])
        else:
            self.action = _initialise(nn.Linear(width, TACTICAL_ACTIONS), gain=0.01)
            self.value = _initialise(nn.Linear(width, 1), gain=1.0)
            self.q = _twin(width, TACTICAL_ACTIONS)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """The squad token's output, (rows, width)."""
        rows = state.shape[0]
        units = self.members + self.threats
        squad = self.squad_in(state[:, :tokens_module.SQUAD_TOKEN]) + self.kinds[0]
        body = state[:, tokens_module.SQUAD_TOKEN:tokens_module.SQUAD_TOKEN + units * tokens_module.UNIT_SIZE]
        body = body.reshape(rows, units, tokens_module.UNIT_SIZE)
        present = state[:, tokens_module.SQUAD_TOKEN + units * tokens_module.UNIT_SIZE:] > 0.5
        ours = self.member_in(body[:, :self.members]) + self.kinds[1]
        theirs = self.threat_in(body[:, self.members:]) + self.kinds[2]
        sequence = torch.cat([squad.unsqueeze(1), ours, theirs], dim=1)
        padding = torch.cat([torch.zeros(rows, 1, dtype=torch.bool, device=state.device), ~present], dim=1)
        return self.encoder(sequence, src_key_padding_mask=padding)[:, 0]

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden = self.encode(state)
        logits, value = self.action(hidden), self.value(hidden).squeeze(-1)
        if self.residual:
            flat_logits, flat_value = self.flat(state[:, :TACTICAL_SIZE])
            logits, value = flat_logits + logits, flat_value + value
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, value

    def critic(self, state: torch.Tensor):
        hidden = self.encode(state)
        q, value = torch.stack([head(hidden) for head in self.q]), self.value(hidden).squeeze(-1)
        if self.residual:
            flat_q, flat_value = self.flat.critic(state[:, :TACTICAL_SIZE])
            q, value = flat_q + q, flat_value + value
        return q, value


class _Token(nn.Module):
    """A linear head on one token of the operational network's packed hidden tensor."""

    def __init__(self, width: int, tokens: int, index: int, outputs: int) -> None:
        super().__init__()
        self.width, self.tokens, self.index = width, tokens, index
        self.linear = _initialise(nn.Linear(width, outputs), gain=1.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden.reshape(*hidden.shape[:-1], self.tokens, self.width)[..., self.index, :])


class OperationalSetNet(nn.Module):
    """The board as one global token, a token per region slot, per squad slot and per transport slot, to a region chosen by attention of the decided squad's token over the region tokens, then a plan for the chosen region, and a value from the global token.

    `hidden` packs the encoder's outputs with two summaries, the decided squad's output token and the mean of the present transports' output tokens, as one flat row per decision, so the deciders can repeat and index it as they do the flat network's.
    """

    KIND = "set"
    LAYER = "operations"

    def __init__(self, width: int = WIDTH, heads: int = HEADS, depth: int = DEPTH["operations"]) -> None:
        super().__init__()
        self.config = {"width": width, "heads": heads, "depth": depth}
        self.width = width
        self.tokens = 1 + OPERATIONAL_REGIONS + SQUAD_SLOTS + TRANSPORT_SLOTS + 2
        self.global_in = nn.Linear(GLOBAL_SIZE, width)
        self.region_in = nn.Linear(REGION_SIZE, width)
        self.squad_in = nn.Linear(SQUAD_SIZE + 1, width)
        self.transport_in = nn.Linear(TRANSPORT_SIZE, width)
        self.kinds = nn.Parameter(torch.zeros(4, width))
        self.slots = nn.Embedding(OPERATIONAL_REGIONS, width)
        self.encoder = _encoder(width, heads, depth)
        self.region_query = nn.Linear(width, width)
        self.region_key = nn.Linear(width, width)
        self.region_bias = _initialise(nn.Linear(width, 1), gain=0.01)
        self.plan_body = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.plan = _initialise(nn.Linear(width, OPERATIONAL_PLANS), gain=0.01)
        self.value = _Token(width, self.tokens, 0, 1)
        self.q_region = _twin(2 * width, 1)
        self.q_plan = _twin(width, OPERATIONAL_PLANS)

    def hidden(self, state: torch.Tensor, squad: torch.Tensor) -> torch.Tensor:
        rows = state.shape[0]
        at = 0
        board = state[:, at:at + GLOBAL_SIZE]
        at += GLOBAL_SIZE
        regions = state[:, at:at + OPERATIONAL_REGIONS * REGION_SIZE].reshape(rows, OPERATIONAL_REGIONS, REGION_SIZE)
        at += OPERATIONAL_REGIONS * REGION_SIZE
        squads = state[:, at:at + SQUAD_SLOTS * SQUAD_SIZE].reshape(rows, SQUAD_SLOTS, SQUAD_SIZE)
        at += SQUAD_SLOTS * SQUAD_SIZE
        transports = state[:, at:at + TRANSPORT_SLOTS * TRANSPORT_SIZE].reshape(rows, TRANSPORT_SLOTS, TRANSPORT_SIZE)
        # The first feature of every region, squad and transport row is its validity flag.
        present = torch.cat([torch.ones(rows, 1, dtype=torch.bool, device=state.device), regions[..., 0] > 0.5,
                             (squads[..., 0] > 0.5) | (squad > 0.5), transports[..., 0] > 0.5], dim=1)
        sequence = torch.cat([
            (self.global_in(board) + self.kinds[0]).unsqueeze(1),
            self.region_in(regions) + self.kinds[1] + self.slots.weight,
            self.squad_in(torch.cat([squads, squad.unsqueeze(-1).to(squads.dtype)], dim=-1)) + self.kinds[2],
            self.transport_in(transports) + self.kinds[3],
        ], dim=1)
        out = self.encoder(sequence, src_key_padding_mask=~present)
        base = 1 + OPERATIONAL_REGIONS
        decided = (out[:, base:base + SQUAD_SLOTS] * squad.unsqueeze(-1).to(out.dtype)).sum(dim=1)
        carried = present[:, base + SQUAD_SLOTS:].unsqueeze(-1).to(out.dtype)
        mean = (out[:, base + SQUAD_SLOTS:] * carried).sum(dim=1) / carried.sum(dim=1).clamp(min=1.0)
        return torch.cat([out, decided.unsqueeze(1), mean.unsqueeze(1)], dim=1).reshape(rows, -1)

    def _unpack(self, hidden: torch.Tensor):
        packed = hidden.reshape(*hidden.shape[:-1], self.tokens, self.width)
        return packed[..., 1:1 + OPERATIONAL_REGIONS, :], packed[..., -2, :], packed[..., -1, :]

    def regions(self, hidden: torch.Tensor, region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        regions, decided, _ = self._unpack(hidden)
        logits = (self.region_key(regions) * self.region_query(decided).unsqueeze(-2)).sum(-1) / math.sqrt(self.width)
        logits = logits + self.region_bias(regions).squeeze(-1)
        return logits if region_mask is None else logits.masked_fill(region_mask <= 0, MASKED)

    def _plan_features(self, hidden: torch.Tensor, region: Optional[torch.Tensor]) -> torch.Tensor:
        regions, decided, mean = self._unpack(hidden)
        if region is None:
            count = regions.shape[-2]
            return self.plan_body(torch.cat([regions, decided.unsqueeze(-2).expand(-1, count, -1),
                                             mean.unsqueeze(-2).expand(-1, count, -1)], dim=-1))
        chosen = regions[torch.arange(regions.shape[0], device=regions.device), region]
        return self.plan_body(torch.cat([chosen, decided, mean], dim=-1))

    def plans(self, hidden: torch.Tensor, region: torch.Tensor,
              plan_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.plan(self._plan_features(hidden, region))
        return logits if plan_mask is None else logits.masked_fill(plan_mask <= 0, MASKED)

    def plans_all(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.plan(self._plan_features(hidden, None))

    def critic_regions(self, hidden: torch.Tensor) -> torch.Tensor:
        regions, decided, _ = self._unpack(hidden)
        paired = torch.cat([regions, decided.unsqueeze(-2).expand_as(regions)], dim=-1)
        return torch.stack([head(paired).squeeze(-1) for head in self.q_region])

    def critic_plans(self, hidden: torch.Tensor, region: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self._plan_features(hidden, region)
        return torch.stack([head(features) for head in self.q_plan])

    def forward(self, state: torch.Tensor, squad: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None,
                plan_mask: Optional[torch.Tensor] = None,
                region: Optional[torch.Tensor] = None):
        hidden = self.hidden(state, squad)
        regions = self.regions(hidden, region_mask)
        if region is None:
            region = regions.argmax(dim=-1)
        return regions, self.plans(hidden, region, plan_mask), self.value(hidden).squeeze(-1)


class EconomicSetNet(nn.Module):
    """The economy's context token and one token per offer, attending over each other, to one logit per offer and a value from the context token."""

    KIND = "set"
    LAYER = "economy"

    def __init__(self, width: int = WIDTH, heads: int = HEADS, depth: int = DEPTH["economy"]) -> None:
        super().__init__()
        self.config = {"width": width, "heads": heads, "depth": depth}
        self.context_in = nn.Linear(ECONOMIC_CONTEXT_SIZE, width)
        self.offer_in = nn.Linear(INVESTMENT_SIZE, width)
        self.kinds = nn.Parameter(torch.zeros(2, width))
        self.encoder = _encoder(width, heads, depth)
        self.score = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), _initialise(nn.Linear(width, 1), gain=0.01))
        self.value = _initialise(nn.Linear(width, 1), gain=1.0)
        self.q = _twin(2 * width, 1)

    def encode(self, state: torch.Tensor):
        rows = state.shape[0]
        context = state[:, :ECONOMIC_CONTEXT_SIZE]
        offers = state[:, ECONOMIC_CONTEXT_SIZE:].reshape(rows, INVESTMENT_SLOTS, INVESTMENT_SIZE)
        present = torch.cat([torch.ones(rows, 1, dtype=torch.bool, device=state.device), offers[..., 0] > 0.5], dim=1)
        sequence = torch.cat([(self.context_in(context) + self.kinds[0]).unsqueeze(1),
                              self.offer_in(offers) + self.kinds[1]], dim=1)
        out = self.encoder(sequence, src_key_padding_mask=~present)
        head = out[:, 0]
        paired = torch.cat([out[:, 1:], head.unsqueeze(1).expand(-1, INVESTMENT_SLOTS, -1)], dim=-1)
        return head, paired

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        head, paired = self.encode(state)
        logits = self.score(paired).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, MASKED)
        return logits, self.value(head).squeeze(-1)

    def critic(self, state: torch.Tensor):
        head, paired = self.encode(state)
        return torch.stack([q(paired).squeeze(-1) for q in self.q]), self.value(head).squeeze(-1)


SET_NETS = {"tactics": TacticalSetNet, "operations": OperationalSetNet, "economy": EconomicSetNet}
