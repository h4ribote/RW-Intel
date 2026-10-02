"""The built-in AI as a teacher of the operational layer.

An episode with two AI contestants is observed from the side of one of them, and the script chain runs over that player's units as a playback runs it over a person's: the organisation layer forms squads, and the operational layer asks for a decision when it would in a match. The agent sends the AI players' own orders on each operational observation, and the answer is read out of them by the same inference a replay uses (`rwintel.replay.human.infer_decision`).

A live match cannot be read ahead, so a decision is answered at first from the orders over the review interval that ended at it, and revised when its step is filed, from the orders over the review interval that started at it, as far as the match has got by then. A decision the later orders say nothing recordable about keeps its place in the trajectory with both labels at -1, which imitation skips.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..control.policy.operations import REVIEW_MS
from ..replay.commands import NO_UNIT
from ..replay.human import (CANCEL_UNLOAD, LOAD_INTO, LOAD_UP, OTHER_MOVEMENT, REFUSED, UNLOAD, OrderBook,
                            infer_decision, new_counts)
from ..wire import AI_ORDER_NO_UNIT, AiOrderKind
from .deciders import Choice

#: The order book's kind for each kind of AI order; a stop names no destination and is not filed.
_KINDS = {
    AiOrderKind.MOVE: "move",
    AiOrderKind.ATTACK_MOVE: "attackMove",
    AiOrderKind.ATTACK: "attack",
    AiOrderKind.OTHER_MOVEMENT: OTHER_MOVEMENT,
    AiOrderKind.LOAD_INTO: LOAD_INTO,
    AiOrderKind.LOAD_UP: LOAD_UP,
    AiOrderKind.UNLOAD: UNLOAD,
    AiOrderKind.CANCEL_UNLOAD: CANCEL_UNLOAD,
}


@dataclass(frozen=True)
class _Slot:
    slot: int
    unit: Optional[int]


@dataclass(frozen=True)
class _Slots:
    """The lift layer's slots as they stood at a decision, which is all the inference reads of it."""

    slots: Tuple[_Slot, ...]


@dataclass(frozen=True)
class _Board:
    view: object
    squad: object
    region_mask: List[float]
    plan_masks: List[List[float]]
    logistics: Optional[_Slots]


class BuiltinOperations:
    """Answers the operational layer with what the watched AI player did with the squad in question."""

    #: The AI is the teacher, so what this decider plays is written as the label.
    teaches = True

    def __init__(self, review_ms: int = REVIEW_MS, weight: float = 1.0) -> None:
        self.review_ms = review_ms
        self.weight = weight
        self.book = OrderBook()
        #: Decisions by the basis they were finally labelled with, and refusals by reason, for the run's report.
        self.counts: Dict[str, int] = new_counts()
        #: Decisions the orders before them gave no answer for, so none was taken.
        self.unanswered = 0
        #: AI orders read, by kind name, and the passengers they boarded, builders apart from the rest, since a builder's lift is an economic decision and teaches the operational layer nothing.
        self.orders: Dict[str, int] = {}
        self.boarded: Dict[str, int] = {"builders": 0, "others": 0}
        self._boards: Dict[Tuple[int, int], _Board] = {}

    def observe(self, view) -> None:
        """Files the watched player's orders an operational observation carries."""
        slot = view.observation.slot
        builders = {sighting.unit.id for sighting in getattr(view, "builders", ())}
        for order in view.observation.ai_orders:
            if order.issuer != slot:
                continue
            name = AiOrderKind(order.kind).name
            self.orders[name] = self.orders.get(name, 0) + 1
            if order.kind in (AiOrderKind.LOAD_INTO, AiOrderKind.LOAD_UP):
                passengers = order.units if order.kind == AiOrderKind.LOAD_INTO else (order.target,)
                for unit in passengers:
                    key = "builders" if unit in builders else "others"
                    self.boarded[key] += 1
            kind = _KINDS.get(AiOrderKind(order.kind))
            if kind is None:
                continue
            target = NO_UNIT if order.target == AI_ORDER_NO_UNIT else order.target
            self.book.add(order.time_ms, kind, order.units, order.x, order.y, target)

    def choose(self, state, slot, region_mask, plan_masks) -> Optional[Choice]:
        raise TypeError("the built-in AI's decision is read off the board, so it is asked for with choose_on_board")

    def choose_on_board(self, view, squad, state, region_mask, plan_masks, now_ms: int,
                        logistics=None) -> Optional[Choice]:
        """The decision the orders over the review interval that ended now amount to, kept with the board so that it can be revised from the orders that follow."""
        slots = _Slots(tuple(_Slot(held.slot, held.unit) for held in logistics.slots)) if logistics is not None else None
        board = _Board(view=view, squad=_snapshot(squad), region_mask=list(region_mask),
                       plan_masks=[list(row) for row in plan_masks], logistics=slots)
        decision = infer_decision(view, board.squad, self.book, now_ms - self.review_ms, now_ms, board.region_mask,
                                  board.plan_masks, slots)
        if decision.refused:
            self.unanswered += 1
            return None
        self._boards[(squad.id, now_ms)] = board
        return Choice(action=decision.region, second=decision.plan, meta=self._meta(decision))

    def revise(self, step, now_ms: int) -> bool:
        """Relabels a step being filed from the orders over the review interval that started at its decision, up to `now_ms`; False when the step was not answered here."""
        board = self._boards.pop((step.squad, step.at_ms), None)
        if board is None:
            return False
        end = min(step.at_ms + self.review_ms, max(step.at_ms, now_ms))
        decision = infer_decision(board.view, board.squad, self.book, step.at_ms, end, board.region_mask,
                                  board.plan_masks, board.logistics)
        self.counts[decision.basis] += 1
        if decision.refused:
            step.label = step.second_label = -1
            step.meta = dict(step.meta, basis=REFUSED, reason=decision.basis)
            return True
        step.action = step.label = decision.region
        step.second = step.second_label = decision.plan
        step.meta = self._meta(decision)
        return True

    def _meta(self, decision) -> dict:
        return {"source": "builtin", "weight": round(self.weight * decision.weight, 4),
                "agreement": round(decision.agreement, 4), "basis": decision.basis}

    def report(self) -> str:
        """The decisions by basis, the refusals by reason, the AI orders read and the passengers boarded, as one line."""
        counts = ", ".join(f"{basis} {count}" for basis, count in self.counts.items())
        orders = ", ".join(f"{kind.lower()} {count}" for kind, count in sorted(self.orders.items()))
        boarded = ", ".join(f"{kind} {count}" for kind, count in self.boarded.items())
        return f"{counts}; unanswered {self.unanswered}; AI orders: {orders or 'none'}; boarded: {boarded}"


def _snapshot(squad):
    """A copy of the squad record that later periods cannot change under it."""
    copied = copy.copy(squad)
    copied.members = list(squad.members)
    return copied


def merge(deciders: Sequence[BuiltinOperations]) -> BuiltinOperations:
    """One decider's worth of counts summed over several, for a run's report."""
    total = BuiltinOperations()
    for decider in deciders:
        for basis, count in decider.counts.items():
            total.counts[basis] += count
        total.unanswered += decider.unanswered
        for kind, count in decider.orders.items():
            total.orders[kind] = total.orders.get(kind, 0) + count
        for kind, count in decider.boarded.items():
            total.boarded[kind] += count
    return total
