"""What each layer is paid for.

There is one rule and the rest follows from it: a layer is paid for how well it met the contract its superior handed down, and it never sees its superior's reward. Only the strategic layer is paid the match. That is what removes credit assignment across layer boundaries — the thing the design gives as its reason for not learning the interfaces between layers — and it is also what shortens the tactical horizon to the length of one errand rather than the length of a match, which is the only horizon this sample budget can learn over at all.

Both layers are shaped potentially. A shaping term of the form gamma times the potential after minus the potential before cannot change which policy is optimal, whatever the potential is, so a term that turns out to have been a bad idea costs sample efficiency and never correctness. Everything continuous here is therefore in a potential, and only the discrete outcome of an errand — taken and held, out of time, or lost — is paid directly.

The potentials are written in shares rather than in credits for the same reason the features are: an errand fought over four tanks and an errand fought over forty are the same errand, and a reward that grew with the size of the armies would make the same behaviour worth more later in a match than earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from ..wire import Status
from ..control.policy.contracts import SquadRecord
from ..control.policy.view import WorldView

#: How much of the tactical potential is standing on the ground the contract named, against how much is not having spent the budget. Holding weighs more because taking the region is what the errand is; the budget is a constraint on how, not the object.
HOLDING_WEIGHT = 0.7
SPENDING_WEIGHT = 0.3

#: Paid once, when an errand ends. Taking and holding the region is the errand done; running out of time is a failure that at least did not cost anything in particular; being reported as losing means the budget went and the region did not come, which is the outcome the whole cost budget mechanism exists to make expensive.
COMPLETE_REWARD = 1.0
EXPIRED_REWARD = -0.5
LOSING_REWARD = -1.0

#: Paid when a squad ceases to exist while under contract. Distinguished from being reported as losing because a squad that is gone cannot be withdrawn and the layer had every period until then to withdraw it.
WIPED_REWARD = -1.0

#: How much over budget an errand may run before the overspend is charged against a completion. A region taken at twice its allowance was not taken at the price it was commissioned at.
OVERSPEND_PENALTY = 0.5

#: Discount used in the shaping term. Not the discount a learner uses for its returns, though it should be given the same one: this is only the factor that makes the shaping telescope, and a mismatch would leave a residue that does change the optimal policy.
DISCOUNT = 0.99

#: What the operational potential is made of: how much of what the strategic layer said it wanted is actually being stood on, how far ahead our army is, and how much of the allowance has gone.
PRIORITY_WEIGHT = 0.5
EDGE_WEIGHT = 0.4
ALLOWANCE_WEIGHT = 0.1


def _share(part: float, against: float) -> float:
    total = part + against
    return part / total if total > 0 else 0.5


@dataclass
class Outcome:
    """One period's pay for one decision, and whether the thing it was paid for has finished."""

    reward: float = 0.0
    done: bool = False
    #: What ended the errand, for a log that has to be read by a person rather than by an optimiser.
    reason: str = ""


@dataclass
class _Mission:
    """A tactical layer's memory of one errand, which is the span its reward is cut across."""

    issued_at_ms: int
    potential: float = 0.0
    started: bool = False


class TacticalReward:
    """Pays the tactical layer for the errand it was given, one squad at a time.

    An errand is bounded by its contract. When the operational layer issues a new one the old episode ends and a new one begins, which is exactly the boundary the design wants the tactical horizon to close at: ten to sixty seconds, not a match. That the boundary is drawn by another layer's decision is not a problem to be solved but the arrangement itself — the contract is the unit of work, so it is the unit of pay.
    """

    def __init__(self) -> None:
        self.missions: Dict[int, _Mission] = {}

    def forget(self, squad_id: int) -> None:
        self.missions.pop(squad_id, None)

    def step(self, squad: SquadRecord, view: WorldView, game_time_ms: int) -> Outcome:
        contract = squad.contract
        if contract is None:
            self.forget(squad.id)
            return Outcome()

        mission = self.missions.get(squad.id)
        if mission is None or mission.issued_at_ms != contract.issued_at_ms:
            # A fresh contract is a fresh errand. The potential is taken now and paid from the next period, so that the step which merely received the contract is not paid for the board it arrived on.
            mission = _Mission(issued_at_ms=contract.issued_at_ms,
                               potential=self._potential(squad, view))
            self.missions[squad.id] = mission
            return Outcome()

        potential = self._potential(squad, view)
        reward = DISCOUNT * potential - mission.potential
        mission.potential = potential

        terminal, reason = self._terminal(squad, contract)
        if terminal is None:
            return Outcome(reward=reward)
        self.forget(squad.id)
        return Outcome(reward=reward + terminal, done=True, reason=reason)

    def _potential(self, squad: SquadRecord, view: WorldView) -> float:
        contract = squad.contract
        target = view.region(contract.target_region) if contract is not None else None
        holding = _share(target.our_value, target.enemy_value) if target is not None else 0.5
        budget = contract.cost_budget if contract is not None else 0.0
        spent = min(1.0, squad.losses / budget) if budget > 0 else 0.0
        return HOLDING_WEIGHT * holding + SPENDING_WEIGHT * (1.0 - spent)

    def _terminal(self, squad: SquadRecord, contract) -> Tuple[Optional[float], str]:
        if not squad.members:
            return WIPED_REWARD, "wiped"
        if squad.status is Status.COMPLETE:
            overspend = 0.0
            if contract.cost_budget > 0 and squad.losses > contract.cost_budget:
                overspend = OVERSPEND_PENALTY * min(1.0, squad.losses / contract.cost_budget - 1.0)
            return COMPLETE_REWARD - overspend, "complete"
        if squad.status is Status.LOSING:
            return LOSING_REWARD, "losing"
        if squad.status is Status.EXPIRED:
            return EXPIRED_REWARD, "expired"
        return None, ""


class OperationalReward:
    """Pays the operational layer for meeting the strategic layer's orders.

    Three things are in the potential and the match is not one of them. The design gives the terminal result of the match to the strategic layer alone, and a layer that could see it would be learning to win rather than learning to carry out the orders it was given — which sounds like an improvement until the strategic layer is changed and everything below it has to be learnt again. What is here instead is the strategic layer's own statement of what it wants: which regions it called valuable, how far ahead the army is, and how much of the loss allowance has gone.
    """

    def __init__(self) -> None:
        self.potential: Optional[float] = None
        self.spent = 0.0

    def reset(self) -> None:
        self.potential = None
        self.spent = 0.0

    def step(self, view: WorldView, orders, squads: Sequence[SquadRecord]) -> Outcome:
        potential = self._potential(view, orders, squads)
        if self.potential is None:
            self.potential = potential
            return Outcome()
        reward = DISCOUNT * potential - self.potential
        self.potential = potential
        return Outcome(reward=reward)

    def _potential(self, view: WorldView, orders, squads: Sequence[SquadRecord]) -> float:
        priorities = orders.priorities if orders is not None else {}
        wanted = sum(priorities.values())
        covered = 0.0
        for region in view.regions:
            priority = priorities.get(region.id, 0.0)
            if priority > 0:
                covered += priority * _share(region.our_value, region.enemy_value)
        coverage = covered / wanted if wanted > 0 else 0.5

        ours = sum(s.value for s in view.fighters)
        theirs = sum(s.value for s in view.enemies)
        edge = _share(ours, theirs)

        allowance = orders.loss_allowance if orders is not None else 0.0
        losses = sum(squad.losses for squad in squads)
        spent = min(1.0, losses / allowance) if allowance > 0 else 0.0

        return PRIORITY_WEIGHT * coverage + EDGE_WEIGHT * edge + ALLOWANCE_WEIGHT * (1.0 - spent)
