"""What each layer is paid for.

There is one rule and the rest follows from it: a layer is paid for how well it met the contract its superior handed down, and it never sees its superior's reward. Only the strategic layer is paid the match. That is what removes credit assignment across layer boundaries — the thing the design gives as its reason for not learning the interfaces between layers — and it is also what shortens the tactical horizon to the length of one errand rather than the length of a match, which is the only horizon this sample budget can learn over at all.

Both layers are shaped potentially. A shaping term of the form gamma times the potential after minus the potential before cannot change which policy is optimal, whatever the potential is, so a term that turns out to have been a bad idea costs sample efficiency and never correctness. Everything continuous here is therefore in a potential, and only the discrete outcome of an errand — taken and held, out of time, or lost — is paid directly.

That guarantee holds on one condition, and it is a convention rather than an observation: the potential of the state an errand ended in is taken to be nought. The shaping only telescopes to the difference of two potentials if the last term is nought less what was being held, and a last term that used the real potential of the ending board would leave a residue which depends on where the errand ended, which is exactly the dependence potential shaping exists to remove. The terminal states are not board positions the potential is defined on anyway, so what they are worth is this side's to decide, and nought is the choice that makes the arithmetic hold.

The potentials are written in shares rather than in credits for the same reason the features are: an errand fought over four tanks and an errand fought over forty are the same errand, and a reward that grew with the size of the armies would make the same behaviour worth more later in a match than earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..wire import Status
from ..control.policy.contracts import SquadRecord
from ..control.policy.view import WorldView

#: What the tactical potential is made of: standing on the ground the contract named, not having spent the allowance, and having destroyed more than has been lost.
#:
#: The exchange weighs most, and the reason is credit assignment rather than importance. An errand is about a hundred and twenty five decisions long at the tactical rate, and with a discount of a hundredth and a trace of five hundredths the terminal reaches the first of them weighted by about four ten-thousandths. Nothing the layer decides in the opening of a fight is therefore taught by how the fight ended; what teaches it is the shaping, and the shaping only teaches the right thing if it points where the terminal points. Trading well is what the terminal is, so trading well is what the dense term has to be. Measured before this term existed, two hundred updates moved the return not at all while the entropy of the policy climbed steadily, which is the signature of a gradient made of noise and a shaping term aimed elsewhere.
#:
#: The old pair on their own aimed somewhere else in a way worth naming: not having spent the allowance rewards not fighting, and a layer whose only dense signal says that will find the quietest way through every engagement.
HOLDING_WEIGHT = 0.35
SPENDING_WEIGHT = 0.15
EXCHANGE_WEIGHT = 0.5

#: Credits added to both sides of the exchange ratio before it is taken, so that a fight in which nothing has happened yet reads as even rather than swinging to one end on the first shot. A tank costs about this, which makes the first kill worth about two thirds rather than all of the term.
EXCHANGE_PRIOR = 300.0

#: Paid once, when an errand ends. Taking and holding the region is the errand done; running out of time is a failure that at least did not cost anything in particular; being reported as losing means the budget went and the region did not come, which is the outcome the whole cost budget mechanism exists to make expensive.
COMPLETE_REWARD = 1.0
EXPIRED_REWARD = -0.5
LOSING_REWARD = -1.0

#: Paid when a squad ceases to exist while under contract. Distinguished from being reported as losing because a squad that is gone cannot be withdrawn and the layer had every period until then to withdraw it.
WIPED_REWARD = -1.0

#: How much over budget an errand may run before the overspend is charged against a completion. A region taken at twice its allowance was not taken at the price it was commissioned at.
OVERSPEND_PENALTY = 0.5

#: Discount used in the shaping term. Not the discount a learner uses for its returns, though it should be given the same one: this is only the factor that makes the shaping telescope, and a mismatch would leave a residue that does change the optimal policy.
#:
#: This is the figure for an errand that is a fragment of a longer match. A constructed fight is not that — it is one errand from beginning to end, over inside a minute — and what it is discounted at is handed in rather than read from here. See the fight-scoped figures in the rollout.
DISCOUNT = 0.99

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
    #: True on the period a new contract replaced the one a squad was working to. The errand that was replaced did not fail and did not finish; it stopped being the thing the squad is doing, so a caller keeping trajectories has to end the old one here rather than letting the decisions of two errands sit in one.
    renewed: bool = False


@dataclass
class _Mission:
    """A tactical layer's memory of one errand, which is the span its reward is cut across."""

    issued_at_ms: int
    potential: float = 0.0
    #: What the potential was when the errand opened, kept so that an errand ended from outside can be paid its terminal with the whole of its shaping cancelled rather than with the opening term left standing. See `OperationalReward.close`.
    opening: float = 0.0
    started: bool = False
    #: True once this errand has been paid its one terminal, so that it cannot be paid another.
    ended: bool = False


class TacticalReward:
    """Pays the tactical layer for the errand it was given, one squad at a time.

    An errand is bounded by its contract. When the operational layer issues a new one the old episode ends and a new one begins, which is exactly the boundary the design wants the tactical horizon to close at: ten to sixty seconds, not a match. That the boundary is drawn by another layer's decision is not a problem to be solved but the arrangement itself — the contract is the unit of work, so it is the unit of pay.

    An errand is paid one terminal and no more. The memory of a finished errand is therefore kept rather than dropped, keyed by the moment its contract was issued: the conditions that end one — the ground taken, the allowance gone, the deadline past — are read from the board and stay true for as long as the board stays that way, so an errand whose memory was dropped would be started afresh on the next period, meet the same condition, and be paid again. Measured before this was so, a squad reported as losing was paid the whole of that penalty every other period until its fight ended, and one run of four hundred and forty fights paid four thousand three hundred and forty nine of them.

    Where an errand ends is not the same question everywhere it is used. In a match the operational layer reissues contracts, so the conditions of the contract are what bound the errand and it is right that they end it. On a board where engagements are constructed there is one contract for the whole fight and nothing reissues it, so ending the errand early would leave the rest of the fight unpaid while the squad went on fighting it. `status_terminals` is which of the two this is.
    """

    def __init__(self, status_terminals: bool = True, discount: float = DISCOUNT) -> None:
        self.missions: Dict[int, _Mission] = {}
        self.status_terminals = status_terminals
        # The factor the shaping telescopes with, which has to be the one the returns are discounted at or the shaping leaves a residue and stops being harmless. An argument rather than the constant because the two places this layer is trained have errands of different lengths: a match reissues contracts and an errand is a fragment of it, while a constructed fight is one errand from end to end and is short enough to be discounted at nothing at all.
        self.discount = discount

    def forget(self, squad_id: int) -> None:
        self.missions.pop(squad_id, None)

    def ended(self, squad_id: int) -> bool:
        """Whether this squad's errand has already been paid its terminal, which is what a caller ending errands from outside has to ask before paying another."""
        mission = self.missions.get(squad_id)
        return mission is not None and mission.ended

    def close(self, squad_id: int) -> float:
        """Hands back the potential this squad's errand was last valued at and forgets the errand, so that a caller ending it from outside can pay the shaping term itself.

        Shaping is only harmless if it telescopes to nothing over an episode, and it only does that if the potential of a terminal state is taken to be nought. An errand closed from outside has therefore to be paid its last shaping term as nought minus whatever was being held, and this is where that figure comes from. Nought when nothing is held, which is the case for a squad whose errand ended in the same period it was given, so the answer is always a number that can be paid.
        """
        mission = self.missions.pop(squad_id, None)
        return mission.potential if mission is not None else 0.0

    def step(self, squad: SquadRecord, view: WorldView, game_time_ms: int,
             killed: float = 0.0) -> Outcome:
        contract = squad.contract
        if contract is None:
            self.forget(squad.id)
            return Outcome()

        mission = self.missions.get(squad.id)
        if mission is None or mission.issued_at_ms != contract.issued_at_ms:
            # A fresh contract is a fresh errand. The potential is taken now and paid from the next period, so that the step which merely received the contract is not paid for the board it arrived on.
            replaced = mission is not None
            mission = _Mission(issued_at_ms=contract.issued_at_ms,
                               potential=self._potential(squad, view, killed))
            self.missions[squad.id] = mission
            return Outcome(renewed=replaced)

        # An errand that has already been paid its terminal is over. Decisions taken about the squad afterwards belong to no errand until a contract issues a new one, and paying them would be paying for work nobody asked for.
        if mission.ended:
            return Outcome()

        terminal, reason = self._terminal(squad, contract)
        if terminal is not None:
            # The potential of a state an errand ended in is nought by convention, so the last shaping term is nought less whatever was being held rather than the discounted potential of the ending board. Paying the real potential would leave a residue proportional to it, the residue would differ between one ending and another, and a shaping term that differs by outcome is a term that moves which policy is best -- the single thing potential-based shaping was chosen to rule out. Taking the ground is where it bit hardest: that board scores near the top of the potential, so a completion was worth about two thirds of a point more than the terminal it is defined to be worth.
            mission.ended = True
            return Outcome(reward=(0.0 - mission.potential) + terminal, done=True, reason=reason)

        potential = self._potential(squad, view, killed)
        reward = self.discount * potential - mission.potential
        mission.potential = potential
        return Outcome(reward=reward)

    def _potential(self, squad: SquadRecord, view: WorldView, killed: float = 0.0) -> float:
        contract = squad.contract
        target = view.region(contract.target_region) if contract is not None else None
        holding = _share(target.our_value, target.enemy_value) if target is not None else 0.5
        budget = contract.cost_budget if contract is not None else 0.0
        spent = min(1.0, squad.losses / budget) if budget > 0 else 0.0
        # Worth destroyed against worth lost, over the life of this contract, with a unit's price added to both so that nothing having happened reads as even. This is the running form of the figure an engagement is finally scored on, which is the whole reason it is here.
        exchange = _share(killed + EXCHANGE_PRIOR, squad.losses + EXCHANGE_PRIOR)
        return (HOLDING_WEIGHT * holding + SPENDING_WEIGHT * (1.0 - spent)
                + EXCHANGE_WEIGHT * exchange)

    def _terminal(self, squad: SquadRecord, contract) -> Tuple[Optional[float], str]:
        # Where one contract stands for a whole fight, what ends the fight ends the errand and nothing else does. Being reported as losing is then something the layer is meant to act on rather than something the errand is over because of, and the squad that acts on it goes on fighting under the same contract for another half minute; ending its errand there would leave every decision in that half minute unpaid. A squad destroyed is no exception, even though its errand plainly is over: what it is worth depends on how much it took with it, and the only figure that knows that is the score of the fight, which is handed in from outside.
        if not self.status_terminals:
            return None, ""
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
    """Pays the operational layer for meeting the strategic layer's orders, one squad at a time.

    The match is not in the potential. The design gives the terminal result of the match to the strategic layer alone, and a layer that could see it would be learning to win rather than to carry out the orders it was given — which sounds like an improvement until the strategic layer is changed and everything below it has to be learnt again. What is here instead is the strategic layer's own statement of what it wants: how much of the ground it called valuable is being stood on.

    Per squad, and region-specific, because a single global board figure written identically into every squad's step was the disease. The per-decision advantage barely depended on which region a squad was sent to, so only the entropy bonus had a consistent gradient and the policy spread toward uniform while the return sat still: a dead gradient. The potential of a decision is the priority-weighted domination of the one region that decision's contract named — the share of that region that is ours, less an even split, times what the strategic layer said the region was worth — so a squad sent to a region it took and a squad sent to one it lost are paid differently, and the shaping already points where the choice does. Keyed by the contract's issue time exactly as the tactical layer's errand is, so a squad handed a new region begins a fresh mission and the two are not run into one trajectory.

    The domination rather than the bare share, and that is not a presentational choice: shaping telescopes, so what survives an errand is the terminal less the potential the errand opened at, and a potential written in a different quantity from the terminal leaves the difference between the two quantities standing in every return. See `_potential` for what that residue was and which way it pointed.

    There is no status terminal here. What ends an operational errand — the region taken, the deadline past — is not read from the board and paid the way the tactical layer's is; the constructed arena that this per-squad form exists for pays a region-domination terminal from outside through `finish`, and a match pays none at all (the match result is the strategic layer's). So `step` only ever shapes and renews, and `close`/`ended` are here for the outside terminal to telescope against.
    """

    def __init__(self, discount: float = DISCOUNT) -> None:
        self.missions: Dict[int, _Mission] = {}
        # The factor the shaping telescopes with, which has to be the one the returns are discounted at or the shaping leaves a residue and stops being harmless. An argument rather than the constant for the same reason the tactical layer's is: a match discounts an operational errand as a fragment of itself, while the constructed arena is one errand from end to end and discounts it at nothing.
        self.discount = discount

    def forget(self, squad_id: int) -> None:
        self.missions.pop(squad_id, None)

    def ended(self, squad_id: int) -> bool:
        """Whether this squad's errand has already been paid its terminal, which a caller ending errands from outside asks before paying another."""
        mission = self.missions.get(squad_id)
        return mission is not None and mission.ended

    def close(self, squad_id: int) -> float:
        """Hands back how far this squad's errand moved its potential — the last valuation less the opening one — and forgets the errand, so that a caller ending it from outside pays `terminal − (last − opening)` and the errand's whole return comes to the terminal exactly. Nought when nothing is held.

        The cancellation is exact at a discount of one, which is where it has to be: a terminal only ever arrives from the constructed arena, an arena contest is one whole bounded errand, and such an errand is discounted at nothing. A match pays no operational terminal at all, so nothing here runs under the match's discount.

        The difference and not the last valuation, which is what it was, and the distinction is the whole alignment of this layer's signal. The shaping sums over an errand to `last − opening`; subtracting only `last` leaves `− opening` standing in the return, and the opening valuation is taken from the region the decision itself named. So the residue was a function of the action: a squad sent at ground the enemy already held opened near nought and kept its whole terminal, while a squad sent to hold ground that was already ours opened near the top and had that much taken off it. **Taking a region from lost to level paid, and holding a region that was already won paid nothing**, although the arena's own score says the first is worth nothing and the second is worth half that region's priority. Cancelling the opening term as well leaves the return equal to the terminal, whatever the terminal is: under the arena's region credit that is the contracted region's own contribution to the side score, and under its marginal credit the part of that contribution the squad's own units account for. The identity this guarantees is the terminal alone, not any particular reading of it.
        """
        mission = self.missions.pop(squad_id, None)
        return mission.potential - mission.opening if mission is not None else 0.0

    def reset(self) -> None:
        self.missions.clear()

    def step(self, squad: SquadRecord, view: WorldView, orders) -> Outcome:
        contract = squad.contract
        if contract is None:
            self.forget(squad.id)
            return Outcome()

        mission = self.missions.get(squad.id)
        if mission is None or mission.issued_at_ms != contract.issued_at_ms:
            # A fresh contract is a fresh errand. The potential is taken now and paid from the next period, so the step that merely received the contract is not paid for the board it arrived on. It is kept as the opening as well, because what the errand returns has to be the terminal alone and the shaping has to cancel whole — see `close`.
            replaced = mission is not None
            opening = self._potential(squad, view, orders)
            mission = _Mission(issued_at_ms=contract.issued_at_ms, potential=opening, opening=opening)
            self.missions[squad.id] = mission
            return Outcome(renewed=replaced)

        if mission.ended:
            return Outcome()

        potential = self._potential(squad, view, orders)
        reward = self.discount * potential - mission.potential
        mission.potential = potential
        return Outcome(reward=reward)

    def _potential(self, squad: SquadRecord, view: WorldView, orders) -> float:
        contract = squad.contract
        if contract is None:
            return 0.0
        region = view.region(contract.target_region)
        if region is None:
            return 0.0
        # The priority the strategic layer put on the region this squad is contracted to, times how much of that region is ours less an even split. A region the strategic layer did not ask for carries no priority and so no shaping, which is the point: the layer is paid for meeting the orders, not for holding ground nobody wanted.
        #
        # The domination and not the bare share, because the potential has to be written in the same quantity as the terminal it telescopes against, and it was not. The shaping cancels over an errand except for its first term, so a decision's return comes to the terminal less the potential it started at: with the potential at `priority * share` on nought to one and the arena's terminal at `priority * (share - a half)`, that return carried a standing `- 0.5 * priority` — an offset with no board in it at all, which every squad paid in proportion to how valuable the region it was sent to was. Sending a squad at the most wanted region on the board cost it half of that region's priority before the fighting was scored, and the ground it could win back was at most the same priority again. The layer was being taught to leave the valuable ground alone. Written as the domination the offset is nought and the return is exactly `priority * (share at the horizon - share at issue)`, which is what the errand did to the quantity the arena is measured by.
        priority = orders.priorities.get(region.id, 0.0) if orders is not None else 0.0
        return priority * (_share(region.our_value, region.enemy_value) - 0.5)
