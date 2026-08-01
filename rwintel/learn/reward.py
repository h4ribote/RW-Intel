"""What each layer is paid for.

There is one rule and the rest follows from it: a layer is paid for how well it met the contract its superior handed down, and it never sees its superior's reward. Only the strategic layer is paid the match. That is what removes credit assignment across layer boundaries — the thing the design gives as its reason for not learning the interfaces between layers — and it is also what shortens the tactical horizon to the length of one errand rather than the length of a match, which is the only horizon this sample budget can learn over at all.

All three layers are shaped potentially. A shaping term of the form gamma times the potential after minus the potential before cannot change which policy is optimal, whatever the potential is, so a term that turns out to have been a bad idea costs sample efficiency and never correctness. Everything continuous here is therefore in a potential, and only the discrete outcome of an errand — taken and held, out of time, or lost — is paid directly.

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
#: The exchange weighs most, and the reason was credit assignment rather than importance — in a regime that is no longer the one this layer is trained in. An errand is about a hundred and twenty five decisions long at the tactical rate, and where an errand is a fragment of a match, discounted at ninety-nine hundredths with an advantage trace of ninety-five, the terminal reaches the first of those decisions weighted by about four ten-thousandths. Nothing decided in the opening of a fight is taught there by how the fight ended; what teaches it is the shaping, and the shaping only teaches the right thing if it points where the terminal points. Trading well is what the terminal is, so trading well is what the dense term had to be. Measured before this term existed, two hundred updates moved the return not at all while the entropy of the policy climbed steadily, which is the signature of a gradient made of noise and a shaping term aimed elsewhere.
#:
#: On a constructed fight that argument does not hold, and none of these three weights reaches the policy at all. A fight is one whole errand, discounted at nothing with a trace of one, and at those figures the shaping telescopes over the errand into a single term: the return of a decision is the score of the fight less the potential held at that decision, and the advantage is that less what the critic expected. The potential is read at the top of the period, before the departure is chosen, because the previous decision is settled first and the choice is made afterwards — so the whole dense term enters every return as a quantity already fixed when the action is taken. A quantity that does not depend on the action contributes nothing to the policy gradient, and a critic that has learnt it leaves the advantage exactly the number it would have been with no shaping at all. What these weights still do on a fight is give the critic a different target to fit; what they cannot do is teach one departure over another, so an arm run without the exchange term could not measure a difference in what was taught. The figure is left where it is rather than retuned for that reason: on a fight it cannot matter, and off a fight it is the figure that was argued for.
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


class StrategicReward:
    """Pays the strategic layer the match, which is the one layer in this design that is paid the match at all.

    Everything below it is paid for meeting the contract handed down to it and never sees the result; this is where that stops. The rule's whole point is that credit assignment does not cross a layer boundary, and it does not have to here, because there is nothing above this layer to hand it a contract — what it is asked for IS the match, so what it is paid is the match.

    One errand per match and not one per squad. The layer decides one thing about the whole side every ten seconds, so there is nothing to key a mission by: the errand opens with the match and closes with it, and `close`/`ended` are here for the outside terminal to telescope against. What the two layers' `close` hand back is not the same figure, and the difference is deliberate: the operational one hands back the movement since its errand opened, so that a contest returns its terminal exactly, while this one hands back only the last valuation, so that the shaping telescopes at any discount and the match returns the terminal less the potential it opened at. See this class's own `close` for why that trade goes this way here.

    The potential is the running form of the very quantity the match is scored on. An episode that was cut off is scored on the military edge — the value standing on our side against the strongest opponent's, as a ratio — and that same edge can be read off any board in flight, so the shaping points exactly where the terminal points. That is the property the tactical layer's exchange term was chosen for and the one the operational layer's per-region potential had to be rewritten to get: a dense term that disagrees with the terminal teaches the opening of an episode to do the opposite of what the ending pays for.

    Two limits on that alignment, and both are honest rather than incidental. **The economy term cannot be read in flight at all**: the score's economy component compares our income with the enemy's, and the enemy's income is not observable from inside a match (only our own aggregates are, and the design says so). So the potential carries the military weight alone, and if the score's weights are ever fitted away from military-only, this potential answers a different question than the terminal and has to be revisited. **And the military edge here is read from what this side commands against what it can see**, where the terminal's is read from the standing the game reports for every team at the end. With the fog off those are the same board; with it on, the potential is a partial reading of the quantity the terminal settles.
    """

    def __init__(self, discount: float = DISCOUNT, military_weight: float = 1.0) -> None:
        #: One errand for the whole match, so this is one mission and not a dictionary of them.
        self.mission: Optional[_Mission] = None
        # The factor the shaping telescopes with, handed in for the reason the other two layers' is: whoever builds the layer knows how long the errand is. A match is one whole errand from this layer's seat, which is the case discounted at nothing.
        self.discount = discount
        # What share of the score the potential is the running form of. The blend the episodes are scored with is military-only today, and this is that weight rather than a one so that the two cannot silently disagree.
        self.military_weight = military_weight

    def forget(self) -> None:
        self.mission = None

    def ended(self) -> bool:
        """Whether this match's errand has already been paid its terminal, which a caller ending it from outside asks before paying another."""
        return self.mission is not None and self.mission.ended

    def close(self) -> float:
        """Hands back the valuation this match was last held at and forgets it, so that a caller ending it from outside pays `terminal − that`. Nought where nothing is held, which is a match that ended before a first decision was taken.

        The last valuation and not the movement since the opening, which is what it was. Handing back the movement made the match's whole return come to the terminal exactly, which reads well and is only true at a discount of one: the shaping paid over a match is the sum of `discount × potential after − potential before`, and that sum is the movement only when the discount is one. Below one the two differ by an amount that depends on the path the match took, so what was left standing in the return was a residue a policy could move — which is the single thing potential-based shaping is chosen to rule out, and this layer's discount is a run's to set.

        Paid this way the match returns the terminal less the potential it opened at, at any discount. That opening is the board the match started on, before this layer had taken a decision, so it is a constant of the episode and not something a posture can move; what it costs is that the mean return of a run is no longer readable as the mean score of its matches, and what it buys is that the shaping cannot change which posture is best.
        """
        mission, self.mission = self.mission, None
        return mission.potential if mission is not None else 0.0

    def reset(self) -> None:
        self.mission = None

    def step(self, report) -> Outcome:
        """One strategic period's shaping, paid to the decision the period before it left waiting.

        The first period opens the errand and pays nothing: the potential is read there and paid from the next period, so the decision that merely started the match is not paid for the board it started on.
        """
        potential = self._potential(report)
        if self.mission is None:
            # No opening is kept, because nothing here reads one: this layer's terminal is paid against the last valuation alone, which is what makes the shaping telescope at any discount. The field exists for the layer whose terminal cancels the opening as well — see `OperationalReward.close`.
            self.mission = _Mission(issued_at_ms=0, potential=potential)
            return Outcome()
        if self.mission.ended:
            return Outcome()
        reward = self.discount * potential - self.mission.potential
        self.mission.potential = potential
        return Outcome(reward=reward)

    def _potential(self, report) -> float:
        """The military edge as the score defines it: ours less theirs over their sum, on −1 to +1, which is twice the share less a half. Written as the edge rather than as the share so that it is the same number the terminal is, and so that an even board is nought and carries no constant a discount below one would charge for.

        Everything standing on both sides, which is what the game's own standing counts and therefore what the terminal is a ratio of. The report also carries the mobile armed force on both sides, and that is the right pair for a loss allowance and the wrong one here: a potential that left our buildings out of our side and theirs into theirs — which is what the report used to offer as its only pair — is not the running form of anything the match is scored on.
        """
        ours = float(getattr(report, "our_value", 0.0))
        theirs = float(getattr(report, "enemy_value", 0.0))
        return self.military_weight * 2.0 * (_share(ours, theirs) - 0.5)


class OperationalReward:
    """Pays the operational layer for meeting the strategic layer's orders, one squad at a time.

    The match is not in the potential. The design gives the terminal result of the match to the strategic layer alone, and a layer that could see it would be learning to win rather than to carry out the orders it was given — which sounds like an improvement until the strategic layer is changed and everything below it has to be learnt again. What is here instead is the strategic layer's own statement of what it wants: how much of the ground it called valuable is being stood on.

    Per squad, and region-specific, because a single global board figure written identically into every squad's step was the disease. The per-decision advantage barely depended on which region a squad was sent to, so only the entropy bonus had a consistent gradient and the policy spread toward uniform while the return sat still: a dead gradient. The potential of a decision is the priority-weighted domination of the one region that decision's contract named — the share of that region that is ours, less an even split, times what the strategic layer said the region was worth — so a squad sent to a region it took and a squad sent to one it lost are paid differently, and the shaping already points where the choice does.

    One ledger a squad, opened when the squad is first seen and never re-based, whichever board is in force. It was once re-based at every fresh contract, on the ground that two contracts are two errands; what that missed is that both errands are priced out of the same table — the region block's priority times its domination — so the difference across the boundary is a difference of one quantity and the only thing re-basing did was decline to pay it. A policy could therefore keep what a region had earned it and walk away before the region lost it, which is the identical fault the scored ledger is written the way it is to prevent.

    Which board is in force decides which of two quantities a period is paid in, and it is one or the other and never both. In a match the region block above is the whole of the signal: there is no operational terminal at all — the match result is the strategic layer's — and nothing outside can read anything better, so the block's own movement is what the layer is taught by. On a contest that scores its own ground the arena hands in a figure every period and `step` pays the movement of that instead, through `_scored`; the block is not added to it, because two densities of one objective can be summed and two different quantities cannot.

    The two are not the same figure and were once written here as though they were. On the constructed operations arena the scored figure is read off the unit rows — a health-weighted disc of fixed radius about a contest point — while the block is the game's own region cell at unit prices, a Voronoi block about a region centre with this side's free base counted into it: not the same table and not the same ground. Nothing would make them telescope against each other, which is why the switch is per board rather than per squad — a trajectory some of whose steps were paid in one and some in the other sums to neither.

    There is no status terminal here either way. What ends an operational errand — the region taken, the deadline past — is not read from the board and paid the way the tactical layer's is; a contest pays its terminal from outside through `finish` and a match pays none at all. So `step` only ever shapes, and `close`/`ended` are here for the outside terminal to telescope against.
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
        """Hands back everything this squad has already been paid — the last valuation less the opening one — and forgets the errand, so that a caller ending it from outside pays `terminal − (last − opening)` and the errand's whole return comes to the terminal exactly. Nought when nothing is held.

        One statement covers both boards, which is why this needs no case of its own. On the region block the opening is the valuation the errand began at and the difference is the errand's own movement, which is the sum of the shaping terms already paid. On a scored contest the opening is nought and the difference is the last figure, which is likewise the sum of the differences already paid, every period having been paid one. Either way what is handed back is what has been paid, so what the caller adds is what has not.

        The cancellation is exact at a discount of one, which is where it has to be: a terminal only ever arrives from the constructed arena, an arena contest is one whole bounded errand, and such an errand is discounted at nothing. A match pays no operational terminal at all, so nothing here runs under the match's discount.

        The difference and not the last valuation, which is what it was, and the distinction is the whole alignment of this layer's signal. The shaping sums over an errand to `last − opening`; subtracting only `last` leaves `− opening` standing in the return, and the opening valuation is taken from the region the decision itself named. So the residue was a function of the action: a squad sent at ground the enemy already held opened near nought and kept its whole terminal, while a squad sent to hold ground that was already ours opened near the top and had that much taken off it. **Taking a region from lost to level paid, and holding a region that was already won paid nothing**, although the arena's own score says the first is worth nothing and the second is worth half that region's priority. Cancelling the opening term as well leaves the return equal to the terminal, whatever the terminal is: under the arena's region credit that is the contracted region's own contribution to the side score, and under its marginal credit the part of that contribution the squad's own units account for. The identity this guarantees is the terminal alone, not any particular reading of it.
        """
        mission = self.missions.pop(squad_id, None)
        return mission.potential - mission.opening if mission is not None else 0.0

    def reset(self) -> None:
        self.missions.clear()

    def step(self, squad: SquadRecord, view: WorldView, orders,
             figure: Optional[float] = None) -> Outcome:
        if figure is not None:
            return self._scored(squad, figure)

        contract = squad.contract
        potential = self._potential(squad, view, orders)

        mission = self.missions.get(squad.id)
        if mission is None:
            # The first period this squad is seen in opens the ledger. The potential is taken now and paid from the next period, so the step that merely brought the squad into view is not paid for the board it arrived on. It is kept as the opening as well, so that what has been paid can be handed back whole — see `close`.
            mission = _Mission(issued_at_ms=contract.issued_at_ms if contract is not None else 0,
                               potential=potential, opening=potential)
            self.missions[squad.id] = mission
            return Outcome()

        if mission.ended:
            return Outcome()

        # One ledger from the first period to the last, never re-based on a fresh contract, which is the ledger the scored board already keeps and for the same reason. Re-basing paid nothing at all in the period that re-tasked a squad: the rise the old region had been paying for was kept and the fall that was coming was never charged, so a policy could bank a gain and duck a loss by writing a new contract — a change of objective wearing the clothes of a change of density, and one this layer could help itself to at will. Paid as a difference of the same quantity, the period that re-tasks a squad hands back the ground it is leaving and takes on the ground it is going to, in one number, and the whole episode telescopes to its last reading.
        #
        # A squad between contracts is paid rather than forgotten, for the same reason: it is holding no contracted ground, its potential is nought, and the honest payment is nought less what the ledger held. Dropped instead, the ledger would restart with that hand-back unpaid.
        reward = self.discount * potential - mission.potential
        mission.potential = potential
        if contract is not None:
            mission.issued_at_ms = contract.issued_at_ms
        return Outcome(reward=reward)

    def _scored(self, squad: SquadRecord, figure: float) -> Outcome:
        """Pays this period the movement of the squad's own scored figure, which is what a contest hands in when it can read its own ground.

        Three properties are load-bearing and none of them is arithmetic convenience.

        The ledger opens at nought and is never re-based. There is no comparison of the contract's issue time here, so a squad handed a different region does not start again: the period that re-tasks it pays the new region's figure less the old region's figure, which hands back everything banked on the ground it is leaving. Re-basing instead would let a squad keep what it gained on one region and open clean on another, so that a rise could be banked and a fall ducked by re-tasking — a change of objective wearing the clothes of a change of density, and one a policy can help itself to at will. What makes this the right ledger and not merely a different one is that the payments then telescope: every period pays a difference of the same quantity, so the whole episode sums to the last reading of it, which is exactly the terminal the contest pays at its horizon. The objective is unchanged and only its density changes.

        A period in which the squad holds no contract is paid rather than forgotten, which is why this branch sits above the guard that drops a contract-less squad's mission. A squad between contracts has simply not moved any ground, its figure is nought, and the honest payment is nought less whatever the ledger held — the same hand-back a re-tasking makes. Dropping the mission there would restart the ledger at nought with that hand-back unpaid, and the episode would then pay everything banked before the gap a second time.

        The ledger's opening stays nought, and the horizon depends on it. `close` returns the last figure less the opening, and the contest's `finish` pays its terminal less that, so the last payment is the last two figures' difference only because the opening is nought. A scored mission given a non-nought opening would leave every period payment right and the horizon wrong.
        """
        mission = self.missions.get(squad.id)
        if mission is None:
            contract = squad.contract
            mission = _Mission(issued_at_ms=contract.issued_at_ms if contract is not None else 0,
                               potential=0.0, opening=0.0)
            self.missions[squad.id] = mission
        if mission.ended:
            # Nothing in this class ever ends a mission, so this is inert as things stand. It is here because `finish` asks the same question before paying a terminal, and if anything ever did end a scored errand mid-episode the payments would have to stop with it: a ledger that went on moving after its terminal had been paid would no longer sum to that terminal.
            return Outcome()
        reward = self.discount * figure - mission.potential
        mission.potential = figure
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
        # The domination and not the bare share, and the reason is the discount rather than any agreement with a terminal. A potential carrying a constant term does not telescope that term away wherever the discount is below one: the shaping is `discount * potential after - potential before`, so a constant `c` inside it is paid as `c * (discount - 1)` every single period, for as long as the errand stands. With the potential at `priority * share` on nought to one the constant is half the region's priority, and in a match, at ninety-nine hundredths, a squad was charged a two-hundredth of the priority of the region it was sent to every period it stayed on the errand — a standing charge for having been pointed at ground the strategic layer wanted, with no board in it at all and nothing it could do about it. Written as the domination the constant is nought and the shaping is the movement of the ground alone, which is the only part of it a decision can answer for.
        #
        # A whole constructed errand, discounted at one, is the case where that argument does not bite and neither reading would differ: at a discount of one the constant cancels in every difference. It is written for the match, where it does.
        priority = orders.priorities.get(region.id, 0.0) if orders is not None else 0.0
        return priority * (_share(region.our_value, region.enemy_value) - 0.5)
