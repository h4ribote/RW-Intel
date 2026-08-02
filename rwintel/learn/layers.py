"""The three learnt layers, each a script layer with one decision taken out and replaced.

Everything except the decision is inherited. The tactical layer's mission reports, its bookkeeping of what a squad has destroyed, the rule that a departure other than holding is re-issued every period; the operational layer's pricing of a mission against the strategic allowance, its deadlines, the rule that a contract is only re-issued when it differs from the one held — all of that is the same code running. What is overridden is one method each: which of the departures, and which region under which task.

That is the design's own boundary and not a convenience. Layers are learnt one at a time against frozen neighbours, and a learnt layer must be substitutable for the script layer in the sense that the rest of the system cannot tell which is present. Inheriting rather than reimplementing is how that is guaranteed rather than hoped for: there is no second copy of the rules to drift.

Reward arrives one period late by construction. A decision cannot be paid until the board has moved under it, so each period pays the previous decision from the board that has now arrived, and only then makes a new one. This is the same one-period structure the interface already has for the action itself, and keeping the two aligned means a step in the buffer covers exactly one period of game time.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..wire import Deviation, RegionState, Status, Task
from ..control.policy.contracts import DOCTRINES, MissionReport, OperationsOrders, SquadRecord
from ..control.policy.operations import Operations
from ..control.policy.strategy import Strategy
from ..control.policy.tactics import Tactics
from ..control.policy.view import WorldView
from .deciders import Choice
from .encoding import (POSTURES, operational_slots, operational_state, region_mask, strategic_state,
                       tactical_state, task_mask)
from ..eval.scoring import OPENING_WEIGHTS
from .reward import (DISCOUNT as REWARD_DISCOUNT, WIPED_REWARD, OperationalReward, StrategicReward,
                     TacticalReward)
from .rollout import Rollout, Step

log = logging.getLogger(__name__)


class LearntTactics(Tactics):
    """The tactical layer with the choice of departure taken from a decider instead of from the rule ladder."""

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, status_terminals: bool = True,
                 discount: float = REWARD_DISCOUNT, withhold=()) -> None:
        # Withholding reaches the inherited ladder and not the decider, which is the point of it: an arm built with no decider IS the ladder, so this is how the ladder is measured against itself with one of its branches taken away.
        super().__init__(session, catalogue, withhold)
        self.decider = decider
        self.rollout = rollout
        self.instance = instance
        # Whether the conditions written into a contract are allowed to end an errand, which they should wherever something reissues contracts and should not where one contract stands for a whole constructed fight that nothing will reissue.
        #
        # The discount is handed in with it and for the same reason: the two cases differ in how long an errand is, and the shaping has to telescope at whatever the returns are discounted at. Whoever builds the layer knows which case this is, so whoever builds it owns both.
        self.reward = TacticalReward(status_terminals=status_terminals, discount=discount)
        #: The decision each squad is owed payment for, held until the next period says what it earned.
        self.pending: Dict[int, Step] = {}
        #: Shaping earned in a period where this squad had no decision waiting to be paid, carried to the next one that has. A layer does not take a decision about every squad every period — the inherited rule passes over a squad whose tactical command a human holds, and one none of whose members are in sight, while leaving its contract standing — and the reward advances that squad's potential regardless. Dropped, those periods punch holes in a sum that only means anything because it telescopes: what an errand returns is its terminal less the potential it opened at, and only if every increment in between was paid to something. Carried, the telescope closes again.
        self.owed: Dict[int, float] = {}
        #: Every squad this layer has settled a period for and not yet seen leave. What it is for is `_gone`: the buffer, the carried payments and the reward's own ledger are all keyed by squad number, the numbers are reused, and something has to know which of them are still the board's.
        self.tracked: set = set()
        #: How many errands were closed for each reason, so that a run can be asked whether its terminals are firing at all rather than having it guessed at from the shape of the returns. An errand that never terminates is paid nothing but shaping, and shaping sums to nothing, so a policy learning from trajectories that never close is learning from noise.
        self.terminals: Counter = Counter()
        self._view: Optional[WorldView] = None
        self._now = 0

    def decide(self, view: WorldView, squads: List[SquadRecord], game_time_ms: int):
        # Held on the instance because the method this class exists to override is handed only what the rule needs, and what a network needs is the board. Overriding the caller instead would mean copying the reporting half of the layer, which is the half that has to stay identical.
        self._view = view
        self._now = game_time_ms
        self._settle(view, squads, game_time_ms)
        return super().decide(view, squads, game_time_ms)

    def _settle(self, view: WorldView, squads: Sequence[SquadRecord], game_time_ms: int) -> None:
        """Pays every outstanding decision from the board that has just arrived, closes the errands that have ended, and ends the trajectory of any errand a new contract has just replaced."""
        if self.rollout is None:
            return
        present = {squad.id for squad in squads}
        self.tracked |= present
        for squad in squads:
            # What this squad has destroyed since its contract was issued, which the inherited layer already counts in order to judge the exchange for itself. It is a period behind, because the count for this period is made further down while the departure is being chosen, and a shaping term is a difference of two potentials so a lag applied to both ends of it cancels.
            track = self.tracks.get(squad.id)
            # Nothing destroyed yet where the contract in hand is not the one the tally was kept against. The inherited layer starts a fresh tally whenever a contract is issued, but it does that further down, while this runs first; without the test the opening potential of a new errand is taken with the last errand's kills still on it. In the arena, where the squad numbers are reused fight after fight, that is every fight: measured, an opening potential of 0.575 read as 0.78 with three thousand credits of a previous fight's kills still counted, and the whole of the first decision's payment is the difference.
            killed = 0.0
            if track is not None and squad.contract is not None and track.issued_at_ms == squad.contract.issued_at_ms:
                killed = track.killed
            outcome = self.reward.step(squad, view, game_time_ms, killed=killed)
            step = self.pending.pop(squad.id, None)
            if step is not None:
                step.reward = outcome.reward + self.owed.pop(squad.id, 0.0)
                step.done = outcome.done
                if outcome.done:
                    self.terminals[outcome.reason] += 1
                self.rollout.add((self.instance, squad.id), step)
            elif outcome.done:
                # An errand that ended in a period this layer took no decision in. The inherited rule passes over a squad whose tactical command a human holds and one none of whose members are in sight, leaving its contract standing, and the reward goes on reading the board for it either way — so the period that finds the contract complete, the deadline past or the squad losing can be a period with nothing of this layer's outstanding. Carried into `owed`, which is what every other unpaid period does, the terminal would be handed to whatever decision came next, and that decision belongs to no errand: this one is over and the next contract has not arrived. So the payment reaches back to the last decision there was and closes the trajectory on it, exactly as `finish` does for a squad that left the board.
                payment = outcome.reward + self.owed.pop(squad.id, 0.0)
                if self.rollout.close_with((self.instance, squad.id), payment):
                    self.terminals[outcome.reason] += 1
            else:
                self.owed[squad.id] = self.owed.get(squad.id, 0.0) + outcome.reward
            if outcome.renewed:
                # Whatever was carried belonged to the errand being cut, whose potentials are measured against different ground; it cannot be paid into the errand that replaces it.
                self.owed.pop(squad.id, None)
                # A contract is the unit of work and so the unit of pay, so a squad handed a different one has begun a different errand and the decisions of the two must not share a trajectory: advantage estimation would otherwise run what the new errand earned backwards into decisions taken for the old. Cut rather than closed, because the errand that was replaced did not fail — it stopped being observed, and its last decision is bootstrapped from its own value estimate as any other unobserved ending is. What that decision is paid is nothing, since the potentials of two contracts are measured against different ground and different allowances and a difference between them is not a shaping term.
                self.rollout.cut((self.instance, squad.id), reason="renewed")
        for squad_id in self._gone(present):
            # A squad that has left the board between periods cannot be paid from anything, so its last decision is cut off rather than scored, and anything carried for it goes with the errand.
            self.tracked.discard(squad_id)
            self.pending.pop(squad_id, None)
            self.owed.pop(squad_id, None)
            self.rollout.cut((self.instance, squad_id), reason="left")
            self.reward.forget(squad_id)

    def _gone(self, present: Sequence[int]) -> List[int]:
        """Every squad this layer has been settling that the board no longer reports.

        Read off every squad seen rather than off the decisions awaiting payment alone, which is what it was. A squad leaves the board in a period this layer took no decision about it as often as not — it is destroyed while out of contact, or the organisation layer folds it into another — and its trajectory was then left open with its shaping ledger and its carried payment standing. Squad numbers are handed round a small pool and the arena reuses them fight after fight, so the next squad to be given that number went on filing decisions into the trajectory of the one that died, and opened its errand against a potential measured on the dead squad's ground.
        """
        held = self.tracked | set(self.pending) | set(self.owed)
        return [squad_id for squad_id in sorted(held) if squad_id not in present]

    def finish(self, squad: SquadRecord, terminal: float, reason: str) -> None:
        """Ends this squad's errand from outside, paying the decision still waiting on it as the last of the trajectory.

        Whoever is running the fight knows when it is over; the layer only sees periods. Without this the decision taken in the final period of a fight is left outstanding and is paid, several seconds of game time later, out of the first period of whatever fight the squad number is next used for — which strings the two together into one trajectory and lets the advantage of the second flow backwards into the decisions of the first. Squad numbers are reused promptly wherever the same few slots are handed round, so that is the ordinary case rather than a corner of it.

        The payment is the outcome being handed in plus the last shaping term, taken against a terminal potential of nought so that the shaping over the errand telescopes away to nothing and cannot change which policy is the best one.

        An errand that ended on its own conditions but has since gone on collecting decisions is the opposite case: the decisions after it belong to no errand, so the one still waiting is cut rather than paid, and cutting bootstraps it from its own value estimate as any decision that merely stopped being observed is.

        A squad that no longer exists is the third case and the reason this cannot simply give up when nothing is outstanding. The period that finds it destroyed has no squad left to decide anything, so no decision is taken and none is left waiting; the last one there was has already gone into the trajectory as an ordinary step. The ending is therefore added to that step where it lies. Without this the trajectory would be cut instead, which asserts that the errand went on unobserved rather than that the squad died doing it, and being destroyed would then cost the layer nothing at all.
        """
        self._end(squad.id, terminal, reason)

    def wiped(self, squads: Sequence[int]) -> None:
        """Pays the terminal for squads the game destroyed outright, which the chain names because nothing else can.

        A squad is retired by the organisation layer in the same period its last unit dies, so this layer is never handed a board with an empty squad standing on it, and the condition its own reward tests for — a contract held by a squad with nobody in it — cannot fire in a match at all. Without this, losing a whole squad on an errand read as a squad that stopped being reported: its trajectory was cut and bootstrapped from its own value estimate, so being destroyed cost the layer nothing. The design names being destroyed as one of the four endings this layer is paid for, and this is where a match makes that true.

        Only destruction reaches here. A squad folded into a neighbour or broken up for spares is retired too and is not this: its units are alive and under somebody else's command, so its last decision is cut as any unobserved ending is, which is what `_settle` already does for it.
        """
        for squad_id in squads:
            self._end(squad_id, WIPED_REWARD, "wiped")
            self.tracked.discard(squad_id)

    def _end(self, squad_id: int, terminal: float, reason: str) -> None:
        """The three cases of an errand ended from outside, taken by squad number so that a caller holding a record and a caller holding only a number end an errand by the same code."""
        if self.reward.ended(squad_id):
            self.pending.pop(squad_id, None)
            self.owed.pop(squad_id, None)
            if self.rollout is not None:
                self.rollout.cut((self.instance, squad_id), reason="spent")
            self.reward.forget(squad_id)
            return
        payment = terminal + (0.0 - self.reward.close(squad_id)) + self.owed.pop(squad_id, 0.0)
        step = self.pending.pop(squad_id, None)
        if step is None:
            if self.rollout is not None and self.rollout.close_with((self.instance, squad_id), payment):
                self.terminals[reason] += 1
            return
        step.reward = payment
        step.done = True
        self.terminals[reason] += 1
        if self.rollout is not None:
            self.rollout.add((self.instance, squad_id), step)

    def _departures(self, fights) -> List[Deviation]:
        """Every squad's departure in one ask of the decider, and one step recorded for each, in the order the squads were read.

        One ask a side and not one a squad. A network answers through a batching server whose window is sized against the wall period, so a layer that asked squad by squad would spend a window on each, and the fixed one-period decision lag the interface promises would become a lag that depends on how many squads a side happens to have. The states are all built before anything is asked, which is what lets them go over in one request; that costs nothing, because building a state reads the board and writes nothing, so a state built early is the state that would have been built late.
        """
        view = self._view
        if view is None:
            return super()._departures(fights)
        if not fights:
            # Nothing to order this period, so nothing to ask. Load-bearing and not merely tidy: the batching server answers an empty group without touching its queue, but the path with no server behind it puts the states through the network directly, and a forward pass on no rows is a shape error rather than an empty answer. A side whose squads are all human-held or all off the board is an ordinary period, not a fault, so it has to return here.
            return []
        # Every departure is always available. Withdrawing from a fight that is going well is a bad idea and not an illegal one, and a mask that encoded which were sensible would be the rule ladder again, hidden.
        requests = [(tactical_state(fight.squad, fight.members, fight.threats, fight.losses,
                                    fight.track.killed, view, self._now),
                     [1.0] * len(Deviation))
                    for fight in fights]
        if self.decider is None:
            # No decider means the inherited rule decides and this class is only writing down what it chose, which is how the script is turned into a teacher: what comes out is a state and an action of exactly the form a learnt layer emits. Written as a loop rather than as a comprehension because a comprehension is a function of its own with no `self` argument, and the zero-argument `super()` below would raise inside one.
            choices = []
            for fight in fights:
                choices.append(Choice(action=int(super()._departure(
                    fight.squad, fight.members, fight.threats, fight.losses, fight.track))))
        else:
            choices = self.decider.choose_many(requests)
        if self.rollout is not None:
            for fight, (state, mask), choice in zip(fights, requests, choices):
                self.pending[fight.squad.id] = Step(
                    state=state, action=choice.action, mask=mask, log_prob=choice.log_prob,
                    value=choice.value, squad=fight.squad.id, at_ms=self._now)
        return [Deviation(choice.action) for choice in choices]

    def park(self) -> None:
        """Files every decision still waiting for payment into its trajectory, and ends nothing.

        Split out of the flush below because one arena drives both sides out of one instance number and one buffer. Cutting is by owner, and both sides are the same owner, so the first side to flush cut the other side's live trajectories — trajectories whose last decisions were still sitting in that side's pending ledger and had not been filed. Those decisions then landed in a fresh trajectory of their own and the errand came apart in two, one half bootstrapped from a step that was not its last. Whoever closes a board with two layers on it parks both before either flushes.
        """
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()

    def flush(self) -> None:
        """Ends every open errand at the end of an episode, without yet releasing the episode's trajectories to be drained. They did not fail; they stopped being observed, so they are bootstrapped rather than treated as terminal.

        Separated from the seal that follows it because the operational chain has to mark the episode's interference in between: the decisions still owed payment are added here, then the intruder's touched squads are tainted, and only then are the trajectories sealed. Sealing here instead would let a decision the intruder touched be sealed and drained before it was tainted.
        """
        if self.rollout is None:
            return
        self.park()
        # Every errand open here is cut rather than ended, so whatever was carried for it is carried no further.
        self.owed.clear()
        self.tracked.clear()
        # This instance's errands only. One buffer serves every instance of a run, and an episode ending here says nothing about the fight another instance is in the middle of.
        self.rollout.cut_all(owner=self.instance, reason="episode")

    def close(self) -> None:
        """Ends every open errand and releases this episode's trajectories to the trainer. The tactical arena has no intruder, so there is nothing to taint between the flush and the seal; the two are one call here and split only where an intruder sits above the layer."""
        self.flush()
        if self.rollout is not None:
            self.rollout.seal(self.instance)


class LearntStrategy(Strategy):
    """The strategic layer with the choice of posture taken from a decider instead of from the transition rule.

    One decision every ten seconds for the whole side, and everything else this layer emits — the allocation, the technology cap, the target mix, the region priorities, the loss allowance, whether the posture presses — is that posture read through the inherited tables. Replacing the rule and inheriting the tables is what makes this substitutable for the script in the sense the comparison needs: a learnt posture that beats the script's is a better posture and not a different layer.

    There is one errand and it is the match. Nothing keys a mission by squad here, nothing is re-issued, and the terminal arrives from outside through `conclude` when the match is over — which is the same shape the arenas' `finish` has, and for the same reason: whoever runs the match knows how it ended, and the layer only ever sees periods.

    A human pinning the posture stops this layer deciding at all, exactly as it stops the script's rule, because the inherited `decide` reads `forced` first. No decision is then recorded and none is paid, which is right: the design says the results of what somebody else commanded are to be kept out of the learning signal, and a pinned posture is precisely that.
    """

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, discount: float = REWARD_DISCOUNT) -> None:
        super().__init__(session, catalogue)
        self.decider = decider
        self.rollout = rollout
        self.instance = instance
        # The potential is the running form of the very quantity the match is scored on, so the weight it is taken at is the score's own rather than a one written here. Handed in from the scoring module so that the two cannot silently disagree: the moment the weights are fitted away from military-only, this reads the same figure the score does, and where the score has weight this layer cannot observe in flight — the enemy's income is not visible from inside a match — the disagreement is the stated limit rather than a hidden one.
        self.reward = StrategicReward(discount=discount, military_weight=OPENING_WEIGHTS.military)
        #: The one decision awaiting payment, held until the next period says what the board did under it.
        self.pending: Optional[Step] = None
        #: Shaping earned in a period where no decision was waiting to be paid, carried to the next one that has, for the reason the other two layers' ledgers of the same name give. It fires here whenever a human has the posture pinned: the board goes on moving and this layer is not the one moving it, so the term is carried rather than dropped and the sum still telescopes.
        self.owed = 0.0
        self.terminals: Counter = Counter()
        self._regions: List = []
        self._contact: Optional[Dict] = None
        self._now = 0

    #: What a trajectory of this layer's decisions is keyed by within its instance. Not a squad — this layer commands no squad — and negative so that it can never collide with one, since the buffer is shared with whichever other layer a run happens to be collecting from.
    KEY = -1

    def decide(self, report, regions, game_time_ms: int, contact=None):
        # Held on the instance because the method this class exists to override is handed only the front report, and what a network needs is the board the report was taken from. Overriding the caller instead would mean copying the half of the layer that turns a posture into orders, which is the half that has to stay identical.
        self._regions = list(regions)
        self._contact = contact
        self._now = game_time_ms
        self._settle(report)
        return super().decide(report, regions, game_time_ms, contact)

    def _settle(self, report) -> None:
        """Pays the decision still waiting from the board that has just arrived.

        Before the inherited `decide` samples the histories, so that the payment is made from the board this period brought rather than from the board plus this period's own sample. The two would give the same potential — the potential reads the report and not the histories — and the order is the one the other two layers keep: settle what is outstanding, then decide.
        """
        if self.rollout is None:
            return
        outcome = self.reward.step(report)
        step, self.pending = self.pending, None
        if step is not None:
            step.reward = outcome.reward + self.owed
            self.owed = 0.0
            step.done = outcome.done
            self.rollout.add((self.instance, self.KEY), step)
        else:
            self.owed += outcome.reward

    def _transition(self, report):
        """Which posture, from the decider rather than from the rule.

        With no decider this falls through to the inherited rule and writes down what it chose, which is how the script becomes a teacher for this layer: what comes out is a state and an action of exactly the form a learnt layer emits.

        Every posture is legal and there is no mask. The rule never selects TECH and that is a property of the rule — the design says so outright and gives a human the way to select it — so a mask forbidding it would be the rule written again in the mask's clothing, which is what the tactical space refuses masks for.
        """
        state = strategic_state(report, self._regions, self._now, self.income_history,
                                self.loss_history, self.most_enemy_bases, self._contact, self.posture)
        mask = [1.0] * len(POSTURES)
        if self.decider is None:
            choice = Choice(action=int(super()._transition(report)))
        else:
            choice = self.decider.choose_many([(state, mask)])[0]
        if not 0 <= choice.action < len(POSTURES):
            return super()._transition(report)
        if self.rollout is not None:
            self.pending = Step(state=state, action=choice.action, mask=mask,
                                log_prob=choice.log_prob, value=choice.value,
                                squad=self.KEY, at_ms=self._now)
        return POSTURES[choice.action]

    def conclude(self, terminal: float, reason: str = "match") -> None:
        """Ends the match's errand from outside, paying the decision still waiting on it as the last of the trajectory.

        The match's own result, which is the only terminal this design ever pays a strategic decision and the only place a layer is paid the match at all. The payment is the result handed in plus the last shaping term taken against a terminal potential of nought, so the shaping over the match telescopes away and cannot change which policy is best. The match therefore returns the result less the potential the match opened at, at whatever discount the run is taking its returns at — see `StrategicReward.close` for why that is the identity to want rather than the tidier one it replaced.

        The three cases are the ones the other two layers' finishers handle. An errand already paid its terminal has its still-waiting decision cut rather than paid twice. A match that ended with no decision waiting — the posture was pinned through its last period, or the match was called between periods — has the payment added to the last step there was. And a match that took no strategic decision at all has nothing to pay and nothing to cut.
        """
        if self.rollout is None:
            return
        if self.reward.ended():
            self.pending = None
            self.owed = 0.0
            self.rollout.cut((self.instance, self.KEY), reason="spent")
            self.reward.forget()
            return
        payment = terminal + (0.0 - self.reward.close()) + self.owed
        self.owed = 0.0
        step, self.pending = self.pending, None
        if step is None:
            if self.rollout.close_with((self.instance, self.KEY), payment):
                self.terminals[reason] += 1
            return
        step.reward = payment
        step.done = True
        self.terminals[reason] += 1
        self.rollout.add((self.instance, self.KEY), step)

    def park(self) -> None:
        """Files the decision still waiting for payment, and ends nothing. One layer a side here, so no board parks two of these; it is defined for the same reason the other two layers define it, so that whoever closes a board can park everything before anything cuts."""
        if self.rollout is None:
            return
        if self.pending is not None:
            self.rollout.add((self.instance, self.KEY), self.pending)
            self.pending = None

    def flush(self) -> None:
        """Ends the match's errand at the end of an episode without yet releasing it, for the reason the other two layers' flush is split from their close: the chain marks the episode's interference in between.

        A match that reached `conclude` has nothing left here. What this is for is the episode that ended some other way — a run stopped by hand, an instance that dropped — where the decision still waiting belongs to no result and is bootstrapped rather than paid.
        """
        if self.rollout is None:
            return
        self.park()
        self.owed = 0.0
        self.rollout.cut_all(owner=self.instance, reason="episode")
        self.reward.reset()

    def close(self) -> None:
        self.flush()
        if self.rollout is not None:
            self.rollout.seal(self.instance)


class LearntOperations(Operations):
    """The operational layer with the choice of where and what taken from a decider."""

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, discount: float = REWARD_DISCOUNT) -> None:
        super().__init__(session, catalogue)
        self.decider = decider
        self.rollout = rollout
        self.instance = instance
        # The discount is handed in for the same reason the tactical layer's is: a match discounts an operational errand as a fragment of itself, while the constructed operations arena is one errand from end to end and discounts it at nothing. Whoever builds the layer knows which case this is.
        self.reward = OperationalReward(discount=discount)
        self.pending: Dict[int, Step] = {}
        #: Shaping earned in a period where this squad had no decision waiting to be paid, carried to the next one that has, for the reason the tactical layer's ledger of the same name gives. It bites harder here: the inherited operational rule leaves a squad worn below the health it will task at all out of the decision entirely while its contract stands, and the arena wears squads down by construction.
        self.owed: Dict[int, float] = {}
        #: Every squad this layer has settled a period for and not yet seen leave, which is what `_gone` reads. The tactical layer's ledger of the same name gives the reason.
        self.tracked: set = set()
        #: How many errands were closed for each reason, so a run can be asked whether its terminals are firing. An operational errand takes its terminal only from outside, through finish, so this stays empty in a match and fills in the arena.
        self.terminals: Counter = Counter()
        #: What the board a contest is scored on last read for each of this side's squads, handed in from outside before the period's decisions are settled against it. None in a match, where there is no such board and the region block is the only signal there is. The switch is per board and not per squad on purpose: a trajectory whose steps were paid in two different quantities sums to neither.
        self._standing: Optional[Dict[int, float]] = None
        self._state: List[float] = []
        #: What the period's state was built from, kept so that the one feature which depends on the squad being asked can be written without reading the board a second time.
        self._view_of = None
        self._regions: List[float] = []
        #: The regions in the order the state was written in, kept from the period they were written so that a chosen slot is decoded against the very list it was offered over. Recomputing it at the decision would read the same view twice and give the same answer, but nothing would say so.
        self._slots: List = []
        #: The first squad number this side was ever handed, which is what the squad rows and the one-hot are offset by. Taken once and never raised: one process drives both sides of the constructed arena out of one numbering, so this side's squads are some run of numbers that does not start at nought, and a squad dying must not renumber the ones above it.
        self._base: Optional[int] = None
        self._view: Optional[WorldView] = None
        self._spawns = tuple(region.id for region in getattr(session, "regions", ()) if region.spawn)

    def standing(self, figures: Dict[int, float]) -> None:
        """Takes what the scored board reads for each of this side's squads, for the period about to be settled.

        Reached from outside exactly as `finish` is, and for the same reason: what a squad's errand is worth is a statement about ground that only whoever runs the contest can read, and this layer sees periods and nothing else. Everything of one period at once, by squad, because the discs are read once for the whole board — several squads may be sent to one of them, and both sides are paid off the one reading so that their figures stay exact negatives.

        A match never calls this, so the figures stay unset there and the region block goes on being the whole of the signal.

        Nothing of this figure reaches the observation the decider is handed: the squad row carries no contracted region and no scored standing, so a critic cannot subtract the part of a return that is already fixed when the action is taken. That is variance and not bias — the figure is read before the decision, so what it adds to a return is an action-independent offset and the estimator stays unbiased for the same objective — but it is the one thing that blunts this credit, and putting the contracted region or the figure itself into the squad row is what would remove it.
        """
        self._standing = dict(figures)

    def decide(self, view: WorldView, orders: OperationsOrders, squads: List[SquadRecord],
               reports: List[MissionReport], game_time_ms: int):
        self._view = view
        self._settle(view, orders, squads)
        if squads:
            first = min(squad.id for squad in squads)
            self._base = first if self._base is None else min(self._base, first)
        # Everything but the one feature that is about the squad being asked. The rest of the board is read once for the period, since it is the same board for every squad and reading it again per squad would be the same answer at four times the cost.
        self._view_of = (view, orders, squads, game_time_ms)
        self._state = operational_state(view, orders, squads, game_time_ms, self._spawns, self.base)
        self._regions = region_mask(view)
        self._slots = operational_slots(view)
        return super().decide(view, orders, squads, reports, game_time_ms)

    @property
    def base(self) -> int:
        """The offset this side's squad rows are written at, nought until a squad has been seen."""
        return 0 if self._base is None else self._base

    def _settle(self, view: WorldView, orders: OperationsOrders, squads: Sequence[SquadRecord]) -> None:
        """Pays each squad's previous decision from the board that has now arrived, one squad at a time.

        Each decision is paid the shaping of the region its own contract named, not one board-wide figure shared out to all of them. The figure that paid all of them was the disease: what the strategic layer asks for is region by region, and a squad sent to a region it took has to be paid differently from one sent to a region it lost, or the advantage does not depend on the choice and the gradient is dead.

        Where a contest reads its own ground and has handed its reading in, that is what a period is paid the movement of instead of the region block, and the same figure the contest pays at its horizon is then the last of the same series. Either way a trajectory has one currency from the first decision to the last, so there is no errand boundary left in it: a squad handed a new contract has its old ground handed back and its new ground taken on in one payment, and its trajectory runs on. The match used to cut here on the ground that two contracts price against different regions, which is true and is not a reason: both prices come out of the one region table, so the difference across the boundary is a difference of one quantity and declining to pay it let a squad keep a rise and walk away from the fall.

        A squad gone from the board — folded into another by the organisation layer, disbanded, or wiped — is cut rather than ended: its last decision is bootstrapped from its own value estimate, as any decision that merely stopped being observed is, not closed against a terminal potential of nought. Marking it done would teach the critic that every state a squad turns over from, a routine merge of a healthy squad included, is worth nothing from here, corrupting the baseline every other squad's advantage is taken against. What ends an operational errand as a terminal comes only from outside, through finish, which the constructed arena calls at the horizon.
        """
        if self.rollout is None:
            return
        present = {squad.id for squad in squads}
        self.tracked |= present
        for squad in squads:
            # Nothing where no contest reads this board, which is a match; the contest's reading where one does. Read per board rather than per squad, and the reading a contest hands in covers every squad it has, so the default here is a guard and not a path — a squad paid out of the region block for a period while its neighbours were paid out of the discs would leave a trajectory summing to neither quantity.
            figure = None if self._standing is None else self._standing.get(squad.id, 0.0)
            outcome = self.reward.step(squad, view, orders, figure=figure)
            step = self.pending.pop(squad.id, None)
            if step is not None:
                step.reward = outcome.reward + self.owed.pop(squad.id, 0.0)
                step.done = outcome.done
                self.rollout.add((self.instance, squad.id), step)
            else:
                self.owed[squad.id] = self.owed.get(squad.id, 0.0) + outcome.reward
        for squad_id in self._gone(present):
            self.tracked.discard(squad_id)
            self.pending.pop(squad_id, None)
            self.owed.pop(squad_id, None)
            self.rollout.cut((self.instance, squad_id), reason="left")
            self.reward.forget(squad_id)

    def _gone(self, present: Sequence[int]) -> List[int]:
        """Every squad this layer is still holding something for that the board no longer reports, read off everything held rather than off the decisions awaiting payment alone. The tactical layer's ledger of the same name gives the reason, and it bites harder here: this layer decides about a squad only when the inherited rule finds it taskable, so a squad worn below that line and then folded into another leaves nothing pending and everything else standing."""
        held = self.tracked | set(self.pending) | set(self.owed)
        return [squad_id for squad_id in sorted(held) if squad_id not in present]

    def finish(self, squad: SquadRecord, terminal: float, reason: str) -> None:
        """Ends this squad's errand from outside, paying the decision still waiting on it as the last of its trajectory. Structurally the tactical layer's finish: whoever runs the contest knows when it is over, the layer only sees periods.

        The payment is the terminal handed in plus the last shaping term taken against a terminal potential of nought, so the shaping over the errand telescopes away and cannot change which policy is best. A squad whose errand already took its terminal has its still-waiting decision cut instead of paid, and a squad gone between periods has no decision waiting, so the terminal is added to the last step there was through the buffer's close-with — exactly the three cases the tactical finish handles.

        The same arithmetic makes this the last of a series where a contest has been paying every period. `close` hands back everything already paid, so what is added here is the terminal less that — the movement of the scored figure over the final period — and the episode still returns the terminal exactly. Nothing in this method knows which of the two it is doing, and nothing needs to.
        """
        if self.reward.ended(squad.id):
            self.pending.pop(squad.id, None)
            self.owed.pop(squad.id, None)
            if self.rollout is not None:
                self.rollout.cut((self.instance, squad.id), reason="spent")
            self.reward.forget(squad.id)
            return
        payment = terminal + (0.0 - self.reward.close(squad.id)) + self.owed.pop(squad.id, 0.0)
        step = self.pending.pop(squad.id, None)
        if step is None:
            if self.rollout is not None and self.rollout.close_with((self.instance, squad.id), payment):
                self.terminals[reason] += 1
            return
        step.reward = payment
        step.done = True
        self.terminals[reason] += 1
        if self.rollout is not None:
            self.rollout.add((self.instance, squad.id), step)

    def _settled(self, view: WorldView, orders, squad: SquadRecord, chosen, avoid):
        """Whatever was chosen, unchanged.

        The script layer keeps a squad on the errand it is already running unless a mission report gives a reason to change, because two regions of nearly equal score trade places whenever a shot lands in either and a squad re-tasked on that difference walks between them for ever. A learnt layer is being asked to make exactly that judgement itself, so overriding its answer with the rule would be measuring the rule. Nothing is lost by removing it: a contract that names the same region and task as the one already held is not re-issued, so a policy that decides to stay costs nothing.

        With no decider this class is recording what the script chose rather than replacing it, so the script's own rule has to stand or what is written down is not what the script does.

        And what is written down has to be the answer this returns, not the one it was handed. The step is recorded where the choice is made, one call earlier, and that call is the SCORING of the board — which the rule then overrules whenever the squad is running a mission that has not stalled, failed, finished or expired. That overruling is not a detail of the rule: it is what makes the handwritten ladder's errand last a hundred and twenty-seven decisions instead of one, and it fires on the great majority of its decisions. A teacher written from the scoring alone is a teacher for a policy that re-scores the board every period and never holds an errand — which is not the ladder, and is a fair description of what fitting to it produced. So the label is corrected here to the contract that actually goes out.
        """
        if self.decider is not None:
            return chosen
        settled = super()._settled(view, orders, squad, chosen, avoid)
        self._relabel(squad, settled)
        return settled

    def _relabel(self, squad: SquadRecord, settled) -> None:
        """Rewrites the decision waiting on this squad to name what the rule settled on, where that differs from what the scoring picked.

        A region past the last row of the block cannot be written down at all — a label outside the head would be fitted as if it named some other place — so the decision is dropped rather than recorded wrongly, which is the same choice `_pick` makes when the scoring itself lands outside the block.
        """
        step = self.pending.get(squad.id)
        if step is None:
            return
        if settled is None:
            self.pending.pop(squad.id, None)
            return
        task, region = settled
        slot = next((index for index, row in enumerate(self._slots) if row.id == region.id), -1)
        if slot < 0:
            self.pending.pop(squad.id, None)
            return
        step.action = slot
        step.second = int(task)

    def _pick(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
              avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        if self._view is None or not self._state:
            return super()._pick(view, orders, squad, avoid)
        tasks = task_mask(squad.doctrine)
        if not any(tasks) or not any(self._regions):
            return None
        # The board as this squad sees it, which differs from the period's shared reading in one feature: which region it is already on its way to. Built per squad because that is what it is — a fact about the decision and not about the board — and cheap enough at four squads a period to be built rather than patched.
        state = self._state
        contract = squad.contract
        if contract is not None:
            view_of, orders_of, squads_of, now_of = self._view_of
            state = operational_state(view_of, orders_of, squads_of, now_of, self._spawns, self.base,
                                      contracted=contract.target_region)
        if self.decider is None:
            chosen = super()._pick(view, orders, squad, avoid)
            if chosen is None:
                return None
            # The script picks a region and what is written down has to be the SLOT it sits in, since that is what a policy fitted to this teacher will answer with. A region beyond the last slot — a map with more places on it than the block has rows — is still sent to, because this class is recording what the script does and must not change it, and simply not written down: a label outside the head would be fitted as if it named some other place.
            slot = next((index for index, region in enumerate(self._slots) if region.id == chosen[1].id), -1)
            if slot < 0:
                return chosen
            choice = Choice(action=slot, second=int(chosen[0]))
        else:
            choice = self.decider.choose(state, squad.id - self.base, self._regions, tasks)
        if choice is None:
            return None
        if not 0 <= choice.action < len(self._slots):
            return None
        region = self._slots[choice.action]
        if self.rollout is not None:
            self.pending[squad.id] = Step(
                state=state, action=choice.action, mask=list(self._regions),
                second=choice.second, second_mask=list(tasks),
                log_prob=choice.total_log_prob, value=choice.value,
                squad=squad.id, slot=squad.id - self.base,
                at_ms=view.observation.game_time_ms)
        return Task(choice.second), region

    def park(self) -> None:
        """Files every decision still waiting for payment into its trajectory, and ends nothing. The tactical layer's method of the same name gives the reason: one arena drives four layers out of one instance number and one buffer, and the first of them to cut would otherwise cut the others' errands out from under decisions they had not filed yet."""
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()

    def flush(self) -> None:
        """Ends every open period at the end of an episode, without yet releasing its trajectories to be drained. Split from the seal for the same reason the tactical layer's is: the operational chain marks the episode's interference between the two, and this is the layer an intruder actually interferes with."""
        if self.rollout is None:
            return
        self.park()
        # Every errand open here is cut rather than ended, so whatever was carried for it is carried no further.
        self.owed.clear()
        self.tracked.clear()
        self.rollout.cut_all(owner=self.instance, reason="episode")
        self.reward.reset()
        # An episode's boards belong to that episode: whatever the last contest read says nothing about the next one's ground, and a reading left standing would be paid against a ledger that has just been cleared.
        self._standing = None

    def close(self) -> None:
        """Ends every open period and releases this episode's trajectories to the trainer. The operational chain reaches the seal through its policy's close, which taints first; this direct close is the path with no intruder above it, where flush and seal are one call."""
        self.flush()
        if self.rollout is not None:
            self.rollout.seal(self.instance)
