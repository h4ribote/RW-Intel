"""The two learnt layers, each a script layer with one decision taken out and replaced.

Everything except the decision is inherited. The tactical layer's mission reports, its bookkeeping of what a squad has destroyed, the rule that a departure other than holding is re-issued every period; the operational layer's pricing of a mission against the strategic allowance, its deadlines, the rule that a contract is only re-issued when it differs from the one held — all of that is the same code running. What is overridden is one method each: which of the five departures, and which region under which task.

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
from ..control.policy.tactics import Tactics
from ..control.policy.view import Sighting, WorldView
from .deciders import Choice
from .encoding import operational_state, region_mask, tactical_state, task_mask
from .reward import DISCOUNT as REWARD_DISCOUNT, OperationalReward, TacticalReward
from .rollout import Rollout, Step

log = logging.getLogger(__name__)


class LearntTactics(Tactics):
    """The tactical layer with the choice of departure taken from a decider instead of from the rule ladder."""

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, status_terminals: bool = True,
                 discount: float = REWARD_DISCOUNT) -> None:
        super().__init__(session, catalogue)
        self.decider = decider
        self.rollout = rollout
        self.instance = instance
        # Whether the conditions written into a contract are allowed to end an errand, which they should wherever something reissues contracts and should not where one contract stands for a whole constructed fight that nothing will reissue.
        #
        # The discount is handed in with it and for the same reason: the two cases differ in how long an errand is, and the shaping has to telescope at whatever the returns are discounted at. Whoever builds the layer knows which case this is, so whoever builds it owns both.
        self.reward = TacticalReward(status_terminals=status_terminals, discount=discount)
        #: The decision each squad is owed payment for, held until the next period says what it earned.
        self.pending: Dict[int, Step] = {}
        #: Shaping earned in a period where this squad had no decision waiting to be paid, carried to the next one that has. A layer does not take a decision about every squad every period — the inherited rule skips a squad worn below the health it will task at all, and leaves its contract standing — while the reward advances that squad's potential regardless. Dropped, those periods punch holes in a sum that only means anything because it telescopes: what an errand returns is its terminal less the potential it opened at, and only if every increment in between was paid to something. Carried, the telescope closes again.
        self.owed: Dict[int, float] = {}
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
            else:
                self.owed[squad.id] = self.owed.get(squad.id, 0.0) + outcome.reward
            if outcome.renewed:
                # Whatever was carried belonged to the errand being cut, whose potentials are measured against different ground; it cannot be paid into the errand that replaces it.
                self.owed.pop(squad.id, None)
                # A contract is the unit of work and so the unit of pay, so a squad handed a different one has begun a different errand and the decisions of the two must not share a trajectory: advantage estimation would otherwise run what the new errand earned backwards into decisions taken for the old. Cut rather than closed, because the errand that was replaced did not fail — it stopped being observed, and its last decision is bootstrapped from its own value estimate as any other unobserved ending is. What that decision is paid is nothing, since the potentials of two contracts are measured against different ground and different allowances and a difference between them is not a shaping term.
                self.rollout.cut((self.instance, squad.id))
        for squad_id in [key for key in self.pending if key not in present]:
            # A squad that has left the board between periods cannot be paid from anything, so its last decision is cut off rather than scored, and anything carried for it goes with the errand.
            self.pending.pop(squad_id, None)
            self.owed.pop(squad_id, None)
            self.rollout.cut((self.instance, squad_id))
            self.reward.forget(squad_id)

    def finish(self, squad: SquadRecord, terminal: float, reason: str) -> None:
        """Ends this squad's errand from outside, paying the decision still waiting on it as the last of the trajectory.

        Whoever is running the fight knows when it is over; the layer only sees periods. Without this the decision taken in the final period of a fight is left outstanding and is paid, several seconds of game time later, out of the first period of whatever fight the squad number is next used for — which strings the two together into one trajectory and lets the advantage of the second flow backwards into the decisions of the first. Squad numbers are reused promptly wherever the same few slots are handed round, so that is the ordinary case rather than a corner of it.

        The payment is the outcome being handed in plus the last shaping term, taken against a terminal potential of nought so that the shaping over the errand telescopes away to nothing and cannot change which policy is the best one.

        An errand that ended on its own conditions but has since gone on collecting decisions is the opposite case: the decisions after it belong to no errand, so the one still waiting is cut rather than paid, and cutting bootstraps it from its own value estimate as any decision that merely stopped being observed is.

        A squad that no longer exists is the third case and the reason this cannot simply give up when nothing is outstanding. The period that finds it destroyed has no squad left to decide anything, so no decision is taken and none is left waiting; the last one there was has already gone into the trajectory as an ordinary step. The ending is therefore added to that step where it lies. Without this the trajectory would be cut instead, which asserts that the errand went on unobserved rather than that the squad died doing it, and being destroyed would then cost the layer nothing at all.
        """
        if self.reward.ended(squad.id):
            self.pending.pop(squad.id, None)
            self.owed.pop(squad.id, None)
            if self.rollout is not None:
                self.rollout.cut((self.instance, squad.id))
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

    def _departure(self, squad: SquadRecord, members: List[Sighting], threats: List[Sighting],
                   losses: float, track) -> Deviation:
        view = self._view
        if view is None:
            return super()._departure(squad, members, threats, losses, track)
        state = tactical_state(squad, members, threats, losses, track.killed, view, self._now)
        # Every departure is always available. Withdrawing from a fight that is going well is a bad idea and not an illegal one, and a mask that encoded which were sensible would be the rule ladder again, hidden.
        mask = [1.0] * len(Deviation)
        if self.decider is None:
            # No decider means the inherited rule decides and this class is only writing down what it chose, which is how the script is turned into a teacher: what comes out is a state and an action of exactly the form a learnt layer emits.
            choice = Choice(action=int(super()._departure(squad, members, threats, losses, track)))
        else:
            choice = self.decider.choose(state, mask)
        if self.rollout is not None:
            self.pending[squad.id] = Step(
                state=state, action=choice.action, mask=mask, log_prob=choice.log_prob,
                value=choice.value, squad=squad.id, at_ms=self._now)
        return Deviation(choice.action)

    def flush(self) -> None:
        """Ends every open errand at the end of an episode, without yet releasing the episode's trajectories to be drained. They did not fail; they stopped being observed, so they are bootstrapped rather than treated as terminal.

        Separated from the seal that follows it because the operational chain has to mark the episode's interference in between: the decisions still owed payment are added here, then the intruder's touched squads are tainted, and only then are the trajectories sealed. Sealing here instead would let a decision the intruder touched be sealed and drained before it was tainted.
        """
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()
        # Every errand open here is cut rather than ended, so whatever was carried for it is carried no further.
        self.owed.clear()
        # This instance's errands only. One buffer serves every instance of a run, and an episode ending here says nothing about the fight another instance is in the middle of.
        self.rollout.cut_all(owner=self.instance)

    def close(self) -> None:
        """Ends every open errand and releases this episode's trajectories to the trainer. The tactical arena has no intruder, so there is nothing to taint between the flush and the seal; the two are one call here and split only where an intruder sits above the layer."""
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
        #: How many errands were closed for each reason, so a run can be asked whether its terminals are firing. An operational errand takes its terminal only from outside, through finish, so this stays empty in a match and fills in the arena.
        self.terminals: Counter = Counter()
        self._state: List[float] = []
        self._regions: List[float] = []
        self._view: Optional[WorldView] = None
        self._spawns = tuple(region.id for region in getattr(session, "regions", ()) if region.spawn)

    def decide(self, view: WorldView, orders: OperationsOrders, squads: List[SquadRecord],
               reports: List[MissionReport], game_time_ms: int):
        self._view = view
        self._settle(view, orders, squads)
        self._state = operational_state(view, orders, squads, game_time_ms, self._spawns)
        self._regions = region_mask(view)
        return super().decide(view, orders, squads, reports, game_time_ms)

    def _settle(self, view: WorldView, orders: OperationsOrders, squads: Sequence[SquadRecord]) -> None:
        """Pays each squad's previous decision from the board that has now arrived, one squad at a time.

        Each decision is paid the shaping of the region its own contract named, not one board-wide figure shared out to all of them. The figure that paid all of them was the disease: what the strategic layer asks for is region by region, and a squad sent to a region it took has to be paid differently from one sent to a region it lost, or the advantage does not depend on the choice and the gradient is dead.

        A squad handed a new contract has begun a different errand and its old trajectory is cut, because advantage estimation would otherwise run what the new errand earned backwards into the decisions of the old, whose potential is measured against different ground. A squad gone from the board — folded into another by the organisation layer, disbanded, or wiped — is cut rather than ended: its last decision is bootstrapped from its own value estimate, as any decision that merely stopped being observed is, not closed against a terminal potential of nought. Marking it done would teach the critic that every state a squad turns over from, a routine merge of a healthy squad included, is worth nothing from here, corrupting the baseline every other squad's advantage is taken against. What ends an operational errand as a terminal comes only from outside, through finish, which the constructed arena calls at the horizon.
        """
        if self.rollout is None:
            return
        present = {squad.id for squad in squads}
        for squad in squads:
            outcome = self.reward.step(squad, view, orders)
            step = self.pending.pop(squad.id, None)
            if step is not None:
                step.reward = outcome.reward + self.owed.pop(squad.id, 0.0)
                step.done = outcome.done
                self.rollout.add((self.instance, squad.id), step)
            else:
                self.owed[squad.id] = self.owed.get(squad.id, 0.0) + outcome.reward
            if outcome.renewed:
                # Whatever was carried belonged to the errand being cut, whose potential was measured against different ground.
                self.owed.pop(squad.id, None)
                self.rollout.cut((self.instance, squad.id))
        for squad_id in [key for key in self.pending if key not in present]:
            self.pending.pop(squad_id, None)
            self.owed.pop(squad_id, None)
            self.rollout.cut((self.instance, squad_id))
            self.reward.forget(squad_id)

    def finish(self, squad: SquadRecord, terminal: float, reason: str) -> None:
        """Ends this squad's errand from outside, paying the decision still waiting on it as the last of its trajectory. Structurally the tactical layer's finish: whoever runs the contest knows when it is over, the layer only sees periods.

        The payment is the terminal handed in plus the last shaping term taken against a terminal potential of nought, so the shaping over the errand telescopes away and cannot change which policy is best. A squad whose errand already took its terminal has its still-waiting decision cut instead of paid, and a squad gone between periods has no decision waiting, so the terminal is added to the last step there was through the buffer's close-with — exactly the three cases the tactical finish handles.
        """
        if self.reward.ended(squad.id):
            self.pending.pop(squad.id, None)
            self.owed.pop(squad.id, None)
            if self.rollout is not None:
                self.rollout.cut((self.instance, squad.id))
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
        """
        if self.decider is None:
            return super()._settled(view, orders, squad, chosen, avoid)
        return chosen

    def _pick(self, view: WorldView, orders: OperationsOrders, squad: SquadRecord,
              avoid: Optional[int]) -> Optional[Tuple[Task, RegionState]]:
        if self._view is None or not self._state:
            return super()._pick(view, orders, squad, avoid)
        tasks = task_mask(squad.doctrine)
        if not any(tasks) or not any(self._regions):
            return None
        if self.decider is None:
            chosen = super()._pick(view, orders, squad, avoid)
            if chosen is None:
                return None
            choice = Choice(action=int(chosen[1].id), second=int(chosen[0]))
        else:
            choice = self.decider.choose(self._state, squad.id, self._regions, tasks)
        if choice is None:
            return None
        region = view.region(choice.action)
        if region is None:
            return None
        if self.rollout is not None:
            self.pending[squad.id] = Step(
                state=self._state, action=choice.action, mask=list(self._regions),
                second=choice.second, second_mask=list(tasks),
                log_prob=choice.total_log_prob, value=choice.value,
                squad=squad.id, at_ms=view.observation.game_time_ms)
        return Task(choice.second), region

    def flush(self) -> None:
        """Ends every open period at the end of an episode, without yet releasing its trajectories to be drained. Split from the seal for the same reason the tactical layer's is: the operational chain marks the episode's interference between the two, and this is the layer an intruder actually interferes with."""
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()
        # Every errand open here is cut rather than ended, so whatever was carried for it is carried no further.
        self.owed.clear()
        self.rollout.cut_all(owner=self.instance)
        self.reward.reset()

    def close(self) -> None:
        """Ends every open period and releases this episode's trajectories to the trainer. The operational chain reaches the seal through its policy's close, which taints first; this direct close is the path with no intruder above it, where flush and seal are one call."""
        self.flush()
        if self.rollout is not None:
            self.rollout.seal(self.instance)
