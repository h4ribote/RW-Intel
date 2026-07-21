"""The two learnt layers, each a script layer with one decision taken out and replaced.

Everything except the decision is inherited. The tactical layer's mission reports, its bookkeeping of what a squad has destroyed, the rule that a departure other than holding is re-issued every period; the operational layer's pricing of a mission against the strategic allowance, its deadlines, the rule that a contract is only re-issued when it differs from the one held — all of that is the same code running. What is overridden is one method each: which of the five departures, and which region under which task.

That is the design's own boundary and not a convenience. Layers are learnt one at a time against frozen neighbours, and a learnt layer must be substitutable for the script layer in the sense that the rest of the system cannot tell which is present. Inheriting rather than reimplementing is how that is guaranteed rather than hoped for: there is no second copy of the rules to drift.

Reward arrives one period late by construction. A decision cannot be paid until the board has moved under it, so each period pays the previous decision from the board that has now arrived, and only then makes a new one. This is the same one-period structure the interface already has for the action itself, and keeping the two aligned means a step in the buffer covers exactly one period of game time.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..wire import Deviation, RegionState, Status, Task
from ..control.policy.contracts import DOCTRINES, MissionReport, OperationsOrders, SquadRecord
from ..control.policy.operations import Operations
from ..control.policy.tactics import Tactics
from ..control.policy.view import Sighting, WorldView
from .deciders import Choice
from .encoding import operational_state, region_mask, tactical_state, task_mask
from .reward import OperationalReward, TacticalReward
from .rollout import Rollout, Step

log = logging.getLogger(__name__)


class LearntTactics(Tactics):
    """The tactical layer with the choice of departure taken from a decider instead of from the rule ladder."""

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1) -> None:
        super().__init__(session, catalogue)
        self.decider = decider
        self.rollout = rollout
        self.instance = instance
        self.reward = TacticalReward()
        #: The decision each squad is owed payment for, held until the next period says what it earned.
        self.pending: Dict[int, Step] = {}
        self._view: Optional[WorldView] = None
        self._now = 0

    def decide(self, view: WorldView, squads: List[SquadRecord], game_time_ms: int):
        # Held on the instance because the method this class exists to override is handed only what the rule needs, and what a network needs is the board. Overriding the caller instead would mean copying the reporting half of the layer, which is the half that has to stay identical.
        self._view = view
        self._now = game_time_ms
        self._settle(view, squads, game_time_ms)
        return super().decide(view, squads, game_time_ms)

    def _settle(self, view: WorldView, squads: Sequence[SquadRecord], game_time_ms: int) -> None:
        """Pays every outstanding decision from the board that has just arrived, and closes the errands that have ended."""
        if self.rollout is None:
            return
        present = {squad.id for squad in squads}
        for squad in squads:
            outcome = self.reward.step(squad, view, game_time_ms)
            step = self.pending.pop(squad.id, None)
            if step is None:
                continue
            step.reward = outcome.reward
            step.done = outcome.done
            self.rollout.add((self.instance, squad.id), step)
        for squad_id in [key for key in self.pending if key not in present]:
            # A squad that has left the board between periods cannot be paid from anything, so its last decision is cut off rather than scored.
            self.pending.pop(squad_id, None)
            self.rollout.cut((self.instance, squad_id))
            self.reward.forget(squad_id)

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

    def close(self) -> None:
        """Ends every open errand at the end of an episode. They did not fail; they stopped being observed, so they are bootstrapped rather than treated as terminal."""
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()
        self.rollout.cut_all()


class LearntOperations(Operations):
    """The operational layer with the choice of where and what taken from a decider."""

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1) -> None:
        super().__init__(session, catalogue)
        self.decider = decider
        self.rollout = rollout
        self.instance = instance
        self.reward = OperationalReward()
        self.pending: Dict[int, Step] = {}
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
        """Pays the previous period's decisions.

        One figure pays all of them, because what the strategic layer asked for is a statement about the whole board and not about any one squad: how much of the ground it called valuable is being stood on, how far ahead the army is, how much of the allowance has gone. Cutting that per squad would need an attribution nothing in the observation supports, and inventing one would be a stronger claim than the measurement can carry.
        """
        if self.rollout is None:
            return
        outcome = self.reward.step(view, orders, squads)
        present = {squad.id for squad in squads}
        for squad_id, step in list(self.pending.items()):
            step.reward = outcome.reward
            step.done = squad_id not in present
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()

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

    def close(self) -> None:
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()
        self.rollout.cut_all()
        self.reward.reset()
