"""Where decisions and what they earned are kept until there is enough of them to learn from.

A trajectory here is one squad's errand, or one match's worth of operational decisions, and not one episode of the game. That is the design's horizon rule made concrete: the tactical layer's credit assignment closes at the contract boundary, so a squad given a new errand starts a new trajectory even though the match has not ended, and a squad destroyed ends one even though the others go on.

Decisions taken about a squad somebody interfered with are collected and marked rather than dropped at the point of collection. The design requires them to be kept out of the learning signal — praising a policy for what a human's squad achieved, or blaming it for what a human's squad lost, teaches the wrong thing — but which squads those were is not known until the interference happens, which may be several periods after the decision. Marking late and filtering at the end is the only order that works.

Advantages are computed here rather than in the optimiser because they need the trajectory intact and in order, and because the boundary between a trajectory that ended and one that was merely cut off when the episode stopped is a property of the collection: an errand still running when the match was called has to be bootstrapped from its last value estimate, and one that finished must not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: Discount, matching the one the shaping term telescopes with. A shaping term built with one discount and returns computed with another leaves a residue that does change which policy is optimal, which is the one thing potential-based shaping was chosen to avoid.
DISCOUNT = 0.99

#: How far the advantage estimator trades bias for variance. Nine tenths is the usual place to start and this project has no measurement that argues for anywhere else yet.
TRACE = 0.95


@dataclass
class Step:
    """One decision, the board it was taken from, and what followed."""

    state: List[float]
    action: int
    #: Which actions were legal, so that the optimiser scores the same restricted distribution the decision was drawn from.
    mask: List[float]
    #: A second action drawn alongside the first where a layer chooses two things at once, as the operational layer chooses a region and a task.
    second: int = -1
    second_mask: Sequence[float] = ()
    log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    done: bool = False
    #: Filled in when the trajectory is closed.
    advantage: float = 0.0
    ret: float = 0.0
    #: True once anyone outside the command chain has touched the squad this decision was about.
    tainted: bool = False
    #: Which squad it was about, so that interference discovered later can find it.
    squad: int = -1
    at_ms: int = 0


@dataclass
class Trajectory:
    key: object
    steps: List[Step] = field(default_factory=list)
    #: True when the last step ended the errand rather than the collection being cut off.
    finished: bool = False
    #: The value of the state after the last step, used to bootstrap a trajectory that was cut off.
    tail_value: float = 0.0


class Rollout:
    """A batch under construction. Live trajectories by key, finished ones waiting to be drained."""

    def __init__(self, discount: float = DISCOUNT, trace: float = TRACE) -> None:
        self.discount = discount
        self.trace = trace
        self.live: Dict[object, Trajectory] = {}
        self.done: List[Trajectory] = []

    def __len__(self) -> int:
        return sum(len(t.steps) for t in self.done) + sum(len(t.steps) for t in self.live.values())

    def add(self, key: object, step: Step) -> None:
        trajectory = self.live.get(key)
        if trajectory is None:
            trajectory = self.live[key] = Trajectory(key=key)
        trajectory.steps.append(step)
        if step.done:
            trajectory.finished = True
            self.done.append(trajectory)
            del self.live[key]

    def close_with(self, key: object, terminal: float) -> bool:
        """Adds a terminal payment to the last decision of a live trajectory and ends it there, and says whether there was one to end.

        For the case where what ended the work is known only after the last decision about it was already paid and filed. A squad destroyed is the example this exists for: the period that discovers it is a period with no squad left to decide anything, so there is no outstanding decision to hang the ending on, and the last one there was has already gone into the trajectory as an ordinary step. Reaching back to it is the only way to say that the work ended rather than stopped being watched, and the difference between those two is the difference between a nought bootstrap and a value one.
        """
        trajectory = self.live.get(key)
        if trajectory is None or not trajectory.steps:
            return False
        trajectory.steps[-1].reward += terminal
        trajectory.steps[-1].done = True
        trajectory.finished = True
        self.done.append(trajectory)
        del self.live[key]
        return True

    def cut(self, key: object, tail_value: Optional[float] = None) -> None:
        """Ends a trajectory that has not finished on its own — the match was called, or the squad passed out of this layer's hands. Its last step is bootstrapped rather than treated as terminal, because the errand did not fail, it merely stopped being observed.

        With no estimate offered, the last step's own value stands in for the one after it. That is an approximation and a much better one than nought: nought asserts that the errand was worth nothing from the moment observation stopped, which would teach a policy that having a match called on it is a failure, and matches are called on a fixed clock that no policy can affect.
        """
        trajectory = self.live.pop(key, None)
        if trajectory is None or not trajectory.steps:
            return
        trajectory.tail_value = trajectory.steps[-1].value if tail_value is None else tail_value
        self.done.append(trajectory)

    def cut_all(self, tail_value: Optional[float] = None) -> None:
        for key in list(self.live):
            self.cut(key, tail_value)

    def taint(self, squads: Iterable[int]) -> None:
        """Marks every decision taken about these squads, in trajectories still open and in trajectories already finished, as one somebody else interfered with."""
        touched = set(squads)
        if not touched:
            return
        for trajectory in list(self.done) + list(self.live.values()):
            for step in trajectory.steps:
                if step.squad in touched:
                    step.tainted = True

    def drain(self, keep_tainted: bool = False) -> List[Step]:
        """Every finished trajectory's steps, with advantages and returns filled in, oldest first. Live trajectories are left alone: they are still accruing."""
        out: List[Step] = []
        for trajectory in self.done:
            self._finish(trajectory)
            out.extend(trajectory.steps)
        self.done = []
        if keep_tainted:
            return out
        return [step for step in out if not step.tainted]

    def _finish(self, trajectory: Trajectory) -> None:
        """Generalised advantage estimation over one trajectory, backwards. The bootstrap after the final step is nought for an errand that ended and the value estimate for one that was cut off, which is the whole of the difference between the two cases."""
        following = 0.0 if trajectory.finished else trajectory.tail_value
        advantage = 0.0
        for step in reversed(trajectory.steps):
            delta = step.reward + self.discount * following - step.value
            advantage = delta + self.discount * self.trace * advantage
            step.advantage = advantage
            step.ret = advantage + step.value
            following = step.value


def normalise(steps: Sequence[Step]) -> None:
    """Centres and scales the advantages of a batch in place. Without it the size of a gradient step depends on how eventful the episodes in the batch happened to be, which makes a learning rate that worked once fail on the next batch for no reason to do with the policy."""
    if len(steps) < 2:
        return
    mean = sum(step.advantage for step in steps) / len(steps)
    variance = sum((step.advantage - mean) ** 2 for step in steps) / len(steps)
    spread = variance ** 0.5
    if spread < 1e-8:
        for step in steps:
            step.advantage = 0.0
        return
    for step in steps:
        step.advantage = (step.advantage - mean) / spread
