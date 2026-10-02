"""Where decisions and what they earned are kept until there is enough of them to learn from, and until they can be written down.

A trajectory here is one squad's errand, or one match's worth of operational decisions, and not one episode of the game. That is the design's horizon rule made concrete: the tactical layer's credit assignment closes at the contract boundary, so a squad given a new errand starts a new trajectory even though the match has not ended, and a squad destroyed ends one even though the others go on.

Decisions taken about a squad somebody interfered with are collected and marked rather than dropped at the point of collection. The design requires them to be kept out of the learning signal -praising a policy for what a human's squad achieved, or blaming it for what a human's squad lost, teaches the wrong thing -but which squads those were is not known until the interference happens, which may be several periods after the decision. Marking late and filtering at the end is the only order that works.

Advantages are computed here rather than in the optimiser because they need the trajectory intact and in order, and because the boundary between a trajectory that ended and one that was merely cut off when the episode stopped is a property of the collection: an errand still running when the match was called has to be bootstrapped from its last value estimate, and one that finished must not be.

A closed trajectory goes two ways. The optimiser drains it as soon as there is a batch, and a recorder (`sink`) receives it when the episode it belongs to is sealed: only then is everything about it final, since interference found at the end of an episode marks decisions taken long before.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

#: Discount, matching the one the shaping term telescopes with. A shaping term built with one discount and returns computed with another leaves a residue that does change which policy is optimal, which is the one thing potential-based shaping was chosen to avoid.
DISCOUNT = 0.99

#: How far the advantage estimator trades bias for variance. Nine tenths is the usual place to start and this project has no measurement that argues for anywhere else yet.
TRACE = 0.95

#: Discount for an errand that is a whole constructed fight, which is to say none at all.
#:
#: This is arithmetic rather than taste. Decisions are taken five times a second and a fight runs to a minute, so an errand in the arena is around a hundred decisions and three hundred at its longest. At a hundredth off per decision the terminal -which is the score of the fight, the very quantity the run is judged on -reaches the first decision of a long fight weighted by 0.99^300, about a twentieth, and the trace of 0.95 cuts the estimator's own reach to 1/(1-0.99*0.95), about seventeen decisions. Everything decided before the last few seconds would then be taught by the shaping alone, and shaping is by construction the one part of the reward that cannot change which policy is best.
#:
#: A fight is a finite episode with a real terminal, so it can simply be left undiscounted. Then the return of every decision in a fight is the score of that fight less the potential held at that decision, the shaping cancels out of the advantage against a fitted critic, and what is left is exactly the right question: how much better did this fight go than was expected from here.
FIGHT_DISCOUNT = 1.0

#: The trace for the same case, which is one for the same reason the discount is: the terminal arrives at a decision n steps earlier weighted by (discount times trace) to the n, so undiscounting with a trace short of one moves the truncation from the discount to the trace and leaves it where it was.
FIGHT_TRACE = 1.0


@dataclass
class Step:
    """One decision, the board it was taken from, and what followed."""

    state: List[float]
    #: What was played. The optimiser learns about this action, and the probabilities below are the ones it was drawn with.
    action: int
    #: Which actions were legal, so that the optimiser scores the same restricted distribution the decision was drawn from.
    mask: List[float]
    #: A second action drawn alongside the first where a layer chooses two things at once, as the operational layer chooses a region and a plan.
    second: int = -1
    #: Which second actions were legal, one row per first action laid end to end, since which plans are open depends on the region.
    second_mask: Sequence[float] = ()
    #: The log probability of what was played under the policy that played it, both parts together.
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
    #: Periods between this decision and the next one in its trajectory. A layer that decides only when something calls for it holds one decision across several periods; its reward is the discounted sum over them, and the next value is discounted by the discount to this power. Nought for a decision followed by another in the same period, as the economy's investments within one period are: no game time passes between them, so nothing is discounted.
    periods: int = 1
    #: What the decider said about where the decision came from: a decision inferred from a person's play carries its source, its weight and how well the person's orders agreed with it.
    meta: Dict[str, object] = field(default_factory=dict)
    #: What the teacher -the layer's own judge, or the person for a decision inferred from a replay- would have chosen on this board, and its distribution over both parts; -1 and empty where there was no teacher.
    label: int = -1
    second_label: int = -1
    soft: List[float] = field(default_factory=list)
    second_soft: List[float] = field(default_factory=list)
    #: The distributions what was played was drawn from: over the first part, and over the second given the first that was played. Empty where the decider could not say.
    probabilities: Sequence[float] = ()
    second_probabilities: Sequence[float] = ()
    #: The update count of the network that played, nought for anything that is not a learning network.
    version: int = 0
    #: The raw quantities each payment this decision received was priced from (`reward.TACTICAL_SIGNALS` and its siblings), in the order they were paid.
    signals: List[Tuple[float, ...]] = field(default_factory=list)
    #: What the state was encoded from (`materials`), or None when nothing is recording.
    materials: object = None
    #: The arena engagement this decision was taken in, -1 outside one.
    engagement: int = -1
    #: What the network read, where it differs from the recorded state: the tactical set row a set network decided on. Kept for the optimiser and never written to a dataset, which rebuilds it from the materials.
    net_state: Optional[List[float]] = None


@dataclass
class Trajectory:
    key: object
    steps: List[Step] = field(default_factory=list)
    #: True when the last step ended the errand rather than the collection being cut off.
    finished: bool = False
    #: The value of the state after the last step, used to bootstrap a trajectory that was cut off.
    tail_value: float = 0.0

    @property
    def owner(self) -> object:
        return self.key[0] if isinstance(self.key, tuple) and self.key else None


def estimate(rewards: Sequence[float], values: Sequence[float], periods: Sequence[int], finished: bool,
             tail_value: float, discount: float, trace: float) -> Tuple[List[float], List[float]]:
    """Generalised advantage estimation over one trajectory, backwards, as advantages and returns. The bootstrap after the final step is nought for an errand that ended and the value estimate for one that was cut off, which is the whole of the difference between the two cases.

    A step that spans several periods discounts what follows it by the discount raised to that many periods, and the trace likewise, so that a decision held for half a minute is discounted as half a minute rather than as one period, and one followed within its own period by the next is not discounted at all."""
    following = 0.0 if finished else tail_value
    advantage = 0.0
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        span = max(0, periods[index])
        factor = discount ** span
        delta = rewards[index] + factor * following - values[index]
        advantage = delta + factor * trace ** span * advantage
        advantages[index] = advantage
        returns[index] = advantage + values[index]
        following = values[index]
    return advantages, returns


#: What a sink is handed when an episode is sealed: its closed trajectories, a description of the episode, and the discount and trace they were collected under.
Sink = Callable[[List[Trajectory], dict, float, float], None]


class Rollout:
    """A batch under construction. Live trajectories by key, finished ones waiting to be drained, and closed ones waiting for their episode to be sealed.

    One buffer serves every instance of a run, each on its own thread, so every change to it is made under one lock.
    """

    def __init__(self, discount: float = DISCOUNT, trace: float = TRACE, sink: Optional[Sink] = None,
                 retain: bool = True) -> None:
        self.discount = discount
        self.trace = trace
        self.live: Dict[object, Trajectory] = {}
        self.done: List[Trajectory] = []
        #: Where sealed trajectories are written, or None when nothing is recording.
        self.sink = sink
        #: Whether closed trajectories are kept for `drain`. A run with no optimiser keeps none, so that its buffer holds an episode at a time rather than the whole run.
        self.retain = retain
        self._unsealed: Dict[object, List[Trajectory]] = {}
        self._lock = threading.RLock()

    @property
    def recording(self) -> bool:
        return self.sink is not None

    def __len__(self) -> int:
        with self._lock:
            return sum(len(t.steps) for t in self.done) + sum(len(t.steps) for t in self.live.values())

    def finished_steps(self) -> int:
        with self._lock:
            return sum(len(t.steps) for t in self.done)

    def _closed(self, trajectory: Trajectory) -> None:
        if self.retain:
            self.done.append(trajectory)
        if self.sink is not None:
            self._unsealed.setdefault(trajectory.owner, []).append(trajectory)

    def add(self, key: object, step: Step) -> None:
        with self._lock:
            trajectory = self.live.get(key)
            if trajectory is None:
                trajectory = self.live[key] = Trajectory(key=key)
            trajectory.steps.append(step)
            if step.done:
                trajectory.finished = True
                del self.live[key]
                self._closed(trajectory)

    def close_with(self, key: object, terminal: float, signal: Optional[Tuple[float, ...]] = None) -> bool:
        """Adds a terminal payment to the last decision of a live trajectory and ends it there, and says whether there was one to end.

        For the case where what ended the work is known only after the last decision about it was already paid and filed. A squad destroyed is the example this exists for: the period that discovers it is a period with no squad left to decide anything, so there is no outstanding decision to hang the ending on, and the last one there was has already gone into the trajectory as an ordinary step. Reaching back to it is the only way to say that the work ended rather than stopped being watched, and the difference between those two is the difference between a nought bootstrap and a value one.
        """
        with self._lock:
            trajectory = self.live.get(key)
            if trajectory is None or not trajectory.steps:
                return False
            last = trajectory.steps[-1]
            last.reward += terminal
            if signal is not None:
                last.signals.append(signal)
            last.done = True
            trajectory.finished = True
            del self.live[key]
            self._closed(trajectory)
            return True

    def last(self, key: object) -> Optional[Step]:
        with self._lock:
            trajectory = self.live.get(key)
            return trajectory.steps[-1] if trajectory is not None and trajectory.steps else None

    def cut(self, key: object, tail_value: Optional[float] = None) -> None:
        """Ends a trajectory that has not finished on its own -the match was called, or the squad passed out of this layer's hands. Its last step is bootstrapped rather than treated as terminal, because the errand did not fail, it merely stopped being observed.

        With no estimate offered, the last step's own value stands in for the one after it. That is an approximation and a much better one than nought: nought asserts that the errand was worth nothing from the moment observation stopped, which would teach a policy that having a match called on it is a failure, and matches are called on a fixed clock that no policy can affect.
        """
        with self._lock:
            trajectory = self.live.pop(key, None)
            if trajectory is None or not trajectory.steps:
                return
            trajectory.tail_value = trajectory.steps[-1].value if tail_value is None else tail_value
            self._closed(trajectory)

    def cut_all(self, tail_value: Optional[float] = None, owner: object = None) -> None:
        """Ends every trajectory still open, or every one belonging to one owner.

        The owner matters because one buffer serves every instance of a run: a trajectory is keyed by the instance it was collected on and the squad it is about, so cutting the whole buffer when one instance finishes an episode would reach into the other instances and cut the fight each of them is in the middle of. Those fights would then end with a bootstrap where they were about to be paid their score, which is the one payment the arena exists to make.

        Nothing is passed when the run itself is shutting down, which is the case the whole buffer is meant to be cut in.
        """
        with self._lock:
            for key in list(self.live):
                if owner is None or (isinstance(key, tuple) and key and key[0] == owner):
                    self.cut(key, tail_value)

    def taint(self, squads: Iterable[int], owner: object = None) -> None:
        """Marks every decision taken about these squads, in trajectories still open and in trajectories closed but not yet drained or sealed, as one somebody else interfered with. With an owner, only that instance's trajectories, since a squad number names a different squad on every instance."""
        touched = set(squads)
        if not touched:
            return
        with self._lock:
            pending = [t for group in self._unsealed.values() for t in group]
            for trajectory in list(self.done) + list(self.live.values()) + pending:
                if owner is not None and trajectory.owner != owner:
                    continue
                for step in trajectory.steps:
                    if step.squad in touched:
                        step.tainted = True

    def seal(self, owner: object, episode: dict) -> int:
        """Hands every trajectory of this owner closed since its last seal to the sink, as belonging to `episode`, and says how many decisions that was. Called once an episode is over and everything about its decisions is known."""
        with self._lock:
            trajectories = self._unsealed.pop(owner, [])
        if self.sink is None or not trajectories:
            return 0
        self.sink(trajectories, episode, self.discount, self.trace)
        return sum(len(t.steps) for t in trajectories)

    def seal_all(self, episode: dict) -> int:
        """Seals every owner at once, for a run shutting down whose episodes were never closed one by one."""
        with self._lock:
            owners = list(self._unsealed)
        return sum(self.seal(owner, dict(episode, instance=owner if isinstance(owner, int) else -1))
                   for owner in owners)

    def drain(self, keep_tainted: bool = False) -> List[Step]:
        """Every finished trajectory's steps, with advantages and returns filled in, oldest first. Live trajectories are left alone: they are still accruing."""
        with self._lock:
            finished, self.done = self.done, []
        out: List[Step] = []
        for trajectory in finished:
            self._finish(trajectory)
            out.extend(trajectory.steps)
        if keep_tainted:
            return out
        return [step for step in out if not step.tainted]

    def _finish(self, trajectory: Trajectory) -> None:
        steps = trajectory.steps
        advantages, returns = estimate([s.reward for s in steps], [s.value for s in steps], [s.periods for s in steps],
                                       trajectory.finished, trajectory.tail_value, self.discount, self.trace)
        for step, advantage, ret in zip(steps, advantages, returns):
            step.advantage = advantage
            step.ret = ret


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
