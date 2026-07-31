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

#: Discount for an errand that is a whole constructed fight, which is to say none at all.
#:
#: This is arithmetic rather than taste. Decisions are taken five times a second and a fight runs to a minute, so an errand in the arena is a hundred and thirteen decisions on average and three hundred at its longest. At a hundredth off per decision the terminal — which is the score of the fight, which is the very quantity the run is judged on — reaches the first decision of a fight weighted by 0.99^300, about a twentieth, and the trace of nine hundredths cuts the advantage estimator's own reach to 1/(1-0.99*0.95), about seventeen decisions, which is three and a half seconds of a fight lasting twenty. Everything decided before the last few seconds was therefore taught by the shaping alone, and shaping is by construction the one part of the reward that cannot change which policy is best. The layer was being trained on the only term that provably does not matter.
#:
#: A fight is a finite episode with a real terminal, so it can simply be left undiscounted. Then the return of every decision in a fight is the score of that fight less the potential held at that decision, the shaping cancels out of the advantage against a fitted critic, and what is left is exactly the right question: how much better did this fight go than was expected from here. Nothing about the shaping's guarantee is given up — it telescopes at any discount, this one included.
FIGHT_DISCOUNT = 1.0

#: The trace for the same case, which is one for the same reason the discount is.
#:
#: The trace is what actually decides how far the terminal reaches, and discounting at one does not on its own fix that: the terminal arrives at a decision n steps earlier weighted by (discount times trace) to the n, so a trace of ninety-seven hundredths over the hundred and thirty decisions of an ordinary fight delivers it at about two hundredths whatever the discount is. Undiscounting with a trace short of one moves the truncation from the discount to the trace and leaves it where it was.
#:
#: What a trace below one buys is variance reduction through the critic, and here there is almost none to buy. Every intermediate reward in a fight is a shaping difference, and shaping cancels against a critic that has learnt it, so the deltas between the first decision and the last carry nearly nothing and the estimator spends its bias budget on them for no return. At one, the advantage of a decision is exactly the score the fight came to, less the potential held at that decision, less what the critic expected: the whole fight, judged by how much better it went than was expected from there. That is the question the layer is being asked.
FIGHT_TRACE = 1.0


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
    #: Which squad it was about, so that interference discovered later can find it. The squad's own number as the game and the intruder know it, and therefore not a row of anything.
    squad: int = -1
    #: Which row of the squad block the decision was asked about, where a layer is asked one squad at a time. That is a different number from the squad's own, and the difference is the offset a side's numbering starts at: one process drives both sides of the constructed arena out of one numbering, so the other side's squads are some run of numbers that does not begin at nought. Kept apart from `squad` because the two are read by different things and were once the same field — the decider was asked with the offset row and the optimiser rebuilt the one-hot from the raw number, so the network was updated against a row it had not been asked about wherever the two differ. Minus one where the layer has no such row, which is every layer but the operational one.
    slot: int = -1
    at_ms: int = 0


@dataclass
class Trajectory:
    key: object
    steps: List[Step] = field(default_factory=list)
    #: True when the last step ended the errand rather than the collection being cut off.
    finished: bool = False
    #: Why a trajectory that did not finish was cut, so that a batch can be asked where its decisions went instead of having it read off the code. The layers name four places a cut is made: a new contract replaced the errand, the squad left the board, the episode closed around it, and its errand had already taken its terminal so the decisions after it belong to no errand at all. A trajectory that finished carries none, and a cut made by a caller that named no reason is recorded as unobserved. Nothing reads this to decide anything; it exists because the share of a batch no terminal ever reached is the first thing to ask of a layer that will not learn, and it was not a figure this buffer could produce.
    reason: str = ""
    #: The value of the state after the last step, used to bootstrap a trajectory that was cut off.
    tail_value: float = 0.0
    #: True once the episode this trajectory finished in has closed and its interference has been marked, which is the point past which the trainer may drain it. A trajectory that finished in the middle of an episode is added to the finished set the moment it ends, several periods before the episode closes and the intruder's touched set is complete; drained in that gap it would carry an interfered-with decision into the update untainted, because the tainting has not run yet. Held undrainable until sealed, it cannot.
    sealed: bool = False


@dataclass
class Census:
    """What one drained batch was made of, in the figures that say whether anything paid from outside reached the decisions in it.

    A trajectory that was cut is one no terminal ever reached. Its last decision is bootstrapped from its own value estimate, so at a discount and a trace of one that decision's advantage is exactly its own reward, and every earlier decision in it is anchored by nothing but the difference between two of the critic's own estimates — while the critic's regression target on those same steps is its own later estimate. A batch mostly made of cut trajectories is therefore mostly trained on the critic's opinion of itself, and its mean return is that opinion drifting rather than the layer learning. None of that can be told from the losses, the entropy or the return, which is why the shares are counted here and reported with every update instead of being argued from the code.

    The counts are of the batch as it will be handed to the optimiser, so decisions dropped for having been interfered with are not in them.
    """

    steps: int = 0
    #: Decisions lying in a trajectory that ended on a terminal, which are the only ones anything paid from outside can reach. The rest are taught by the shaping and the critic alone.
    paid_steps: int = 0
    #: Trajectories that ended on a terminal, and the ones that were cut, counted by what cut them.
    finished: int = 0
    cut: Dict[str, int] = field(default_factory=dict)
    #: Decisions whose advantage is exactly nought before the batch is normalised. Exactly rather than nearly, because the number worth watching is produced by an exact cancellation and not by a small quantity: the last step of a cut trajectory has `reward + value − value`, which is its own reward, and a period that pays nothing makes that identically nought. A batch in which this is a large share is a batch where that share of the sampled actions carries no information about itself at all, and after normalisation they all carry one identical nonzero number instead, which reinforces whatever the policy currently draws.
    zero_advantage: int = 0
    #: How many different advantages the batch holds. A batch of many decisions and a handful of distinct advantages is the same fault seen from the other side, and it is the one figure that separates a critic which has fitted something from one which has not.
    distinct: int = 0

    @classmethod
    def of(cls, trajectories: Sequence[Trajectory], steps: Sequence[Step]) -> "Census":
        kept = {id(step) for step in steps}
        census = cls(steps=len(steps))
        for trajectory in trajectories:
            if trajectory.finished:
                census.finished += 1
                census.paid_steps += sum(1 for step in trajectory.steps if id(step) in kept)
            else:
                census.cut[trajectory.reason] = census.cut.get(trajectory.reason, 0) + 1
        census.zero_advantage = sum(1 for step in steps if step.advantage == 0.0)
        census.distinct = len({step.advantage for step in steps})
        return census

    def as_dict(self) -> dict:
        return {"steps": self.steps, "paid_steps": self.paid_steps, "finished": self.finished,
                "cut": dict(self.cut), "zero_advantage": self.zero_advantage, "distinct": self.distinct}


class Rollout:
    """A batch under construction. Live trajectories by key, finished ones waiting to be drained."""

    def __init__(self, discount: float = DISCOUNT, trace: float = TRACE) -> None:
        self.discount = discount
        self.trace = trace
        self.live: Dict[object, Trajectory] = {}
        self.done: List[Trajectory] = []
        #: What the last drain took, kept so that whoever spends a batch can report how it was made up. Overwritten by each drain rather than accumulated, because the question it answers is about one update.
        self.census = Census()

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

    def cut(self, key: object, tail_value: Optional[float] = None,
            reason: str = "unobserved") -> None:
        """Ends a trajectory that has not finished on its own — the match was called, or the squad passed out of this layer's hands. Its last step is bootstrapped rather than treated as terminal, because the errand did not fail, it merely stopped being observed.

        With no estimate offered, the last step's own value stands in for the one after it. That is an approximation and a much better one than nought: nought asserts that the errand was worth nothing from the moment observation stopped, which would teach a policy that having a match called on it is a failure, and matches are called on a fixed clock that no policy can affect.

        The reason is written down and never read back by anything that decides: what it is for is the census, where a batch has to be able to say how much of itself no terminal ever reached and what took the rest away.
        """
        trajectory = self.live.pop(key, None)
        if trajectory is None or not trajectory.steps:
            return
        trajectory.tail_value = trajectory.steps[-1].value if tail_value is None else tail_value
        trajectory.reason = reason
        self.done.append(trajectory)

    def cut_all(self, tail_value: Optional[float] = None, owner: object = None,
                reason: str = "unobserved") -> None:
        """Ends every trajectory still open, or every one belonging to one owner.

        The owner matters because one buffer serves every instance of a run: a trajectory is keyed by the instance it was collected on and the squad it is about, so cutting the whole buffer when one instance finishes an episode reaches into eleven other instances and cuts the fight each of them is in the middle of. Those fights then end with a bootstrap where they were about to be paid their score, which is the one payment the arena exists to make. With a dozen instances each finishing an episode every half minute and a fight lasting about twenty seconds, that was most of them.

        Nothing is passed when the run itself is shutting down, which is the case the whole buffer is meant to be cut in.
        """
        for key in list(self.live):
            if owner is None or (isinstance(key, tuple) and key and key[0] == owner):
                self.cut(key, tail_value, reason)

    def taint(self, owner: object, squads: Iterable[int]) -> None:
        """Marks every decision this owner's instance took about these squads, in trajectories still open and in trajectories already finished, as one somebody else interfered with.

        Scoped to the owner, exactly as cut_all is, and for the same reason. One buffer serves every instance of a run and a trajectory is keyed by (instance, squad); the squad-number pool is identical on every instance, the organisation layer capping squads at eight. Tainting by the bare squad number would reach out of the instance the interference actually happened on and mark every other instance's clean decisions about the same number, dropping them from the update by a factor of the instance count. Matched against the trajectory key's first element, so a trajectory keyed some other way is left alone.
        """
        touched = set(squads)
        if not touched:
            return
        for trajectory in list(self.done) + list(self.live.values()):
            key = trajectory.key
            if not (isinstance(key, tuple) and key and key[0] == owner):
                continue
            for step in trajectory.steps:
                if step.squad in touched:
                    step.tainted = True

    def seal(self, owner: object = None) -> None:
        """Marks this owner's finished trajectories — every one when no owner is given — as ready to be drained into an update.

        Called at the end of an episode's close, after any interference with the episode has been marked. Until then a trajectory that finished in the middle of the episode sits in the finished set drainable, and the trainer thread runs on its own clock: it can pull that trajectory into an update before the episode closes and the tainting runs, so an intruder-touched decision leaks into the gradient untainted. Sealing only at close, after the taint, is what shuts that window. Scoped to the owner exactly as taint and cut_all are, because one buffer serves every instance of a run and an episode closing on one instance says nothing about the fight another is still in the middle of. Setting a flag in place rather than moving the trajectory keeps this safe to call from an instance's own thread while the trainer reads the same set: the trainer only ever removes trajectories, and only the ones already sealed.
        """
        for trajectory in self.done:
            key = trajectory.key
            if owner is None or (isinstance(key, tuple) and key and key[0] == owner):
                trajectory.sealed = True

    def drain(self, keep_tainted: bool = False, sealed_only: bool = False) -> List[Step]:
        """Every finished trajectory's steps, with advantages and returns filled in, oldest first. Live trajectories are left alone: they are still accruing.

        With ``sealed_only`` the trainer takes only the trajectories an episode's close has sealed and leaves the rest in place, so a trajectory that finished mid-episode is never drained before the tainting that decides whether it is clean has run. The finishing paths that take the whole buffer at once — the last update of a run and the collecting run's single read — seal everything first and drain without the gate.
        """
        if sealed_only:
            ready = [trajectory for trajectory in self.done if trajectory.sealed]
            self.done = [trajectory for trajectory in self.done if not trajectory.sealed]
        else:
            ready = self.done
            self.done = []
        out: List[Step] = []
        for trajectory in ready:
            self._finish(trajectory)
            out.extend(trajectory.steps)
        kept = out if keep_tainted else [step for step in out if not step.tainted]
        # Taken after the advantages are filled in and before anything normalises them, which is the only moment the exact cancellations are visible: normalising centres and scales the batch, so a mass of identical noughts comes out as a mass of one identical nonzero number and cannot be told from a batch that learnt something.
        self.census = Census.of(ready, kept)
        return kept

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
