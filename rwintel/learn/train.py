"""The optimiser, and the awkward fact that the environment cannot be paused.

Every textbook policy gradient alternates: collect a batch with fixed parameters, update, collect again. Nothing here can do that. The games are running in their own processes at ten times speed and they do not stop while an update is computed; there is no step function to withhold. So collection is continuous and the update happens on its own thread whenever enough has accumulated, which means a batch is always collected under parameters that drift slightly during its own collection.

That is survivable for exactly one reason, and it is the reason this is a proximal method rather than a plain policy gradient: every step carries the log probability the behaviour policy actually assigned to it, and the objective is a ratio against that, clipped. A modest lag between the parameters that acted and the parameters being updated is the case importance ratios exist for. What is not survivable is a large lag, so the batch is kept small enough to be collected in well under a minute of wall clock and the update is kept short.

Weights are read by the inference path on other threads while this one writes them, so both take the same lock. Inference is already batched into a handful of calls a second and an update is a few milliseconds, so the contention is not worth avoiding; a torn read of a weight matrix would be, and would show up as a policy that occasionally behaved like nothing in particular.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch
from torch import nn

from .net import entropy, one_hot_slot
from .rollout import Rollout, Step, normalise

log = logging.getLogger(__name__)

#: How far a single update may move the probability of an action it is reweighting. The clip is the whole of what makes this method safe against the drift described above.
CLIP = 0.2

#: How much the value head's error counts against the policy's, sharing a trunk.
VALUE_WEIGHT = 0.5

#: How much an undecided policy is worth. Small, but not nought: with this sample budget a policy that collapses onto one action in the first few hundred steps never recovers, because it stops producing the evidence that would argue it out.
#:
#: Small because the intended way to start a run is from an imitation of the handwritten layer, and that policy is deliberately narrow: it reproduces a rule ladder, so it is about nine tenths sure of itself and its entropy is a quarter of an even policy's. At a fiftieth this term does not preserve that narrowness, it removes it. Measured, two hundred updates at a fiftieth took the entropy from four tenths to nine tenths while the return did not move at all — the policy was not being improved, it was being dissolved, and the term paying for the dissolution was the only one with a consistent gradient. What is left here is enough to keep a policy from closing an action off entirely and not enough to undo where it started.
ENTROPY_WEIGHT = 0.002

#: The largest gradient norm allowed through. A single episode in which a squad was wiped can otherwise produce a step that undoes an hour.
MAX_GRADIENT = 0.5

LEARNING_RATE = 3e-4

#: Passes over each batch. More than one is what makes the method worth its ratio machinery; many more, with a batch this small, overfits it.
EPOCHS = 4

MINIBATCH = 256

#: Steps that make one update. Small deliberately: a batch is collected while the parameters that collected it are already moving, and a clipped ratio covers a modest lag rather than a large one, so this is kept to what fills in well under a minute of wall clock.
BATCH = 1024


@dataclass
class Report:
    """One update, in the numbers that say whether it did anything."""

    updates: int = 0
    steps: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    mean_return: float = 0.0
    mean_reward: float = 0.0
    clipped: float = 0.0
    #: True while the update was a value-only one, so that a run of flat policy losses at the start of a log is read as the critic catching up rather than as a policy that has stopped moving.
    warming: bool = False

    def as_dict(self) -> dict:
        return {key: (round(value, 5) if isinstance(value, float) else value)
                for key, value in vars(self).items()}


class Optimiser:
    """Proximal policy optimisation over the steps a rollout has finished with.

    One class serves both layers. The only difference between them is that the operational layer chooses two things at once, so its log probability is the sum of two and its entropy bonus is the sum of two; everything else — the ratio, the clip, the value target, the gradient clipping — is identical, and writing it twice would be two things to keep in step for no gain.
    """

    def __init__(self, net: nn.Module, device=None, learning_rate: float = LEARNING_RATE,
                 two_headed: bool = False, warmup: int = 0,
                 entropy_weight: float = ENTROPY_WEIGHT) -> None:
        self.net = net
        self.device = device
        self.two_headed = two_headed
        #: How hard the objective pushes the policy back towards choosing evenly. A run starting from noise wants this, because a policy that collapses onto one action in its first few hundred steps stops producing the evidence that would overturn it. A run starting from an imitation of the handwritten layer wants much less of it: that policy is already narrow on purpose, and the term does not preserve the narrowness it does not know the reason for.
        self.entropy_weight = entropy_weight
        #: Updates at the start of a run that fit the value head and nothing else. Meant for a policy that arrived from somewhere — an imitation of the handwritten layer, an earlier run — because its critic did not arrive with it: the value head is still random, so every advantage the first few thousand steps produce is noise of about the size of the returns, and a policy gradient taken against that dismantles the policy before it has been paid for anything. Nought is right for a run starting from a fresh policy, where there is nothing to protect.
        self.warmup = warmup
        self.optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
        #: Held by anything that reads the weights, which is every inference call on every other thread.
        self.lock = threading.Lock()
        #: Held for the whole of one update, which the lock above deliberately is not. A warm-up freezes the trunk by setting a flag on the module, and a flag on the module is not something a per-minibatch lock protects: two updates overlapping would have the one that finished first lift the other's freeze half way through and let the value loss into the shared trunk, which is the one thing a warm-up exists to prevent. The mirror image is as bad and quieter -- an ordinary update running inside somebody else's freeze drops its policy gradient and reports nothing unusual. Overlap is not hypothetical, because the run's last update is spent by whoever is shutting the run down while the trainer thread may still be inside one.
        self.updating = threading.Lock()
        self.report = Report()

    def update(self, steps: Sequence[Step]) -> Report:
        if not steps:
            return self.report
        with self.updating:
            return self._update(steps)

    def _update(self, steps: Sequence[Step]) -> Report:
        normalise(steps)
        states = torch.tensor([step.state for step in steps], dtype=torch.float32, device=self.device)
        actions = torch.tensor([step.action for step in steps], dtype=torch.long, device=self.device)
        masks = torch.tensor([step.mask for step in steps], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([step.advantage for step in steps], dtype=torch.float32, device=self.device)
        returns = torch.tensor([step.ret for step in steps], dtype=torch.float32, device=self.device)
        old = torch.tensor([step.log_prob for step in steps], dtype=torch.float32, device=self.device)
        slots = seconds = None
        if self.two_headed:
            slots = torch.stack([one_hot_slot(step.squad, device=self.device) for step in steps])
            seconds = (torch.tensor([step.second for step in steps], dtype=torch.long, device=self.device),
                       torch.tensor([list(step.second_mask) for step in steps], dtype=torch.float32,
                                    device=self.device))

        count = len(steps)
        order = torch.randperm(count, device=self.device)
        totals = [0.0, 0.0, 0.0, 0.0]
        batches = 0
        warming = self.report.updates < self.warmup
        if warming:
            self._hold_policy(True)
        try:
            for _ in range(EPOCHS):
                for start in range(0, count, MINIBATCH):
                    index = order[start:start + MINIBATCH]
                    with self.lock:
                        losses = self._step(warming, states[index], actions[index], masks[index],
                                            advantages[index], returns[index], old[index],
                                            None if slots is None else slots[index],
                                            None if seconds is None else (seconds[0][index], seconds[1][index]))
                    for position, value in enumerate(losses):
                        totals[position] += value
                    batches += 1
        finally:
            # Put back however the update ended, including badly. A run that left the trunk frozen after a failed update would go on collecting and go on reporting and never move the policy again.
            if warming:
                self._hold_policy(False)

        self.report = Report(
            updates=self.report.updates + 1,
            steps=count,
            policy_loss=totals[0] / max(1, batches),
            value_loss=totals[1] / max(1, batches),
            entropy=totals[2] / max(1, batches),
            clipped=totals[3] / max(1, batches),
            mean_return=float(returns.mean()),
            mean_reward=sum(step.reward for step in steps) / count,
            warming=warming,
        )
        return self.report

    def _hold_policy(self, held: bool) -> None:
        """Stops everything but the value head from moving, or lets it move again.

        The trunk has to be held as well as the action head, and that is the whole point rather than a precaution. It is shared, so a value loss allowed through to it moves the features the action head reads: the policy would be taken apart by the fitting of its own critic, which is precisely the thing a warm-up is being run to prevent. What is wanted at the end of one is a critic that has caught up with an actor that has not moved.
        """
        spared = {id(parameter) for parameter in self.net.value.parameters()}
        for parameter in self.net.parameters():
            if id(parameter) not in spared:
                parameter.requires_grad_(not held)

    def _step(self, warming, states, actions, masks, advantages, returns, old, slots, second):
        if self.two_headed:
            logits, task_logits, values = self.net(states, slots, masks, second[1])
        else:
            logits, values = self.net(states, masks)
            task_logits = None
        value_loss = torch.nn.functional.mse_loss(values, returns)

        if warming:
            # Only the critic is being fitted, so only its error is charged. Nothing of the policy is computed at all rather than computed and thrown away: with the trunk and the action head held still there is no parameter left for either the ratio or the entropy bonus to reach, and reporting both as nought is what makes the log say that no policy step was taken here.
            self._descend(VALUE_WEIGHT * value_loss)
            return 0.0, value_loss.item(), 0.0, 0.0

        log_prob = _log_prob(logits, actions)
        spread = entropy(logits)
        if task_logits is not None:
            # A layer that chooses two things at once decided one thing, so the probability of what it did is the product of the two and the entropy of what it might have done is the sum.
            log_prob = log_prob + _log_prob(task_logits, second[0])
            spread = spread + entropy(task_logits)

        ratio = torch.exp(log_prob - old)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        bonus = spread.mean()
        self._descend(policy_loss + VALUE_WEIGHT * value_loss - self.entropy_weight * bonus)
        with torch.no_grad():
            share = float(((ratio - 1.0).abs() > CLIP).float().mean())
            return policy_loss.item(), value_loss.item(), bonus.item(), share

    def _descend(self, loss: torch.Tensor) -> None:
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRADIENT)
        self.optimiser.step()


def _log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).log_prob(actions)


class Trainer(threading.Thread):
    """Watches a rollout and updates whenever there is enough in it.

    Running as a thread rather than as a loop the runner drives is what lets the games keep going. The alternative — stopping the world between batches — is not available in a process that does not own the clock.
    """

    def __init__(self, rollout: Rollout, optimiser: Optimiser, batch: int = BATCH,
                 on_update: Optional[Callable[[Report], None]] = None) -> None:
        super().__init__(name="trainer", daemon=True)
        self.rollout = rollout
        self.optimiser = optimiser
        self.batch = batch
        self.on_update = on_update
        self.reports: List[Report] = []
        # Named _halt rather than _stop on purpose: this class is a Thread, and Thread._stop is a method the standard library calls on itself from inside join, through _wait_for_tstate_lock, the moment the thread has finished. An instance attribute called _stop shadows that method, so join tries to call this Event and raises TypeError: 'Event' object is not callable. That is exactly where finish fails — it sets the flag, then joins — so every training run would crash on the way out, after the last episode and before the trained parameters were saved. The name is the whole of the fix.
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            finished = sum(len(t.steps) for t in self.rollout.done)
            if finished < self.batch:
                self._halt.wait(0.25)
                continue
            steps = self.rollout.drain()
            if not steps:
                continue
            report = self.optimiser.update(steps)
            self.reports.append(report)
            log.info("update %d over %d step(s): policy %+.4f value %.4f entropy %.3f return %+.3f reward %+.4f clipped %.2f%s",
                     report.updates, report.steps, report.policy_loss, report.value_loss,
                     report.entropy, report.mean_return, report.mean_reward, report.clipped,
                     "   (warming the value head, the policy is held still)" if report.warming else "")
            if self.on_update is not None:
                self.on_update(report)

    def finish(self) -> Optional[Report]:
        """Stops collecting and spends whatever is left. A last partial batch is worth taking: the alternative is throwing away the most recent and most relevant experience of the run."""
        self._halt.set()
        self.join(timeout=5.0)
        self.rollout.cut_all()
        steps = self.rollout.drain()
        if not steps:
            return self.reports[-1] if self.reports else None
        report = self.optimiser.update(steps)
        self.reports.append(report)
        return report
