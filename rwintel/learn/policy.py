"""The command chain with one layer learnt and the rest frozen.

This is the arrangement the design insists on for every training run: exactly one layer is being changed, everything else is the script, and the interference of an intruder is present because the operational layer has to be robust to it and because evaluation is done with it too. A run that moved two layers at once could not attribute the difference it measured to either of them, and the sample budget does not allow the number of runs it would take to find out which.

Building it by substitution rather than by assembly is what keeps the two comparable. The learnt policy is a script policy with one attribute replaced, so everything the comparison holds constant is held constant by construction rather than by care.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ..control.policy import ScriptPolicy
from .layers import LearntOperations, LearntTactics
from .rollout import Rollout

log = logging.getLogger(__name__)

TACTICAL = "tactics"
OPERATIONAL = "operations"


class LearningPolicy(ScriptPolicy):
    """The chain with one layer taken from a decider. Everything else is the script it is measured against."""

    def __init__(self, session, layer: str, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1) -> None:
        super().__init__(session)
        self.layer = layer
        if layer == TACTICAL:
            self.tactics = LearntTactics(session, self.catalogue, decider, rollout, instance)
        elif layer == OPERATIONAL:
            self.operations = LearntOperations(session, self.catalogue, decider, rollout, instance)
        else:
            raise ValueError(f"no layer named {layer!r}: expected {TACTICAL!r} or {OPERATIONAL!r}")

    def close(self) -> None:
        """Ends the episode's open trajectories, and marks every decision about a squad somebody interfered with.

        The marking happens here rather than where the interference happened because an intervention half way through an errand invalidates the decisions taken before it as well as after: the outcome the whole errand is paid on is no longer the outcome of what this layer chose. Doing it at the end is the only point at which the full list of interfered-with squads is known.
        """
        learnt = getattr(self, self.layer, None)
        rollout = getattr(learnt, "rollout", None)
        # Flush this instance's outstanding decisions into the buffer before tainting, not after. The layer's close is where the decision each squad was still owed payment for is finally added; run after the taint, those freshly added steps escape it, so the last decision about an interfered squad — a seized or rewritten squad still on the board at the episode's end — would enter the update untainted while the decisions before it were dropped.
        if hasattr(learnt, "close"):
            learnt.close()
        if rollout is not None:
            for commander in self.outside:
                touched = getattr(getattr(commander, "log", None), "touched", None)
                if touched:
                    # This instance only. The buffer is shared across instances and keyed by (instance, squad); tainting by the bare squad number would drop every other instance's clean decisions about the same number.
                    rollout.taint(learnt.instance, touched)


def learning_arm(layer: str, decider_for: Callable[[object], object],
                 rollout: Rollout, intruders: Optional[Callable[[object], object]] = None) -> Callable:
    """An arm that runs the chain with one layer learnt, for the evaluation runner and the training runner alike.

    The decider is built per session rather than shared, because it is the thing that knows which instance it is answering for; the network behind it is shared, which is the whole point of batching the inference across instances.
    """

    def build(session) -> LearningPolicy:
        policy = LearningPolicy(session, layer, decider_for(session), rollout, session.instance)
        if intruders is not None:
            intruder = intruders(session)
            if intruder is not None:
                intruder.organisation = policy.organisation
                policy.outside.append(intruder)
        return policy

    return build
