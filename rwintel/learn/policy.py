"""The command chain with one layer learnt and the rest frozen.

This is the arrangement the design insists on for every training run: exactly one layer is being changed and every other layer is held still, and the interference of an intruder is present because the operational layer has to be robust to it and because evaluation is done with it too. Held still means the handwritten script unless the run names a trained layer to freeze, which is the second half of the learning order — a layer is meant to be learnt against the layers already improved beneath it, and `frozen` is where those go. A run that moved two layers at once could not attribute the difference it measured to either of them, and the sample budget does not allow the number of runs it would take to find out which.

Building it by substitution rather than by assembly is what keeps the two comparable. The learnt policy is a script policy with one attribute replaced, so everything the comparison holds constant is held constant by construction rather than by care.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from ..control.policy import ScriptPolicy
from ..eval.scoring import score
from .layers import LearntOperations, LearntStrategy, LearntTactics
from .rollout import Rollout

log = logging.getLogger(__name__)

TACTICAL = "tactics"
OPERATIONAL = "operations"
STRATEGIC = "strategy"

#: Every layer this chain can have replaced by a learnt one, in the order the design learns them: the tactical layer first because it can be trained without playing matches, then the operational layer against a frozen tactical one, and the strategic layer last because a match is the only board its choice is made on.
LAYERS = (TACTICAL, OPERATIONAL, STRATEGIC)


class LearningPolicy(ScriptPolicy):
    """The chain with one layer taken from a decider. Everything else is the script it is measured against."""

    def __init__(self, session, layer: str, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, frozen: Optional[Dict[str, object]] = None) -> None:
        super().__init__(session)
        self.layer = layer
        #: Other layers replaced by trained ones and held still, which is the second half of what the design means by learning one layer against frozen neighbours. Without them the neighbours are the handwritten script, which is what every run so far has been trained against — a coherent thing to measure and not the thing the learning order asks for once a layer below has actually been improved.
        self.frozen = dict(frozen or {})
        if layer in self.frozen:
            raise ValueError("the %s layer cannot be both trained and frozen in one policy" % layer)
        for other, held in self.frozen.items():
            # No rollout and no instance: a frozen layer is read, not learnt from, so with nowhere to record a decision it records none. Handing it this run's buffer would splice one layer's states and actions into another layer's trajectories, and the trainer would feed the lot to a network that reads neither.
            self._replace(session, other, held, None, -1)
        self._replace(session, layer, decider, rollout, instance)

    def _replace(self, session, layer: str, decider, rollout: Optional[Rollout], instance: int) -> None:
        """Puts one learnt layer in place of the script's. One place rather than three so that a layer being trained and a layer being frozen are built by the same code, and cannot come to differ in anything but the rollout."""
        if layer == TACTICAL:
            self.tactics = LearntTactics(session, self.catalogue, decider, rollout, instance)
        elif layer == OPERATIONAL:
            self.operations = LearntOperations(session, self.catalogue, decider, rollout, instance)
        elif layer == STRATEGIC:
            self.strategy = LearntStrategy(session, self.catalogue, decider, rollout, instance)
        else:
            raise ValueError("no layer named %r: expected one of %s" % (layer, ", ".join(repr(l) for l in LAYERS)))

    def conclude(self, episode) -> None:
        """Hands the layer the result of the match, where the layer is one that is paid the match.

        Only the strategic layer is, and that is the design's rule rather than an accident of what is implemented: every other layer is paid for meeting the contract handed down to it and never sees its superior's reward, which is what keeps credit assignment from crossing a layer boundary. So this reaches a `conclude` that only the strategic layer defines, and is a no-op under the other two.

        Called by whoever is running the match, before the policy is closed, because the result is a statement about the episode that only the session holds — the same shape as an arena's `finish`, which is handed in from outside for exactly the same reason. The score is the project's own episode score: a decision saturates it, and a match cut off by the clock is scored on the board it was cut off with.

        A match interfered with is scored as it stands. Squad-level interference taints the decisions taken about those squads, and the strategic decision is about no squad — it is about the whole side, whose result is the result including whatever an intruder did to it. That is the same convention the design states for evaluation, which is to be done with intruders present.
        """
        concluded = getattr(getattr(self, self.layer, None), "conclude", None)
        if concluded is not None:
            concluded(score(episode), "match")

    def close(self) -> None:
        """Ends the episode's open trajectories, and marks every decision about a squad somebody interfered with.

        The marking happens here rather than where the interference happened because an intervention half way through an errand invalidates the decisions taken before it as well as after: the outcome the whole errand is paid on is no longer the outcome of what this layer chose. Doing it at the end is the only point at which the full list of interfered-with squads is known.
        """
        learnt = getattr(self, self.layer, None)
        rollout = getattr(learnt, "rollout", None)
        # Flush this instance's outstanding decisions into the buffer before tainting, not after. The layer's flush is where the decision each squad was still owed payment for is finally added; run after the taint, those freshly added steps escape it, so the last decision about an interfered squad — a seized or rewritten squad still on the board at the episode's end — would enter the update untainted while the decisions before it were dropped. Flush, not close: close would also seal, and a sealed trajectory is drainable, so sealing before the taint would reopen exactly the window the taint closes. The seal is run below, once the taint has marked every touched decision.
        if hasattr(learnt, "flush"):
            learnt.flush()
        if rollout is not None:
            for commander in self.outside:
                touched = getattr(getattr(commander, "log", None), "touched", None)
                if touched:
                    # This instance only. The buffer is shared across instances and keyed by (instance, squad); tainting by the bare squad number would drop every other instance's clean decisions about the same number.
                    rollout.taint(learnt.instance, touched)
            # Now that the episode's interference is marked, release this instance's finished trajectories to the trainer. Unconditional, not only when something was touched: a run without an intruder still has to seal, or the trainer — which now drains only sealed trajectories — never takes anything and the policy never moves. A trajectory that finished mid-episode has waited in the buffer undrainable until this point, which is the whole of the fix; without it, the trainer thread could have drained and updated it several periods before the taint above ran.
            rollout.seal(learnt.instance)


def learning_arm(layer: str, decider_for: Callable[[object], object],
                 rollout: Rollout, intruders: Optional[Callable[[object], object]] = None,
                 frozen_for: Optional[Callable[[], Dict[str, object]]] = None) -> Callable:
    """An arm that runs the chain with one layer learnt, for the evaluation runner and the training runner alike.

    The decider is built per session rather than shared, because it is the thing that knows which instance it is answering for; the network behind it is shared, which is the whole point of batching the inference across instances. Frozen layers are built the same way and for the same reason.
    """

    def build(session) -> LearningPolicy:
        policy = LearningPolicy(session, layer, decider_for(session), rollout, session.instance,
                                frozen=frozen_for() if frozen_for is not None else None)
        if intruders is not None:
            intruder = intruders(session)
            if intruder is not None:
                intruder.organisation = policy.organisation
                policy.outside.append(intruder)
        return policy

    return build
