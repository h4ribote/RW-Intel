"""The command chain with one layer learnt and the rest frozen.

This is the arrangement the design insists on for every training run: exactly one layer is being changed, everything else is the script, and the interference of an intruder is present because the operational layer has to be robust to it and because evaluation is done with it too. A run that moved two layers at once could not attribute the difference it measured to either of them, and the sample budget does not allow the number of runs it would take to find out which.

Building it by substitution rather than by assembly is what keeps the two comparable. The learnt policy is a script policy with one attribute replaced, so everything the comparison holds constant is held constant by construction rather than by care.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from ..control.policy import ScriptPolicy
from .layers import LearntEconomy, LearntOperations, LearntTactics
from .rollout import Rollout

log = logging.getLogger(__name__)

TACTICAL = "tactics"
OPERATIONAL = "operations"
ECONOMIC = "economy"

#: Every layer that can be learnt, by the name runs and teacher files know it by.
LAYERS = (TACTICAL, OPERATIONAL, ECONOMIC)


class LearningPolicy(ScriptPolicy):
    """The chain with one layer taken from a decider. Everything else is the script it is measured against."""

    def __init__(self, session, layer: str, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, **layer_options) -> None:
        super().__init__(session)
        self.layer = layer
        #: How many of each outside commander's logged events have already been acted on.
        self._seen_events: Dict[int, int] = {}
        if layer == TACTICAL:
            self.tactics = LearntTactics(session, self.catalogue, decider, rollout, instance, **layer_options)
        elif layer == OPERATIONAL:
            self.operations = LearntOperations(session, self.catalogue, decider, rollout, instance, **layer_options)
        elif layer == ECONOMIC:
            self.economy = LearntEconomy(session, self.catalogue, decider, rollout, instance,
                                         options=self.options, **layer_options)
        else:
            raise ValueError(f"no layer named {layer!r}: expected one of {', '.join(LAYERS)}")

    def decide(self, observation):
        answer = super().decide(observation)
        if self.layer == OPERATIONAL:
            self._taint_standing()
        return answer

    def _taint_standing(self) -> None:
        """Marks the operational decision standing for each squad an outside commander has just touched.

        An operational decision is paid for the periods it stands, so an intervention spoils the one standing when it happened and no other: the decisions before it were paid for periods nobody interfered with, and the squad is decided about afresh once it is back in this layer's hands.
        """
        for index, commander in enumerate(self.outside):
            events = getattr(getattr(commander, "log", None), "events", None)
            if events is None:
                continue
            seen = self._seen_events.get(index, 0)
            for event in events[seen:]:
                for squad in (event.get("squad"), event.get("into")):
                    if squad is not None and squad >= 0:
                        self.operations.taint(squad)
            self._seen_events[index] = len(events)

    def plan(self, observation):
        planned = super().plan(observation)
        if self.layer == OPERATIONAL:
            # Carried in the episode's statistics so that a record says how often the learnt layer was actually asked, which is far less than once a period.
            self.statistics.decisions = self.operations.decisions
            self.statistics.achievement = round(self.operations.achieved / max(1, self.operations.periods), 4)
        elif self.layer == ECONOMIC:
            self.statistics.decisions = self.economy.decisions
        return planned

    def close(self, score: Optional[float] = None) -> None:
        """Ends the episode's open trajectories, paying the match score to the layer when the match ended with one, and marks every decision about a squad somebody interfered with, before the layer seals the episode's record.

        The marking happens here rather than where the interference happened because an intervention half way through an errand invalidates the decisions taken before it as well as after: the outcome the whole errand is paid on is no longer the outcome of what this layer chose. Doing it at the end is the only point at which the full list of interfered-with squads is known.
        """
        learnt = getattr(self, self.layer, None)
        rollout = getattr(learnt, "rollout", None)
        # The operational layer has been marked as interventions happened, decision by decision; marking every decision about a touched squad here would throw away the periods nobody interfered with.
        if rollout is not None and self.layer == TACTICAL:
            for commander in self.outside:
                touched = getattr(getattr(commander, "log", None), "touched", None)
                if touched:
                    # This instance's squads only: a squad number names a different squad on every instance sharing the buffer.
                    rollout.taint(touched, owner=learnt.instance)
        if hasattr(learnt, "close"):
            learnt.close(score)


def learning_arm(layer: str, decider_for: Callable[[object], object],
                 rollout: Rollout, intruders: Optional[Callable[[object], object]] = None,
                 **layer_options) -> Callable:
    """An arm that runs the chain with one layer learnt, for the evaluation runner and the training runner alike.

    The decider is built per session rather than shared, because it is the thing that knows which instance it is answering for; the network behind it is shared, which is the whole point of batching the inference across instances.
    """

    def build(session) -> LearningPolicy:
        policy = LearningPolicy(session, layer, decider_for(session), rollout, session.instance, **layer_options)
        if intruders is not None:
            intruder = intruders(session)
            if intruder is not None:
                intruder.organisation = policy.organisation
                policy.outside.append(intruder)
        return policy

    return build
