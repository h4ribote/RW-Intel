"""The script intruder: a commander that interferes on purpose.

The design requires robustness to interruption to be trained rather than asserted. A human at the intervention interface pulls units out of a squad half way through its mission, takes a squad away for a minute and gives it back somewhere else, or rewrites an errand for a reason the chain cannot see; a policy learnt against a world where none of that happens treats every contract as a promise and has no answer when one is broken. So the learning environment contains something that breaks them, and evaluation is run with it too, because a number measured without interruption is not the number the system will be operated at.

It is not a test fixture. Everything here goes out through the same interface a human uses, in the same contract form, and the game side cannot tell the two apart — which is the point: what makes the training realistic is precisely that the intruder is indistinguishable from the person it stands in for. What it does not model is judgement. Its choices are random within the design's stated rates, which is deliberately the hardest version of the problem: a human's interventions are at least usually sensible, so a chain that copes with unmotivated ones copes with motivated ones.

The rates below are per operational period for the whole side, not per squad. Read per squad they would produce an intervention every few seconds on a full board, which is not interference but a second commander.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..wire import BLOCK_REGIONS, Commander, Observation, Stance, Task
from .intervention import Interface, Intervention, Kind, Recorder
from .policy.contracts import DOCTRINES, SquadRecord

log = logging.getLogger(__name__)

#: Chance per operational period that units are pulled out of some squad.
DETACH_CHANCE = 0.02

#: The share of a squad that goes when they are, drawn uniformly between the two.
DETACH_SHARE = (0.10, 0.50)

#: Chance per operational period that a squad is taken over whole.
SEIZE_CHANCE = 0.005

#: Chance per operational period that a squad's contract is rewritten from outside.
REWRITE_CHANCE = 0.01

#: How long anything taken is held, in game milliseconds, drawn uniformly between the two.
HOLD_MS = (20000, 90000)

#: What a rewritten or intruder-issued contract is funded with, as a share of the squad's own worth. The intruder is not costing the errand, it is disturbing it, so the figure only has to be one the tactical layer will act on rather than immediately abandon.
BUDGET_SHARE = 0.5

#: The least a rewritten contract may be funded with, for a squad worth almost nothing.
MINIMUM_BUDGET = 200.0

#: How long an intruder's contract runs before the chain would call it expired.
DEADLINE_MS = 120000


@dataclass
class Holding:
    """Something the intruder has taken and will give back."""

    squad: int
    until_ms: int
    #: True when the squad is one the intruder raised for units it pulled out, rather than one it took whole.
    raised: bool = False


@dataclass
class Log:
    """What the intruder did over an episode. The design says the results of squads that were interfered with are to be kept out of the learning signal, and this is what says which those were."""

    events: List[dict] = field(default_factory=list)
    #: Squads touched at any point, which is the set a learning run excludes.
    touched: set = field(default_factory=set)

    def note(self, intervention: Intervention) -> None:
        self.events.append(intervention.as_dict())
        self.touched.add(intervention.squad)
        # A squad units were moved into was interfered with as surely as the one they came out of: it is fighting its contract with a composition the operational layer did not give it, so its result says nothing about the layer either.
        if intervention.kind is Kind.REASSIGN and intervention.into >= 0:
            self.touched.add(intervention.into)

    def as_dict(self) -> dict:
        return {"events": list(self.events), "touched": sorted(self.touched)}


class Intruder:
    """A commander that interferes at the design's stated rates. Plugged into a policy's list of outside commanders, exactly as a human's interface is."""

    def __init__(self, organisation=None, seed: int = 0, recorder: Optional[Recorder] = None,
                 instance: int = -1, name: str = "intruder") -> None:
        self.random = random.Random(seed)
        self.interface = Interface(organisation=organisation, recorder=recorder,
                                   name=name, instance=instance)
        self.log = Log()
        self.holdings: Dict[int, Holding] = {}
        self._regions: List[int] = []

    @property
    def organisation(self):
        return self.interface.organisation

    @organisation.setter
    def organisation(self, layer) -> None:
        self.interface.organisation = layer

    def intervene(self, action, view, squads: List[SquadRecord], observation: Observation):
        # Interference is decided on the operational period because that is the clock the design quotes its rates against, and because a contract is what it interferes with.
        if observation.blocks & BLOCK_REGIONS:
            self._regions = [region.id for region in observation.regions]
            self._release(observation.game_time_ms)
            self._draw(squads, observation)
        applied = self.interface.intervene(action, view, squads, observation)
        for intervention in applied:
            self.log.note(intervention)
            # Which slot detached units went into is settled when the request is drained, not when it is made, so the holding is opened here.
            if intervention.kind is Kind.REASSIGN and intervention.into in self.interface.own:
                self.holdings[intervention.into] = Holding(
                    squad=intervention.into,
                    until_ms=intervention.at_ms + self.random.randint(*HOLD_MS), raised=True)
        return applied

    # ---- what it decides to do --------------------------------------------------------

    def _draw(self, squads: List[SquadRecord], observation: Observation) -> None:
        free = [squad for squad in squads
                if squad.id not in self.holdings and squad.members and squad.machine]
        if not free:
            return
        if self.random.random() < DETACH_CHANCE:
            self._detach(self.random.choice(free), observation)
        if self.random.random() < SEIZE_CHANCE:
            candidates = [s for s in free if s.id not in self.holdings]
            if candidates:
                self._seize(self.random.choice(candidates), observation)
        if self.random.random() < REWRITE_CHANCE:
            # A squad somebody else is holding is not one to rewrite the errand of. The override that makes the row apply at all would apply it over the holder's own, and the two would then alternate on the same squad with nothing deciding between them.
            candidates = [s for s in squads if s.id not in self.holdings and s.members and s.machine]
            if candidates:
                self._rewrite(self.random.choice(candidates), observation)

    def _detach(self, squad: SquadRecord, observation: Observation) -> None:
        """Takes part of a squad away. The units go into a squad of the intruder's own rather than loose, because loose units are picked straight back up by the reinforcement rule and the squad would be whole again a period later, which is not an interruption but a delay."""
        share = self.random.uniform(*DETACH_SHARE)
        count = max(1, int(round(len(squad.members) * share)))
        if count >= len(squad.members):
            count = len(squad.members) - 1
        if count <= 0:
            return
        self.interface.reassign(squad.id, self.random.sample(list(squad.members), count))

    def _seize(self, squad: SquadRecord, observation: Observation) -> None:
        """Takes a squad whole, moves it, and hands it back later somewhere other than where it was found — which is the case the design names, because a squad returned in a place the chain did not put it is the one whose contract has silently stopped meaning anything."""
        self.interface.take(squad.id, int(Commander.OPERATIONS | Commander.TACTICS))
        self._order(squad, observation)
        self.holdings[squad.id] = Holding(squad=squad.id, until_ms=self._until(observation))

    def _rewrite(self, squad: SquadRecord, observation: Observation) -> None:
        """Replaces a squad's errand without taking the squad. The chain goes on fighting it and does not notice, which is the mildest of the three and the most common thing a person actually does."""
        self._order(squad, observation)

    def _order(self, squad: SquadRecord, observation: Observation) -> None:
        region = self._somewhere()
        if region is None:
            return
        tasks = DOCTRINES[squad.doctrine].tasks or (Task.ATTACK,)
        self.interface.write(
            squad.id, task=self.random.choice(list(tasks)), target_region=region,
            stance=Stance.AGGRESSIVE,
            cost_budget=max(MINIMUM_BUDGET, BUDGET_SHARE * squad.value),
            deadline_ms=observation.game_time_ms + DEADLINE_MS,
        )

    def _somewhere(self) -> Optional[int]:
        return self.random.choice(self._regions) if self._regions else None

    def _until(self, observation: Observation) -> int:
        return observation.game_time_ms + self.random.randint(*HOLD_MS)

    # ---- giving things back -----------------------------------------------------------

    def _release(self, now: int) -> None:
        for holding in list(self.holdings.values()):
            if now < holding.until_ms:
                continue
            self.interface.give_back(holding.squad)
            del self.holdings[holding.squad]


def interference(seed: int = 0, recorder: Optional[Recorder] = None):
    """A way of building one intruder per episode, which is what a server is handed rather than an intruder itself.

    An intruder is per episode because what it is holding is a fact about one match, and because its log is what the episode record carries: a single intruder across a run would report the whole run's interference against every episode in it and no learning run could tell which squads to leave out of which.

    The seed is derived from the run's own seed, the instance and the episode index, so the same run interferes the same way twice while two instances of it do not interfere identically — which matters because identical interference across a batch would be a systematic difference between the arms rather than the noise it is meant to be. It does not make the match reproduce: the game does not, whatever it is seeded with. What it makes reproducible is the intruder, so that an odd result can be read back against exactly the interference that produced it.
    """

    def make(session) -> Intruder:
        return Intruder(seed=seed + session.instance * 1000 + len(session.records),
                        recorder=recorder, instance=session.instance)

    return make
