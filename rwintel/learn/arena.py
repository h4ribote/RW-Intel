"""A board on which fights are built one after another, so the tactical layer can be trained without playing matches.

The tactical layer's sample budget is not the match budget. An errand lasts ten to sixty seconds and a match lasts fifteen minutes, so training the tactical layer inside matches would spend the overwhelming majority of the wall clock simulating economies, build orders and marches that its decision has no bearing on. Everything needed to avoid that is already there: the engine's own spawn command creates units through the ordinary command route, the room settings can start a match with nothing on the board, and the sandbox flag makes every player's units answerable to this one process. Put together, that is an arena — one long episode in which engagements are constructed, fought, swept away and constructed again.

Both sides are driven from here, which is the point of the sandbox flag and is what makes the opponent something other than the built-in AI. The opposing side runs the same tactical layer over the same board read from the other side, so that what a learnt layer is measured against is the script layer doing exactly its job, and so that self-play needs no second process and no network.

There is no instruction that removes a unit, because the game has none: the only way to unmake a unit is to kill it. So an engagement is not cleared, it is finished — the survivors of both sides are set on each other until there are none — and the next engagement is built somewhere else on the map. That is slower than deleting them would be and it is the only method that keeps every change on the command route, which is the property the whole approach depends on.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..wire import (
    Action,
    Contract,
    Observation,
    Stance,
    Status,
    SquadAssignment,
    Task,
    encode_action,
)
from ..control.policy.catalogue import Catalogue
from ..control.policy.contracts import Doctrine, SquadRecord, TaskContract
from ..control.policy.tactics import Tactics
from ..control.policy.view import build as build_view

log = logging.getLogger(__name__)

#: Squad slots the two sides of an engagement occupy. Fixed, because nothing else is competing for the eight: the arena is the whole policy for this episode.
OURS = 0
THEIRS = 1

#: How long a fight is allowed before it is called and swept, in game milliseconds. The design's stated tactical horizon is ten to sixty seconds; this is the top of it, and most fights end well inside it.
FIGHT_MS = 60000

#: How long the survivors are given to finish each other off before the next engagement is built anyway. A remainder is tolerable — the next site is chosen away from it — but an unbounded wait is not.
SWEEP_MS = 20000

#: How long a fight may go without a casualty on either side before it is called.
#:
#: Two forces that have stopped hurting each other are not about to start. Measured: an engagement that ends with somebody destroyed takes twenty to thirty seconds, and one that ends on the clock spends its whole minute with both sides nearly intact, so waiting the full minute for those buys nothing and costs the arena a third of its time. Long enough that a squad manoeuvring for position is not mistaken for one that has given up.
STALL_MS = 12000

#: How long to wait for spawned units to appear before giving up on an engagement and trying again. Spawning goes through the command queue, so it takes a step or two rather than being instantaneous.
SPAWN_WAIT_MS = 12000

#: World units between the two sides when they are put down.
#:
#: Measured rather than chosen. At seven hundred the two sides converged to a median of a hundred and sixty and stopped there, which is outside a tank's reach of a hundred and thirty, and three quarters of the engagements then stood off until the clock ran out with almost nobody hurt: the engine's attack-move halts a unit when it acquires something, and acquisition happens at sight range while shooting needs weapon range, so two forces walking at each other come to rest in the gap between the two and stay there. Starting inside that gap is what makes the fight begin. It is still outside the reach of the shorter-ranged types, so which side shoots first is still something the layer's choice can affect.
SEPARATION = 250.0

#: Credits each side is built out of, drawn uniformly. Small fights and large ones teach different things and the layer has to answer both.
FORCE_VALUE = (1200.0, 5000.0)

#: How lopsided a fight may be, as the weaker side's share of the stronger. A layer that only ever saw even fights would never learn that some fights are to be broken off, which is one of the five departures.
IMBALANCE = (0.5, 1.0)

#: The most units either side is built from, so that one engagement cannot fill the board.
MAX_UNITS = 14

#: The fewest units a side is built from, which is what bounds how expensive a type may be for the budget it is drawn against. A fight is between formations, and one machine against a formation is a different problem from the one the five departures are about.
MINIMUM_FORCE = 3

#: How much of a terminal reward the outcome of a fight is worth, for a fight that ended without the contract itself reaching one of its own conclusions. One means that destroying the other side without a scratch is paid exactly what taking the contracted ground is paid, which is the largest this can be set to without teaching a layer to prefer a massacre to the errand it was given.
#:
#: Something has to be paid here, and measurement is the reason. Most fights end neither by one side being destroyed nor on the clock: they end with two forces that have stopped hurting each other, and under a scheme that paid only the discrete conclusions those fights were worth precisely nothing to either side. Nothing is the best score available in a fight that can only go badly, so a layer paid that way is being taught to stand off and wait, which is the opposite of what the arena exists to teach.
TERMINAL_OUTCOME_WEIGHT = 1.0

#: What multiple of a squad's present worth its contract will let it lose, drawn per side per engagement.
#:
#: Drawn rather than fixed because the allowance is one of the features the layer reads and one of the three things that make a mission be reported as losing. Pinned at the whole worth of the squad, being reported as losing means being all but destroyed, so the report never arrives in time to be acted on and the feature never moves; a layer trained that way has never seen the board on which the decision to break off is the right one, and meets it for the first time in a match, where the operational layer hands down allowances far tighter than a squad's whole worth.
BUDGET_SHARE = (0.3, 1.2)

#: How much of a fight's score is explained by which side the draw made stronger, per unit of the strength share.
#:
#: Fitted once, over a thousand fights of one layer against another, and then left alone. Refitting it as a run goes on would make the reward move under the policy for reasons that have nothing to do with the policy; a fixed multiple is unbiased whatever its value, because the draw happens before either side has decided anything, and only how much variance it removes depends on getting it near right. At this value it takes a third of the variance out.
STRENGTH_SLOPE = 2.2


def _lost(started: float, left: float) -> float:
    """The share of a side's worth that was destroyed, between nought and one.

    Bounded at both ends and defined as nothing when there was nothing to lose, so that a side built from no units, or one that somehow ends worth more than it began, still yields a number a reward can be paid from.
    """
    if started <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (started - left) / started))


@dataclass
class Engagement:
    """One fight, from the moment it is spawned to the moment the ground is clear."""

    index: int
    site: Tuple[float, float]
    #: What each side was worth when the two squads were formed, which is the worth of what actually arrived rather than of what was ordered. The two differ: placement is per unit and the engine refuses ground it will not build on, so part of an order can be stillborn while the rest of it fights. Scored against the order, a fight in which four of a dozen tanks never appeared would pay a loss nobody suffered.
    our_value: float = 0.0
    their_value: float = 0.0
    #: What the spawn order cost, kept beside what arrived so that a run producing thin fights can be told from one producing small ones.
    our_ordered: float = 0.0
    their_ordered: float = 0.0
    #: How many units were ordered for each side, which is what says whether an order has finished arriving. Spawning goes through the command queue a unit at a time, so a side can be half there while the other is whole, and forming on the first arrival puts the late half of an order on the board outside the squad that is being scored.
    our_count: int = 0
    their_count: int = 0
    ours_left: int = 0
    theirs_left: int = 0
    #: What each side was still worth when the fight was called, which is what turns a fight into a score rather than a tally of who was left standing.
    our_left_value: float = 0.0
    their_left_value: float = 0.0
    seconds: float = 0.0
    #: True when the fight was called because neither side had hurt the other for a while, rather than because it ended or ran out of time.
    stalled: bool = False
    #: How close the two sides ever came to each other, in world units. A fight in which this never falls below the weapons' reach was not a fight, and telling that case from a fight that was genuinely even is the difference between an arena that produces engagements and one that produces marches.
    closest: float = 1e9

    @property
    def outcome(self) -> float:
        """How the fight went for our side, from minus one to plus one: the share of the enemy's worth destroyed, less the share of ours lost, less what being dealt the stronger side is worth on its own.

        Antisymmetric, which is the property the whole measurement rests on. The other side's figure is exactly this one negated — the last term included, since one side's share of the total strength is one less the other's — so a run of self-play must average to nought and any departure from nought is a left-right asymmetry in the arena rather than a policy that has learnt something, while against the handwritten layer an average above nought is the same statement as having beaten it. What is learnt from and what is reported are then one quantity.

        Written in shares rather than in credits because the two sides are built to a deliberately uneven draw. A difference of worth would pay for having been dealt the stronger side, and a layer can improve that score without ever fighting differently.

        Shares alone do not finish the job, which is what the last term is for. The stronger side loses a smaller fraction of itself as well as fewer credits, so the score still rises with the draw: measured over a thousand fights it correlated with the share of the total strength dealt to this side at nearly six tenths, and the draw is made before either layer has decided anything. Subtracting a fixed multiple of that share removes a third of the variance without moving the average, because the draw is independent of how either side plays. What is left is how well a side did for the hand it was dealt, which is what both the reward and the comparison are trying to be.
        """
        strength = self.our_value + self.their_value
        share = self.our_value / strength if strength > 0 else 0.5
        return (_lost(self.their_value, self.their_left_value)
                - _lost(self.our_value, self.our_left_value)
                - STRENGTH_SLOPE * (share - 0.5))

    def as_dict(self) -> dict:
        return {"index": self.index, "our_value": round(self.our_value), "their_value": round(self.their_value),
                "our_ordered": round(self.our_ordered), "their_ordered": round(self.their_ordered),
                "ours_left": self.ours_left, "theirs_left": self.theirs_left,
                "seconds": round(self.seconds, 1), "closest": round(self.closest),
                "stalled": self.stalled, "outcome": round(self.outcome, 4)}


@dataclass
class Statistics:
    """What an arena episode did, in the shape the episode record expects so that a run of engagements is journalled like any other."""

    engagements: int = 0
    spawned: int = 0
    #: Engagements that were built but where the units never appeared, so no fight took place. Counted separately because it is wasted time rather than a result, and because it is the first thing to look at when the arena is producing less than it should.
    stillborn: int = 0
    won: int = 0
    lost: int = 0
    drawn: int = 0
    #: The three ways a fight is drawn, kept apart because they say different things about the arena. A stalled fight is two forces that stopped hurting each other and was called early; an expired one ran the whole minute out with both sides still standing; a mutual one is both sides destroyed within the same period, which is a fight fought to the end rather than one that never happened. A run made almost entirely of the first two is an arena producing stand-offs rather than engagements, and the drawn count alone cannot show that.
    stalled: int = 0
    expired: int = 0
    mutual: int = 0
    tactical: int = 0
    decisions: int = 0
    #: Every fight's outcome, kept whole so that the spread can be taken over the episode. Only the mean, the spread and the count go into the episode record: the list is as long as the run and says nothing per fight that the history does not already carry.
    outcomes: List[float] = field(default_factory=list)
    #: How many errands were closed for each reason, summed over both sides. Present so that a run can be asked directly whether its terminals fired, which is otherwise only inferable by reading the code and guessing.
    terminals: Dict[str, int] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)

    @property
    def fought(self) -> int:
        return self.won + self.lost + self.drawn

    @property
    def outcome_mean(self) -> float:
        return sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    @property
    def outcome_sd(self) -> float:
        """How widely the outcomes were spread, which is what says how many fights an assertion about the mean would need. Nought for a single fight, which has no spread rather than an unknown one."""
        if len(self.outcomes) < 2:
            return 0.0
        mean = self.outcome_mean
        return (sum((value - mean) ** 2 for value in self.outcomes) / len(self.outcomes)) ** 0.5

    def as_dict(self) -> dict:
        return {"engagements": self.engagements, "spawned": self.spawned,
                "stillborn": self.stillborn, "fought": self.fought, "won": self.won,
                "lost": self.lost, "drawn": self.drawn, "stalled": self.stalled,
                "expired": self.expired, "mutual": self.mutual, "tactical": self.tactical,
                "decisions": self.decisions, "outcome_mean": round(self.outcome_mean, 4),
                "outcome_sd": round(self.outcome_sd, 4), "terminals": dict(self.terminals),
                "history": self.history[-32:]}


class Arena:
    """The policy an arena episode runs under. Builds engagements, fights both sides of them, and pays the layer being trained."""

    def __init__(self, session, tactics: Optional[Callable] = None,
                 opponent: Optional[Callable] = None, seed: int = 0,
                 enemy_slot: Optional[int] = None,
                 outcome_weight: float = TERMINAL_OUTCOME_WEIGHT) -> None:
        self.session = session
        self.outcome_weight = outcome_weight
        self.catalogue = Catalogue(session.types, session.assets)
        self.random = random.Random(seed)
        # The layers are built here rather than handed in already made, because both sides have to read the same type catalogue as the arena that spawns their units: a layer classifying a unit from a different table would sort the same tank into a different role.
        self.tactics = tactics(session, self.catalogue) if tactics else Tactics(session, self.catalogue)
        self.opponent = opponent(session, self.catalogue) if opponent else Tactics(session, self.catalogue)
        self.statistics = Statistics()
        self.enemy_slot = enemy_slot

        self.squads: Dict[int, SquadRecord] = {}
        self.phase = "opening"
        self.engagement: Optional[Engagement] = None
        self.until_ms = 0
        self.known: set = set()
        self.sites: List[Tuple[float, float]] = []
        self.last_regions: List = []
        self._sandbox_sent = False
        #: How many units were standing in the current fight when its count last changed, and when that was, which is how a fight that has stopped being one is recognised.
        self._alive = 0
        self._changed_ms = 0

    # ---- the one entry point ----------------------------------------------------------

    def decide(self, observation: Observation) -> Optional[bytes]:
        if not self._sandbox_sent:
            # Sandbox first and on its own, because until it is set the units of the other side are not this process's to command and every order to them would be dropped.
            self.session.scenario([], sandbox=True)
            self._sandbox_sent = True
            self.sites = self._sites()
            self.known = {unit.id for unit in observation.unit_states}

        view = build_view(observation, self.catalogue, None, self.last_regions)
        self.last_regions = view.regions
        action = Action()
        now = observation.game_time_ms

        self._fold(observation)
        if self.phase == "opening" or self.phase == "clear":
            self._begin(observation, action, now)
        elif self.phase == "spawning":
            self._form(observation, action, now)
        elif self.phase == "fighting":
            self._fight(observation, view, action, now)
        elif self.phase == "sweeping":
            self._sweep(observation, action, now)

        if not (action.squads or action.contracts or action.deviations or action.production):
            return None
        return encode_action(action)

    # ---- building a fight --------------------------------------------------------------

    def _begin(self, observation: Observation, action: Action, now: int) -> None:
        site = self._site(observation)
        if site is None:
            return
        budget = self.random.uniform(*FORCE_VALUE)
        weaker = self.random.uniform(*IMBALANCE)
        ours_first = self.random.random() < 0.5
        our_budget = budget if ours_first else budget * weaker
        their_budget = budget * weaker if ours_first else budget

        angle = self.random.uniform(0, 2 * math.pi)
        offset = (math.cos(angle) * SEPARATION / 2, math.sin(angle) * SEPARATION / 2)
        our_place = (site[0] - offset[0], site[1] - offset[1])
        their_place = (site[0] + offset[0], site[1] + offset[1])

        our_force = self._force(our_budget)
        their_force = self._force(their_budget)
        if not our_force or not their_force:
            return

        spawns: List[float] = []
        spawns.extend(self._rows(our_force, self._our_slot(observation), our_place))
        spawns.extend(self._rows(their_force, self._their_slot(observation), their_place))
        self.session.scenario(spawns)

        self.known = {unit.id for unit in observation.unit_states}
        # What the sides are worth is left until they are formed, because what is ordered here and what appears there are not always the same units.
        self.engagement = Engagement(index=self.statistics.engagements, site=site,
                                     our_ordered=sum(kind.price for kind in our_force),
                                     their_ordered=sum(kind.price for kind in their_force),
                                     our_count=len(our_force), their_count=len(their_force))
        self.statistics.engagements += 1
        self.statistics.spawned += len(our_force) + len(their_force)
        self.phase = "spawning"
        self.until_ms = now + SPAWN_WAIT_MS
        log.debug("engagement %d at %.0f,%.0f: %.0f against %.0f credits ordered",
                  self.engagement.index, site[0], site[1],
                  self.engagement.our_ordered, self.engagement.their_ordered)

    def _form(self, observation: Observation, action: Action, now: int) -> None:
        """Takes the units that have appeared since the spawn was ordered and makes two squads of them.

        Waits for the whole of both orders rather than for the first unit of each. An order arrives over several periods, and one side's is submitted before the other's, so forming as soon as both have somebody systematically leaves more of the second side outside its squad than of the first. Those units stand on the board, join in the fighting, and are not counted in what the squad was worth or in what is left of it, which shows up as a score that favours one side of the board for no reason to do with either policy. If the wait runs out, whatever arrived is what fights, and that is honest because both figures the score uses are then taken from the same units.
        """
        fresh = [unit for unit in observation.unit_states if unit.id not in self.known]
        ours = [unit.id for unit in fresh if not unit.hostile]
        theirs = [unit.id for unit in fresh if unit.hostile]
        engagement = self.engagement
        whole = (engagement is None
                 or (len(ours) >= engagement.our_count and len(theirs) >= engagement.their_count))
        if not ours or not theirs or (not whole and now < self.until_ms):
            if now >= self.until_ms:
                log.info("engagement %d at %.0f,%.0f never appeared, trying another",
                         self.engagement.index if self.engagement else -1,
                         self.engagement.site[0] if self.engagement else 0.0,
                         self.engagement.site[1] if self.engagement else 0.0)
                self.statistics.stillborn += 1
                self._blame_site()
                self.phase = "clear"
            return

        self.squads = {
            OURS: self._record(OURS, ours, observation),
            THEIRS: self._record(THEIRS, theirs, observation),
        }
        if self.engagement is not None:
            # The fight is between what is standing here, so this is where the two figures the score divides by are taken. They are on the same footing as the worth left at the end, which the game reports as the price of a squad's surviving members.
            self.engagement.our_value = self.squads[OURS].value
            self.engagement.their_value = self.squads[THEIRS].value
        action.squads.append(SquadAssignment(squad=OURS, units=ours))
        action.squads.append(SquadAssignment(squad=THEIRS, units=theirs,
                                             owner=self._their_slot(observation)))

        deadline = now + FIGHT_MS
        for squad_id, other in ((OURS, THEIRS), (THEIRS, OURS)):
            squad = self.squads[squad_id]
            target = self._region_of(self.squads[other])
            # Drawn inside the loop so that the two sides get separate allowances, as they would from an operational layer pricing two missions against what each is for.
            budget = max(200.0, squad.value * self.random.uniform(*BUDGET_SHARE))
            contract = TaskContract(squad=squad_id, task=Task.ATTACK, target_region=target,
                                    stance=Stance.AGGRESSIVE, cost_budget=budget,
                                    deadline_ms=deadline, issued_at_ms=now)
            squad.contract = contract
            action.contracts.append(Contract(
                squad=squad_id, task=contract.task, stance=contract.stance,
                target_region=contract.target_region, cost_budget=contract.cost_budget,
                deadline_ms=contract.deadline_ms, issued_at_ms=contract.issued_at_ms,
                override=True))

        self.phase = "fighting"
        self.until_ms = deadline
        self._alive = len(ours) + len(theirs)
        self._changed_ms = now
        if self.engagement is not None:
            self.engagement.seconds = 0.0

    # ---- fighting it -------------------------------------------------------------------

    def _fight(self, observation: Observation, view, action: Action, now: int) -> None:
        ours = self.squads.get(OURS)
        theirs = self.squads.get(THEIRS)
        if ours is None or theirs is None:
            self.phase = "clear"
            return

        if ours.members and theirs.members and self.engagement is not None:
            self.engagement.closest = min(self.engagement.closest,
                                          math.hypot(ours.x - theirs.x, ours.y - theirs.y))

        deviations, _ = self.tactics.decide(view, [ours], now)
        action.deviations.extend(deviations)
        their_view = build_view(observation, self.catalogue, None, self.last_regions, invert=True)
        their_deviations, _ = self.opponent.decide(their_view, [theirs], now)
        action.deviations.extend(their_deviations)
        self.statistics.tactical += 1
        self.statistics.decisions += len(deviations)

        if not ours.members or not theirs.members or now >= self.until_ms:
            self._call(ours, theirs, now)
            return
        alive = len(ours.members) + len(theirs.members)
        if alive != self._alive:
            self._alive, self._changed_ms = alive, now
        elif now - self._changed_ms >= STALL_MS:
            self._call(ours, theirs, now, stalled=True)

    def _call(self, ours: SquadRecord, theirs: SquadRecord, now: int, stalled: bool = False) -> None:
        engagement = self.engagement
        outcome = 0.0
        if engagement is not None:
            engagement.stalled = stalled
            engagement.ours_left = len(ours.members)
            engagement.theirs_left = len(theirs.members)
            # A side with nothing left is worth nothing, said here rather than taken from the squad block: the game stops reporting a squad that no longer exists, so the last figure it sent would otherwise stand as the worth of survivors there are none of.
            engagement.our_left_value = ours.value if ours.members else 0.0
            engagement.their_left_value = theirs.value if theirs.members else 0.0
            engagement.seconds = (now - (ours.contract.issued_at_ms if ours.contract else now)) / 1000.0
            outcome = engagement.outcome
            self.statistics.outcomes.append(outcome)
            self.statistics.history.append(engagement.as_dict())
        if ours.members and not theirs.members:
            self.statistics.won += 1
        elif theirs.members and not ours.members:
            self.statistics.lost += 1
        else:
            self.statistics.drawn += 1
            if not ours.members and not theirs.members:
                # The last of both sides went within the same period. Asked first, because having nobody left says more about how a fight ended than the clock does, and counted apart from the fights the clock ended: a run of engagements fought to the last unit would otherwise read as a run of engagements in which nothing happened.
                self.statistics.mutual += 1
            elif stalled:
                self.statistics.stalled += 1
            else:
                self.statistics.expired += 1

        # The fight is over for both sides at the same instant, so both layers are told before the board is swept. Each is handed its own side of the outcome, which is the same number with the sign the other way up.
        self._end(self.tactics, ours, self.outcome_weight * outcome)
        self._end(self.opponent, theirs, self.outcome_weight * -outcome)
        self._tally()
        self.phase = "sweeping"
        self.until_ms = now + SWEEP_MS

    @staticmethod
    def _end(layer, squad: SquadRecord, terminal: float) -> None:
        """Tells one side's layer that its fight has ended, so that the decision it is still owed payment for is paid now instead of being carried into the next fight built on the same squad number.

        A layer that keeps no trajectories has nothing to end and says so by not offering the method, which is the ordinary case for the handwritten layer on either side of an arena run.
        """
        finish = getattr(layer, "finish", None)
        if finish is not None:
            finish(squad, terminal, "called")

    def _tally(self) -> None:
        """Collects how the two layers have been closing their errands.

        Recomputed from the layers' running totals rather than accumulated here, and refreshed as each fight is called rather than only at the end of the episode, because the episode record is taken from these statistics before the layers are closed. A count filled in only on the way out would be written to the journal empty, and a measurement that is absent from the record it exists for might as well not have been taken.
        """
        terminals: Dict[str, int] = {}
        for layer in (self.tactics, self.opponent):
            for reason, count in getattr(layer, "terminals", {}).items():
                terminals[reason] = terminals.get(reason, 0) + int(count)
        self.statistics.terminals = terminals

    def _sweep(self, observation: Observation, action: Action, now: int) -> None:
        """Sets whatever is left of both sides on each other, because there is no command that removes a unit and a board that is never cleared fills up.

        There is only anything to do here while both sides still have somebody. A fight that ended by one side being destroyed has nothing left to set against anything, and waiting out the sweep in that case is the commonest thing the arena did with its time: the winner stands about for twenty seconds while the next engagement, which is built somewhere else on the map regardless, waits for a clock that is measuring nothing.
        """
        ours = self.squads.get(OURS)
        theirs = self.squads.get(THEIRS)
        contested = (ours is not None and theirs is not None and ours.members and theirs.members)
        if not contested or now >= self.until_ms:
            self.phase = "clear"
            self.squads = {}
            return
        for squad_id, other in ((OURS, THEIRS), (THEIRS, OURS)):
            squad = self.squads[squad_id]
            if squad.contract is None:
                continue
            target = self._region_of(self.squads[other])
            if target == squad.contract.target_region:
                continue
            squad.contract = replace(squad.contract, target_region=target, issued_at_ms=now)
            action.contracts.append(Contract(
                squad=squad_id, task=Task.ATTACK, stance=Stance.AGGRESSIVE,
                target_region=target, cost_budget=squad.contract.cost_budget,
                deadline_ms=now + SWEEP_MS, issued_at_ms=now, override=True))

    # ---- keeping the two squads in step with the board ---------------------------------

    def _fold(self, observation: Observation) -> None:
        alive = {unit.id for unit in observation.unit_states}
        by_id = {state.id: state for state in observation.squads}
        for squad_id, squad in self.squads.items():
            squad.members = [member for member in squad.members if member in alive]
            state = by_id.get(squad_id)
            if state is None:
                continue
            squad.value = state.value
            squad.formed_value = state.formed_value
            squad.x, squad.y = state.x, state.y
            squad.spread = state.spread
            squad.losses = state.losses
            squad.status = Status(state.status)

    def _record(self, squad_id: int, members: Sequence[int], observation: Observation) -> SquadRecord:
        by_id = {unit.id: unit for unit in observation.unit_states}
        doctrine = Doctrine.VANGUARD
        for member in members:
            unit = by_id.get(member)
            found = self.catalogue.doctrine_for(unit.type_index) if unit is not None else None
            if found is not None:
                doctrine = found
                break
        value = sum(self.catalogue.value(by_id[m].type_index) for m in members if m in by_id)
        centre_x = sum(by_id[m].x for m in members if m in by_id) / max(1, len(members))
        centre_y = sum(by_id[m].y for m in members if m in by_id) / max(1, len(members))
        return SquadRecord(id=squad_id, doctrine=doctrine, members=list(members), value=value,
                           formed_value=value, x=centre_x, y=centre_y)

    # ---- where and what ----------------------------------------------------------------

    def _sites(self) -> List[Tuple[float, float]]:
        """Places an engagement may be built on.

        Region centres and resource points both, because a region centre is the mean of the points that formed it and can therefore fall on water or on a cliff, where the engine refuses to place anything and the engagement is stillborn. A resource point is ground something can be built on by definition, so it is ground a unit can be put on. Measured before this: with region centres alone, one instance in a run of eight lost fourteen of its twenty-two engagements to placements that never appeared.
        """
        places = [(region.x, region.y) for region in getattr(self.session, "regions", ())]
        content = getattr(self.session, "map_content", None)
        if content is not None:
            places.extend(content.to_world(tile) for tile in content.resources)
        return places

    def _blame_site(self) -> None:
        """Takes the site of a stillborn engagement out of the pool. Whether ground will take a unit is not something this side can ask, so the only way to find out is to try, and the only thing worth doing with the answer is to remember it."""
        if self.engagement is None or len(self.sites) <= 2:
            return
        site = self.engagement.site
        self.sites = [place for place in self.sites if place != site]

    def _site(self, observation: Observation) -> Optional[Tuple[float, float]]:
        """Somewhere to build the next fight, as far as possible from whatever is still standing. Survivors of an earlier engagement that could not be swept must not wander into the next one, or the fight the layer is paid for is not the fight it was given."""
        if not self.sites:
            self.sites = self._sites()
        if not self.sites:
            return None
        standing = [(unit.x, unit.y) for unit in observation.unit_states]
        if not standing:
            return self.random.choice(self.sites)

        def clearance(site: Tuple[float, float]) -> float:
            return min(math.hypot(site[0] - x, site[1] - y) for x, y in standing)

        best = max(clearance(site) for site in self.sites)
        return self.random.choice([site for site in self.sites if clearance(site) >= best * 0.9])

    def _force(self, budget: float) -> List:
        """A random force worth about the budget, drawn from the types that can fight on the ground.

        Restricted to what moves on the ground and shoots at the ground on purpose. A fight between aircraft and units that cannot elevate is not a fight, and an engagement in which one side cannot be reached teaches the layer only that nothing it does matters.

        A type is only considered if the whole force could be built from it, which is what keeps an engagement from consisting of one enormous machine. The registry runs from a three hundred credit tank to experimental units costing hundreds of times that, and a rule that simply spent until the budget ran out would put a single one of the latter on the board against a dozen of the former and call it a fight.
        """
        pool = [kind for kind in self.catalogue.types
                if kind.mobile and kind.armed and kind.hits_land and kind.price > 0
                and kind.movement in ("LAND", "HOVER")
                and kind.price <= budget / MINIMUM_FORCE]
        if not pool:
            return []
        force: List = []
        spent = 0.0
        while len(force) < MAX_UNITS:
            kind = self.random.choice(pool)
            if spent + kind.price > budget:
                if force:
                    break
                continue
            force.append(kind)
            spent += kind.price
        return force

    def _rows(self, force: Sequence, slot: int, place: Tuple[float, float]) -> List[float]:
        """Spawn rows for one side: type, player slot, position, count. Units are scattered a little so that they do not all arrive on the same point and spend the first seconds pushing each other apart."""
        rows: List[float] = []
        for index, kind in enumerate(force):
            angle = 2 * math.pi * index / max(1, len(force))
            radius = 40.0 + 12.0 * index
            rows.extend([float(kind.index), float(slot),
                         place[0] + math.cos(angle) * radius,
                         place[1] + math.sin(angle) * radius, 1.0])
        return rows

    def _our_slot(self, observation: Observation) -> int:
        return observation.slot

    def _their_slot(self, observation: Observation) -> int:
        """Whose the opposing side of an engagement is.

        The game side settles this as it builds the room and reports it, because it depends on which slots the room filled and which of those the map had nowhere to put. What is wanted is a player that never had a base: one that owns nothing has no income, nothing to build with and nothing to think about, so the only thing that moves its units is this process.
        """
        if self.enemy_slot is not None:
            return self.enemy_slot
        reported = getattr(self.session, "sparring_slot", -1)
        if reported >= 0:
            return reported
        return 1 if observation.slot != 1 else 0

    def _region_of(self, squad: SquadRecord) -> int:
        """The region slot nearest a squad, which is how a position is named to a contract. An arena map has regions like any other, and the nearest one is a good enough name for 'over there'."""
        best = 0
        best_distance = float("inf")
        for region in self.last_regions:
            distance = math.hypot(region.x - squad.x, region.y - squad.y)
            if distance < best_distance:
                best, best_distance = region.id, distance
        return best

    def close(self) -> None:
        self._tally()
        for layer in (self.tactics, self.opponent):
            if hasattr(layer, "close"):
                layer.close()
