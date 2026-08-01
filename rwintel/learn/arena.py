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
from ..control.policy.view import build as build_view, rehome

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
#: Two forces that have stopped hurting each other are not about to start. Measured on the arena as it then stood: an engagement that ends with somebody destroyed takes twenty to thirty seconds, and one that ends on the clock spent its whole minute with both sides nearly intact, so waiting the full minute for those bought nothing and cost the arena a third of its time. With the cut-off in place almost nothing reaches the clock — 15 fights of the 21212 recorded since — so what the cut-off now decides is when a fight that has already had its casualties is swept, not whether an empty minute is paid for. Long enough that a squad manoeuvring for position is not mistaken for one that has given up.
#:
#: What removing it costs and buys has since been measured on both settings at once, as two arms of one run drawing the same fights. It buys decisiveness: fights that end with a side destroyed go from 31 per cent to 49. It costs throughput, because the fights that would have been called run their whole minute — the same wall clock produced 1380 fights against 894 — and it widens the scatter of the score from 0.569 to 0.643, which is what a sample size is paid in. It does not move the baseline: the two arms differ by -0.022 with two standard errors of 0.031, where a run of the handwritten layer against itself has to average nought. That last was once believed otherwise and was the stated reason for keeping the cut-off; the belief came from a run whose episodes all drew the same fights over again, and it did not survive drawing them afresh.
STALL_MS = 12000

#: How long to wait for spawned units to appear before giving up on an engagement and trying again. Spawning goes through the command queue, so it takes a step or two rather than being instantaneous.
SPAWN_WAIT_MS = 12000

#: World units between the two sides when they are put down.
#:
#: Measured rather than chosen. At seven hundred the two sides converged to a median of a hundred and sixty and stopped there, which is outside a tank's reach of a hundred and thirty, and three quarters of the engagements then stood off until the clock ran out with almost nobody hurt: the engine's attack-move halts a unit when it acquires something, and acquisition happens at sight range while shooting needs weapon range, so two forces walking at each other come to rest in the gap between the two and stay there. Starting inside that gap is what makes the fight begin. It is still outside the reach of the shorter-ranged types, so which side shoots first is still something the layer's choice can affect.
SEPARATION = 250.0

#: Credits each side is built out of, drawn uniformly. Small fights and large ones teach different things and the layer has to answer both.
FORCE_VALUE = (1200.0, 5000.0)

#: How lopsided a fight may be, as the weaker side's share of the stronger. A layer that only ever saw even fights would never learn that some fights are to be broken off, which is one of the departures.
#:
#: Half rather than lower, and the reason has been measured twice, differently each time. The first reading was that lowering the floor to three tenths broke the board: the self-play average, which has to be nought, went from -0.017 to -0.073. That reading did not survive drawing fresh fights every episode, which is what those runs had not been doing; at fifty-odd distinct draws the interval on either figure was about a seventh, and the sign of the lopsided one flipped from one seed to the next.
#:
#: The second reading is what the setting now rests on, and it is about sharpness rather than fairness. Both floors are fair - the handwritten layer against itself comes back at +0.013 and +0.002 with two standard errors of about 0.03 - and the lower floor does make fights more decisive, from 31 per cent ending with a side destroyed to 37. What it does not do is make the measurement any sharper. Pinning a layer to one departure and taking the difference from the baseline on the very same fights, the loss it costs comes out at -0.0555 under the even floor and -0.0600 under the lopsided one, at two standard errors of 0.025 and 0.024, so the same claim costs the same number of fights either way. What the lopsided draw does move is the unpaired spread, from 0.57 to 0.63, which is paid for in sample size. There would be a reason to lower it - a layer that never meets a fight it ought to break off never learns to - but the arena's own arithmetic says the measurement gains nothing, so the floor stays where the design put it.
IMBALANCE = (0.5, 1.0)

#: This side's departure is decided and submitted before the other side's, every period, which is the arrangement the arena ran under while a left-right lean was being measured in the fighting itself.
OURS_FIRST = "ours"

#: The other side's first instead, which is the same arrangement with the sides exchanged. It is not a setting to run under; it is the measurement that decides whether the order is what the lean is made of, because a lean made of the order has to change sign when the order does and one made of anything else cannot.
THEIRS_FIRST = "theirs"

#: Which side leads changes from period to period, so that whatever a period's leader gains falls on both sides equally over a fight. The same argument the spawn orders are interleaved under, applied to the decisions.
ALTERNATING = "alternate"

DECISION_ORDERS = (OURS_FIRST, THEIRS_FIRST, ALTERNATING)

#: The most units either side is built from, so that one engagement cannot fill the board.
MAX_UNITS = 14

#: The fewest units a side is built from, which is what bounds how expensive a type may be for the budget it is drawn against. A fight is between formations, and one machine against a formation is a different problem from the one the departures are about.
MINIMUM_FORCE = 3

#: How long to let the opening board settle before the first engagement is built, in game milliseconds.
#:
#: A spawn-point player is given a headquarters and a builder at the start of an episode, and they are not all on the board in the first frame the arena reads: they arrive a step or two in. The snapshot of what was already standing, against which every later arrival is judged to be a freshly spawned unit of a fight, is taken before an engagement is built — so building the first one immediately takes that snapshot too early, misses the base, and then the squad former counts this side's own headquarters as a unit that has just arrived for the fight and sweeps it into the squad. The fight is then one this side scores an extra headquarters in and the baseless opposing side never can, which is a bias in the self-play baseline that has to be nought. Waiting until the set of standing units stops changing folds the base into the snapshot, after which no engagement mistakes it for part of a fight; this is the longest that wait may run before the first engagement is built regardless, for the case of a board that never settles or never fills. It is paid once an episode, because after the first engagement the base is standing and every later snapshot already holds it.
SETTLE_MS = 5000

#: How much of a terminal reward the outcome of a fight is worth, for a fight that ended without the contract itself reaching one of its own conclusions. One means that destroying the other side without a scratch is paid exactly what taking the contracted ground is paid, which is the largest this can be set to without teaching a layer to prefer a massacre to the errand it was given.
#:
#: Something has to be paid here, and measurement is the reason. Most fights end neither by one side being destroyed nor on the clock: they end with two forces that have stopped hurting each other, and under a scheme that paid only the discrete conclusions those fights were worth precisely nothing to either side. Nothing is the best score available in a fight that can only go badly, so a layer paid that way is being taught to stand off and wait, which is the opposite of what the arena exists to teach.
TERMINAL_OUTCOME_WEIGHT = 1.0

#: What multiple of a squad's present worth its contract will let it lose, drawn per side per engagement.
#:
#: Drawn rather than fixed because the allowance is one of the features the layer reads and one of the three things that make a mission be reported as losing. Pinned at the whole worth of the squad, being reported as losing means being all but destroyed, so the report never arrives in time to be acted on and the feature never moves; a layer trained that way has never seen the board on which the decision to break off is the right one, and meets it for the first time in a match, where the operational layer hands down allowances far tighter than a squad's whole worth.
BUDGET_SHARE = (0.3, 1.2)

#: How much of a fight's score is explained by which side the draw made stronger, subtracted so that being dealt the stronger side is not paid for on its own. One multiple per reading of a fight, because the two readings scatter differently and each is de-noised best by its own.
#:
#: The score rises with the strength share at a correlation near two thirds, and that share is settled at the draw, before either layer has decided anything. Subtracting a fixed multiple of it is therefore a control variate: policy-invariant, since a policy cannot move a quantity fixed before it acts, and antisymmetric, since one side's share of the total is one less the other's — so self-play still averages nought whatever the multiple is, and the self-check that governs everything here still governs it. What the subtraction buys is variance, and the interval on every claim the arena makes is paid in the spread of its fights.
#:
#: It was nought for a while, and the story is why the multiple is trusted now. Fitted at 2.2 on an earlier arena it removed a third of the variance and did not survive the self-check: the handwritten layer against itself came back at -0.070 over 901 fights where it has to be nought, because the formed shares were not symmetric. Both structural causes are since gone — Arena._interleave submits the two spawn orders a unit at a time, and SETTLE_MS folds this side's own base into the opening snapshot before any fight is built — and the formed shares are even: over the 3329 self-play fights the multiple is fitted on, the mean share is 0.502.
#:
#: So it has been earned back. Fitted by least squares on those fights it is 3.96 on the sparse reading and 3.68 on the health reading, stable across the five seeds they were drawn from: leaving any one seed out moves it by under 0.05. It removes 41 per cent of the variance on the sparse reading (spread 0.579 to 0.443) and 47 per cent on the health reading (0.507 to 0.371), reproduced on fresh self-play seeds the fit never saw — per-fight spread 0.48 and 0.41 corrected against 0.58 and 0.53 raw, over eleven hundred more fights.
#:
#: What that variance is on decides what the reduction buys, and it is worth being exact about. It is the per-fight scatter, so it halves the fights a self-play self-check or any single-arm score needs, and it lowers the variance of the terminal the tactical layer is paid. It does NOT tighten a policy-against-baseline duel, which is the comparison a run is actually judged by: that is paired on the same draw, so the share — and the term, which is a function of the share alone — is common to both arms and cancels in their difference. That cancellation is the whole reason pairing removes the draw already; the term removes the same draw from the places pairing does not reach, the unpaired self-play mean and the reward.
#:
#: The check it has to pass is structural rather than numeric: the term is antisymmetric, so it cannot put a left-right lean into a score that was even, and the formed shares are symmetric now (mean 0.502), so the asymmetry that sank the 2.2 fit — formed shares of 53 against 50 — is gone. What self-play still leans by, up to about four hundredths and different from one seed to the next, is the arena's own lean and not the term's; it is why a policy is measured against a baseline drawn under its own seed, which cancels it, rather than against nought. Over the five fit seeds the corrected mean pools to +0.009. The figures carry the measurement — an arena drawing its fights differently, at another imbalance floor or force range, would want them re-fit, since the multiple that flattens the draw's variance depends on how the draw is spread.
STRENGTH_SLOPE_KILLS = 3.96
STRENGTH_SLOPE_HEALTH = 3.68

#: Scoring a fight on what is left standing, which is what every figure this project has quoted was taken on.
#:
#: A unit counts for the whole of its price until the moment it dies and for nothing after, so damage short of a kill is invisible. That is the sparse reading of a fight and it is the one the ceiling was measured against.
BY_KILLS = "kills"

#: Scoring a fight on what is left standing weighted by how much of it is left, so that a unit at a tenth of its health counts for a tenth of its price.
#:
#: Two thirds of fights end with neither side destroyed, because a fight is called twelve seconds after its last casualty. Over the 21212 arena fights recorded so far — leaving out the arms pinned to one departure, since pinning moves that rate by thirty points either way — 13733 of them, 64.7 per cent with two standard errors of 0.7 points, were called that way rather than by a body count or by the clock.
#:
#: That is not a fight in which nothing happened, and the sparse reading does not score it as one. The call wants twelve quiet seconds, not an untouched pair of forces, so every casualty taken before the quiet counts in full: of those 13733 fights only 13 lost nobody at all and only 33 came out at exactly nought, and the median one has lost 32.8 per cent of this side's worth and destroyed 35.5 per cent of the other's. It was believed for a while that all of them scored nought, and that belief came from reading the cut-off as a fight nobody died in rather than as one that had gone quiet.
#:
#: What weighting by health buys is therefore sharpness rather than a signal where there was none: it counts the damage standing on the survivors, which is the one thing the sparse reading cannot see until it has killed something. Read with the strength-share multiple at nought, so that fights drawn before and after that multiple was fitted can be pooled, the per-fight spread over those 21212 is 0.585 sparse against 0.518 on health — a fifth of the variance, and so a fifth off the fights any claim about a mean costs. On a stall-called fight the two readings differ by 0.101 on average and disagree about which side did better in one fight in ten.
#:
#: Antisymmetry is untouched: the two sides' figures are the same subtraction with the terms exchanged, so a run of the handwritten layer against itself still has to average nought and the self-check that governs everything here still governs it.
BY_HEALTH = "health"

SCORES = (BY_KILLS, BY_HEALTH)


def _lost(started: float, left: float) -> float:
    """The share of a side's worth that was destroyed, between nought and one.

    Bounded at both ends and defined as nothing when there was nothing to lose, so that a side built from no units, or one that somehow ends worth more than it began, still yields a number a reward can be paid from.
    """
    if started <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (started - left) / started))


def _tally(force: Sequence) -> Dict[int, int]:
    """How many of each type an order asks for, which is what a squad is later filled against."""
    counts: Dict[int, int] = {}
    for kind in force:
        counts[kind.index] = counts.get(kind.index, 0) + 1
    return counts


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _spread(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


@dataclass
class Engagement:
    """One fight, from the moment it is spawned to the moment the ground is clear."""

    index: int
    site: Tuple[float, float]
    #: Where each side was put down, which is the two points the site is the midpoint of. Kept on the fight rather than recomputed because they are what each side's view is anchored to: a constructed fight has no base, and the direction a squad has come from is the one direction its encoding reads that is not read off the fight itself. The two are exact reflections of each other about the site, so a squad and its mirror are measured from mirrored origins and read as one fight; anchoring both sides to one point would put the frame straight back.
    our_place: Tuple[float, float] = (0.0, 0.0)
    their_place: Tuple[float, float] = (0.0, 0.0)
    #: What each side was worth when the two squads were formed, which is the worth of what actually arrived rather than of what was ordered. The two differ: placement is per unit and the engine refuses ground it will not build on, so part of an order can be stillborn while the rest of it fights. Scored against the order, a fight in which four of a dozen tanks never appeared would pay a loss nobody suffered.
    our_value: float = 0.0
    their_value: float = 0.0
    #: What the spawn order cost, kept beside what arrived so that a run producing thin fights can be told from one producing small ones.
    our_ordered: float = 0.0
    their_ordered: float = 0.0
    #: How many units were ordered for each side, which is what says whether an order has finished arriving. Spawning goes through the command queue a unit at a time, so a squad is only formed once the count it was ordered at is standing; the two orders are interleaved so that neither runs ahead of the other, and forming before both are whole would leave the units still queued on the board outside the squad that is being scored.
    our_count: int = 0
    their_count: int = 0
    #: What was ordered, type by type, so that a squad can be made of what was commissioned for this fight and of nothing else.
    #:
    #: A count is not enough on its own. Two things arrive on the board that nobody commissioned: the headquarters and the builder a spawn-point player begins an episode with, which appear a step or two apart so that the wait for the opening board to settle can pass between them, and the units of an engagement that was abandoned as stillborn, whose spawn commands are never withdrawn and which turn up while the next fight is being formed. Both were measured: the builder joined this side's squad in about seven of every ten first fights, worth five hundred credits it never had to lose, and those fights scored a tenth of a point above every other fight in the run. Neither can join a squad that is filled against the order that was actually placed.
    our_wanted: Dict[int, int] = field(default_factory=dict)
    their_wanted: Dict[int, int] = field(default_factory=dict)
    ours_left: int = 0
    theirs_left: int = 0
    #: What each side was still worth when the fight was called, which is what turns a fight into a score rather than a tally of who was left standing.
    our_left_value: float = 0.0
    their_left_value: float = 0.0
    #: The same two figures with every survivor counted at the share of its health it still has, which is what makes a fight that nobody died in worth something. Taken beside the other pair rather than instead of it, so that one run reports the score both ways and the sparse reading every earlier figure was quoted on stays readable.
    our_left_health: float = 0.0
    their_left_health: float = 0.0
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

        The last term takes out what the draw was worth, which is a control variate rather than part of the fight: the strength share is fixed before either layer acts, so subtracting a multiple of it cannot change which policy is best and cannot break the antisymmetry above, and what it removes is variance — about two fifths of it, since the score and the share correlate near two thirds. The multiple was nought while a self-check was outstanding and is now the fitted value (see STRENGTH_SLOPE_KILLS); the sparse and health readings carry their own, because they scatter differently.
        """
        return self._scored(self.our_left_value, self.their_left_value, STRENGTH_SLOPE_KILLS)

    @property
    def outcome_health(self) -> float:
        """The same score with every survivor counted at the share of its health it still holds.

        The sparse reading above cannot tell a fight in which both sides walked away untouched from one in which both were shot to a tenth of themselves and neither quite died. Both of those are rare — of the 21212 fights recorded so far, 13 ended with nobody destroyed at all — because a fight is called twelve seconds after its last casualty rather than for want of one, and two thirds of fights are called that way with their dead already taken. So what this reading adds is not a score where the other had none, but the part of the score the other cannot reach: the damage left standing on the survivors, worth 0.101 of score on the average stall-called fight and enough to turn its sign once in ten.

        Not identical to the sparse reading on a fight that ended in a body count, which was once claimed here: the side that was destroyed is worth nothing under either, but the side that won walks away damaged and only this reading counts that, so the two differ on 89.2 per cent of the one-sided annihilations recorded, by 0.135 averaged over all of them. The two agree exactly only when the surviving side is untouched. Antisymmetric for the same reason the other is, being the same subtraction with the sides exchanged.
        """
        return self._scored(self.our_left_health, self.their_left_health, STRENGTH_SLOPE_HEALTH)

    def scored(self, how: str) -> float:
        return self.outcome_health if how == BY_HEALTH else self.outcome

    def _scored(self, ours_left: float, theirs_left: float, slope: float) -> float:
        strength = self.our_value + self.their_value
        share = self.our_value / strength if strength > 0 else 0.5
        return (_lost(self.their_value, theirs_left)
                - _lost(self.our_value, ours_left)
                - slope * (share - 0.5))

    def as_dict(self) -> dict:
        return {"index": self.index, "our_value": round(self.our_value), "their_value": round(self.their_value),
                "our_ordered": round(self.our_ordered), "their_ordered": round(self.their_ordered),
                "ours_left": self.ours_left, "theirs_left": self.theirs_left,
                "our_left_value": round(self.our_left_value), "their_left_value": round(self.their_left_value),
                "our_left_health": round(self.our_left_health), "their_left_health": round(self.their_left_health),
                "seconds": round(self.seconds, 1), "closest": round(self.closest),
                "stalled": self.stalled, "outcome": round(self.outcome, 4),
                "outcome_health": round(self.outcome_health, 4)}


@dataclass
class Statistics:
    """What an arena episode did, in the shape the episode record expects so that a run of engagements is journalled like any other."""

    engagements: int = 0
    spawned: int = 0
    #: Engagements that were built but where the units never appeared, so no fight took place. Counted separately because it is wasted time rather than a result, and because it is the first thing to look at when the arena is producing less than it should. Measured over every episode record kept from the runs behind this design, it is two in 26420 engagements built, and both of those were the first engagement of an episode at one point on one map that would not take a unit — so what this counts now is ground rather than a board filled up by a weak layer, and a run that reports more than a handful of these is reporting something new.
    stillborn: int = 0
    won: int = 0
    lost: int = 0
    drawn: int = 0
    #: The three ways a fight is drawn, kept apart because they say different things about the arena. A stalled fight is two forces that stopped hurting each other and was called early, which is the ordinary ending here rather than an absent one: it takes twelve quiet seconds and not an untouched pair, and the median one has already destroyed about a third of each side. An expired one ran the whole minute out with both sides still standing, and a run with many of those is an arena producing stand-offs rather than engagements. A mutual one is both sides destroyed within the same period, which is a fight fought to the end rather than one that never happened. The drawn count alone shows none of the three, which is why they are counted apart.
    stalled: int = 0
    expired: int = 0
    mutual: int = 0
    tactical: int = 0
    decisions: int = 0
    #: Every fight's outcome, kept whole so that the spread can be taken over the episode. Only the mean, the spread and the count go into the episode record: the list is as long as the run and says nothing per fight that the history does not already carry.
    outcomes: List[float] = field(default_factory=list)
    #: The same fights scored on health rather than on bodies. Kept beside rather than instead, because every ceiling this project has quoted was measured on the sparse reading and a run that reported only the other could not be read against any of them.
    health_outcomes: List[float] = field(default_factory=list)
    #: How many errands were closed for each reason, summed over both sides. Present so that a run can be asked directly whether its terminals fired, which is otherwise only inferable by reading the code and guessing.
    #: How the fights of this episode were drawn: how far apart the two sides were put down, how long a fight may go without a casualty before it is called, the weaker side's smallest share of the stronger, and which reading of a fight was paid as the terminal. Journalled with the episode for the reason the constructed operations arena journals its own draw — none of them reaches the episode settings, and two runs drawn under different ones are two different instruments, so a comparison that pooled them would be reading the change of instrument as a difference between the arms. The separation is the sharpest of them: it decides how much of a fight the departures can still decide at all.
    separation: float = SEPARATION
    stall_ms: int = STALL_MS
    imbalance_floor: float = IMBALANCE[0]
    score: str = BY_HEALTH
    terminals: Dict[str, int] = field(default_factory=dict)
    #: Every fight of the episode, one row each, and all of them.
    #:
    #: It used to be the last thirty two, which is the number of fights a long episode has after the point where the board has stopped being even. Everything anybody wanted to ask of this list — whether a run drifts as its board fills, what the score looks like early against late — is a question about the fights that were dropped, and asking it of what was left gave an answer drawn from the wrong half. A row is about a hundred bytes and an episode is a handful of fights at the length the arena now runs at.
    history: List[dict] = field(default_factory=list)

    @property
    def fought(self) -> int:
        return self.won + self.lost + self.drawn

    @property
    def outcome_mean(self) -> float:
        return _mean(self.outcomes)

    @property
    def outcome_sd(self) -> float:
        """How widely the outcomes were spread, which is what says how many fights an assertion about the mean would need. Nought for a single fight, which has no spread rather than an unknown one."""
        return _spread(self.outcomes)

    @property
    def health_outcome_mean(self) -> float:
        return _mean(self.health_outcomes)

    @property
    def health_outcome_sd(self) -> float:
        return _spread(self.health_outcomes)

    def as_dict(self) -> dict:
        return {"engagements": self.engagements, "spawned": self.spawned,
                "stillborn": self.stillborn, "fought": self.fought, "won": self.won,
                "lost": self.lost, "drawn": self.drawn, "stalled": self.stalled,
                "expired": self.expired, "mutual": self.mutual, "tactical": self.tactical,
                "decisions": self.decisions, "outcome_mean": round(self.outcome_mean, 4),
                "outcome_sd": round(self.outcome_sd, 4),
                "health_outcome_mean": round(self.health_outcome_mean, 4),
                "health_outcome_sd": round(self.health_outcome_sd, 4),
                "separation": round(self.separation, 1), "stall_ms": self.stall_ms,
                "imbalance_floor": round(self.imbalance_floor, 4), "score": self.score,
                "terminals": dict(self.terminals), "history": self.history}


class Arena:
    """The policy an arena episode runs under. Builds engagements, fights both sides of them, and pays the layer being trained.

    One of these is built per episode, and everything about the fights it builds — where, how big, how uneven, made of what — comes out of the seed it is handed. That seed therefore has to advance from episode to episode, or every episode of an instance replays the same fights and a run reports its fight count as a sample size it does not have. It did, for a while, and the arithmetic that came out of it was wrong by a factor of thirty to fifty. The caller owns the derivation, because only the caller knows which episode this is and which arm of a comparison it belongs to.
    """

    def __init__(self, session, tactics: Optional[Callable] = None,
                 opponent: Optional[Callable] = None, seed: int = 0,
                 enemy_slot: Optional[int] = None,
                 outcome_weight: float = TERMINAL_OUTCOME_WEIGHT,
                 stall_ms: int = STALL_MS,
                 imbalance_floor: float = IMBALANCE[0],
                 decision_order: str = OURS_FIRST,
                 score: str = BY_HEALTH,
                 separation: float = SEPARATION) -> None:
        self.session = session
        self.outcome_weight = outcome_weight
        if score not in SCORES:
            raise ValueError(f"no score named {score!r}: expected one of {', '.join(SCORES)}")
        # Which reading of a fight is handed to the layers as their terminal. Both are computed and both are journalled whatever this says; what it settles is only which one is paid, because a layer can be paid on one reading and reported on the other but it cannot be paid on two. Health by default, and stated in one place only: a second default sitting here would be a figure a run could be paid on without anything having asked for it.
        self.score = score
        # How long a fight may go without a casualty before it is called. An argument rather than the constant because how decisive the arena's fights are is one of the things a run may want to ask about: a layer's choices can only be worth as much as the fights they are made in, and a fight that is called at the first quiet spell is one where declining to fight costs nothing.
        self.stall_ms = stall_ms
        # The weaker side's smallest share of the stronger. An argument rather than the constant because how lopsided the draw is decides how many fights end with a side destroyed and how widely the score scatters, and whether that trade is worth taking is a question only a run of both settings answers.
        self.imbalance_floor = imbalance_floor
        # How far apart the two sides are put down. An argument rather than the constant for the reason the stall is one, and it is the sharpest of the three: this distance is what decides how much of a fight the departures can still decide. Inside the reach of everything drawn, both sides are shooting from the first frame and what is left to choose is small; outside it, whether a squad closes, backs off or stands is the whole fight. The figure the constant states was measured against the engine's halting behaviour and is the one every recorded measurement was taken at, so a run that moves it is a different instrument and its numbers do not pair with theirs.
        self.separation = separation
        if decision_order not in DECISION_ORDERS:
            raise ValueError(f"no decision order named {decision_order!r}: expected one of {', '.join(DECISION_ORDERS)}")
        self.decision_order = decision_order
        self.catalogue = Catalogue(session.types, session.assets)
        self.random = random.Random(seed)
        #: A second stream, for the one draw that has to look at the board: where to put the next fight.
        #:
        #: The draw of a fight — the two budgets, which side is the stronger, the angle, and the two forces — has to be the same for two arms of one run, because that is the whole of what makes a duel paired: the two arms meet the same fights and what is left between them is the play. Drawn out of one stream with the placement, it was not. Choosing a site is a choice among the places that are clearest of whatever is still standing, and what is still standing is a fact about how the previous fights went, so the number of candidates differs between two arms and `random.choice` over a list of n consumes an amount of the underlying stream that depends on n. One arm therefore stepped off the other's stream part way through an episode, and every fight after that was a different fight: the pairing went on subtracting them as though they were the same draw.
        #:
        #: Split, the k-th fight of an episode is the same draw on both arms, since every other consumption is a fixed count per fight. What cannot be paired is the site itself — where a fight is clear of the survivors is a fact about the board and the board is what the policies made of it — so what remains between two arms is the same forces at the same odds on possibly different ground, rather than different forces.
        self.placement = random.Random(seed ^ 0x5EED51E5)
        # The layers are built here rather than handed in already made, because both sides have to read the same type catalogue as the arena that spawns their units: a layer classifying a unit from a different table would sort the same tank into a different role.
        self.tactics = tactics(session, self.catalogue) if tactics else Tactics(session, self.catalogue)
        self.opponent = opponent(session, self.catalogue) if opponent else Tactics(session, self.catalogue)
        # The draw is written into the statistics at construction rather than at scoring, so that an episode which produced no fight at all still says under what instrument it was run.
        self.statistics = Statistics(separation=separation, stall_ms=stall_ms,
                                     imbalance_floor=imbalance_floor, score=score)
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
            # Left empty, not seeded with this frame's units, so that the settle below has a change to detect: seeded with the current set, the first period would already read as unchanged and the settle would pass before the base had finished appearing, which is the very thing it is there to wait out.
            self.known = set()
            # The longest the opening settle may run before the first engagement is built regardless. See SETTLE_MS.
            self.until_ms = observation.game_time_ms + SETTLE_MS

        view = build_view(observation, self.catalogue, None, self.last_regions)
        self.last_regions = view.regions
        # Anchored where this side was put down, not where the map says home is. The builder's fallback is the region nearest this process's base, which is one point for the whole board and therefore the same point for both sides — under which a squad and its exact reflection read as two different fights.
        if self.engagement is not None:
            rehome(view, point=self.engagement.our_place)
        action = Action()
        now = observation.game_time_ms

        self._fold(observation)
        if self.phase == "opening":
            self._settle(observation, action, now)
        elif self.phase == "clear":
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

    def _settle(self, observation: Observation, action: Action, now: int) -> None:
        """Holds off the first engagement until the opening board has stopped changing, then builds it.

        The units a spawn-point player starts with arrive over the first steps rather than all at once, so the record of what was already standing has to be taken after they have, or this side's own base is counted as freshly spawned for the first fight and swept into its squad. Each period the record is refreshed to whatever is standing; the first engagement is built once that set has held from one period to the next, or once the settle has run its length, whichever comes first. See SETTLE_MS.
        """
        current = {unit.id for unit in observation.unit_states}
        settled = bool(current) and current == self.known
        self.known = current
        if settled or now >= self.until_ms:
            self._begin(observation, action, now)

    def _begin(self, observation: Observation, action: Action, now: int) -> None:
        site = self._site(observation)
        if site is None:
            return
        budget = self.random.uniform(*FORCE_VALUE)
        weaker = self.random.uniform(self.imbalance_floor, IMBALANCE[1])
        ours_first = self.random.random() < 0.5
        our_budget = budget if ours_first else budget * weaker
        their_budget = budget * weaker if ours_first else budget

        angle = self.random.uniform(0, 2 * math.pi)
        offset = (math.cos(angle) * self.separation / 2, math.sin(angle) * self.separation / 2)
        our_place = (site[0] - offset[0], site[1] - offset[1])
        their_place = (site[0] + offset[0], site[1] + offset[1])

        our_force = self._force(our_budget)
        their_force = self._force(their_budget)
        if not our_force or not their_force:
            return

        our_rows = self._rows(our_force, self._our_slot(observation), our_place)
        their_rows = self._rows(their_force, self._their_slot(observation), their_place)
        self.session.scenario(self._interleave(our_rows, their_rows))

        self.known = {unit.id for unit in observation.unit_states}
        # What the sides are worth is left until they are formed, because what is ordered here and what appears there are not always the same units.
        self.engagement = Engagement(index=self.statistics.engagements, site=site,
                                     our_place=our_place, their_place=their_place,
                                     our_ordered=sum(kind.price for kind in our_force),
                                     their_ordered=sum(kind.price for kind in their_force),
                                     our_count=len(our_force), their_count=len(their_force),
                                     our_wanted=_tally(our_force), their_wanted=_tally(their_force))
        self.statistics.engagements += 1
        self.statistics.spawned += len(our_force) + len(their_force)
        self.phase = "spawning"
        self.until_ms = now + SPAWN_WAIT_MS
        log.debug("engagement %d at %.0f,%.0f: %.0f against %.0f credits ordered",
                  self.engagement.index, site[0], site[1],
                  self.engagement.our_ordered, self.engagement.their_ordered)

    def _form(self, observation: Observation, action: Action, now: int) -> None:
        """Takes the units that have appeared since the spawn was ordered and makes two squads of them.

        Waits for the whole of both orders rather than for the first unit of each. The two orders are interleaved a unit at a time (_interleave), so they arrive at the same rate rather than one side's whole order before the other's; but at any single moment the two can still be a unit apart, and forming on the first arrival would leave whatever had not yet come outside its squad. Those units stand on the board, join in the fighting, and are counted neither in what the squad was worth nor in what is left of it, which would show up as a score that favours one side for no reason to do with either policy. If the wait runs out, whatever arrived is what fights, and that is honest because the interleave has cut both sides to the same depth and both figures the score uses are then taken from the same units.
        """
        fresh = [unit for unit in observation.unit_states if unit.id not in self.known]
        engagement = self.engagement
        ours = self._commissioned(fresh, False, engagement.our_wanted if engagement else None)
        theirs = self._commissioned(fresh, True, engagement.their_wanted if engagement else None)
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

        their_view = build_view(observation, self.catalogue, None, self.last_regions, invert=True)
        # The reflection of this side's anchor, so that the layer fighting from the other seat measures the same fight from the mirrored origin.
        if self.engagement is not None:
            rehome(their_view, point=self.engagement.their_place)
        sides = [(self.tactics, ours, view), (self.opponent, theirs, their_view)]
        if not self._ours_leads():
            sides.reverse()
        for layer, squad, board in sides:
            deviations, _ = layer.decide(board, [squad], now)
            action.deviations.extend(deviations)
            # Only this side's decisions are counted, because the count is what the run reports as its own output and the opposing layer's decisions are the environment rather than the product.
            if layer is self.tactics:
                self.statistics.decisions += len(deviations)
        self.statistics.tactical += 1

        if not ours.members or not theirs.members or now >= self.until_ms:
            self._call(ours, theirs, now, observation)
            return
        alive = len(ours.members) + len(theirs.members)
        if alive != self._alive:
            self._alive, self._changed_ms = alive, now
        elif now - self._changed_ms >= self.stall_ms:
            self._call(ours, theirs, now, observation, stalled=True)

    def _ours_leads(self) -> bool:
        """Whether this side's departure is decided and submitted ahead of the other side's this period.

        It ought not to matter. Both layers read the same frame, neither can see what the other chose, and the two sets of orders are carried on one action to one period of the simulation. But a left-right lean that the score cannot absorb has been measured in the fighting itself — buried in the noise on an even draw and out of it on a lopsided one — and the order the two sides are decided in is the only thing about a period that is not symmetric between them, so it is the first candidate and the one that can be settled by measurement rather than by reading: a lean made of the order reverses when the order does, and a lean made of anything else does not.

        The alternating setting is what a lean made of the order would be answered with rather than corrected for, on the same argument the two spawn orders are interleaved under: whatever a period's leader gains is then dealt to each side in half the periods of every fight instead of to one side in all of them. The parity is read off the count of periods the layers have decided in, which is already kept and is incremented once per period after the decisions are taken.
        """
        if self.decision_order == THEIRS_FIRST:
            return False
        if self.decision_order == ALTERNATING:
            return self.statistics.tactical % 2 == 0
        return True

    def _call(self, ours: SquadRecord, theirs: SquadRecord, now: int,
              observation: Observation, stalled: bool = False) -> None:
        engagement = self.engagement
        outcome = 0.0
        if engagement is not None:
            engagement.stalled = stalled
            engagement.ours_left = len(ours.members)
            engagement.theirs_left = len(theirs.members)
            # A side with nothing left is worth nothing, said here rather than taken from the squad block: the game stops reporting a squad that no longer exists, so the last figure it sent would otherwise stand as the worth of survivors there are none of.
            engagement.our_left_value = ours.value if ours.members else 0.0
            engagement.their_left_value = theirs.value if theirs.members else 0.0
            engagement.our_left_health = self._health_worth(ours, observation)
            engagement.their_left_health = self._health_worth(theirs, observation)
            engagement.seconds = (now - (ours.contract.issued_at_ms if ours.contract else now)) / 1000.0
            outcome = engagement.scored(self.score)
            self.statistics.outcomes.append(engagement.outcome)
            self.statistics.health_outcomes.append(engagement.outcome_health)
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
            # The worth this squad was formed with is left as the arena set it. The game side keeps that figure as a running maximum over the life of a squad number and never lowers it, which is right where a squad is reinforced over a match and wrong here: the arena hands the same two numbers to every fight of an episode, so after the first big fight the figure is the largest force either slot ever held rather than the force standing in this one. It is divided into the present worth to make the health of the squad, which is one of the layer's inputs, so a stale denominator is an input that says a fresh force is already half destroyed. Measured over the runs in flight, most fights after the first of an episode opened with that input already below full at full strength.
            squad.x, squad.y = state.x, state.y
            squad.spread = state.spread
            squad.losses = state.losses
            squad.status = Status(state.status)

    @staticmethod
    def _commissioned(fresh: Sequence, hostile: bool, wanted: Optional[Dict[int, int]]) -> List[int]:
        """The units of one side of a fight: what has newly appeared, taken against the order that was placed for it, type by type and no more of a type than were asked for.

        Filling a squad with everything that newly appeared is what let two kinds of stranger into a fight. One is this side's own base: a spawn-point player is given a headquarters and a builder, they arrive a step apart, and the wait for the opening board to settle can pass between the two, after which the builder is a unit that has just appeared and joins the squad. The other is an engagement abandoned as stillborn, whose spawn commands stay in the queue and whose units surface later, inside the window in which the next fight is being formed. Neither was commissioned, and a fight is between what was commissioned.

        With no order to take against — which is only the case if a fight is being formed without one — everything of the right side is taken, since there is nothing to say what does not belong.
        """
        taken: List[int] = []
        left = dict(wanted) if wanted else None
        for unit in fresh:
            if bool(unit.hostile) != hostile:
                continue
            if left is None:
                taken.append(unit.id)
                continue
            remaining = left.get(unit.type_index, 0)
            if remaining <= 0:
                continue
            left[unit.type_index] = remaining - 1
            taken.append(unit.id)
        return taken

    def _health_worth(self, squad: SquadRecord, observation: Observation) -> float:
        """What a side is worth counting every survivor at the share of its health it still holds.

        Taken from the unit rows rather than from the squad block, because the squad block carries the price of what is standing and nothing about how much of it is standing. The units of the squad are whatever is both in its membership and still on the board; the membership is pruned every period against what the board reports, so a unit that is in it is one the board still has.

        A type with no maximum health recorded counts whole, which is the same thing the sparse reading says about it and is therefore the reading that cannot make the two disagree for a reason that is not about the fight.
        """
        members = set(squad.members)
        if not members:
            return 0.0
        worth = 0.0
        for unit in observation.unit_states:
            if unit.id not in members:
                continue
            share = 1.0 if unit.max_health <= 0 else unit.health / unit.max_health
            worth += self.catalogue.value(unit.type_index) * min(1.0, max(0.0, share))
        return worth

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
        # Out of the placement stream and never out of the draw's, because how many candidates are clear enough to choose between is a fact about the board this policy made, and a choice among n of them consumes an amount of the stream that depends on n. See `placement`.
        if not standing:
            return self.placement.choice(self.sites)

        def clearance(site: Tuple[float, float]) -> float:
            return min(math.hypot(site[0] - x, site[1] - y) for x, y in standing)

        best = max(clearance(site) for site in self.sites)
        return self.placement.choice([site for site in self.sites if clearance(site) >= best * 0.9])

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

    def _rows(self, force: Sequence, slot: int, place: Tuple[float, float]) -> List[List[float]]:
        """One spawn row per unit for one side: type, player slot, position, count. Units are scattered a little so that they do not all arrive on the same point and spend the first seconds pushing each other apart. Returned a row at a time rather than run together, because the two sides' rows are interleaved before either order is sent."""
        rows: List[List[float]] = []
        for index, kind in enumerate(force):
            angle = 2 * math.pi * index / max(1, len(force))
            radius = 40.0 + 12.0 * index
            rows.append([float(kind.index), float(slot),
                         place[0] + math.cos(angle) * radius,
                         place[1] + math.sin(angle) * radius, 1.0])
        return rows

    @staticmethod
    def _interleave(ours: Sequence[Sequence[float]],
                    theirs: Sequence[Sequence[float]]) -> List[float]:
        """Merges the two sides' spawn rows into the one order that is sent, so that neither side is submitted ahead of the other.

        Spawning goes through the command queue a unit at a time and the queue is drained in the order it was filled, so a side whose whole order is submitted before the other's is the more completely on the board when the spawn wait is called: its late units sit nearer the front. Running the two orders one after the other, this side first, handed this side a few points of the strength that forms into its squad for no reason to do with either policy — drawn out as a left-right lean it was three points, which is enough to bias the self-play score the arena is measured against.

        Taken one unit from each side in turn, the two orders stay at the same depth in the queue throughout, so a wait that runs out cuts both sides to the same degree rather than only the second. Which side leads a pair alternates, so the one unit a side is unavoidably ahead by inside a pair falls on each side equally over the force. When one order is longer its tail runs on alone, which leans on the side that was dealt the larger force rather than on a fixed side, and which side that is was already drawn even.
        """
        rows: List[float] = []
        for index in range(max(len(ours), len(theirs))):
            pair = [ours[index] if index < len(ours) else None,
                    theirs[index] if index < len(theirs) else None]
            if index % 2:
                pair.reverse()
            for row in pair:
                if row is not None:
                    rows.extend(row)
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
        layers = [layer for layer in (self.tactics, self.opponent) if hasattr(layer, "close")]
        # Both sides file what they are still owed payment for before either of them ends anything. The two layers of a self-played fight share one instance number and one buffer, and a trajectory is cut by owner, so closing them one after the other had the first side's cut reach into the second side's live errands — which still had their last decisions sitting unfiled — and split each of them in two.
        for layer in layers:
            if hasattr(layer, "park"):
                layer.park()
        for layer in layers:
            layer.close()
