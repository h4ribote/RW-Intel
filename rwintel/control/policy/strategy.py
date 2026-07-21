"""The strategic layer: one posture every ten seconds, and what that posture means to the layers below.

This is the layer the design says is worth the least to learn and the most to hand over. Its action space is five wide, its credit assignment horizon is the whole match, and a human with an hour of the game already knows when to stop expanding and start arming. So it is a script, and it is written to be read and argued with rather than fitted: the posture is chosen by a rule anyone can state in a sentence, and everything else here is that posture read through a table.

Nothing continuous crosses this boundary by choice. The allocation is a table lookup on the posture, and so are the loss allowance and the shape of the priorities, because the moment this layer emits tuned numbers directly a human taking it over has to emit tuned numbers too. What is left continuous is only what the board decides rather than the posture: which region is which, and how much has been contacted.

The transition rule is the design's opening one and no more: expand from the start, arm once income has levelled off, decide once the enemy is down to one base, defend while ground keeps being taken. It never selects TECH. That is not an oversight — TECH is reachable, has a full row in every table here, and is exactly the sort of call a human makes from outside the match ("they are massing air, buy the tier") and a script has no business making from inside it. `forced` is how it gets selected.

Income is the one figure read as a history rather than as a level, because "levelled off" is not a property of a number. The history is sampled on the strategic period from the game clock, so it means the same thing whatever the wall clock and whatever speed multiplier the instance is running at.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from ...wire import RegionState
from .catalogue import Catalogue
from .contracts import ALLOCATION, EconomyOrders, FrontReport, OperationsOrders, Posture, Role

#: The strategic period in game time, per the design's 0.1Hz. Structural: the caller runs on it and the income history is sampled by it.
STRATEGY_PERIOD_MS = 10000

#: A call sooner than this after the last one is a repeat rather than a fresh period, and must not push a duplicate sample into the income history.
SAMPLE_GUARD_MS = STRATEGY_PERIOD_MS // 2

#: Initial value: strategic periods of income kept, which at 0.1Hz is fifty seconds of history. Short enough that a stall is noticed while there is still a match left to arm for, long enough that one extractor finishing does not read as growth.
INCOME_WINDOW = 5

#: Initial value: growth across the whole window, as a share of the oldest sample, below which income counts as levelled off. Five per cent over fifty seconds is a plateau; a working expansion does far better than that.
INCOME_PLATEAU_GROWTH = 0.05

#: Initial value: income below this is treated as not yet started rather than as levelled off, so that the opening — flat and near zero until the first extractor stands — does not read as a plateau and send us arming with nothing to arm from.
INCOME_PLATEAU_FLOOR = 10.0

#: Initial value: strategic periods of region losses kept. The rule is about ground being taken repeatedly, not about one region changing hands, so it needs a window rather than a level.
LOSS_WINDOW = 3

#: Initial value: how many periods within that window must show a fresh loss before the posture goes to defending. Two of three also sets the release: two quiet periods and it lets go, which is the hysteresis that stops the posture oscillating on a single raid.
LOSS_PERIODS = 2

#: Initial value: a recent-loss count standing at or above this is being pushed back whatever the periods say. The report's figure counts regions lost lately and may sit at a ceiling while ground keeps going, in which case there are no fresh losses to count and only the level shows it.
LOSS_STANDING = 3

#: Initial value: shares of military spending by role for each posture. Expanding buys builders because the thing worth spending on is more ground; arming and deciding buy armour and guns; defending buys anti-air and enough builders to keep turrets going up.
TARGET_MIX: Dict[Posture, Dict[Role, float]] = {
    Posture.EXPAND: {Role.ARMOUR: 0.40, Role.ARTILLERY: 0.05, Role.ANTI_AIR: 0.05, Role.FAST: 0.10, Role.BUILDER: 0.40},
    Posture.TECH: {Role.ARMOUR: 0.40, Role.ARTILLERY: 0.15, Role.ANTI_AIR: 0.10, Role.FAST: 0.05, Role.BUILDER: 0.30},
    Posture.ARM: {Role.ARMOUR: 0.50, Role.ARTILLERY: 0.25, Role.ANTI_AIR: 0.10, Role.FAST: 0.10, Role.BUILDER: 0.05},
    Posture.DEFEND: {Role.ARMOUR: 0.45, Role.ARTILLERY: 0.15, Role.ANTI_AIR: 0.20, Role.FAST: 0.05, Role.BUILDER: 0.15},
    Posture.DECIDE: {Role.ARMOUR: 0.55, Role.ARTILLERY: 0.25, Role.ANTI_AIR: 0.10, Role.FAST: 0.10, Role.BUILDER: 0.00},
}

#: Initial value: share of contacted enemy worth that has to be fast-moving before the mix answers it with anti-air. Below this it is a scout and not a wing.
AIR_CONTACT_SHARE = 0.25

#: Initial value: how much of the mix moves into anti-air when that happens, taken from the other roles in proportion so the shape of the rest is kept.
ANTI_AIR_SHIFT = 0.25

#: Initial value: distance from home at which a region is worth half what the same region would be worth at home. Priorities fall off with it so that a posture prefers the near instance of whatever it is after.
DISTANCE_SCALE = 2000.0

#: Initial value: what may be lost across every mission at once, as a multiple of our own commanded worth. Expanding pays almost nothing for ground it can take uncontested; defending pays for ground it already has; deciding is the design's "near unlimited" and is set high enough that the operational layer never treats it as a constraint.
LOSS_ALLOWANCE_SHARE: Dict[Posture, float] = {
    Posture.EXPAND: 0.15,
    Posture.TECH: 0.10,
    Posture.ARM: 0.35,
    Posture.DEFEND: 0.60,
    Posture.DECIDE: 5.00,
}

#: Initial value: credits of allowance granted regardless of our worth, so that an opening with four tanks on the board can still be told to fight for something.
LOSS_ALLOWANCE_FLOOR = 500.0

#: Whether the posture presses. Expanding presses outward for ground and deciding presses for the base; arming holds the front the design says to maintain, and the other two hold what they have.
OFFENSIVE: Dict[Posture, bool] = {
    Posture.EXPAND: True,
    Posture.TECH: False,
    Posture.ARM: False,
    Posture.DEFEND: False,
    Posture.DECIDE: True,
}

#: The wire reports income per second and the technology cap is per minute, which is the unit a build order reasons in.
SECONDS_PER_MINUTE = 60.0


def _nearness(region: RegionState) -> float:
    """How much a region counts for by virtue of being reachable. Everything here wants the near one of two otherwise equal places, and nothing here wants a hard cut-off at some radius."""
    return 1.0 / (1.0 + region.distance_from_home / DISTANCE_SCALE)


def _normalise(scores: Dict[int, float], regions: List[RegionState]) -> Dict[int, float]:
    """Priorities are relative, so the best region on the board is a 1 whatever the raw numbers were. When a posture finds nothing it wants at all, distance from home is the fallback ordering: the operational layer is owed a preference every period, and "near" is the least wrong one."""
    top = max(scores.values(), default=0.0)
    if top <= 0.0:
        return {r.id: _nearness(r) for r in regions}
    return {region_id: min(1.0, score / top) for region_id, score in scores.items()}


class Strategy:
    """The posture and its consequences. Holds only the two histories the transition rule needs, which is the whole of this layer's state."""

    def __init__(self, session, catalogue: Catalogue) -> None:
        self.session = session
        self.catalogue = catalogue
        self.posture = Posture.EXPAND
        #: Set by a human to pin the posture. This is the interface the design has in mind for this layer, and the only route by which TECH is ever selected.
        self.forced: Optional[Posture] = None
        self.income_history: Deque[float] = deque(maxlen=INCOME_WINDOW)
        #: Fresh losses per period rather than the report's running figure, so that ground taken ten periods ago stops arguing for defending.
        self.loss_history: Deque[int] = deque(maxlen=LOSS_WINDOW)
        self.lost_regions = 0
        #: The most bases the enemy has ever been seen holding, which is what makes "one left" mean driven back rather than merely counted.
        self.most_enemy_bases = 0
        self.sampled_at_ms = -STRATEGY_PERIOD_MS

    def decide(self, report: FrontReport, regions: List[RegionState], game_time_ms: int,
               contact: Optional[Dict[Role, float]] = None) -> tuple[EconomyOrders, OperationsOrders]:
        self._sample(report, game_time_ms)
        self.posture = self.forced if self.forced is not None else self._transition(report)
        allocation = ALLOCATION[self.posture]

        economy = EconomyOrders(
            posture=self.posture,
            allocation=allocation,
            # A cap in credits per minute rather than a share, because a build order asks "can I afford the tier this minute" and cannot ask that of a ratio.
            tech_cap=allocation.tech * report.income * SECONDS_PER_MINUTE,
            target_mix=self._mix(contact),
        )
        operations = OperationsOrders(
            posture=self.posture,
            priorities=self._priorities(regions),
            offensive=OFFENSIVE[self.posture],
            loss_allowance=max(LOSS_ALLOWANCE_FLOOR, LOSS_ALLOWANCE_SHARE[self.posture] * report.military_value),
        )
        return economy, operations

    # ---- the transition rule ---------------------------------------------------------

    def _sample(self, report: FrontReport, game_time_ms: int) -> None:
        if game_time_ms - self.sampled_at_ms < SAMPLE_GUARD_MS:
            return
        self.sampled_at_ms = game_time_ms
        self.income_history.append(report.income)
        self.most_enemy_bases = max(self.most_enemy_bases, report.enemy_bases)
        self.loss_history.append(max(0, report.lost_regions - self.lost_regions))
        self.lost_regions = report.lost_regions

    def _transition(self, report: FrontReport) -> Posture:
        """The design's opening rule, in the order the conditions override each other. Deciding comes first because one enemy base left is a terminal condition and its near unlimited allowance subsumes defending anyway; ground being taken comes next because losing it makes every other plan moot; the income plateau is what is left once neither of those is happening. TECH is unreachable from here by design and is selected through `forced`."""
        # One base left means the enemy has been driven back to their last, not that we can only see one of them. In a match of two that is true from the opening whistle, so what is tested is that they held more at some point and hold one now; otherwise every match would open by committing everything to a final battle against an opponent at full strength.
        if report.enemy_bases == 1 and self.most_enemy_bases > 1:
            return Posture.DECIDE
        if self._overrun(report):
            return Posture.DEFEND
        if self._income_levelled_off():
            return Posture.ARM
        return Posture.EXPAND

    def _income_levelled_off(self) -> bool:
        if len(self.income_history) < INCOME_WINDOW:
            return False
        oldest, newest = self.income_history[0], self.income_history[-1]
        if newest < INCOME_PLATEAU_FLOOR:
            return False
        return newest - oldest < INCOME_PLATEAU_GROWTH * max(oldest, 1.0)

    def _overrun(self, report: FrontReport) -> bool:
        if report.lost_regions >= LOSS_STANDING:
            return True
        return sum(1 for lost in self.loss_history if lost > 0) >= LOSS_PERIODS

    # ---- what the posture means ------------------------------------------------------

    def _mix(self, contact: Optional[Dict[Role, float]]) -> Dict[Role, float]:
        mix = dict(TARGET_MIX[self.posture])
        if not contact:
            return mix
        total = sum(contact.values())
        # The roles carry no air flag: a type is classified FAST exactly when it moves by AIR or HOVER, so fast contact is the closest thing to air contact anything reports upward. It over-answers hovercraft, which is the cheap direction to be wrong in.
        if total <= 0.0 or contact.get(Role.FAST, 0.0) / total < AIR_CONTACT_SHARE:
            return mix
        rest = 1.0 - mix[Role.ANTI_AIR]
        if rest <= 0.0:
            return mix
        for role in mix:
            mix[role] *= (1.0 - ANTI_AIR_SHIFT) if role != Role.ANTI_AIR else 1.0
        mix[Role.ANTI_AIR] += ANTI_AIR_SHIFT * rest
        return mix

    def _priorities(self, regions: List[RegionState]) -> Dict[int, float]:
        """What each region is worth to this posture, from 0 to 1, keyed by the map's own region id. Read off the posture: expanding prizes resource ground nobody holds, defending prizes what we already stand on, deciding prizes wherever the enemy still is."""
        scores = {region.id: self._score(region) for region in regions}
        return _normalise(scores, regions)

    def _score(self, region: RegionState) -> float:
        near = _nearness(region)
        if self.posture == Posture.EXPAND:
            # Unheld resource ground, in preference to ground already producing; contested ground is worth less than empty ground because expanding is not what pays for a fight.
            if region.held_by_us:
                return 0.2 * region.resources * near
            return region.resources * near * (0.4 if region.held_by_enemy else 1.0)
        if self.posture == Posture.TECH:
            # Home and its neighbours, and nothing the enemy is standing on: the posture exists to buy time, not ground.
            return 0.0 if region.enemy_value > 0.0 else (1.0 + region.resources) * near * near
        if self.posture == Posture.ARM:
            # The front, meaning ground we hold that the enemy has been seen near. Ground we hold and nobody contests still counts, at a fraction, so the priorities are not empty in a quiet phase.
            held = 1.0 if region.held_by_us else 0.3
            return held * (1.0 + region.resources) * near * (2.0 if region.enemy_value > 0.0 else 1.0)
        if self.posture == Posture.DEFEND:
            # What we hold, weighted by what is there to lose and by how hard it is being pushed.
            if not region.held_by_us:
                return 0.0
            return (region.our_value + 100.0 * region.resources) * (2.0 if region.enemy_value > 0.0 else 1.0)
        # Deciding: the enemy's remaining base, which is whatever region still has their worth standing in it, and their resource ground after that.
        return region.enemy_value + 200.0 * region.held_by_enemy
