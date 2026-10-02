"""The strategic layer: one posture every ten seconds, and what that posture means to the layers below.

This is the layer the design says is worth the least to learn and the most to hand over. Its action space is five wide, its credit assignment horizon is the whole match, and a human with an hour of the game already knows when to stop expanding and start arming. So it is a script, and it is written to be read and argued with rather than fitted: the posture is chosen by a rule anyone can state in a sentence, and everything else here is that posture read through a table.

Nothing continuous crosses this boundary by choice. The allocation is a table lookup on the posture, and so are the loss allowance and the shape of the priorities, because the moment this layer emits tuned numbers directly a human taking it over has to emit tuned numbers too. What is left continuous is only what the board decides rather than the posture: which region is which, and how much has been contacted.

The transition rule is the design's opening one with the enemy's army weighed in: expand from the start, arm once income has levelled off, decide once the enemy is down to one base or, switched to push, once our army is far larger than theirs, defend while ground keeps being taken or while our army is far smaller than theirs. A posture other than deciding or defending is held for a minimum time, and arming turns back to expanding only on clear growth, so that one extractor finishing does not flip the chain between pressing and holding. Being ahead does not change the posture but makes the operational layer press; only a lead large enough to decide on does. It never selects TECH. That is not an oversight -TECH is reachable, has a full row in every table here, and is exactly the sort of call a human makes from outside the match ("they are massing air, buy the tier") and a script has no business making from inside it. `forced` is how it gets selected.

Income is the one figure read as a history rather than as a level, because "levelled off" is not a property of a number. The history is sampled on the strategic period from the game clock, so it means the same thing whatever the wall clock and whatever speed multiplier the instance is running at.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Set

from ...wire import RegionState
from .catalogue import Catalogue
from .contracts import ALLOCATION, EconomyOrders, FrontReport, OperationsOrders, Posture, Role
from .ground import enemy_spawns, frontier
from .options import Options

#: The strategic period in game time, per the design's 0.1Hz. Structural: the caller runs on it and the income history is sampled by it.
STRATEGY_PERIOD_MS = 10000

#: A call sooner than this after the last one is a repeat rather than a fresh period, and must not push a duplicate sample into the income history.
SAMPLE_GUARD_MS = STRATEGY_PERIOD_MS // 2

#: Initial value: strategic periods of income kept, which at 0.1Hz is fifty seconds of history. Short enough that a stall is noticed while there is still a match left to arm for, long enough that one extractor finishing does not read as growth.
INCOME_WINDOW = 5

#: Initial value: growth across the whole window, as a share of the oldest sample, below which income counts as levelled off. Five per cent over fifty seconds is a plateau; a working expansion does far better than that.
INCOME_PLATEAU_GROWTH = 0.05

#: Initial value: income below this is treated as not yet started rather than as levelled off, so that the opening -flat and near zero until the first extractor stands -does not read as a plateau and send us arming with nothing to arm from.
INCOME_PLATEAU_FLOOR = 10.0

#: Initial value: growth across the income window, as a share of the oldest sample, that turns arming back to expanding. Larger than the plateau threshold, so that income hovering at the plateau does not flip the posture every period.
RESUME_GROWTH = 0.15

#: Initial value: game time a posture is held before the rule may move off it, except into deciding or defending, which answer events that do not wait.
MIN_POSTURE_MS = 30000

#: Postures entered as soon as the rule asks for them, whatever the dwell says.
URGENT = (Posture.DECIDE, Posture.DEFEND)

#: Initial value: our army's worth against the enemy's at which the operational layer is told to press whatever the posture, when not switched to push. Switched, the odds, the loss allowance while pressing, the odds a decision is gone for and held to, and the odds below which the posture goes to defending are the searched values `tuning.push_odds`, `push_allowance_share`, `decide_odds`, `decide_hold_odds` and `yield_odds`.
PRESS_ODDS = 1.5

#: Initial value: credits of army the smaller side needs before the odds are read at all. Below it the ratio is one tank against none and says nothing about who is ahead.
ODDS_FLOOR = 1000.0

#: Initial value: how much more the first region of the expansion plan is worth to an expanding posture, so squads clear the region the builder is heading to.
PLAN_WEIGHT = 2.0

#: Initial value: how much a region with enemy extractors is worth to an arming posture that is pressing, against held ground at the front. Going for their income is how an army that is ahead converts the lead.
PRESS_WEIGHT = 2.0

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

#: Initial value: share of contacted enemy worth that has to be flying before the mix answers it with anti-air. Below this it is a scout and not a wing.
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

    def __init__(self, session, catalogue: Catalogue, options: Options = Options()) -> None:
        self.session = session
        self.catalogue = catalogue
        self.options = options
        self.tuning = options.tuning
        self.spawns = [region for region in session.regions if region.spawn]
        self.posture = Posture.EXPAND
        #: When the posture last changed, which is what the dwell is measured from.
        self.changed_at_ms = 0
        #: Game time spent in each posture and how often it changed, kept for the episode record.
        self.time_in: Dict[Posture, int] = {}
        self.changes = 0
        self.decided_at_ms: Optional[int] = None
        #: Whether our army is far enough ahead that the operational layer is told to press.
        self.pressing = False
        #: Regions to expand into, safest first, as last handed down.
        self.expansion: List[int] = []
        #: Set by a human to pin the posture. This is the interface the design has in mind for this layer, and the only route by which TECH is ever selected.
        self.forced: Optional[Posture] = None
        self.income_history: Deque[float] = deque(maxlen=INCOME_WINDOW)
        #: Fresh losses per period rather than the report's running figure, so that ground taken ten periods ago stops arguing for defending.
        self.loss_history: Deque[int] = deque(maxlen=LOSS_WINDOW)
        self.lost_regions = 0
        #: The most bases the enemy has ever been seen holding, which is what makes "one left" mean driven back rather than merely counted.
        self.most_enemy_bases = 0
        self.sampled_at_ms = -STRATEGY_PERIOD_MS
        #: The regions anything of ours could get to from home, on foot or carried by a kind of transport the catalogue has; None while the map is not known, when every region counts. A region outside it is neither planned onto nor given a priority.
        self.reachable: Optional[Set[int]] = None

    def decide(self, report: FrontReport, regions: List[RegionState], game_time_ms: int,
               contact: Optional[Dict[Role, float]] = None,
               home: Optional[RegionState] = None, airborne: float = 0.0) -> tuple[EconomyOrders, OperationsOrders]:
        self._sample(report, game_time_ms)
        self._keep_time(game_time_ms)
        chosen = self.forced if self.forced is not None else self._transition(report, game_time_ms)
        if chosen != self.posture:
            self.changed_at_ms = game_time_ms
            self.changes += 1
        self.posture = chosen
        allocation = ALLOCATION[self.posture]
        # A pinned posture is a human's choice of what the chain does, pressing included, so the odds only speak for the chain's own choice.
        self.pressing = self.forced is None and self._ahead(report)
        regions = [region for region in regions if self.reachable is None or region.id in self.reachable]
        self.expansion = frontier(regions, home, enemy_spawns(self.spawns, home), chained=self.options.frontier)

        economy = EconomyOrders(
            posture=self.posture,
            allocation=allocation,
            # A cap in credits per minute rather than a share, because a build order asks "can I afford the tier this minute" and cannot ask that of a ratio.
            tech_cap=allocation.tech * report.income * SECONDS_PER_MINUTE,
            target_mix=self._mix(contact, airborne),
            expansion=list(self.expansion),
        )
        share = LOSS_ALLOWANCE_SHARE[self.posture]
        if self.pressing and self.options.push:
            share = max(share, self.tuning.push_allowance_share)
        operations = OperationsOrders(
            posture=self.posture,
            priorities=self._priorities(regions),
            offensive=OFFENSIVE[self.posture] or self.pressing,
            loss_allowance=max(LOSS_ALLOWANCE_FLOOR, share * report.military_value),
            expansion=list(self.expansion),
        )
        return economy, operations

    def _keep_time(self, game_time_ms: int) -> None:
        """Adds the time since the last decision to the posture that was standing through it."""
        if self.decided_at_ms is not None and game_time_ms > self.decided_at_ms:
            self.time_in[self.posture] = self.time_in.get(self.posture, 0) + game_time_ms - self.decided_at_ms
        self.decided_at_ms = game_time_ms

    # ---- the transition rule ---------------------------------------------------------

    def _sample(self, report: FrontReport, game_time_ms: int) -> None:
        if game_time_ms - self.sampled_at_ms < SAMPLE_GUARD_MS:
            return
        self.sampled_at_ms = game_time_ms
        self.income_history.append(report.income)
        self.most_enemy_bases = max(self.most_enemy_bases, report.enemy_bases)
        self.loss_history.append(max(0, report.lost_regions - self.lost_regions))
        self.lost_regions = report.lost_regions

    def _transition(self, report: FrontReport, game_time_ms: int) -> Posture:
        """The rule's answer, held back while the standing posture is younger than MIN_POSTURE_MS unless the answer is one that cannot wait."""
        wanted = self._rule(report)
        if (self.options.dwell and wanted != self.posture and wanted not in URGENT
                and game_time_ms - self.changed_at_ms < MIN_POSTURE_MS):
            return self.posture
        return wanted

    def _rule(self, report: FrontReport) -> Posture:
        """The design's opening rule, in the order the conditions override each other. Deciding comes first because one enemy base left is a terminal condition and its near unlimited allowance subsumes defending anyway; ground being taken or an army far smaller than theirs comes next because losing ground makes every other plan moot; the income plateau is what is left once neither of those is happening. TECH is unreachable from here by design and is selected through `forced`."""
        # One base left means the enemy has been driven back to their last, not that we can only see one of them. In a match of two that is true from the opening whistle, so what is tested is that they held more at some point and hold one now; otherwise every match would open by committing everything to a final battle against an opponent at full strength.
        if report.enemy_bases == 1 and self.most_enemy_bases > 1:
            return Posture.DECIDE
        # Switched to push, an army far enough ahead goes for the decision on the odds alone, and holds to it until the odds have fallen well back, so that one lost skirmish does not call the attack off.
        if self.options.push and self.options.relative and self._odds(report) is not None:
            odds = self._odds(report)
            if odds >= self.tuning.decide_odds or (self.posture == Posture.DECIDE and odds >= self.tuning.decide_hold_odds):
                return Posture.DECIDE
        if self._overrun(report) or self._outmatched(report):
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
        growth = RESUME_GROWTH if self.options.dwell and self.posture == Posture.ARM else INCOME_PLATEAU_GROWTH
        return newest - oldest < growth * max(oldest, 1.0)

    def _outmatched(self, report: FrontReport) -> bool:
        """Whether the enemy's army is so much larger than ours that holding what we have is all it can be asked to do."""
        if not self.options.relative or report.enemy_military < ODDS_FLOOR:
            return False
        return report.military_value < self.tuning.yield_odds * report.enemy_military

    def _ahead(self, report: FrontReport) -> bool:
        """Whether our army is far enough ahead of theirs to press whatever the posture says."""
        if not self.options.relative or report.military_value < ODDS_FLOOR:
            return False
        return report.military_value >= (self.tuning.push_odds if self.options.push else PRESS_ODDS) * report.enemy_military

    @staticmethod
    def _odds(report: FrontReport) -> Optional[float]:
        """Our army's worth over the enemy's, or None while ours is below ODDS_FLOOR, when the ratio says nothing. An enemy with no army at all gives unbounded odds."""
        if report.military_value < ODDS_FLOOR:
            return None
        return report.military_value / report.enemy_military if report.enemy_military > 0 else float("inf")

    def _overrun(self, report: FrontReport) -> bool:
        if report.lost_regions >= LOSS_STANDING:
            return True
        return sum(1 for lost in self.loss_history if lost > 0) >= LOSS_PERIODS

    # ---- what the posture means ------------------------------------------------------

    def _mix(self, contact: Optional[Dict[Role, float]], airborne: float) -> Dict[Role, float]:
        """The posture's target mix, with a share moved into anti-air when enough of what the squads have run into flies. Flying rather than fast: the fast role also covers hovercraft, and anti-air bought against hovercraft cannot shoot them."""
        mix = dict(TARGET_MIX[self.posture])
        if not contact:
            return mix
        total = sum(contact.values())
        if total <= 0.0 or airborne / total < AIR_CONTACT_SHARE:
            return mix
        rest = 1.0 - mix[Role.ANTI_AIR]
        if rest <= 0.0:
            return mix
        for role in mix:
            mix[role] *= (1.0 - ANTI_AIR_SHIFT) if role != Role.ANTI_AIR else 1.0
        mix[Role.ANTI_AIR] += ANTI_AIR_SHIFT * rest
        return mix

    def _priorities(self, regions: List[RegionState]) -> Dict[int, float]:
        """What each region is worth to this posture, from 0 to 1, keyed by the map's own region id. Read off the posture: expanding prizes resource ground nobody holds, defending prizes what we already stand on, deciding prizes wherever the enemy still is. A region nothing of ours could get to is worth nothing."""
        scores = {region.id: self._score(region) if self.reachable is None or region.id in self.reachable else 0.0
                  for region in regions}
        return _normalise(scores, regions)

    def _score(self, region: RegionState) -> float:
        near = _nearness(region)
        if self.posture == Posture.EXPAND:
            # Unheld resource ground, in preference to ground already producing; contested ground is worth less than empty ground because expanding is not what pays for a fight. The region the builder goes to next is worth most, so that the army is there first.
            if region.held_by_us:
                return 0.2 * region.resources * near
            planned = PLAN_WEIGHT if self.options.cover and self.expansion and region.id == self.expansion[0] else 1.0
            return region.resources * near * (0.4 if region.held_by_enemy else 1.0) * planned
        if self.posture == Posture.TECH:
            # Home and its neighbours, and nothing the enemy is standing on: the posture exists to buy time, not ground.
            return 0.0 if region.enemy_value > 0.0 else (1.0 + region.resources) * near * near
        if self.posture == Posture.ARM:
            # The front, meaning ground we hold that the enemy has been seen near. Ground we hold and nobody contests still counts, at a fraction, so the priorities are not empty in a quiet phase.
            held = 1.0 if region.held_by_us else 0.3
            front = held * (1.0 + region.resources) * near * (2.0 if region.enemy_value > 0.0 else 1.0)
            # Pressing, the enemy's extractors are worth going for as well: their income is what keeps an army that is behind in the match.
            if self.pressing and region.held_by_enemy:
                return max(front, PRESS_WEIGHT * (1.0 + region.held_by_enemy) * near)
            return front
        if self.posture == Posture.DEFEND:
            # What we hold, weighted by what is there to lose and by how hard it is being pushed.
            if not region.held_by_us:
                return 0.0
            return (region.our_value + 100.0 * region.resources) * (2.0 if region.enemy_value > 0.0 else 1.0)
        # Deciding: the enemy's remaining base, which is whatever region still has their worth standing in it, and their resource ground after that.
        return region.enemy_value + 200.0 * region.held_by_enemy
