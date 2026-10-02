"""The script's decisions for the three layers that are learnt, taken from the encoded state and nothing else.

A learnt layer is a copy of the script layer with one decision replaced, and the script is its teacher. A teacher that decides on something its pupil cannot see is one the pupil can only partly imitate, and the difference is lost the moment the pupil takes over. So the rules the economy, the operational and the tactical layers decide by are written here as functions of exactly the numbers the networks read (`encoding`), and whatever a rule needs that the encoding did not carry was added to the encoding rather than looked up on the side.

Each judge scores every action rather than naming one. The action it takes is the best scored, and the scores themselves are what a teacher file keeps beside it: a softened copy of them is a distribution over the actions that says which were close, which a single label cannot.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from ...wire import REGION_SLOTS, SQUAD_SLOTS, Deviation, Status, Task
from .contracts import DOCTRINES, Doctrine
from .options import Options
from .tuning import Tuning
from .encoding import (
    COUNT_SCALE,
    DOCTRINE_LIST,
    ECONOMIC_CONTEXT,
    ECONOMIC_CONTEXT_SIZE,
    GLOBAL_FEATURES,
    GLOBAL_SIZE,
    INCOME_SCALE,
    INVESTMENT_FEATURES,
    INVESTMENT_SIZE,
    INVESTMENT_SLOTS,
    LOSS_SCALE,
    PRICE_SCALE,
    RAISE_STAGES,
    RANGE_SCALE,
    REGION_FEATURES,
    REGION_SIZE,
    SPREAD_SCALE,
    SQUAD_FEATURES,
    SQUAD_SIZE,
    SQUAD_SIZE_SCALE,
    STATUSES,
    TACTICAL_FEATURES,
    TASKS,
    TRANSPORT_FEATURES,
    TRANSPORT_SIZE,
    TREASURY_SCALE,
    MEANS,
    OPERATIONAL_PLANS,
    Investment,
)

# ---- the tactical judge -------------------------------------------------------------------

#: Losses this close to the contract's budget are as good as spent, and the mission is worth breaking off before the rest of it goes too. The game side calls a mission losing at seven tenths, so this sits above that: hearing "losing" is information for operations, acting on it is a decision made here.
BUDGET_CLOSE_SHARE = 0.8

#: Below this much lost the exchange ratio is one cheap unit's worth of noise and says nothing about how the fight is going.
EXCHANGE_MIN_LOSS = 200.0

#: Enemy worth destroyed under this multiple of our own worth lost is a trade to walk away from. Below one rather than at one because a squad that is merely breaking even is still spending a budget it was given for a purpose.
EXCHANGE_BAD_RATIO = 0.6

#: A squad whose members stand within this of their centre is bunched enough for one area weapon to cover several of them. The game side scatters to 140, so a squad already looser than this has nothing to gain by scattering again.
BUNCHED_SPREAD = 140.0

#: This share of the squad hit inside the recent window at once reads as one weapon covering the squad rather than as several weapons picking at it. Without an area radius on the type table this is the whole of the evidence for an area weapon.
SPLASH_HIT_SHARE = 0.5

#: Fewer members than this and there is no formation to scatter, only units to send in separate directions.
SPREAD_MIN_MEMBERS = 3

#: How much further the squad has to reach than what is shooting at it before backing away while shooting pays for the ground given up. Under this the gap closes again before the squad has fired.
KITE_RANGE_MARGIN = 80.0

#: Fewer shooters than this and concentrating them changes nothing, because the engine's own target acquisition already has them on the same few enemies.
FOCUS_MIN_SHOOTERS = 3

#: With fewer enemies than this there is nothing to concentrate away from: the squad is already fighting the only thing present.
FOCUS_MIN_ENEMIES = 2

#: A count written as a share of SQUAD_SIZE_SCALE is compared against a whole number with this much allowance for rounding.
_EPSILON = 1e-6

_TACTICAL = {name: index for index, name in enumerate(TACTICAL_FEATURES)}


class TacticsJudge:
    """The tactical rule ladder, read off the tactical features. With `predict`, a squad also withdraws from a fight the combat table expects it to lose, at `tuning.predict_withdraw` or worse, before the losses bear that out, and the whole way out at `tuning.predict_withdraw_far` or worse."""

    def __init__(self, predict: bool = True, tuning: Tuning = Tuning()) -> None:
        self.predict = predict
        self.tuning = tuning

    def scores(self, state: Sequence[float]) -> List[float]:
        """A score for every departure: the rungs of the ladder that apply, highest first in the ladder's order, holding at nought, and the rest below it."""
        ladder = self._ladder(state)
        scores = [-1.0] * len(Deviation)
        scores[Deviation.HOLD] = 0.0
        for rank, departure in enumerate(ladder):
            scores[departure] = max(scores[departure], float(len(ladder) - rank))
        return scores

    def choose(self, state: Sequence[float]) -> Deviation:
        scores = self.scores(state)
        return Deviation(max(range(len(scores)), key=lambda index: (scores[index], -index)))

    def _ladder(self, state: Sequence[float]) -> List[Deviation]:
        """Every departure whose rung applies, in the order of what would be worst to get wrong: breaking off, scattering from area fire, keeping out of reach, and concentrating."""
        def f(name: str) -> float:
            return state[_TACTICAL[name]]

        engaged, under_fire = f("engaged") > 0, f("under_fire") > 0
        if not engaged and not under_fire:
            return []
        rungs: List[Deviation] = []
        losing = f("status_losing") > 0
        spent = losing or (f("budget_share") > 0 and f("budget_spent") >= BUDGET_CLOSE_SHARE - _EPSILON)
        losses = f("losses") * LOSS_SCALE
        bad_trade = losses >= EXCHANGE_MIN_LOSS - _EPSILON and f("exchange") < EXCHANGE_BAD_RATIO / (1.0 + EXCHANGE_BAD_RATIO)
        if spent or bad_trade:
            rungs.append(Deviation.WITHDRAW_FAR if losing else Deviation.WITHDRAW)
        if self.predict and engaged and f("predicted") <= self.tuning.predict_withdraw:
            rungs.append(Deviation.WITHDRAW_FAR if f("predicted") <= self.tuning.predict_withdraw_far else Deviation.WITHDRAW)
        members = f("members") * SQUAD_SIZE_SCALE
        if (members >= SPREAD_MIN_MEMBERS - _EPSILON and f("spread") * SPREAD_SCALE <= BUNCHED_SPREAD + _EPSILON
                and f("artillery_present") > 0 and f("share_under_fire") >= SPLASH_HIT_SHARE - _EPSILON):
            rungs.append(Deviation.SPREAD)
        if f("our_reach") > 0 and f("their_reach") > 0 and f("reach_advantage") * RANGE_SCALE >= KITE_RANGE_MARGIN - _EPSILON:
            rungs.append(Deviation.KITE)
        if (f("armed_members") * SQUAD_SIZE_SCALE >= FOCUS_MIN_SHOOTERS - _EPSILON
                and f("enemies_near") * SQUAD_SIZE_SCALE >= FOCUS_MIN_ENEMIES - _EPSILON and f("target_in_reach") > 0):
            rungs.append(Deviation.FOCUS_THREAT if f("long_in_reach") > 0 else Deviation.FOCUS)
        return rungs


# ---- the operational judge ----------------------------------------------------------------

#: How much of a region's worth a march of half the distance scale costs, which is REACH_COST per 2000 world units with the region row's distances quoted against 4000. Opening value.
REACH_COST = 0.8

#: What one resource point adds to holding or raiding a region, with the row's resource count quoted against four. Opening value.
RESOURCE_WORTH = 0.6

#: How much the enemy in a region we hold raises it as something to garrison, and how much it puts a raid off. Opening values.
THREAT_WEIGHT = 0.3
RAID_RISK = 0.8

#: How much our own strength standing in a region puts a garrison off it, so that garrisons spread over what we hold. Opening value.
CROWDING_COST = 0.3


#: Without `concentrate`, the share of the force in a region a vanguard needs before it surrounds, which is the old rule's odds of two to one. Opening value.
ENCIRCLE_SHARE = 2.0 / 3.0

_GLOBAL = {name: index for index, name in enumerate(GLOBAL_FEATURES)}
_REGION = {name: index for index, name in enumerate(REGION_FEATURES)}
_SQUAD = {name: index for index, name in enumerate(SQUAD_FEATURES)}
_TRANSPORT = {name: index for index, name in enumerate(TRANSPORT_FEATURES)}

#: Tasks whose completion is the state they exist to keep.
HOLDING_TASKS = (Task.DEFEND, Task.ESCORT)


class _Board:
    """The encoded state read by name."""

    def __init__(self, state: Sequence[float], slot: int) -> None:
        self.state = state
        self.slot = slot

    def glob(self, name: str) -> float:
        return self.state[_GLOBAL[name]]

    def region(self, region: int, name: str) -> float:
        return self.state[GLOBAL_SIZE + region * REGION_SIZE + _REGION[name]]

    def squad(self, name: str, slot: Optional[int] = None) -> float:
        slot = self.slot if slot is None else slot
        return self.state[GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + slot * SQUAD_SIZE + _SQUAD[name]]

    def one_hot(self, prefix: str, options: Sequence, slot: Optional[int] = None) -> Optional[int]:
        values = [self.squad(f"{prefix}_{option.name.lower()}", slot) for option in options]
        best = max(range(len(values)), key=lambda index: values[index])
        return best if values[best] > 0 else None

    def idle_garrison(self, slot: int) -> bool:
        """Whether the squad in a slot is a garrison of ours with nothing to do: one holding no contract, or one whose ground is quiet."""
        if self.squad("valid", slot) <= 0 or self.squad("taskable", slot) <= 0 or self.squad("held", slot) > 0:
            return False
        if self.one_hot("doctrine", DOCTRINE_LIST, slot) != DOCTRINE_LIST.index(Doctrine.GARRISON):
            return False
        return self.one_hot("task", TASKS, slot) is None or self.one_hot("status", STATUSES, slot) == STATUSES.index(Status.COMPLETE)


class OperationsJudge:
    """Where a squad goes and what it does there, read off the operational features of the squad being decided about. With `concentrate`, a vanguard attacks only where the combat table expects it to win together with whatever else is bound there, is drawn to where others are already bound, and gathers at a rally point when nowhere can be won. With `cover`, the free garrison nearest the next region of the expansion plan goes to stand on it."""

    def __init__(self, concentrate: bool = True, cover: bool = True, tuning: Tuning = Tuning()) -> None:
        self.concentrate = concentrate
        self.cover = cover
        #: With `concentrate`: the predicted outcome, from -1 to +1, a vanguard needs before it attacks a region (`attack_edge`) and at which it surrounds rather than pushes straight in (`encircle_edge`); how much the strength already bound for a region draws another vanguard to it (`concentration`); how much the predicted outcome itself counts (`predict_weight`); and how far forward a rally point is drawn when nothing can be attacked (`forward`).
        self.tuning = tuning

    def scores(self, state: Sequence[float], slot: int, region_mask: Sequence[float],
               plan_masks: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        """A score for every region slot and for every plan, the plans scored for the region the regions' scores put first. Regions the rule would not send the squad to score below every one it would; slots the masks forbid score negative infinity.

        A plan is a task and a means. Of the plans open in that region, the one with the rule's task and the rule's means scores highest, the rule's task by another means next, and the rest below them. The means is walking where the squad can walk, otherwise the open transport already carrying the squad, and otherwise the open transport standing nearest the squad, read off the transport block.
        """
        board = _Board(state, slot)
        live = [r for r in range(len(region_mask)) if region_mask[r] > 0 and board.region(r, "valid") > 0]
        region_scores = [-math.inf] * len(region_mask)
        plan_scores = [-math.inf] * OPERATIONAL_PLANS
        if not live:
            return region_scores, plan_scores
        candidates, task = self._candidates(board, live)
        floor = min(candidates.values()) - 2.0 if candidates else 0.0
        for r in live:
            region_scores[r] = candidates.get(r, floor)
        best = max(range(len(region_scores)), key=lambda index: (region_scores[index], -index))
        mask = plan_masks[best]
        means = self._means(board, mask, int(task))
        for plan, allowed in enumerate(mask):
            if allowed > 0:
                plan_scores[plan] = (1.0 if plan // MEANS == int(task) else 0.0) + (0.5 if plan % MEANS == means else 0.0)
        return region_scores, plan_scores

    def choose(self, state: Sequence[float], slot: int, region_mask: Sequence[float],
               plan_masks: Sequence[Sequence[float]]) -> Optional[Tuple[int, int]]:
        """Where and which plan, as region slot and plan index."""
        regions, plans = self.scores(state, slot, region_mask, plan_masks)
        if not any(math.isfinite(s) for s in regions) or not any(math.isfinite(s) for s in plans):
            return None
        region = max(range(len(regions)), key=lambda index: (regions[index], -index))
        plan = max(range(len(plans)), key=lambda index: (plans[index], -index))
        return region, plan

    @staticmethod
    def _means(board: _Board, mask: Sequence[float], task: int) -> int:
        """The means index (0 for walking, k + 1 for the transport in slot k) the rule takes for a task: walking when it is open, otherwise the open transport already carrying the squad, otherwise the open transport nearest the squad."""
        open_means = [m for m in range(MEANS) if mask[task * MEANS + m] > 0]
        if not open_means or 0 in open_means:
            return 0
        start = GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + SQUAD_SLOTS * SQUAD_SIZE

        def feature(m: int, name: str) -> float:
            return board.state[start + (m - 1) * TRANSPORT_SIZE + _TRANSPORT[name]]

        return min(open_means, key=lambda m: (-feature(m, "mine"), feature(m, "distance"), m))

    # ---- the rules ------------------------------------------------------------------------

    def _candidates(self, board: _Board, live: List[int]):
        doctrine_index = board.one_hot("doctrine", DOCTRINE_LIST)
        doctrine = DOCTRINE_LIST[doctrine_index] if doctrine_index is not None else Doctrine.VANGUARD
        status_index = board.one_hot("status", STATUSES)
        status = STATUSES[status_index] if status_index is not None else Status.ACTIVE
        task_index = board.one_hot("task", TASKS)
        held_task = TASKS[task_index] if task_index is not None else None
        current = next((r for r in live if board.region(r, "current_target") > 0), None)
        running = status == Status.ACTIVE or (status == Status.COMPLETE and held_task in HOLDING_TASKS)
        # A mission that has stopped moving, is being lost, is finished or cannot be reached takes its region off the table, which is what makes the search find a different answer rather than the same one.
        avoid = current if not running and status in (Status.STALLED, Status.LOSING, Status.COMPLETE, Status.UNREACHABLE) else None
        home = min(live, key=lambda r: (board.region(r, "distance"), r))
        allowed = DOCTRINES[doctrine].tasks

        if status == Status.LOSING and Task.WITHDRAW in allowed:
            return {home: 1.0}, Task.WITHDRAW
        if doctrine == Doctrine.GARRISON:
            return self._garrison(board, live, avoid, home), Task.DEFEND
        if doctrine == Doctrine.RAID:
            return self._raid(board, live, avoid), Task.RAID
        if doctrine == Doctrine.VANGUARD:
            return self._vanguard(board, live, avoid, home)
        if doctrine in (Doctrine.FLEET, Doctrine.AIRWING):
            # A fleet and an air wing go where a vanguard would, among the regions they can reach, and press straight in: neither surrounds.
            scores, _ = self._vanguard(board, live, avoid, home)
            return scores, Task.ATTACK
        return {home: 1.0}, Task.ESCORT

    def _reach(self, board: _Board, r: int) -> float:
        return REACH_COST * board.region(r, "from_squad")

    def _garrison(self, board: _Board, live: List[int], avoid: Optional[int], home: int):
        """The most valuable ground we already draw an income from; or, with `cover`, for the free garrison nearest the next region of the expansion plan, that region when nothing of ours stands there or is bound there, so that the builder finds it covered."""
        if self.cover and board.idle_garrison(board.slot):
            mine = board.squad("plan_distance")
            nearest = all((mine, board.slot) <= (board.squad("plan_distance", other), other)
                          for other in range(SQUAD_SLOTS) if other != board.slot and board.idle_garrison(other))
            for r in live:
                if (nearest and r != avoid and board.region(r, "plan") >= 1.0 - _EPSILON
                        and board.region(r, "ours_present") <= 0 and board.region(r, "committed") <= 0):
                    return {r: 1.0}
        ours = [r for r in live if r != avoid and board.region(r, "held_by_us") > 0 and board.region(r, "resources") > 0]
        if not ours:
            ours = [r for r in live if board.region(r, "held_by_us") > 0] or [home]
        return {r: (board.region(r, "priority") + RESOURCE_WORTH * board.region(r, "resources")
                    + THREAT_WEIGHT * board.region(r, "enemy_present")
                    - CROWDING_COST * board.region(r, "ours_present") - self._reach(board, r)) for r in ours}

    def _raid(self, board: _Board, live: List[int], avoid: Optional[int]):
        """Something that pays and that nothing is guarding."""
        candidates = [r for r in live if r != avoid and board.region(r, "resources") > 0 and board.region(r, "held_by_us") <= 0]
        if not candidates:
            candidates = [r for r in live if r != avoid and board.region(r, "held_by_us") <= 0] or live
        return {r: (board.region(r, "priority") + RESOURCE_WORTH * board.region(r, "resources")
                    - RAID_RISK * board.region(r, "enemy_present") - self._reach(board, r)) for r in candidates}

    def _vanguard(self, board: _Board, live: List[int], avoid: Optional[int], home: int):
        """The contested region worth the most less the march there; with `concentrate`, only one the squad would win together with what is bound there and what of ours stands there, drawn towards where others are bound, and a rally point behind the front when there is none."""
        contested = [r for r in live if r != avoid and (board.region(r, "enemy_present") > 0 or board.region(r, "held_by_enemy") > 0)]
        if board.glob("offensive") <= 0:
            contested = [r for r in contested if board.region(r, "held_by_us") > 0]
        if not contested:
            contested = [r for r in live if r != avoid and board.region(r, "held_by_us") <= 0]
        if not self.concentrate:
            if not contested:
                return {home: 1.0}, Task.ATTACK
            scores = {r: (board.region(r, "priority") - self._reach(board, r)
                          - CROWDING_COST * board.region(r, "ours_present")) for r in contested}
            best = max(scores, key=lambda r: (scores[r], -board.region(r, "distance")))
            surround = board.region(best, "enemy_present") > 0 and board.region(best, "force_edge") >= ENCIRCLE_SHARE
            return scores, (Task.ENCIRCLE if surround else Task.ATTACK)

        knobs = self.tuning
        winnable = [r for r in contested if board.region(r, "predicted") >= knobs.attack_edge]
        if winnable:
            scores = {r: (board.region(r, "priority") - self._reach(board, r)
                          + knobs.concentration * board.region(r, "committed")
                          + knobs.predict_weight * board.region(r, "predicted"))
                      for r in winnable}
            best = max(scores, key=lambda r: (scores[r], -board.region(r, "distance")))
            surround = board.region(best, "enemy_present") > 0 and board.region(best, "predicted") >= knobs.encircle_edge
            return scores, (Task.ENCIRCLE if surround else Task.ATTACK)
        # Nowhere can be won alone: gather where our strength already is and others are bound, as far forward as that ground goes.
        rallies = [r for r in live if r != avoid and (board.region(r, "held_by_us") > 0 or board.region(r, "ours_present") > 0)] or [home]
        scores = {r: (knobs.concentration * board.region(r, "committed") + board.region(r, "ours_present")
                      + knobs.forward * board.region(r, "distance") - self._reach(board, r)) for r in rallies}
        return scores, Task.ATTACK


# ---- the economic judge -------------------------------------------------------------------

#: Extractors that have to be standing before the first factory is worth 700 credits.
FACTORY_EXTRACTORS = 2

#: Military share above which the factory is worth running at all. Opening value, set below the lowest share any posture carries so that even an expanding opening keeps something in the field.
MILITARY_SHARE_THRESHOLD = 0.15

#: Economy share above which a second builder is worth its 500 credits rather than another extractor. Opening value: below the arming postures rather than between them, so that only a final battle stops us keeping two.
#:
#: Set above the arming share it reads as "stop expanding", which is not what arming means. A match turns to arming early, and with the threshold above that share the whole of it is then played on the one builder it started with: nothing is left to raise a second factory while the first is busy, or to replace the builder when it dies, and the economy stops growing at the moment the army starts costing.
EXPANSION_SHARE_THRESHOLD = 0.25

#: Share above which ground that has seen the enemy is worth a turret. Opening value; read off the military share for want of a defensive one, so only the postures that are arming fortify.
DEFENCE_SHARE_THRESHOLD = 0.45

#: Income, in the engine's own units, below which a factory is not taken off production to raise its tier. Opening value.
FACTORY_UPGRADE_INCOME = 25.0

#: How much more a factory's next tier has to be worth, by the best thing it would then make, before the factory is taken off production to raise it. Opening value.
RAISE_GAIN = 0.1

_CONTEXT = {name: index for index, name in enumerate(ECONOMIC_CONTEXT)}
_OFFER = {name: index for index, name in enumerate(INVESTMENT_FEATURES)}

#: How far apart two rungs of the build order score, and the most the order within one rung adds. The order within a rung stays below the gap, so no rung is ever overtaken by the one beneath it.
_RUNG_SPAN = 0.9


class EconomyJudge:
    """The build order, read off the economic features: which investment comes next.

    The build order is a ladder of rungs read top down, each with its gate, and the investment taken is the first one a rung lets through. That is written here as a score per slot: a rung scores above every rung beneath it, the order within a rung adds less than the gap between two rungs, stopping scores nought, and an offer no rung lets through scores below stopping. The rungs, highest first, are rebuilding a lost factory, an extractor at home, the first factory, a builder, a transport when one is wanted, the expansion plan's ground with ground a builder already stands on saved for (when switched to claim it first), a tier raise, the second factory, the army, another factory for a banked treasury, the rest of the plan's ground and a turret. What the switches of `options` turn off is turned off here.

    Choosing an offer the treasury cannot pay for yet is how saving for it is said, and three rungs do it: a tier raise, the second factory, and a unit the army is better off waiting for.
    """

    def __init__(self, options: Options = Options()) -> None:
        self.options = options
        self.tuning = options.tuning

    def scores(self, state: Sequence[float]) -> List[float]:
        rungs = self._rungs(state)
        scores = [-math.inf] * INVESTMENT_SLOTS
        for slot in range(INVESTMENT_SLOTS):
            if self._row(state, slot)("valid") > 0:
                scores[slot] = -1.0
        scores[0] = 0.0
        for rank, members in enumerate(rungs):
            base = float(len(rungs) - rank)
            for position, slot in enumerate(members):
                scores[slot] = max(scores[slot], base + _RUNG_SPAN * (1.0 - position / len(members)))
        return scores

    def choose(self, state: Sequence[float]) -> int:
        scores = self.scores(state)
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    @staticmethod
    def _row(state: Sequence[float], slot: int) -> Callable[[str], float]:
        start = ECONOMIC_CONTEXT_SIZE + slot * INVESTMENT_SIZE
        return lambda name: state[start + _OFFER[name]]

    def _rungs(self, state: Sequence[float]) -> List[List[int]]:
        """Every rung of the build order, highest first, each as the slots it lets through in the order it prefers them."""
        def c(name: str) -> float:
            return state[_CONTEXT[name]]

        rows = {slot: self._row(state, slot) for slot in range(1, INVESTMENT_SLOTS)}
        rows = {slot: row for slot, row in rows.items() if row("valid") > 0}

        def of(kind: Investment) -> List[int]:
            return [slot for slot, row in rows.items() if row(f"kind_{kind.name.lower()}") > 0]

        def affordable(slot: int) -> bool:
            return rows[slot]("affordable") > 0

        knobs = self.tuning
        built = round(c("factories") * COUNT_SCALE)
        pending = c("pending_factory") > 0
        rebuild = self.options.recovery and c("rebuilding") > 0 and built == 0
        income = c("income") * INCOME_SCALE
        budget = c("budget") * TREASURY_SCALE
        near_cap = c("near_cap") > 0
        land = [s for s in of(Investment.FACTORY) if rows[s]("land_factory") > 0 and affordable(s)]
        extractors = of(Investment.EXTRACTOR)
        home = [s for s in extractors if rows[s]("home") > 0 and affordable(s)]
        plan = sorted((s for s in extractors if rows[s]("home") <= 0 and affordable(s)),
                      key=lambda s: (rows[s]("contested"), rows[s]("safety"), s))
        best_factory = sorted(of(Investment.FACTORY),
                              key=lambda s: (-rows[s]("worth"), -rows[s]("land_factory"), s))[:1]

        rungs: List[List[int]] = []
        rungs.append(land[:1] if rebuild else [])
        rungs.append(home)
        rungs.append(land[:1] if built == 0 and not rebuild
                     and round(c("extractors") * 2 * COUNT_SCALE) >= FACTORY_EXTRACTORS else [])
        builders = [s for s in of(Investment.BUILDER) if affordable(s)]
        wanted = c("builder_shortfall") > 0 and (c("economy_share") >= EXPANSION_SHARE_THRESHOLD or c("builders") <= 0)
        rungs.append(builders if wanted else [])
        # A transport when ground or a squad is waiting on one and none can serve, once a factory stands: without one nothing ever leaves an island.
        transports = sorted((s for s in of(Investment.UNIT) if rows[s]("transport") > 0 and affordable(s)),
                            key=lambda s: (rows[s]("price"), s))
        rungs.append(transports[:1] if c("transport_wanted") > 0 and built >= 1 else [])
        # Ground a builder is already standing on is saved for when it cannot be paid yet: that builder was sent there, often carried across water, for nothing else.
        waiting = sorted((s for s in extractors if rows[s]("home") <= 0 and rows[s]("ready") > 0 and not affordable(s)),
                         key=lambda s: (rows[s]("contested"), rows[s]("safety"), s))
        rungs.append(plan + waiting[:1] if self.options.expand_first else [])
        rungs.append(self._raise(rows, of(Investment.RAISE), c, income, near_cap))
        second = built == 1 and income >= knobs.second_factory_income and not pending and not rebuild
        rungs.append(best_factory if second else [])
        rungs.append(self._army(rows, of(Investment.UNIT), near_cap)
                     if c("military_share") >= MILITARY_SHARE_THRESHOLD else [])
        banked = (1 <= built < int(knobs.max_factories) and budget >= knobs.banked_credits and not near_cap
                  and not pending)
        rungs.append([s for s in best_factory if affordable(s)] if banked else [])
        rungs.append(plan)
        rungs.append([s for s in of(Investment.TURRET) if affordable(s)]
                     if c("military_share") >= DEFENCE_SHARE_THRESHOLD else [])
        return rungs

    def _raise(self, rows, raises: List[int], c, income: float, near_cap: bool) -> List[int]:
        """The one tier raise the build order takes, at the lowest stage it will consider, or none. A factory is raised only while another keeps producing or the unit cap has stopped production anyway, with the income to spare, one at a time, and when switched to choose only for a next tier worth RAISE_GAIN more or near the cap. The raise is saved for when it cannot be paid for yet, unless saving is switched off, it costs more than `tuning.upgrade_saving`, or it is an extractor's third tier while our army is the smaller."""
        def considered(slot: int) -> bool:
            row = rows[slot]
            if row("stage_1") <= 0:
                return True
            if c("factory_raising") > 0 or income < FACTORY_UPGRADE_INCOME:
                return False
            if round(c("factories") * COUNT_SCALE) < 2 and c("at_cap") <= 0:
                return False
            return not self.options.choose or near_cap or row("gain") > RAISE_GAIN

        staged = sorted((s for s in raises if considered(s)),
                        key=lambda s: [rows[s](f"stage_{stage}") for stage in range(RAISE_STAGES)], reverse=True)
        if not staged:
            return []
        slot = staged[0]
        row = rows[slot]
        if row("affordable") > 0:
            return [slot]
        if not self.options.saving or row("price") * PRICE_SCALE > self.tuning.upgrade_saving:
            return []
        if row("stage_2") > 0 and c("army_edge") < 0.5:
            return []
        return [slot]

    def _army(self, rows, units: List[int], near_cap: bool) -> List[int]:
        """What the next idle factory makes, and after it what it would make instead.

        When switched to choose: the unit within reach that buys the most fighting strength against what the enemy fields, per credit or per unit near the cap, the cheaper and then the earlier slot winning ties. When the treasury cannot pay for it now, a unit it can pay for is made if it is worth `tuning.save_share` of it, and otherwise the factory waits for it, which is choosing it unpaid.

        Otherwise the roles in the order the target mix asks for them: the cheapest of the first role whose cheapest can be paid for, or near the cap the dearest of the first role that has one the budget covers.
        """
        if self.options.choose:
            key = "unit_efficiency" if near_cap else "efficiency"
            reach = [s for s in units if rows[s]("fighting") > 0 and rows[s]("reach") > 0]
            if not reach:
                return []
            ranked = sorted(reach, key=lambda s: (-rows[s](key), rows[s]("price"), s))
            best = ranked[0]
            paid = [s for s in ranked if rows[s]("affordable") > 0]
            if rows[best]("affordable") > 0:
                return paid
            if paid and rows[paid[0]](key) >= self.tuning.save_share * rows[best](key):
                return paid + [best]
            return [best] + paid
        fielded = sorted({rows[s]("role_rank") for s in units if rows[s]("role_rank") > 0}, reverse=True)
        for rank in fielded:
            members = [s for s in units if rows[s]("role_rank") == rank]
            flag = "priciest_fitting" if near_cap else "cheapest_of_role"
            chosen = [s for s in members if rows[s](flag) > 0 and rows[s]("affordable") > 0]
            if chosen:
                return chosen[:1]
        return []


#: Opening value: the temperature a teacher's scores are softened at before they are written down. The tactical ladder's rungs are a whole point apart, so at this temperature the rung after the one taken keeps a small share and the rest next to nothing; the operational scores are closer together and keep more.
TEACHER_TEMPERATURE = 0.25


def distribution(scores: Sequence[float], temperature: float) -> List[float]:
    """The scores softened into a distribution over the actions they allow: a softmax at the temperature, with forbidden actions, scored negative infinity, at nought."""
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return [0.0] * len(scores)
    top = max(finite)
    weights = [math.exp((s - top) / max(temperature, 1e-6)) if math.isfinite(s) else 0.0 for s in scores]
    total = sum(weights)
    return [w / total for w in weights]
