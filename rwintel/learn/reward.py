"""What each layer is paid for.

The tactical layer is paid for how well it met the contract its superior handed down: that is what shortens its horizon to the length of one errand, which is the only horizon this sample budget can learn a fight over. The operational layer and the economy are paid the match: the score the match ends with (`rwintel.eval.scoring.score`, the same function every evaluation reads), paid once when it ends, so that the return of their decisions is the quantity they are judged by. The strategic layer's orders are not in their pay; met or not, they do not move the score.

Every layer is shaped potentially. A shaping term of the form gamma times the potential after minus the potential before cannot change which policy is optimal, whatever the potential is, so a term that turns out to have been a bad idea costs sample efficiency and never correctness. The potential of the state a trajectory ends in is taken to be nought: the last shaping term is nought less what was being held, which is what makes the shaping telescope to the difference of two potentials.

The match layers are shaped by the score expected from the board (`predictor`), at a discount of one. A decision is then paid the change in the expected score over the periods it stood, the match's end pays the score less the last expectation, and the payments over a match add up to its score less the expectation it opened on.

Every payment is made in two steps: the board is read into a signal, a row of the raw quantities the payment depends on, and the row is priced by a pure function of the run's terms (`TacticalTerms`, `OperationalTerms`, `EconomicTerms`). The rows are what a dataset keeps beside each decision, so a payment can be priced again offline under other terms by the same function the layer paid with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Optional, Sequence, Tuple

from ..wire import Status
from ..control.policy.contracts import SquadRecord
from ..control.policy.view import WorldView
from .predictor import Predictor, default_predictor

#: What the tactical potential is made of: standing on the ground the contract named, not having spent the allowance, and having destroyed more than has been lost.
#:
#: The exchange weighs most, and the reason is credit assignment rather than importance. In a match an errand is a fragment discounted per decision, so nothing the layer decides in the opening of a fight is taught by how the fight ended; what teaches it is the shaping, and the shaping only teaches the right thing if it points where the terminal points. Trading well is what the terminal is, so trading well is what the dense term has to be.
HOLDING_WEIGHT = 0.35
SPENDING_WEIGHT = 0.15
EXCHANGE_WEIGHT = 0.5

#: Credits added to both sides of the exchange ratio before it is taken, so that a fight in which nothing has happened yet reads as even rather than swinging to one end on the first shot. A tank costs about this, which makes the first kill worth about two thirds rather than all of the term.
EXCHANGE_PRIOR = 300.0

#: Paid once, when an errand ends. Taking and holding the region is the errand done; running out of time is a failure that at least did not cost anything in particular; being reported as losing means the budget went and the region did not come, which is the outcome the whole cost budget mechanism exists to make expensive.
COMPLETE_REWARD = 1.0
EXPIRED_REWARD = -0.5
LOSING_REWARD = -1.0

#: Paid when a squad ceases to exist while under contract. Distinguished from being reported as losing because a squad that is gone cannot be withdrawn and the layer had every period until then to withdraw it.
WIPED_REWARD = -1.0

#: How much over budget an errand may run before the overspend is charged against a completion. A region taken at twice its allowance was not taken at the price it was commissioned at.
OVERSPEND_PENALTY = 0.5

#: Discount for an errand that is a fragment of a match. A constructed fight is discounted at what the rollout hands in instead.
DISCOUNT = 0.99

#: Scoring a fight on what is left standing: a unit counts for the whole of its price until the moment it dies and for nothing after, so damage short of a kill is invisible.
BY_KILLS = "kills"

#: Scoring a fight on what is left standing weighted by how much of it is left, so that a unit at a tenth of its health counts for a tenth of its price. Most fights are called with two damaged forces still standing, which the sparse reading scores at nought whatever the damage was.
BY_HEALTH = "health"

SCORES = (BY_KILLS, BY_HEALTH)

#: How much of a terminal reward the outcome of a fight is worth, for a fight that ended without the contract itself reaching one of its own conclusions. One means that destroying the other side without a scratch is paid exactly what taking the contracted ground is paid, which is the largest this can be set to without teaching a layer to prefer a massacre to the errand it was given.
OUTCOME_WEIGHT = 1.0

#: Discount per period of the match layers' default reward. The shaping is the change in the expected score, so nothing is lost by not discounting, and the match score reaches every decision taken in the match whole.
MATCH_DISCOUNT = 1.0

#: How much of the match score the end of a match pays. One makes the return of a decision the score less the expectation it was taken on.
TERMINAL_WEIGHT = 1.0

#: The two potentials a match layer can be shaped by: the expected score (`predictor`), or our share of the worth on the board times `share_weight`.
PREDICTED = "predicted"
SHARE = "share"
SHAPINGS = (PREDICTED, SHARE)

#: Weight of the worth share when it is the potential.
SHARE_WEIGHT = 0.4

#: What the operational layer is paid each period, under `achievement_weight`, for the ground the strategic layer asked for: the priority-weighted control of those regions, from -1 (all of it the enemy's) to +1 (all of it ours).
ACHIEVEMENT_RATE = 0.05

#: Credits added to both sides of a region's force balance, so that an empty region reads as nobody's rather than swinging to one side on the first unit that walks in. About one tank.
CONTROL_PRIOR = 300.0

#: How much the extractors standing in a region count towards controlling it, against the force balance. Ground is held by what draws income from it as much as by what stands on it.
EXTRACTOR_WEIGHT = 0.5

#: Charged each period, under `achievement_weight`, for every whole allowance the missions have lost beyond the strategic layer's loss allowance, on the same scale as the achievement flow.
OVERSPEND_WEIGHT = 0.05

#: Credits of exchange that make one unit of the operational layer's local exchange payment: about three tanks.
EXCHANGE_SCALE = 1000.0


def _share(part: float, against: float) -> float:
    total = part + against
    return part / total if total > 0 else 0.5


def value_share(view: WorldView) -> float:
    """Our share of the worth standing on the board, units and buildings of both sides alike, which is the quantity the match score reads."""
    return _share(sum(s.value for s in view.ours), sum(s.value for s in view.enemies))


def ground_share(view: WorldView) -> float:
    """Our share of the resource points either side holds."""
    return _share(sum(r.held_by_us for r in view.regions), sum(r.held_by_enemy for r in view.regions))


# ---- the signals ---------------------------------------------------------------------------

#: What a tactical signal row holds, in order. One row is one payment made to one decision; a decision closed after its own payment was made (a squad destroyed between periods) carries a second row for the ending.
TACTICAL_SIGNALS = ("kind", "holding_before", "unspent_before", "exchange_before",
                    "holding_after", "unspent_after", "exchange_after",
                    "reason", "overrun", "outcome_kills", "outcome_health")

#: The kinds of tactical payment: nothing, a shaping difference, an ending the contract's own conditions reached, and an ending the arena called from outside.
NOTHING, SHAPING, ENDED, CALLED = 0, 1, 2, 3

#: Why an errand ended, as the codes a tactical row carries. The empty name is the row of a payment that ended nothing.
REASONS = ("", "complete", "expired", "losing", "wiped", "called")

#: What an operational signal row holds: one row per period the decision stood, in order, the nth discounted by the discount to the nth, and a last row when the match ended under it. The remaining time is in game seconds, -1 where it is not known.
OPERATIONAL_SIGNALS = ("kind", "remaining_before", "remaining_after", "achievement", "overspend",
                       "value_before", "value_after", "ground_before", "ground_after", "exchange", "score")

#: What an economic signal row holds, as for the operational layer.
ECONOMIC_SIGNALS = ("kind", "remaining_before", "remaining_after", "value_before", "value_after",
                    "ground_before", "ground_after", "score")

#: The kinds of a match layer's row: the opening period of a match, which has no board before it and pays no shaping; a period; and the end of the match, which pays the score and reads only the board before it.
OPENED, ELAPSED, CLOSED = 0, 1, 2

#: Potential components, as (holding, unspent, exchange).
Components = Tuple[float, float, float]

#: A match layer's board, as (remaining seconds, worth share, ground share).
Board = Tuple[float, float, float]

_NONE: Components = (0.0, 0.0, 0.0)


def tactical_row(kind: int, before: Components = _NONE, after: Components = _NONE, reason: str = "",
                 overrun: float = 0.0, kills: float = 0.0, health: float = 0.0) -> Tuple[float, ...]:
    return (float(kind), *before, *after, float(REASONS.index(reason)), overrun, kills, health)


def operational_row(kind: int, before: Board, after: Board, achievement: float = 0.0, overspend: float = 0.0,
                    exchange: float = 0.0, score: float = 0.0) -> Tuple[float, ...]:
    return (float(kind), before[0], after[0], achievement, overspend, before[1], after[1], before[2], after[2],
            exchange, score)


def economic_row(kind: int, before: Board, after: Board, score: float = 0.0) -> Tuple[float, ...]:
    return (float(kind), before[0], after[0], before[1], after[1], before[2], after[2], score)


# ---- the terms a run is paid under ---------------------------------------------------------

@dataclass(frozen=True)
class TacticalTerms:
    """The figures a tactical row is priced at. Everything a run may change is here, including the discount the shaping telescopes with and which reading of a fight is paid."""

    discount: float = DISCOUNT
    holding_weight: float = HOLDING_WEIGHT
    spending_weight: float = SPENDING_WEIGHT
    exchange_weight: float = EXCHANGE_WEIGHT
    complete: float = COMPLETE_REWARD
    expired: float = EXPIRED_REWARD
    losing: float = LOSING_REWARD
    wiped: float = WIPED_REWARD
    overspend_penalty: float = OVERSPEND_PENALTY
    outcome_weight: float = OUTCOME_WEIGHT
    score: str = BY_HEALTH

    def __post_init__(self) -> None:
        # A misspelt reading would leave a run reporting both readings while being paid on the one nobody asked for, and nothing in its logs would say so.
        if self.score not in SCORES:
            raise ValueError(f"no score named {self.score!r}: expected one of {', '.join(SCORES)}")

    def potential(self, components: Sequence[float]) -> float:
        holding, unspent, exchange = components
        return self.holding_weight * holding + self.spending_weight * unspent + self.exchange_weight * exchange

    def terminal(self, reason: str, overrun: float) -> float:
        if reason == "complete":
            return self.complete - self.overspend_penalty * overrun
        return {"expired": self.expired, "losing": self.losing, "wiped": self.wiped}.get(reason, 0.0)

    def pay(self, row: Sequence[float]) -> float:
        kind = int(row[0])
        before, after = row[1:4], row[4:7]
        if kind == SHAPING:
            return self.discount * self.potential(after) - self.potential(before)
        if kind == ENDED:
            return (0.0 - self.potential(before)) + self.terminal(REASONS[int(row[7])], row[8])
        if kind == CALLED:
            outcome = row[10] if self.score == BY_HEALTH else row[9]
            return (0.0 - self.potential(before)) + self.outcome_weight * outcome
        return 0.0

    def reward(self, rows: Sequence[Sequence[float]]) -> float:
        """What a decision with these rows was paid: the sum of its payments, which follow each other within one period."""
        return sum(self.pay(row) for row in rows)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MatchTerms:
    """What the operational layer and the economy share: the discount, the score at the end of the match, the potential, and flows of the two shares paid each period on top."""

    discount: float = MATCH_DISCOUNT
    terminal_weight: float = TERMINAL_WEIGHT
    shaping: str = PREDICTED
    share_weight: float = SHARE_WEIGHT
    #: Paid each period per unit of the worth share and of the ground share, each written from -1 to +1. Nought by default: the score at the end is the pay, and a flow is a second objective beside it.
    value_flow: float = 0.0
    ground_flow: float = 0.0
    predictor: Predictor = field(default_factory=default_predictor)

    def __post_init__(self) -> None:
        if self.shaping not in SHAPINGS:
            raise ValueError(f"no shaping named {self.shaping!r}: expected one of {', '.join(SHAPINGS)}")
        if isinstance(self.predictor, dict):
            object.__setattr__(self, "predictor", Predictor.from_dict(self.predictor))

    def potential(self, remaining: float, value: float, ground: float) -> float:
        if self.shaping == PREDICTED:
            return self.predictor.potential(remaining, value, ground)
        return self.share_weight * value

    def flows(self, value: float, ground: float) -> float:
        return self.value_flow * (2.0 * value - 1.0) + self.ground_flow * (2.0 * ground - 1.0)

    def board_pay(self, kind: int, before: Board, after: Board, score: float) -> float:
        """The part of a row's payment that is the same for both layers: flows and shaping over a period, the score less the potential held at the end of the match."""
        if kind == CLOSED:
            return self.terminal_weight * score - self.potential(*before)
        if kind == ELAPSED:
            return self.flows(after[1], after[2]) + self.discount * self.potential(*after) - self.potential(*before)
        return 0.0

    def reward(self, rows: Sequence[Sequence[float]]) -> float:
        """What a decision that stood over these rows was paid, each discounted by how many periods the decision had already stood. The end of a match reads the board the last period ended on, so it is discounted as that board's potential was and the shaping cancels against it exactly."""
        paid = 0.0
        periods = 0
        for row in rows:
            paid += self.discount ** periods * self.pay(row)
            if int(row[0]) != CLOSED:
                periods += 1
        return paid

    def pay(self, row: Sequence[float]) -> float:
        raise NotImplementedError

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["predictor"] = self.predictor.as_dict()
        return out


@dataclass(frozen=True)
class OperationalTerms(MatchTerms):
    """The figures an operational row is priced at."""

    #: A multiplier on the pay for the strategic layer's orders: their achievement as a flow, less a charge for losses beyond their allowance. Nought by default, since the orders being met does not move the match score.
    achievement_weight: float = 0.0
    achievement_rate: float = ACHIEVEMENT_RATE
    overspend_weight: float = OVERSPEND_WEIGHT
    #: How much a decision is also paid its own squad's exchange, worth destroyed less worth lost per EXCHANGE_SCALE credits, over the periods it stands.
    local_exchange: float = 0.0

    def pay(self, row: Sequence[float]) -> float:
        kind = int(row[0])
        before, after = (row[1], row[5], row[7]), (row[2], row[6], row[8])
        paid = self.board_pay(kind, before, after, row[10]) + self.local_exchange * row[9] / EXCHANGE_SCALE
        if kind == ELAPSED and self.achievement_weight:
            paid += self.achievement_weight * (self.achievement_rate * row[3] - self.overspend_weight * row[4])
        return paid


@dataclass(frozen=True)
class EconomicTerms(MatchTerms):
    """The figures an economic row is priced at."""

    def pay(self, row: Sequence[float]) -> float:
        kind = int(row[0])
        return self.board_pay(kind, (row[1], row[3], row[5]), (row[2], row[4], row[6]), row[7])


#: The named sets of terms a match layer's run can be paid under: the default, and the flows of the two shares that the default replaced, kept as the comparison it is measured against.
REWARDS = ("predicted", "flows")

#: Discount and trace of each named set, per layer.
_FLOWS: Dict[str, Tuple[dict, float]] = {
    "operations": (dict(discount=0.95, shaping=SHARE, value_flow=0.05, local_exchange=0.1), 0.9),
    "economy": (dict(discount=0.99, shaping=SHARE, value_flow=0.05, ground_flow=0.02), 0.95),
}

#: Trace of the default terms.
MATCH_TRACE = 0.95


def preset(layer: str, name: str) -> Tuple[dict, float]:
    """The terms a named reward sets for a match layer, as keywords of its terms, and the trace it is collected at."""
    if layer not in _FLOWS:
        raise ValueError(f"the named rewards are for the operational layer and the economy, not {layer!r}")
    if name == "predicted":
        return {}, MATCH_TRACE
    if name == "flows":
        terms, trace = _FLOWS[layer]
        return dict(terms), trace
    raise ValueError(f"no reward named {name!r}: expected one of {', '.join(REWARDS)}")


# ---- the tactical layer --------------------------------------------------------------------

@dataclass
class Outcome:
    """One period's pay for one decision, the signal it was priced from, and whether the thing it was paid for has finished."""

    reward: float = 0.0
    done: bool = False
    #: What ended the errand, for a log that has to be read by a person rather than by an optimiser.
    reason: str = ""
    #: True on the period a new contract replaced the one a squad was working to. The errand that was replaced did not fail and did not finish; it stopped being the thing the squad is doing, so a caller keeping trajectories has to end the old one here rather than letting the decisions of two errands sit in one.
    renewed: bool = False
    #: The signal row the payment was priced from.
    signal: Tuple[float, ...] = ()


@dataclass
class _Mission:
    """A tactical layer's memory of one errand, which is the span its reward is cut across."""

    issued_at_ms: int
    components: Components = _NONE
    potential: float = 0.0
    #: True once this errand has been paid its one terminal, so that it cannot be paid another.
    ended: bool = False


class TacticalReward:
    """Pays the tactical layer for the errand it was given, one squad at a time.

    An errand is bounded by its contract. When the operational layer issues a new one the old episode ends and a new one begins, which is exactly the boundary the design wants the tactical horizon to close at: ten to sixty seconds, not a match. That the boundary is drawn by another layer's decision is not a problem to be solved but the arrangement itself -the contract is the unit of work, so it is the unit of pay.

    An errand is paid one terminal and no more. The memory of a finished errand is therefore kept rather than dropped, keyed by the moment its contract was issued: the conditions that end one -the ground taken, the allowance gone, the deadline past -are read from the board and stay true for as long as the board stays that way, so an errand whose memory was dropped would be started afresh on the next period, meet the same condition, and be paid again.

    Where an errand ends is not the same question everywhere it is used. In a match the operational layer reissues contracts, so the conditions of the contract are what bound the errand and it is right that they end it. On a board where engagements are constructed there is one contract for the whole fight and nothing reissues it, so ending the errand early would leave the rest of the fight unpaid while the squad went on fighting it. `status_terminals` is which of the two this is.
    """

    def __init__(self, status_terminals: bool = True, discount: float = DISCOUNT,
                 terms: Optional[TacticalTerms] = None) -> None:
        self.missions: Dict[int, _Mission] = {}
        self.status_terminals = status_terminals
        # The discount the shaping telescopes with has to be the one the returns are discounted at, and the two places this layer is trained have errands of different lengths, so it is handed in with the rest of the terms.
        self.terms = terms if terms is not None else TacticalTerms(discount=discount)

    @property
    def discount(self) -> float:
        return self.terms.discount

    def forget(self, squad_id: int) -> None:
        self.missions.pop(squad_id, None)

    def ended(self, squad_id: int) -> bool:
        """Whether this squad's errand has already been paid its terminal, which is what a caller ending errands from outside has to ask before paying another."""
        mission = self.missions.get(squad_id)
        return mission is not None and mission.ended

    def call(self, squad_id: int, kills: float, health: float) -> Outcome:
        """Ends this squad's errand from outside on the score of the fight it was in, both readings of it, and forgets the errand.

        The last shaping term is paid as nought less the potential held, which is what makes the shaping over the errand telescope away. A squad whose errand ended in the period it was given holds nothing, so the payment is then the score alone.
        """
        mission = self.missions.pop(squad_id, None)
        held = mission.components if mission is not None else _NONE
        row = tactical_row(CALLED, before=held, reason="called", kills=kills, health=health)
        return Outcome(reward=self.terms.pay(row), done=True, reason="called", signal=row)

    def step(self, squad: SquadRecord, view: WorldView, game_time_ms: int,
             killed: float = 0.0) -> Outcome:
        contract = squad.contract
        if contract is None:
            self.forget(squad.id)
            return Outcome(signal=tactical_row(NOTHING))

        mission = self.missions.get(squad.id)
        if mission is None or mission.issued_at_ms != contract.issued_at_ms:
            # A fresh contract is a fresh errand. The potential is taken now and paid from the next period, so that the step which merely received the contract is not paid for the board it arrived on.
            replaced = mission is not None
            components = self.components(squad, view, killed)
            self.missions[squad.id] = _Mission(issued_at_ms=contract.issued_at_ms, components=components,
                                               potential=self.terms.potential(components))
            return Outcome(renewed=replaced, signal=tactical_row(NOTHING))

        # An errand that has already been paid its terminal is over. Decisions taken about the squad afterwards belong to no errand until a contract issues a new one, and paying them would be paying for work nobody asked for.
        if mission.ended:
            return Outcome(signal=tactical_row(NOTHING))

        reason, overrun = self._terminal(squad, contract)
        if reason:
            # The potential of a state an errand ended in is nought by convention, so the last shaping term is nought less whatever was being held rather than the discounted potential of the ending board. Paying the real potential would leave a residue proportional to it, the residue would differ between one ending and another, and a shaping term that differs by outcome is a term that moves which policy is best.
            mission.ended = True
            row = tactical_row(ENDED, before=mission.components, reason=reason, overrun=overrun)
            return Outcome(reward=self.terms.pay(row), done=True, reason=reason, signal=row)

        components = self.components(squad, view, killed)
        row = tactical_row(SHAPING, before=mission.components, after=components)
        mission.components, mission.potential = components, self.terms.potential(components)
        return Outcome(reward=self.terms.pay(row), signal=row)

    @staticmethod
    def components(squad: SquadRecord, view: WorldView, killed: float = 0.0) -> Components:
        """The three parts of the potential: our share of the force on the contracted ground, the share of the allowance not yet lost, and worth destroyed against worth lost over the life of the contract with a unit's price added to both so that nothing having happened reads as even."""
        contract = squad.contract
        target = view.region(contract.target_region) if contract is not None else None
        holding = _share(target.our_value, target.enemy_value) if target is not None else 0.5
        budget = contract.cost_budget if contract is not None else 0.0
        spent = min(1.0, squad.losses / budget) if budget > 0 else 0.0
        exchange = _share(killed + EXCHANGE_PRIOR, squad.losses + EXCHANGE_PRIOR)
        return holding, 1.0 - spent, exchange

    def _terminal(self, squad: SquadRecord, contract) -> Tuple[str, float]:
        """Which of the contract's own endings the squad has reached, and by how much a completion overran its allowance, from nought to one."""
        # Where one contract stands for a whole fight, what ends the fight ends the errand and nothing else does. A squad destroyed is no exception: what it is worth depends on how much it took with it, and the only figure that knows that is the score of the fight, which is handed in from outside.
        if not self.status_terminals:
            return "", 0.0
        if not squad.members:
            return "wiped", 0.0
        if squad.status is Status.COMPLETE:
            overrun = 0.0
            if contract.cost_budget > 0 and squad.losses > contract.cost_budget:
                overrun = min(1.0, squad.losses / contract.cost_budget - 1.0)
            return "complete", overrun
        if squad.status is Status.LOSING:
            return "losing", 0.0
        if squad.status is Status.EXPIRED:
            return "expired", 0.0
        return "", 0.0


# ---- the match layers ----------------------------------------------------------------------

def control(region) -> float:
    """How much a region is ours, from -1 to +1: the balance of force standing in it, plus the balance of extractors drawing from it."""
    force = (region.our_value - region.enemy_value) / (region.our_value + region.enemy_value + CONTROL_PRIOR)
    pools = max(1, region.resources)
    held = EXTRACTOR_WEIGHT * (region.held_by_us - region.held_by_enemy) / pools
    return max(-1.0, min(1.0, force + held))


class _MatchReward:
    """The board a match layer was last paid from, so that each period's row reads the board before it and the end of the match reads the last."""

    def __init__(self, terms: MatchTerms) -> None:
        self.terms = terms
        self.board: Optional[Board] = None

    @property
    def discount(self) -> float:
        return self.terms.discount

    def reset(self) -> None:
        self.board = None

    def _advance(self, view: WorldView, remaining: float) -> Tuple[int, Board, Board]:
        board = (float(remaining), value_share(view), ground_share(view))
        before = board if self.board is None else self.board
        kind = OPENED if self.board is None else ELAPSED
        self.board = board
        return kind, before, board


class OperationalReward(_MatchReward):
    """Pays the operational layer, once per operational period, from the board, and once more when the match ends, the match score.

    One figure pays every decision standing in a period: the board is a statement about the whole match and not about any one squad. A squad's own exchange is added per decision where the terms ask for it.
    """

    def __init__(self, terms: Optional[OperationalTerms] = None) -> None:
        super().__init__(terms if terms is not None else OperationalTerms())

    def signal(self, view: WorldView, orders, squads: Sequence[SquadRecord], remaining: float = -1.0) -> Tuple[float, ...]:
        """The period that has just elapsed, as an operational row with no exchange in it; `with_exchange` adds a squad's own."""
        kind, before, after = self._advance(view, remaining)
        return operational_row(kind, before, after, achievement=self.achievement(view, orders),
                               overspend=self.overspend(orders, squads))

    def end(self, score: float) -> Optional[Tuple[float, ...]]:
        """The row the end of the match pays, or None when no period was ever read."""
        if self.board is None:
            return None
        return operational_row(CLOSED, self.board, self.board, score=score)

    @staticmethod
    def with_exchange(row: Tuple[float, ...], exchange: float) -> Tuple[float, ...]:
        return row[:9] + (exchange,) + row[10:]

    @staticmethod
    def achievement(view: WorldView, orders) -> float:
        """Priority-weighted control of the regions the strategic layer asked for, from -1 to +1; nought when it asked for nothing."""
        priorities = orders.priorities if orders is not None else {}
        wanted = 0.0
        held = 0.0
        for region in view.regions:
            priority = priorities.get(region.id, 0.0)
            if priority > 0:
                wanted += priority
                held += priority * control(region)
        return held / wanted if wanted > 0 else 0.0

    @staticmethod
    def overspend(orders, squads: Sequence[SquadRecord]) -> float:
        """Allowances lost beyond the first, which is nought while the missions keep within what the strategic layer said it would pay."""
        allowance = orders.loss_allowance if orders is not None else 0.0
        if allowance <= 0:
            return 0.0
        return max(0.0, sum(squad.losses for squad in squads) / allowance - 1.0)


class EconomicReward(_MatchReward):
    """Pays the economy, once per operational period, from the board its spending has turned into, and once more when the match ends, the match score."""

    def __init__(self, terms: Optional[EconomicTerms] = None) -> None:
        super().__init__(terms if terms is not None else EconomicTerms())

    def step(self, view: WorldView, remaining: float = -1.0) -> Outcome:
        """The payment for the period that has just elapsed. Nothing on the first period of a match, which has no period before it."""
        kind, before, after = self._advance(view, remaining)
        row = economic_row(kind, before, after)
        return Outcome(reward=self.terms.pay(row), signal=row)

    def end(self, score: float) -> Optional[Tuple[float, ...]]:
        """The row the end of the match pays, or None when no period was ever read."""
        if self.board is None:
            return None
        return economic_row(CLOSED, self.board, self.board, score=score)
