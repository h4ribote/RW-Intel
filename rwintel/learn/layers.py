"""The three learnt layers, each a script layer with one decision taken out and replaced.

Everything except the decision is inherited. The tactical layer's mission reports, its bookkeeping of what a squad has destroyed, the rule that a departure other than holding is re-issued every period; the operational layer's pricing of a mission against the strategic allowance, its deadlines, the rule that a contract is only re-issued when it differs from the one held; the economy's list of what could be bought, where a building goes and which builder places it - all of that is the same code running. What is overridden is one method each: which of the seven departures, which region under which task, and which investment comes next.

That is the design's own boundary and not a convenience. Layers are learnt one at a time against frozen neighbours, and a learnt layer must be substitutable for the script layer in the sense that the rest of the system cannot tell which is present. Inheriting rather than reimplementing is how that is guaranteed rather than hoped for: there is no second copy of the rules to drift.

Reward arrives one period late by construction. A decision cannot be paid until the board has moved under it, so each period pays the previous decision from the board that has now arrived, and only then makes a new one. This is the same one-period structure the interface already has for the action itself, and keeping the two aligned means a step in the buffer covers exactly one period of game time.

Whenever a decision is written down, the layer's own judge is asked about the same board and its answer is written beside what was played, so that every record -a network's, an exploring pupil's, a rule's- is also a record of what the script would have done there. When the rollout is recording, what the state was encoded from is kept as well (`materials`).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..wire import Deviation
from ..control.policy.contracts import EconomyOrders, MissionReport, OperationsOrders, Replacement, SquadRecord
from ..control.policy.economy import Economy
from ..control.policy.encoding import EconomicBoard, Offer
from ..control.policy.judgement import TEACHER_TEMPERATURE, distribution
from ..control.policy.operations import REVIEW_MS, Operations
from ..control.policy.options import Options
from ..control.policy.tactics import Tactics
from ..control.policy.view import Sighting, WorldView
from . import materials, tokens
from .deciders import Choice, one_hot
from .reward import (
    BY_HEALTH,
    DISCOUNT as REWARD_DISCOUNT,
    OUTCOME_WEIGHT,
    EconomicReward,
    EconomicTerms,
    OperationalReward,
    OperationalTerms,
    TacticalReward,
    TacticalTerms,
)
from .rollout import Rollout, Step

log = logging.getLogger(__name__)


def _best(scores: Sequence[float]) -> int:
    """The judge's answer from its scores, ties to the lower index as the judges break them."""
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def episode_of(session) -> dict:
    """What a sealed trajectory records about the episode it belongs to, as the session knows it. A session that is not a live game session -a replay playback, a test- answers with what it has."""
    records = getattr(session, "records", None) or []
    ending = getattr(session, "ending", "finished")
    settings = getattr(session, "settings", None)
    seed = getattr(settings, "seed", None)
    job = getattr(session, "job", None)
    return {"instance": int(getattr(session, "instance", -1)), "attempt": int(getattr(session, "attempt", 0)),
            "episode": len(records) + 1 if ending == "finished" else -1, "arm": str(getattr(session, "arm", "")),
            "map": str(getattr(session, "map_path", "") or ""),
            "order": int(getattr(session, "order", 0)),
            "seed": int(seed) + len(records) if isinstance(seed, int) else -1, "ending": ending,
            "replay": str(getattr(job, "stem", "")) if job is not None else ""}


def context_of(layer) -> dict:
    """The type table and the combat table a layer encoded its states against, as plain data, which is what its recorded materials are encoded again against."""
    from dataclasses import asdict

    combat = getattr(layer, "combat", None)
    return {"types": [asdict(kind) for kind in layer.catalogue.types],
            "combat": combat.snapshot() if combat is not None else None}


def _bind(decider, judge) -> None:
    """Hands the layer's judge to a decider that answers with it (`ScriptChoice`, or `Labelled` around one)."""
    bind = getattr(decider, "bind", None)
    if bind is not None:
        bind(judge)


def _seal(layer) -> None:
    rollout = layer.rollout
    if rollout is not None and rollout.recording:
        rollout.seal(layer.instance, dict(episode_of(layer.session), context=context_of(layer)))


def remaining_seconds(session, game_time_ms: int) -> float:
    """Game seconds left before the match is cut off, or -1 where the session states no limit, as a replay playback or a test does not."""
    limit = getattr(getattr(session, "settings", None), "max_seconds", 0)
    if not isinstance(limit, (int, float)) or limit <= 0:
        return -1.0
    return max(0.0, float(limit) - game_time_ms / 1000.0)


def _end(step: Step, terms, row) -> None:
    """Pays the end of the match to the decision standing when it ended, which closes its trajectory. It reads the board the last period ended on, so it is discounted as that board's potential was."""
    step.reward += terms.discount ** step.periods * terms.pay(row)
    step.signals.append(row)
    step.done = True


class LearntTactics(Tactics):
    """The tactical layer with the choice of departure taken from a decider instead of from the rule ladder."""

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, status_terminals: bool = True,
                 discount: float = REWARD_DISCOUNT, temperature: float = TEACHER_TEMPERATURE,
                 outcome_weight: float = OUTCOME_WEIGHT, score: str = BY_HEALTH,
                 terms: Optional[TacticalTerms] = None) -> None:
        super().__init__(session, catalogue)
        self.decider = decider
        _bind(decider, self.judge)
        self.rollout = rollout
        self.instance = instance
        #: The type table a set network's unit tokens are read against; the set row needs no combat table.
        self._set_context = materials.Context(catalogue=catalogue, combat=None)
        #: How soft the distribution written beside a teacher's decision is: the temperature its scores are softened at.
        self.temperature = temperature
        # Whether the conditions written into a contract are allowed to end an errand, which they should wherever something reissues contracts and should not where one contract stands for a whole constructed fight that nothing will reissue.
        #
        # The discount is handed in with it and for the same reason: the two cases differ in how long an errand is, and the shaping has to telescope at whatever the returns are discounted at. Whoever builds the layer knows which case this is, so whoever builds it owns both. `terms`, when given, states every figure at once and the three before it are not read.
        self.reward = TacticalReward(status_terminals=status_terminals,
                                     terms=terms or TacticalTerms(discount=discount, outcome_weight=outcome_weight,
                                                                  score=score))
        #: The decision each squad is owed payment for, held until the next period says what it earned.
        self.pending: Dict[int, Step] = {}
        #: How many errands were closed for each reason, so that a run can be asked whether its terminals are firing at all rather than having it guessed at from the shape of the returns.
        self.terminals: Counter = Counter()

    def decide(self, view: WorldView, squads: List[SquadRecord], game_time_ms: int):
        self._settle(view, squads, game_time_ms)
        return super().decide(view, squads, game_time_ms)

    def _settle(self, view: WorldView, squads: Sequence[SquadRecord], game_time_ms: int) -> None:
        """Pays every outstanding decision from the board that has just arrived, closes the errands that have ended, and ends the trajectory of any errand a new contract has just replaced."""
        if self.rollout is None:
            return
        present = {squad.id for squad in squads}
        for squad in squads:
            # What this squad has destroyed since its contract was issued, which the inherited layer already counts. It is a period behind, because the count for this period is made further down while the departure is being chosen, and a shaping term is a difference of two potentials so a lag applied to both ends of it cancels.
            track = self.tracks.get(squad.id)
            # Nothing destroyed yet where the contract in hand is not the one the tally was kept against: the inherited layer starts a fresh tally whenever a contract is issued, but it does that further down, while this runs first, and without the test the opening potential of a new errand would be taken with the last errand's kills still on it.
            killed = 0.0
            if track is not None and squad.contract is not None and track.issued_at_ms == squad.contract.issued_at_ms:
                killed = track.killed
            outcome = self.reward.step(squad, view, game_time_ms, killed=killed)
            step = self.pending.pop(squad.id, None)
            if step is not None:
                step.reward = outcome.reward
                step.signals.append(outcome.signal)
                step.done = outcome.done
                if outcome.done:
                    self.terminals[outcome.reason] += 1
                self.rollout.add((self.instance, squad.id), step)
            if outcome.renewed:
                # A contract is the unit of work and so the unit of pay, so a squad handed a different one has begun a different errand and the decisions of the two must not share a trajectory. Cut rather than closed, because the errand that was replaced did not fail -it stopped being observed- and its last decision is bootstrapped from its own value estimate. What that decision is paid is nothing, since the potentials of two contracts are measured against different ground and different allowances.
                self.rollout.cut((self.instance, squad.id))
        for squad_id in [key for key in self.pending if key not in present]:
            # A squad that has left the board between periods cannot be paid from anything, so its last decision is cut off rather than scored.
            self.pending.pop(squad_id, None)
            self.rollout.cut((self.instance, squad_id))
            self.reward.forget(squad_id)

    def finish(self, squad: SquadRecord, kills: float, health: float, reason: str = "called",
               engagement: int = -1) -> None:
        """Ends this squad's errand from outside, on the score of the fight it was in, both readings of it, paying the decision still waiting on it as the last of the trajectory.

        Whoever is running the fight knows when it is over; the layer only sees periods. Without this the decision taken in the final period of a fight is left outstanding and is paid, several seconds of game time later, out of the first period of whatever fight the squad number is next used for, which strings the two together into one trajectory.

        An errand that ended on its own conditions but has since gone on collecting decisions is the opposite case: the decisions after it belong to no errand, so the one still waiting is cut rather than paid.

        A squad that no longer exists is the third case. The period that finds it destroyed has no squad left to decide anything, so the ending is added to the last decision there was, where it lies in the trajectory. Without this the trajectory would be cut instead, which asserts that the errand went on unobserved rather than that the squad died doing it.
        """
        key = (self.instance, squad.id)
        if self.reward.ended(squad.id):
            self.pending.pop(squad.id, None)
            if self.rollout is not None:
                self.rollout.cut(key)
            self.reward.forget(squad.id)
            return
        outcome = self.reward.call(squad.id, kills, health)
        step = self.pending.pop(squad.id, None)
        if step is None:
            if self.rollout is not None:
                last = self.rollout.last(key)
                if self.rollout.close_with(key, outcome.reward, outcome.signal):
                    self.terminals[reason] += 1
                    last.engagement = engagement
            return
        step.reward = outcome.reward
        step.signals.append(outcome.signal)
        step.done = True
        step.engagement = engagement
        self.terminals[reason] += 1
        if self.rollout is not None:
            self.rollout.add(key, step)

    def _departure(self, squad: SquadRecord, members: List[Sighting], threats: List[Sighting],
                   losses: float, track) -> Deviation:
        if self._view is None:
            return super()._departure(squad, members, threats, losses, track)
        state = self.state(squad, members, threats, losses, track)
        # Every departure is always available. Withdrawing from a fight that is going well is a bad idea and not an illegal one, and a mask that encoded which were sensible would be the rule ladder again, hidden.
        mask = [1.0] * len(Deviation)
        recording = self.rollout is not None
        set_input = getattr(self.decider, "set_input", False)
        found = None
        if (recording and self.rollout.recording) or set_input:
            found = materials.tactical(squad, members, threats, losses, track.killed, self._view, self._now)
        scores = self.judge.scores(state) if recording or self.decider is None else None
        label = _best(scores) if scores is not None else -1
        net_state = None
        if self.decider is None:
            # No decider means the inherited rule decides, which is how the script is turned into a teacher: what comes out is a state and an action of exactly the form a learnt layer emits.
            choice = Choice(action=label, probabilities=one_hot(label, len(mask)))
        elif set_input:
            # A set network reads the decision's units one by one; the judge and the record keep the flat features.
            net_state = tokens.set_state(materials.tables(found), self._set_context, features=state)
            choice = self.decider.choose(net_state, mask)
        else:
            choice = self.decider.choose(state, mask)
        if recording:
            self.pending[squad.id] = Step(
                state=state, action=choice.action, mask=mask, log_prob=choice.log_prob, value=choice.value,
                squad=squad.id, at_ms=self._now, meta=dict(choice.meta or {}), label=label,
                soft=distribution(scores, self.temperature), probabilities=list(choice.probabilities or ()),
                version=choice.version, net_state=net_state,
                materials=found if self.rollout.recording else None)
        return Deviation(choice.action)

    def release(self) -> None:
        """Files every decision still waiting on payment into its trajectory, leaving the trajectory open. Both sides of an arena share one rollout, and each side's last decisions have to be in their trajectories before either side closes the instance's errands."""
        if self.rollout is None:
            return
        for squad_id, step in list(self.pending.items()):
            self.rollout.add((self.instance, squad_id), step)
        self.pending.clear()

    def close(self, score: Optional[float] = None) -> None:
        """Ends every open errand at the end of an episode. They did not fail; they stopped being observed, so they are bootstrapped rather than treated as terminal. The match score is not the tactical layer's pay, so it is not read."""
        if self.rollout is None:
            return
        self.release()
        # This instance's errands only. One buffer serves every instance of a run, and an episode ending here says nothing about the fight another instance is in the middle of.
        self.rollout.cut_all(owner=self.instance)
        _seal(self)


class LearntOperations(Operations):
    """The operational layer with the choice of where and what taken from a decider, made only when something calls for a choice.

    A squad is decided about when it has no contract, when its mission report says the errand stalled, is going badly, is finished (other than a garrison holding its ground) or is out of time, and otherwise once every REVIEW_MS; in between it keeps the errand it holds and no decision is taken. The script layer keeps a squad on its errand for the same reasons, and a layer that re-drew every squad's target each period would re-issue contracts on every change of its mind and reset the losses each mission is judged by.

    Each decision is paid, while it stands, the discounted sum of the per-period payments, and is filed with the number of periods it stood for so that what follows it is discounted by that many periods. The decisions standing when the match ends are paid its score and end their trajectories there.
    """

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None,
                 instance: int = -1, review_ms: int = REVIEW_MS, terms: Optional[OperationalTerms] = None,
                 temperature: float = TEACHER_TEMPERATURE) -> None:
        super().__init__(session, catalogue, review_ms=review_ms)
        self.decider = decider
        _bind(decider, self.judge)
        self.rollout = rollout
        self.instance = instance
        #: How soft the distributions written beside a teacher's decision are.
        self.temperature = temperature
        #: Per squad, the contract its mission report was last read under and the destroyed and lost worth it said, so that each period counts only what changed.
        self._exchange: Dict[int, Tuple[int, float, float]] = {}
        # The discount per period is the one the returns are computed at, so that the shaping telescopes; whoever builds the layer hands the same figure to both.
        self.reward = OperationalReward(terms)
        #: The decision each squad is under, still collecting what it earns.
        self.pending: Dict[int, Step] = {}
        #: Decisions taken over the episode, for the record.
        self.decisions = 0
        #: The achievement of the strategic orders summed over the episode's periods, and how many periods that was, so that a record can say how well the orders were met beside how the match went.
        self.achieved = 0.0
        self.periods = 0

    @property
    def local_exchange(self) -> float:
        return self.reward.terms.local_exchange

    def decide(self, view: WorldView, orders: OperationsOrders, squads: List[SquadRecord],
               reports: List[MissionReport], game_time_ms: int):
        # A decider that reads what the players did, as the built-in AI's teacher does, is shown every operational observation.
        observe = getattr(self.decider, "observe", None)
        if observe is not None:
            observe(view)
        self._now = game_time_ms
        self._settle(view, orders, squads, reports)
        return super().decide(view, orders, squads, reports, game_time_ms)

    def _settle(self, view: WorldView, orders: OperationsOrders, squads: Sequence[SquadRecord],
                reports: Sequence[MissionReport] = ()) -> None:
        """Pays the period that has just elapsed to every decision still standing, and cuts the trajectories of squads that have left the board.

        One figure pays all of them, because the board is a statement about the whole match and not about any one squad; each is also paid its own squad's exchange at the rate the run asked for, nought by default. A squad that is gone is cut rather than ended: the payment is for the board, which goes on after it, so its last decision is bootstrapped from its own value estimate.
        """
        board = self.reward.signal(view, orders, squads, remaining_seconds(self.session, self._now))
        self.achieved += board[3]
        self.periods += 1
        exchanges = self._exchanges(squads, reports)
        if self.rollout is None:
            return
        terms = self.reward.terms
        for squad_id, step in self.pending.items():
            row = OperationalReward.with_exchange(board, exchanges.get(squad_id, 0.0))
            step.reward += terms.discount ** step.periods * terms.pay(row)
            step.signals.append(row)
            step.periods += 1
        present = {squad.id for squad in squads}
        for squad_id in [key for key in self.pending if key not in present]:
            self._file(squad_id)
            self.rollout.cut((self.instance, squad_id))
            self.decided_at.pop(squad_id, None)

    def _exchanges(self, squads: Sequence[SquadRecord], reports: Sequence[MissionReport]) -> Dict[int, float]:
        """Each squad's exchange over the period that has just elapsed, worth destroyed less worth lost, read off the mission reports; counted from nought whenever the squad is under a different contract from the last period, since a report counts from the issue of its contract."""
        issued = {squad.id: squad.contract.issued_at_ms if squad.contract is not None else -1 for squad in squads}
        changes: Dict[int, float] = {}
        seen: Dict[int, Tuple[int, float, float]] = {}
        for report in reports:
            if report.squad not in issued:
                continue
            under = issued[report.squad]
            last = self._exchange.get(report.squad)
            destroyed, lost = (last[1], last[2]) if last is not None and last[0] == under else (0.0, 0.0)
            changes[report.squad] = (report.destroyed - destroyed) - (report.losses - lost)
            seen[report.squad] = (under, report.destroyed, report.losses)
        self._exchange = seen
        return changes

    def _file(self, squad_id: int) -> None:
        step = self.pending.pop(squad_id, None)
        if step is None or self.rollout is None:
            return
        step.periods = max(1, step.periods)
        # A teacher that answered before it could see what followed relabels the step from what did.
        revise = getattr(self.decider, "revise", None)
        if revise is not None:
            revise(step, self._now)
        self.rollout.add((self.instance, squad_id), step)

    def _choose(self, state: List[float], squad: SquadRecord, regions: List[float],
                masks: List[List[float]]) -> Optional[Tuple[int, int]]:
        """Where and which plan, from the decider, and written down as a step of the squad's trajectory. Without a decider the judge answers, which is how the script is turned into a teacher."""
        recording = self.rollout is not None
        judged = self.judge.scores(state, squad.id, regions, masks) if recording or self.decider is None else None
        if self.decider is None:
            chosen = self.judge.choose(state, squad.id, regions, masks)
            if chosen is None:
                return None
            choice = Choice(action=chosen[0], second=chosen[1], probabilities=one_hot(chosen[0], len(regions)),
                            second_probabilities=one_hot(chosen[1], len(masks[chosen[0]])))
        else:
            # A decider that reads the board itself, as one inferring a person's decision from what they did next does, is handed it; every other decider sees only the encoded state, as a network does.
            board = getattr(self.decider, "choose_on_board", None)
            choice = (board(self._view, squad, state, regions, masks, self._now, logistics=self.logistics)
                      if board is not None
                      else self.decider.choose(state, squad.id, regions, masks))
            if choice is None:
                return None
        self.decisions += 1
        if recording:
            if getattr(self.decider, "teaches", False):
                # The decider is the teacher, as a person is in a replay: what was played is the answer to learn.
                label, second_label, soft, second_soft = choice.action, choice.second, [], []
            else:
                region_scores, plan_scores = judged
                finite = any(score > float("-inf") for score in region_scores)
                label = _best(region_scores) if finite else -1
                second_label = _best(plan_scores) if finite else -1
                soft = distribution(region_scores, self.temperature)
                second_soft = distribution(plan_scores, self.temperature)
            self._file(squad.id)
            self.pending[squad.id] = Step(
                state=state, action=choice.action, mask=list(regions), second=choice.second,
                second_mask=[value for row in masks for value in row], log_prob=choice.total_log_prob,
                value=choice.value, squad=squad.id,
                at_ms=self._now, periods=0, meta=dict(choice.meta or {}), label=label, second_label=second_label,
                soft=soft, second_soft=second_soft, probabilities=list(choice.probabilities or ()),
                second_probabilities=list(choice.second_probabilities or ()), version=choice.version,
                materials=materials.operational(self._view, self._orders, self._squads, self._spawns, squad, self._now,
                                                self.access(self._view, squad))
                if self.rollout.recording else None)
        return choice.action, choice.second

    def taint(self, squad_id: int) -> None:
        """Marks the decision standing for this squad as one somebody outside the chain interfered with."""
        step = self.pending.get(squad_id)
        if step is not None:
            step.tainted = True

    def close(self, score: Optional[float] = None) -> None:
        """Ends the match's trajectories: with its score, those of the decisions standing at its end are paid the score and ended; without one, as when the match was stopped or the connection lost, they are cut and bootstrapped."""
        if self.rollout is None:
            return
        ending = self.reward.end(score) if score is not None else None
        for squad_id in list(self.pending):
            if ending is not None:
                _end(self.pending[squad_id], self.reward.terms, ending)
            self._file(squad_id)
        self.rollout.cut_all(owner=self.instance)
        _seal(self)
        self.reward.reset()
        self.decided_at.clear()
        self._exchange = {}


#: What an economy's trajectory is keyed by beside its instance. There is one economy to an instance, so one trajectory, running the length of the match.
ECONOMY_KEY = "economy"


class LearntEconomy(Economy):
    """The build order with the choice of the next investment taken from a decider.

    A period's investments are chosen one after another, and each is a decision of its own. The ones within a period follow each other with no game time between them, so each is filed with nought periods and nothing it is followed by is discounted; the last of a period stands until the next period's first, collecting the discounted payment of every period in between, as an operational decision collects the periods it stands for. A period in which nothing but stopping was on offer takes no decision.
    """

    def __init__(self, session, catalogue, decider, rollout: Optional[Rollout] = None, instance: int = -1,
                 options: Options = Options(), terms: Optional[EconomicTerms] = None,
                 temperature: float = TEACHER_TEMPERATURE) -> None:
        super().__init__(session, catalogue, options)
        self.decider = decider
        _bind(decider, self.judge)
        self.rollout = rollout
        self.instance = instance
        self.temperature = temperature
        # The discount per period is the one the returns are computed at, so that the shaping telescopes; whoever builds the layer hands the same figure to both.
        self.reward = EconomicReward(terms)
        #: The last investment chosen, still collecting what the periods after it earn.
        self.pending: Optional[Step] = None
        #: Investments chosen over the episode, for the record.
        self.decisions = 0
        self._now = 0

    def decide(self, view: WorldView, orders: EconomyOrders, replacements: Sequence[Replacement]):
        self._settle(view)
        self._now = view.observation.game_time_ms
        return super().decide(view, orders, replacements)

    def _settle(self, view: WorldView) -> None:
        """Pays the period that has just elapsed to the investment still standing."""
        outcome = self.reward.step(view, remaining_seconds(self.session, view.observation.game_time_ms))
        if self.pending is not None:
            self.pending.reward += self.reward.discount ** self.pending.periods * outcome.reward
            self.pending.signals.append(outcome.signal)
            self.pending.periods += 1

    def _file(self) -> None:
        if self.pending is not None and self.rollout is not None:
            self.rollout.add((self.instance, ECONOMY_KEY), self.pending)
        self.pending = None

    def _choose(self, state: List[float], mask: List[float], slots: Sequence[Optional[Offer]],
                board: Optional[EconomicBoard] = None) -> Optional[int]:
        """The next investment from the decider, written down as a step of the match's trajectory. Without a decider the judge answers, which is how the script is turned into a teacher."""
        recording = self.rollout is not None
        scores = self.judge.scores(state) if recording or self.decider is None else None
        label = _best(scores) if scores is not None else -1
        if self.decider is None:
            choice = Choice(action=label, probabilities=one_hot(label, len(mask)))
        else:
            choice = self.decider.choose(state, mask)
        self.decisions += 1
        if recording:
            self._file()
            self.pending = Step(state=state, action=choice.action, mask=list(mask), log_prob=choice.log_prob,
                                value=choice.value, squad=0, at_ms=self._now, periods=0,
                                meta=dict(choice.meta or {}), label=label, soft=distribution(scores, self.temperature),
                                probabilities=list(choice.probabilities or ()), version=choice.version,
                                materials=materials.economic(board, slots)
                                if self.rollout.recording and board is not None else None)
        return choice.action

    def close(self, score: Optional[float] = None) -> None:
        """Ends the match's trajectory: with its score, the investment standing at its end is paid the score and the trajectory ends there; without one, as when the match was stopped or the connection lost, it is cut and bootstrapped from its last value estimate."""
        if self.rollout is None:
            return
        ending = self.reward.end(score) if score is not None else None
        if ending is not None and self.pending is not None:
            _end(self.pending, self.reward.terms, ending)
        self._file()
        self.rollout.cut_all(owner=self.instance)
        _seal(self)
        self.reward.reset()
