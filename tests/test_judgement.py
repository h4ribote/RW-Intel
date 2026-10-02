"""The script's decisions for the learnt layers, read off encoded states alone, and how a fit is made against them."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rwintel.control.policy.contracts import ALLOCATION, Doctrine, Posture, Role
from rwintel.control.policy.encoding import (
    INVESTMENT_SLOTS,
    EconomicBoard,
    Investment,
    Offer,
    economic_state,
    investment_mask,
    lay_out,
    GLOBAL_FEATURES,
    GLOBAL_SIZE,
    MEANS,
    OPERATIONAL_PLANS,
    OPERATIONAL_SIZE,
    TRANSPORT_FEATURES,
    TRANSPORT_SIZE,
    plan_of,
    REGION_FEATURES,
    REGION_SIZE,
    SQUAD_FEATURES,
    SQUAD_SIZE,
    TACTICAL_FEATURES,
    TACTICAL_SIZE,
)
from rwintel.control.policy.judgement import EconomyJudge, OperationsJudge, TacticsJudge, distribution
from rwintel.control.policy.options import Options
from rwintel.control.policy.tuning import Tuning
from rwintel.learn.deciders import Choice, Labelled
from rwintel.learn.imitation import _cross_entropy
from rwintel.wire import REGION_SLOTS, SQUAD_SLOTS, Deviation, Status, Task

# ---- the tactical judge ---------------------------------------------------------------------


def _fight(**features):
    state = [0.0] * TACTICAL_SIZE
    state[TACTICAL_FEATURES.index("status_active")] = 1.0
    for name, value in features.items():
        state[TACTICAL_FEATURES.index(name)] = value
    return state


def test_a_squad_nobody_is_fighting_holds():
    assert TacticsJudge().choose(_fight()) == Deviation.HOLD


def test_a_spent_mission_breaks_off_and_the_whole_way_out_when_it_is_losing():
    spent = _fight(engaged=1.0, budget_share=0.5, budget_spent=0.9)
    assert TacticsJudge().choose(spent) == Deviation.WITHDRAW
    losing = _fight(engaged=1.0, status_active=0.0, status_losing=1.0)
    assert TacticsJudge().choose(losing) == Deviation.WITHDRAW_FAR


def test_a_fight_the_table_expects_to_lose_is_left_before_the_losses_say_so_when_switched_to_predict():
    knobs = Tuning()
    doomed = _fight(engaged=1.0, predicted=(knobs.predict_withdraw + knobs.predict_withdraw_far) / 2)
    assert TacticsJudge().choose(doomed) == Deviation.WITHDRAW
    assert TacticsJudge().choose(_fight(engaged=1.0, predicted=knobs.predict_withdraw_far - 0.05)) == Deviation.WITHDRAW_FAR
    assert TacticsJudge().choose(_fight(engaged=1.0, predicted=knobs.predict_withdraw + 0.05)) == Deviation.HOLD
    assert TacticsJudge(predict=False).choose(doomed) == Deviation.HOLD


def test_out_reaching_kites_and_enough_shooters_concentrate_on_a_gun_in_reach_first():
    assert TacticsJudge().choose(_fight(engaged=1.0, our_reach=0.5, their_reach=0.2, reach_advantage=0.3)) == Deviation.KITE
    focus = dict(engaged=1.0, armed_members=3 / 12, enemies_near=2 / 12, target_in_reach=1.0)
    assert TacticsJudge().choose(_fight(**focus)) == Deviation.FOCUS
    assert TacticsJudge().choose(_fight(long_in_reach=1.0, **focus)) == Deviation.FOCUS_THREAT


def test_the_scores_put_the_taken_departure_first_and_soften_into_a_distribution():
    scores = TacticsJudge().scores(_fight(engaged=1.0, our_reach=0.5, their_reach=0.2, reach_advantage=0.3,
                                          armed_members=3 / 12, enemies_near=2 / 12, target_in_reach=1.0))
    assert max(range(len(scores)), key=scores.__getitem__) == Deviation.KITE
    assert scores[Deviation.FOCUS] > scores[Deviation.HOLD] > scores[Deviation.SPREAD]
    soft = distribution(scores, 0.25)
    assert abs(sum(soft) - 1.0) < 1e-9 and soft[Deviation.KITE] > soft[Deviation.FOCUS] > soft[Deviation.SPREAD]


# ---- the operational judge ------------------------------------------------------------------


class _Board:
    """An operational state built by name."""

    def __init__(self):
        self.state = [0.0] * OPERATIONAL_SIZE

    def glob(self, **features):
        for name, value in features.items():
            self.state[GLOBAL_FEATURES.index(name)] = value
        return self

    def region(self, slot, **features):
        features.setdefault("valid", 1.0)
        for name, value in features.items():
            self.state[GLOBAL_SIZE + slot * REGION_SIZE + REGION_FEATURES.index(name)] = value
        return self

    def squad(self, slot, doctrine, status=Status.ACTIVE, task=None, **features):
        base = GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + slot * SQUAD_SIZE
        values = {"valid": 1.0, "taskable": 1.0, f"doctrine_{doctrine.name.lower()}": 1.0,
                  f"status_{status.name.lower()}": 1.0, **features}
        if task is not None:
            values[f"task_{task.name.lower()}"] = 1.0
        for name, value in values.items():
            self.state[base + SQUAD_FEATURES.index(name)] = value
        return self


REGIONS = [1.0] * 4 + [0.0] * (REGION_SLOTS - 4)


def _plans(tasks, carried=None):
    """Plan masks for the four regions on the board: every allowed task by walking, except where `carried` names a region and the transport slots that can carry the squad there instead."""
    carried = carried or {}
    rows = []
    for region in range(REGION_SLOTS):
        row = [0.0] * OPERATIONAL_PLANS
        if region < 4:
            means = [k + 1 for k in carried[region]] if region in carried else [0]
            for task in tasks:
                for m in means:
                    row[int(task) * MEANS + m] = 1.0
        rows.append(row)
    return rows


VANGUARD_TASKS = _plans((Task.ATTACK, Task.ENCIRCLE))
GARRISON_TASKS = _plans((Task.DEFEND, Task.ESCORT))
RAID_TASKS = _plans((Task.RAID, Task.WITHDRAW))


def _walking(task):
    return plan_of(task, -1)


def _front():
    """Home at slot 0, our ground at 1, and two contested regions at 2 and 3 with the second the more valuable."""
    return (_Board().glob(offensive=1.0)
            .region(0, distance=0.0, held_by_us=0.25, ours_present=0.2)
            .region(1, distance=0.2, held_by_us=0.25, ours_present=0.1)
            .region(2, distance=0.4, enemy_present=0.3, priority=0.4)
            .region(3, distance=0.5, enemy_present=0.3, priority=0.9))


def test_a_vanguard_attacks_only_where_it_is_expected_to_win_when_concentrating():
    board = _front().region(2, predicted=0.5).region(3, predicted=-0.4).squad(0, Doctrine.VANGUARD)
    assert OperationsJudge().choose(board.state, 0, REGIONS, VANGUARD_TASKS) == (2, _walking(Task.ATTACK))
    assert OperationsJudge(concentrate=False).choose(board.state, 0, REGIONS, VANGUARD_TASKS)[0] == 3


def test_a_vanguard_joins_where_others_are_bound_and_surrounds_with_odds_to_spare():
    edge = Tuning().encircle_edge
    board = (_front().region(2, predicted=edge / 2, committed=0.4, priority=0.5)
             .region(3, predicted=edge / 2, priority=0.5).squad(0, Doctrine.VANGUARD))
    region, plan = OperationsJudge().choose(board.state, 0, REGIONS, VANGUARD_TASKS)
    assert region == 2 and plan == _walking(Task.ATTACK)
    alone = _front().region(3, predicted=min(1.0, edge + 0.1)).squad(0, Doctrine.VANGUARD)
    assert OperationsJudge().choose(alone.state, 0, REGIONS, VANGUARD_TASKS) == (3, _walking(Task.ENCIRCLE))


def test_a_vanguard_that_can_win_nowhere_gathers_forward_where_our_strength_is_bound():
    board = (_front().region(2, predicted=-0.2).region(3, predicted=-0.6)
             .region(1, committed=0.3).squad(0, Doctrine.VANGUARD))
    assert OperationsJudge().choose(board.state, 0, REGIONS, VANGUARD_TASKS) == (1, _walking(Task.ATTACK))


def test_the_nearest_free_garrison_covers_the_next_expansion_and_no_other():
    board = (_front().region(3, plan=1.0, enemy_present=0.0)
             .squad(0, Doctrine.GARRISON, status=Status.COMPLETE, task=Task.DEFEND, plan_distance=0.5)
             .squad(1, Doctrine.GARRISON, status=Status.COMPLETE, task=Task.DEFEND, plan_distance=0.2))
    assert OperationsJudge().choose(board.state, 1, REGIONS, GARRISON_TASKS) == (3, _walking(Task.DEFEND))
    assert OperationsJudge().choose(board.state, 0, REGIONS, GARRISON_TASKS)[0] != 3
    assert OperationsJudge(cover=False).choose(board.state, 1, REGIONS, GARRISON_TASKS)[0] != 3


def test_a_losing_raid_withdraws_home():
    board = _front().squad(0, Doctrine.RAID, status=Status.LOSING, task=Task.RAID).region(2, current_target=1.0)
    assert OperationsJudge().choose(board.state, 0, REGIONS, RAID_TASKS) == (0, _walking(Task.WITHDRAW))


def test_ground_that_cannot_be_walked_to_is_reached_by_the_nearest_open_transport():
    """Region 2 is across water, open by the transports in slots 0 and 2; slot 2 stands nearer the squad, so it is the means the rule takes."""
    board = _front().region(2, predicted=0.5).region(3, predicted=-0.4).squad(0, Doctrine.VANGUARD)
    start = GLOBAL_SIZE + REGION_SLOTS * REGION_SIZE + SQUAD_SLOTS * SQUAD_SIZE
    for slot, distance in ((0, 0.6), (2, 0.1)):
        board.state[start + slot * TRANSPORT_SIZE + TRANSPORT_FEATURES.index("valid")] = 1.0
        board.state[start + slot * TRANSPORT_SIZE + TRANSPORT_FEATURES.index("distance")] = distance
    masks = _plans((Task.ATTACK, Task.ENCIRCLE), carried={2: [0, 2]})
    assert OperationsJudge().choose(board.state, 0, REGIONS, masks) == (2, plan_of(Task.ATTACK, 2))


def test_the_region_scores_rank_what_was_chosen_first_and_mask_what_does_not_exist():
    board = _front().region(2, predicted=0.5).squad(0, Doctrine.VANGUARD)
    regions, tasks = OperationsJudge().scores(board.state, 0, REGIONS, VANGUARD_TASKS)
    assert max(range(len(regions)), key=regions.__getitem__) == 2
    assert all(math.isinf(regions[slot]) for slot in range(4, REGION_SLOTS))
    soft = distribution(regions, 0.25)
    assert abs(sum(soft) - 1.0) < 1e-9 and all(soft[slot] == 0.0 for slot in range(4, REGION_SLOTS))


# ---- fitting to the teacher ------------------------------------------------------------------


def test_a_written_distribution_replaces_the_smoothing_as_the_target():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 0.0, 0.0]])
    mask = torch.ones(2, 3)
    chosen = torch.tensor([0, 1])
    soft = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.0, 0.0]])
    rows = torch.tensor([True, False])
    loss = _cross_entropy(logits, chosen, mask, 0.0, None, soft, rows)
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = (-(soft[0] * log_probs[0]).sum() - log_probs[1, 1]) / 2
    assert torch.allclose(loss, expected)


def test_an_exploring_pupil_says_what_mixture_it_drew_from():
    """What an exploring pupil plays is drawn from its own distribution most of the time and evenly from the legal actions the rest, and the probability it records for what it played is that mixture's. A record whose probabilities were the pupil's alone would misstate every action the exploration substituted, which is exactly the action a learner off the policy needs priced correctly."""
    pupil_probabilities = [0.7, 0.3] + [0.0] * (len(Deviation) - 2)

    class _Pupil:
        def choose(self, state, mask):
            return Choice(action=Deviation.HOLD, log_prob=math.log(0.7), probabilities=list(pupil_probabilities))

    mask = [1.0] * len(Deviation)
    played = Labelled(_Pupil()).choose([0.0] * TACTICAL_SIZE, mask)
    assert played.action == Deviation.HOLD and abs(math.exp(played.log_prob) - 0.7) < 1e-12

    exploring = Labelled(_Pupil(), explore=0.5, seed=3)
    drawn = set()
    for _ in range(60):
        choice = exploring.choose([0.0] * TACTICAL_SIZE, mask)
        drawn.add(choice.action)
        expected = 0.5 * pupil_probabilities[choice.action] + 0.5 / len(mask)
        assert abs(math.exp(choice.log_prob) - expected) < 1e-12
        assert abs(sum(choice.probabilities) - 1.0) < 1e-12
    assert len(drawn) > 2


# ---- the economic judge ---------------------------------------------------------------------


def _economy(offers, posture=Posture.ARM, **board):
    """The economy's encoded board with these offers on it, and the slots they were laid out in."""
    allocation = ALLOCATION[posture]
    fields = dict(credits=2000.0, reserve=0.0, income=20.0, units=10, unit_cap=100, near_cap=False, at_cap=False,
                  factories=1, pending_factory=False, free_factories=1, factory_raising=False, builders=2,
                  builder_shortfall=0, idle_builders=1, extractors=2, home_open=0, plan_open=0, posture=posture,
                  economy_share=allocation.economy, military_share=allocation.military, tech_share=allocation.tech,
                  tech_cap=0.0, tech_fund=0.0, tech_fund_limit=8000.0, tech_level=1, target_mix={Role.ARMOUR: 1.0})
    fields.update(board)
    slots = lay_out([Offer(kind=Investment.STOP)] + list(offers))
    return economic_state(EconomicBoard(**fields), slots), slots


def _where(slots, kind, **match):
    return next(index for index, offer in enumerate(slots)
                if offer is not None and offer.kind == kind and all(getattr(offer, k) == v for k, v in match.items()))


def _armour(index=0, price=350.0, efficiency=1.0):
    return Offer(kind=Investment.UNIT, price=price, type_index=index, efficiency=efficiency,
                 unit_efficiency=efficiency * price, fighting=True, role=Role.ARMOUR, role_rank=1.0)


def test_the_build_order_takes_its_rungs_top_down_and_stopping_outranks_what_no_rung_lets_through():
    """An extractor at home before the army, the army before a turret, all of them before stopping; a builder nobody is short of scores below stopping, and a board with nothing else on it stops."""
    offers = [Offer(kind=Investment.EXTRACTOR, price=700.0, region=0, home=True),
              _armour(), Offer(kind=Investment.TURRET, price=500.0, type_index=9, contact=True),
              Offer(kind=Investment.BUILDER, price=500.0, type_index=2)]
    state, slots = _economy(offers)
    scores = EconomyJudge().scores(state)
    extractor, unit = _where(slots, Investment.EXTRACTOR), _where(slots, Investment.UNIT)
    turret, builder = _where(slots, Investment.TURRET), _where(slots, Investment.BUILDER)
    assert scores[extractor] > scores[unit] > scores[turret] > scores[0] == 0.0 > scores[builder]
    assert EconomyJudge().choose(state) == extractor
    assert all(math.isinf(scores[slot]) for slot in range(INVESTMENT_SLOTS) if slots[slot] is None)

    lone, _ = _economy([Offer(kind=Investment.BUILDER, price=500.0, type_index=2)])
    assert EconomyJudge().choose(lone) == 0


def test_a_unit_worth_waiting_for_is_chosen_unpaid_and_one_nearly_as_good_is_made_instead():
    """The strongest unit within reach costs more than the treasury holds. A unit it can pay for that is worth less than `save_share` of it leaves the factory waiting, which is choosing the strong one unpaid; one worth more is made instead."""
    save_share = Tuning().save_share
    heavy = _armour(index=6, price=1200.0, efficiency=2.0)
    weak, _ = _economy([heavy, _armour(index=0, efficiency=2.0 * (save_share - 0.1))], credits=500.0)
    strong, slots = _economy([heavy, _armour(index=0, efficiency=2.0 * (save_share + 0.1))], credits=500.0)
    assert EconomyJudge().choose(weak) == _where(slots, Investment.UNIT, type_index=6)
    assert EconomyJudge().choose(strong) == _where(slots, Investment.UNIT, type_index=0)


def test_the_judges_answer_is_the_likeliest_in_the_distribution_written_beside_it():
    offers = [Offer(kind=Investment.EXTRACTOR, price=700.0, region=1, plan_rank=1.0), _armour()]
    state, slots = _economy(offers, posture=Posture.EXPAND)
    scores = EconomyJudge().scores(state)
    softened = distribution(scores, 0.25)
    assert max(range(INVESTMENT_SLOTS), key=lambda slot: softened[slot]) == EconomyJudge().choose(state)
    assert abs(sum(softened) - 1.0) < 1e-9
    assert all(softened[slot] == 0.0 for slot, allowed in enumerate(investment_mask(slots)) if not allowed)


def test_the_plans_ground_goes_before_the_army_only_when_switched_to_claim_it_first():
    offers = [Offer(kind=Investment.EXTRACTOR, price=700.0, region=1, plan_rank=1.0), _armour()]
    state, slots = _economy(offers, posture=Posture.EXPAND)
    assert EconomyJudge().choose(state) == _where(slots, Investment.EXTRACTOR)
    assert EconomyJudge(Options(expand_first=False)).choose(state) == _where(slots, Investment.UNIT)
