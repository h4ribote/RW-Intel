"""What the strategic layer and the ground it hands down promise, held to.

The posture rule is a few lines, and what is pinned is what those lines are for: what the squads run into reaches the target mix, a posture is not given up on a flicker of income, our army against theirs decides pressing and yielding, and the expansion plan grows out from held ground and reaches both layers that act on it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.policy import ScriptPolicy
from rwintel.control.policy.contracts import FrontReport, MissionReport, Posture, Role
from rwintel.control.policy.ground import FRONTIER_REACH, frontier
from rwintel.control.policy.options import BASELINE, Options, parse
from rwintel.control.policy.strategy import INCOME_WINDOW, MIN_POSTURE_MS, STRATEGY_PERIOD_MS, Strategy
from rwintel.control.policy.tuning import Tuning
from rwintel.data.regions import Region
from rwintel.wire import RegionState, Status


class _Session:
    regions = [Region(id=0, x=0.0, y=0.0, radius=0.0, resources=1, spawn=True),
               Region(id=5, x=6000.0, y=0.0, radius=0.0, resources=1, spawn=True)]


def _region(region_id, x, resources=2, held=0, enemy_held=0, ours=0.0, theirs=0.0):
    return RegionState(id=region_id, resources=resources, held_by_us=held, held_by_enemy=enemy_held, x=x, y=0.0,
                       our_value=ours, enemy_value=theirs, enemy_seen_at_ms=0, distance_from_home=abs(x))


def _report(income=50.0, ours=0.0, theirs=0.0):
    return FrontReport(income=income, credits=0.0, military_value=ours, enemy_value=theirs, enemy_military=theirs,
                       held=1, enemy_held=1, lost_regions=0, enemy_bases=1)


HOME = _region(0, 0.0, resources=1, held=1)
BOARD = [HOME, _region(1, 1200.0), _region(2, 2600.0), _region(3, 4200.0, enemy_held=1)]


def _levelled(strategy, start_ms=0, income=50.0, **report):
    """Feeds the layer a full window of flat income, which is the plateau that turns expanding into arming."""
    t = start_ms
    for _ in range(INCOME_WINDOW):
        strategy.decide(_report(income=income, **report), BOARD, t, home=HOME)
        t += STRATEGY_PERIOD_MS
    return t


def test_the_frontier_grows_out_from_held_ground_and_skips_the_enemys():
    """Unchained, only what is in reach of home is offered; chained, holding the first region brings the next one into reach. A region the enemy has an extractor in is never offered."""
    assert BOARD[2].x - BOARD[1].x <= FRONTIER_REACH < BOARD[2].x
    assert frontier(BOARD, HOME, [], chained=False) == [1]
    held = [HOME, _region(1, 1200.0, held=1), BOARD[2], BOARD[3]]
    assert frontier(held, HOME, [], chained=True) == [1, 2]
    assert frontier(held, HOME, [], chained=False) == [1]


def test_the_expansion_plan_reaches_the_economy_and_the_operational_layer():
    economy, operations = Strategy(_Session(), None).decide(_report(), BOARD, 0, home=HOME)
    assert economy.expansion == operations.expansion == [1]


def test_flying_contact_bends_the_mix_towards_anti_air_and_hovercraft_does_not():
    contact = {Role.FAST: 3000.0, Role.ARMOUR: 1000.0}
    plain, _ = Strategy(_Session(), None).decide(_report(), BOARD, 0, home=HOME)
    flying, _ = Strategy(_Session(), None).decide(_report(), BOARD, 0, contact=contact, home=HOME, airborne=3000.0)
    hovering, _ = Strategy(_Session(), None).decide(_report(), BOARD, 0, contact=contact, home=HOME, airborne=0.0)
    assert flying.target_mix[Role.ANTI_AIR] > plain.target_mix[Role.ANTI_AIR]
    assert hovering.target_mix == plain.target_mix


def test_the_chain_hands_what_its_squads_ran_into_to_the_strategic_layer():
    policy = ScriptPolicy.__new__(ScriptPolicy)
    policy.reports = [MissionReport(squad=0, status=Status.ACTIVE, contact={Role.FAST: 500.0}),
                      MissionReport(squad=1, status=Status.ACTIVE, contact={Role.FAST: 250.0, Role.ARMOUR: 100.0})]
    assert policy._contact() == {Role.FAST: 750.0, Role.ARMOUR: 100.0}


def test_arming_is_held_through_a_little_growth_but_not_without_the_dwell():
    """A plateau turns expanding into arming. Growth of ten per cent across the window is not enough to turn back with the dwell on, and is enough without it."""
    for options, expected in ((Options(), Posture.ARM), (Options(dwell=False), Posture.EXPAND)):
        strategy = Strategy(_Session(), None, options)
        t = _levelled(strategy)
        assert strategy.posture == Posture.ARM
        t += MIN_POSTURE_MS
        for step in range(INCOME_WINDOW):
            strategy.decide(_report(income=50.0 + step * 1.25), BOARD, t, home=HOME)
            t += STRATEGY_PERIOD_MS
        assert strategy.posture == expected, options


def test_a_young_posture_is_held_but_defending_is_not_kept_waiting():
    strategy = Strategy(_Session(), None)
    strategy.posture, strategy.changed_at_ms = Posture.ARM, 0
    strategy.income_history.extend([10.0, 20.0, 30.0, 40.0, 50.0])
    assert strategy._transition(_report(), MIN_POSTURE_MS - 1) == Posture.ARM
    assert strategy._transition(_report(), MIN_POSTURE_MS) == Posture.EXPAND
    assert strategy._transition(_report(ours=1000.0, theirs=5000.0), 1) == Posture.DEFEND


def test_an_army_well_ahead_presses_and_one_far_behind_defends():
    strategy = Strategy(_Session(), None)
    t = _levelled(strategy, ours=2000.0, theirs=1000.0)
    _, operations = strategy.decide(_report(ours=2000.0, theirs=1000.0), BOARD, t, home=HOME)
    assert operations.posture == Posture.ARM and operations.offensive
    # Pressing under arming makes the enemy's extractors worth going for.
    assert operations.priorities[3] > 0.0

    unmoved = Strategy(_Session(), None, Options(relative=False))
    t = _levelled(unmoved, ours=3000.0, theirs=1000.0)
    _, operations = unmoved.decide(_report(ours=3000.0, theirs=1000.0), BOARD, t, home=HOME)
    assert not operations.offensive

    behind = Strategy(_Session(), None)
    _, operations = behind.decide(_report(ours=400.0, theirs=2000.0), BOARD, 0, home=HOME)
    assert operations.posture == Posture.DEFEND


def test_a_large_lead_goes_for_the_decision_and_holds_to_it_until_the_odds_fall_well_back():
    """Switched to push, an army at the deciding odds decides; between the holding odds and the deciding odds it holds on, and only once the lead has fallen below the holding odds does the chain let the decision go. Not switched, the odds alone never decide."""
    knobs = Tuning()
    decisive, holding, fallen = 1000.0 * knobs.decide_odds * 1.05, 1000.0 * (knobs.decide_odds + knobs.decide_hold_odds) / 2, 1000.0 * knobs.decide_hold_odds * 0.95
    strategy = Strategy(_Session(), None)
    t = _levelled(strategy, ours=decisive, theirs=1000.0)
    assert strategy.posture == Posture.DECIDE
    _, operations = strategy.decide(_report(ours=holding, theirs=1000.0), BOARD, t, home=HOME)
    assert operations.posture == Posture.DECIDE and operations.offensive
    _, operations = strategy.decide(_report(ours=fallen, theirs=1000.0), BOARD, t + STRATEGY_PERIOD_MS, home=HOME)
    assert operations.posture != Posture.DECIDE
    unpushed = Strategy(_Session(), None, Options(push=False))
    _levelled(unpushed, ours=decisive, theirs=1000.0)
    assert unpushed.posture == Posture.ARM


def test_pressing_carries_a_larger_loss_allowance_when_switched_to_push():
    """Switched to push, a lead just past the tuned pressing odds presses, and pressing may lose more than holding the front the same posture would; a lead just short of them does not press."""
    knobs = Tuning()
    ahead = 1000.0 * min(knobs.push_odds * 1.05, (knobs.push_odds + knobs.decide_odds) / 2)
    pushed = Strategy(_Session(), None)
    t = _levelled(pushed, ours=ahead, theirs=1000.0)
    _, pressing = pushed.decide(_report(ours=ahead, theirs=1000.0), BOARD, t, home=HOME)
    assert pressing.posture == Posture.ARM and pressing.offensive
    unpushed = Strategy(_Session(), None, Options(push=False, relative=False))
    t = _levelled(unpushed, ours=ahead, theirs=1000.0)
    _, holding = unpushed.decide(_report(ours=ahead, theirs=1000.0), BOARD, t, home=HOME)
    assert not holding.offensive and pressing.loss_allowance > holding.loss_allowance
    short = Strategy(_Session(), None)
    t = _levelled(short, ours=1000.0 * knobs.push_odds * 0.95, theirs=1000.0)
    _, waiting = short.decide(_report(ours=1000.0 * knobs.push_odds * 0.95, theirs=1000.0), BOARD, t, home=HOME)
    assert not waiting.offensive


def test_a_pinned_posture_does_not_press_on_the_odds():
    strategy = Strategy(_Session(), None)
    strategy.forced = Posture.ARM
    _, operations = strategy.decide(_report(ours=9000.0, theirs=1000.0), BOARD, 0, home=HOME)
    assert not operations.offensive


def test_time_is_kept_per_posture():
    strategy = Strategy(_Session(), None)
    t = _levelled(strategy)
    strategy.decide(_report(), BOARD, t, home=HOME)
    assert sum(strategy.time_in.values()) == t and strategy.changes == 1


def test_options_parse_left_to_right_over_the_defaults():
    assert parse("") == Options()
    assert parse("baseline") == BASELINE
    assert parse("baseline,frontier=on") == Options(**{**vars(BASELINE), "frontier": True})
    assert parse("tech=all,convoy=off") == Options(tech="all", convoy=False)
    for bad in ("frontier", "nothing=on", "tech=maybe", "dwell=sometimes"):
        try:
            parse(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
