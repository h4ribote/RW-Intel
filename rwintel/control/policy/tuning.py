"""The opening values of the script chain that are searched rather than argued over.

Every rule of the chain carries numbers, and most of them are opening values: set for a stated reason and never measured. The ones collected here are those whose setting plausibly moves the match score, and each comes with the range a search draws it from (`RANGES`). They live on an object handed to the layers rather than in module constants because the arms of a comparison run side by side in one process, so an arm that set a module constant would set it for every other arm too.

An arm names a setting as `script:tune.<name>=<value>`; the defaults here are the chain's own values.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple


@dataclass(frozen=True)
class Tuning:
    #: The build order: the share of the best type's worth a type that can be paid for now has to reach to be made instead of waiting for the best; how far ahead, in game seconds of income, the treasury is reckoned when deciding whether a type is within reach at all.
    save_share: float = 0.6295
    save_horizon_s: float = 33.4176
    #: Income, in the engine's own units, above which a second factory can be kept fed: a little over what two extractors bring in.
    second_factory_income: float = 33.5571
    #: Credits in the treasury above which another factory is worth more than anything else a builder could be doing: the price of a factory and a few tanks. Also the line the ledger counts a banked treasury by.
    banked_credits: float = 2013.029
    #: The most factories the build order raises. Compared against a count, so only the whole part matters.
    max_factories: float = 7.0
    #: The most a tier raise may cost for the army to hold its price back while it is saved for.
    upgrade_saving: float = 5367.1349
    #: The organisation layer: the most garrisons raised when switched to raise vanguards. Compared against a count, so only the whole part matters.
    max_garrisons: float = 1.0
    # The strategic layer: the odds it presses at, decides at and lets a decision go at, the loss allowance while pressing, and the odds it falls back to defending at.
    push_odds: float = 1.5804
    decide_odds: float = 3.4749
    decide_hold_odds: float = 1.4038
    push_allowance_share: float = 1.0001
    yield_odds: float = 0.6399
    # The operational judge: the predicted outcome a vanguard attacks at and surrounds at, how strongly strength already bound draws another squad, how much the prediction counts, and how far forward a rally point is drawn.
    attack_edge: float = 0.0408
    encircle_edge: float = 0.5048
    concentration: float = 0.3283
    predict_weight: float = 1.2694
    forward: float = 0.4574
    # The tactical judge: the predicted outcome a squad withdraws at and withdraws the whole way at.
    predict_withdraw: float = -0.4807
    predict_withdraw_far: float = -0.5031
    # The combat model: what outranging the enemy is worth, and the least share of an enemy's damage taken to reach a type it cannot reach.
    range_bonus: float = 0.2096
    unreachable_floor: float = 0.0698


#: The range each value is searched over, as its least and its most.
RANGES: Dict[str, Tuple[float, float]] = {
    "save_share": (0.5, 1.0),
    "save_horizon_s": (20.0, 120.0),
    "second_factory_income": (10.0, 50.0),
    "banked_credits": (1500.0, 6000.0),
    "max_factories": (3.0, 8.0),
    "upgrade_saving": (0.0, 6000.0),
    "max_garrisons": (0.0, 4.0),
    "push_odds": (1.0, 1.6),
    "decide_odds": (1.6, 4.0),
    "decide_hold_odds": (1.1, 2.0),
    "push_allowance_share": (0.3, 1.5),
    "yield_odds": (0.3, 0.8),
    "attack_edge": (-0.2, 0.5),
    "encircle_edge": (0.3, 0.9),
    "concentration": (0.0, 2.0),
    "predict_weight": (0.0, 1.5),
    "forward": (0.0, 1.0),
    "predict_withdraw": (-0.8, -0.1),
    "predict_withdraw_far": (-1.0, -0.5),
    "range_bonus": (0.0, 1.0),
    "unreachable_floor": (0.03, 0.3),
}

NAMES = tuple(field.name for field in fields(Tuning))


def describe(tuning: Tuning) -> str:
    """The values that differ from the defaults, as an arm writes them."""
    default = Tuning()
    parts = [f"tune.{name}={_format(getattr(tuning, name))}" for name in NAMES
             if getattr(tuning, name) != getattr(default, name)]
    return ",".join(parts)


def _format(value: float) -> str:
    """Short, and exact for any value written to ten significant figures or fewer, so that an arm named for a setting reads back as that setting."""
    return f"{value:.10g}"
