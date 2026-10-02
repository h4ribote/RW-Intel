"""The searched opening values: every one has a range around its default, and an arm can set any of them."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rwintel.control.policy.options import parse
from rwintel.control.policy.tuning import NAMES, RANGES, Tuning, describe


def test_every_searched_value_has_a_range_that_holds_its_default():
    assert set(RANGES) == set(NAMES)
    default = Tuning()
    for name, (low, high) in RANGES.items():
        assert low < high and low <= getattr(default, name) <= high, name


def test_an_arm_sets_a_searched_value_and_names_it_back_the_same_way():
    options = parse("push=off,tune.attack_edge=0.1,tune.max_factories=6")
    assert not options.push and options.tuning.attack_edge == 0.1 and options.tuning.max_factories == 6.0
    assert describe(options.tuning) == "tune.max_factories=6,tune.attack_edge=0.1"
    assert parse("baseline,tune.save_share=0.9").tuning.save_share == 0.9
    with pytest.raises(ValueError):
        parse("tune.nothing=1")
    with pytest.raises(ValueError):
        parse("tune.save_share=high")
