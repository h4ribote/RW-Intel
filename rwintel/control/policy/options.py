"""Switches on the script chain's rules, so that each rule can be measured against the chain without it.

A rule of the script earns its place by what it does to the match score, and a match does not repeat, so the only fair comparison is two arms run interleaved in the same run. Each switch here names one rule and its alternative; `baseline` sets every switch to the setting the chain had before these switches existed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Dict, Tuple

from .tuning import NAMES as TUNING_NAMES, Tuning

#: The prefix that names a searched opening value in an arm.
TUNE = "tune."

#: Values a switch written as on or off accepts.
_TRUE = ("on", "true", "1", "yes")
_FALSE = ("off", "false", "0", "no")

#: Modes the technology budget may run in: paid from a fund for factory tiers only, for every tier raise, or not a budget at all and only a ban when zero.
TECH_MODES = ("factory", "all", "off")

#: Modes the builder target may run in: grown with the open ground in the expansion plan, or the fixed opening value.
BUILDER_MODES = ("scaled", "fixed")


@dataclass(frozen=True)
class Options:
    #: Expansion ground grows outward from every region we hold, rather than stopping at a fixed distance from home.
    frontier: bool = True
    #: How many builders the economy keeps.
    builders: str = "scaled"
    #: Builders are ordered from the command centre as well as from factories.
    hq: bool = True
    #: A garrison is sent onto the next region of the expansion plan before the builder arrives.
    cover: bool = True
    #: What the squads run into reaches the strategic layer and bends the target mix.
    contact: bool = True
    #: The strategic layer weighs our army against the enemy's: pressing when ahead, defending when far behind.
    relative: bool = True
    #: A posture is held for a minimum time, and arming turns back to expanding only on clear growth.
    dwell: bool = True
    #: Reinforcements for a distant squad gather and leave together.
    convoy: bool = True
    #: How the strategic layer's technology cap is spent.
    tech: str = "factory"
    #: Open ground in the expansion plan is claimed before tier raises and the army, rather than from what they leave over.
    expand_first: bool = True
    #: Resource points whose placements never stand are given up on, and each factory placement tries fresh ground.
    retry: bool = True
    #: A tier raise that cannot be paid for yet has its price held back from the army until it can; off, raises are bought only from what production leaves over.
    saving: bool = True
    #: A match that has lost every factory raises another beside home ahead of everything else, and a builder standing still away from its site is given something else to do.
    recovery: bool = True
    #: What to make, which factory to raise and when to raise its tier are chosen by the fighting strength a credit buys against what the enemy fields (`combat.CombatTable`); off, the factories fill the posture's target mix role by role with the cheapest type in each.
    choose: bool = True
    #: The chain presses at smaller odds and with a larger loss allowance, and goes for the decision on a large enough lead whatever the enemy's bases say.
    push: bool = True
    #: A squad in a fight the combat table expects it to lose withdraws before the losses bear that out.
    predict: bool = True
    #: A vanguard attacks only where the combat table expects it to win together with what else is bound there, is drawn to where other squads are bound, and gathers at a rally point when nowhere can be won; off, vanguards spread over the contested ground.
    concentrate: bool = True
    #: No more than `tuning.max_garrisons` garrisons are raised, and a unit a vanguard would take waits loose rather than being pushed over strength into another kind of squad, so that the army gathers into vanguards; off, garrisons take every tank that arrives and the army stands on defence.
    vanguards: bool = True
    #: The opening values a search moves (`tuning.Tuning`), set in an arm as `tune.<name>=<value>`.
    tuning: Tuning = Tuning()


#: The chain before these switches existed: every new rule off, every mode at its original setting, and the older saving rule on.
BASELINE = Options(frontier=False, builders="fixed", hq=False, cover=False, contact=False, relative=False, dwell=False,
                   convoy=False, tech="off", expand_first=False, retry=False, saving=True, recovery=False, choose=False, push=False, predict=False, concentrate=False, vanguards=False)

_CHOICES: Dict[str, Tuple[str, ...]] = {"builders": BUILDER_MODES, "tech": TECH_MODES}


def parse(text: str) -> Options:
    """Options from a comma separated list of `name=value`, applied left to right over the defaults. The word `baseline` resets every switch to its original setting, so `baseline,frontier=on` measures one rule on its own; `tune.<name>=<value>` sets one of the searched opening values."""
    options = Options()
    names = {field.name for field in fields(Options)} - {"tuning"}
    for part in filter(None, (p.strip() for p in text.split(","))):
        if part == "baseline":
            options = replace(BASELINE, tuning=options.tuning)
            continue
        name, sep, value = part.partition("=")
        name, value = name.strip(), value.strip().lower()
        if sep and name.startswith(TUNE):
            knob = name[len(TUNE):]
            if knob not in TUNING_NAMES:
                raise ValueError(f"unknown tuned value {knob!r}: expected one of {', '.join(TUNING_NAMES)}")
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f"tuned value {knob} takes a number, not {value!r}") from None
            options = replace(options, tuning=replace(options.tuning, **{knob: number}))
            continue
        if not sep or name not in names:
            raise ValueError(f"unknown script option {part!r}: expected baseline or one of {', '.join(sorted(names))} as name=value")
        if name in _CHOICES:
            if value not in _CHOICES[name]:
                raise ValueError(f"script option {name} takes one of {', '.join(_CHOICES[name])}, not {value!r}")
            options = replace(options, **{name: value})
        elif value in _TRUE:
            options = replace(options, **{name: True})
        elif value in _FALSE:
            options = replace(options, **{name: False})
        else:
            raise ValueError(f"script option {name} takes on or off, not {value!r}")
    return options
