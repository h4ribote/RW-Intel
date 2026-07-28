"""What a recorded thing was made by, as a digest of the code that made it.

A name is not a claim about meaning. A set of parameters carries the list of what its slots are called and that list cannot say what they hold; a teacher carries the same list and cannot say which rule answered the boards below it. Both failures have happened here and neither was visible. The strategic cut compared this side's army with the enemy's whole side until it was corrected to armies against armies, and the slot stayed named `military_edge` while every file stayed byte for byte identical. The tactical ladder answered a fifth of its boards by walking in on a squad that was outranged until that was measured out of it, and a teacher recorded the day before states exactly the encoding a teacher recorded the day after states.

So a recorded thing states a digest of the code that produced it, and two digests that disagree retire the file. What is digested is walked from an entry point outward — the instructions of each function, the names it touches, the constants it carries, and the module-level figures any of them reads — so that a helper rewritten under an unchanged caller is caught and a change to one layer does not retire another's.

Three limits, and all three are stated rather than worked around.

**Prose does not count.** A function's docstring is skipped where the caller knows what it is. The explanations in this codebase are the bulk of it, and a digest that turned every edit to one into a retired training run is one nobody could keep.

**The walk stops at what the named namespaces do not define.** Built-ins, imported types and anything reached through an object at run time are outside it. A quantity can change meaning further out — in the view a feature is read off, or in what the game reports — and no digest taken here would move.

**A table defined elsewhere is digested as a table.** What is not a number, a name, a container or an enumerated member is digested as its kind alone, because a container built out of hashed keys does not repr in a fixed order and a digest that changed from one run to the next would retire everything every time anything loaded it.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, List, Sequence


def value_digest(value: Any) -> str:
    """A figure as text that says the same thing on every run.

    Written out rather than left to `repr` because the orders inside a set or a dictionary of hashed keys are not fixed between runs, and a digest that moved would be a digest that retires every file every time.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str, bytes)):
        return repr(value)
    if isinstance(value, Enum):
        return "%s.%s" % (type(value).__name__, value.name)
    if isinstance(value, (tuple, list)):
        return "[%s]" % ",".join(value_digest(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return "{%s}" % ",".join(sorted(value_digest(item) for item in value))
    if isinstance(value, dict):
        return "{%s}" % ",".join(sorted("%s:%s" % (value_digest(key), value_digest(item))
                                        for key, item in value.items()))
    return type(value).__name__


def code_digest(code, docstring: str = None) -> str:
    """One code object as a digest of what it does rather than of how it reads: the instructions, the names it touches and the constants it carries.

    A nested code object — a comprehension, a lambda — is digested the same way, since that is where a fair share of the arithmetic in this project actually lives.
    """
    parts = [code.co_name, str(code.co_argcount), code.co_code.hex(),
             ",".join(code.co_names), ",".join(code.co_varnames)]
    for index, constant in enumerate(code.co_consts):
        if index == 0 and docstring is not None and constant == docstring:
            continue
        if hasattr(constant, "co_code"):
            parts.append(code_digest(constant))
        elif isinstance(constant, (set, frozenset, dict)):
            # The only constants whose `repr` is not fixed between runs, and the compiler makes them: a membership test against a literal set becomes a frozenset constant. Everything else is written by `repr`, which is what the digests already recorded were computed with, so that changing this machinery does not retire every file that ever stated one.
            parts.append(value_digest(constant))
        else:
            parts.append(repr(constant))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def digest(namespaces: Sequence[Any], entries: Sequence[str], length: int = 16) -> str:
    """What a set of entry points is made of, as one short digest.

    The namespaces are searched in the order given, which is what lets a class be handed in before the module it lives in: a method reaching `self._spent` puts `_spent` in its own code's names, and the class is where that resolves. Walked from the entries outward rather than taken over a whole module, so that two things defined side by side retire separately when only one of them moves.
    """
    seen: set = set()
    queue: List[str] = list(entries)
    parts: List[str] = []
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        holder = next((space for space in namespaces if hasattr(space, name)), None)
        if holder is None:
            continue
        seen.add(name)
        value = getattr(holder, name)
        code = getattr(value, "__code__", None)
        if code is None:
            parts.append("%s=%s" % (name, value_digest(value)))
            continue
        parts.append("%s:%s" % (name, code_digest(code, getattr(value, "__doc__", None))))
        queue.extend(sorted(set(code.co_names)))
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()[:length]


def rule_digests() -> dict:
    """What each layer's handwritten rule is made of, one digest apiece.

    A teacher is a recording of a rule, and the rule is the half of it nothing could state. The eighth tactical departure is the case that showed it: the ladder answered a fifth of its boards by walking in on a squad that was outranged, that was measured to cost it 0.0675 and taken out, and a teacher recorded before the change states exactly the encoding one recorded after states. Fitting to the old file would produce a network imitating a rule that no longer exists, and reporting a perfectly ordinary accuracy for it.

    Walked from each layer's own deciding entry point, with the class handed in before its module so a method reaching its own helpers resolves. What is deliberately NOT here is the reporting half of these layers: what a teacher records is the choice, so what has to be digested is what makes the choice.
    """
    from ..control.policy import operations as operations_module
    from ..control.policy import strategy as strategy_module
    from ..control.policy import tactics as tactics_module

    return {
        "tactics": digest([tactics_module.Tactics, tactics_module], ["_departure"]),
        "operations": digest([operations_module.Operations, operations_module],
                             ["_plan", "_target", "_settled", "_pick", "_vanguard", "_garrison",
                              "_raid", "_admit", "_priority", "_reach", "_price", "_changed"]),
        "strategy": digest([strategy_module.Strategy, strategy_module],
                           ["_transition", "_overrun", "_income_levelled_off", "_mix",
                            "_priorities", "_score"]),
    }
