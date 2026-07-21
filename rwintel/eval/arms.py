"""The things a comparison compares.

An arm is a name and a way of making a policy. The point of naming them is that a number is only worth keeping if what produced it is written down beside it, and "the script policy" stops identifying anything the moment there is more than one way to run it.

What can be varied here is deliberately narrow. A comparison is only meaningful when one thing differs and everything else is held, so an arm is built by taking the chain as it is and pinning one decision, rather than by assembling a different chain. Pinning the posture is the first of those because the interface for it already exists and is not a test fixture: it is the same handle a human uses to take the strategic layer over, which the design calls the layer worth the least to learn and the most to hand across.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from ..control.policy import ScriptPolicy, script_policy
from ..control.policy.contracts import Posture

Arm = Tuple[str, Callable]


def pinned(posture: Posture) -> Callable:
    """The chain with the strategic layer held at one posture for the whole match."""

    def build(session) -> ScriptPolicy:
        policy = script_policy(session)
        policy.strategy.forced = posture
        return policy

    return build


def parse(name: str) -> Arm:
    """An arm from its written name: `script` for the chain as it decides for itself, or a posture's name for the chain pinned to it."""
    if name == "script":
        return name, script_policy
    key = name.upper()
    if key in Posture.__members__:
        return name.lower(), pinned(Posture[key])
    raise ValueError(f"no arm named {name!r}: expected 'script' or one of {', '.join(p.name.lower() for p in Posture)}")


def parse_all(names: List[str]) -> List[Arm]:
    arms = [parse(name) for name in names]
    if len({name for name, _ in arms}) != len(arms):
        raise ValueError("two arms of a comparison cannot share a name")
    return arms
