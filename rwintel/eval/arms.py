"""The things a comparison compares.

An arm is a name and a way of making a policy. The point of naming them is that a number is only worth keeping if what produced it is written down beside it, and "the script policy" stops identifying anything the moment there is more than one way to run it.

What can be varied here is deliberately narrow. A comparison is only meaningful when one thing differs and everything else is held, so an arm is built by taking the chain as it is and pinning one decision, rather than by assembling a different chain. Pinning the posture is the first of those because the interface for it already exists and is not a test fixture: it is the same handle a human uses to take the strategic layer over, which the design calls the layer worth the least to learn and the most to hand across.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from ..control.policy import ScriptPolicy, script_policy
from ..control.policy import options as script_options
from ..control.policy.contracts import Posture

Arm = Tuple[str, Callable]

#: Prefix of the arms that run the script chain with some of its rules switched, as ablations.
SCRIPT_OPTIONS = "script:"


def switched(options: script_options.Options) -> Callable:
    """The chain with some of its rules switched off or set to another mode."""

    def build(session) -> ScriptPolicy:
        return script_policy(session, options)

    return build


def pinned(posture: Posture) -> Callable:
    """The chain with the strategic layer held at one posture for the whole match."""

    def build(session) -> ScriptPolicy:
        policy = script_policy(session)
        policy.strategy.forced = posture
        return policy

    return build


#: Written prefixes of the arms that run a learnt operational layer from a parameter file: drawing from its distribution, as it is operated, or taking its most likely answer.
LEARNT_OPERATIONS = {"operations": False, "operations-greedy": True}

#: The same for a learnt economy.
LEARNT_ECONOMY = {"economy": False, "economy-greedy": True}

#: Prefix of the arms that pin the operational layer to one fixed rule, as ablations.
PINNED_OPERATIONS = "ops-"

#: The arm that plays the economy's judge through the learnt layer's machinery.
ECONOMY_SCRIPT = "eco-script"

#: Networks already loaded, by layer, file and whether they answer greedily, so that every session of a run shares one network and one batching server.
_loaded: Dict[Tuple[str, str, bool], Callable] = {}

#: The device asked for with `--device`, or None for the serving rule of the learning runner (`learn.__main__._serving_device`): a set network on a graphics card when there is one, everything else on the processor.
DEVICE: Optional[str] = None


def _decider(layer: str, path: str, greedy: bool) -> Callable:
    """What answers for a learnt layer from the network in `path`, loaded once however many sessions ask."""
    key = (layer, os.path.abspath(path), greedy)
    if key not in _loaded:
        if not os.path.exists(path):
            raise ValueError(f"no parameters at {path}")
        from ..learn import models
        from ..learn.__main__ import _serving_device
        from ..learn.deciders import NetworkChoice, NetworkOperations, choice_batcher, operational_batcher
        from ..learn.policy import OPERATIONAL

        device = _serving_device(DEVICE, path)
        net = models.load(path, layer, device)
        if layer == OPERATIONAL:
            decider = NetworkOperations(net, device, batcher=operational_batcher(net, device=device, greedy=greedy),
                                        greedy=greedy)
        else:
            decider = NetworkChoice(net, device, batcher=choice_batcher(net, device=device, greedy=greedy), greedy=greedy)
        _loaded[key] = lambda session: decider
    return _loaded[key]


def learnt(layer: str, path: str, greedy: bool) -> Callable:
    """The chain with one layer answered by the network in `path`. Nothing is recorded and nothing is learnt; the parameters stay as loaded."""
    decider_for = _decider(layer, path, greedy)

    def build(session):
        from ..learn.policy import LearningPolicy

        return LearningPolicy(session, layer, decider_for(session), None, session.instance)

    return build


def learnt_operations(path: str, greedy: bool) -> Callable:
    """The chain with its operational layer answered by the network in `path`."""
    from ..learn.policy import OPERATIONAL

    return learnt(OPERATIONAL, path, greedy)


def learnt_economy(path: str, greedy: bool) -> Callable:
    """The chain with its economy answered by the network in `path`."""
    from ..learn.policy import ECONOMIC

    return learnt(ECONOMIC, path, greedy)


def economy_script(session):
    """The chain with its economy's judge played through the learnt layer's machinery, which decides the same as the script and also records how many investments were chosen."""
    from ..learn.policy import ECONOMIC, LearningPolicy

    return LearningPolicy(session, ECONOMIC, None, None, session.instance)


def pinned_operations(rule: str) -> Callable:
    """The chain with its operational layer pinned to one fixed rule, or with `script` the script's own rule played through the learnt layer's machinery, which decides the same and also records how well the strategic orders were met."""
    from ..learn.deciders import PinnedOperations

    if rule != "script":
        PinnedOperations(rule)

    def build(session):
        from ..learn.policy import OPERATIONAL, LearningPolicy

        decider = None if rule == "script" else PinnedOperations(
            rule, seed=1000003 * max(0, session.instance) + len(session.records))
        return LearningPolicy(session, OPERATIONAL, decider, None, session.instance)

    return build


def name_of(written: str) -> str:
    """The name an arm written this way is journalled under, without building it: a learnt arm is named after its file and a posture by its name in lower case."""
    prefix, _, path = written.partition(":")
    if path and (prefix in LEARNT_OPERATIONS or prefix in LEARNT_ECONOMY):
        return f"{prefix}-{os.path.splitext(os.path.basename(path))[0]}"
    if written.upper() in Posture.__members__ and not written.startswith(PINNED_OPERATIONS):
        return written.lower()
    return written


def parse(name: str) -> Arm:
    """An arm from its written name.

    `script` is the chain as it decides for itself and a posture's name is the chain pinned to it. `script:<name>=<value>[,...]` is the chain with some of its rules switched (`rwintel.control.policy.options`), and `script:baseline` is the chain with all of them at their original setting. `operations:<path>` and `operations-greedy:<path>` replace the operational layer with the network saved at the path, drawing from it or taking its most likely answer, and `economy:<path>` and `economy-greedy:<path>` do the same for the economy; the arm is named after the file. `ops-<rule>`, for any rule of `PinnedOperations.RULES` (`ops-home`, `ops-nearest`, `ops-carry` ...), pins the operational layer to that fixed rule, and `ops-script` plays the script's rule through the learnt layer so that its record carries the same operational statistics; `eco-script` does that for the economy.
    """
    if name == "script":
        return name, script_policy
    if name.startswith(SCRIPT_OPTIONS):
        return name, switched(script_options.parse(name[len(SCRIPT_OPTIONS):]))
    if name == ECONOMY_SCRIPT:
        return name, economy_script
    prefix, _, path = name.partition(":")
    if path and prefix in LEARNT_OPERATIONS:
        return name_of(name), learnt_operations(path, LEARNT_OPERATIONS[prefix])
    if path and prefix in LEARNT_ECONOMY:
        return name_of(name), learnt_economy(path, LEARNT_ECONOMY[prefix])
    if name.startswith(PINNED_OPERATIONS):
        return name, pinned_operations(name[len(PINNED_OPERATIONS):])
    key = name.upper()
    if key in Posture.__members__:
        return name_of(name), pinned(Posture[key])
    raise ValueError(f"no arm named {name!r}: expected 'script', script:<name>=<value>[,...], "
                     f"one of {', '.join(p.name.lower() for p in Posture)}, "
                     f"operations:<path>, operations-greedy:<path>, economy:<path>, economy-greedy:<path>, "
                     f"ops-<rule> for a rule of home|nearest|weakest|richest|spawn|random|carry, or eco-script")


def parse_all(names: List[str]) -> List[Arm]:
    arms = [parse(name) for name in names]
    if len({name for name, _ in arms}) != len(arms):
        raise ValueError("two arms of a comparison cannot share a name")
    return arms
