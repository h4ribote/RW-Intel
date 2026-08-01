"""The things a comparison compares.

An arm is a name and a way of making a policy. The point of naming them is that a number is only worth keeping if what produced it is written down beside it, and "the script policy" stops identifying anything the moment there is more than one way to run it.

What can be varied here is deliberately narrow. A comparison is only meaningful when one thing differs and everything else is held, so an arm is built by taking the chain as it is and replacing one decision, rather than by assembling a different chain. Pinning the posture is the first of those because the interface for it already exists and is not a test fixture: it is the same handle a human uses to take the strategic layer over, which the design calls the layer worth the least to learn and the most to hand across. Loading a learnt operational layer is the second, and it holds everything else the same in the same way: the chain is the script with exactly one decision taken from a network, so beating the script means that one decision got better and nothing else moved.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from ..control.policy import ScriptPolicy, script_policy
from ..control.policy.contracts import Posture
from ..control.policy.operations import Concentrated as _Concentrated

log = logging.getLogger(__name__)

Arm = Tuple[str, Callable]

#: How a learnt-layer arm names the file it loads, as in `ops:local/operations.pt` or `strategy:local/strategy.pt`. The prefix is what tells a path from a posture, and it names the layer. Two of the three are what a match measures on their own: the operational layer, whose deployment is read against the script's, and the strategic layer, whose posture is read against the rule's and against a posture pinned for the whole match. **The tactical prefix is here for a different reason.** A tactical layer's own strength is read on the engagement arena and nowhere else, but a layer trained with a trained fighter frozen beneath it was trained in that environment, and measuring it with the handwritten fighter underneath measures it somewhere else. The chain an arm carries has to be the chain the layer was trained in, or the number is about neither.
OPERATIONAL_PREFIXES = ("ops", "operations")
STRATEGIC_PREFIXES = ("strategy", "strat")
TACTICAL_PREFIXES = ("tactics", "tactical")


def pinned(posture: Posture) -> Callable:
    """The chain with the strategic layer held at one posture for the whole match."""

    def build(session) -> ScriptPolicy:
        policy = script_policy(session)
        policy.strategy.forced = posture
        return policy

    return build


def parse(name: str) -> Arm:
    """An arm from its written name: `script` for the chain as it decides for itself, or a posture's name for the chain pinned to it. A learnt operational arm (`ops:<path>`) is not built here because it loads a network and stands up a batching server, neither of which belongs in a name a control process running only scripts must be able to parse without a tensor library; it is built by `build_all`."""
    if name == "script":
        return name, script_policy
    key = name.upper()
    if key in Posture.__members__:
        return name.lower(), pinned(Posture[key])
    raise ValueError(f"no arm named {name!r}: expected 'script', 'ops:<path>', 'strategy:<path>', 'tactics:<path>', 'ops-pin', 'ops-concentrate', or one of {', '.join(p.name.lower() for p in Posture)}")


#: What joins two learnt layers into one arm, as in `ops:local/operations.pt+strategy:local/strategy.pt`. Chosen because a path may contain a comma on some systems and never contains a plus in this project's own naming.
LAYER_JOIN = "+"


#: Every prefix that names a layer, so that one list answers both "is this a learnt arm" and "which layer is it".
LEARNT_PREFIXES = OPERATIONAL_PREFIXES + STRATEGIC_PREFIXES + TACTICAL_PREFIXES


def _is_learnt(name: str) -> bool:
    for part in name.split(LAYER_JOIN):
        prefix, separator, _ = part.partition(":")
        if not separator or prefix not in LEARNT_PREFIXES:
            return False
    return True


def _width_of(state) -> dict:
    """The hidden width the parameters in this file were fitted at, as the keyword every one of the three networks takes.

    Read off the file rather than left at the module's default, because all three networks begin with one linear map from the features to the width, so a file states its own width and a default-width network refuses to take a wider one. Nothing here can build a fresh network — an arm is always a file — so there is no flag this can come to disagree with. An empty answer where the file has no first layer to read, which leaves the network at its default and the strict load below to refuse it.
    """
    from ..learn.net import INPUT_WEIGHT

    weight = state.get(INPUT_WEIGHT) if isinstance(state, dict) else None
    if weight is None or not hasattr(weight, "dim") or weight.dim() != 2:
        return {}
    return {"width": int(weight.shape[0])}


def learnt(name: str, device: Optional[str] = None, greedy: bool = False) -> Tuple[Arm, object]:
    """A learnt layer, or a chain of them, loaded from files, as an arm of a match comparison.

    Every layer the arm does not name is the script it is measured against, exactly as a training run holds it, and each layer named is handed no rollout, so with nowhere to record a decision it records none: this reads the networks, it does not learn them. One network is loaded and one batching server answers every instance's decisions through it, the same arrangement the duel uses for the tactical layer; the server is returned for the run to stop, because a `(name, build)` pair has nowhere to keep it.

    Refused rather than started from nothing when the file is not there, which is what the duel does and for the same reason: a comparison that quietly scored a freshly initialised policy would produce a perfectly plausible number about a policy nobody asked about.

    The intruder the design requires under evaluation is not built in here. It is attached uniformly to every arm by the session from the run's `--intrude`, so building one into this arm alone would disturb the learnt side and not the script it is measured against.

    How the policy is read — drawn from, or taken at its likeliest action — is the run's to say and is said out loud, because it is a real difference and one this project has an open question about. Every match measurement recorded so far was taken drawn, which is why that is the default; the arena's measuring runner reads its learnt arm greedily, and a number taken one way is not a number taken the other.
    """
    import os

    parts = [part.strip() for part in name.split(LAYER_JOIN) if part.strip()]
    if not parts:
        raise ValueError(f"a learnt arm needs a path, as in 'ops:local/operations.pt', not {name!r}")

    # Imported here rather than at the top of the module so that a control process running only script and posture arms never loads the tensor library, which is the same discipline the deciders keep.
    import torch

    from ..learn.deciders import (NetworkOperations, NetworkStrategy, NetworkTactics,
                                  operational_batcher, strategic_batcher, tactical_batcher)
    from ..learn.net import (EncodingRefused, OperationalNet, StrategicNet, TacticalNet,
                             load_encoded)
    from ..learn.policy import OPERATIONAL, STRATEGIC, TACTICAL, LearningPolicy

    # What each prefix names and what builds it. Written once here rather than as a chain of conditionals, because a third layer turned every two-way choice into a place the three could come to disagree about which network reads which file.
    kinds = {prefix: (layer, build_net, build_batcher, build_decider)
             for prefixes, layer, build_net, build_batcher, build_decider in (
                 (OPERATIONAL_PREFIXES, OPERATIONAL, OperationalNet, operational_batcher, NetworkOperations),
                 (STRATEGIC_PREFIXES, STRATEGIC, StrategicNet, strategic_batcher, NetworkStrategy),
                 (TACTICAL_PREFIXES, TACTICAL, TacticalNet, tactical_batcher, NetworkTactics))
             for prefix in prefixes}

    # The games this process is scored beside run on these cores; a library that helps itself to all of them turns every inference into a fight with the simulation it is measuring.
    torch.set_num_threads(2)
    where = torch.device(device) if device else torch.device("cpu")
    loaded = []
    batchers = []
    try:
        for part in parts:
            prefix, _, path = part.partition(":")
            path = path.strip()
            if not path:
                raise ValueError(f"a learnt arm needs a path, as in 'ops:local/operations.pt', not {part!r}")
            if not os.path.exists(path):
                raise ValueError(f"there are no parameters at {path} to measure")
            if prefix not in kinds:
                raise ValueError(f"no layer named {prefix!r} to load in the arm {name!r}: expected one of {', '.join(sorted(kinds))}")
            layer, build_net, build_batcher, build_decider = kinds[prefix]
            if any(layer == held for held, _, _, _ in loaded):
                raise ValueError(f"the {layer} layer is named twice in the arm {name!r}, so one of the two would never play")
            state = torch.load(path, map_location=where)
            # The width is read off the file rather than left at the module's default, exactly as the arena's frozen tactical layer reads it. A network trained wider than the default loaded into a default-width one raises on the shapes, so a run that swept the width could measure nothing it had trained: the option existed and the measurement it was for did not.
            net = build_net(**_width_of(state)).to(where)
            # Refused for the same reason a missing file is: a number about parameters that read the board differently from the way they were fitted to read it is a plausible number about nothing, and the shapes all match, so nothing later in the run would notice.
            try:
                avowal = load_encoded(net, state)
            except EncodingRefused as refused:
                raise ValueError(f"the parameters at {path} cannot be measured: {refused}")
            if avowal:
                # Said out loud beside the arm it is about, because a number is only worth keeping if what produced it is written down beside it, and what produced this one is parameters whose feature list nothing in the file could prove.
                log.warning("the feature list at %s is a person's word and not a fit's record: %s", path, avowal)
            batcher = build_batcher(net, device=where, greedy=greedy)
            batchers.append(batcher)
            loaded.append((layer, net, batcher, build_decider))
            log.info("the %s arm reads the %s layer at %s %s", _stem(name), layer, path,
                     "at its likeliest action" if greedy else "by drawing from it")
    except BaseException:
        # A later layer failing must not leave an earlier one's inference thread running against an arm that will never play.
        for batcher in batchers:
            batcher.stop()
        raise

    def build(session) -> LearningPolicy:
        # A fresh decider per session because it answers for one instance; the network behind it is shared, which is the whole point of batching the inference across instances. No rollout anywhere, so every layer here decides and writes nothing down. The decider carries the same reading as the server, since one of them answers when there is a server and the other when there is not.
        deciders = {layer: build_decider(net, where, batcher, greedy=greedy)
                    for layer, net, batcher, build_decider in loaded}
        # One of them is named as the arm's layer and the rest are frozen beside it, which is the same construction either way: a frozen layer and a measured one differ only in the rollout, and neither has one here.
        first = loaded[0][0]
        return LearningPolicy(session, first, deciders[first], None, session.instance,
                              frozen={layer: decider for layer, decider in deciders.items() if layer != first})

    # One layer hands back its own server rather than a wrapper around it, because callers that read a single
    # layer's answers directly - the tests that check what a network replies - hold the server itself.
    return (_stem(name), build), (batchers[0] if len(batchers) == 1 else _Batchers(batchers))


class _Batchers:
    """Several inference servers behind one arm, standing in for one. The run holds one object per arm, stops it when it is done and reports how its batching went, and an arm that carries two learnt layers has two of everything to answer for rather than one."""

    def __init__(self, batchers) -> None:
        self.batchers = list(batchers)

    def stop(self) -> None:
        for batcher in self.batchers:
            batcher.stop()

    @property
    def calls(self) -> int:
        return sum(batcher.calls for batcher in self.batchers)

    @property
    def served(self) -> int:
        return sum(batcher.served for batcher in self.batchers)

    @property
    def batch_size(self) -> float:
        """How the batching went across every server this arm started, pooled over their calls rather than averaged over the servers. Two layers deciding at different periods make very different numbers of calls, and a mean of the two means would let the rarer layer's figure count as much as the commoner one's."""
        return self.served / self.calls if self.calls else 0.0


def _stem(name: str) -> str:
    """What an arm carrying learnt layers is called: the file stems joined the way the layers were, so a chain of two is named for both and never merges with either alone in a journal."""
    import os

    return LAYER_JOIN.join(os.path.splitext(os.path.basename(part.partition(":")[2].strip()))[0]
                           for part in name.split(LAYER_JOIN) if part.strip())


def concentrating(name: str = "ops-concentrate") -> Arm:
    """The chain with the operational layer's choice of region replaced by the one the strategic layer wants most, for the match runner.

    The arena says this arm is worth measuring in a match. On the constructed board it beats the handwritten ladder by about 0.08 to 0.10 of the side score and takes about a fifth of the discs it could only gain by taking, against the ladder's twentieth — and taking ground is not an arena-shaped skill: income comes from extractors standing on resource points, so ground is upstream of the economy and the economy is where this chain loses. Measured over 480 matches, the chain finishes on 35.5 income against a difficulty-1 opponent's 77.9 and 9,157 credits of standing value against 25,662.

    Whether the arena's advantage carries into a match is what this arm was built to ask, and it has since been answered for this one: over 96 matches the concentrating chain scored 0.065 above the ladder's, which is the same direction and the same order as the 0.076 to 0.095 it stood above it on the constructed board (see the operational record). That is one arm on two maps rather than a general statement about the arena, and the design's reason for doubting it stands: that board scores holding and taking ground and nothing else, while a match asks what the ground was for. No network, so nothing is loaded and nothing has to be torn down.
    """
    from ..learn.policy import OPERATIONAL, LearningPolicy

    def build(session) -> LearningPolicy:
        # The concentrating rule is a script layer, not a decider, so it replaces the operational layer whole rather than answering for it. No rollout: read from, not learnt from.
        policy = script_policy(session)
        policy.operations = _Concentrated(session, policy.catalogue)
        return policy

    return (name, build)


def pinned_operational(name: str = "ops-pin") -> Arm:
    """The chain with the operational layer pinned to one legal region and task, for the match runner.

    The operational analogue of the arena's `--pin`: a constant deployment against which a match is read, to find whether the region-and-task choice moves the match at all. If the score with every squad sent to the same region on the same task is the same as the script's careful choice, the choice was not what moved the match. No network, so nothing is loaded and nothing has to be torn down, and it needs no path.
    """
    from ..learn.deciders import PinnedRegion
    from ..learn.policy import OPERATIONAL, LearningPolicy

    def build(session) -> LearningPolicy:
        # No rollout: read from, not learnt from.
        return LearningPolicy(session, OPERATIONAL, PinnedRegion(), None, session.instance)

    return (name, build)


def build_all(names: List[str], device: Optional[str] = None, greedy: bool = False) -> Tuple[List[Arm], List[object]]:
    """Every arm of a comparison, and the inference servers any of them started.

    Script and pinned-posture arms need nothing torn down and start no server. A learnt operational arm loads a network once and answers every instance's decisions through one batching server, so the server is returned alongside the arms for the run to stop when it is done. Two arms of a comparison cannot share a name: journalled and reported under one name they would merge into one, and the run would silently measure half of what it was asked for.
    """
    arms: List[Arm] = []
    batchers: List[object] = []
    try:
        for name in names:
            if name == "ops-pin":
                arms.append(pinned_operational(name))
            elif name == "ops-concentrate":
                arms.append(concentrating(name))
            elif _is_learnt(name):
                arm, batcher = learnt(name, device, greedy)
                arms.append(arm)
                batchers.append(batcher)
            else:
                arms.append(parse(name))
        if len({name for name, _ in arms}) != len(arms):
            raise ValueError("two arms of a comparison cannot share a name")
    except BaseException:
        # A later arm failing must not leave an earlier learnt arm's inference thread running against a run that will never start.
        for batcher in batchers:
            batcher.stop()
        raise
    return arms, batchers
