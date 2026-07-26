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

#: How a learnt-layer arm names the file it loads, as in `ops:local/operations.pt` or `strategy:local/strategy.pt`. The prefix is what tells a path from a posture, and it names the layer because two of the three are measured on a match: the operational layer, whose deployment is read against the script's, and the strategic layer, whose posture is read against the rule's and against a posture pinned for the whole match. The tactical layer is not among them — it is measured on the engagement arena, off any match at all.
OPERATIONAL_PREFIXES = ("ops", "operations")
STRATEGIC_PREFIXES = ("strategy", "strat")


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
    raise ValueError(f"no arm named {name!r}: expected 'script', 'ops:<path>', 'strategy:<path>', 'ops-pin', 'ops-concentrate', or one of {', '.join(p.name.lower() for p in Posture)}")


def _is_learnt(name: str) -> bool:
    prefix, separator, _ = name.partition(":")
    return bool(separator) and prefix in OPERATIONAL_PREFIXES + STRATEGIC_PREFIXES


def learnt(name: str, device: Optional[str] = None, greedy: bool = False) -> Tuple[Arm, object]:
    """A learnt layer, loaded from a file, as an arm of a match comparison.

    Everything but the operational decision is the script it is measured against, exactly as a training run holds it, and the layer is handed no rollout, so with nowhere to record a decision it records none: this reads the network, it does not learn it. One network is loaded and one batching server answers every instance's decisions through it, the same arrangement the duel uses for the tactical layer; the server is returned for the run to stop, because a `(name, build)` pair has nowhere to keep it.

    Refused rather than started from nothing when the file is not there, which is what the duel does and for the same reason: a comparison that quietly scored a freshly initialised policy would produce a perfectly plausible number about a policy nobody asked about.

    The intruder the design requires under evaluation is not built in here. It is attached uniformly to every arm by the session from the run's `--intrude`, so building one into this arm alone would disturb the learnt side and not the script it is measured against.

    How the policy is read — drawn from, or taken at its likeliest action — is the run's to say and is said out loud, because it is a real difference and one this project has an open question about. Every match measurement recorded so far was taken drawn, which is why that is the default; the arena's measuring runner reads its learnt arm greedily, and a number taken one way is not a number taken the other.
    """
    import os

    prefix, _, path = name.partition(":")
    path = path.strip()
    if not path:
        raise ValueError(f"a learnt arm needs a path, as in 'ops:local/operations.pt', not {name!r}")
    if not os.path.exists(path):
        raise ValueError(f"there are no parameters at {path} to measure")

    # Imported here rather than at the top of the module so that a control process running only script and posture arms never loads the tensor library, which is the same discipline the deciders keep.
    import torch

    from ..learn.deciders import (NetworkOperations, NetworkStrategy, operational_batcher,
                                  strategic_batcher)
    from ..learn.net import EncodingRefused, OperationalNet, StrategicNet, load_encoded
    from ..learn.policy import OPERATIONAL, STRATEGIC, LearningPolicy

    strategic = prefix in STRATEGIC_PREFIXES
    layer = STRATEGIC if strategic else OPERATIONAL
    # The games this process is scored beside run on these cores; a library that helps itself to all of them turns every inference into a fight with the simulation it is measuring.
    torch.set_num_threads(2)
    where = torch.device(device) if device else torch.device("cpu")
    net = (StrategicNet() if strategic else OperationalNet()).to(where)
    state = torch.load(path, map_location=where)
    # Refused for the same reason a missing file is: a number about parameters that read the board differently from the way they were fitted to read it is a plausible number about nothing, and the shapes all match, so nothing later in the run would notice.
    try:
        avowal = load_encoded(net, state)
    except EncodingRefused as refused:
        raise ValueError(f"the parameters at {path} cannot be measured: {refused}")
    if avowal:
        # Said out loud beside the arm it is about, because a number is only worth keeping if what produced it is written down beside it, and what produced this one is parameters whose feature list nothing in the file could prove.
        log.warning("the feature list at %s is a person's word and not a fit's record: %s", path, avowal)
    batcher = (strategic_batcher if strategic else operational_batcher)(net, device=where, greedy=greedy)
    log.info("the %s arm reads the %s layer at %s %s", os.path.splitext(os.path.basename(path))[0], layer, path,
             "at its likeliest action" if greedy else "by drawing from it")

    def build(session) -> LearningPolicy:
        # A fresh decider per session because it answers for one instance; the network behind it is shared, which is the whole point of batching the inference across instances. No rollout, so the layer decides and writes nothing down. The decider carries the same reading as the server, since one of them answers when there is a server and the other when there is not.
        decider = (NetworkStrategy if strategic else NetworkOperations)(net, where, batcher, greedy=greedy)
        return LearningPolicy(session, layer, decider, None, session.instance)

    return (os.path.splitext(os.path.basename(path))[0], build), batcher


def concentrating(name: str = "ops-concentrate") -> Arm:
    """The chain with the operational layer's choice of region replaced by the one the strategic layer wants most, for the match runner.

    The arena says this arm is worth measuring in a match. On the constructed board it beats the handwritten ladder by about 0.08 to 0.10 of the side score and takes about a fifth of the discs it could only gain by taking, against the ladder's twentieth — and taking ground is not an arena-shaped skill: income comes from extractors standing on resource points, so ground is upstream of the economy and the economy is where this chain loses. Measured over 480 matches, the chain finishes on 35.5 income against a difficulty-1 opponent's 77.9 and 9,157 credits of standing value against 25,662.

    Whether the arena's advantage carries into a match is exactly what has never been measured, and the design says why it might not: that board scores holding and taking ground and nothing else, while a match asks what the ground was for. No network, so nothing is loaded and nothing has to be torn down.
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
