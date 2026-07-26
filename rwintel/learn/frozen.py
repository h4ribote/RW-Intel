"""Trained layers read off files and held still underneath the one a run is training.

The design's rule for learning is that exactly one layer moves and every other layer is frozen, and it says which layer the frozen ones should be: the trained ones, in the order tactical, operational, strategic. Until this module existed only half of that was reachable. A training run could freeze the script beneath itself — that is what a `LearningPolicy` does by construction — and the constructed operations arena could freeze trained tactical parameters under both sides of its board, but there was no way at all to put a trained layer under a run in an ordinary match. So the second half of the order, and the whole of the third, could only be run against the handwritten layers they were meant to have improved on.

What is frozen here is read and never learnt from. A frozen layer is handed no rollout, so with nowhere to record a decision it records none, and it is read at its likeliest action rather than drawn from: drawing is the exploration a run needs of the layer it is training, and these are not being trained. That is the same arrangement the arena's frozen fighter has, for the same reasons, and this module is where the two share their loading rather than keeping a second copy of it.

A frozen layer is part of the instrument. Two runs made under different frozen layers are two different environments, and a comparison that pooled them would report the change of environment as a difference between the arms — which is why every layer named here is identified by a digest of the parameters themselves rather than by the path they were read from. A path is a nickname that changes underneath itself: a training run overwrites whatever its save names.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def digest(path: str) -> str:
    """What a set of parameters is called: the first half of the SHA-256 of the file's own bytes.

    The parameters and not the path, for the reason the arena's frozen fighter is named the same way — the path is a nickname a later run overwrites, and two runs made under two different networks would otherwise claim one instrument.
    """
    with open(path, "rb") as handle:
        return "sha256:" + hashlib.sha256(handle.read()).hexdigest()[:16]


def load_layer(layer: str, path: str, device=None) -> Tuple[Callable, object, str]:
    """One trained layer off a file: a factory that makes its decider, the batching server the network answers through, and the name the parameters go by.

    Refused rather than loaded blind when the file is missing, and refused when the feature list beside the parameters is not the one this layer now reads. Both refusals matter more for a frozen layer than for a layer being trained: a run that began from a freshly initialised policy is at least learning, while an instrument built from noise is a whole run measured against nothing, journalled under a trained layer's name, with nothing downstream able to tell.

    One network and one server per layer, shared by every instance of the run, because the network is what is being held still and one copy batched across the instances is the whole point of batching.
    """
    from .policy import OPERATIONAL, STRATEGIC, TACTICAL

    if layer not in (TACTICAL, OPERATIONAL, STRATEGIC):
        raise ValueError("no layer named %r to freeze" % layer)
    if not path:
        raise ValueError("a frozen %s layer needs a path, as in %s:local/%s.pt" % (layer, layer, layer))
    if not os.path.exists(path):
        raise ValueError("there are no parameters at %s to freeze under the run" % path)

    # Imported here rather than at the top of the module, so that a control process running only scripts never loads the tensor library. Every module here that touches a network keeps the same discipline.
    import torch

    from .deciders import (NetworkOperations, NetworkStrategy, NetworkTactics, operational_batcher,
                           strategic_batcher, tactical_batcher)
    from .net import EncodingRefused, OperationalNet, StrategicNet, TacticalNet, load_encoded

    # The games this run is learning from are on these cores; a library that helps itself to all of them turns every inference into a fight with the simulation.
    torch.set_num_threads(2)
    where = torch.device(device) if device else torch.device("cpu")
    build_net, build_batcher, build_decider = {
        TACTICAL: (TacticalNet, tactical_batcher, NetworkTactics),
        OPERATIONAL: (OperationalNet, operational_batcher, NetworkOperations),
        STRATEGIC: (StrategicNet, strategic_batcher, NetworkStrategy),
    }[layer]
    net = build_net().to(where)
    try:
        avowal = load_encoded(net, torch.load(path, map_location=where))
    except EncodingRefused as refused:
        raise ValueError("the parameters at %s cannot be frozen as the %s layer: %s" % (path, layer, refused))
    if avowal:
        # Worth saying at every run rather than only where the word was written: an instrument accepted on somebody's word is still an instrument accepted on somebody's word.
        log.warning("the feature list at %s is a person's word and not a fit's record: %s", path, avowal)
    batcher = build_batcher(net, device=where, greedy=True)
    name = digest(path)
    log.info("freezing the %s layer at %s (%s) beneath this run, read at its likeliest action and recording nothing",
             layer, path, name)
    return (lambda: build_decider(net, where, batcher, greedy=True)), batcher, name


@dataclass
class Frozen:
    """Every layer a run holds still beneath the one it is training, and what it takes to tear them down.

    Empty when a run named none, which is the ordinary case and the one every measurement so far was made under: the layers below are then the handwritten script, which is also what a learnt layer is measured against.
    """

    deciders: Dict[str, Callable] = field(default_factory=dict)
    batchers: List[object] = field(default_factory=list)
    #: What each frozen layer is, by its parameters, so that a run can journal the instrument it was made under.
    names: Dict[str, str] = field(default_factory=dict)

    def build(self) -> Dict[str, object]:
        """A fresh decider for each frozen layer, for one session. The deciders are per session because a decider answers for one instance; the networks behind them are shared."""
        return {layer: make() for layer, make in self.deciders.items()}

    def stop(self) -> None:
        for batcher in self.batchers:
            batcher.stop()

    def as_dict(self) -> Dict[str, str]:
        return dict(self.names)


def frozen_layers(spec: Optional[str], device=None, training: Optional[str] = None) -> Frozen:
    """The layers named in a `tactics:PATH,operations:PATH` specification, loaded and held still.

    Nothing at all when nothing was named. A layer named twice is refused, and so is the layer the run is training: freezing the very layer being trained would leave a run whose gradient goes to a network nothing is reading, reporting updates about a policy that never plays.
    """
    frozen = Frozen()
    if not spec:
        return frozen
    try:
        for piece in str(spec).split(","):
            piece = piece.strip()
            if not piece:
                continue
            layer, separator, path = piece.partition(":")
            if not separator:
                raise ValueError("a frozen layer is written layer:path, as in tactics:local/tactics.pt, not %r" % piece)
            layer, path = layer.strip(), path.strip()
            if layer in frozen.deciders:
                raise ValueError("the %s layer was named twice" % layer)
            if training is not None and layer == training:
                raise ValueError("this run is training the %s layer, so it cannot also freeze one" % layer)
            make, batcher, name = load_layer(layer, path, device)
            frozen.deciders[layer] = make
            frozen.batchers.append(batcher)
            frozen.names[layer] = name
    except BaseException:
        # A later layer failing must not leave an earlier one's inference thread running against a run that will never start.
        frozen.stop()
        raise
    return frozen
