"""Model files: one format for every network, flat or set, of every layer.

A file holds `{"format": 1, "layer", "kind": "flat" | "set", "config": {...}, "version": int, "state": state_dict}`, written atomically. `config` is the constructor's keyword arguments, plus whatever a learner adds about how the parameters were made (a reward scale, for example), which the constructor does not take. `load` builds the right class from the file, and also reads a bare state dict as written before this format, as the flat network of its layer, its widths read off the parameter shapes.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Optional

import torch
from torch import nn

from .. import paths
from ..control.policy.encoding import OPERATIONAL_SIZE, SQUAD_SLOTS, TACTICAL_SIZE
from .net import EconomicNet, OperationalNet, TacticalNet
from .setnet import SET_NETS

log = logging.getLogger(__name__)

FORMAT = 1

FLAT_NETS = {"tactics": TacticalNet, "operations": OperationalNet, "economy": EconomicNet}

#: Parameter names a bare state dict written before the action-value heads existed does not carry.
_LATER_HEADS = ("q.", "q_region.", "q_plan.")


def limit_card(device: torch.device, share: Optional[float]) -> None:
    """Limits this process to `share` of a graphics card's memory, so that the caching allocator returns its unused blocks before going over and raises an out-of-memory error at the share instead of spilling into shared host memory; None, or a device other than a graphics card, sets no limit."""
    if share is None:
        return
    if not 0.0 < share <= 1.0:
        raise ValueError(f"a share of the card is above nought and at most one, not {share}")
    if device.type != "cuda":
        return
    torch.cuda.set_per_process_memory_fraction(float(share), device.index)
    total = torch.cuda.get_device_properties(device).total_memory
    log.info("this process may take %g of the card, %.2f GB", share, share * total / 1e9)


def kind_of(net: nn.Module) -> str:
    return getattr(net, "KIND", "flat")


def layer_of(net: nn.Module) -> str:
    return getattr(net, "LAYER")


def _class(layer: str, kind: str):
    table = SET_NETS if kind == "set" else FLAT_NETS
    if kind not in ("flat", "set") or layer not in table:
        raise ValueError(f"no {kind} network for the {layer} layer")
    return table[layer]


def build(layer: str, kind: str = "flat", config: Optional[dict] = None) -> nn.Module:
    """A fresh network of a layer and kind from constructor keywords; keys the constructor does not take are ignored."""
    cls = _class(layer, kind)
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{name: value for name, value in (config or {}).items() if name in accepted})


def file_config(written: dict) -> dict:
    """A model file's config with the class's `FILE_DEFAULTS` under it, which is what a file written before a constructor keyword existed was built with."""
    defaults = getattr(_class(written["layer"], written["kind"]), "FILE_DEFAULTS", {})
    return {**defaults, **written["config"]}


def save(net: nn.Module, path: str, version: int = 0, extra: Optional[dict] = None) -> None:
    """Writes a network with what is needed to build it again, replacing `path` in one step."""
    config = dict(getattr(net, "config", {}))
    config.update(extra or {})
    record = {"format": FORMAT, "layer": layer_of(net), "kind": kind_of(net), "config": config,
              "version": int(version), "state": {name: value.detach().cpu() for name, value in net.state_dict().items()}}
    with paths.replacing(path, "wb") as handle:
        torch.save(record, handle)


def read(path: str) -> dict:
    """A model file as written, with a bare state dict presented in the same form (`format` 0, kind flat)."""
    written = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(written, dict) and "format" in written and "state" in written:
        return written
    return {"format": 0, "layer": _bare_layer(written), "kind": "flat", "config": _bare_config(written),
            "version": 0, "state": written}


def _bare_layer(state: dict) -> str:
    inputs = state["body.0.weight"].shape[1]
    if inputs == TACTICAL_SIZE:
        return "tactics"
    if inputs == OPERATIONAL_SIZE + SQUAD_SLOTS:
        return "operations"
    return "economy"


def _bare_config(state: dict) -> dict:
    config = {"width": int(state["body.0.weight"].shape[0])}
    if "offer.0.weight" in state:
        config["offer_width"] = int(state["offer.0.weight"].shape[0])
    return config


def load(path: str, layer: Optional[str] = None, device=None, kind: Optional[str] = None) -> nn.Module:
    """The network in a model file, of the class the file names, on `device`, in evaluation mode. `layer` and `kind`, when given, are checked against the file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no parameters at {path}")
    written = read(path)
    if layer is not None and written["layer"] != layer:
        raise ValueError(f"{path} holds a network of the {written['layer']} layer, not of the {layer} layer")
    if kind is not None and written["kind"] != kind:
        raise ValueError(f"{path} holds a {written['kind']} network, not a {kind} one")
    config = file_config(written)
    net = build(written["layer"], written["kind"], config)
    missing, unexpected = net.load_state_dict(written["state"], strict=False)
    late = [name for name in missing if name.startswith(_LATER_HEADS)]
    if unexpected or len(late) != len(missing) or (late and written["format"] != 0):
        raise ValueError(f"{path} does not fit a {written['kind']} {written['layer']} network: "
                         f"missing {sorted(missing)}, unexpected {sorted(unexpected)}")
    net.version = int(written.get("version", 0))
    net.file_config = config
    return net.to(device).eval()


def load_into(net: nn.Module, path: str, device=None) -> nn.Module:
    """Loads the parameters of a model file into a network already built, which has to be of the same layer, kind and shape."""
    loaded = load(path, layer_of(net), kind=kind_of(net))
    net.load_state_dict(loaded.state_dict())
    return net.to(device)
