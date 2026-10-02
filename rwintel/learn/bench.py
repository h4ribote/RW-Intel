"""What one inference call costs, per batch size and device: the figures the batching limits (`inference.LIMITS`) are chosen from.

    python -m rwintel.learn bench --layer tactics --net set --device cuda
    python -m rwintel.learn bench --layer operations --load local/models/ops.pt --device cpu --sizes 1,8,64

Requests are synthetic and of the right shape: tactical set rows with a random number of members and threats present, flat states drawn at random, operational states with every region and plan open. Each size is warmed up and then timed through the real `evaluate_single` / `evaluate_operational`, including the transfer of the answers back (`_fetch`), so a figure is what a batcher's call costs.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from .. import paths
from ..control.policy.encoding import (
    ECONOMIC_CONTEXT_SIZE,
    ECONOMIC_SIZE,
    GLOBAL_SIZE,
    INVESTMENT_SIZE,
    INVESTMENT_SLOTS,
    OPERATIONAL_PLANS,
    OPERATIONAL_REGIONS,
    OPERATIONAL_SIZE,
    REGION_SIZE,
    SQUAD_SLOTS,
    TACTICAL_ACTIONS,
    TACTICAL_SIZE,
)
from . import models, tokens
from .deciders import evaluate_operational, evaluate_single

log = logging.getLogger(__name__)

WARMUP = 10


def _tactical_requests(count: int, set_rows: bool, draw: np.random.Generator) -> List[tuple]:
    mask = [1.0] * TACTICAL_ACTIONS
    if not set_rows:
        return [(draw.standard_normal(TACTICAL_SIZE).tolist(), mask) for _ in range(count)]
    flags = tokens.SQUAD_TOKEN + (tokens.MEMBER_CAP + tokens.THREAT_CAP) * tokens.UNIT_SIZE
    requests = []
    for _ in range(count):
        row = draw.standard_normal(tokens.SET_SIZE)
        row[flags:] = 0.0
        row[flags:flags + int(draw.integers(1, tokens.MEMBER_CAP + 1))] = 1.0
        threats = int(draw.integers(0, tokens.THREAT_CAP + 1))
        row[flags + tokens.MEMBER_CAP:flags + tokens.MEMBER_CAP + threats] = 1.0
        requests.append((row.tolist(), mask))
    return requests


def _economic_requests(count: int, draw: np.random.Generator) -> List[tuple]:
    requests = []
    for _ in range(count):
        state = draw.standard_normal(ECONOMIC_SIZE)
        for slot in range(INVESTMENT_SLOTS):
            state[ECONOMIC_CONTEXT_SIZE + slot * INVESTMENT_SIZE] = 1.0
        requests.append((state.tolist(), [1.0] * INVESTMENT_SLOTS))
    return requests


def _operational_requests(count: int, draw: np.random.Generator) -> List[tuple]:
    requests = []
    for index in range(count):
        state = draw.standard_normal(OPERATIONAL_SIZE)
        for region in range(OPERATIONAL_REGIONS):
            state[GLOBAL_SIZE + region * REGION_SIZE] = 1.0
        requests.append((state.tolist(), index % SQUAD_SLOTS, [1.0] * OPERATIONAL_REGIONS,
                         [[1.0] * OPERATIONAL_PLANS for _ in range(OPERATIONAL_REGIONS)]))
    return requests


def requests_for(layer: str, kind: str, count: int, seed: int = 0) -> List[tuple]:
    """`count` synthetic requests of the shape a `layer` network of `kind` is asked."""
    draw = np.random.default_rng(seed)
    if layer == "tactics":
        return _tactical_requests(count, kind == "set", draw)
    if layer == "economy":
        return _economic_requests(count, draw)
    return _operational_requests(count, draw)


def run(layer: str, kind: str = "flat", load: Optional[str] = None, device="cpu",
        sizes: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256), repeats: int = 200) -> dict:
    """Times the real evaluation of a `layer` network, loaded from `load` or freshly built of `kind`, for each batch size."""
    device = torch.device(device)
    net = models.load(load, layer, device) if load else models.build(layer, kind).to(device).eval()
    kind = models.kind_of(net)
    evaluate = evaluate_operational if layer == "operations" else evaluate_single
    rows = []
    for size in sizes:
        requests = requests_for(layer, kind, size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(WARMUP):
            evaluate(net, requests, device)
        timings = []
        for _ in range(repeats):
            began = time.perf_counter()
            evaluate(net, requests, device)
            timings.append((time.perf_counter() - began) * 1000.0)
        timings.sort()
        median = statistics.median(timings)
        rows.append({"size": int(size), "median_ms": round(median, 4),
                     "p95_ms": round(timings[min(len(timings) - 1, int(0.95 * len(timings)))], 4),
                     "us_per_decision": round(median * 1000.0 / size, 2),
                     "peak_gpu_mb": round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 2)
                     if device.type == "cuda" else 0.0})
    return {"layer": layer, "kind": kind, "device": str(device), "load": load or "", "repeats": repeats,
            "parameters": sum(p.numel() for p in net.parameters()), "rows": rows, "limits": suggested(rows)}


def suggested(rows: Sequence[dict]) -> dict:
    """The batching limits the rule in `inference.LIMITS` reads off a bench: the largest size whose call takes at most twice the single-decision call, and a window of the single-decision latency where that exceeds the processor's window."""
    from .inference import LIMITS

    single = next((row["median_ms"] for row in rows if row["size"] == 1), None)
    if single is None:
        return {}
    fitting = [row["size"] for row in rows if row["median_ms"] <= 2.0 * single]
    window = max(LIMITS["cpu"][0], single / 1000.0)
    return {"window": round(window, 6), "max_batch": max(fitting) if fitting else 1}


def table(result: dict) -> str:
    lines = [f"{result['layer']} {result['kind']} on {result['device']} ({result['parameters']} parameters)",
             "| size | median ms | p95 ms | us per decision | peak GPU MB |", "| --- | --- | --- | --- | --- |"]
    for row in result["rows"]:
        lines.append(f"| {row['size']} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['us_per_decision']:.1f} | "
                     f"{row['peak_gpu_mb']:.1f} |")
    if result["limits"]:
        lines.append(f"limits by the rule: window {result['limits']['window']:.4f} s, max batch {result['limits']['max_batch']}")
    return "\n".join(lines)


def bench_command(arguments) -> int:
    from .__main__ import _device

    device = _device(arguments.device or "auto")
    models.limit_card(device, arguments.card_share)
    sizes = [int(size) for size in str(arguments.sizes).split(",") if size.strip()]
    result = run(arguments.layer, arguments.net, arguments.load, device, sizes, arguments.repeats)
    print(table(result))
    report = arguments.report or os.path.join(paths.local_root(), "reports", "bench", f"{paths.stamp()}.json")
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
    with paths.replacing(report, "w") as handle:
        json.dump(result, handle, indent=1)
    log.info("bench report written to %s", report)
    return 0
