"""The decision-latency meter: how much game time passes between an observation and the step that applies its answer.

Each observation carries the number of the observation whose answer the agent applied at the head of the step that produced it (`Observation.answered`, the timing block), and the game time of that step. A lag is the game time of the applying step less the game time of the observation answered. On the fixed and replay clocks every lag is one tactical period, whatever the machine's speed; a lag of any other length, or an answer never applied, is what the meter is kept to show. The tactical layer answers every observation; the operational layer and the economy decide on the observations that carry the region block, so their figures cover those alone.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

#: The layers a summary reports: the tactical layer on every observation, the operational layer (and the economy beside it) on those carrying the region block.
LAYERS = ("tactics", "operations")


class Latency:
    """Lags of one episode's answers, fed by the session in observation order."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        #: Per observation in arrival order: its number, game time, whether it carried the region block, and the lag of its answer once applied.
        self._seen: List[List] = []
        #: Where in `_seen` the latest observation of each number is; numbers wrap, and an answer is to the latest observation of its number.
        self._latest: Dict[int, int] = {}

    def observed(self, number: int, game_time_ms: int, operational: bool) -> None:
        self._latest[int(number)] = len(self._seen)
        self._seen.append([int(number), int(game_time_ms), bool(operational), None])

    def answered(self, number: int, game_time_ms: int) -> None:
        """The answer to observation `number` was applied at the step whose game time is `game_time_ms`; -1 says none was."""
        if number < 0:
            return
        index = self._latest.get(int(number))
        if index is None or self._seen[index][3] is not None:
            return
        self._seen[index][3] = int(game_time_ms) - self._seen[index][1]

    def lags(self, layer: str = "tactics") -> List[int]:
        return [entry[3] for entry in self._rows(layer) if entry[3] is not None]

    def _rows(self, layer: str) -> List[List]:
        return [entry for entry in self._seen if layer == "tactics" or entry[2]]

    def summary(self, period_ms: int) -> dict:
        """Per layer: answers applied, observations whose answer never was (the episode's last observation is not counted, since no step followed it), and the mean, 95th percentile and largest lag in milliseconds and in tactical periods. An episode with no observations reports zeros."""
        last = len(self._seen) - 1
        out = {}
        for layer in LAYERS:
            rows = [(index, entry) for index, entry in enumerate(self._seen) if layer == "tactics" or entry[2]]
            lags = sorted(entry[3] for _, entry in rows if entry[3] is not None)
            missed = sum(1 for index, entry in rows if entry[3] is None and index != last)
            mean = sum(lags) / len(lags) if lags else 0.0
            p95 = float(lags[min(len(lags) - 1, max(0, math.ceil(0.95 * len(lags)) - 1))]) if lags else 0.0
            largest = float(lags[-1]) if lags else 0.0
            period = float(period_ms) if period_ms > 0 else 1.0
            out[layer] = {"answers": len(lags), "missed": missed, "mean_ms": round(mean, 2), "p95_ms": round(p95, 2),
                          "max_ms": round(largest, 2), "mean_periods": round(mean / period, 3),
                          "p95_periods": round(p95 / period, 3), "max_periods": round(largest / period, 3)}
        return out

    def steady(self, period_ms: int) -> Tuple[bool, Optional[int]]:
        """Whether every applied answer lagged exactly one tactical period, and the first lag that did not."""
        for entry in self._seen:
            if entry[3] is not None and entry[3] != period_ms:
                return False, entry[3]
        return True, None
