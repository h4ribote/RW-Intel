"""One evaluation for many instances at once.

Every game process asks the tactical layer for a decision about every squad every period, and each of those asks is a handful of floats. Answered one at a time that is hundreds of separate calls a second across the instances, almost all of it dispatch overhead for a network small enough to fit the budget in the first place. Requests from different instances are therefore held and answered together.

A request is held until every thread that is in the middle of a decision is waiting on a batcher (see `rwintel.control.deciding`), because from then on nothing more can arrive before an answer goes out. The window is only the bound on the wait when that cannot be known, which is when a caller is outside any marked decision. It is short on purpose: games on the wall clock take an answer that arrives after its period as no answer, and games on the fixed clock stand still while they wait for it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ..control import deciding

#: Per device kind, the longest a request waits for company in seconds when it is not known whether more can come, and the most that are ever answered in one call.
#: The processor's cap is about the number of instances, beyond which there is nothing to gain, and a cap keeps a stall from turning into an unboundedly large call.
#: The graphics card's pair follows from `python -m rwintel.learn bench`: its cap is the largest batch whose call takes at most twice as long as a call for one decision, and its window stays at the processor's unless one decision alone takes longer than that, in which case the window is that single-decision latency.
#: It is the pair that rule reads on an idle card for the tactical set network, the network played at scale on the card.
LIMITS = {"cpu": (0.004, 64), "cuda": (0.004, 64)}

WINDOW, MAX_BATCH = LIMITS["cpu"]


def limits(device=None) -> tuple:
    """`(window, max_batch)` for the device a batcher evaluates on; the processor's for anything not named in `LIMITS`."""
    kind = getattr(device, "type", None) or (str(device).split(":")[0] if device is not None else "cpu")
    return LIMITS.get(kind, LIMITS["cpu"])


@dataclass
class _Ticket:
    request: object
    #: Whether the caller was counted as a deciding thread waiting here, which whoever answers has to undo.
    counted: bool = False
    reply: object = None
    ready: threading.Event = field(default_factory=threading.Event)
    failed: Optional[BaseException] = None


class Batcher:
    """Holds requests briefly, answers them in one call, and hands each caller its own answer back.

    Callers block. What blocks is the control process's own per-instance thread, between the observation it has already read and the action it has not yet sent.
    """

    def __init__(self, evaluate: Callable[[List[object]], Sequence[object]],
                 window: float = WINDOW, max_batch: int = MAX_BATCH) -> None:
        self.evaluate = evaluate
        self.window = window
        self.max_batch = max_batch
        self._queue: List[_Ticket] = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()
        #: How the batching is actually going, which is the only way to tell a window that is helping from one that is merely adding latency.
        self.calls = 0
        self.served = 0

    @property
    def batch_size(self) -> float:
        return self.served / self.calls if self.calls else 0.0

    def submit(self, request: object) -> object:
        ticket = _Ticket(request=request, counted=deciding.inside())
        with deciding.condition:
            if self._stop:
                raise RuntimeError("the inference server has been stopped")
            self._queue.append(ticket)
            if ticket.counted:
                deciding.start_waiting()
            deciding.condition.notify_all()
        ticket.ready.wait()
        if ticket.failed is not None:
            raise ticket.failed
        return ticket.reply

    def stop(self) -> None:
        with deciding.condition:
            self._stop = True
            deciding.condition.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with deciding.condition:
                while not self._queue and not self._stop:
                    deciding.condition.wait(0.05)
                if self._stop and not self._queue:
                    return
                deadline = time.monotonic() + self.window
                while (not self._stop and len(self._queue) < self.max_batch and not deciding.everyone_waiting()
                       and time.monotonic() < deadline):
                    deciding.condition.wait(max(0.0, deadline - time.monotonic()))
                batch, self._queue = self._queue[:self.max_batch], self._queue[self.max_batch:]
                # Released from the count before they are answered, so that no other batcher reads them as still waiting once they are busy again.
                for ticket in batch:
                    if ticket.counted:
                        deciding.stop_waiting()
            if batch:
                self._answer(batch)

    def _answer(self, batch: List[_Ticket]) -> None:
        try:
            replies = self.evaluate([ticket.request for ticket in batch])
        except BaseException as error:  # a failure has to reach the callers, or every one of them waits for ever
            for ticket in batch:
                ticket.failed = error
                ticket.ready.set()
            return
        self.calls += 1
        self.served += len(batch)
        for ticket, reply in zip(batch, replies):
            ticket.reply = reply
            ticket.ready.set()
