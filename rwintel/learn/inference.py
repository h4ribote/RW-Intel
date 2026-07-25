"""One evaluation for many instances at once.

Eight game processes at ten times speed ask the tactical layer for a decision about fifty times a second each, and each of those asks is a handful of floats. Answered one at a time that is four hundred separate calls onto the card a second, almost all of it launch overhead for a network small enough to fit the budget in the first place. The fixed one-period decision lag the interface already has is what makes the fix free: a period is twenty milliseconds of wall clock at ten times speed, and nothing is waiting on the answer inside it, so requests arriving within a few milliseconds of each other can be held and answered together. Eight times the batch, an eighth of the calls, and no change to what any instance observes.

The window is short on purpose. It is not a queue depth or a throughput knob: the moment it approaches a period, an instance's answer arrives after the period it was for, and the decision lag stops being one period and starts depending on how busy the machine is — which is precisely the thing the interface design refuses, because it makes the environment differ between training and operation.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, ContextManager, Dict, List, Optional, Sequence

#: How long a request waits for company, in seconds. A few milliseconds against a twenty millisecond period leaves the lag where it was.
WINDOW = 0.004

#: The most that are ever answered in one call. Beyond the number of instances there is nothing to gain, and a cap keeps a stall from turning into an unboundedly large call.
MAX_BATCH = 64


@dataclass
class _Ticket:
    request: object
    reply: object = None
    ready: threading.Event = field(default_factory=threading.Event)
    failed: Optional[BaseException] = None


class Batcher:
    """Holds requests briefly, answers them in one call, and hands each caller its own answer back.

    Callers block. That is correct here and nowhere else in this process: the game thread never waits for a decision, so what blocks is the control process's own per-instance thread, between the observation it has already read and the action it has not yet sent.
    """

    def __init__(self, evaluate: Callable[[List[object]], Sequence[object]],
                 window: float = WINDOW, max_batch: int = MAX_BATCH,
                 guard: Optional[ContextManager] = None) -> None:
        self.evaluate = evaluate
        self.window = window
        self.max_batch = max_batch
        #: Held across the forward pass where a training run is writing to the very parameters it reads. The optimiser takes the same lock around one minibatch step, so without this the reader was the only party not taking a lock that exists to be taken by two, and a batch could be answered from a network half of whose weights had been stepped and half of which had not. Nothing downstream can see that: the action and its log probability come from one pass, so the ratio the update needs is still self-consistent, and the run reports nothing unusual. It is left out of the queue handling on purpose, so the window still collects arrivals while a step is in flight. A run with no optimiser hands nothing in and holds nothing.
        self.guard: ContextManager = guard if guard is not None else contextlib.nullcontext()
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
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
        return self.submit_many([request])[0]

    def submit_many(self, requests: Sequence[object]) -> List[object]:
        """Several requests queued before any of them is waited on, and each caller's own answers handed back in the order it asked.

        This is the whole of what one caller with several requests needs, and it is the reason such a caller cannot simply loop over the single entry: that one appends a ticket and then blocks on it, so a second request is not even in the queue until the first has been answered, and a caller with four of them spends four windows rather than one. Queued together they are one window and, at the sizes anyone submits, one call. The queue is extended under one acquisition of the lock the server slices it under, so the server sees either none of a group or all of it.

        More than the batch cap in one group is answered over consecutive calls rather than refused; every ticket keeps its own identity through the queue, so the answers still come back one per request and in order. A failure is raised only once every ticket of the group has been waited on. That is not because a ticket left unwaited would strand anything — the server drains and answers its queue whether or not anyone is still listening — but so that the caller returns a failure describing the group rather than the first slice of it, and the cost of it is that a group larger than the cap whose first slice fails still waits out the windows the later slices need before it can raise. A reader shortening the failure path should know that is what they are shortenilf collected with tickets still to be answered.
        """
        tickets = [_Ticket(request=request) for request in requests]
        if not tickets:
            return []
        with self._arrived:
            if self._stop:
                raise RuntimeError("the inference server has been stopped")
            self._queue.extend(tickets)
            self._arrived.notify()
        failure: Optional[BaseException] = None
        for ticket in tickets:
            ticket.ready.wait()
            if ticket.failed is not None and failure is None:
                failure = ticket.failed
        if failure is not None:
            raise failure
        return [ticket.reply for ticket in tickets]

    def stop(self) -> None:
        with self._arrived:
            self._stop = True
            self._arrived.notify_all()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._arrived:
                while not self._queue and not self._stop:
                    self._arrived.wait(0.05)
                if self._stop and not self._queue:
                    return
            # The window is waited outside the lock so that arrivals during it are collected rather than shut out.
            time.sleep(self.window)
            with self._arrived:
                batch, self._queue = self._queue[:self.max_batch], self._queue[self.max_batch:]
            if not batch:
                continue
            self._answer(batch)

    def _answer(self, batch: List[_Ticket]) -> None:
        try:
            with self.guard:
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
