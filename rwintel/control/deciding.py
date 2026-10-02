"""Which control threads are in the middle of a decision, and how many of those are waiting on a batched evaluation.

A session marks the span of each decision with `deciding()`. A batcher counts the callers it holds that are inside such a span, and once every thread that is deciding is waiting on one batcher or another, no further request can arrive until one of them is answered: that is the moment to evaluate, rather than the end of a fixed window. Games on the fixed clock wait for every answer, so a window spent waiting for company that cannot come is time the simulations stand still.

One condition serves every batcher in the process, because the question is about all of them at once.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

#: Guards the counts below and is what batchers wait on.
condition = threading.Condition()

_deciding = 0
_waiting = 0
_local = threading.local()


@contextmanager
def deciding() -> Iterator[None]:
    """Marks the calling thread as inside a decision for the duration."""
    global _deciding
    with condition:
        _deciding += 1
        condition.notify_all()
    _local.inside = True
    try:
        yield
    finally:
        _local.inside = False
        with condition:
            _deciding -= 1
            condition.notify_all()


def inside() -> bool:
    """Whether the calling thread is inside a decision."""
    return getattr(_local, "inside", False)


def start_waiting() -> None:
    """Counts one more deciding thread as waiting on a batcher. Called with `condition` held."""
    global _waiting
    _waiting += 1


def stop_waiting() -> None:
    """Counts one fewer. Called with `condition` held, by whoever answers, before the waiting thread is released."""
    global _waiting
    _waiting -= 1


def everyone_waiting() -> bool:
    """Whether some thread is deciding and every such thread is waiting on a batcher. Called with `condition` held."""
    return _deciding > 0 and _waiting >= _deciding
