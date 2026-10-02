"""Where an episode is written down as it finishes.

A run that is only summarised at the end can be summarised once. Everything the design says to keep -the settings and who the opponent was, the score and what it is made of, whether it was decided and when, what each layer decided and how much of it came off -is kept per episode so that a question asked later can be answered without playing the matches again. That matters here more than usual: the same settings do not reproduce the same match, so an episode that is not written down at the time cannot be recovered by running it again.

One JSON object per line, appended and flushed as each episode ends. A run that is interrupted keeps everything up to the interruption, which a single document written at the end would not.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Iterator, List, Optional

from .. import paths


class Journal:
    """Appends episodes to a file as they finish. Safe to hand to every session: they finish on their own threads."""

    def __init__(self, path: str) -> None:
        self.path = paths.ensure_parent(path)
        self._lock = threading.Lock()
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, record) -> None:
        line = json.dumps(record.as_dict(), ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def read(path: str) -> List[dict]:
    """Every episode a journal holds. Written and read as lines so that a run still in progress can be read from another process."""
    return list(stream(path))


def stream(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def default_path(name: Optional[str] = None) -> str:
    """Where a run writes when nothing was asked for: a file of its own under local/episodes, so that runs with different settings never share a journal."""
    return os.path.join(paths.episodes(), f"{name or 'run'}-{paths.stamp()}.jsonl")
