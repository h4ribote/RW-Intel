"""Actors that follow a learner: the weights a game-playing run serves, replaced whenever the learner publishes newer ones.

A learner in follow mode (`offline --follow`) writes its model file and then a sidecar `<model file>.json` naming the version it holds. A `Reloader` reads the sidecar every few seconds; when the version has risen it loads the file away from the inference lock, checks that the file holds the version the sidecar names (a file still being replaced is tried again on the next tick), and copies the parameters into the served network under the lock its batcher evaluates inside. A file of another layer, kind or shape is refused with one error line and the served weights stay as they were.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from torch import nn

from . import models

log = logging.getLogger(__name__)


def sidecar_path(path: str) -> str:
    """Where a published model file's sidecar is written."""
    return path + ".json"


def read_sidecar(path: str) -> Optional[dict]:
    """The sidecar beside a model file, or None while there is none or it cannot be read whole."""
    try:
        with open(sidecar_path(path), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


class Reloader:
    """Keeps a served network at the latest version a learner has published to `path`; `version` is the file version being served."""

    def __init__(self, net: nn.Module, path: str, lock: threading.Lock, seconds: float) -> None:
        self.net = net
        self.path = path
        self.lock = lock
        self.seconds = seconds
        self.version = int(getattr(net, "version", 0))
        self.reloads = 0
        #: Set once a published file did not fit the served network; nothing more is loaded after it.
        self.refused = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reloader", daemon=True)

    def start(self) -> "Reloader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.seconds):
            try:
                self.poll()
            except Exception as error:  # a failed tick leaves the served weights as they were and is tried again
                log.warning("reloading %s failed: %s", self.path, error)

    def poll(self) -> bool:
        """One look at the sidecar; True when newer weights were swapped in."""
        if self.refused:
            return False
        sidecar = read_sidecar(self.path)
        if sidecar is None or int(sidecar.get("version", 0)) <= self.version:
            return False
        wanted = int(sidecar["version"])
        try:
            written = models.read(self.path)
        except Exception:
            return False
        if int(written.get("version", 0)) != wanted:
            return False
        config = dict(getattr(self.net, "config", {}))
        try:
            written_config = models.file_config(written)
        except ValueError:
            written_config = written["config"]
        shape = {name: written_config.get(name) for name in config}
        if written["layer"] != models.layer_of(self.net) or written["kind"] != models.kind_of(self.net) or shape != config:
            log.error("%s now holds a %s %s network of config %s, which does not fit the served %s %s network of config %s; "
                      "it is not reloaded", self.path, written["kind"], written["layer"], shape,
                      models.kind_of(self.net), models.layer_of(self.net), config)
            self.refused = True
            return False
        fresh = models.build(written["layer"], written["kind"], written_config)
        fresh.load_state_dict(written["state"])
        state = fresh.state_dict()
        with self.lock:
            self.net.load_state_dict(state)
            self.version = wanted
        self.reloads += 1
        log.info("serving version %d of %s", wanted, self.path)
        return True
