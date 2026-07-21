"""Two game processes in one lockstep match, and the question that can only be asked once they are.

Every episode this project has run so far has been one process playing by itself against the built-in AI. That is not a lockstep session: with nobody to keep in step with, the engine never compares its world against another and there is nothing that could be observed to disagree. So one assumption has stood unchecked under everything built on top of it — that units created through the engine's spawn command travel the ordinary command route and therefore stay in step — and it is the assumption the whole tactical training environment rests on, because that environment does nothing but create units.

Checking it needs two processes in one match, and the engine has everything for that: a real host that binds a port, a connector that joins one, and a checksum of the world exchanged every few hundred frames whose verdict the host keeps per client. So the check is not a piece of apparatus to be invented but a run to be made: host, join, play, spawn, and read back what the engine says about whether the two worlds still agree.

Which process does which is decided here rather than at the launcher, because it depends on which instance connects and the launcher does not know. The joining process is held until the host has actually reported its match started: the engine's connector gives up after a few seconds and a join attempted against a port nobody is listening on yet fails outright, so the order matters and is worth arranging rather than retrying into.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

#: How long a joining process waits for the host to report its match started before trying anyway. Generous, because what it is waiting through is a map load.
HOST_WAIT_SECONDS = 120.0


@dataclass
class Pairing:
    """Shared between the sessions of one paired match: who hosts, on what port, and whether the host is up yet."""

    #: The instance number that hosts. Everyone else joins it.
    host_instance: int = 0
    address: str = "127.0.0.1"
    port: int = 5123
    #: Set once the hosting instance has reported that its match has begun, which is the moment a join can succeed.
    up: threading.Event = field(default_factory=threading.Event)

    def hosts(self, session) -> bool:
        return session.instance == self.host_instance

    def instruction(self, session) -> Dict[str, object]:
        """The fields the start instruction carries for this instance's part in the match."""
        if self.hosts(session):
            # Released as the instruction goes out rather than when the match begins. A match cannot be joined once it has started, so the other process has to be dialling while the host is still loading its map and standing in the room; the joining side retries until the port answers, and the host waits in the room until somebody is there.
            self.up.set()
            return {"host": True, "port": self.port, "name": f"host-{session.instance}"}
        if not self.up.wait(HOST_WAIT_SECONDS):
            log.warning("instance %d is joining %s:%d without having heard the host open a room",
                        session.instance, self.address, self.port)
        return {"join": f"{self.address}:{self.port}", "name": f"client-{session.instance}"}

    def started(self, session) -> None:
        """Nothing to do. The order that matters is settled before the match begins, not by its beginning."""


@dataclass
class SpawnProbe:
    """Creates units during a live paired match, which is the thing whose safety is in question.

    It runs on the host because that is where the per client verdict is recorded, and it spawns something small and ordinary rather than anything exotic: what is being asked is whether the command route keeps the two worlds in step, not whether some particular unit does.
    """

    #: How many times to spawn over the episode.
    times: int = 6
    #: Game milliseconds between spawns.
    every_ms: int = 20000
    #: Units created each time, for the player the observation says this process is.
    count: int = 3
    _done: int = 0
    _next_ms: int = 0

    def consider(self, session, observation) -> bool:
        """Spawns if it is time to. Returns whether anything was sent, so that a caller can say so."""
        if self._done >= self.times or observation.game_time_ms < self._next_ms:
            return False
        kind = _something_ordinary(session)
        if kind is None:
            return False
        home = _somewhere_of_ours(observation)
        if home is None:
            return False
        rows: List[float] = []
        for index in range(self.count):
            rows.extend([float(kind.index), float(observation.slot),
                         home[0] + 60.0 * index, home[1] + 60.0, 1.0])
        session.scenario(rows)
        self._done += 1
        self._next_ms = observation.game_time_ms + self.every_ms
        log.info("instance %d spawned %d %s at %.0f,%.0f (%d of %d)",
                 session.instance, self.count, kind.lookup, home[0], home[1], self._done, self.times)
        return True


def _something_ordinary(session):
    """The cheapest thing a factory can actually turn out, which is what an ordinary player's first command would produce anyway.

    Restricted to what a factory builds rather than to whatever the registry prices lowest, because the registry contains the creatures a nest spawns and the deployed forms of other units, and the cheapest thing in it is one of those. What is being asked is whether creating an ordinary unit disturbs a shared match, so the unit had better be an ordinary one.
    """
    from .policy.catalogue import Catalogue

    catalogue = Catalogue(session.types, session.assets)
    candidates = [kind for kind in session.types
                  if kind.mobile and kind.armed and kind.price > 0
                  and kind.movement in ("LAND", "HOVER") and catalogue.builds("landFactory", kind)]
    return min(candidates, key=lambda kind: kind.price) if candidates else None


def _somewhere_of_ours(observation):
    ours = [unit for unit in observation.unit_states if not unit.hostile]
    if not ours:
        return None
    return (sum(u.x for u in ours) / len(ours), sum(u.y for u in ours) / len(ours))


class Probe:
    """The spawn probe as a commander outside the chain.

    It creates units and issues no orders, so it is not a policy and does not belong inside the chain: what is under test is whether creating units disturbs a match that is otherwise ordinary, and the chain playing that ordinary match is exactly what must not be altered to ask the question. The seam for anything that commands from outside already exists, and this goes through it like a person or an intruder does.
    """

    def __init__(self, session, times: int, every_ms: int = 20000, count: int = 3) -> None:
        self.session = session
        self.probe = SpawnProbe(times=times, every_ms=every_ms, count=count)
        #: Set by the session, as it is on every outside commander. Nothing here wants a squad, so nothing here reads it.
        self.organisation = None

    def intervene(self, action, view, squads, observation):
        self.probe.consider(self.session, observation)
        return []


def probing(times: int, pairing: Optional[Pairing] = None):
    """A way of building the probe per session, which creates units only on the process that hosts: the host is the side the engine records each client's verdict on, and there is no reason for both sides to spawn to answer one question."""

    def build(session) -> Probe:
        wanted = times if pairing is None or pairing.hosts(session) else 0
        return Probe(session, wanted)

    return build
