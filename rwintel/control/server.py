"""The listening side.

One thread per connected instance. Threads rather than a single loop because each session's work is a policy call, and the point at which that becomes a batched call across instances is the point at which this grows a queue; until then a thread per instance is the shape that makes the sequence per instance obvious.

Listening does not stop once every instance has dialled in. An instance that loses the link retries, and the episode it is running carries on meanwhile, so a connection arriving later is usually one coming back rather than a new one. Which it is, is settled by the instance number in its HELLO: a session already known under that number is handed the new connection and keeps everything it had.
"""

from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..data import AssetPaths
from ..wire import Kind, read_frame
from .session import EpisodeSettings, Session

log = logging.getLogger(__name__)


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8642
    instances: int = 1
    episodes: int = 1
    episode: EpisodeSettings = field(default_factory=EpisodeSettings)
    assets: Optional[AssetPaths] = None


class Server:
    def __init__(self, settings: ServerSettings, policy_factory: Callable[[Session], object]):
        self.settings = settings
        self.policy_factory = policy_factory
        self.sessions: List[Session] = []
        self._lock = threading.Lock()
        self._done = threading.Event()

    def _finished(self) -> bool:
        """True once every instance that was asked for has been seen and has run all its episodes."""
        with self._lock:
            if len(self.sessions) < self.settings.instances:
                return False
            return all(len(session.records) >= self.settings.episodes for session in self.sessions)

    def serve(self) -> List[Session]:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Not SO_REUSEADDR: on Windows it lets a second process bind a port that is already listening, and which of the two then receives a connection is undefined. A stale control process would silently keep serving the agents while the new one looked healthy. Failing to bind is the behaviour that is wanted here.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((self.settings.host, self.settings.port))
        listener.listen(max(1, self.settings.instances))
        listener.settimeout(1.0)
        log.info("listening on %s:%d for %d instance(s)",
                 self.settings.host, self.settings.port, self.settings.instances)

        threads: List[threading.Thread] = []
        try:
            while not self._done.is_set() and not self._finished():
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    continue
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                thread = threading.Thread(target=self._run, args=(connection, address),
                                          name=f"link-{len(threads)}", daemon=True)
                thread.start()
                threads.append(thread)
        finally:
            listener.close()

        for thread in threads:
            thread.join()
        return self.sessions

    def stop(self) -> None:
        self._done.set()

    def _session_for(self, instance: int, connection, address) -> Session:
        """The session this connection belongs to, which is an existing one whenever the instance has been seen before."""
        with self._lock:
            for session in self.sessions:
                if session.instance == instance:
                    session.rebind(connection, address)
                    return session
            session = Session(connection, address, self.settings.episode, self.policy_factory,
                              self.settings.assets, self.settings.episodes)
            self.sessions.append(session)
            return session

    def _run(self, connection, address) -> None:
        session: Optional[Session] = None
        try:
            while not self._done.is_set():
                frame = read_frame(connection)
                if frame is None:
                    break
                if frame.kind == Kind.HELLO:
                    # Which session this is cannot be known before the HELLO, because the instance number is in it.
                    instance = Session.instance_in(frame.body)
                    session = self._session_for(instance, connection, address)
                    session.on_hello(frame.body)
                elif session is None:
                    log.warning("frame of kind %s from %s before its hello, ignored", frame.kind, address)
                elif frame.kind == Kind.EPISODE:
                    session.on_episode(frame.body)
                elif frame.kind == Kind.OBSERVATION:
                    session.on_observation(frame.body)
                if session is not None and len(session.records) >= self.settings.episodes:
                    break
        except (ConnectionError, OSError) as error:
            log.info("link %s ended: %s", address, error)
        except Exception:
            log.exception("link %s failed", address)
        finally:
            try:
                connection.close()
            except OSError:
                pass
