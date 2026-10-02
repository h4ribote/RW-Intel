"""The listening side.

One thread per connected instance. Threads rather than a single loop because each session's work is a policy call, and the point at which that becomes a batched call across instances is the point at which this grows a queue; until then a thread per instance is the shape that makes the sequence per instance obvious.

Listening does not stop once every instance has dialled in. An instance that loses the link retries, and the episode it is running carries on meanwhile, so a connection arriving later is usually one coming back rather than a new one. Which it is, is settled by the instance number in its HELLO: a session already known under that number is handed the new connection and keeps everything it had.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from ..data import AssetPaths
from ..wire import Kind, read_frame
from .session import EpisodeSettings, Session

log = logging.getLogger(__name__)


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8642
    instances: int = 1
    #: Episodes each arm runs on each instance.
    episodes: int = 1
    episode: EpisodeSettings = field(default_factory=EpisodeSettings)
    assets: Optional[AssetPaths] = None
    #: The policies to run as (name, factory) pairs. More than one turns the run into a comparison, with the arms alternating within each instance.
    arms: List[Tuple[str, Callable]] = field(default_factory=list)
    #: Where every episode is written as it finishes, or None to keep nothing.
    journal: Optional[object] = None
    #: Ways of building a commander outside the chain, each called once per episode with the session. The intervention console puts one here and so does the script intruder; the order is the order they are consulted in, and the last has the final word about a squad, so a person belongs after an intruder.
    outside: List[Callable] = field(default_factory=list)
    #: Set to put every connected instance into one shared lockstep match rather than giving each its own, which is the only arrangement in which the engine has a second world to compare its own against.
    pairing: Optional[object] = None


def open_listener(host: str, port: int, backlog: int = 1) -> socket.socket:
    """The control process's listening socket, polled once a second so that a stop request is noticed.

    On Linux SO_REUSEADDR lets a restarted control process bind while the previous one's connections sit in TIME_WAIT, and still refuses a port another process is listening on, so a stale control process makes this one fail to start rather than share the agents with it.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(max(1, backlog))
    except OSError:
        listener.close()
        raise
    listener.settimeout(1.0)
    return listener


class Server:
    def __init__(self, settings: ServerSettings, policy_factory: Optional[Callable[[Session], object]] = None):
        self.settings = settings
        self.arms = list(settings.arms) or [("script", policy_factory)]
        if any(factory is None for _, factory in self.arms):
            raise ValueError("a server needs either arms or a policy factory")
        self.sessions: List[Session] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._threads: List[threading.Thread] = []

    def _finished(self) -> bool:
        """True once every instance that was asked for has been seen and has run all its episodes."""
        with self._lock:
            if len(self.sessions) < self.settings.instances:
                return False
            wanted = self.settings.episodes * len(self.arms)
            return all(len(session.records) >= wanted for session in self.sessions)

    def serve(self) -> List[Session]:
        listener = open_listener(self.settings.host, self.settings.port, self.settings.instances)
        log.info("listening on %s:%d for %d instance(s)",
                 self.settings.host, self.settings.port, self.settings.instances)

        threads = self._threads
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

    def wait(self, timeout: float) -> bool:
        """Waits up to `timeout` seconds in all for the link threads to end after a stop, and says whether they all did. A link thread notices the stop at its next frame."""
        deadline = time.monotonic() + timeout
        for thread in list(self._threads):
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def close_unfinished(self, timeout: float = 10.0) -> None:
        """After a stop, closes the policy of every episode still under way as `stopped`, once the link threads that drive them have ended. A session whose thread has not ended is left alone, since its policy may still be deciding."""
        if not self.wait(timeout):
            log.warning("a link thread did not end within %.0fs of the stop; its episode is left unclosed", timeout)
        alive = {thread.name for thread in self._threads if thread.is_alive()}
        with self._lock:
            sessions = list(self.sessions)
        for session in sessions:
            if getattr(session, "thread_name", None) not in alive:
                session.close_episode("stopped")

    def abandon(self) -> None:
        """Stops the run because an instance cannot play, and releases every link thread, since one may be blocked reading from an agent that has nothing more to send."""
        self._done.set()
        with self._lock:
            connections = [session.connection for session in self.sessions]
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    @property
    def failures(self) -> List[Tuple[int, str]]:
        """The instances that reported they could not play, with the reason each gave."""
        with self._lock:
            return [(session.instance, session.failure) for session in self.sessions if session.failure is not None]

    def session(self, instance: int) -> Optional[Session]:
        """The session of one instance, which is how anything outside the link threads -the intervention console above all -reaches the policy that is currently deciding for it."""
        with self._lock:
            return next((s for s in self.sessions if s.instance == instance), None)

    def _session_for(self, instance: int, connection, address) -> Session:
        """The session this connection belongs to, which is an existing one whenever the instance has been seen before."""
        with self._lock:
            for session in self.sessions:
                if session.instance == instance:
                    session.rebind(connection, address)
                    return session
            session = Session(connection, address, self.settings.episode, self.arms,
                              self.settings.assets, self.settings.episodes, self.settings.journal,
                              self.settings.outside)
            session.pairing = self.settings.pairing
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
                    session.thread_name = threading.current_thread().name
                    session.on_hello(frame.body)
                elif session is None:
                    log.warning("frame of kind %s from %s before its hello, ignored", frame.kind, address)
                elif frame.kind == Kind.EPISODE:
                    session.on_episode(frame.body)
                elif frame.kind == Kind.TERRAIN:
                    session.on_terrain(frame.body)
                elif frame.kind == Kind.OBSERVATION:
                    session.on_observation(frame.body, frame.flags)
                if session is not None and session.failure is not None:
                    self.abandon()
                    break
                if session is not None and len(session.records) >= session.episodes_wanted:
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
