"""Whether a port is free for a hosting game instance to bind.

A host that cannot bind its match port says so only in the game's own log and carries on, and the pair then waits for a match that never begins. Checking before either process is started says it where it can still be fixed.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from typing import List, Optional


def holder(port: int) -> Optional[str]:
    """What is listening on the port, as `ss` reports it, or None when that cannot be found out."""
    ss = shutil.which("ss")
    if ss is None:
        return None
    try:
        shown = subprocess.run([ss, "-H", "-l", "-n", "-p", "-t", "-u", f"sport = :{port}"],
                               capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = " ".join(shown.stdout.split())
    return text or None


def check_free(port: int) -> Optional[str]:
    """None when both TCP and UDP can bind the port on every address, otherwise a sentence saying which cannot."""
    for kind, name in ((socket.SOCK_STREAM, "TCP"), (socket.SOCK_DGRAM, "UDP")):
        probe = socket.socket(socket.AF_INET, kind)
        try:
            if kind == socket.SOCK_STREAM:
                # The JVM binds its server sockets with SO_REUSEADDR on Linux, so connections of an earlier match still in TIME_WAIT do not stop the host and must not fail this check either.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("0.0.0.0", port))
        except OSError as error:
            owner = holder(port)
            return f"{name} port {port} is in use ({error.strerror})" + (f": {owner}" if owner else "")
        finally:
            probe.close()
    return None


def local_addresses() -> List[str]:
    """This machine's non-loopback IPv4 addresses: the one outgoing traffic leaves from, then any the host name resolves to. Empty when there are none, which leaves the loopback as the only way in."""
    found: List[str] = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting a datagram socket only chooses a route; nothing is sent.
        probe.connect(("192.0.2.1", 9))
        found.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    try:
        found.extend(entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return list(dict.fromkeys(address for address in found if not address.startswith("127.") and address != "0.0.0.0"))


def join_addresses(port: int, addresses: Optional[List[str]] = None) -> List[str]:
    """Where a game client joins a match hosted on this machine: the loopback first, which is the way in from the same machine and through a forwarded localhost, then this machine's own addresses."""
    found = local_addresses() if addresses is None else addresses
    return [f"localhost:{port}", *(f"{address}:{port}" for address in found)]


def check_listener_free(host: str, port: int) -> Optional[str]:
    """None when a control process could bind its listening port, otherwise a sentence saying why not.

    Bound the way the control process binds it, with SO_REUSEADDR, so that only a live listener fails the check and connections of an earlier run still closing do not.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as error:
        owner = holder(port)
        return f"TCP port {port} on {host} is in use ({error.strerror})" + (f": {owner}" if owner else "")
    finally:
        probe.close()
    return None
