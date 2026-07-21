"""Framing: a fixed header followed by a body.

Everything is little endian. The header carries the protocol version so that a mismatched pair refuses the connection instead of misreading each other's bytes; the field layout is derived from a disassembly of an obfuscated game and will not survive a game update untouched.
"""

from __future__ import annotations

import enum
import socket
import struct
from dataclasses import dataclass
from typing import Optional

#: ASCII "RWIN". Present so a stream that has lost sync fails loudly at the next header.
MAGIC = 0x4E495752

PROTOCOL_VERSION = 1

_HEADER = struct.Struct("<IHHHHI")
HEADER_SIZE = _HEADER.size


class Kind(enum.IntEnum):
    """Frame kinds. Low numbers travel from the game, high numbers travel to it."""

    HELLO = 0x01
    EPISODE = 0x02
    OBSERVATION = 0x10
    ACTION = 0x20
    CONTROL = 0x30


@dataclass
class Frame:
    kind: Kind
    instance: int
    flags: int
    body: bytes


def encode(kind: Kind, instance: int, body: bytes, flags: int = 0) -> bytes:
    return _HEADER.pack(MAGIC, PROTOCOL_VERSION, int(kind), instance, flags, len(body)) + body


def decode_header(header: bytes):
    """Returns (kind, instance, flags, body_length). Raises on a bad magic or a version mismatch."""
    magic, version, kind, instance, flags, length = _HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError(f"not a frame header: magic {magic:#x}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
    return Kind(kind), instance, flags, length


def _recv_exactly(connection: socket.socket, count: int) -> Optional[bytes]:
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(connection: socket.socket) -> Optional[Frame]:
    """Reads one whole frame, or returns None when the peer has closed the connection."""
    header = _recv_exactly(connection, HEADER_SIZE)
    if header is None:
        return None
    kind, instance, flags, length = decode_header(header)
    body = _recv_exactly(connection, length) if length else b""
    if body is None:
        return None
    return Frame(kind=kind, instance=instance, flags=flags, body=body)
