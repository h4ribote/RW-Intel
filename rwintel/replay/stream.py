"""Reading the engine's own serialisation, which a replay is written in from end to end.

It is Java's data stream: big-endian numbers, booleans as one byte, strings as a two byte length followed by modified UTF-8. On top of that the engine nests named blocks, each a string name, a four byte length and that many bytes, and reads a block's contents from those bytes alone. A reader that knows a block's length can therefore skip the parts of it that it has no use for and still land on the next block exactly.

What a field means can depend on the version of the stream that wrote it, and the engine tests that version before reading each field added since the first. The version is carried here so the readers above can make the same tests.
"""

from __future__ import annotations

import gzip
import struct
from typing import Optional, Tuple

#: The version a stream is read at before anything says otherwise, which is the engine's own default: every versioned field is present.
NEWEST = 999999

#: The value the engine writes after a section as a check that its reader consumed exactly what was written.
MARK = 12345


class ReplayFormatError(ValueError):
    """Bytes that do not read as the engine wrote them: truncated, misframed, or not a replay at all."""


class Reader:
    """A cursor over bytes in the engine's serialisation."""

    def __init__(self, data: bytes, version: int = NEWEST) -> None:
        self.data = data
        self.position = 0
        self.version = version

    @property
    def remaining(self) -> int:
        return len(self.data) - self.position

    def take(self, count: int) -> bytes:
        if count < 0 or self.position + count > len(self.data):
            raise ReplayFormatError(f"wanted {count} byte(s) at offset {self.position} of {len(self.data)}")
        start = self.position
        self.position += count
        return self.data[start:self.position]

    def _unpack(self, fmt: str, size: int):
        return struct.unpack(fmt, self.take(size))[0]

    def byte(self) -> int:
        return self._unpack(">b", 1)

    def boolean(self) -> bool:
        return self.take(1)[0] != 0

    def short(self) -> int:
        return self._unpack(">h", 2)

    def int(self) -> int:
        return self._unpack(">i", 4)

    def long(self) -> int:
        return self._unpack(">q", 8)

    def float(self) -> float:
        return self._unpack(">f", 4)

    def utf(self) -> str:
        length = self._unpack(">H", 2)
        return self.take(length).decode("utf-8", errors="replace")

    def optional_utf(self) -> Optional[str]:
        """A string preceded by a flag saying whether there is one."""
        return self.utf() if self.boolean() else None

    def enum(self) -> Optional[int]:
        """An enum constant as its ordinal, or None where the engine wrote -1 for none."""
        ordinal = self.int()
        return None if ordinal == -1 else ordinal

    def sized(self) -> bytes:
        """A byte array preceded by its length."""
        return self.take(self.int())

    def block(self) -> Tuple[str, "Reader"]:
        """The next named block, as its name and a reader over its contents alone, at this stream's version."""
        name = self.utf()
        return name, Reader(self.sized(), self.version)

    def expect_block(self, name: str) -> "Reader":
        found, contents = self.block()
        if found != name:
            raise ReplayFormatError(f"expected a block named {name!r} at offset {self.position}, found {found!r}")
        return contents

    def compressed_block(self, name: str) -> "Reader":
        """A block whose contents the engine wrote through gzip."""
        contents = self.expect_block(name)
        try:
            return Reader(gzip.decompress(contents.data), self.version)
        except (OSError, EOFError) as error:
            raise ReplayFormatError(f"block {name!r} does not decompress: {error}") from error
