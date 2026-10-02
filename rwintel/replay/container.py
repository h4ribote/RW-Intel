"""A replay file as a whole: its header, the save the match started from, and the records that follow.

The header names the file as a replay and carries the game's version code, the save version and the version string. The save version is also the version the rest of the file is read at. The save that follows is the engine's ordinary save of the starting position, gzip compressed; what is taken from it here is the path of the map, which a playback needs before the game has said anything.

Everything after the save is a sequence of named blocks, each read to its own length:

- `rc`: a frame and a player command (see commands.py)
- `wait`: a frame the recording held until
- `cs`: a frame and the world checksum the recording side took there
- `es`: a frame and the extra checksums (see timing.EXTRA_CHECKSUMS)
- `chat`: a frame, a sender slot, the sender's name and the message
- `resync`: a frame and a full save the recording side resynchronised to
- `end` and `endReplayMetaData`: the recording was closed properly

A block with any other name is counted and skipped, which its length allows.
"""

from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .commands import SYSTEM, Command, read_command
from .stream import MARK, Reader, ReplayFormatError
from .timing import Clock

MAGIC = "rustedWarfareReplay"
SAVE_MAGIC = "rustedWarfareSave"
SAVE_END = "<SAVE END>"

_MAP_PATH = re.compile(rb"maps/[\x20-\x7e]+?\.tmx")


@dataclass(frozen=True)
class Header:
    #: The game's version code (`l.c(true)`), which the engine refuses to play a replay recorded under another of.
    game_version: int
    #: The save version, which is also the version every block of the file is read at.
    save_version: int
    version_name: str


@dataclass(frozen=True)
class Chat:
    frame: int
    slot: int
    sender: Optional[str]
    message: Optional[str]


@dataclass(frozen=True)
class ExtraChecksums:
    frame: int
    #: Whether the recording side traced every checksum rather than taking the ordinary long form.
    traced: bool
    values: Tuple[int, ...]


@dataclass
class Replay:
    header: Header
    #: The map as the engine names it, relative to its assets directory, or "" when the save does not say.
    map_path: str
    commands: List[Command] = field(default_factory=list)
    checksums: List[Tuple[int, int]] = field(default_factory=list)
    extra_checksums: List[ExtraChecksums] = field(default_factory=list)
    waits: List[int] = field(default_factory=list)
    chats: List[Chat] = field(default_factory=list)
    resyncs: List[int] = field(default_factory=list)
    #: Whether the recording closed with an `end` block, which a match whose game was stopped mid recording lacks.
    ended: bool = False
    #: Blocks of a name this reader does not know, by name.
    unknown: Dict[str, int] = field(default_factory=dict)
    #: Blocks by name, known ones included.
    blocks: Dict[str, int] = field(default_factory=dict)

    @property
    def clock(self) -> Clock:
        return Clock.of((c.frame, c.step_rate) for c in self.commands if c.step_rate is not None)

    @property
    def last_frame(self) -> int:
        """The latest frame any record names."""
        frames = [c.frame for c in self.commands] + [f for f, _ in self.checksums] + \
                 [e.frame for e in self.extra_checksums] + self.waits
        return max(frames, default=0)

    def players(self) -> List[int]:
        """The slots that issued commands, the engine's own excepted."""
        return sorted({c.player for c in self.commands if c.player != SYSTEM})

    def by_player(self, slot: int) -> List[Command]:
        return [c for c in self.commands if c.player == slot]


def read_header(reader: Reader) -> Header:
    magic = reader.utf()
    if magic != MAGIC:
        raise ReplayFormatError(f"not a replay: the file starts with {magic!r} rather than {MAGIC!r}")
    game_version = reader.int()
    save_version = reader.int()
    reader.version = save_version
    name = reader.utf()
    reader.boolean()
    return Header(game_version=game_version, save_version=save_version, version_name=name)


def read_save(contents: Reader) -> bytes:
    """The decompressed starting save out of the `gamesave` block."""
    magic = contents.utf()
    if magic != SAVE_MAGIC:
        raise ReplayFormatError(f"the starting save opens with {magic!r} rather than {SAVE_MAGIC!r}")
    contents.int()
    contents.int()
    contents.boolean()
    save = contents.compressed_block("saveCompression").data
    if contents.short() != MARK:
        raise ReplayFormatError("the starting save is not followed by the engine's section mark")
    if contents.utf() != SAVE_END:
        raise ReplayFormatError("the starting save does not end where the engine ends it")
    return save


def map_path_of(save: bytes) -> str:
    """The map path the starting save names, which is the first map path in it."""
    found = _MAP_PATH.search(save)
    return found.group().decode("ascii") if found else ""


def read_replay(source: Union[str, bytes, os.PathLike]) -> Replay:
    """Reads a whole replay, from a path or from its bytes."""
    data = source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
    reader = Reader(bytes(data))
    header = read_header(reader)
    save = read_save(reader.expect_block("gamesave"))
    replay = Replay(header=header, map_path=map_path_of(save))
    counts: collections.Counter = collections.Counter()
    unknown: collections.Counter = collections.Counter()
    while reader.remaining:
        tag, block = reader.block()
        counts[tag] += 1
        if tag == "rc":
            frame = block.int()
            replay.commands.append(read_command(block.expect_block("c"), frame))
        elif tag == "cs":
            frame = block.int()
            replay.checksums.append((frame, block.long()))
        elif tag == "es":
            frame = block.int()
            values = Reader(block.sized(), block.version)
            traced = values.byte() & 1 == 1
            replay.extra_checksums.append(
                ExtraChecksums(frame=frame, traced=traced, values=tuple(values.long() for _ in range(values.int()))))
        elif tag == "wait":
            replay.waits.append(block.int())
        elif tag == "chat":
            frame = block.int()
            replay.chats.append(Chat(frame=frame, slot=block.int(), sender=block.optional_utf(),
                                     message=block.optional_utf()))
        elif tag == "resync":
            replay.resyncs.append(block.int())
        elif tag in ("end", "endReplayMetaData"):
            replay.ended = replay.ended or tag == "end"
        else:
            unknown[tag] += 1
    replay.blocks = dict(counts)
    replay.unknown = dict(unknown)
    return replay
