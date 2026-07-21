"""The frame format spoken between the in-process agent and the control process.

The layout is fixed in `docs/project/05-interface.md`. Anything changed here has to be changed in `agent/Wire.java` as well, and the protocol version has to be raised so a mismatched pair refuses to talk rather than misreading each other.
"""

from .frames import (
    Frame,
    HEADER_SIZE,
    Kind,
    PROTOCOL_VERSION,
    decode_header,
    encode,
    read_frame,
)
from .observation import Observation, RegionState, SquadState, UnitState, decode_observation
from .action import Action, Contract, Production, SquadAssignment, encode_action

__all__ = [
    "Action",
    "Contract",
    "Frame",
    "HEADER_SIZE",
    "Kind",
    "Observation",
    "PROTOCOL_VERSION",
    "Production",
    "RegionState",
    "SquadAssignment",
    "SquadState",
    "UnitState",
    "decode_header",
    "decode_observation",
    "encode",
    "encode_action",
    "read_frame",
]
