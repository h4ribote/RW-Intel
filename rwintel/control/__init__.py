"""The process the game agents connect to.

One control process, several game processes. The asymmetry is not a preference: inference is meant to be batched across instances rather than held per instance, so the side holding the policies has to be the single one. It listens and the agents dial in, because an instance comes and goes across a training run while this side lives through the whole of it.
"""

from .server import Server, ServerSettings
from .session import Session

__all__ = ["Server", "ServerSettings", "Session"]
