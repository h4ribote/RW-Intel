"""Where our units are lost, which is the half of the exchange the score turns on and no layer measures for itself.

Every loss is booked in credits three ways: by the role of what was lost, by whether it was in a squad or loose, and by whether it went down within reach of an armed enemy building. A side that out-earns its opponent and still falls behind is losing units somewhere, and these three cuts separate the usual answers: the wrong things built, reinforcements met alone on the way, and attacks into fixed defences.

A loss is placed where the unit was last seen, since a unit reported lost is no longer on the board, so a unit that appears and dies between two frames has no place and is not counted as lost under defences. A lost building is booked under its role only.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from ...wire import NO_SQUAD
from .contracts import Role
from .view import WorldView

#: The event kind the agent sends for a unit of ours that has been lost.
EVENT_UNIT_LOST = 2

#: How far beyond its own range an armed enemy building still counts as what a loss happened under, allowing for the unit having moved in the last period before it died.
TURRET_MARGIN = 100.0


class LossAudit:
    def __init__(self) -> None:
        #: Where our units were last seen.
        self.seen: Dict[int, Tuple[float, float]] = {}
        #: Credits lost, by the keys `role_<name>`, `squad`, `loose` and `under_defences`.
        self.lost: Dict[str, float] = {}

    def update(self, view: WorldView) -> None:
        """Books the losses this frame reports against where each unit was last seen, then remembers where our units are now."""
        defences = [e for e in view.enemies if e.role == Role.STRUCTURE and e.kind is not None and e.kind.armed]
        for event in view.observation.events:
            if event.kind != EVENT_UNIT_LOST:
                continue
            role = view.catalogue.role(event.type_index)
            self._book(f"role_{role.name.lower()}", event.value)
            if role == Role.STRUCTURE:
                continue
            self._book("loose" if event.squad == NO_SQUAD else "squad", event.value)
            last = self.seen.get(event.unit)
            if last is not None and any(math.hypot(d.unit.x - last[0], d.unit.y - last[1]) <= d.kind.range + TURRET_MARGIN
                                        for d in defences):
                self._book("under_defences", event.value)
        if view.observation.unit_states:
            self.seen = {s.unit.id: (s.unit.x, s.unit.y) for s in view.ours}

    def _book(self, key: str, value: float) -> None:
        self.lost[key] = self.lost.get(key, 0.0) + value

    def summary(self) -> Dict[str, float]:
        return {key: round(value) for key, value in sorted(self.lost.items())}
