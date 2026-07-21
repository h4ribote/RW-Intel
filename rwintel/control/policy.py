"""A first script policy: enough of every layer to close the loop end to end.

This is not the script policy the design calls for. That one has five layers with contracts between them, squads with a lifetime, doctrines, and postures; this has one squad, a fixed opening, and a single rule for departing from the contract. What it does have is a decision at every point the interface has a field for, which is what makes it useful now: it exercises economy, organisation, operations and tactics against a real match, so the format is tested by something that plays rather than by something that echoes.

Where it makes a choice the design has already fixed, it makes that choice: value is the sum of prices, a squad is addressed as a whole, places are named by region.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from ..wire import Action, Contract, Observation, Production, SquadAssignment, UnitState, encode_action
from ..wire.action import Deviation, ProductionKind, Stance, Task

log = logging.getLogger(__name__)

#: A building standing on a resource point is within this of it, and nothing else is.
ON_RESOURCE = 40.0

#: How long a placement is assumed to be on its way before the point is offered again.
PLACEMENT_GRACE_MS = 15000

#: The one squad this policy forms.
MAIN_SQUAD = 0

#: Below this the squad stays home; the opening is not worth throwing four tanks away on.
ATTACK_STRENGTH = 6


class ScriptPolicy:
    def __init__(self, session):
        self.session = session
        self.extractor = session.type_by_lookup("extractor")
        self.factory = session.type_by_lookup("landFactory")
        self.tank = session.type_by_lookup("tank")
        self.builder = session.type_by_lookup("builder")

        self.resource_points: List[Tuple[float, float]] = []
        if session.map_content is not None:
            self.resource_points = [session.map_content.to_world(t) for t in session.map_content.resources]

        self.placed_at: Dict[int, int] = {}
        self.last_economy_ms = -10 ** 9
        self.squad_members: List[int] = []

    # ---- entry point -----------------------------------------------------------------

    def decide(self, observation: Observation) -> Optional[bytes]:
        types = self.session.types
        ours = [u for u in observation.unit_states if not u.hostile]
        enemies = [u for u in observation.unit_states if u.hostile]

        buildings, builders, fighters = [], [], []
        for unit in ours:
            kind = types[unit.type_index] if 0 <= unit.type_index < len(types) else None
            if kind is None:
                continue
            if kind.building:
                buildings.append((unit, kind))
            elif kind.builder:
                builders.append((unit, kind))
            else:
                fighters.append((unit, kind))

        action = Action()
        self._organise(action, fighters)
        self._operate(action, observation, fighters, enemies)
        if observation.blocks & 1:  # the region block marks an operational period, which is when the economy decides
            self._build(action, observation, buildings, builders)
            log.debug("t=%5ds credits=%6.0f buildings=%d builders=%d fighters=%d enemies=%d production=%s",
                      observation.game_time_ms // 1000, observation.credits, len(buildings),
                      len(builders), len(fighters), len(enemies),
                      [(p.kind.name, self.session.types[p.type_index].lookup) for p in action.production])

        if not action.squads and not action.contracts and not action.production:
            return None
        return encode_action(action)

    # ---- organisation ----------------------------------------------------------------

    def _organise(self, action: Action, fighters) -> None:
        """One squad holding everything that can fight. The real organisation layer forms several by doctrine."""
        members = sorted(unit.id for unit, _ in fighters)
        if members != self.squad_members:
            self.squad_members = members
            action.squads.append(SquadAssignment(squad=MAIN_SQUAD, units=members))

    # ---- operations ------------------------------------------------------------------

    def _operate(self, action: Action, observation: Observation, fighters, enemies) -> None:
        if not self.session.regions:
            return
        squad = next((s for s in observation.squads if s.id == MAIN_SQUAD), None)
        strong_enough = len(fighters) >= ATTACK_STRENGTH

        target = self._enemy_home() if strong_enough else self.session.home
        if target is None:
            return

        # The whole point of a budget is that the tactical layer never has to know why it is what it is. Half the squad's worth is what this opening is willing to pay for a push.
        budget = int(sum(self.session.types[u.type_index].price for u, _ in fighters) * 0.5)

        deviation = Deviation.HOLD
        if squad is not None and squad.status == 2:
            deviation = Deviation.WITHDRAW
        elif squad is not None and enemies and strong_enough:
            deviation = Deviation.FOCUS if self._enemies_near(squad, enemies) else Deviation.HOLD

        action.contracts.append(Contract(
            squad=MAIN_SQUAD,
            task=Task.ATTACK if strong_enough else Task.DEFEND,
            stance=Stance.AGGRESSIVE,
            target_region=target.id,
            deviation=deviation,
            cost_budget=budget,
            deadline_ms=observation.game_time_ms + 120000,
        ))

    @staticmethod
    def _enemies_near(squad, enemies: List[UnitState]) -> bool:
        return any(math.hypot(e.x - squad.x, e.y - squad.y) < 400 for e in enemies)

    def _enemy_home(self):
        """The spawn region furthest from ours, which on a symmetric map is where the opponent started."""
        home = self.session.home
        spawns = [r for r in self.session.regions if r.spawn]
        if not spawns:
            return None
        if home is None:
            return spawns[0]
        return max(spawns, key=lambda r: home.distance_to(r))

    # ---- economy ---------------------------------------------------------------------

    def _build(self, action: Action, observation: Observation, buildings, builders) -> None:
        now = observation.game_time_ms
        self.last_economy_ms = now

        occupied = [(u.x, u.y) for u, _ in buildings]
        free = []
        for index, point in enumerate(self.resource_points):
            if now - self.placed_at.get(index, -10 ** 9) < PLACEMENT_GRACE_MS:
                continue
            if any(math.hypot(bx - point[0], by - point[1]) < ON_RESOURCE for bx, by in occupied):
                continue
            free.append((index, point))

        idle_builders = [u for u, _ in builders if u.orders == 0]
        have_factory = any(kind.lookup == "landFactory" for _, kind in buildings)

        if idle_builders and self.factory is not None and not have_factory:
            extractors = sum(1 for _, kind in buildings if kind.lookup == "extractor")
            if extractors >= 2 and observation.credits >= self.factory.price:
                unit = idle_builders.pop(0)
                action.production.append(Production(
                    producer=unit.id, type_index=self.factory.index,
                    kind=ProductionKind.BUILDING, x=unit.x + 120, y=unit.y + 120))

        if idle_builders and self.extractor is not None and free and observation.credits >= self.extractor.price:
            unit = idle_builders.pop(0)
            index, point = min(free, key=lambda item: math.hypot(item[1][0] - unit.x, item[1][1] - unit.y))
            self.placed_at[index] = now
            action.production.append(Production(
                producer=unit.id, type_index=self.extractor.index,
                kind=ProductionKind.BUILDING, x=point[0], y=point[1]))

        if self.tank is not None and observation.credits >= self.tank.price:
            for unit, kind in buildings:
                if kind.lookup != "landFactory" or unit.built < 255:
                    continue
                action.production.append(Production(
                    producer=unit.id, type_index=self.tank.index, kind=ProductionKind.UNIT))
                break


def script_policy(session) -> ScriptPolicy:
    return ScriptPolicy(session)
