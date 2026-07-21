"""A board on which fights are built one after another, so the tactical layer can be trained without playing matches.

The tactical layer's sample budget is not the match budget. An errand lasts ten to sixty seconds and a match lasts fifteen minutes, so training the tactical layer inside matches would spend the overwhelming majority of the wall clock simulating economies, build orders and marches that its decision has no bearing on. Everything needed to avoid that is already there: the engine's own spawn command creates units through the ordinary command route, the room settings can start a match with nothing on the board, and the sandbox flag makes every player's units answerable to this one process. Put together, that is an arena — one long episode in which engagements are constructed, fought, swept away and constructed again.

Both sides are driven from here, which is the point of the sandbox flag and is what makes the opponent something other than the built-in AI. The opposing side runs the same tactical layer over the same board read from the other side, so that what a learnt layer is measured against is the script layer doing exactly its job, and so that self-play needs no second process and no network.

There is no instruction that removes a unit, because the game has none: the only way to unmake a unit is to kill it. So an engagement is not cleared, it is finished — the survivors of both sides are set on each other until there are none — and the next engagement is built somewhere else on the map. That is slower than deleting them would be and it is the only method that keeps every change on the command route, which is the property the whole approach depends on.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..wire import (
    Action,
    Contract,
    Observation,
    Stance,
    Status,
    SquadAssignment,
    Task,
    encode_action,
)
from ..control.policy.catalogue import Catalogue
from ..control.policy.contracts import Doctrine, SquadRecord, TaskContract
from ..control.policy.tactics import Tactics
from ..control.policy.view import build as build_view

log = logging.getLogger(__name__)

#: Squad slots the two sides of an engagement occupy. Fixed, because nothing else is competing for the eight: the arena is the whole policy for this episode.
OURS = 0
THEIRS = 1

#: How long a fight is allowed before it is called and swept, in game milliseconds. The design's stated tactical horizon is ten to sixty seconds; this is the top of it, and most fights end well inside it.
FIGHT_MS = 60000

#: How long the survivors are given to finish each other off before the next engagement is built anyway. A remainder is tolerable — the next site is chosen away from it — but an unbounded wait is not.
SWEEP_MS = 20000

#: How long a fight may go without a casualty on either side before it is called.
#:
#: Two forces that have stopped hurting each other are not about to start. Measured: an engagement that ends with somebody destroyed takes twenty to thirty seconds, and one that ends on the clock spends its whole minute with both sides nearly intact, so waiting the full minute for those buys nothing and costs the arena a third of its time. Long enough that a squad manoeuvring for position is not mistaken for one that has given up.
STALL_MS = 12000

#: How long to wait for spawned units to appear before giving up on an engagement and trying again. Spawning goes through the command queue, so it takes a step or two rather than being instantaneous.
SPAWN_WAIT_MS = 12000

#: World units between the two sides when they are put down.
#:
#: Measured rather than chosen. At seven hundred the two sides converged to a median of a hundred and sixty and stopped there, which is outside a tank's reach of a hundred and thirty, and three quarters of the engagements then stood off until the clock ran out with almost nobody hurt: the engine's attack-move halts a unit when it acquires something, and acquisition happens at sight range while shooting needs weapon range, so two forces walking at each other come to rest in the gap between the two and stay there. Starting inside that gap is what makes the fight begin. It is still outside the reach of the shorter-ranged types, so which side shoots first is still something the layer's choice can affect.
SEPARATION = 250.0

#: Credits each side is built out of, drawn uniformly. Small fights and large ones teach different things and the layer has to answer both.
FORCE_VALUE = (1200.0, 5000.0)

#: How lopsided a fight may be, as the weaker side's share of the stronger. A layer that only ever saw even fights would never learn that some fights are to be broken off, which is one of the five departures.
IMBALANCE = (0.5, 1.0)

#: The most units either side is built from, so that one engagement cannot fill the board.
MAX_UNITS = 14

#: The fewest units a side is built from, which is what bounds how expensive a type may be for the budget it is drawn against. A fight is between formations, and one machine against a formation is a different problem from the one the five departures are about.
MINIMUM_FORCE = 3


@dataclass
class Engagement:
    """One fight, from the moment it is spawned to the moment the ground is clear."""

    index: int
    site: Tuple[float, float]
    our_value: float = 0.0
    their_value: float = 0.0
    ours_left: int = 0
    theirs_left: int = 0
    seconds: float = 0.0
    #: True when the fight was called because neither side had hurt the other for a while, rather than because it ended or ran out of time.
    stalled: bool = False
    #: How close the two sides ever came to each other, in world units. A fight in which this never falls below the weapons' reach was not a fight, and telling that case from a fight that was genuinely even is the difference between an arena that produces engagements and one that produces marches.
    closest: float = 1e9

    def as_dict(self) -> dict:
        return {"index": self.index, "our_value": round(self.our_value), "their_value": round(self.their_value),
                "ours_left": self.ours_left, "theirs_left": self.theirs_left,
                "seconds": round(self.seconds, 1), "closest": round(self.closest),
                "stalled": self.stalled}


@dataclass
class Statistics:
    """What an arena episode did, in the shape the episode record expects so that a run of engagements is journalled like any other."""

    engagements: int = 0
    spawned: int = 0
    #: Engagements that were built but where the units never appeared, so no fight took place. Counted separately because it is wasted time rather than a result, and because it is the first thing to look at when the arena is producing less than it should.
    stillborn: int = 0
    won: int = 0
    lost: int = 0
    drawn: int = 0
    tactical: int = 0
    decisions: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def fought(self) -> int:
        return self.won + self.lost + self.drawn

    def as_dict(self) -> dict:
        return {"engagements": self.engagements, "spawned": self.spawned,
                "stillborn": self.stillborn, "fought": self.fought, "won": self.won,
                "lost": self.lost, "drawn": self.drawn, "tactical": self.tactical,
                "decisions": self.decisions, "history": self.history[-32:]}


class Arena:
    """The policy an arena episode runs under. Builds engagements, fights both sides of them, and pays the layer being trained."""

    def __init__(self, session, tactics: Optional[Callable] = None,
                 opponent: Optional[Callable] = None, seed: int = 0,
                 enemy_slot: Optional[int] = None) -> None:
        self.session = session
        self.catalogue = Catalogue(session.types, session.assets)
        self.random = random.Random(seed)
        # The layers are built here rather than handed in already made, because both sides have to read the same type catalogue as the arena that spawns their units: a layer classifying a unit from a different table would sort the same tank into a different role.
        self.tactics = tactics(session, self.catalogue) if tactics else Tactics(session, self.catalogue)
        self.opponent = opponent(session, self.catalogue) if opponent else Tactics(session, self.catalogue)
        self.statistics = Statistics()
        self.enemy_slot = enemy_slot

        self.squads: Dict[int, SquadRecord] = {}
        self.phase = "opening"
        self.engagement: Optional[Engagement] = None
        self.until_ms = 0
        self.known: set = set()
        self.sites: List[Tuple[float, float]] = []
        self.last_regions: List = []
        self._sandbox_sent = False
        #: How many units were standing in the current fight when its count last changed, and when that was, which is how a fight that has stopped being one is recognised.
        self._alive = 0
        self._changed_ms = 0

    # ---- the one entry point ----------------------------------------------------------

    def decide(self, observation: Observation) -> Optional[bytes]:
        if not self._sandbox_sent:
            # Sandbox first and on its own, because until it is set the units of the other side are not this process's to command and every order to them would be dropped.
            self.session.scenario([], sandbox=True)
            self._sandbox_sent = True
            self.sites = self._sites()
            self.known = {unit.id for unit in observation.unit_states}

        view = build_view(observation, self.catalogue, None, self.last_regions)
        self.last_regions = view.regions
        action = Action()
        now = observation.game_time_ms

        self._fold(observation)
        if self.phase == "opening" or self.phase == "clear":
            self._begin(observation, action, now)
        elif self.phase == "spawning":
            self._form(observation, action, now)
        elif self.phase == "fighting":
            self._fight(observation, view, action, now)
        elif self.phase == "sweeping":
            self._sweep(observation, action, now)

        if not (action.squads or action.contracts or action.deviations or action.production):
            return None
        return encode_action(action)

    # ---- building a fight --------------------------------------------------------------

    def _begin(self, observation: Observation, action: Action, now: int) -> None:
        site = self._site(observation)
        if site is None:
            return
        budget = self.random.uniform(*FORCE_VALUE)
        weaker = self.random.uniform(*IMBALANCE)
        ours_first = self.random.random() < 0.5
        our_budget = budget if ours_first else budget * weaker
        their_budget = budget * weaker if ours_first else budget

        angle = self.random.uniform(0, 2 * math.pi)
        offset = (math.cos(angle) * SEPARATION / 2, math.sin(angle) * SEPARATION / 2)
        our_place = (site[0] - offset[0], site[1] - offset[1])
        their_place = (site[0] + offset[0], site[1] + offset[1])

        our_force = self._force(our_budget)
        their_force = self._force(their_budget)
        if not our_force or not their_force:
            return

        spawns: List[float] = []
        spawns.extend(self._rows(our_force, self._our_slot(observation), our_place))
        spawns.extend(self._rows(their_force, self._their_slot(observation), their_place))
        self.session.scenario(spawns)

        self.known = {unit.id for unit in observation.unit_states}
        self.engagement = Engagement(index=self.statistics.engagements, site=site,
                                     our_value=sum(kind.price for kind in our_force),
                                     their_value=sum(kind.price for kind in their_force))
        self.statistics.engagements += 1
        self.statistics.spawned += len(our_force) + len(their_force)
        self.phase = "spawning"
        self.until_ms = now + SPAWN_WAIT_MS
        log.debug("engagement %d at %.0f,%.0f: %.0f against %.0f credits",
                  self.engagement.index, site[0], site[1],
                  self.engagement.our_value, self.engagement.their_value)

    def _form(self, observation: Observation, action: Action, now: int) -> None:
        """Takes the units that have appeared since the spawn was ordered and makes two squads of them."""
        fresh = [unit for unit in observation.unit_states if unit.id not in self.known]
        ours = [unit.id for unit in fresh if not unit.hostile]
        theirs = [unit.id for unit in fresh if unit.hostile]
        if not ours or not theirs:
            if now >= self.until_ms:
                log.info("engagement %d at %.0f,%.0f never appeared, trying another",
                         self.engagement.index if self.engagement else -1,
                         self.engagement.site[0] if self.engagement else 0.0,
                         self.engagement.site[1] if self.engagement else 0.0)
                self.statistics.stillborn += 1
                self._blame_site()
                self.phase = "clear"
            return

        self.squads = {
            OURS: self._record(OURS, ours, observation),
            THEIRS: self._record(THEIRS, theirs, observation),
        }
        action.squads.append(SquadAssignment(squad=OURS, units=ours))
        action.squads.append(SquadAssignment(squad=THEIRS, units=theirs,
                                             owner=self._their_slot(observation)))

        deadline = now + FIGHT_MS
        for squad_id, other in ((OURS, THEIRS), (THEIRS, OURS)):
            squad = self.squads[squad_id]
            target = self._region_of(self.squads[other])
            contract = TaskContract(squad=squad_id, task=Task.ATTACK, target_region=target,
                                    stance=Stance.AGGRESSIVE, cost_budget=max(200.0, squad.value),
                                    deadline_ms=deadline, issued_at_ms=now)
            squad.contract = contract
            action.contracts.append(Contract(
                squad=squad_id, task=contract.task, stance=contract.stance,
                target_region=contract.target_region, cost_budget=contract.cost_budget,
                deadline_ms=contract.deadline_ms, issued_at_ms=contract.issued_at_ms,
                override=True))

        self.phase = "fighting"
        self.until_ms = deadline
        self._alive = len(ours) + len(theirs)
        self._changed_ms = now
        if self.engagement is not None:
            self.engagement.seconds = 0.0

    # ---- fighting it -------------------------------------------------------------------

    def _fight(self, observation: Observation, view, action: Action, now: int) -> None:
        ours = self.squads.get(OURS)
        theirs = self.squads.get(THEIRS)
        if ours is None or theirs is None:
            self.phase = "clear"
            return

        if ours.members and theirs.members and self.engagement is not None:
            self.engagement.closest = min(self.engagement.closest,
                                          math.hypot(ours.x - theirs.x, ours.y - theirs.y))

        deviations, _ = self.tactics.decide(view, [ours], now)
        action.deviations.extend(deviations)
        their_view = build_view(observation, self.catalogue, None, self.last_regions, invert=True)
        their_deviations, _ = self.opponent.decide(their_view, [theirs], now)
        action.deviations.extend(their_deviations)
        self.statistics.tactical += 1
        self.statistics.decisions += len(deviations)

        if not ours.members or not theirs.members or now >= self.until_ms:
            self._call(ours, theirs, now)
            return
        alive = len(ours.members) + len(theirs.members)
        if alive != self._alive:
            self._alive, self._changed_ms = alive, now
        elif now - self._changed_ms >= STALL_MS:
            self._call(ours, theirs, now, stalled=True)

    def _call(self, ours: SquadRecord, theirs: SquadRecord, now: int, stalled: bool = False) -> None:
        engagement = self.engagement
        if engagement is not None:
            engagement.stalled = stalled
        if engagement is not None:
            engagement.ours_left = len(ours.members)
            engagement.theirs_left = len(theirs.members)
            engagement.seconds = (now - (ours.contract.issued_at_ms if ours.contract else now)) / 1000.0
            self.statistics.history.append(engagement.as_dict())
        if ours.members and not theirs.members:
            self.statistics.won += 1
        elif theirs.members and not ours.members:
            self.statistics.lost += 1
        else:
            self.statistics.drawn += 1
        self.phase = "sweeping"
        self.until_ms = now + SWEEP_MS

    def _sweep(self, observation: Observation, action: Action, now: int) -> None:
        """Sets whatever is left of both sides on each other, because there is no command that removes a unit and a board that is never cleared fills up.

        There is only anything to do here while both sides still have somebody. A fight that ended by one side being destroyed has nothing left to set against anything, and waiting out the sweep in that case is the commonest thing the arena did with its time: the winner stands about for twenty seconds while the next engagement, which is built somewhere else on the map regardless, waits for a clock that is measuring nothing.
        """
        ours = self.squads.get(OURS)
        theirs = self.squads.get(THEIRS)
        contested = (ours is not None and theirs is not None and ours.members and theirs.members)
        if not contested or now >= self.until_ms:
            self.phase = "clear"
            self.squads = {}
            return
        for squad_id, other in ((OURS, THEIRS), (THEIRS, OURS)):
            squad = self.squads[squad_id]
            if squad.contract is None:
                continue
            target = self._region_of(self.squads[other])
            if target == squad.contract.target_region:
                continue
            squad.contract = replace(squad.contract, target_region=target, issued_at_ms=now)
            action.contracts.append(Contract(
                squad=squad_id, task=Task.ATTACK, stance=Stance.AGGRESSIVE,
                target_region=target, cost_budget=squad.contract.cost_budget,
                deadline_ms=now + SWEEP_MS, issued_at_ms=now, override=True))

    # ---- keeping the two squads in step with the board ---------------------------------

    def _fold(self, observation: Observation) -> None:
        alive = {unit.id for unit in observation.unit_states}
        by_id = {state.id: state for state in observation.squads}
        for squad_id, squad in self.squads.items():
            squad.members = [member for member in squad.members if member in alive]
            state = by_id.get(squad_id)
            if state is None:
                continue
            squad.value = state.value
            squad.formed_value = state.formed_value
            squad.x, squad.y = state.x, state.y
            squad.spread = state.spread
            squad.losses = state.losses
            squad.status = Status(state.status)

    def _record(self, squad_id: int, members: Sequence[int], observation: Observation) -> SquadRecord:
        by_id = {unit.id: unit for unit in observation.unit_states}
        doctrine = Doctrine.VANGUARD
        for member in members:
            unit = by_id.get(member)
            found = self.catalogue.doctrine_for(unit.type_index) if unit is not None else None
            if found is not None:
                doctrine = found
                break
        value = sum(self.catalogue.value(by_id[m].type_index) for m in members if m in by_id)
        centre_x = sum(by_id[m].x for m in members if m in by_id) / max(1, len(members))
        centre_y = sum(by_id[m].y for m in members if m in by_id) / max(1, len(members))
        return SquadRecord(id=squad_id, doctrine=doctrine, members=list(members), value=value,
                           formed_value=value, x=centre_x, y=centre_y)

    # ---- where and what ----------------------------------------------------------------

    def _sites(self) -> List[Tuple[float, float]]:
        """Places an engagement may be built on.

        Region centres and resource points both, because a region centre is the mean of the points that formed it and can therefore fall on water or on a cliff, where the engine refuses to place anything and the engagement is stillborn. A resource point is ground something can be built on by definition, so it is ground a unit can be put on. Measured before this: with region centres alone, one instance in a run of eight lost fourteen of its twenty-two engagements to placements that never appeared.
        """
        places = [(region.x, region.y) for region in getattr(self.session, "regions", ())]
        content = getattr(self.session, "map_content", None)
        if content is not None:
            places.extend(content.to_world(tile) for tile in content.resources)
        return places

    def _blame_site(self) -> None:
        """Takes the site of a stillborn engagement out of the pool. Whether ground will take a unit is not something this side can ask, so the only way to find out is to try, and the only thing worth doing with the answer is to remember it."""
        if self.engagement is None or len(self.sites) <= 2:
            return
        site = self.engagement.site
        self.sites = [place for place in self.sites if place != site]

    def _site(self, observation: Observation) -> Optional[Tuple[float, float]]:
        """Somewhere to build the next fight, as far as possible from whatever is still standing. Survivors of an earlier engagement that could not be swept must not wander into the next one, or the fight the layer is paid for is not the fight it was given."""
        if not self.sites:
            self.sites = self._sites()
        if not self.sites:
            return None
        standing = [(unit.x, unit.y) for unit in observation.unit_states]
        if not standing:
            return self.random.choice(self.sites)

        def clearance(site: Tuple[float, float]) -> float:
            return min(math.hypot(site[0] - x, site[1] - y) for x, y in standing)

        best = max(clearance(site) for site in self.sites)
        return self.random.choice([site for site in self.sites if clearance(site) >= best * 0.9])

    def _force(self, budget: float) -> List:
        """A random force worth about the budget, drawn from the types that can fight on the ground.

        Restricted to what moves on the ground and shoots at the ground on purpose. A fight between aircraft and units that cannot elevate is not a fight, and an engagement in which one side cannot be reached teaches the layer only that nothing it does matters.

        A type is only considered if the whole force could be built from it, which is what keeps an engagement from consisting of one enormous machine. The registry runs from a three hundred credit tank to experimental units costing hundreds of times that, and a rule that simply spent until the budget ran out would put a single one of the latter on the board against a dozen of the former and call it a fight.
        """
        pool = [kind for kind in self.catalogue.types
                if kind.mobile and kind.armed and kind.hits_land and kind.price > 0
                and kind.movement in ("LAND", "HOVER")
                and kind.price <= budget / MINIMUM_FORCE]
        if not pool:
            return []
        force: List = []
        spent = 0.0
        while len(force) < MAX_UNITS:
            kind = self.random.choice(pool)
            if spent + kind.price > budget:
                if force:
                    break
                continue
            force.append(kind)
            spent += kind.price
        return force

    def _rows(self, force: Sequence, slot: int, place: Tuple[float, float]) -> List[float]:
        """Spawn rows for one side: type, player slot, position, count. Units are scattered a little so that they do not all arrive on the same point and spend the first seconds pushing each other apart."""
        rows: List[float] = []
        for index, kind in enumerate(force):
            angle = 2 * math.pi * index / max(1, len(force))
            radius = 40.0 + 12.0 * index
            rows.extend([float(kind.index), float(slot),
                         place[0] + math.cos(angle) * radius,
                         place[1] + math.sin(angle) * radius, 1.0])
        return rows

    def _our_slot(self, observation: Observation) -> int:
        return observation.slot

    def _their_slot(self, observation: Observation) -> int:
        """Whose the opposing side of an engagement is.

        The game side settles this as it builds the room and reports it, because it depends on which slots the room filled and which of those the map had nowhere to put. What is wanted is a player that never had a base: one that owns nothing has no income, nothing to build with and nothing to think about, so the only thing that moves its units is this process.
        """
        if self.enemy_slot is not None:
            return self.enemy_slot
        reported = getattr(self.session, "sparring_slot", -1)
        if reported >= 0:
            return reported
        return 1 if observation.slot != 1 else 0

    def _region_of(self, squad: SquadRecord) -> int:
        """The region slot nearest a squad, which is how a position is named to a contract. An arena map has regions like any other, and the nearest one is a good enough name for 'over there'."""
        best = 0
        best_distance = float("inf")
        for region in self.last_regions:
            distance = math.hypot(region.x - squad.x, region.y - squad.y)
            if distance < best_distance:
                best, best_distance = region.id, distance
        return best

    def close(self) -> None:
        for layer in (self.tactics, self.opponent):
            if hasattr(layer, "close"):
                layer.close()
