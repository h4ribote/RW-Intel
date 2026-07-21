"""The build order: where the next credits go.

The economy is the one layer whose decision is a queue rather than a plan. It is asked twice a second what to spend on, and the answer that beats every clever alternative in an opening is a fixed ladder of conditions read top down, because the ordering already encodes everything a build order knows: an extractor before a factory because income compounds and a tank does not, the home region before the ones beyond it because a building outside what we hold is a gift to the enemy, a turret only where something has actually been seen.

What the strategic layer contributes is not a place in that ladder but a weight on it. The posture's shares are shares of what is worth spending on, never a division of the credits themselves, so they are read here as the gates on the steps they belong to: the military share decides whether the factory runs at all, the economy share whether a second builder is worth 500 credits, and the target mix decides which of several things the factory that is running should turn out. A step whose share is below its threshold is simply skipped and the credits fall through to the next one, which is what makes the ladder degrade into "expand only" or "arm only" without any of the steps knowing about a posture.

Two things are worth stating because they are not visible in the code. Placements go only to a builder with no order at all, since a builder given a second placement abandons the first and the half built structure is a total loss; and a placement is remembered for a while after it is issued, because the building it will become does not exist yet and the point it was sent to would otherwise look empty to the next period and be offered again. The same reasoning covers a factory just told to produce: nothing on the wire says a factory is busy, so the layer has to remember that it asked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from ...wire import Production
from ...wire.action import ProductionKind
from .catalogue import Catalogue
from .contracts import EconomyOrders, Replacement, Role
from .view import WorldView

#: A building standing on a resource point is within this of it, and nothing else is.
ON_RESOURCE = 40.0

#: How long a placement is assumed to be on its way before the ground it was sent to is offered again. Opening value: long enough to cover the walk to a neighbouring region as well as the seventeen seconds of building, since a point offered again while a builder is still walking to it is a point two builders are sent to.
PLACEMENT_GRACE_MS = 45000

#: How long a builder may be left on one placement before it is given something else to do. Opening value: comfortably longer than a walk to a neighbouring region plus the seventeen seconds of building, because this is a last resort for a builder sent somewhere it cannot reach. Set anywhere near the honest length of the job it interrupts the job instead of rescuing it, and the builder starts over every time without ever finishing.
PLACEMENT_TIMEOUT_MS = 120000

#: How long a factory is left alone after being told to make something. Opening value: one operational period, which is to say almost nothing.
#:
#: The engine gives a factory a queue and works through it on its own, so what this layer owes it is a queue that is never empty, not an estimate of how long each unit takes. Modelling the factory as busy for the length of a build looks careful and is not: the estimate is always wrong, and every period it is wrong by is a period the factory stood idle with credits in the treasury. The real bound on production is what can be paid for, and that is already enforced.
FACTORY_BUSY_MS = 2000

#: Extractors that have to be standing before the first factory is worth 700 credits.
FACTORY_EXTRACTORS = 2

#: Builders worth keeping. Opening value: one places while the other walks, and a third is idle more often than not until there is a second front to build on.
BUILDER_TARGET = 2

#: Military share above which the factory is worth running at all. Opening value, set below the lowest share any posture carries so that even an expanding opening keeps something in the field.
MILITARY_SHARE_THRESHOLD = 0.15

#: Economy share above which a second builder is worth its 500 credits rather than another extractor. Opening value: below the arming postures rather than between them, so that only a final battle stops us keeping two.
#:
#: Set above the arming share it reads as "stop expanding", which is not what arming means. A match turns to arming early, and with the threshold above that share the whole of it is then played on the one builder it started with: nothing is left to raise a second factory while the first is busy, or to replace the builder when it dies, and the economy stops growing at the moment the army starts costing.
EXPANSION_SHARE_THRESHOLD = 0.25

#: Share above which ground that has seen the enemy is worth a turret. Opening value; read off the military share for want of a defensive one, so only the postures that are arming fortify.
DEFENCE_SHARE_THRESHOLD = 0.45

#: How recently the enemy must have been seen in a region for it to still count as contacted. Opening value: long enough that a raid that passed through is still worth answering, short enough that the front moving on stops the spending.
CONTACT_WINDOW_MS = 60000

#: How close an existing building of ours makes a turret redundant, which is what stops a contacted region from collecting one turret per period.
TURRET_SPACING = 250.0

#: How far from a region's centre a turret stands when the region has no radius to speak of, which is every region grown from a single resource point. Opening value.
TURRET_STANDOFF = 200.0

#: How far a region's centre may be from home for it to count as adjacent. Opening value: regions are agglomerated at 400 world units, so this is a few region widths and not a whole board.
ADJACENT_REGION_DISTANCE = 1600.0

#: Enemy worth in a region that a builder may be sent into anyway. Opening value, about two of the cheapest things that move: a scout passing through is not a reason to stop expanding, and anything heavier is a reason to send a squad first.
ESTABLISHED_ENEMY = 700.0

#: How far from the placing builder a factory goes, far enough not to fight the builder for its own footprint.
FACTORY_OFFSET = 120.0

#: Income above which the economy can keep a second factory fed. Opening value in the engine's own income units, roughly what two extractors bring in: a single factory cannot spend the income of a working expansion, and credits sitting in the treasury are an army that was not built.
SECOND_FACTORY_INCOME = 25.0

#: How much the squads' stated shortfall counts against the posture's target mix when the two disagree. Opening value: half, so a mix is bent towards what the front is asking for without being overridden by it.
REPLACEMENT_WEIGHT = 0.5

#: Roles the factory is asked to fill from the target mix. Builders are left out because the build order spends on them by its own rule and not by the mix, and the rest cannot be produced.
FIELDED_ROLES = (Role.ARMOUR, Role.ARTILLERY, Role.ANTI_AIR, Role.FAST)

#: The one type the build order has to name outright. An extractor announces itself by only standing on a resource pool and a turret by being a building that shoots, but nothing on the wire distinguishes a factory from any other building, so the design's own name for it is used.
FACTORY_LOOKUP = "landFactory"

_LONG_AGO = -10 ** 9

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourcePoint:
    """A place an extractor may stand. Not a thing in the world — a flag on a map tile — which is why the positions come from the map file and not from the observation."""

    index: int
    x: float
    y: float
    region: int


@dataclass
class _Purse:
    """What is left to commit this period. A period's orders are issued in one batch and the engine debits each only as it starts, so the layer has to do this arithmetic itself or it will promise the same credits three times."""

    credits: float

    def afford(self, price: float) -> bool:
        return self.credits >= price

    def spend(self, price: float) -> None:
        self.credits -= price


class Economy:
    def __init__(self, session, catalogue: Catalogue) -> None:
        self.session = session
        self.catalogue = catalogue
        self.points: List[ResourcePoint] = _resource_points(session)
        #: The type that may only stand on a resource pool, which is what an extractor is and the only way to know one without reading its name.
        self.extractor = _cheapest(k for k in catalogue.types if k.extractor)
        self.radius: Dict[int, float] = {region.id: region.radius for region in session.regions}
        self.spawns = [region for region in session.regions if region.spawn]
        #: When each placement or production was last issued, keyed by what it was for. This is the whole of the layer's memory.
        self.issued_at: Dict[object, int] = {}

    def decide(self, view: WorldView, orders: EconomyOrders,
               replacements: Sequence[Replacement]) -> List[Production]:
        now = view.observation.game_time_ms
        purse = _Purse(view.observation.credits)
        idle = [sighting.unit for sighting in view.builders if self._available(sighting.unit, now)]
        out: List[Production] = []
        tech = self._tech(view, orders)

        # An extractor counts towards the factory as soon as it is committed, not only once it is standing. It takes about seventeen seconds to raise and the credits it will bring are already spent; waiting for it to finish before placing the factory costs the opening that whole time for no decision that could still change.
        extractors = sum(1 for s in view.buildings if s.kind is not None and s.kind.extractor)
        extractors += sum(1 for point in self.points
                          if 0 <= now - self.issued_at.get(point.index, _LONG_AGO) < PLACEMENT_GRACE_MS
                          and not self._occupied(view, point))
        factories = [s for s in view.buildings if s.kind is not None and s.kind.lookup == FACTORY_LOOKUP]
        free = [s.unit for s in factories
                if s.unit.built >= 255 and now - self.issued_at.get(("factory", s.unit.id), _LONG_AGO) >= FACTORY_BUSY_MS]
        # A factory ordered but not yet standing is still a factory as far as deciding to build another goes, but only for as long as one could still be on its way. Counted from the last placement rather than kept, or the first order would stand in for a factory for the rest of the match and no second one would ever be worth building.
        awaited = self.issued_at.get("factory-placement", _LONG_AGO)
        placed = 1 if 0 <= now - awaited < PLACEMENT_GRACE_MS and not factories else 0
        built = len(factories) + placed

        # The order of what follows is the build order itself, and each step takes from the same builders and the same purse: reordering the blocks is a change of policy, not a refactor.
        if view.home is not None:
            self._extract(out, view, purse, idle, now, {view.home.id})

        if built == 0 and extractors >= FACTORY_EXTRACTORS:
            self._place_factory(out, purse, idle, now)

        # Every factory that is standing idle, not one of them. A factory is the only thing that turns credits into an army, and an opening that leaves one waiting a period at a time while the treasury grows has already lost the match it is saving for.
        while free and orders.allocation.military >= MILITARY_SHARE_THRESHOLD:
            role = self._wanted(view, orders, replacements)
            kind = self.catalogue.cheapest(role, tech, producer=FACTORY_LOOKUP)
            if kind is None or not purse.afford(kind.price):
                break
            self._produce(out, purse, free.pop(0), kind, now)

        self._extract(out, view, purse, idle, now, self._adjacent_held(view))

        if len(view.builders) < BUILDER_TARGET and orders.allocation.economy >= EXPANSION_SHARE_THRESHOLD and free:
            kind = self.catalogue.cheapest(Role.BUILDER, tech)
            if kind is not None and purse.afford(kind.price):
                self._produce(out, purse, free.pop(0), kind, now)

        if orders.allocation.military >= DEFENCE_SHARE_THRESHOLD:
            self._fortify(out, view, purse, idle, now, tech)

        if built == 1 and view.observation.income >= SECOND_FACTORY_INCOME:
            self._place_factory(out, purse, idle, now)

        # The build order is a ladder of conditions, and what a stalled economy needs said is which rung it stopped on. Every term of every gate, once a period, is small beside a period and is the difference between reading a stall and guessing at one.
        log.debug("credits=%.0f left=%.0f builders=%d/%d extractors=%d factories=%d(%d) idlefactories=%d want=%s openings=%d/%d ordered=%s",
                  view.observation.credits, purse.credits, len(idle), len(view.builders),
                  extractors, len(factories), built, len(free),
                  self.catalogue.cheapest(self._wanted(view, orders, replacements), tech, producer=FACTORY_LOOKUP),
                  self._open_points(view, now, {view.home.id} if view.home else set()),
                  self._open_points(view, now, self._adjacent_held(view)),
                  [self.catalogue.kind(p.type_index).lookup for p in out])

        return out

    # ---- steps -----------------------------------------------------------------------

    def _extract(self, out: List[Production], view: WorldView, purse: _Purse, idle: List,
                 now: int, regions: Set[int]) -> None:
        """An extractor onto every unheld point in the given regions that there is a builder and the money for, safest and nearest to home first.

        Ground is ranked from home rather than from the builder that would walk to it. The two orderings differ exactly where it matters: on a map between two players the nearest unclaimed pool to a builder that has just finished at home is often the one in the middle, and sending the only builder an opening has into the middle is how an opening ends. Ranking from home, and putting any region an enemy is standing in behind every region where none is, keeps the expansion behind the front instead of through it.
        """
        kind = self.extractor
        if kind is None or not regions:
            return
        contested = {region.id for region in view.regions if region.enemy_value > 0 or region.held_by_enemy > 0}
        home = view.home
        safety = self._safety(home)
        while idle and purse.afford(kind.price):
            open_points = [point for point in self.points
                           if point.region in regions
                           and now - self.issued_at.get(point.index, _LONG_AGO) >= PLACEMENT_GRACE_MS
                           and not self._occupied(view, point)]
            if not open_points:
                return
            point = min(open_points, key=lambda p: (p.region in contested, safety(p)))
            builder = min(idle, key=lambda b: math.hypot(point.x - b.x, point.y - b.y))
            self.issued_at[point.index] = now
            self.issued_at[("builder", builder.id)] = now
            idle.remove(builder)
            purse.spend(kind.price)
            out.append(Production(producer=builder.id, type_index=kind.index,
                                  kind=ProductionKind.BUILDING, x=point.x, y=point.y))

    def _open_points(self, view: WorldView, now: int, regions: Set[int]) -> int:
        """Resource points in those regions that are neither taken nor already being walked to."""
        return sum(1 for point in self.points
                   if point.region in regions
                   and now - self.issued_at.get(point.index, _LONG_AGO) >= PLACEMENT_GRACE_MS
                   and not self._occupied(view, point))

    def _available(self, builder, now: int) -> bool:
        """Whether a builder may be given a placement. Nothing queued means it is free; a placement it has been sitting on for too long means it is not going to arrive, and giving it another is how it is recovered."""
        if builder.queued == 0:
            self.issued_at.pop(("builder", builder.id), None)
            return True
        since = self.issued_at.get(("builder", builder.id))
        return since is not None and now - since >= PLACEMENT_TIMEOUT_MS

    def _place_factory(self, out: List[Production], purse: _Purse, idle: List, now: int) -> None:
        kind = self.session.type_by_lookup(FACTORY_LOOKUP)
        if kind is None or not idle or not purse.afford(kind.price):
            return
        builder = idle.pop(0)
        self.issued_at[("builder", builder.id)] = now
        self.issued_at["factory-placement"] = now
        purse.spend(kind.price)
        out.append(Production(producer=builder.id, type_index=kind.index, kind=ProductionKind.BUILDING,
                              x=builder.x + FACTORY_OFFSET, y=builder.y + FACTORY_OFFSET))

    def _fortify(self, out: List[Production], view: WorldView, purse: _Purse, idle: List,
                 now: int, tech: int) -> None:
        """One turret on the home-facing edge of the nearest region the enemy has been seen in. Nearest rather than most recent, because the contact worth answering with a building is the one closest to what the buildings are for."""
        kind = self._turret_kind(tech)
        home = view.home
        if kind is None or home is None or not idle or not purse.afford(kind.price):
            return
        contacted = [region for region in view.regions
                     if region.enemy_seen_at_ms > 0
                     and now - region.enemy_seen_at_ms <= CONTACT_WINDOW_MS
                     and now - self.issued_at.get(("turret", region.id), _LONG_AGO) >= PLACEMENT_GRACE_MS]
        for region in sorted(contacted, key=lambda r: (r.distance_from_home, r.id)):
            x, y = self._facing_home(region, home)
            if any(math.hypot(s.unit.x - x, s.unit.y - y) < TURRET_SPACING for s in view.buildings):
                continue
            builder = min(idle, key=lambda b: math.hypot(b.x - x, b.y - y))
            idle.remove(builder)
            self.issued_at[("builder", builder.id)] = now
            self.issued_at[("turret", region.id)] = now
            purse.spend(kind.price)
            out.append(Production(producer=builder.id, type_index=kind.index,
                                  kind=ProductionKind.BUILDING, x=x, y=y))
            return

    def _produce(self, out: List[Production], purse: _Purse, factory, kind, now: int) -> None:
        self.issued_at[("factory", factory.id)] = now
        purse.spend(kind.price)
        out.append(Production(producer=factory.id, type_index=kind.index, kind=ProductionKind.UNIT))

    # ---- what to make ------------------------------------------------------------------

    def _wanted(self, view: WorldView, orders: EconomyOrders,
                replacements: Sequence[Replacement]) -> Role:
        """Which role the factory fills next: the one furthest below the posture's target mix, bent towards whatever the squads have said they are short of. Shares rather than counts, because the mix is a share of military spending and a heavy tank is two light ones."""
        wanted = {role: share for role, share in orders.target_mix.items()
                  if share > 0 and role in FIELDED_ROLES}
        if not wanted:
            return Role.ARMOUR
        held: Dict[Role, float] = {}
        total = 0.0
        for sighting in view.fighters:
            held[sighting.role] = held.get(sighting.role, 0.0) + sighting.value
            total += sighting.value
        short: Dict[Role, int] = {}
        for replacement in replacements:
            short[replacement.role] = short.get(replacement.role, 0) + replacement.count
        demanded = sum(short.values())

        def deficit(role: Role) -> float:
            standing = held.get(role, 0.0) / total if total > 0 else 0.0
            asked = short.get(role, 0) / demanded if demanded > 0 else 0.0
            return (wanted[role] - standing) + REPLACEMENT_WEIGHT * asked

        return max(sorted(wanted), key=deficit)

    def _tech(self, view: WorldView, orders: EconomyOrders) -> int:
        """The technology level the economy will buy at. The build order has no step that raises it, so what the cap decides is whether the higher tier a base already unlocks may be fielded at all: zero pins the army to tier one whatever is standing."""
        if orders.tech_cap <= 0:
            return 1
        return max((s.kind.tech for s in view.buildings if s.kind is not None and s.unit.built >= 255), default=1)

    def _turret_kind(self, tech: int):
        """The cheapest building that shoots at the ground. Recognised by what it does rather than by its name, so a definition file that renames or replaces the built-in turret changes nothing."""
        return _cheapest(k for k in self.catalogue.types
                         if k.building and k.armed and k.hits_land and k.tech <= tech)

    # ---- ground ------------------------------------------------------------------------

    def _occupied(self, view: WorldView, point: ResourcePoint) -> bool:
        return any(math.hypot(s.unit.x - point.x, s.unit.y - point.y) < ON_RESOURCE for s in view.buildings)

    def _safety(self, home):
        """Ranks ground to expand onto: how far it is from home, less how much deeper into our own half it lies than the enemy's.

        Distance alone sends an opening into the middle of the board. On a map between two players the centre pool is usually the nearest unclaimed one, and it is nearer the enemy's approach than anything behind it: it is where the first army arrives, and the only builder an opening has is what would be standing there. Subtracting the margin — how much closer the point is to us than to them — makes a slightly further pool deep in our own ground beat a slightly nearer one on the line, which is the trade an opening wants. Whether an enemy is standing there right now is a separate and stronger test, applied ahead of this one.
        """
        enemies = []
        if home is not None and len(self.spawns) >= 2:
            start = min(self.spawns, key=lambda r: math.hypot(r.x - home.x, r.y - home.y))
            enemies = [r for r in self.spawns if r.id != start.id]

        def rank(point: ResourcePoint) -> float:
            reach = math.hypot(point.x - home.x, point.y - home.y) if home is not None else 0.0
            if not enemies:
                return reach
            theirs = min(math.hypot(point.x - r.x, point.y - r.y) for r in enemies)
            return reach - (theirs - reach)

        return rank

    def _adjacent_held(self, view: WorldView) -> Set[int]:
        """Nearby regions our side holds, meaning ones we are not being out-weighed in and the enemy has not already drawn from.

        Holding cannot mean "already has an extractor of ours", or the step that expands onto new ground would only ever offer ground already taken and the economy would stop at the point it opened on. Nor can it mean "no enemy anywhere near": the observation is omniscient for now, so one scout crossing a region would veto it for as long as it kept walking, and on a map whose home region holds a single resource point that is the difference between an economy and none at all. What it means is that the enemy is not established there — no extractor of theirs on the ground, and nothing standing that a builder could not be sent past.
        """
        home = view.home
        if home is None:
            return set()
        return {region.id for region in view.regions
                if region.id != home.id and region.held_by_enemy == 0
                and region.enemy_value <= max(ESTABLISHED_ENEMY, region.our_value)
                and math.hypot(region.x - home.x, region.y - home.y) <= ADJACENT_REGION_DISTANCE}

    def _facing_home(self, region, home) -> Tuple[float, float]:
        """The point on a region's edge that lies towards our own base, which is the side an attack on it has to come through and the side a turret can be reached to repair."""
        reach = max(self.radius.get(region.id, 0.0), TURRET_STANDOFF)
        dx, dy = home.x - region.x, home.y - region.y
        span = math.hypot(dx, dy)
        if span <= 1.0:  # the contacted region is home itself, and there is no direction to face
            return region.x, region.y
        return region.x + dx / span * reach, region.y + dy / span * reach


def _cheapest(types):
    """The least expensive of a set of types, ignoring anything the registry prices at nothing since that is a type the engine does not sell."""
    candidates = [kind for kind in types if kind.price > 0]
    return min(candidates, key=lambda kind: kind.price) if candidates else None


def _resource_points(session) -> List[ResourcePoint]:
    """Every resource point on the map with the region it falls in, worked out the same way the region table sent to the game was, so both sides agree on which point belongs where."""
    content = session.map_content
    if content is None or not session.regions:
        return []
    points: List[ResourcePoint] = []
    for index, tile in enumerate(content.resources):
        x, y = content.to_world(tile)
        nearest = min(session.regions, key=lambda r: (r.x - x) ** 2 + (r.y - y) ** 2)
        points.append(ResourcePoint(index=index, x=x, y=y, region=nearest.id))
    return points
