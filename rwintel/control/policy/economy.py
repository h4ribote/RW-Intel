"""The build order: where the next credits go.

The economy is the one layer whose decision is a queue rather than a plan. It is asked twice a second what to spend on, and it answers one investment at a time: every period it lists what it could do now -a unit from an idle factory, a builder, an extractor on open ground, a factory, a tier raise, a turret- and asks which comes next, carries that out, and asks again, until the answer is to stop. The list and the carrying out are this module's; which comes next is the judge's (`judgement.EconomyJudge`), which reads the same encoded numbers a learnt layer reads, and a learnt layer replaces only that answer.

The judge's answer is a fixed ladder of conditions read top down, because the ordering already encodes everything a build order knows: an extractor before a factory because income compounds and a tank does not, the home region before the ones beyond it because a building outside what we hold is a gift to the enemy, a turret only where something has actually been seen. What the strategic layer contributes is not a place in that ladder but a weight on it: the posture's shares are the gates on the rungs they belong to, and the target mix decides which of several things an idle factory should turn out.

Choosing something the treasury cannot pay for yet is how saving for it is said. Its price is held back from everything chosen after it in the period, and it is not offered again until the next one.

Two things are worth stating because they are not visible in the code. Placements go only to a builder with no order at all, since a builder given a second placement abandons the first and the half built structure is a total loss; and a placement is remembered for a while after it is issued, because the building it will become does not exist yet and the point it was sent to would otherwise look empty to the next period and be offered again. The same reasoning covers a factory just told to produce: nothing on the wire says a factory is busy, so the layer has to remember that it asked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...wire import BLOCK_MENUS, Production
from ...wire.action import ProductionKind
from .catalogue import Catalogue
from .combat import CombatTable, Profile
from .contracts import EconomyOrders, Replacement, Role
from .encoding import (
    FIELDED,
    INVESTMENT_SLOTS,
    EconomicBoard,
    Investment,
    Offer,
    economic_state,
    investment_mask,
    lay_out,
)
from .contracts import Domain, Function
from .ground import enemy_spawns, established, safety_rank
from .judgement import EconomyJudge
from .logistics import TRANSPORT_SLOTS
from .options import Options
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

#: Builders worth keeping. Opening value: one places while the other walks, and a third is idle more often than not until there is a second front to build on.
BUILDER_TARGET = 2

#: How recently the enemy must have been seen in a region for it to still count as contacted. Opening value: long enough that a raid that passed through is still worth answering, short enough that the front moving on stops the spending.
CONTACT_WINDOW_MS = 60000

#: How close an existing building of ours makes a turret redundant, which is what stops a contacted region from collecting one turret per period.
TURRET_SPACING = 250.0

#: How far from a region's centre a turret stands when the region has no radius to speak of, which is every region grown from a single resource point. Opening value.
TURRET_STANDOFF = 200.0

#: Open resource points in the expansion plan per builder kept beyond BUILDER_TARGET. Opening value: one builder places while another walks, so a third pays once there are several places to walk to.
POINTS_PER_BUILDER = 3

#: The most builders the economy keeps however much ground is open. Opening value.
MAX_BUILDERS = 5

#: The most the technology fund may hold, so that a long spell without a raise to buy does not bank a whole tier ahead of time. Opening value.
TECH_FUND_LIMIT = 8000.0

#: How far from the placing builder a factory goes, far enough not to fight the builder for its own footprint.
FACTORY_OFFSET = 120.0

#: The turn between successive factory placements, which spreads any number of attempts evenly around the builder.
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

#: Placements on one resource point that never stood before the point is given up on for the match. Opening value.
POINT_ATTEMPTS = 2

#: How long a builder with an errand may stand still away from its site before it is taken to be walking nowhere and given something else. Opening value: several operational periods, longer than a builder waits for a path to clear.
STUCK_MS = 20000

#: How far a builder may drift and still be standing still, which allows for the jostling of units around it.
STUCK_MOVE = 24.0

#: How near its site a builder is raising the building rather than walking to it. Opening value: the footprint of a large building and the builder's own reach.
SITE_REACH = 200.0

#: How long a builder ordered from a factory counts as on its way, so that one missing builder is not ordered again every period while the first is still being made. Opening value.
BUILDER_ORDER_GRACE_MS = 45000

#: Share of the unit cap above which each unit is chosen for the most worth it can carry rather than for the least credits. Below it the credits run out first and worth is price whichever unit buys it; above it the slots run out first. Opening value.
CAP_SHARE = 0.6

#: How long a building told to raise its tier is left alone before being told again, which covers the raise itself. Opening value.
UPGRADE_GRACE_MS = 60000

#: How much the squads' stated shortfall counts against the posture's target mix when the two disagree. Opening value: half, so a mix is bent towards what the front is asking for without being overridden by it.
REPLACEMENT_WEIGHT = 0.5

#: Roles the factory is asked to fill from the target mix. Builders are left out because the build order spends on them by its own rung and not by the mix, and the rest cannot be produced.
FIELDED_ROLES = FIELDED

#: Transports worth keeping at most, which is the number of slots the lift layer holds them in.
MAX_TRANSPORTS = TRANSPORT_SLOTS

#: How long after a builder is sent across to a region no other is sent there, which covers a lift's crossing. Opening value.
FERRY_GRACE_MS = 60000

#: How long a builder handed to a lift counts as being carried before the lift layer has reported holding it.
FERRY_PENDING_MS = 5000

#: How long a builder given a placement counts as busy before its queue shows the order. The game takes an order on a later frame than it is sent, so for at least one period the builder reads as having nothing queued.
PLACEMENT_PENDING_MS = 5000

#: How long a transport ordered from a factory counts as on its way. Opening value, as for a builder.
TRANSPORT_ORDER_GRACE_MS = 45000

_LONG_AGO = -10 ** 9

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourcePoint:
    """A place an extractor may stand. Not a thing in the world -a flag on a map tile -which is why the positions come from the map file and not from the observation."""

    index: int
    x: float
    y: float
    region: int


#: What a type the combat table has no numbers for is taken to be: something that does not fight.
_UNARMED = Profile(price=0.0, hp=0.0, dps=0.0, range=0.0, flying=False, hits_air=False, hits_land=False, measured=False)


@dataclass
class _Purse:
    """What is left to commit this period. A period's orders are issued in one batch and the engine debits each only as it starts, so the layer has to do this arithmetic itself or it will promise the same credits three times."""

    credits: float

    def afford(self, price: float) -> bool:
        return self.credits >= price

    def spend(self, price: float) -> None:
        self.credits -= price


@dataclass
class _Books:
    """One period's working state: what is left to spend and to spend it with, as the picks of the period use it up."""

    view: WorldView
    orders: EconomyOrders
    now: int
    purse: _Purse
    idle: List
    #: Finished factories not told to produce lately, in the order they are handed work.
    free: List
    #: Factories that were to make something this period and are waiting for it to be paid for instead.
    waiting: List
    #: Our factories standing at the start of the period.
    factories: List
    tech: int
    expansion: Set[int]
    enemy: List[Tuple[int, float]]
    at_cap: bool
    near_cap: bool
    #: The fielded roles in the order the factories fill them.
    ranked: List[Role]
    rank: Callable[[float, float], float]
    contested: Set[int]
    #: Whether there is a fight on the water, which is when warships are worth building.
    water_front: bool = False
    #: What the squads have said they are short of.
    replacements: Sequence[Replacement] = ()
    #: Credits held back for what was chosen and cannot be paid for yet.
    reserve: float = 0.0
    #: What may be done only once a period, and has been: a builder, a factory, a tier raise, a turret, and whatever was saved for.
    used: Set[object] = field(default_factory=set)
    out: List[Production] = field(default_factory=list)
    picks: int = 0
    #: Fighting strength per credit and per unit, by type, against this period's enemy.
    strength: Dict[Tuple[int, str], float] = field(default_factory=dict)

    @property
    def budget(self) -> float:
        return self.purse.credits - self.reserve


@dataclass
class Ledger:
    """How well credits were turned into things over an episode: what sat in the treasury, how often a factory was left without an order and a builder without a job, and when the opening's milestones were reached. A stalled build order shows up here rather than as an error.

    A factory's production queue is not on the wire, so a factory left without an order may still be working through earlier ones; what the count says is how often this layer held production back, for want of credits beyond the reserve, at the unit cap, or below the military share.
    """

    periods: int = 0
    credits: float = 0.0
    factory_periods: int = 0
    unordered_factory_periods: int = 0
    builder_periods: int = 0
    idle_builder_periods: int = 0
    #: Game time at which each milestone first stood, or -1 if it never did.
    first_factory_ms: int = -1
    second_extractor_ms: int = -1
    second_factory_ms: int = -1
    #: Credits committed by what they were committed to: `army_<role>` for production by role, and `builders`, `extractors`, `factories`, `turrets` and `upgrades`. A commitment is an order issued, which the engine debits only when the work starts, so an order that never starts is counted too.
    spent: Dict[str, float] = field(default_factory=dict)
    #: Periods since the first factory stood, and how many of them had no finished factory at all: the economy that cannot make anything until it rebuilds.
    since_factory_periods: int = 0
    factoryless_periods: int = 0
    #: Periods on which the treasury held more than the banking line (`tuning.banked_credits`), which is an army that was paid for and not built.
    banked_periods: int = 0

    def commit(self, category: str, price: float) -> None:
        self.spent[category] = self.spent.get(category, 0.0) + price

    def summary(self) -> Dict[str, object]:
        return {
            "credits_mean": round(self.credits / self.periods, 1) if self.periods else 0.0,
            "factory_unordered": round(self.unordered_factory_periods / self.factory_periods, 4) if self.factory_periods else 0.0,
            "builder_idle": round(self.idle_builder_periods / self.builder_periods, 4) if self.builder_periods else 0.0,
            "first_factory_s": self.first_factory_ms / 1000 if self.first_factory_ms >= 0 else -1,
            "second_extractor_s": self.second_extractor_ms / 1000 if self.second_extractor_ms >= 0 else -1,
            "second_factory_s": self.second_factory_ms / 1000 if self.second_factory_ms >= 0 else -1,
            "factoryless": round(self.factoryless_periods / self.since_factory_periods, 4) if self.since_factory_periods else 0.0,
            "banked": round(self.banked_periods / self.periods, 4) if self.periods else 0.0,
            "spent": {category: round(credits) for category, credits in sorted(self.spent.items())},
        }


class Economy:
    def __init__(self, session, catalogue: Catalogue, options: Options = Options()) -> None:
        self.session = session
        self.catalogue = catalogue
        self.options = options
        self.tuning = options.tuning
        self.judge = EconomyJudge(options)
        self.points: List[ResourcePoint] = _resource_points(session)
        #: The type that may only stand on a resource pool, which is what an extractor is and the only way to know one without reading its name.
        self.extractor = _cheapest(k for k in catalogue.types if k.extractor)
        self.radius: Dict[int, float] = {region.id: region.radius for region in session.regions}
        self.spawns = [region for region in session.regions if region.spawn]
        #: When each placement or production was last issued, keyed by what it was for.
        self.issued_at: Dict[object, int] = {}
        #: When each builder still on its way was ordered from a factory.
        self.builder_orders: List[int] = []
        #: What each of our buildings and builders offers to produce or place, by unit id, as the engine last reported it.
        self.menus: Dict[int, List[int]] = {}
        #: Where each builder last stood still from, and since when, which is how a builder walking nowhere is recognised.
        self.still_since: Dict[int, Tuple[float, float, int]] = {}
        #: Where each builder was last sent to place something, and the resource point when that was an extractor.
        self.sites: Dict[int, Tuple[float, float]] = {}
        self.extracting: Dict[int, int] = {}
        #: Credits the strategic layer's technology cap has put aside for tier raises and not yet spent, and when it was last topped up.
        self.tech_fund = 0.0
        self.funded_at_ms: Optional[int] = None
        self.ledger = Ledger()
        #: Resource points seen with a building on them since their last placement, and how many placements on each never stood.
        self.stood: Set[int] = set()
        self.failures: Dict[int, int] = {}
        #: Factory placements issued so far, which is what turns the next one to fresh ground.
        self.factory_attempts = 0
        #: What each type is worth in a fight, which is what the choice of what to make and which factory to raise reads.
        self.combat = CombatTable.load(catalogue, tuning=options.tuning)
        #: Our buildings as last seen, by id, which is what says which kind and tier a menu belongs to.
        self._standing: Dict[int, object] = {}
        #: The factory the build order opens with: of the buildings a builder places that make something that fights, the cheapest whose first tier makes something that fights on the ground.
        self.base_factory = _cheapest(k for k in (catalogue.kind(i) for i in catalogue.factories)
                                      if k is not None and any(catalogue.has(i, Function.COMBAT) and catalogue.domain(i) == Domain.GROUND
                                                               for i in k.menu))
        #: The building types that count as factories: the base factory, and, when switched to choose, every building a builder places that makes something that fights.
        self.factory_kinds: Set[int] = self._factory_kinds()
        #: The lift layer, which carries a builder to ground it cannot walk to; handed over by the policy each period.
        self.logistics = None
        #: Builders on their way across water, by builder id, and the regions they were sent to, with the game time the lift was asked for.
        self.ferried: Dict[int, int] = {}
        self.ferry_regions: Dict[int, int] = {}
        #: When each transport still on its way was ordered from a factory.
        self.transport_orders: List[int] = []
        #: Whether an enemy warship has been seen this match, which is what makes warships worth building.
        self.warships_seen = False

    def _factory_kinds(self) -> Set[int]:
        base = {self.base_factory.index} if self.base_factory is not None else set()
        if not self.options.choose:
            return base
        return base | set(self.catalogue.factories)

    def _makes(self, factory_kind, tier: int = 1) -> List:
        """What a kind of building makes at a tier: the menu of one of ours standing at that tier when there is one, which is the engine's own answer; else, at the first tier, the menu the catalogue reads off the engine's sample of the kind; else what the definition files say a building of that name makes at that tier."""
        for unit_id, menu in self.menus.items():
            standing = self._standing.get(unit_id)
            if standing is not None and standing.type_index == factory_kind.index and max(1, standing.level) == tier:
                return [k for k in (self.catalogue.kind(i) for i in menu) if k is not None]
        if tier == 1 and factory_kind.menu:
            return [k for k in (self.catalogue.kind(i) for i in factory_kind.menu) if k is not None]
        definitions = getattr(self.catalogue, "definitions", {}) or {}
        definition = definitions.get(factory_kind.lookup) or definitions.get(factory_kind.name)
        names = set(definition.builds) if definition is not None else set()
        return [k for k in self.catalogue.types if k.lookup in names and k.tech == tier]

    def decide(self, view: WorldView, orders: EconomyOrders,
               replacements: Sequence[Replacement]) -> List[Production]:
        now = view.observation.game_time_ms
        if view.observation.blocks & BLOCK_MENUS:
            if log.isEnabledFor(logging.DEBUG) and view.observation.menus != self.menus:
                log.debug("menus %s", {unit: [self.catalogue.kind(i).lookup if self.catalogue.kind(i) else i for i in menu]
                                       for unit, menu in view.observation.menus.items()})
            self.menus = dict(view.observation.menus)
        self._standing = {s.unit.id: s.unit for s in view.buildings}
        self._track_builders(view, now)
        books = self._open_books(view, orders, replacements, now)

        # One investment at a time, each one taken against what the ones before it left: the purse, the idle builders and factories, and the reserve for what is being saved for.
        while books.picks < INVESTMENT_SLOTS:
            slots = lay_out(self._offers(books))
            if not any(offer is not None for offer in slots[1:]):
                break
            board = self._board(books)
            state = economic_state(board, slots)
            pick = self._choose(state, investment_mask(slots), slots, board)
            books.picks += 1
            if pick is None or pick <= 0 or slots[pick] is None:
                break
            self._carry_out(slots[pick], books)

        self._keep_ledger(view, now, books.factories, books.free + books.waiting, books.idle)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("builders %s", [(s.unit.id, round(s.unit.x), round(s.unit.y), s.unit.queued, s.unit.order)
                                      for s in view.builders])
            # What a stalled economy needs said is where the credits stopped: once a period, every figure the choice was made on.
            log.debug("credits=%.0f left=%.0f reserve=%.0f units=%d/%d tech=%d builders=%d/%d factories=%d idlefactories=%d waiting=%d picks=%d ordered=%s",
                      view.observation.credits, books.purse.credits, books.reserve, view.observation.units,
                      view.observation.unit_cap, books.tech, len(books.idle), len(view.builders),
                      len(books.factories), len(books.free), len(books.waiting), books.picks,
                      ["upgrade" if p.kind == ProductionKind.UPGRADE else self.catalogue.kind(p.type_index).lookup
                       for p in books.out])
        return books.out

    def _choose(self, state: List[float], mask: List[float], slots: Sequence[Optional[Offer]],
                board: Optional[EconomicBoard] = None) -> Optional[int]:
        """Which slot comes next, given the encoded board, which slots hold an offer, the offers and the board they were encoded from. The judge's answer; a learnt economy answers here instead and nowhere else."""
        return self.judge.choose(state)

    # ---- the period's books -------------------------------------------------------------

    def _open_books(self, view: WorldView, orders: EconomyOrders, replacements: Sequence[Replacement],
                    now: int) -> _Books:
        observation = view.observation
        tech = self._tech(view, orders)
        self._fund(orders, now)
        factories = [s for s in view.buildings if s.kind is not None and s.kind.index in self.factory_kinds]
        free = [s.unit for s in factories
                if s.unit.built >= 255 and now - self.issued_at.get(("factory", s.unit.id), _LONG_AGO) >= FACTORY_BUSY_MS]
        self.builder_orders = [at for at in self.builder_orders if 0 <= now - at < BUILDER_ORDER_GRACE_MS]
        self.transport_orders = [at for at in self.transport_orders if 0 <= now - at < TRANSPORT_ORDER_GRACE_MS]
        home = view.home
        return _Books(
            view=view, orders=orders, now=now, purse=_Purse(observation.credits),
            idle=[sighting.unit for sighting in view.builders if self._available(sighting.unit, now)],
            free=free, waiting=[], factories=factories, tech=tech,
            expansion=self._expansion(view, orders), enemy=self._enemy_mix(view),
            at_cap=observation.unit_cap > 0 and observation.units >= observation.unit_cap,
            near_cap=observation.unit_cap > 0 and observation.units >= CAP_SHARE * observation.unit_cap,
            ranked=self._ranked(view, orders, replacements),
            rank=safety_rank(home, enemy_spawns(self.spawns, home)),
            contested={region.id for region in view.regions if region.enemy_value > 0 or region.held_by_enemy > 0},
            replacements=list(replacements),
            water_front=self._water_front(view),
        )

    def _built(self, books: _Books) -> Tuple[int, bool]:
        """Factories standing, counting one ordered and not yet standing when none stands, and whether a placement is still on its way. Counted from the last placement rather than kept, or the first order would stand in for a factory for the rest of the match and no second one would ever be worth building."""
        awaited = self.issued_at.get("factory-placement", _LONG_AGO)
        pending = 0 <= books.now - awaited < PLACEMENT_GRACE_MS
        return len(books.factories) + (1 if pending and not books.factories else 0), pending

    def _extractors(self, view: WorldView, now: int) -> int:
        """Extractors standing and on their way. An extractor counts as soon as it is committed: it takes about seventeen seconds to raise and the credits it will bring are already spent, and waiting for it to finish before placing the factory costs the opening that whole time for no decision that could still change."""
        standing = sum(1 for s in view.buildings if s.kind is not None and s.kind.extractor)
        return standing + sum(1 for point in self.points
                              if 0 <= now - self.issued_at.get(point.index, _LONG_AGO) < PLACEMENT_GRACE_MS
                              and not self._occupied(view, point))

    def _board(self, books: _Books) -> EconomicBoard:
        view, orders, now = books.view, books.orders, books.now
        observation = view.observation
        built, pending = self._built(books)
        ours = sum(s.value for s in view.ours)
        theirs = sum(s.value for s in view.enemies)
        army = sum(s.value for s in view.fighters)
        enemy_army = sum(s.value for s in view.enemy_fighters)
        flying = sum(s.value for s in view.enemy_fighters if s.kind is not None and s.kind.movement == "AIR")
        held = sum(region.held_by_us for region in view.regions)
        enemy_held = sum(region.held_by_enemy for region in view.regions)
        on_hand = len(view.builders) + len(self.builder_orders)
        home = {view.home.id} if view.home is not None else set()
        return EconomicBoard(
            credits=books.purse.credits, reserve=books.reserve, income=observation.income,
            units=observation.units, unit_cap=observation.unit_cap, near_cap=books.near_cap, at_cap=books.at_cap,
            factories=built, pending_factory=pending, free_factories=len(books.free),
            factory_raising=any(0 <= now - self.issued_at.get(("upgrade", s.unit.id), _LONG_AGO) < UPGRADE_GRACE_MS
                                for s in books.factories),
            builders=len(view.builders),
            builder_shortfall=self._builder_target(view, now, books.expansion) - on_hand,
            idle_builders=len(books.idle), extractors=self._extractors(view, now),
            home_open=self._open_points(view, now, home), plan_open=self._open_points(view, now, books.expansion),
            posture=orders.posture, economy_share=orders.allocation.economy,
            military_share=orders.allocation.military, tech_share=orders.allocation.tech,
            tech_cap=orders.tech_cap, tech_fund=self.tech_fund, tech_fund_limit=TECH_FUND_LIMIT,
            tech_level=books.tech, target_mix=dict(orders.target_mix),
            value_edge=_share(ours, theirs), army_edge=_share(army, enemy_army),
            ground_edge=_share(held, enemy_held), enemy_air=flying / enemy_army if enemy_army > 0 else 0.0,
            rebuilding=built == 0 and self.ledger.first_factory_ms >= 0,
            transports=len(view.transports) + len(self.transport_orders),
            transport_wanted=bool(self._transport_wanted(books)), water_front=books.water_front,
            picks=books.picks, game_time_ms=now,
        )

    # ---- what the strategic layer hands down --------------------------------------------

    def _expansion(self, view: WorldView, orders: EconomyOrders) -> Set[int]:
        """The regions of the expansion plan the enemy has not become established in since it was drawn up. The plan arrives on the strategic period, and a region can be taken in between."""
        return {region.id for region in view.regions if region.id in orders.expansion and not established(region)}

    def _headquarters(self, view: WorldView, now: int) -> List:
        """Finished command centres not told to produce within FACTORY_BUSY_MS, when builders may be ordered from them."""
        if not self.options.hq:
            return []
        return [s.unit for s in view.buildings
                if s.kind is not None and s.kind.index in self.catalogue.headquarters and s.unit.built >= 255
                and now - self.issued_at.get(("factory", s.unit.id), _LONG_AGO) >= FACTORY_BUSY_MS]

    def _builder_target(self, view: WorldView, now: int, expansion: Set[int]) -> int:
        """How many builders are worth keeping: two, and one more for every few resource points open in the expansion plan, up to MAX_BUILDERS. Ground nobody is walking to is income nobody is collecting."""
        if self.options.builders == "fixed":
            return BUILDER_TARGET
        return min(MAX_BUILDERS, BUILDER_TARGET + self._open_points(view, now, expansion) // POINTS_PER_BUILDER)

    def _fund(self, orders: EconomyOrders, now: int) -> None:
        """Tops up the technology fund by the strategic layer's cap, which is in credits per minute, for the game time since the last period."""
        if self.funded_at_ms is not None and now > self.funded_at_ms:
            self.tech_fund = min(TECH_FUND_LIMIT, self.tech_fund + orders.tech_cap * (now - self.funded_at_ms) / 60000.0)
        self.funded_at_ms = now

    def _funded(self, stage: int) -> bool:
        """Whether a raise at this stage is paid from the technology fund rather than from the economy's own spending. `factory` puts only the factory's tier under the cap, since that is what decides the technology the army is built at, and leaves raising extractors to the economy as the investment in income it is."""
        mode = self.options.tech
        return mode == "all" or (mode == "factory" and stage == 1)

    def _keep_ledger(self, view: WorldView, now: int, factories: List, free: List, idle: List) -> None:
        ledger = self.ledger
        ledger.periods += 1
        ledger.credits += view.observation.credits
        finished = [s for s in factories if s.unit.built >= 255]
        ledger.factory_periods += len(finished)
        ledger.unordered_factory_periods += len(free)
        ledger.builder_periods += len(view.builders)
        ledger.idle_builder_periods += len(idle)
        extractors = sum(1 for s in view.buildings if s.kind is not None and s.kind.extractor and s.unit.built >= 255)
        if ledger.first_factory_ms < 0 and len(finished) >= 1:
            ledger.first_factory_ms = now
        if ledger.second_factory_ms < 0 and len(finished) >= 2:
            ledger.second_factory_ms = now
        if ledger.second_extractor_ms < 0 and extractors >= 2:
            ledger.second_extractor_ms = now
        if ledger.first_factory_ms >= 0:
            ledger.since_factory_periods += 1
            if not finished:
                ledger.factoryless_periods += 1
        if view.observation.credits > self.tuning.banked_credits:
            ledger.banked_periods += 1

    # ---- what could be done now ------------------------------------------------------------

    def _offers(self, books: _Books) -> List[Offer]:
        offers = [Offer(kind=Investment.STOP)]
        # Ground first: a point only a transport could take a builder to is what tells the lift layer a transport is wanted, which the units on offer then answer.
        offers.extend(self._extractor_offers(books))
        offers.extend(self._unit_offers(books))
        builder = self._builder_offer(books)
        if builder is not None:
            offers.append(builder)
        offers.extend(self._factory_offers(books))
        offers.extend(self._raise_offers(books))
        turret = self._turret_offer(books)
        if turret is not None:
            offers.append(turret)
        return offers

    def _strength(self, books: _Books, type_index: int, per: str) -> float:
        key = (type_index, per)
        if key not in books.strength:
            books.strength[key] = self.combat.efficiency(type_index, books.enemy, per)
        return books.strength[key]

    def _unit_offers(self, books: _Books) -> List[Offer]:
        """Every type an idle factory offers that fills a fielded role or fights, once each whichever factories offer it, warships only while there is a fight on the water; and, while the lift layer has a request no transport could serve and there is room for another, every transport that would load what was asked for. Nothing at the unit cap, where nothing more can be made."""
        if books.at_cap or not books.free:
            return []
        view = books.view
        wanted = self._transport_wanted(books)
        kinds: Dict[int, object] = {}
        for factory in books.free:
            for kind in self._offered(factory):
                if kind.tech > books.tech or kind.builder or kind.building or kind.index in kinds:
                    continue
                if self._fighting([kind], books.water_front) or (
                        self.catalogue.role(kind.index) in FIELDED_ROLES and self.catalogue.domain(kind.index) != Domain.NAVAL):
                    kinds[kind.index] = kind
                elif wanted and self.catalogue.role(kind.index) == Role.TRANSPORT and wanted <= set(kind.carries):
                    kinds[kind.index] = kind
        if not kinds:
            return []
        army = sum(s.value for s in view.fighters)
        fielded: Dict[int, float] = {}
        for sighting in view.fighters:
            fielded[sighting.unit.type_index] = fielded.get(sighting.unit.type_index, 0.0) + sighting.value
        asked: Dict[Role, int] = {}
        for replacement in books.replacements:
            asked[replacement.role] = asked.get(replacement.role, 0) + replacement.count
        demanded = sum(asked.values())
        wealth = books.purse.credits + view.observation.income * self.tuning.save_horizon_s
        order = {role: position for position, role in enumerate(books.ranked)}
        offers = []
        for kind in kinds.values():
            role = self.catalogue.role(kind.index)
            profile = self.combat.profile(kind.index) or _UNARMED
            offers.append(Offer(
                kind=Investment.UNIT, price=float(kind.price), type_index=kind.index, in_reach=kind.price <= wealth,
                efficiency=self._strength(books, kind.index, "credit"),
                unit_efficiency=self._strength(books, kind.index, "unit"),
                fighting=bool(self._fighting([kind], books.water_front)), role=role,
                transport=role == Role.TRANSPORT, naval=self.catalogue.domain(kind.index) == Domain.NAVAL,
                role_rank=1.0 - order[role] / len(FIELDED_ROLES) if role in order else 0.0,
                flying=profile.flying, hits_air=profile.hits_air, hits_land=profile.hits_land,
                range=profile.range, tier=kind.tech,
                army_share=fielded.get(kind.index, 0.0) / army if army > 0 else 0.0,
                asked=asked.get(role, 0) / demanded if demanded else 0.0,
                payload=kind))
        return offers

    def _builder_offer(self, books: _Books) -> Optional[Offer]:
        """A builder from the command centre, or else from the first idle factory whose menu has one, narrowed to what that building makes: the cheapest thing anywhere that reads as a builder is a scenario creature no building here produces. The command centre comes first so the factories stay on the army, and because it is the only thing left that can make one when the last builder died before a factory stood."""
        if "builder" in books.used:
            return None
        for producer in self._headquarters(books.view, books.now)[:1] + list(books.free):
            kind = _cheapest(k for k in self._offered(producer) if k.builder and k.tech <= books.tech)
            if kind is not None:
                return Offer(kind=Investment.BUILDER, price=float(kind.price), type_index=kind.index,
                             payload=(producer, kind))
        return None

    def _extractor_offers(self, books: _Books) -> List[Offer]:
        """An extractor for every region of home and the expansion plan with an open point and an idle builder able to place one there: the point the region offers is its safest.

        Ground is ranked from home rather than from the builder that would walk to it. The two orderings differ exactly where it matters: on a map between two players the nearest unclaimed pool to a builder that has just finished at home is often the one in the middle, and sending the only builder an opening has into the middle is how an opening ends.
        """
        kind = self.extractor
        view = books.view
        able = [b for b in books.idle if self._can_place(b, kind)] if kind is not None else []
        if not able:
            return []
        home = view.home
        regions = set(books.expansion) | ({home.id} if home is not None else set())
        plan = list(books.orders.expansion)
        offers = []
        for region in view.regions:
            if region.id not in regions or ("extractor", region.id) in books.used:
                continue
            points = [p for p in self.points if p.region == region.id and self._open(view, p, books.now)]
            if not points:
                continue
            point = min(points, key=lambda p: books.rank(p.x, p.y))
            # A point no idle builder can get to is offered only when a builder can be carried there, and taking it then asks for the lift rather than placing anything, so it costs nothing yet.
            walkers = [b for b in able if self._reaches(b, point.x, point.y)]
            lift = not walkers
            if lift and not self._ferriable(able, region.id, books.now):
                continue
            ready = any(math.hypot(b.x - region.x, b.y - region.y) <= max(self.radius.get(region.id, 0.0), SITE_REACH) * 2
                        for b in walkers)
            offers.append(Offer(
                kind=Investment.EXTRACTOR, price=0.0 if lift else float(kind.price), type_index=kind.index,
                region=region.id, home=home is not None and region.id == home.id, contested=region.id in books.contested,
                safety=books.rank(point.x, point.y),
                plan_rank=1.0 - plan.index(region.id) / len(plan) if region.id in plan else 0.0,
                distance=region.distance_from_home, open_points=len(points), lift=lift, ready=ready, payload=point))
        return offers

    def _factory_offers(self, books: _Books) -> List[Offer]:
        """A factory of every kind an idle builder can place: the land factory, and when switched to choose every kind of building that makes something that fights. A match that has had a factory and lost every one may, when switched to recover, take a builder off its errand for it."""
        if "factory" in books.used:
            return []
        builders = list(books.idle)
        built, _ = self._built(books)
        if not builders and built == 0 and self.ledger.first_factory_ms >= 0 and self.options.recovery:
            builders = self._commandeer(books.view)
        if not builders:
            return []
        base = self.base_factory
        kinds = [base] if base is not None else []
        if self.options.choose:
            placeable = {index for b in builders for index in self.menus.get(b.id, ())}
            kinds += [k for k in (self.catalogue.kind(i) for i in sorted(self.factory_kinds))
                      if k is not None and k.price > 0 and (base is None or k.index != base.index)
                      and (not placeable or k.index in placeable)]
        standing: Dict[int, int] = {}
        for s in books.factories:
            standing[s.kind.index] = standing.get(s.kind.index, 0) + 1
        offers = []
        for kind in kinds:
            able = [b for b in builders if self._can_place(b, kind)]
            if not able:
                continue
            worth = max((self._strength(books, k.index, "credit") for k in self._fighting(self._makes(kind))), default=0.0)
            offers.append(Offer(kind=Investment.FACTORY, price=float(kind.price), type_index=kind.index,
                                land_factory=base is not None and kind.index == base.index, worth=worth,
                                count=standing.get(kind.index, 0), payload=(kind, able)))
        return offers

    def _raise_offers(self, books: _Books) -> List[Offer]:
        """The tier raise at each stage nearest home: an extractor to its second tier, a factory to its next, an extractor to its third. Only ground no enemy is on is invested in, a building told to raise is left alone while the raise goes through, and a raise the technology mode puts under the strategic layer's cap is offered only once the fund holds its price."""
        if "raise" in books.used:
            return []
        view = books.view
        home = view.home
        best: Dict[int, Tuple[float, int, object, object]] = {}
        for sighting in view.buildings:
            unit, kind = sighting.unit, sighting.kind
            if kind is None or unit.built < 255 or unit.upgrade_price <= 0:
                continue
            if 0 <= books.now - self.issued_at.get(("upgrade", unit.id), _LONG_AGO) < UPGRADE_GRACE_MS:
                continue
            if self._region_of(view, unit.x, unit.y) in books.contested:
                continue
            if kind.extractor:
                stage = 0 if unit.level <= 1 else 2
            elif kind.index in self.factory_kinds:
                stage = 1
            else:
                continue
            if self._funded(stage) and self.tech_fund < unit.upgrade_price:
                continue
            reach = math.hypot(unit.x - home.x, unit.y - home.y) if home is not None else 0.0
            if stage not in best or (reach, unit.id) < best[stage][:2]:
                best[stage] = (reach, unit.id, unit, kind)
        offers = []
        for stage, (_, _, unit, kind) in sorted(best.items()):
            offers.append(Offer(kind=Investment.RAISE, price=float(unit.upgrade_price), stage=stage,
                                gain=self._raise_gain(books, unit, kind) if stage == 1 else 0.0,
                                funded=self._funded(stage), payload=unit))
        return offers

    def _turret_offer(self, books: _Books) -> Optional[Offer]:
        """One turret on the home-facing edge of the nearest region the enemy has been seen in. Nearest rather than most recent, because the contact worth answering with a building is the one closest to what the buildings are for."""
        if "turret" in books.used:
            return None
        kind = self._turret_kind(books.tech)
        view = books.view
        home = view.home
        if kind is None or home is None or not books.idle:
            return None
        now = books.now
        contacted = [region for region in view.regions
                     if region.enemy_seen_at_ms > 0
                     and now - region.enemy_seen_at_ms <= CONTACT_WINDOW_MS
                     and now - self.issued_at.get(("turret", region.id), _LONG_AGO) >= PLACEMENT_GRACE_MS]
        for region in sorted(contacted, key=lambda r: (r.distance_from_home, r.id)):
            x, y = self._facing_home(region, home)
            if any(math.hypot(s.unit.x - x, s.unit.y - y) < TURRET_SPACING for s in view.buildings):
                continue
            able = [b for b in books.idle if self._can_place(b, kind) and self._reaches(b, x, y)]
            if not able:
                continue
            builder = min(able, key=lambda b: math.hypot(b.x - x, b.y - y))
            return Offer(kind=Investment.TURRET, price=float(kind.price), type_index=kind.index, contact=True,
                         distance=region.distance_from_home, payload=(builder, x, y, region.id))
        return None

    # ---- carrying a choice out -----------------------------------------------------------

    def _carry_out(self, offer: Offer, books: _Books) -> None:
        """Does what was chosen, or, when the treasury cannot pay for it yet, holds its price back from everything chosen after it this period and takes it off the table until the next."""
        paid = offer.price <= books.budget
        if offer.kind == Investment.UNIT:
            factory = next((f for f in books.free
                            if any(k.index == offer.type_index for k in self._offered(f))), None)
            if factory is None:
                return
            books.free.remove(factory)
            if paid:
                self._produce(books.out, books.purse, factory, offer.payload, books.now)
            else:
                # The factory waits for it and makes nothing else, and its price is held back from the rest, or what it waits for is spent from under it every period and never bought.
                books.waiting.append(factory)
                books.reserve += offer.price
            return
        if offer.kind == Investment.BUILDER:
            books.used.add("builder")
            if not paid:
                books.reserve += offer.price
                return
            producer, kind = offer.payload
            self.builder_orders.append(books.now)
            if producer in books.free:
                books.free.remove(producer)
            self._produce(books.out, books.purse, producer, kind, books.now)
            return
        if offer.kind == Investment.EXTRACTOR:
            # Carrying a builder across spends nothing yet, so it goes ahead whatever is held back.
            if not paid and not offer.lift:
                books.used.add(("extractor", offer.region))
                books.reserve += offer.price
                return
            self._extract(offer, books)
            return
        if offer.kind == Investment.FACTORY:
            books.used.add("factory")
            if not paid:
                books.reserve += offer.price
                return
            self._place_factory(offer, books)
            return
        if offer.kind == Investment.RAISE:
            books.used.add("raise")
            if not paid:
                books.reserve += offer.price
                return
            self._raise(offer, books)
            return
        if offer.kind == Investment.TURRET:
            books.used.add("turret")
            if not paid:
                books.reserve += offer.price
                return
            builder, x, y, region = offer.payload
            books.idle.remove(builder)
            self._send(builder, x, y, books.now)
            self.issued_at[("turret", region)] = books.now
            books.purse.spend(offer.price)
            self.ledger.commit("turrets", offer.price)
            books.out.append(Production(producer=builder.id, type_index=offer.type_index,
                                        kind=ProductionKind.BUILDING, x=x, y=y))

    def _extract(self, offer: Offer, books: _Books) -> None:
        """An extractor onto the point the offer names, placed by the idle builder nearest it that can get there; or, when none can, a lift to carry to the point's region the idle builder nearest it that a free transport can take, after which it places from there."""
        point = offer.payload
        kind = self.extractor
        able = [b for b in books.idle if self._can_place(b, kind) and self._reaches(b, point.x, point.y)]
        if not able:
            self._ferry([b for b in books.idle if self._can_place(b, kind)], point.region, books)
            return
        builder = min(able, key=lambda b: math.hypot(point.x - b.x, point.y - b.y))
        if point.index in self.issued_at and point.index not in self.stood:
            self.failures[point.index] = self.failures.get(point.index, 0) + 1
        self.stood.discard(point.index)
        self.issued_at[point.index] = books.now
        self._send(builder, point.x, point.y, books.now)
        self.extracting[builder.id] = point.index
        books.idle.remove(builder)
        books.purse.spend(kind.price)
        self.ledger.commit("extractors", kind.price)
        books.out.append(Production(producer=builder.id, type_index=kind.index,
                                    kind=ProductionKind.BUILDING, x=point.x, y=point.y))

    def _place_factory(self, offer: Offer, books: _Books) -> None:
        """A factory beside the first builder able to place it; or, in a match that has had a factory and lost every one, beside home, placed by the able builder nearest home, since the builder it takes may be standing at the front."""
        kind, able = offer.payload
        view = books.view
        built, _ = self._built(books)
        home = view.home if built == 0 and self.ledger.first_factory_ms >= 0 and self.options.recovery else None
        builder = min(able, key=lambda b: math.hypot(b.x - home.x, b.y - home.y)) if home is not None else able[0]
        if builder in books.idle:
            books.idle.remove(builder)
        self.issued_at["factory-placement"] = books.now
        books.purse.spend(kind.price)
        self.ledger.commit("factories", kind.price)
        dx, dy = self._factory_offset()
        x, y = (home.x + dx, home.y + dy) if home is not None else (builder.x + dx, builder.y + dy)
        self._send(builder, x, y, books.now)
        books.out.append(Production(producer=builder.id, type_index=kind.index, kind=ProductionKind.BUILDING, x=x, y=y))

    def _raise(self, offer: Offer, books: _Books) -> None:
        """Raises the building the offer names, paying from the technology fund when the mode puts it there. A factory being raised is not also handed a unit to make."""
        unit = offer.payload
        self.issued_at[("upgrade", unit.id)] = books.now
        if offer.funded:
            self.tech_fund -= offer.price
        books.purse.spend(offer.price)
        self.ledger.commit("upgrades", offer.price)
        books.free = [f for f in books.free if f.id != unit.id]
        books.out.append(Production(producer=unit.id, type_index=0, kind=ProductionKind.UPGRADE))

    # ---- mechanics --------------------------------------------------------------------------

    def _open_points(self, view: WorldView, now: int, regions: Set[int]) -> int:
        """Resource points in those regions that are neither taken nor already being walked to."""
        return sum(1 for point in self.points if point.region in regions and self._open(view, point, now))

    def _open(self, view: WorldView, point: ResourcePoint, now: int) -> bool:
        """Whether a resource point may be offered to a builder: nothing stands on it, no placement on it is still on its way, and, when giving up is switched on, it has not had POINT_ATTEMPTS placements that never stood. Straight-line distance cannot see water, so on a map of islands some points are ones a builder can walk towards for ever."""
        if now - self.issued_at.get(point.index, _LONG_AGO) < PLACEMENT_GRACE_MS:
            return False
        if self._occupied(view, point):
            self.stood.add(point.index)
            return False
        if not self.options.retry:
            return True
        pending = 1 if point.index in self.issued_at and point.index not in self.stood else 0
        return self.failures.get(point.index, 0) + pending < POINT_ATTEMPTS

    def _available(self, builder, now: int) -> bool:
        """Whether a builder may be given a placement. One being carried is not, nor one given a placement too recently for its queue to show it; otherwise nothing queued means it is free; a placement it has been sitting on for too long means it is not going to arrive, and giving it another is how it is recovered. When switched to recover, so is a builder that has stood still away from where it was sent for STUCK_MS: it is walking nowhere.

        A builder found free with an extractor it was sent to place not standing gives the point back, so that it is offered again rather than held for PLACEMENT_GRACE_MS: the order never reached the builder or the engine refused it. The placement counts as one that never stood.
        """
        if self._ferrying(builder, now):
            return False
        since = self.issued_at.get(("builder", builder.id))
        if builder.queued == 0:
            if since is not None and 0 <= now - since < PLACEMENT_PENDING_MS:
                return False
            self.issued_at.pop(("builder", builder.id), None)
            self.sites.pop(builder.id, None)
            point = self.extracting.pop(builder.id, None)
            if point is not None and point not in self.stood and point in self.issued_at:
                self.failures[point] = self.failures.get(point, 0) + 1
                del self.issued_at[point]
            return True
        if since is None:
            return False
        return now - since >= PLACEMENT_TIMEOUT_MS or (self.options.recovery and self._stuck(builder, now))

    def _track_builders(self, view: WorldView, now: int) -> None:
        """Notes, for every builder, since when it has been standing where it is."""
        tracked: Dict[int, Tuple[float, float, int]] = {}
        for sighting in view.builders:
            unit = sighting.unit
            last = self.still_since.get(unit.id)
            if last is not None and math.hypot(unit.x - last[0], unit.y - last[1]) <= STUCK_MOVE:
                tracked[unit.id] = last
            else:
                tracked[unit.id] = (unit.x, unit.y, now)
        self.still_since = tracked

    def _stuck(self, builder, now: int) -> bool:
        """Whether a builder with an errand has stood still for STUCK_MS somewhere other than the site it was sent to. A builder raising a building stands still at the site, which is why the distance matters."""
        still = self.still_since.get(builder.id)
        site = self.sites.get(builder.id)
        if still is None or site is None or now - still[2] < STUCK_MS:
            return False
        return math.hypot(builder.x - site[0], builder.y - site[1]) > SITE_REACH

    def _commandeer(self, view: WorldView) -> List:
        """The builder nearest home, taken off whatever it was doing, for the one errand that outranks every other."""
        home = view.home
        if not view.builders:
            return []
        if home is None:
            return [view.builders[0].unit]
        return [min((s.unit for s in view.builders), key=lambda b: math.hypot(b.x - home.x, b.y - home.y))]

    def _reaches(self, builder, x: float, y: float) -> bool:
        """Whether a builder can get to a point to place there: by walking or crossing under its own power, or for a building ship, from water within its reach of the point. True while the map's passage is not known."""
        reach = self.logistics.reach if self.logistics is not None else None
        if reach is None:
            return True
        if getattr(builder, "carrier", 0):
            return False
        kind = self.catalogue.kind(builder.type_index)
        if kind is None:
            return False
        if self.catalogue.domain(kind.index) == Domain.NAVAL:
            return reach.ship_site((x, y), (builder.x, builder.y), kind.range) is not None
        return reach.reachable(kind.movement, (builder.x, builder.y), (x, y))

    def _carriable(self, builders: Sequence) -> List[Tuple[int, int, str, float, float]]:
        """Builders as the lift layer takes passengers: id, type, movement and position, leaving out building ships, which no transport loads."""
        found = []
        for builder in builders:
            kind = self.catalogue.kind(builder.type_index)
            if kind is not None and self.catalogue.domain(kind.index) != Domain.NAVAL and not getattr(builder, "carrier", 0):
                found.append((builder.id, builder.type_index, kind.movement, builder.x, builder.y))
        return found

    def _ferriable(self, builders: Sequence, region: int, now: int) -> bool:
        """Whether one of these builders can be carried to the region now: no lift is already taking a builder there, and a free transport could take one of them. When none could, the lift layer is told, which is what the build order buys transports against."""
        if self.logistics is None or self.logistics.reach is None:
            return False
        if 0 <= now - self.ferry_regions.get(region, _LONG_AGO) < FERRY_GRACE_MS:
            return False
        passengers = self._carriable(builders)
        for passenger in passengers:
            if self.logistics.options(-1, [passenger[1:]], [region]):
                return True
        if passengers:
            self.logistics.want([passengers[0][1]])
        return False

    def _ferry(self, builders: Sequence, region: int, books: _Books) -> None:
        """Has the lift layer carry to the region the builder nearest it that a free transport can take."""
        if self.logistics is None:
            return
        target = books.view.region(region)
        passengers = self._carriable(builders)
        if target is not None:
            passengers.sort(key=lambda p: math.hypot(p[3] - target.x, p[4] - target.y))
        for passenger in passengers:
            if self.logistics.lift_units([passenger], region, books.now):
                self.ferried[passenger[0]] = books.now
                self.ferry_regions[region] = books.now
                books.used.add(("extractor", region))
                books.idle = [b for b in books.idle if b.id != passenger[0]]
                return

    def _ferrying(self, builder, now: int) -> bool:
        """Whether a builder is the cargo of a lift, or was handed to one too recently for the lift to have reported it."""
        since = self.ferried.get(builder.id)
        if since is None:
            return False
        held = self.logistics is not None and any(builder.id in slot.units for slot in self.logistics.slots)
        if held or now - since < FERRY_PENDING_MS:
            return True
        del self.ferried[builder.id]
        return False

    def _can_place(self, builder, kind) -> bool:
        """Whether a builder offers to place this type, as its menu says. A builder whose menu has not arrived yet is taken to offer everything."""
        menu = self.menus.get(builder.id)
        return menu is None or kind.index in menu

    def _send(self, builder, x: float, y: float, now: int) -> None:
        self.issued_at[("builder", builder.id)] = now
        self.sites[builder.id] = (x, y)
        self.extracting.pop(builder.id, None)

    def _factory_offset(self) -> Tuple[float, float]:
        """Where from the builder a factory is placed. The same diagonal step every time unless retrying is switched on; then each placement turns by the golden angle and every few placements step further out, so that ground the engine refused once is not offered again on the next attempt."""
        attempt = self.factory_attempts
        self.factory_attempts += 1
        if not self.options.retry:
            return FACTORY_OFFSET, FACTORY_OFFSET
        reach = FACTORY_OFFSET * math.sqrt(2.0) * (1.0 + 0.5 * (attempt // 6))
        angle = math.pi / 4.0 + attempt * GOLDEN_ANGLE
        return reach * math.cos(angle), reach * math.sin(angle)

    @staticmethod
    def _region_of(view: WorldView, x: float, y: float):
        """The region whose centre is nearest a point, by id, or None on a board with no regions."""
        if not view.regions:
            return None
        return min(view.regions, key=lambda r: (r.x - x) ** 2 + (r.y - y) ** 2).id

    def _offered(self, producer) -> List:
        """The types a building offers to produce, as its menu says. Before its menu has arrived, which is only the first period it stands, the first-tier menu the catalogue holds for its kind stands in."""
        menu = self.menus.get(producer.id)
        if menu is None:
            own = self.catalogue.kind(producer.type_index)
            menu = own.menu if own is not None else ()
        return [kind for kind in (self.catalogue.kind(index) for index in menu) if kind is not None and kind.price > 0]

    def _fighting(self, kinds: Iterable, naval: bool = False) -> List:
        """Of these types, the ones that fight: able to attack by the engine's own account, mobile, with a weapon on record, and warships only when `naval` says there is a fight on the water, since a ship cannot reach a fight inland."""
        return [k for k in kinds if self.catalogue.has(k.index, Function.COMBAT)
                and (naval or self.catalogue.domain(k.index) != Domain.NAVAL)
                and (self.combat.profile(k.index) or _UNARMED).armed]

    def _transport_wanted(self, books: _Books) -> Set[int]:
        """The passenger types a transport is wanted for: those of the requests no transport could serve, while fewer transports stand or are on order than the lift layer has slots for. Empty when none is wanted."""
        if self.logistics is None or self.logistics.shortfall.count <= 0:
            return set()
        if len(books.view.transports) + len(self.transport_orders) >= MAX_TRANSPORTS:
            return set()
        return set(self.logistics.shortfall.passengers)

    def _water_front(self, view: WorldView) -> bool:
        """Whether there is a fight on the water, which is an enemy warship having been seen at any time this match."""
        if not self.warships_seen:
            self.warships_seen = any(self.catalogue.domain(s.unit.type_index) == Domain.NAVAL for s in view.enemy_fighters)
        return self.warships_seen

    def _enemy_mix(self, view: WorldView) -> List[Tuple[int, float]]:
        """What the enemy fields, as types and how many of each are standing counting health, which is what the choice of what to build is made against. Before anything of theirs has been seen, what we field ourselves stands in for it: the two armies are built from the same catalogue."""
        found: Dict[int, float] = {}
        for sighting in view.enemy_fighters or view.fighters:
            unit = sighting.unit
            share = unit.health / unit.max_health if unit.max_health > 0 else 1.0
            found[unit.type_index] = found.get(unit.type_index, 0.0) + share
        return sorted(found.items())

    def _produce(self, out: List[Production], purse: _Purse, factory, kind, now: int) -> None:
        self.issued_at[("factory", factory.id)] = now
        purse.spend(kind.price)
        role = self.catalogue.role(kind.index)
        if role == Role.TRANSPORT:
            self.transport_orders.append(now)
        self.ledger.commit("builders" if role == Role.BUILDER else f"army_{role.name.lower()}", kind.price)
        out.append(Production(producer=factory.id, type_index=kind.index, kind=ProductionKind.UNIT))

    def _ranked(self, view: WorldView, orders: EconomyOrders,
                replacements: Sequence[Replacement]) -> List[Role]:
        """The fielded roles in the order the factories fill them: furthest below the posture's target mix first, bent towards whatever the squads have said they are short of, and the roles the mix does not ask for after all of those. Shares rather than counts, because the mix is a share of military spending and a heavy tank is two light ones."""
        wanted ={role: share for role, share in orders.target_mix.items()
                  if share > 0 and role in FIELDED_ROLES}
        rest = [role for role in FIELDED_ROLES if role not in wanted]
        if not wanted:
            return [Role.ARMOUR] + [role for role in rest if role != Role.ARMOUR]
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

        return sorted(wanted, key=lambda role: (-deficit(role), role)) + rest

    def _raise_gain(self, books: _Books, unit, kind) -> float:
        """How much more the best thing a factory's next tier makes is worth per credit than the best it makes now, less one; one when it makes nothing that fights now and would then, nought when it would not."""
        now_best = max((self._strength(books, k.index, "credit") for k in self._fighting(self._offered(unit))), default=0.0)
        next_best = max((self._strength(books, k.index, "credit")
                         for k in self._fighting(self._makes(kind, max(1, unit.level) + 1))), default=0.0)
        if now_best > 0:
            return next_best / now_best - 1.0
        return 1.0 if next_best > 0 else 0.0

    def _tech(self, view: WorldView, orders: EconomyOrders) -> int:
        """The technology level the economy will buy at: the highest tier a finished factory of ours stands at, since that is what decides what the factory makes. Other buildings do not count: a raised extractor reports a type of a higher level, and reading that as the factory's would order units it cannot make. The cap decides whether the higher tier may be fielded at all: zero pins the army to tier one whatever is standing."""
        if orders.tech_cap <= 0:
            return 1
        return max([1] + [s.unit.level for s in view.buildings
                          if s.kind is not None and s.unit.built >= 255 and s.kind.index in self.factory_kinds])

    def _turret_kind(self, tech: int):
        """The cheapest building that shoots at the ground. Recognised by what it does rather than by its name, so a definition file that renames or replaces the built-in turret changes nothing."""
        return _cheapest(k for k in self.catalogue.types
                         if k.building and k.armed and k.hits_land and k.tech <= tech)

    # ---- ground ------------------------------------------------------------------------

    def _occupied(self, view: WorldView, point: ResourcePoint) -> bool:
        return any(math.hypot(s.unit.x - point.x, s.unit.y - point.y) < ON_RESOURCE for s in view.buildings)

    def _facing_home(self, region, home) -> Tuple[float, float]:
        """The point on a region's edge that lies towards our own base, which is the side an attack on it has to come through and the side a turret can be reached to repair."""
        reach = max(self.radius.get(region.id, 0.0), TURRET_STANDOFF)
        dx, dy = home.x - region.x, home.y - region.y
        span = math.hypot(dx, dy)
        if span <= 1.0:  # the contacted region is home itself, and there is no direction to face
            return region.x, region.y
        return region.x + dx / span * reach, region.y + dy / span * reach


def _share(part: float, against: float) -> float:
    total = part + against
    return part / total if total > 0 else 0.5


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
