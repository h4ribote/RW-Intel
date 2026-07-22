"""The tactical layer: whether to depart from the contract, and how.

This layer does not drive units. The engine already advances every squad on its contract with path finding, target acquisition and an engagement stance, and that is a competent baseline obtained for nothing; writing tactics as fresh unit control would throw it away and would put the inference count on the number of units rather than on the number of squads. So the only decision here is a departure, one of five, one per squad, and a squad that is doing well gets no order at all.

Each of the five answers one local factor, and there are five because the factors that decide a small fight in this game are being shot at, being covered by an area weapon, and out-reaching what is shooting back — plus concentrating, and doing nothing. Anything that cannot be read off those factors belongs to the operational layer, which is the layer that knows why the squad is where it is.

The upward half matters as much. Once the fog is on, nothing above the fighting can see the enemy at all: the per player aggregates the higher layers read are our own side only, and learning what the enemy is fielding means having stood next to it. The squad in contact is therefore the only sensor the command chain has, and the mission report is the only wire it reports on. It is built properly here while the observation is still omniscient precisely so that turning the fog on changes what the report contains and not whether anything is listening.

Rules rather than choices settle where a departure points: the engine picks the focus target and the fall-back position. What this layer picks is only which of the five, which keeps the action space at five whether a script or a network is deciding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ...wire import Deviation, SquadDeviation, Status
from .catalogue import Catalogue
from .contracts import MissionReport, Role, SquadRecord
from .view import Sighting, WorldView

#: The bit in the commander byte that says a human is driving this squad's units. The command chain reports on such a squad but never orders it: an owner is one person, and a departure sent alongside a human's own orders is the shared control the design refuses to build.
HUMAN_TACTICS = 0b10

#: How far around the squad centre the mission report reads the enemy from. Wider than the engagement radius on purpose: what operations wants is what the squad has run into, including what is coming, and it pays no price for hearing about something a few seconds early.
CONTACT_RADIUS = 900.0

#: How far around the squad centre an enemy counts as part of this fight. Set to the distance the game side reads enemy reach over when it carries out a kite or a fall-back, so that the test made here and the manoeuvre made there are about the same enemies.
ENGAGEMENT_RADIUS = 400.0

#: A hit no older than this means the unit is still under fire rather than merely damaged. Two seconds is several exchanges at the tactical rate and well inside the time a squad takes to disengage.
RECENT_HIT_MS = 2000

#: Losses this close to the contract's budget are as good as spent, and the mission is worth breaking off before the rest of it goes too. The game side calls a mission losing at seven tenths, so this sits above that: hearing "losing" is information for operations, acting on it is a decision made here.
BUDGET_CLOSE_SHARE = 0.8

#: Below this much lost the exchange ratio is one cheap unit's worth of noise and says nothing about how the fight is going.
EXCHANGE_MIN_LOSS = 200.0

#: Enemy worth destroyed under this multiple of our own worth lost is a trade to walk away from. Below one rather than at one because a squad that is merely breaking even is still spending a budget it was given for a purpose.
EXCHANGE_BAD_RATIO = 0.6

#: A squad whose members stand within this of their centre is bunched enough for one area weapon to cover several of them. The game side scatters to 140, so a squad already looser than this has nothing to gain by scattering again.
BUNCHED_SPREAD = 140.0

#: This share of the squad hit inside the recent window at once reads as one weapon covering the squad rather than as several weapons picking at it. Without an area radius on the type table this is the whole of the evidence for an area weapon.
SPLASH_HIT_SHARE = 0.5

#: Fewer members than this and there is no formation to scatter, only units to send in separate directions.
SPREAD_MIN_MEMBERS = 3

#: How much further the squad has to reach than what is shooting at it before backing away while shooting pays for the ground given up. Under this the gap closes again before the squad has fired.
KITE_RANGE_MARGIN = 80.0

#: Fewer shooters than this and concentrating them changes nothing, because the engine's own target acquisition already has them on the same few enemies.
FOCUS_MIN_SHOOTERS = 3

#: With fewer enemies than this there is nothing to concentrate away from: the squad is already fighting the only thing present.
FOCUS_MIN_ENEMIES = 2


@dataclass
class _Track:
    """What one squad's fight looks like over time rather than in one frame.

    Two of the tests are about a trend and cannot be read from a single period. Losses are measured from the worth the squad held when its contract was issued, which is the same origin the game side measures them from, so a restated contract resets them on both sides at once. Kills are counted by watching which enemies stood next to this squad and are then gone from the board; with the fog on that will need the game side's own kill ledger instead, because gone from view will no longer mean gone.
    """

    issued_at_ms: int = -1
    value_at_issue: float = 0.0
    #: Enemy id to worth, for everything that has been inside the engagement radius since the contract was issued.
    engaged: Dict[int, float] = field(default_factory=dict)
    killed: float = 0.0


class Tactics:
    def __init__(self, session, catalogue: Catalogue) -> None:
        self.session = session
        self.catalogue = catalogue
        self.tracks: Dict[int, _Track] = {}

    def decide(self, view: WorldView, squads: List[SquadRecord],
               game_time_ms: int) -> Tuple[List[SquadDeviation], List[MissionReport]]:
        mine: Dict[int, Sighting] = {s.unit.id: s for s in view.ours}
        alive: Set[int] = {s.unit.id for s in view.enemies}
        reported = {squad.id for squad in squads}
        for squad_id in [k for k in self.tracks if k not in reported]:
            del self.tracks[squad_id]

        deviations: List[SquadDeviation] = []
        reports: List[MissionReport] = []
        for squad in squads:
            track = self._track(squad, game_time_ms)
            members = [mine[unit] for unit in squad.members if unit in mine]
            near = view.enemies_near(squad.x, squad.y, ENGAGEMENT_RADIUS)
            threats = [e for e in near if self._threatens(e, members)]
            self._observe(track, threats, alive)

            losses = self._losses(view, squad, track)
            reports.append(MissionReport(
                squad=squad.id,
                status=squad.status,
                contact=view.contact(squad.x, squad.y, CONTACT_RADIUS),
                losses=losses,
            ))

            # A human holding the tactical command of a squad is driving its units directly, so this layer stays off the wire for it and confines itself to reporting what it can see.
            if squad.commander & HUMAN_TACTICS or not members:
                continue
            deviations.append(SquadDeviation(
                squad=squad.id,
                deviation=self._departure(squad, members, threats, losses, track),
            ))
        return deviations, reports

    # ---- the five ----------------------------------------------------------------------

    def _departure(self, squad: SquadRecord, members: List[Sighting], threats: List[Sighting],
                   losses: float, track: _Track) -> Deviation:
        """Which departure, tried in the order of what would be worst to get wrong.

        Breaking off comes first because everything below it is a way of fighting better and none of them helps a fight that should not go on. Scattering comes next because an area weapon on a bunched squad is the fastest way to lose one. Kiting before concentrating because a range advantage is worth more than a focused volley and the two want opposite positions. Concentrating last, as the thing to do when the fight is worth having on the ground it is on.

        Two of the departures also carry the choice a rule on the game side used to make on their behalf. A withdrawal is the short step back that repositions a squad, unless the squad is being destroyed, when it is the whole way out of the fight. A concentration goes onto the weakest enemy, unless a longer-ranged one is close enough to shoot at, when it goes onto that: the gun that out-reaches the squad does the most damage and dies to a focused volley like anything else.
        """
        if not threats and not self._under_fire(members):
            return Deviation.HOLD
        if self._spent(squad, losses) or self._losing_the_exchange(losses, track):
            return Deviation.WITHDRAW_FAR if squad.status == Status.LOSING else Deviation.WITHDRAW
        if self._covered_by_area_fire(squad, members, threats):
            return Deviation.SPREAD
        if self._out_ranges(members, threats) >= KITE_RANGE_MARGIN:
            return Deviation.KITE
        if self._worth_concentrating(members, threats):
            return Deviation.FOCUS_THREAT if self._long_range_in_reach(members, threats) else Deviation.FOCUS
        return Deviation.HOLD

    def _spent(self, squad: SquadRecord, losses: float) -> bool:
        """Whether the mission has cost what it was given to spend. The contract's figure is in credits and so are the losses, which is the reason the design passes an absolute budget rather than a share: the comparison is a subtraction and it means the same thing an hour into the match as it did at the start."""
        if squad.status == Status.LOSING:
            return True
        budget = squad.contract.cost_budget if squad.contract is not None else 0.0
        return budget > 0.0 and losses >= budget * BUDGET_CLOSE_SHARE

    @staticmethod
    def _losing_the_exchange(losses: float, track: _Track) -> bool:
        """Whether this fight is buying less than it costs. Judged over the life of the contract rather than over one period, since a squad that has just traded a tank for a tank has shown nothing yet."""
        return losses >= EXCHANGE_MIN_LOSS and track.killed < losses * EXCHANGE_BAD_RATIO

    @staticmethod
    def _covered_by_area_fire(squad: SquadRecord, members: List[Sighting],
                              threats: List[Sighting]) -> bool:
        """Whether one weapon appears to be covering the squad rather than several picking at it.

        There is no area radius on the type table, so the evidence is circumstantial and deliberately conservative: the squad is tight enough for one blast to reach several of it, most of it took a hit inside the same short window, and something with an artillery's reach is in range. Scattering a squad that is merely losing a firefight costs it its formation for nothing, so all three have to hold.
        """
        if len(members) < SPREAD_MIN_MEMBERS or squad.spread > BUNCHED_SPREAD:
            return False
        if not any(e.role == Role.ARTILLERY for e in threats):
            return False
        hit = sum(1 for m in members if m.unit.since_hit_ms <= RECENT_HIT_MS)
        return hit >= len(members) * SPLASH_HIT_SHARE

    @staticmethod
    def _out_ranges(members: List[Sighting], threats: List[Sighting]) -> float:
        """By how far the squad out-reaches what is shooting at it.

        Ours is the shortest reach among the armed members, because that is the distance at which all of the squad is in the fight and it is also what the game side backs the squad off to. Theirs is the longest among the threats, because that is what decides how close is safe. Taking the two the other way round would order a kite the squad cannot hold.
        """
        reaches = [m.kind.range for m in members if m.kind is not None and m.kind.armed]
        enemy = [e.kind.range for e in threats if e.kind is not None and e.kind.armed]
        if not reaches or not enemy:
            return 0.0
        return min(reaches) - max(enemy)

    @staticmethod
    def _worth_concentrating(members: List[Sighting], threats: List[Sighting]) -> bool:
        """Whether putting the whole squad onto one enemy would change the fight. It needs enough shooters to take something out of the fight sooner than the engine's own spread of targets would, more than one enemy to choose between, and a target inside the squad's reach: the game side only concentrates on what is already in range, so ordering it otherwise produces no command at all."""
        shooters = [m for m in members if m.kind is not None and m.kind.armed]
        if len(shooters) < FOCUS_MIN_SHOOTERS or len(threats) < FOCUS_MIN_ENEMIES:
            return False
        reach = min(m.kind.range for m in shooters)
        centre_x = sum(m.unit.x for m in shooters) / len(shooters)
        centre_y = sum(m.unit.y for m in shooters) / len(shooters)
        return any(math.hypot(e.unit.x - centre_x, e.unit.y - centre_y) <= reach for e in threats)

    @staticmethod
    def _long_range_in_reach(members: List[Sighting], threats: List[Sighting]) -> bool:
        """Whether a longer-ranged enemy is close enough to shoot at, which is the enemy a concentration is better spent on than the weakest one.

        A gun that out-reaches the squad does the most damage while it lives and is no harder to kill than anything else once the squad is in range of it, so taking it out first buys more than shortening the count by one. Judged the same way _worth_concentrating judges reach - the squad's shortest weapon range from the shooters' centre - because a target the squad would have to walk to is not one a focused volley reaches. Artillery is the role the type table gives the long-ranged land types, so it stands in for the reach comparison here.
        """
        shooters = [m for m in members if m.kind is not None and m.kind.armed]
        if not shooters:
            return False
        reach = min(m.kind.range for m in shooters)
        centre_x = sum(m.unit.x for m in shooters) / len(shooters)
        centre_y = sum(m.unit.y for m in shooters) / len(shooters)
        return any(e.role == Role.ARTILLERY
                   and math.hypot(e.unit.x - centre_x, e.unit.y - centre_y) <= reach
                   for e in threats)

    @staticmethod
    def _under_fire(members: List[Sighting]) -> bool:
        """Whether anything is shooting at the squad. This is what turns the layer on at all: a squad moving or waiting is the engine's business, and the time since a unit was last hit is the cheapest signal there is that it has stopped being so."""
        return any(m.unit.since_hit_ms <= RECENT_HIT_MS for m in members)

    @staticmethod
    def _threatens(enemy: Sighting, members: List[Sighting]) -> bool:
        """Whether an enemy can shoot at this squad at all. A squad of aircraft over a line of tanks that cannot elevate is not in a fight, however close the two are, and reading the range difference off units that cannot touch each other would order a kite away from nothing."""
        if enemy.kind is None or not enemy.kind.armed:
            return False
        for member in members:
            if member.kind is None:
                continue
            airborne = member.kind.movement == "AIR"
            reachable = enemy.kind.hits_air if airborne else enemy.kind.hits_land
            if reachable:
                return True
        return False

    # ---- keeping count -----------------------------------------------------------------

    def _track(self, squad: SquadRecord, game_time_ms: int) -> _Track:
        """This squad's memory, reset whenever a new contract is issued. A contract is the boundary the losses and the exchange are measured across, so carrying either over from the last mission would have a squad break off a fresh attack for what a finished one cost."""
        issued = squad.contract.issued_at_ms if squad.contract is not None else -1
        track = self.tracks.get(squad.id)
        if track is None or track.issued_at_ms != issued:
            track = _Track(issued_at_ms=issued, value_at_issue=squad.value)
            self.tracks[squad.id] = track
        return track

    @staticmethod
    def _observe(track: _Track, threats: List[Sighting], alive: Set[int]) -> None:
        """Counts what this squad has destroyed, by remembering everything that came within reach of it and noticing when one of them is no longer on the board. It is an attribution rather than a measurement — a neighbouring squad's kill next to this one is counted here — but the exchange ratio it feeds is a judgement about a place, not about a squad's record."""
        for enemy in threats:
            track.engaged.setdefault(enemy.unit.id, enemy.value)
        for enemy_id in [k for k in track.engaged if k not in alive]:
            track.killed += track.engaged.pop(enemy_id)

    @staticmethod
    def _losses(view: WorldView, squad: SquadRecord, track: _Track) -> float:
        """What this mission has cost so far. The game keeps the same figure against its own ledger and sends it whenever the squad block is on the frame; between those frames it is reconstructed from the worth the squad held when the contract was issued, which is how the game derives it too."""
        row = next((s for s in view.observation.squads if s.id == squad.id), None)
        if row is not None:
            return row.losses
        return max(0.0, track.value_at_issue - squad.value)
