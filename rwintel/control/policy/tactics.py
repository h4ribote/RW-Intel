"""The tactical layer: whether to depart from the contract, and how.

This layer does not drive units. The engine already advances every squad on its contract with path finding, target acquisition and an engagement stance, and that is a competent baseline obtained for nothing; writing tactics as fresh unit control would throw it away and would put the inference count on the number of units rather than on the number of squads. So the only decision here is a departure, one of seven, one per squad, and a squad that is doing well gets no order at all.

Each departure answers one local factor, and the set began at five because the factors that decide a small fight in this game are being shot at, being covered by an area weapon, and out-reaching what is shooting back -plus concentrating, and doing nothing. Two more were added later, not as new kinds of move but as the one parameter a rule used to settle on their behalf: how far a withdrawal commits, and which enemy a concentration goes onto. Anything that cannot be read off those factors belongs to the operational layer, which is the layer that knows why the squad is where it is.

The upward half matters as much. Once the fog is on, nothing above the fighting can see the enemy at all: the per player aggregates the higher layers read are our own side only, and learning what the enemy is fielding means having stood next to it. The squad in contact is therefore the only sensor the command chain has, and the mission report is the only wire it reports on. It is built properly here while the observation is still omniscient precisely so that turning the fog on changes what the report contains and not whether anything is listening.

Rules rather than choices settle where a departure points: the engine picks the focus target and the fall-back position. What this layer picks is only which of the seven, which keeps the action space at the number of kinds whether a script or a network is deciding, and keeps it from ever asking for a continuous value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ...wire import Deviation, SquadDeviation, Status
from .catalogue import Catalogue
from .combat import CombatTable
from .contracts import MissionReport, SquadRecord
from .encoding import tactical_state
from .judgement import TacticsJudge
from .options import Options
from .view import Sighting, WorldView

#: The bit in the commander byte that says a human is driving this squad's units. The command chain reports on such a squad but never orders it: an owner is one person, and a departure sent alongside a human's own orders is the shared control the design refuses to build.
HUMAN_TACTICS = 0b10

#: How far around the squad centre the mission report reads the enemy from. Wider than the engagement radius on purpose: what operations wants is what the squad has run into, including what is coming, and it pays no price for hearing about something a few seconds early.
CONTACT_RADIUS = 900.0

#: How far around the squad centre an enemy counts as part of this fight. Set to the distance the game side reads enemy reach over when it carries out a kite or a fall-back, so that the test made here and the manoeuvre made there are about the same enemies.
ENGAGEMENT_RADIUS = 400.0



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
    def __init__(self, session, catalogue: Catalogue, options: Options = Options()) -> None:
        self.session = session
        self.catalogue = catalogue
        self.tracks: Dict[int, _Track] = {}
        #: The rule ladder, which decides from the encoded fight alone so that a network imitating it sees everything it decided on.
        self.judge = TacticsJudge(predict=options.predict, tuning=options.tuning)
        #: What the fight is expected to come to, which the encoding carries for the judge to read.
        self.combat = CombatTable.load(catalogue, tuning=options.tuning)
        self._view: Optional[WorldView] = None
        self._now = 0

    def decide(self, view: WorldView, squads: List[SquadRecord],
               game_time_ms: int) -> Tuple[List[SquadDeviation], List[MissionReport]]:
        # Held on the instance because the departure is chosen from the encoded fight, and encoding it needs the board and the clock.
        self._view = view
        self._now = game_time_ms
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
                destroyed=track.killed,
                airborne=view.airborne(squad.x, squad.y, CONTACT_RADIUS),
            ))

            # A human holding the tactical command of a squad is driving its units directly, so this layer stays off the wire for it and confines itself to reporting what it can see. A squad being carried is moved by its lift, and the game side carries out no departure for it.
            if squad.commander & HUMAN_TACTICS or not members or squad.status in (Status.AWAITING_LIFT, Status.LIFTING):
                continue
            deviations.append(SquadDeviation(
                squad=squad.id,
                deviation=self._departure(squad, members, threats, losses, track),
            ))
        return deviations, reports

    # ---- the departures ----------------------------------------------------------------

    def state(self, squad: SquadRecord, members: List[Sighting], threats: List[Sighting],
              losses: float, track: _Track) -> List[float]:
        """The squad's fight as the encoding writes it, which is everything the departure is chosen from."""
        return tactical_state(squad, members, threats, losses, track.killed, self._view, self._now, self.combat)

    def _departure(self, squad: SquadRecord, members: List[Sighting], threats: List[Sighting],
                   losses: float, track: _Track) -> Deviation:
        """Which departure, as the rule ladder (`judgement.TacticsJudge`) reads it off the encoded fight.

        The ladder is tried in the order of what would be worst to get wrong. Breaking off comes first because everything below it is a way of fighting better and none of them helps a fight that should not go on: the mission has spent what it was given, or is trading badly, or the combat table expects it to be lost. Scattering comes next because an area weapon on a bunched squad is the fastest way to lose one. Kiting before concentrating because a range advantage is worth more than a focused volley and the two want opposite positions. Concentrating last, as the thing to do when the fight is worth having on the ground it is on. A withdrawal is the whole way out when the squad is reported losing or the fight is expected to be a rout, and a concentration goes onto a gun in reach when there is one.
        """
        return self.judge.choose(self.state(squad, members, threats, losses, track))

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
        """Counts what this squad has destroyed, by remembering everything that came within reach of it and noticing when one of them is no longer on the board. It is an attribution rather than a measurement -a neighbouring squad's kill next to this one is counted here -but the exchange ratio it feeds is a judgement about a place, not about a squad's record."""
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
