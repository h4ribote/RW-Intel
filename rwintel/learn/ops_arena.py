"""The constructed operations arena — a board on which the where-to-send-a-squad choice is measured without playing matches.

The operational choice (which squad, which region, which task) is worth about four hundredths of a match's military edge, and a full match buries that under the economy race and the count of AI players (docs/record/04-operations.md, where a difficulty −2 match does not even resolve in fifteen minutes and the choice sits under ±0.11–0.24 of economy and AI-count scatter). This is the operational analogue of the engagement arena in `arena.py`: the economy is removed by construction — nothing here builds a unit — both sides are handed mirror-equal forces, and the only lever left is where the squads go. Each contested region is a zero-sum contest scored antisymmetrically from a fixed-radius health-weighted catchment, so a run of the handwritten chain against itself must average nought exactly as a fight must.

The whole design is arranged to restore exchange symmetry, which is what makes the self-play mean zero rather than merely making the two side scores exact negatives within an episode. The layout is a single reflection about one chosen centre; the garrisons are mirror-paired by ownership so each side attacks one member of every pair and defends the other and total on-board strength is equal; the score is read off unit positions in a disc rather than off the game's region-block table, so the free base and the last-unit-death flip cannot enter it; and the priorities are one board statement identical to both sides. The trust gate is therefore the statistical self-play mean over many paired episodes, not the tautological within-episode sign check — see `ops_run.py`.

This module builds the board and runs both command chains over a bounded horizon. Driving a live game is deferred: what is exercised without a game is the arithmetic that turns a finished board into a side score and the geometry that lays the board out symmetrically, which is where every exchange-symmetry break would show up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..wire import (
    Action,
    BLOCK_REGIONS,
    Contract,
    Observation,
    SquadAssignment,
    encode_action,
)
from ..control.policy.contracts import Doctrine, OperationsOrders, Posture, SquadRecord
from ..control.policy.operations import Operations
from ..control.policy.tactics import Tactics
from ..control.policy.view import WorldView, build as build_view, rehome
from .arena import Arena, MAX_UNITS, MINIMUM_FORCE, OURS, SETTLE_MS, SPAWN_WAIT_MS, THEIRS

#: How long an episode runs the two chains before it is scored, in game milliseconds. Long enough for a staged squad to march to a contest and fight the garrison, short enough that survivors do not wander into a catchment they were not sent to. Measured: at 120 s the staged squads have not resolved the contests — they survive but are still short of the garrisons — and the score is decided by the mirror garrisons alone, which cancel, giving a trivial self-play zero. At 300 s the squads reach and contest, the domination shares spread, and the choice moves the score. Swept as arms and picked by the resolution gate.
HORIZON_MS = 300000

#: Assorted-doctrine squads staged per side.
OUR_SQUADS = 4

#: Contested offset pairs. Each pair is two congruent points reflected about the centre; both members are scored, so this is half the number of contested regions.
CONTEST_PAIRS = 2

#: Credits a staged squad is drawn out of, uniformly. Small and large deployments teach different things.
SQUAD_VALUE = (1500.0, 4500.0)

#: A region defender's worth as a share of a staged squad's value range, and the scale that share is taken against. One value is drawn per contest pair and placed on both members by ownership, so total garrison strength is equal between the sides.
#:
#: This is the constant that decides whether taking ground is worth doing. A garrison too strong for the squads a side can bring makes the assault unprofitable and the best play is to hold what one already owns, which is a coherent objective but not the one an operational layer has to be good at in a match, where ground must be taken. The scale is a construction argument (`--garrison`) so that the question can be settled by a sweep rather than by the opening value, exactly as the horizon and the catchment radius were.
GARRISON_VALUE = (0.4, 1.2)
GARRISON_SCALE = 3000.0

#: Radius of the disc a contest is scored over, in world units. Sized to the engagement standoff band, because that is where an assaulting squad halts against the garrison: measured, the nearest surviving squad member stopped about 490 units from its contest on Lake and about 280 on the more compact Hills, so a 250-unit disc saw only the garrison and the choice never registered. At 400 the assaulting squads enter the disc, the shares spread off the garrison's nought-or-one, and the choice moves the score. The diameter must stay below the least separation of two contest points so the discs do not overlap, which is what `_draw_pairs` enforces — and at 400 that separation is hard enough to place on a compact map that a fair share of boards refuse, which is a tuning cost of the standoff-sized disc, not a bias (a refused board is never scored, and a board refused for one arm is refused for every arm of the run, since the draw is the seed's). Measured on Hills at seed 70001 with the current draw floor and the clearance from what was standing: 13 boards of 64 refused, the same 13 for each of four arms.
CATCHMENT_RADIUS = 400.0

#: How far a contest point sits from the centre, drawn uniformly. Above the merge distance so a pair's two points fall on distinct regions. The floor a draw actually uses is this or the catchment radius, whichever is larger, because a pair's own two points are twice the offset apart and have to clear the same catchment diameter that two different pairs are held to — a floor below the radius let the one pair every board carries overlap itself, which no later test looked for.
CONTEST_MIN = 350.0
CONTEST_MAX = 700.0

#: How far each staged squad's spawn point is offset from its side's one staging point, so several squads on a side do not all land on the same spot. Reflected exactly for the mirror side, so the two boards stay congruent.
SQUAD_STAGGER = 120.0

#: How far from a staging point a freshly spawned unit is taken to belong to that side's squads. Comfortably beyond the squad stagger and internal scatter, and comfortably inside the march to a contest, so a garrison spawned at a contest point is never swept into a staging squad.
STAGING_REACH = 800.0

#: How many offsets are tried before an episode gives up on placing its contest pairs. A pair is rejected when its two points share a region, collide with a region already taken, sit within a catchment diameter of a point already placed, or reach anything that was already standing when the board was laid out.
MAX_PAIR_ATTEMPTS = 400

#: What point a squad's terminal is read from, mixing the neutral half at nought with the opening ownership of the disc it was sent to at one. At nought the squad is paid the region's absolute domination, and that reading is why a layer trained on this arena learnt to attack nothing: measured on Hills at the default draw, a garrison holds its own disc through the horizon about 88 times in 100 whatever is sent there, while even four squads massed on one enemy disc take it only about 40 times in 100, so absolute domination pays a defender about +0.38 of a priority and an assailant about -0.10, and the best errand a squad can be given is one it was going to be paid for anyway. At one the terminal is what the errand changed on the ground it was sent to, which is the quantity the side score is a priority-weighted mean of, and the same two rates then pay an assault about +0.40 and a redundant defence about nothing.
#:
#: Unlike the engagement arena's strength slope this is NOT policy-invariant, and it is not a control variate: the squad chooses which region it is measured against and therefore chooses which opening it is read from. That is the point rather than a flaw — an errand is worth what it changed, and what it changed cannot be read without knowing where the ground started. It stays antisymmetric at every value, because the two sides' opening shares of one disc sum to one exactly as their final shares do, so the self-play zero remains a statement about the board and says nothing about this.
OPENING_BASELINE = 1.0

#: How a squad's terminal is read off the scored board. `region` pays the whole domination of the region the squad's contract named, which several squads on one region then each take in full; `marginal` pays only the part of it that squad's own surviving units account for. See `OpsArena._finish_side` for what each teaches and what each costs.
CREDITS = ("region", "marginal")

#: Which of them a run uses unless it says otherwise. The region reading is the one every measurement so far was taken under, so it stays the default until the marginal one has been measured against it on the same boards.
CREDIT = "region"

#: What a run calls the tactical layer that fought beneath both sides when that layer was the handwritten `Tactics` ladder — which is what the arena builds for itself when nothing else is handed to it, and what every measurement taken on this arena so far was made under. A run that froze trained tactical parameters under the arena instead names them by their content, so the two can never be mistaken for each other afterwards.
SCRIPT_TACTICS = "script"

#: The doctrines a staged squad may be drawn from. Engineers are excluded because the economy drives them and a contract would land on top of a placement; garrisons are the defenders, drawn separately and never staged as a taskable squad.
_DOCTRINES = (Doctrine.VANGUARD, Doctrine.GARRISON, Doctrine.RAID)


def _tally(force) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for kind in force:
        counts[kind.index] = counts.get(kind.index, 0) + 1
    return counts


@dataclass
class _Pair:
    """One mirror pair of contest points about the centre. The attack member is where this process's squads press and the enemy garrison stands; the defend member is its exact reflection, where the enemy presses and this process's own garrison stands. The two are congruent by construction and resolve to distinct region ids."""

    attack_region: int
    attack_point: Tuple[float, float]
    defend_region: int
    defend_point: Tuple[float, float]


@dataclass
class _Contest:
    """One scored point: the region it names and the world position its catchment is centred on."""

    region_id: int
    point: Tuple[float, float]


@dataclass
class _Garrison:
    """A pre-placed defender, kept apart from the taskable squads so it is never handed to a command layer. GARRISON doctrine has real tasks, so a garrison passed to `Operations` would be tasked and finished, giving one side more squads than the other."""

    side: int
    region_id: int
    point: Tuple[float, float]
    value: float
    tally: Dict[int, int]


@dataclass
class OpsStatistics:
    """What one arena episode scored, in the shape the episode record expects so a run is journalled like any other. One episode is one deployment scored once, so there is one side score."""

    #: True once the horizon was reached and the board scored. An episode cut off before its horizon carries no score, and a run must skip it rather than pool a nought that was never measured.
    scored: bool = False
    #: The priority-weighted mean domination for this process's side, on [−0.5, +0.5]. Exactly the negative of the other side's every episode, so a run of the script chain against itself must pool this to nought.
    side_score: float = 0.0
    #: Contested regions and this side's total garrison worth, kept so a run can be read for what the draw put on the board.
    contests: int = 0
    garrison_value: float = 0.0
    #: Each contested region's final our-share, for a log a person reads.
    shares: Dict[int, float] = field(default_factory=dict)
    #: What each contested region was at the start: which side's garrison stood there (whole where ours did, none where the enemy's did) and what the draw said the region was worth. Journalled beside the final shares because a share on its own cannot say whether a region was one this side had to take or one it had to hold, and the two are the whole question when an arm's advantage might be that it declined to attack at all. A run that only writes the outcome cannot be asked afterwards which discs were which.
    held: Dict[int, float] = field(default_factory=dict)
    priorities: Dict[int, float] = field(default_factory=dict)
    #: Whether the episode was refused because the room exposed no baseless sparring slot to own the enemy side.
    refused: bool = False
    #: The seed this episode's board was drawn from, which is the board's name. Written down so that a later comparison can say which episodes were played on one construction instead of deriving it from the instance and the episode number and the arm count — a derivation that is right until a run is arranged differently and then silently pairs the wrong episodes.
    board: int = 0
    #: How the board was drawn: the horizon in game milliseconds, the catchment radius, the squads staged a side, the contest pairs asked for and the credits a defender was drawn out of. Journalled with the episode because none of these reach the episode settings, and two runs drawn under different ones are two different instruments: a later comparison that pairs them board by board would be reading the change in the instrument as a difference between the arms. Kept here so that comparison can refuse rather than have to be trusted not to.
    horizon_ms: int = 0
    radius: float = 0.0
    squads: int = 0
    pairs: int = 0
    garrison: float = 0.0
    #: Which tactical layer did the fighting beneath both sides: the handwritten ladder, or trained parameters frozen under the arena and named by their content. It belongs with the draw settings above and meets their test word for word — it never reaches the episode settings, and two runs made under different ones are two different arenas. It is a stronger case than the garrison scale rather than a weaker one, because the fighting under an operational choice is the whole of what turns a deployment into a share of a disc: a different fighter moves the disc tallies, the rate at which a garrison holds its own ground through the horizon, the reach a squad ends at, and therefore the horizon and the catchment radius that were both tuned to where an assaulting squad halts.
    tactics: str = SCRIPT_TACTICS
    #: Diagnostics that say whether the staged squads — the thing whose deployment the arena exists to measure — actually reached and contested the catchments, or whether the score was decided by the pre-placed garrisons alone. If the squads never register in a catchment the self-play zero is trivially met by the mirror garrisons and the arena resolves nothing.
    #:
    #: Recorded for both sides and not only this one, because the mirror is a reflection of the board and not of the ground: this side stages from a site the map was searched for, the other from that site's reflection, which is wherever the reflection lands, with a march the reflection cannot make congruent. The arena's answer to that has always been the script arm's self-play zero, and that zero bounds the asymmetry only under the script — a fighter strong enough to exploit a shorter march would convert it into a score that no arm comparison could tell from an operational difference. The two sides' figures side by side are what says whether both deployments reached their contests alike, which is the evidence that was missing when a frozen fighter first made the two sides read a board differently.
    our_alive: int = 0
    our_in_catchment: int = 0
    our_reach: float = 0.0
    their_alive: int = 0
    their_in_catchment: int = 0
    their_reach: float = 0.0
    #: How many of this side's horizon payments actually landed on a decision, read back off the layer's own count of the errands it closed exactly as the engagement arena reads its own, rather than off the number of times the horizon offered one — which is one per staged squad by construction and says nothing. A payment lands only where the layer had a decision of that squad's still waiting or a step of it still in the buffer, so the figure is at most the staged squad count and equals it whenever every squad had a decision recorded at all. It is nought for a run that keeps no trajectories, which is the truth for it: nothing was recorded for a payment to reach.
    #:
    #: This used to be far below the staged count for a reason that has been removed. A squad's trajectory was cut every time a fresh contract replaced its errand, and a cut trajectory cannot take a terminal, so a layer that re-drew its region every period landed almost none of them. Now that every operational period is paid the movement of the squad's own scored figure there is no errand boundary left to cut at, the trajectory runs the whole episode, and the horizon is simply the last of a series of payments.
    terminals: int = 0
    #: `periods` is the operational decisions this side's squads were given over the episode, one per squad per operational frame it held a contract; `errands` is how many distinct contracts those decisions were divided into. Their ratio is the length of an errand in decisions, which is a description of how decisive an arm is — the handwritten ladder holds a squad on the errand it is running while a learnt layer re-draws every period. It is no longer a bound on how far the arena's payment reaches: every period is paid its own movement now, so the whole of the decision mass is reached whatever the ratio comes to. Written down per episode and per arm because the figure differs by arm and cannot be quoted once for the arena.
    periods: int = 0
    errands: int = 0

    def as_dict(self) -> dict:
        return {"scored": self.scored, "side_score": round(self.side_score, 6),
                "contests": self.contests, "garrison_value": round(self.garrison_value, 1),
                "shares": {int(r): round(s, 4) for r, s in self.shares.items()},
                "held": {int(r): round(s, 4) for r, s in self.held.items()},
                "priorities": {int(r): round(w, 4) for r, w in self.priorities.items()},
                "refused": self.refused, "our_alive": self.our_alive,
                "our_in_catchment": self.our_in_catchment, "our_reach": round(self.our_reach, 1),
                "their_alive": self.their_alive, "their_in_catchment": self.their_in_catchment,
                "their_reach": round(self.their_reach, 1),
                "board": self.board, "horizon_ms": self.horizon_ms, "radius": round(self.radius, 1),
                "squads": self.squads, "pairs": self.pairs, "garrison": round(self.garrison, 1),
                "tactics": self.tactics,
                "terminals": self.terminals, "periods": self.periods, "errands": self.errands}


class OpsArena(Arena):
    """Runs one region-domination episode: deploy a mirror board about one centre, run both command chains over a bounded horizon with the economy frozen, score region domination antisymmetrically off a health-weighted catchment, and pay each squad what its errand changed on the region it was sent to.

    That payment is made every operational period rather than once at the horizon. The arena reads its own discs each period and hands each side's layer the figure standing for each of its squads; the layer is paid the movement of that figure, and the horizon hands over the same figure once more as the last of the series. The payments are differences of one quantity and so telescope to the horizon's reading, which is the terminal this arena always paid — the objective is unchanged and only its density is. Paid once, the terminal reached the last errand of a squad and no other, and a layer that re-draws its region every period left almost every decision it took anchored by nothing.

    Subclasses the engagement arena so every geometry and spawn helper — `_sites`, `_site`, `_rows`, `_interleave`, `_commissioned`, `_record`, `_health_worth`, the catalogue and the seeded random — is inherited unchanged and the two arenas cannot drift in how they place or read a board.
    """

    def __init__(self, session, operations=None, opponent=None, tactics=None,
                 tactics_name: str = SCRIPT_TACTICS, seed: int = 0,
                 horizon_ms: int = HORIZON_MS, our_squads: int = OUR_SQUADS,
                 catchment_radius: float = CATCHMENT_RADIUS, contest_pairs: int = CONTEST_PAIRS,
                 opening_baseline: float = OPENING_BASELINE, credit: str = CREDIT,
                 garrison_scale: float = GARRISON_SCALE) -> None:
        super().__init__(session, seed=seed)  # inherits catalogue, random, _sites and every spawn helper
        # The layer under study on this side (a learnt operational layer, or the script for the baseline) and what it is measured against on the other (the script for a duel, its own policy for self-play). Built here rather than handed in already made, for the same reason the engagement arena builds its layers here: both sides must read the same type catalogue as the arena that spawns their units, or a unit would be sorted into a different role on each side. The tactical layer below both actually moves the units and is frozen.
        #
        # One tactical factory serves BOTH sides, which is deliberate and not a shortcut. A trained tactical layer frozen under the arena therefore fights for the enemy exactly as it fights for us, which is the least a mirror can require: were only one side to fight with it the two sides would plainly stop being exchangeable, the script arm's pooled self-play mean would no longer have to be nought, and the arena would lose the only instrument that says whether the board leans. It is also what the learning order means by an operational layer trained against a frozen tactical layer — the whole environment's fighting is that layer, not one side's.
        #
        # It is the least a mirror requires, and what it is enough for differs between the two cuts. The board is a point reflection about a centre, but the other side's view is built by turning the ownership flags over and not by reflecting the coordinates, so the two sides are handed the same function applied to inputs that are congruent in what they mean and not in where they are. A handwritten ladder reading distances and strengths comes out the same either way. So now does a learnt TACTICAL layer, because no tactical feature is measured against the map's axes any more — the direction a squad's errand points is measured from what is shooting at it, and a half turn leaves that alone — and the encoding suite pins it by reading one mirrored board from both sides. The OPERATIONAL vector is not congruent yet: its region and squad rows are laid out by the map's own numbering and by the global squad slot, so congruent ground sits at different offsets for the two sides, and its contact record cannot be turned over at all because the wire's region row carries no counterpart to say when we ourselves were last seen. A learnt operational layer is therefore what still reads a frame here.
        #
        # A frozen tactical layer makes the run a different instrument in the strict sense whichever way that falls: the script arm's self-play zero has to be measured again under it before any arm measured beside it is believed, exactly as it had to be measured for the map and the catchment. That holds all the more since the tactical features changed — parameters fitted before the change read two slots whose meaning has moved, so they have to be refitted and everything measured under them measured again.
        #
        # Two layer objects rather than one, off whatever single network the factory closes over. A tactical layer keeps per-side state — what each of its squads has destroyed, and the board and moment it last saw — and the two sides are handed different boards, this side's view and the other side's inverted and rehomed one, so one shared object would fold the two sides' bookkeeping together and the mirror would stop being a mirror.
        self.our_ops = operations(session, self.catalogue) if operations else Operations(session, self.catalogue)
        self.their_ops = opponent(session, self.catalogue) if opponent else Operations(session, self.catalogue)
        self.our_tac = tactics(session, self.catalogue) if tactics else Tactics(session, self.catalogue)
        self.their_tac = tactics(session, self.catalogue) if tactics else Tactics(session, self.catalogue)
        self.horizon_ms = horizon_ms
        self.our_n = our_squads
        self.radius = catchment_radius
        self.contest_pairs = contest_pairs
        self.opening_baseline = opening_baseline
        if credit not in CREDITS:
            raise ValueError("the terminal a squad is paid is either %s" % " or ".join(CREDITS))
        self.credit = credit
        self.garrison_scale = garrison_scale

        self.phase = "opening"
        # Two taskable dicts and a separate garrison list. The garrisons are never in either taskable dict and are never handed to a command layer.
        self.squads: Dict[int, SquadRecord] = {}      # this side, slots 0 .. N-1
        self.enemy: Dict[int, SquadRecord] = {}       # the other side, slots N .. 2N-1
        self.garrisons: List[_Garrison] = []
        self.pairs: List[_Pair] = []
        self.contests: List[_Contest] = []
        self.mirror_of: Dict[int, int] = {}
        self.orders: Optional[OperationsOrders] = None
        self.priorities: Dict[int, float] = {}
        self.garrison_share: Dict[int, float] = {}    # region -> this side's share of the initial garrison worth, fixed at the draw
        self.our_home_id: Optional[int] = None
        self.their_home_id: Optional[int] = None
        self.our_reports: List = []
        self.their_reports: List = []
        self.side_score = 0.0
        self.refused = False
        self.until_ms = 0
        self.centre: Tuple[float, float] = (0.0, 0.0)
        self._our_pt: Optional[Tuple[float, float]] = None
        self._their_pt: Optional[Tuple[float, float]] = None
        self._wanted: Dict[int, Dict[int, int]] = {}
        self._period = 0
        #: When each of this side's squads was last handed a contract, so that a fresh one can be told from the same one standing. A contract carries the moment it was issued and the operational layer only writes a new one when something about it changed, so a change in this figure is exactly one errand ending and another beginning.
        self._issued: Dict[int, int] = {}
        self._sandbox_sent = False
        # The draw's settings are written into the statistics at construction rather than at scoring, so that an episode which never reaches its horizon still says under what instrument it was run. The name of the tactical layer beneath both sides is one of them: whoever builds the arena is the only one who knows which parameters the factory closes over, and the arena cannot read it back off a layer afterwards.
        self.statistics = OpsStatistics(board=seed, horizon_ms=horizon_ms, radius=catchment_radius,
                                        squads=our_squads, pairs=contest_pairs, garrison=garrison_scale,
                                        tactics=tactics_name)

    # ---- the one entry point (mirrors Arena.decide) ------------------------------------

    def decide(self, observation: Observation) -> Optional[bytes]:
        if not self._sandbox_sent:
            # Sandbox first and on its own, because until it is set the units of the other side are not this process's to command and every order to them would be dropped.
            self.session.scenario([], sandbox=True)
            self._sandbox_sent = True
            self.sites = self._sites()
            # Left empty so the settle below has a change to detect, exactly as the engagement arena leaves it.
            self.known = set()
            self.until_ms = observation.game_time_ms + SETTLE_MS

        view = build_view(observation, self.catalogue, None, self.last_regions)
        self.last_regions = view.regions
        action = Action()
        now = observation.game_time_ms

        self._fold_all(observation)
        if self.phase == "opening":
            self._settle_open(observation, action, now)
        elif self.phase == "spawning":
            self._commission(observation, action, now)
        elif self.phase == "running":
            self._run(observation, view, action, now)

        if not (action.squads or action.contracts or action.deviations):
            return None
        return encode_action(action)

    # ---- opening / deployment ----------------------------------------------------------

    def _settle_open(self, observation: Observation, action: Action, now: int) -> None:
        """Holds the deployment until the opening board has stopped changing, then lays it out. Copied from the engagement arena's settle: the free headquarters and builder a spawn-point player starts with arrive over the first steps, and folding them into the standing snapshot first is what keeps this side's own base out of a squad."""
        current = {unit.id for unit in observation.unit_states}
        settled = bool(current) and current == self.known
        self.known = current
        if settled or now >= self.until_ms:
            self._deploy(observation, action, now)

    def _deploy(self, observation: Observation, action: Action, now: int) -> None:
        """Lays out a board symmetric by reflection about one centre and submits it in a single order."""
        if self.enemy_slot is None and getattr(self.session, "sparring_slot", -1) < 0:
            # No baseless sparring slot to own the enemy squads and garrisons: the enemy cannot be placed, or fights on its own economy and poisons every number. The room settles this as it fills its free slots, so it is read here rather than derived. `Arena._their_slot` falls back to a live slot when none is reported, which is right for a match but wrong here, so the raw report is what the refusal is gated on. Refuse the episode; an override slot handed in for a constructed test stands in for the report.
            self.refused = True
            self.statistics.refused = True
            self.phase = "done"
            return

        centre = self._board_centre()
        our_stage = self._site(observation)
        if our_stage is None:
            return
        their_stage = self._mirror(our_stage, centre)
        self.centre, self._our_pt, self._their_pt = centre, our_stage, their_stage

        # What is already standing when the board is laid out is the free base every player with a starting position is given — the game has no setting that withholds it — and the arena is built on top of it rather than instead of it. Its worth is thousands of credits, it is not hostile, and the sparring side has no counterpart for it, so a scored disc that reached it would hand this side that worth on every board it happened on and on no board the mirror could answer with. The pairs are drawn clear of it instead.
        self.pairs = self._draw_pairs(centre, [(unit.x, unit.y) for unit in observation.unit_states])
        if len(self.pairs) < self.contest_pairs:
            # Not enough distinct, well-separated contests could be placed on this board. Refuse rather than run a lopsided one.
            self.refused = True
            self.statistics.refused = True
            self.phase = "done"
            return
        self._build_contests()

        our_rows, their_rows = self._spawn_rows(observation, centre, our_stage, their_stage)
        # One order, both sides interleaved a unit at a time, so neither side is more completely on the board when the spawn wait is called.
        self.session.scenario(self._interleave(our_rows, their_rows))
        self.known = {unit.id for unit in observation.unit_states}

        self.our_home_id = self._nearest_region_id(our_stage)
        self.their_home_id = self._nearest_region_id(their_stage)
        self._synthesize_orders()
        self.statistics.contests = len(self.contests)
        self.statistics.garrison_value = sum(g.value for g in self.garrisons if g.side == OURS)

        self.phase = "spawning"
        self.until_ms = now + SPAWN_WAIT_MS

    def _spawn_rows(self, observation: Observation, centre, our_stage, their_stage):
        """Every spawn row for both sides. This side's squads and garrisons are built normally; the other side's are the exact reflection of them about the centre, so the whole board is congruent unit for unit and the two sides field equal strength."""
        our_slot = self._our_slot(observation)
        their_slot = self._their_slot(observation)
        our_rows: List[List[float]] = []
        their_rows: List[List[float]] = []

        for slot in range(self.our_n):
            doctrine = self.random.choice(_DOCTRINES)
            force = self._doctrine_force(doctrine, self.random.uniform(*SQUAD_VALUE))
            if not force:
                # A doctrine whose pool is empty on this catalogue falls back to the vanguard's, which is the ground-armour pool every skirmish map has something in.
                force = self._doctrine_force(Doctrine.VANGUARD, self.random.uniform(*SQUAD_VALUE))
            tally = _tally(force)
            # Both sides commission against the same tally: the mirror squad is the same force reflected, so what is wanted of it is identical.
            self._wanted[slot] = tally
            self._wanted[self.our_n + slot] = tally
            rows = self._rows(force, our_slot, self._scatter(our_stage, slot))
            our_rows += rows
            their_rows += [self._reflect_row(row, centre, their_slot) for row in rows]

        self.garrison_share = {}
        # The last figure each squad stood at while it still had units, kept so that a squad wiped out mid-episode stops moving and is not paid for what its allies go on doing on the disc it was sent to. Cleared with the board, since squad numbers are reused from one episode to the next.
        self._frozen = {}
        for pair in self.pairs:
            garrison = self._doctrine_force(Doctrine.GARRISON,
                                            self.random.uniform(*GARRISON_VALUE) * self.garrison_scale)
            if not garrison:
                garrison = self._doctrine_force(Doctrine.GARRISON, self.garrison_scale)
            value = sum(kind.price for kind in garrison)
            # This side's own garrison defends the defend member; its exact reflection is the enemy garrison on the attack member. One draw, placed once by ownership, so the sides' garrison worth is equal.
            defend_rows = self._rows(garrison, our_slot, pair.defend_point)
            our_rows += defend_rows
            their_rows += [self._reflect_row(row, centre, their_slot) for row in defend_rows]
            self.garrisons.append(_Garrison(side=OURS, region_id=pair.defend_region,
                                            point=pair.defend_point, value=value, tally=_tally(garrison)))
            self.garrisons.append(_Garrison(side=THEIRS, region_id=pair.attack_region,
                                            point=pair.attack_point, value=value, tally=_tally(garrison)))
            # This side's share of the initial garrison worth in each region: whole where it owns the garrison, none where the enemy does. Symmetric about a half under the mirror map, which is what keeps the control variate antisymmetric.
            self.garrison_share[pair.defend_region] = 1.0
            self.garrison_share[pair.attack_region] = 0.0

        return our_rows, their_rows

    def _commission(self, observation: Observation, action: Action, now: int) -> None:
        """Takes the units that appeared since the deployment and makes the taskable squads of them, one per staged slot per side, against the order that was placed. Units near the two staging points are the squads; garrison units, which spawn out at the contest points, are excluded by position and never enter a taskable dict."""
        fresh = [unit for unit in observation.unit_states if unit.id not in self.known]
        our_assign = self._assign(fresh, hostile=False, point=self._our_pt,
                                  slots=range(self.our_n))
        their_assign = self._assign(fresh, hostile=True, point=self._their_pt,
                                    slots=range(self.our_n, 2 * self.our_n))
        for slot, units in our_assign.items():
            self.squads[slot] = self._record(slot, units, observation)
            action.squads.append(SquadAssignment(squad=slot, units=units))
        for slot, units in their_assign.items():
            self.enemy[slot] = self._record(slot, units, observation)
            action.squads.append(SquadAssignment(squad=slot, units=units,
                                                 owner=self._their_slot(observation)))

        # Wait for the whole of both sides unless the spawn window has run out, in which case whatever arrived is what runs.
        whole = len(self.squads) >= self.our_n and len(self.enemy) >= self.our_n
        if not whole and now < self.until_ms:
            return
        if not self.squads or not self.enemy:
            # Nothing usable arrived on one side; there is no contest to run.
            self.refused = True
            self.statistics.refused = True
            self.phase = "done"
            return
        self.phase = "running"
        self.until_ms = now + self.horizon_ms

    def _assign(self, fresh, hostile: bool, point, slots) -> Dict[int, List[int]]:
        """The units of one side's staged squads, taken from what newly appeared near that side's staging point, of the right hostility, against each slot's order in turn. A consumable pool rather than the engagement arena's `_commissioned` per slot, because several squads of one side spawn at one point and a unit taken into one must not be taken into the next; the position filter is what keeps a garrison spawned out at a contest point from being swept into a staging squad."""
        if point is None:
            return {}
        reach2 = STAGING_REACH * STAGING_REACH
        pool = [unit for unit in fresh
                if bool(unit.hostile) == hostile
                and (unit.x - point[0]) ** 2 + (unit.y - point[1]) ** 2 <= reach2]
        assigned: Dict[int, List[int]] = {}
        for slot in slots:
            wanted = self._wanted.get(slot)
            if not wanted:
                continue
            left = dict(wanted)
            taken: List[int] = []
            for unit in list(pool):
                remaining = left.get(unit.type_index, 0)
                if remaining <= 0:
                    continue
                left[unit.type_index] = remaining - 1
                taken.append(unit.id)
                pool.remove(unit)
            if taken:
                assigned[slot] = taken
        return assigned

    # ---- orders ------------------------------------------------------------------------

    def _synthesize_orders(self) -> None:
        """Draws the priorities. No strategic layer runs, so the weights are drawn from the seed one per unordered mirror pair and set on both of the pair's regions at once, which is what keeps the board even: a weight drawn region by region would re-randomise the second member and make it an asymmetric board statement. The priority dict is asserted invariant under the mirror map before the run."""
        priorities: Dict[int, float] = {}
        for pair in self.pairs:
            weight = self.random.uniform(0.3, 1.0)
            priorities[pair.attack_region] = weight
            priorities[pair.defend_region] = weight
        for region, weight in priorities.items():
            mirror = self.mirror_of.get(region)
            assert mirror is not None and priorities.get(mirror) == weight, (
                "the synthesised priorities are not invariant under the mirror map")
        self.priorities = priorities
        self.orders = OperationsOrders(posture=Posture.ARM, priorities=priorities,
                                       offensive=self.random.random() < 0.5,
                                       loss_allowance=self.random.uniform(*SQUAD_VALUE))

    # ---- running both chains over the horizon ------------------------------------------

    def _run(self, observation: Observation, view: WorldView, action: Action, now: int) -> None:
        if now >= self.until_ms:
            self._score(observation)
            self.phase = "done"
            return

        operational = bool(observation.blocks & BLOCK_REGIONS)  # region force totals ride these frames
        # Each side reads the board from where it stages out of: its regions measured again from its own staging region, since the wire measures every distance from this process's base and the inverted view would otherwise read our marches as its own, and its bearing anchored at the staging POINT rather than at that region's centre. The two staging points are drawn as exact reflections while the region nearest each of them is whatever the map put there, so the regions need not be a pair at all — and on any board whose region table is not itself symmetric, which is every real map, anchoring the layer beneath at a region's centre would make a squad and its mirror read as two different fights. What this cannot repair is physical march distance or region-geometry congruence; that residual is what the self-play mean is watched for.
        our_view = rehome(view, self.our_home_id, self._our_pt)
        their_view = rehome(
            build_view(observation, self.catalogue, None, self.last_regions, invert=True),
            self.their_home_id, self._their_pt)

        # What every scored disc reads on the board that has just arrived, taken once for the period and before either side decides, so the two sides are paid off one reading of one board and their figures stay exact negatives of each other. Outside the loop below rather than inside it, because the leader alternation would otherwise hand the two sides boards a decision apart.
        shares = self._shares(observation.unit_states) if operational and self.orders is not None else None

        sides = [(self.our_ops, self.our_tac, self.squads, our_view, OURS),
                 (self.their_ops, self.their_tac, self.enemy, their_view, THEIRS)]
        # Alternate the leader on a period-count parity, so whatever a period's leader gains falls on both sides equally over the horizon. Keyed on a monotone period counter like the engagement arena's, not on game time, which would not alternate evenly across irregular operational frames.
        if self._period % 2 == 1:
            sides.reverse()

        for ops, tac, squads, board, side in sides:
            slist = list(squads.values())
            if operational and self.orders is not None:
                reports = self.our_reports if side == OURS else self.their_reports
                # Where the arena's own reading of its scored discs reaches the layer, and it has to arrive before the decision is taken: settling is what pays the decision the last period left waiting, the decision is settled at the top of `decide`, and this is the board it is to be paid from. Discovered on the layer exactly as `finish` is, and for the same reason — what a squad's errand is worth is a statement about ground only whoever runs the contest can read, while the layer only sees periods. A script layer offers no `standing` and takes none, exactly as it offers no `finish`.
                standing = getattr(ops, "standing", None)
                if standing is not None:
                    standing(self._standings(squads, shares, +1.0 if side == OURS else -1.0,
                                             observation.unit_states))
                # Only the taskable squads are handed to the operational layer; the garrisons are not in `squads`, so a garrison is never tasked and never finished.
                contracts, _ = ops.decide(board, self.orders, slist, reports, now)
                for contract in contracts:
                    action.contracts.append(Contract(
                        squad=contract.squad, task=contract.task, stance=contract.stance,
                        target_region=contract.target_region, cost_budget=contract.cost_budget,
                        deadline_ms=contract.deadline_ms, issued_at_ms=contract.issued_at_ms,
                        override=True))
            deviations, out = tac.decide(board, slist, now)
            action.deviations.extend(deviations)
            if side == OURS:
                self.our_reports = out
            else:
                self.their_reports = out
        if operational and self.orders is not None:
            # After both sides have decided, so that a period is counted against the contract that period's decision produced and the last decision before the horizon is counted at all.
            self._survey()
        self._period += 1

    def _survey(self) -> None:
        """Counts one operational period for each of this side's squads that holds a contract, and one errand each time a squad is handed a fresh one.

        This is how decisive an arm is: an episode in which four contracts stood from the staging point to the horizon and one in which they were re-drawn every period score alike, report the same shares and the same terminals, and are told apart by nothing else in the record. It is no longer a statement about how far the arena's payment reaches — every period is now paid the movement of the squad's own scored figure, so the payments reach every decision whatever the errands come to — but it remains the one figure that says whether an arm settled on an errand or kept changing its mind, which is a real difference between arms and a real thing to read a run by.

        Read off the contracts the layers wrote onto the squad records rather than off any layer's own bookkeeping, so that it costs the same and means the same for every arm — the handwritten ladder, the pinned deployment, the concentrating arm and a learnt network alike — and so that an arm which keeps no trajectories is still measured. This side only: the statistics belong to the process's own side and the enemy's periods are the mirror's business.
        """
        for squad in self.squads.values():
            contract = squad.contract
            if contract is None:
                continue
            self.statistics.periods += 1
            if self._issued.get(squad.id) != contract.issued_at_ms:
                self._issued[squad.id] = contract.issued_at_ms
                self.statistics.errands += 1

    # ---- scoring and per-decision terminal ---------------------------------------------

    def _score(self, observation: Observation) -> None:
        """At the horizon, read the antisymmetric side score and pay each squad the domination of the region it was sent to."""
        units = observation.unit_states
        self.side_score = self._side_score(units)
        shares = self._shares(units)

        self.statistics.scored = True
        self.statistics.side_score = self.side_score
        self.statistics.shares = dict(shares)
        self.statistics.held = {region: self.garrison_share.get(region, 0.5) for region in shares}
        self.statistics.priorities = {region: self.priorities.get(region, 0.0) for region in shares}

        # Diagnose whether the staged squads reached the catchments at all, or the mirror garrisons decided the score by themselves. Both sides, because the question the mirror leaves open is whether the two sides reached their contests alike: this side stages from a validated site while the other stages from that site's reflection, which is an arbitrary point on ground nothing checked, with a march the reflection cannot make congruent. One side's figures alone cannot say whether that told, and a stronger fighter on both sides is exactly what would convert such a difference into a score.
        (self.statistics.our_alive, self.statistics.our_in_catchment,
         self.statistics.our_reach) = self._reached(self.squads, units)
        (self.statistics.their_alive, self.statistics.their_in_catchment,
         self.statistics.their_reach) = self._reached(self.enemy, units)

        self._finish_side(self.our_ops, self.squads, shares, +1.0, units)
        self._finish_side(self.their_ops, self.enemy, shares, -1.0, units)
        self._tally()

    def _reached(self, squads: Dict[int, SquadRecord], unit_states) -> Tuple[int, int, float]:
        """How much of one side's staged strength is still on the board, how much of it ended inside a contest's catchment, and how close the nearest of it came to any contest.

        A member is any unit still on the board belonging to one of that side's staged squads; the reach is the nearest approach of the whole side, quoted as a distance, and is minus one where the side has nothing left to measure — a figure outside the range of a distance, so a reader cannot mistake it for having arrived. Written for either side from the same rule, because a diagnosis of whether a deployment reached its ground only means anything beside what the other deployment did on ground laid out to be its reflection.
        """
        members = {m for squad in squads.values() for m in squad.members}
        by_id = {unit.id: unit for unit in unit_states}
        alive = [by_id[m] for m in members if m in by_id]
        radius2 = self.radius * self.radius
        in_catchment = 0
        reach = float("inf")
        for unit in alive:
            for contest in self.contests:
                d2 = (unit.x - contest.point[0]) ** 2 + (unit.y - contest.point[1]) ** 2
                reach = min(reach, d2)
                if d2 <= radius2:
                    in_catchment += 1
                    break
        return len(alive), in_catchment, math.sqrt(reach) if reach != float("inf") else -1.0

    def _tally(self) -> None:
        """Writes down how many of this side's horizon payments actually landed on a decision, taken from the layer's own count of the errands it closed exactly as the engagement arena's tally is.

        Recomputed from the layer rather than counted here, and that is the whole point of it. What this arena can count for itself is how many times it offered a payment, which is one per staged squad holding a contract and is therefore a constant that says nothing; only the layer knows whether the payment reached a decision, because only the layer knows whether that squad still had a step to pay it into. A layer that keeps no trajectories — every arm of the measuring runner — lands none of them, and nought is then the honest figure rather than a fault.

        With every period paid there is no longer an errand boundary for a trajectory to be cut at, so a layer that records a decision about a squad at all keeps that squad's trajectory to the horizon and takes the payment. What still lands nothing is a squad no decision was ever recorded about — one the inherited rule passed over every period for being worn below the health it will task at — so the figure is at most the staged squad count rather than always equal to it.

        This side only: the statistics belong to the process's own side, and the enemy's terminals are the mirror's business.
        """
        self.statistics.terminals = sum(int(count) for count in getattr(self.our_ops, "terminals", {}).values())

    def _side_score(self, unit_states) -> float:
        """The reported and validated side score: the priority-weighted mean domination over every contested region.

        For each contest a catchment is the disc of radius `self.radius` about its point; `our_worth` sums `value(type)·clip(health/max_health)` over the non-hostile units in it and `enemy_worth` the same over the hostile ones. The share is `our_worth/(our_worth+enemy_worth)`, a half when the disc is empty, and the domination is the share less a half on [−0.5, +0.5]. The side score divides by the sum of the priorities — never by credits, squad count or region count — so it is a share independent of army size and of how many squads went where. It is exactly antisymmetric: the other side's worths are these with the hostility flipped, so its share is one less this one's and its domination the negative, and the priorities are one board statement identical to both; hence the two sides sum to nought every episode.
        """
        total_weight = sum(self.priorities.values())
        if total_weight <= 0.0:
            return 0.0
        score = 0.0
        for contest in self.contests:
            weight = self.priorities.get(contest.region_id, 0.0)
            if weight == 0.0:
                continue
            our_worth, enemy_worth = self._catchment_worths(unit_states, contest.point)
            total = our_worth + enemy_worth
            share = our_worth / total if total > 0 else 0.5
            score += weight * (share - 0.5)
        return score / total_weight

    def _catchment_worths(self, unit_states, point, without=()) -> Tuple[float, float]:
        """This side's and the other side's health-weighted worth inside one catchment, split by the hostility flag. Read off the unit rows rather than the region block, so the free base, spectators and any stray are excluded by construction and the last-unit-death flip the price block carries cannot enter. Health-weighted like `Arena._health_worth`: a type with no maximum health counts whole, which is what the sparse reading says of it too.

        `without` leaves a set of units out of the count, which is how the marginal credit asks what the catchment would have read had one squad not been standing in it. Nothing about the score reported for the episode uses it; it exists for the terminal one squad is paid.
        """
        radius2 = self.radius * self.radius
        our_worth = 0.0
        enemy_worth = 0.0
        for unit in unit_states:
            if unit.id in without:
                continue
            if (unit.x - point[0]) ** 2 + (unit.y - point[1]) ** 2 > radius2:
                continue
            share = 1.0 if unit.max_health <= 0 else unit.health / unit.max_health
            worth = self.catalogue.value(unit.type_index) * min(1.0, max(0.0, share))
            if unit.hostile:
                enemy_worth += worth
            else:
                our_worth += worth
        return our_worth, enemy_worth

    def _shares(self, unit_states) -> Dict[int, float]:
        """This side's share of every contested catchment on the board just read, keyed by region id. A disc with nothing in it reads a half, as it does everywhere else here.

        Read once for the whole board rather than once per squad, and that is what keeps the arithmetic honest rather than merely quick. Several squads may be sent to one disc and would otherwise each read it separately; and it is the reading BOTH sides are paid off, the other side's share of a disc being one less this one's, so taking it once before either side decides is what makes the two sides' figures exact negatives.
        """
        shares: Dict[int, float] = {}
        for contest in self.contests:
            our_worth, enemy_worth = self._catchment_worths(unit_states, contest.point)
            total = our_worth + enemy_worth
            shares[contest.region_id] = our_worth / total if total > 0 else 0.5
        return shares

    def _contest(self, region_id) -> Optional["_Contest"]:
        """The scored contest a region names, or nothing where the region is not one of them.

        A scan rather than a map cached when the contests are built: the arena is exercised without a game by assembling it field by field and appending contests directly, so a cached map would be a fixture that goes stale silently while every reading it fed went on looking right. There are four contests and only the marginal credit consults this, so the scan costs nothing.
        """
        for contest in self.contests:
            if contest.region_id == region_id:
                return contest
        return None

    def _standing(self, squad: SquadRecord, shares: Dict[int, float], sign: float,
                  unit_states=()) -> float:
        """What one squad's errand has moved on the disc its contract names, as the board just read stands, signed for the side.

        This is the whole of the arena's credit and it is one expression called from two places: the period loop reads it every operational frame and the horizon reads it once more, so that the differences between successive readings telescope to the last reading exactly. Were the horizon to compute its own figure the identity would be an intention that two expressions had to be kept in step; written this way it is a fact about the code.

        Which reading it is — the region's own outcome, or only the part of it this squad's surviving units account for — is the credit the arena was constructed with, and what each teaches is in `_finish_side`. A region the board put no priority on moves no figure at all, so a squad sent to one is paid nought rather than being paid out of some other quantity.
        """
        if not squad.members:
            # A squad with nothing left on the board cannot move the disc it was sent to, so its figure is frozen where its last surviving unit left it and it is paid no further difference. Without this it goes on collecting, period after period, whatever its allies produce on that disc, and the horizon hands it the whole of a domination it took no part in — the free-rider term extended to a squad that no longer exists, which is precisely the misattribution this credit was built to remove. Frozen rather than nought, because a squad that destroyed a garrison and died doing it did move the disc, from the enemy's hands to nobody's, and that movement is in the side score whether or not anything of the squad survived to stand on it. The freeze is read before the contract is, so a dead squad whose layer goes on writing it errands cannot change its figure by naming a different region.
            return self._frozen.get(squad.id, 0.0)
        region = squad.contract.target_region if squad.contract is not None else None
        weight = self.priorities.get(region, 0.0)
        if weight == 0.0:
            self._frozen[squad.id] = 0.0
            return 0.0
        share = shares.get(region, 0.5)
        if self.credit == "marginal":
            contest = self._contest(region)
            if contest is not None:
                # The marginal reading carries its own origin — what the catchment would have read with this squad's units taken out of it — so the opening is not subtracted from it a second time.
                figure = sign * weight * (share - self._share_without(unit_states, contest, squad.members))
                self._frozen[squad.id] = figure
                return figure
        # Where the errand's outcome is read from: the neutral half when the baseline is nought, the disc's own opening ownership when it is one. The baseline is inside the priority weighting rather than beside it, because it is the point one region's outcome is measured from and not a separate term added to it; weighting the outcome and not its origin would leave a standing payment on every disc that scaled with nothing.
        opening = 0.5 + self.opening_baseline * (self.garrison_share.get(region, 0.5) - 0.5)
        figure = sign * weight * (share - opening)
        self._frozen[squad.id] = figure
        return figure

    def _standings(self, squads: Dict[int, SquadRecord], shares: Dict[int, float], sign: float,
                   unit_states=()) -> Dict[int, float]:
        """Every squad of one side's standing on the board just read, by squad id.

        Total over the side's squads rather than only over the ones on scored ground: a squad contracted to a region the board put no priority on is present with a nought. What that spares the layer is a fallback of its own for a squad it finds missing, and a fallback is exactly what must not exist — a trajectory some of whose steps were paid in the arena's disc reading and some in the region block sums to neither quantity.
        """
        return {squad.id: self._standing(squad, shares, sign, unit_states)
                for squad in squads.values()}

    def _finish_side(self, ops, squads: Dict[int, SquadRecord], shares: Dict[int, float], sign: float,
                     unit_states=()) -> None:
        """Pays every squad of one side what its errand moved on the region its final contract named, plus the opposite sign for the other side exactly as the engagement arena pays outcome and −outcome. A script layer keeps no trajectories and offers no `finish`, so this is a no-op for the self-play baseline; a learnt operational layer routes the terminal back to the operational decision that produced it.

        The figure handed over is `_standing` on the horizon board, which is the same expression the period loop has been paying differences of all episode. So this is the last of a series of payments rather than the only one, and the layer subtracts what it has already been paid: the sum over the episode comes to this number and to nothing else. Nothing about the terminal, the side score or the antisymmetry is changed by the periods being paid — the horizon figure is what it always was.

        There are two ways to say what one squad's deployment earned, and which one is in force is a construction argument because they teach different things.

        Both readings are taken from the opening baseline rather than from an absolute domination, which is the correction that made the arena teachable at all. A terminal read from the neutral half pays a squad for how the ground stands rather than for what its errand did to the ground, and on this arena those are almost opposite things: a garrison keeps its own disc through the horizon about 88 times in 100 whether or not a squad is sent to help it, so the neutral reading pays a redundant defence in full and charges an assault for the two chances in three it fails, and the layer trained under it duly learnt to send nobody anywhere. Read from where the disc started, the same errands pay what they moved.

        `region` pays the region's own outcome, so several squads that converged on one region share the identical figure. It is the plainest reading of "you were sent here and here is how here went", and its flaw is that it pays a squad in full for a region its allies had already taken — the free-rider term, which rewards piling on whether or not the pile helped. On this arena that flaw is the right way round: one squad takes a defended disc about a tenth of the time and four massed on it about four tenths, so a pile is what taking ground is made of, and a reading that pays every member of the pile is the one that can teach it.

        `marginal` pays the difference the squad itself made: the region's domination as it stands, less what the same catchment would have read with that squad's surviving units taken out of it. Several squads on one region then divide what they jointly produced rather than each taking all of it, and a squad that added nothing to a region already won is paid nothing for it. It is the difference reward, and the reason it is the more honest signal is that it is the part of the team's score that this decision actually moved. Its known cost is that a squad wiped out at the horizon has nothing left in the catchment and is paid nothing, however much of the enemy it took with it — the counterfactual it can compute is "had these units not been standing here", not "had this squad never been sent".

        The marginal reading is the change the squad's own units made to this side's score, on the scale the terminal is paid at: the other regions' terms are identical with and without it, so the one region's difference is the whole difference, and it is that difference before the side score's final division by the sum of the board's priorities — proportional to the change in the side score, by a factor fixed within an episode, rather than equal to it. That identity is the reason to prefer it — a squad is paid in the very quantity the arena is measured by, and in no part of it that another squad produced. It is not antisymmetric between the sides, and is not meant to be: both sides can truthfully say a contested disc would have been lost without them, so two opposing squads can both be paid well. A credit is not a score. Neither reading touches the side score the episode is measured by, which is what the self-play zero is a statement about.
        """
        finish = getattr(ops, "finish", None)
        if finish is None:
            return
        for squad in squads.values():
            finish(squad, self._standing(squad, shares, sign, unit_states), "horizon")

    def _share_without(self, unit_states, contest: "_Contest", members: Sequence[int]) -> float:
        """What one contest's catchment would have read with a squad's surviving units taken out of it. An empty disc reads a half, as it does everywhere else, so a squad that was the only thing in a catchment is credited with the whole of taking it."""
        our_worth, enemy_worth = self._catchment_worths(unit_states, contest.point, without=set(members))
        total = our_worth + enemy_worth
        return our_worth / total if total > 0 else 0.5

    def close(self) -> None:
        for layer in (self.our_ops, self.their_ops, self.our_tac, self.their_tac):
            if hasattr(layer, "close"):
                layer.close()

    # ---- keeping both sides' squads in step with the board -----------------------------

    def _fold_all(self, observation: Observation) -> None:
        """Recomputes value, position and membership for every squad from the unit rows, so a foreign-owned enemy squad is tracked without trusting a squad block this process may not own. Mirrors `Arena._fold` across both sides; the score reads `unit_states` and never the squad block, so this feeds the command layers, not the score."""
        alive = {unit.id for unit in observation.unit_states}
        by_id = {unit.id: unit for unit in observation.unit_states}
        for squads in (self.squads, self.enemy):
            for squad in squads.values():
                squad.members = [member for member in squad.members if member in alive]
                if squad.members:
                    squad.value = sum(self.catalogue.value(by_id[m].type_index) for m in squad.members)
                    squad.x = sum(by_id[m].x for m in squad.members) / len(squad.members)
                    squad.y = sum(by_id[m].y for m in squad.members) / len(squad.members)

    # ---- geometry ----------------------------------------------------------------------

    def _doctrine_force(self, doctrine: Doctrine, budget: float) -> List:
        """A random force worth about the budget, drawn from the types a squad of this doctrine will take. Generalises `Arena._force`, which is fixed to the ground-armour pool, by filtering on `catalogue.accepts(doctrine, index)` instead, and keeps the same rules: a type is only considered if the whole force could be built from it, and the force is capped at MAX_UNITS."""
        pool = [kind for kind in self.catalogue.types
                if kind.price > 0 and self.catalogue.accepts(doctrine, kind.index)
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

    def _board_centre(self) -> Tuple[float, float]:
        """The one centre the whole layout reflects about: the mean of the places an engagement may be built on."""
        if not self.sites:
            return (0.0, 0.0)
        return (sum(site[0] for site in self.sites) / len(self.sites),
                sum(site[1] for site in self.sites) / len(self.sites))

    @staticmethod
    def _mirror(point, centre) -> Tuple[float, float]:
        return (2 * centre[0] - point[0], 2 * centre[1] - point[1])

    @staticmethod
    def _reflect_row(row, centre, slot: int) -> List[float]:
        """One spawn row reflected about the centre and reassigned to the other side's slot, so the mirror force is congruent to this side's unit for unit."""
        return [row[0], float(slot), 2 * centre[0] - row[2], 2 * centre[1] - row[3], row[4]]

    def _scatter(self, point, slot: int) -> Tuple[float, float]:
        """A staged squad's own spawn point, offset from its side's one staging point so several squads do not land on the same spot. Deterministic per slot, so the reflection of it is the other side's congruent staging."""
        angle = 2 * math.pi * slot / max(1, self.our_n)
        return (point[0] + math.cos(angle) * SQUAD_STAGGER,
                point[1] + math.sin(angle) * SQUAD_STAGGER)

    def _draw_pairs(self, centre, standing: Sequence[Tuple[float, float]] = ()) -> List[_Pair]:
        """The contest pairs, each an offset drawn about the centre placed as the congruent pair (centre+u, centre−u).

        A pair is kept only when its two points map to distinct regions, neither region is one an earlier pair already took, neither point sits within a catchment diameter of a point already placed, the pair's own two points are that far apart as well, and neither point reaches anything that was already standing on the board.

        The last two are the ones worth naming. A pair's two members are the reflection of each other about the centre, so their separation is twice the offset drawn and can be shorter than the separation demanded of two different pairs; tested only against earlier pairs, a pair could overlap itself, and the disjointness the score is read under would fail on the one pair that is always present. And what is already standing is this side's free base, which the mirrored side has no counterpart for: the engagement arena keeps it out of a fight by siting the fight at maximum clearance from whatever is standing, and this is the same guarantee taken the other way round, by refusing a draw that reaches it. A pair is rejected whole, so the mirror map, the priority invariance and the garrison balance are untouched by either test, and a board where the attempts run out is refused rather than scored lopsided.
        """
        pairs: List[_Pair] = []
        used: set = set()
        placed: List[Tuple[float, float]] = []
        span = 2 * self.radius
        attempts = 0
        while len(pairs) < self.contest_pairs and attempts < MAX_PAIR_ATTEMPTS:
            attempts += 1
            angle = self.random.uniform(0, 2 * math.pi)
            # Drawn from the radius up rather than from the bare floor, so a pair's own two points — which sit twice the offset apart — always clear the same separation demanded between two different pairs. Rejecting short draws afterwards would do the same thing and throw away one draw in seven on a board where placing pairs at all is already the scarce thing.
            magnitude = self.random.uniform(max(CONTEST_MIN, self.radius), CONTEST_MAX)
            offset = (math.cos(angle) * magnitude, math.sin(angle) * magnitude)
            attack = (centre[0] + offset[0], centre[1] + offset[1])
            defend = (centre[0] - offset[0], centre[1] - offset[1])
            attack_region = self._nearest_region_id(attack)
            defend_region = self._nearest_region_id(defend)
            if attack_region is None or defend_region is None or attack_region == defend_region:
                continue
            if attack_region in used or defend_region in used:
                continue
            if any(math.hypot(attack[0] - x, attack[1] - y) < span
                   or math.hypot(defend[0] - x, defend[1] - y) < span for x, y in placed):
                continue
            if any(math.hypot(attack[0] - x, attack[1] - y) < self.radius
                   or math.hypot(defend[0] - x, defend[1] - y) < self.radius for x, y in standing):
                continue
            used.add(attack_region)
            used.add(defend_region)
            placed.extend((attack, defend))
            pairs.append(_Pair(attack_region=attack_region, attack_point=attack,
                               defend_region=defend_region, defend_point=defend))
        return pairs

    def _build_contests(self) -> None:
        """Both members of every pair as scored contests, and the mirror map they define between the contested regions."""
        self.contests = []
        self.mirror_of = {}
        for pair in self.pairs:
            self.contests.append(_Contest(region_id=pair.attack_region, point=pair.attack_point))
            self.contests.append(_Contest(region_id=pair.defend_region, point=pair.defend_point))
            self.mirror_of[pair.attack_region] = pair.defend_region
            self.mirror_of[pair.defend_region] = pair.attack_region

    def _nearest_region_id(self, point) -> Optional[int]:
        """The map region nearest a point, by the session's stable region table. The table is the map decomposition, whose ids are the same the observation reports a region under, so a region named here names the same ground the score and the priorities do."""
        regions = getattr(self.session, "regions", ())
        best: Optional[int] = None
        best_distance = float("inf")
        for region in regions:
            distance = (region.x - point[0]) ** 2 + (region.y - point[1]) ** 2
            if distance < best_distance:
                best, best_distance = region.id, distance
        return best
