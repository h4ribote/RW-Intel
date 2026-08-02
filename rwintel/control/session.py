"""One connected game instance, and the episodes it runs.

The session owns everything that is per instance: which map is loaded, the region table derived from it, the unit catalogue that instance reported, and the policy state. It drives the episode boundaries; the agent only reports what happened.

Regions are derived here rather than in the game process. The rule needs the map file, which is read here anyway, and keeping it on one side means it can be checked without launching the game at all.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data import AssetPaths, MapContent, Region, decompose, read_map
from ..data.regions import order_from
from ..wire import Action, Kind, Observation, decode_observation, encode

log = logging.getLogger(__name__)


@dataclass
class UnitType:
    index: int
    name: str
    lookup: str
    price: int
    tech: int
    building: bool
    builder: bool
    movement: str
    #: Maximum attack range in world units, zero for something that cannot shoot.
    range: float = 0.0
    hits_air: bool = False
    hits_land: bool = True
    #: True for a building that may only stand on a resource pool, which is what an extractor is.
    extractor: bool = False

    @property
    def armed(self) -> bool:
        return self.range > 0.0

    @property
    def mobile(self) -> bool:
        return not self.building and self.movement not in ("", "NONE", "BUILDING")


@dataclass
class EpisodeSettings:
    """What a `start` control frame carries. The defaults are the ones the measurements were taken with."""

    map: str = "Islands"
    opponents: int = 1
    difficulty: int = 1
    contestants: int = 0
    credits: int = 0
    starting_units: int = 1
    income: float = 1.0
    fog: int = 2
    seed: int = 12345
    max_seconds: int = 900
    #: Holds the episode open when only one side has anything standing, which is what an episode used to construct engagements in needs: it starts empty, so the ordinary end test would finish it before the first unit was spawned.
    arena: bool = False


@dataclass
class EpisodeRecord:
    episode: int
    seconds: int
    winner: int
    alive_teams: int
    timeout: bool
    #: The team the observations were taken from, or -3 when it only watched.
    team: int = -1
    standing: List[dict] = field(default_factory=list)
    #: Where the sides stood some time BEFORE the episode ended, and how far into the match that was. The board an episode ends on has, for a decided match, the loser already destroyed, and the scoring weights are asked to predict a winner from a board where both sides are still standing — so a fit made on the final board is a fit on an easier question than the one the score is used on. Empty for an episode too short to have one.
    before: List[dict] = field(default_factory=list)
    before_seconds: int = 0
    #: Which arm of a comparison this episode belongs to. One name for a plain run.
    arm: str = ""
    instance: int = -1
    #: The settings the episode was played under, so that two numbers are only ever compared when they were produced the same way.
    settings: dict = field(default_factory=dict)
    #: What each layer did, which is what separates a bad result from a result and says which layer it came from.
    statistics: dict = field(default_factory=dict)
    #: What anyone outside the chain did to this episode, and to which squads. Kept beside the statistics rather than inside them because it is not a measure of the run: it is what a later learning run reads to decide which squads' results it must throw away, since a squad that was taken over or emptied half way through its mission was not the chain's to be judged on.
    interference: dict = field(default_factory=dict)
    #: Wall clock seconds the episode took, against which its game seconds give the speed actually achieved. The multiplier asked for is a request; what an instance sustains depends on how much is on the board and how many other instances are sharing the machine, and only this says which.
    wall_seconds: float = 0.0
    #: What the engine said about whether the processes sharing this match were still simulating the same one. Empty for the ordinary case of one process to a match, where the question does not arise; where it does, an episode that drifted apart part way through describes nothing and its numbers must not be used.
    synchronisation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "arm": self.arm, "instance": self.instance, "episode": self.episode,
            "seconds": self.seconds, "winner": self.winner, "alive_teams": self.alive_teams,
            "timeout": self.timeout, "team": self.team, "standing": self.standing,
            "before": self.before, "before_seconds": self.before_seconds,
            "settings": self.settings, "statistics": self.statistics,
            "interference": self.interference, "synchronisation": self.synchronisation,
            "wall_seconds": round(self.wall_seconds, 1),
        }

    @property
    def speed(self) -> float:
        """The multiplier the episode actually ran at, which is what an instance sustained rather than what was asked of it."""
        return self.seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def value_edge(self) -> float:
        """How far ahead our side ended, as a share of the value both sides still held.

        This is the main measure, because a win or a loss is not observed often enough to be one: built-in AI against built-in AI produced no decision at all over 69 episodes. Ranging from -1 to +1 keeps it comparable across maps and match lengths.
        """
        playing = [entry for entry in self.standing if entry.get("team", -1) >= 0]
        if len(playing) < 2:
            return 0.0
        ours = next((e for e in playing if e.get("team") == self.team), None)
        if ours is None:
            # Watching rather than playing, so the two contestants are compared with each other.
            ours, rest = playing[0], playing[1:]
        else:
            rest = [e for e in playing if e is not ours]
        mine = ours.get("value", 0)
        theirs = max(e.get("value", 0) for e in rest)
        total = mine + theirs
        return (mine - theirs) / total if total else 0.0


class Session:
    def __init__(self, connection, address, settings: EpisodeSettings, arms,
                 assets: Optional[AssetPaths] = None, episodes: int = 1, journal=None,
                 outside=None):
        self.connection = connection
        self.address = address
        self.settings = settings
        self._sending = threading.Lock()
        #: The policies to run, as (name, factory) pairs. More than one makes the run a comparison.
        self.arms = list(arms)
        self.assets = assets or AssetPaths.default()
        #: Episodes each arm is to run, so a session plays this many times the number of arms.
        self.episodes_wanted = episodes * len(self.arms)
        self.journal = journal
        #: Ways of building a commander outside the chain, called once per episode with this session. A factory rather than a commander so that what a person or an intruder holds is a fact about one match: a squad number means a different squad next episode, and a holding carried across would be a squad nobody took.
        self.outside_factories = list(outside or [])
        #: The outside commanders of the episode currently running.
        self.outside: List = []
        self.arm = self.arms[0][0]

        self.instance = -1
        self.build = ""
        self.types: List[UnitType] = []
        self.by_lookup: Dict[str, UnitType] = {}
        self.map_content: Optional[MapContent] = None
        self.regions: List[Region] = []
        self.home: Optional[Region] = None
        #: The player list the game reported when the episode began, and which of those players an arena episode's opposing side belongs to.
        self.players: List[dict] = []
        self.sparring_slot = -1
        #: When the episode now running began, in wall clock, so that what it cost can be told from what it simulated.
        self.episode_started_at = time.time()
        self.policy = None
        #: How this instance takes part in a match shared with another process, or None for the ordinary case of one process to a match.
        self.pairing = None
        #: What the engine last said about whether the processes in this session are still simulating the same match.
        self.sync: dict = {}
        self.records: List[EpisodeRecord] = []
        self.observations = 0
        #: The last observation decoded, kept so that anything asking this session what the board looks like — the intervention console above all — reads the same frame the policy last decided on rather than a frame of its own.
        self.observation: Optional[Observation] = None
        self.started_at = time.time()

    @staticmethod
    def instance_in(hello: bytes) -> int:
        """The instance number out of a HELLO body, which is what says whether a connection is a new instance or one coming back."""
        return int(json.loads(hello.decode("utf-8")).get("instance", 0))

    def rebind(self, connection, address) -> None:
        try:
            self.connection.close()
        except OSError:
            pass
        self.connection = connection
        self.address = address

    # ---- outgoing --------------------------------------------------------------------

    def _send(self, kind: Kind, body: bytes) -> None:
        # One frame at a time, because more than one thread can be sending. A link that drops mid period is replaced by a new one while the old one is still inside a policy decision, and the two would otherwise interleave their bytes on the same socket; what the agent then reads is a header out of the middle of somebody else's body, and the connection fails on the magic number rather than on anything that says what happened.
        with self._sending:
            self.connection.sendall(encode(kind, max(0, self.instance), body))

    def control(self, payload: dict) -> None:
        """Sends one control instruction. Public because a policy may need to reach for one — an engagement is constructed with `scenario`, and the arena that trains the tactical layer is a policy."""
        self._send(Kind.CONTROL, json.dumps(payload).encode("utf-8"))

    def scenario(self, spawns: List[float], sandbox: Optional[bool] = None) -> None:
        """Builds a situation on the board: the sandbox flag, and units to create as flat rows of type index, player slot, x, y and how many.

        Every unit goes in through the engine's own spawn command rather than being assigned into its state, which is what keeps a lockstep session in step. There is deliberately no instruction to clear the board, because the game has no command that removes a unit; an arena episode is begun with no starting units and filled in instead.
        """
        payload: Dict[str, object] = {"command": "scenario", "spawns": spawns}
        if sandbox is not None:
            payload["sandbox"] = sandbox
        self.control(payload)

    def abort(self) -> None:
        self.control({"command": "abort"})

    def set_speed(self, multiplier: float) -> None:
        self.control({"command": "speed", "value": multiplier})

    def set_omniscient(self, on: bool) -> None:
        self.control({"command": "omniscient", "value": on})

    def ask_for_sync(self, complain: bool = False) -> None:
        """Asks the agent what the engine says about whether the processes in its session are still simulating the same match."""
        self.control({"command": "sync", "assert": complain})

    def start_episode(self) -> None:
        instruction = {
            "command": "start",
            "map": self.settings.map,
            "opponents": self.settings.opponents,
            "difficulty": self.settings.difficulty,
            "contestants": self.settings.contestants,
            "credits": self.settings.credits,
            "startingUnits": self.settings.starting_units,
            "income": self.settings.income,
            "fog": self.settings.fog,
            "seed": self.settings.seed + len(self.records),
            "maxSeconds": self.settings.max_seconds,
            "arena": self.settings.arena,
            "name": f"rw-intel-{max(0, self.instance)}",
        }
        if self.pairing is not None:
            instruction.update(self.pairing.instruction(self))
        self.control(instruction)

    # ---- incoming --------------------------------------------------------------------

    def on_hello(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        self.instance = int(payload.get("instance", 0))
        self.build = str(payload.get("build", ""))
        self.types = [
            UnitType(index=i, name=entry["name"], lookup=entry.get("lookup", entry["name"]),
                     price=int(entry.get("price", 0)), tech=int(entry.get("tech", 1)),
                     building=bool(entry.get("building", False)),
                     builder=bool(entry.get("builder", False)),
                     movement=str(entry.get("movement", "")),
                     range=float(entry.get("range", 0.0)),
                     hits_air=bool(entry.get("hitsAir", False)),
                     hits_land=bool(entry.get("hitsLand", True)),
                     extractor=bool(entry.get("extractor", False)))
            for i, entry in enumerate(payload.get("unitTypes", []))
        ]
        self.by_lookup = {t.lookup: t for t in self.types}
        log.info("instance %d connected on build %s with %d unit types",
                 self.instance, self.build or "?", len(self.types))

        # An instance that has already run everything asked of it is finished with, and a HELLO from it is the agent redialling because the link was closed on it. Starting another episode here would have it play on for ever, one episode per reconnection, and each of those episodes would also be counted.
        if len(self.records) >= self.episodes_wanted:
            log.debug("instance %d has run its episodes; ignoring its reconnection", self.instance)
            return

        # A HELLO in the middle of a running episode is a reconnection, not a new instance. Starting the episode again would throw away a match the agent has been running on its own meanwhile, which is exactly what the degraded mode is for.
        if payload.get("running"):
            log.info("instance %d rejoined episode %d in progress", self.instance, payload.get("episode", 0))
            self._resume(str(payload.get("map", "")))
        else:
            self.start_episode()

    def on_episode(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        if payload.get("event") == "started":
            self.sync = dict(payload.get("sync", {}))
            self._on_started(payload)
            if self.pairing is not None:
                self.pairing.started(self)
            return
        if payload.get("event") == "sync":
            # A report about synchronisation, not the end of anything. Counting it as an episode would have the two sides disagreeing about how many have been run.
            self.sync = {key: value for key, value in payload.items() if key != "event"}
            log.info("instance %d sync: %s", self.instance, self.sync)
            return

        # What the layers did is read off the policy before it is put down, and the policy is told the episode is over before anything is written down. A policy that is collecting decisions has trajectories still open at this point, and an errand that was still running when the match was called did not fail: it stopped being observed, which is a different thing and is scored differently.
        statistics = self.policy.statistics.as_dict() if hasattr(self.policy, "statistics") else {}
        # The interference is read off the policy's own commanders here, before the policy is put down, for the same reason the statistics are: `_close_policy` sets `self.policy` to None, and `_interference` reads the outside commanders' logs off that policy. Gathered after the put-down it read an empty list every time, so every episode was journalled as undisturbed even when an intruder had been rewriting contracts throughout it — the one confusion this field exists to prevent.
        interference = self._interference()
        self.sync = dict(payload.get("sync", self.sync))

        record = EpisodeRecord(
            episode=int(payload.get("episode", 0)),
            seconds=int(payload.get("seconds", 0)),
            winner=int(payload.get("winner", -1)),
            alive_teams=int(payload.get("aliveTeams", 0)),
            timeout=bool(payload.get("timeout", False)),
            team=int(payload.get("team", -1)),
            standing=list(payload.get("standing", [])),
            before=list(payload.get("before", [])),
            before_seconds=int(payload.get("beforeSeconds", 0)),
            arm=self.arm,
            instance=self.instance,
            settings=vars(self.settings).copy(),
            statistics=statistics,
            synchronisation=dict(payload.get("sync", {})),
            interference=interference,
            wall_seconds=max(0.0, time.time() - self.episode_started_at),
        )
        # The record is built before the policy is put down, and handed to it, because how the match ended is a statement only this side holds and a layer paid the match has no other way to be told it. That is the same arrangement an arena has with the layer it scores — the runner knows when the work is over and what it came to, the layer only ever sees periods — and it has to happen before the close, since the close is what ends the open trajectory the result is owed to.
        self._conclude_policy(record)
        self._close_policy()
        self.records.append(record)
        if self.journal is not None:
            self.journal.write(record)
        log.info("instance %d episode %d finished: %ds winner=%d timeout=%s edge=%+.3f standing=%s",
                 self.instance, record.episode, record.seconds, record.winner,
                 record.timeout, record.value_edge, record.standing)
        if len(self.records) < self.episodes_wanted:
            self.start_episode()
        else:
            log.info("instance %d has run its episodes", self.instance)

    def _conclude_policy(self, record: EpisodeRecord) -> None:
        """Tells a policy how the match it has just played came out, where the policy is one that wants to know.

        The script chain does not: no layer of it is paid anything. A learning run whose layer is paid the match — which the design says is the strategic layer and only that one — takes the result here and pays its last decision with it. Anything that goes wrong doing so must not lose the episode that was actually played, exactly as with the close below, so it is reported and the run goes on.
        """
        concluded = getattr(self.policy, "conclude", None)
        if concluded is None:
            return
        try:
            concluded(record)
        except Exception:
            log.exception("instance %d failed to hand its policy the result of the episode", self.instance)

    def _close_policy(self) -> None:
        """Lets a policy finish with the episode. Nothing the script chain does needs this; a policy that is collecting for a learning run has state that only means anything once it knows no further observation is coming."""
        closer = getattr(self.policy, "close", None)
        try:
            if closer is not None:
                closer()
        except Exception:
            # A failure to tidy up must not lose the episode that was actually played, which is what the record about to be written is.
            log.exception("instance %d failed to close its policy", self.instance)
        # Put down rather than kept until the next episode replaces it. Between the end of one episode and the start of the next there is a map load, which is seconds to minutes of wall clock, and anything asking this session what it is doing in that window — a person at the console above all — has to be told that it is doing nothing rather than shown the match that has just ended.
        self.policy = None
        self.outside = []

    def _on_started(self, payload: dict) -> None:
        self.episode_started_at = time.time()
        self._load_map(str(payload.get("map", "")))
        self.players = list(payload.get("players", []))
        # Which player the opposing side of a constructed engagement is spawned for. Settled game side, because it depends on how the room filled its free slots, which is not visible from here.
        self.sparring_slot = int(payload.get("sparringSlot", -1))
        # Arms alternate within a session rather than one arm being run after the other. Two arms measured in sequence differ by whatever else changed about the machine between them, and the whole point of the comparison is that nothing else changed.
        self.arm, factory = self.arms[len(self.records) % len(self.arms)]
        self.policy = factory(self)
        self._attach_outside()
        self.observations = 0
        log.info("instance %d episode %d on %s (%s): %d regions, players %s",
                 self.instance, payload.get("episode", 0), os.path.basename(str(payload.get("map", ""))),
                 self.arm, len(self.regions), payload.get("players", []))

    def _resume(self, map_path: str) -> None:
        """Picks an episode back up after a reconnection. The squads are still there; what has to be rebuilt is this side's view of them, and the observation carries the identifiers that does it."""
        self._load_map(map_path)
        self.arm, factory = self.arms[len(self.records) % len(self.arms)]
        self.policy = factory(self)
        self._attach_outside()

    def _attach_outside(self) -> None:
        """Builds this episode's commanders outside the chain and hands them to the policy.

        Each is given the organisation layer of the policy it is about to amend, because the cap of eight squads is kept there and nowhere else: a commander that wants a squad of its own for units it has pulled out of another borrows the slot rather than choosing a number, or two commanders would eventually name the same squad and the observation would describe whichever wrote last.

        They are consulted in the order they were configured, and the last word about a squad belongs to whoever is consulted last, so a run with both an intruder and a person puts the person last.
        """
        self.outside = [make(self) for make in self.outside_factories]
        for commander in self.outside:
            commander.organisation = self.policy.organisation
        if self.policy is not None:
            # Added to whatever the policy brought with it rather than put in its place. A training run builds its own intruder inside the arm, because the run is not meaningful without one, and replacing the list here would quietly remove it.
            self.policy.outside = list(getattr(self.policy, "outside", ())) + list(self.outside)

    def _interference(self) -> dict:
        """What the outside commanders did over the episode, gathered for the record.

        Only a commander that keeps a log contributes, which in practice means the script intruder: a person's interventions are written to their own file as they happen, beside the board each was decided from, because that pairing is the imitation data and a summary at the end of the episode would have lost the state. What the record needs is the other half of the same fact — which squads a learning run must leave out of its signal — and for the intruder this is the only place it is written down.

        What is asked is the policy's own list rather than the one attached here, because a policy may arrive with commanders of its own — a training arm builds its intruder inside itself, since the run means nothing without one — and an episode whose interference went unrecorded would be indistinguishable from an undisturbed one, which is exactly the confusion this field exists to prevent.
        """
        commanders = list(getattr(self.policy, "outside", ())) or self.outside
        events: List[dict] = []
        touched = set()
        for commander in commanders:
            log_of = getattr(commander, "log", None)
            if log_of is None:
                continue
            events.extend(log_of.events)
            touched.update(log_of.touched)
        if not events and not touched:
            return {}
        return {"events": events, "touched": sorted(touched)}

    def _load_map(self, map_path: str) -> None:
        self.map_content = self._read_map(map_path)
        self.regions = decompose(self.map_content) if self.map_content else []
        self.home = None
        # The region table goes over in the order the map decomposition produced, which is stable across runs and across a reconnection. The egocentric order the design asks for is applied where a layer is handed the table, not on the wire: home is not known until something has been built, and renumbering the slots part way through an episode would move the ground under a policy that had learnt what a slot means.
        rows: List[float] = []
        for region in self.regions:
            rows.extend([round(region.x, 2), round(region.y, 2), float(region.resources), 1.0 if region.spawn else 0.0])
        # The resource points go over with the table because a pool is not an object in the world: it is a flag on a map tile, and until an extractor is built there is nothing at the position to find. The game side needs the positions to say who holds what.
        points: List[float] = []
        if self.map_content is not None and self.regions:
            for tile in self.map_content.resources:
                x, y = self.map_content.to_world(tile)
                nearest = min(self.regions, key=lambda r: (r.x - x) ** 2 + (r.y - y) ** 2)
                points.extend([round(x, 2), round(y, 2), float(nearest.id)])
        self.control({"command": "regions", "regions": rows, "resourcePoints": points})

    def _read_map(self, engine_path: str) -> Optional[MapContent]:
        """The engine reports a path relative to its own assets directory, which is where the reader looks anyway."""
        if not engine_path:
            return None
        full = os.path.join(self.assets.assets, engine_path.replace("/", os.sep))
        if not os.path.exists(full):
            log.warning("no map file at %s", full)
            return None
        return read_map(full, self.assets)

    def on_observation(self, body: bytes) -> None:
        observation = decode_observation(body)
        self.observations += 1
        self.observation = observation
        if log.isEnabledFor(logging.DEBUG) and observation.regions:
            log.debug("instance %d t=%ds blocks=%x regions=%d(held %d/%d) squads=%d units=%d(%d enemy) events=%s home=%.0f..%.0f",
                      self.instance, observation.game_time_ms // 1000, observation.blocks, len(observation.regions),
                      sum(r.held_by_us for r in observation.regions), sum(r.held_by_enemy for r in observation.regions),
                      len(observation.squads), len(observation.unit_states),
                      sum(1 for u in observation.unit_states if u.hostile),
                      [(e.kind, e.unit) for e in observation.events],
                      min(r.distance_from_home for r in observation.regions),
                      max(r.distance_from_home for r in observation.regions))
        if self.policy is None:
            return
        if self.home is None and observation.unit_states:
            self.home = self._home_region(observation)
        action = self.policy.decide(observation)
        if action is not None:
            self._send(Kind.ACTION, action)

    def _home_region(self, observation: Observation) -> Optional[Region]:
        """The region our own buildings are in, which is the origin every other region is ordered from."""
        ours = [u for u in observation.unit_states if not u.hostile]
        if not ours or not self.regions:
            return None
        x = sum(u.x for u in ours) / len(ours)
        y = sum(u.y for u in ours) / len(ours)
        return min(self.regions, key=lambda r: (r.x - x) ** 2 + (r.y - y) ** 2)

    def ordered_regions(self) -> List[Region]:
        return order_from(self.regions, self.home)

    def type_by_lookup(self, lookup: str) -> Optional[UnitType]:
        return self.by_lookup.get(lookup)
