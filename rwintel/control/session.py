"""One connected game instance, and the episodes it runs.

The session owns everything that is per instance: which map is loaded, the region table derived from it, the unit catalogue that instance reported, and the policy state. It drives the episode boundaries; the agent only reports what happened.

Regions are derived here rather than in the game process. The rule needs the map file, which is read here anyway, and keeping it on one side means it can be checked without launching the game at all.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

from ..data import AssetPaths, MapContent, Region, decompose, read_map
from ..data.regions import order_from
from ..eval.scoring import components, contestants, score as match_score
from ..wire import (BLOCK_REGIONS, FLAGS_MASK, Action, Kind, Observation, Passage, decode_observation, decode_terrain,
                    encode)
from .deciding import deciding
from .latency import Latency

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
    #: Whether units of the type can attack at all, the engine sample's own answer. A transport or a builder has a range without having a weapon.
    can_attack: bool = False
    #: Maximum attack range in world units for a type that can attack; a builder's is how far it builds.
    range: float = 0.0
    hits_air: bool = False
    hits_land: bool = True
    #: True for a building that may only stand on a resource pool, which is what an extractor is.
    extractor: bool = False
    #: Maximum health, from the engine's own sample unit of the type; nought when the engine keeps none.
    max_hp: float = 0.0
    #: World units a second; nought for what does not move.
    speed: float = 0.0
    #: Slots a transport of the type carries, -1 for a type that carries nothing.
    capacity: int = -1
    #: Slots a unit of the type takes aboard a transport.
    slots: int = 1
    #: Whether a unit of the type offers a tier raise at its first tier.
    upgradable: bool = False
    #: Catalogue indices of what a unit of the type makes or places at its first tier.
    menu: Tuple[int, ...] = ()
    #: For a transport, catalogue indices of the types it would load.
    carries: Tuple[int, ...] = ()

    @property
    def armed(self) -> bool:
        return self.can_attack

    @property
    def transport(self) -> bool:
        return self.capacity > 0

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
    #: Game seconds between the standings kept while an episode runs, 0 for none. They are what a decided match is scored back from at a moment before it was decided; an arena episode keeps none, since it is not scored as a match.
    standing_seconds: int = 30
    #: Holds the episode open when only one side has anything standing, which is what an episode used to construct engagements in needs: it starts empty, so the ordinary end test would finish it before the first unit was spawned.
    arena: bool = False
    #: Maps to play in turn instead of `map`, one round of the arms per map, so that every arm plays the same maps in the same order. Empty for one map.
    maps: List[str] = field(default_factory=list)
    #: Sends the built-in AI players' orders on operational observations (`Observation.ai_orders`).
    ai_orders: bool = False
    #: The AI contestants to observe the episode from, by the order they were kept in, one per episode in turn; the answers are then kept as books and nothing is carried out. Empty to play as the local player.
    watch: List[int] = field(default_factory=list)


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
    #: Whether a hosted match ended because everybody who had joined it left, in which case nobody won it and the standing is where the match was abandoned.
    peer_left: bool = False
    #: The recording this episode concerns: for a match, the file the engine recorded it to (`file`), when it recorded one; for a playback, the file played (`file`), how many recorded checksums the playback disagreed with (`mismatches`), why it ended (`ended`) and whose side it was observed from (`viewpoint`). Empty when there is neither.
    replay: dict = field(default_factory=dict)
    #: The standings taken while the episode ran, every `standing_seconds` of game time, as `second` and `standing` with only the teams that took part. Never shown to a policy: a standing is every side's totals, fog or no fog.
    history: List[dict] = field(default_factory=list)
    #: How many episodes this instance had started when this one began, counting those its game lost and played again; with the instance it names the episode in a recorded dataset.
    attempt: int = 0
    #: Matches the game process had played before this one, as the game counts them: 0 for the first match of an instance, and again 0 after its game was restarted.
    order: int = 0
    #: The map file the episode was played on, as the engine named it.
    map: str = ""
    #: The decision-latency meter's summary (`latency.Latency.summary`): per layer, answers applied, answers missed and the lag from observation to applied answer in milliseconds and tactical periods.
    latency: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "arm": self.arm, "instance": self.instance, "episode": self.episode, "attempt": self.attempt,
            "order": self.order, "map": self.map,
            "seconds": self.seconds, "winner": self.winner, "alive_teams": self.alive_teams,
            "timeout": self.timeout, "team": self.team, "standing": self.standing,
            "settings": self.settings, "statistics": self.statistics,
            "interference": self.interference, "synchronisation": self.synchronisation,
            "peer_left": self.peer_left, "wall_seconds": round(self.wall_seconds, 1),
            "replay": self.replay, "history": self.history, "latency": self.latency,
        }

    @property
    def speed(self) -> float:
        """The multiplier the episode actually ran at, which is what an instance sustained rather than what was asked of it."""
        return self.seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def value_edge(self) -> float:
        """How far ahead our side ended, as a share of the value it and its strongest opponent still held: the military component of the evaluation's score."""
        return components(self.standing, self.team).military


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
        #: The game process's working directory, which is where it keeps its saves and replays.
        self.directory = ""
        self.types: List[UnitType] = []
        self.by_lookup: Dict[str, UnitType] = {}
        self.map_content: Optional[MapContent] = None
        #: What each movement type can cross on the map in play, as the game's path finder has it; None until the game has sent it for the episode.
        self.passage: Optional[Passage] = None
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
        #: The standings the episode now running has reported so far, which become its record's history.
        self.history: List[dict] = []
        #: Why this instance could not play the episode it was asked to, as the agent reported it, or None. A failed instance is finished with: the server stops the run rather than waiting for episodes that will not come.
        self.failure: Optional[str] = None
        self.observations = 0
        #: The game time between each observation and the step that applied its answer, over the episode under way.
        self.latency = Latency()
        #: The agent's tactical period and whether its clock waits for every answer (the fixed and replay clocks), from its HELLO.
        self.tactical_ms = 200
        self.steady = False
        #: The last observation decoded, kept so that anything asking this session what the board looks like -the intervention console above all -reads the same frame the policy last decided on rather than a frame of its own.
        self.observation: Optional[Observation] = None
        self.started_at = time.time()
        #: Episodes started on this instance, counting one its game lost and played again, so that the two plays of the same episode number are told apart.
        self.attempt = 0
        #: Matches the game process had played before the episode under way.
        self.order = 0
        #: The link thread currently driving this session, so that a stopped run can tell whether its episode is still being decided on.
        self.thread_name: Optional[str] = None
        #: The map the episode under way is played on, as the engine named it.
        self.map_path = ""
        #: How the episode now being closed ended: `finished` when the game reported its end, `lost` when the game was lost with it, `rejoined` when a reconnection replaced its policy part way, and `stopped` when the run was stopped under it. Read by a policy as it is closed.
        self.ending = "finished"

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

    def _send(self, kind: Kind, body: bytes, flags: int = 0) -> None:
        # One frame at a time, because more than one thread can be sending. A link that drops mid period is replaced by a new one while the old one is still inside a policy decision, and the two would otherwise interleave their bytes on the same socket; what the agent then reads is a header out of the middle of somebody else's body, and the connection fails on the magic number rather than on anything that says what happened.
        with self._sending:
            self.connection.sendall(encode(kind, max(0, self.instance), body, flags))

    def control(self, payload: dict) -> None:
        """Sends one control instruction. Public because a policy may need to reach for one -an engagement is constructed with `scenario`, and the arena that trains the tactical layer is a policy."""
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
        """On the fixed clock the most game time per real time, 0 for no limit; on the wall clock the engine speed multiplier."""
        self.control({"command": "speed", "value": multiplier})

    def set_omniscient(self, on: bool) -> None:
        self.control({"command": "omniscient", "value": on})

    def ask_for_sync(self, complain: bool = False) -> None:
        """Asks the agent what the engine says about whether the processes in its session are still simulating the same match."""
        self.control({"command": "sync", "assert": complain})

    def episode_map(self) -> str:
        """The map of the episode this session is on, counted by rounds of the arms so that the arms of a comparison meet the maps in step."""
        maps = self.settings.maps
        if not maps:
            return self.settings.map
        return maps[(len(self.records) // max(1, len(self.arms))) % len(maps)]

    def episode_settings(self) -> dict:
        """The settings of the episode this session is on, as the record keeps them: the map it was played on rather than the list it was drawn from."""
        settings = vars(self.settings).copy()
        settings["map"] = self.episode_map()
        settings.pop("maps", None)
        return settings

    def episode_watch(self) -> int:
        """The AI contestant the episode about to start is observed from, or -1 for none."""
        watch = self.settings.watch
        return watch[len(self.records) % len(watch)] if watch else -1

    def start_episode(self) -> None:
        instruction = {
            "command": "start",
            "map": self.episode_map(),
            "opponents": self.settings.opponents,
            "difficulty": self.settings.difficulty,
            "contestants": self.settings.contestants,
            "credits": self.settings.credits,
            "startingUnits": self.settings.starting_units,
            "income": self.settings.income,
            "fog": self.settings.fog,
            "seed": self.settings.seed + len(self.records),
            "maxSeconds": self.settings.max_seconds,
            "standingMs": 0 if self.settings.arena else max(0, self.settings.standing_seconds) * 1000,
            "arena": self.settings.arena,
            "aiOrders": self.settings.ai_orders,
            "watch": self.episode_watch(),
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
        self.directory = str(payload.get("directory", ""))
        self.tactical_ms = int(payload.get("tacticalMs", self.tactical_ms))
        self.steady = bool(payload.get("steady", False))
        self.types = [
            UnitType(index=i, name=entry["name"], lookup=entry.get("lookup", entry["name"]),
                     price=int(entry.get("price", 0)), tech=int(entry.get("tech", 1)),
                     building=bool(entry.get("building", False)),
                     builder=bool(entry.get("builder", False)),
                     movement=str(entry.get("movement", "")),
                     can_attack=bool(entry.get("canAttack", False)),
                     range=float(entry.get("range", 0.0)),
                     hits_air=bool(entry.get("hitsAir", False)),
                     hits_land=bool(entry.get("hitsLand", True)),
                     extractor=bool(entry.get("extractor", False)),
                     max_hp=float(entry.get("hp", 0.0)),
                     speed=float(entry.get("speed", 0.0)),
                     capacity=int(entry.get("capacity", -1)),
                     slots=int(entry.get("slots", 1)),
                     upgradable=bool(entry.get("upgradable", False)),
                     menu=tuple(int(index) for index in entry.get("menu", ())),
                     carries=tuple(int(index) for index in entry.get("carries", ())))
            for i, entry in enumerate(payload.get("unitTypes", []))
        ]
        self.by_lookup = {t.lookup: t for t in self.types}
        log.info("instance %d connected on build %s with %d unit types, %d that can attack, transports %s",
                 self.instance, self.build or "?", len(self.types), sum(1 for t in self.types if t.can_attack),
                 ", ".join(f"{t.name}({t.capacity})" for t in self.types if t.transport) or "none")

        # An instance that has already run everything asked of it is finished with, and a HELLO from it is the agent redialling because the link was closed on it. Starting another episode here would have it play on for ever, one episode per reconnection, and each of those episodes would also be counted.
        if len(self.records) >= self.episodes_wanted:
            log.debug("instance %d has run its episodes; ignoring its reconnection", self.instance)
            return

        # A HELLO in the middle of a running episode is a reconnection, not a new instance. Starting the episode again would throw away a match the agent has been running on its own meanwhile, which is exactly what the degraded mode is for.
        if payload.get("running"):
            log.info("instance %d rejoined episode %d in progress", self.instance, payload.get("episode", 0))
            self._resume(str(payload.get("map", "")))
        else:
            if self.policy is not None:
                # The game was lost in the middle of an episode and has been started again. The episode is not recorded, and its policy is closed so that what it was collecting ends there instead of running on into the next match.
                log.warning("instance %d lost episode %d with its game; it is not recorded and is played again",
                            self.instance, len(self.records) + 1)
                self._close_policy("lost")
                self.history = []
            self.start_episode()

    def on_terrain(self, body: bytes) -> None:
        self.passage = decode_terrain(body)
        log.info("instance %d: terrain %dx%d, components %s", self.instance, self.passage.width, self.passage.height,
                 ", ".join(f"{name} {count}" for name, count in self.passage.components.items()))

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
        if payload.get("event") == "failed":
            self.failure = str(payload.get("reason", "the agent gave no reason"))
            log.error("instance %d could not play episode %d: %s", self.instance, len(self.records) + 1, self.failure)
            return
        if payload.get("event") == "progress":
            # Kept for the record and nothing else. It is every side's totals, which the policy is not to see.
            self.history.append({"second": round(int(payload.get("timeMs", 0)) / 1000.0, 1),
                                 "standing": contestants(payload.get("standing", []))})
            return

        # What the layers did is read off the policy before it is put down, and the policy is told the episode is over before anything is written down. A policy that is collecting decisions has trajectories still open at this point, and an errand that was still running when the match was called did not fail: it stopped being observed, which is a different thing and is scored differently.
        statistics = self.policy.statistics.as_dict() if hasattr(self.policy, "statistics") else {}
        # Read while the commanders are still attached: putting the policy down drops them, and the record would then say the episode was undisturbed.
        interference = self._interference()
        self.sync = dict(payload.get("sync", self.sync))
        self._close_policy("finished", self._score_of(payload))
        latency = self._report_latency()

        record = EpisodeRecord(
            # Counted here rather than taken from the agent, whose count starts again when its game is restarted.
            episode=len(self.records) + 1,
            seconds=int(payload.get("seconds", 0)),
            winner=int(payload.get("winner", -1)),
            alive_teams=int(payload.get("aliveTeams", 0)),
            timeout=bool(payload.get("timeout", False)),
            team=int(payload.get("team", -1)),
            standing=list(payload.get("standing", [])),
            arm=self.arm,
            instance=self.instance,
            settings=self.episode_settings(),
            statistics=statistics,
            synchronisation=dict(payload.get("sync", {})),
            interference=interference,
            wall_seconds=max(0.0, time.time() - self.episode_started_at),
            peer_left=bool(payload.get("peerLeft", False)),
            replay=self._replay_of(payload),
            history=self.history,
            attempt=self.attempt,
            order=self.order,
            map=os.path.basename(self.map_path),
            latency=latency,
        )
        self.history = []
        self.records.append(record)
        if self.journal is not None:
            self.journal.write(record)
        log.info("instance %d episode %d finished: %ds winner=%d timeout=%s peer_left=%s edge=%+.3f standing=%s",
                 self.instance, record.episode, record.seconds, record.winner,
                 record.timeout, record.peer_left, record.value_edge, record.standing)
        if len(self.records) < self.episodes_wanted:
            self.start_episode()
        else:
            log.info("instance %d has run its episodes", self.instance)

    def _report_latency(self) -> dict:
        """The episode's latency summary, logged in one line; on a clock that waits for every answer, a lag other than one tactical period is also logged as a warning."""
        summary = self.latency.summary(self.tactical_ms)
        tactical = summary["tactics"]
        log.info("instance %d episode %d latency: tactical lag mean %.0f ms p95 %.0f ms max %.0f ms "
                 "(%.2f / %.2f / %.2f periods), %d answer(s), %d missed; operational %d answer(s), %d missed",
                 self.instance, len(self.records) + 1, tactical["mean_ms"], tactical["p95_ms"], tactical["max_ms"],
                 tactical["mean_periods"], tactical["p95_periods"], tactical["max_periods"], tactical["answers"],
                 tactical["missed"], summary["operations"]["answers"], summary["operations"]["missed"])
        steady, first = self.latency.steady(self.tactical_ms)
        if self.steady and (not steady or tactical["missed"]):
            log.warning("instance %d episode %d: on a clock that waits for every answer, an answer lagged %s ms and %d "
                        "were missed where every one should land one tactical period (%d ms) after its observation",
                        self.instance, len(self.records) + 1, first if first is not None else self.tactical_ms,
                        tactical["missed"], self.tactical_ms)
        return summary

    def _replay_of(self, payload: dict) -> dict:
        """What a finished event says about a recording, in the record's terms."""
        if not payload.get("replay"):
            return {}
        return {"file": str(payload["replay"])}

    def close_episode(self, ending: str = "stopped") -> None:
        """Closes the policy of an episode that will not be finished, which is what a run being stopped does with the episodes still under way, so that what they were collecting is written down as ending there."""
        if self.policy is not None:
            self._close_policy(ending)

    def _score_of(self, payload: dict) -> Optional[float]:
        """The score a finished match ended with, as an evaluation scores it, or None for an episode that is not scored as a match: an arena episode, one this side only watched, or a hosted match everybody left."""
        team = int(payload.get("team", -1))
        if self.settings.arena or team < 0 or payload.get("peerLeft"):
            return None
        ended = SimpleNamespace(winner=int(payload.get("winner", -1)), team=team, timeout=bool(payload.get("timeout", False)),
                                seconds=int(payload.get("seconds", 0)), standing=list(payload.get("standing", [])),
                                history=self.history)
        return match_score(ended)

    def _close_policy(self, ending: str = "finished", score: Optional[float] = None) -> None:
        """Lets a policy finish with the episode, saying how it ended, and handing a policy that takes it the score the match ended with. Nothing the script chain does needs this; a policy that is collecting for a learning run has state that only means anything once it knows no further observation is coming."""
        self.ending = ending
        closer = getattr(self.policy, "close", None)
        try:
            if closer is not None and score is not None and "score" in inspect.signature(closer).parameters:
                closer(score=score)
            elif closer is not None:
                closer()
        except Exception:
            # A failure to tidy up must not lose the episode that was actually played, which is what the record about to be written is.
            log.exception("instance %d failed to close its policy", self.instance)
        # Put down rather than kept until the next episode replaces it. Between the end of one episode and the start of the next there is a map load, which is seconds to minutes of wall clock, and anything asking this session what it is doing in that window -a person at the console above all -has to be told that it is doing nothing rather than shown the match that has just ended.
        self.policy = None
        self.outside = []

    def _on_started(self, payload: dict) -> None:
        self.episode_started_at = time.time()
        self.history = []
        self.attempt += 1
        # The game counts its matches from 1 and restarts the count with the process.
        self.order = max(0, int(payload.get("episode", 1)) - 1)
        # The terrain frame for this map follows the start; the last map's must not stand in for it meanwhile.
        self.passage = None
        self._load_map(str(payload.get("map", "")))
        self.players = list(payload.get("players", []))
        # Which player the opposing side of a constructed engagement is spawned for. Settled game side, because it depends on how the room filled its free slots, which is not visible from here.
        self.sparring_slot = int(payload.get("sparringSlot", -1))
        # Arms alternate within a session rather than one arm being run after the other. Two arms measured in sequence differ by whatever else changed about the machine between them, and the whole point of the comparison is that nothing else changed.
        self.arm, factory = self.arms[len(self.records) % len(self.arms)]
        self.policy = factory(self)
        self._attach_outside()
        self.observations = 0
        self.latency.reset()
        log.info("instance %d episode %d on %s (%s): %d regions, players %s",
                 self.instance, payload.get("episode", 0), os.path.basename(str(payload.get("map", ""))),
                 self.arm, len(self.regions), payload.get("players", []))

    def _resume(self, map_path: str) -> None:
        """Picks an episode back up after a reconnection. The squads are still there; what has to be rebuilt is this side's view of them, and the observation carries the identifiers that does it.

        The policy that was running is closed first, as `rejoined`, rather than dropped: it may be holding decisions still waiting to be paid, and closing it cuts their trajectories where its view of the episode ended instead of losing them. The new policy carries on under the same attempt."""
        if self.policy is not None:
            self._close_policy("rejoined")
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

        Only a commander that keeps a log contributes, which in practice means the script intruder: a person's interventions are written to their own file as they happen, beside the board each was decided from, because that pairing is the imitation data and a summary at the end of the episode would have lost the state. What the record needs is the other half of the same fact -which squads a learning run must leave out of its signal -and for the intruder this is the only place it is written down.

        What is asked is the policy's own list rather than the one attached here, because a policy may arrive with commanders of its own -a training arm builds its intruder inside itself, since the run means nothing without one -and an episode whose interference went unrecorded would be indistinguishable from an undisturbed one, which is exactly the confusion this field exists to prevent.
        """
        commanders = list(getattr(self.policy, "outside", ())) or self.outside
        events: List[dict] = []
        touched = set()
        intruders = 0
        for commander in commanders:
            log_of = getattr(commander, "log", None)
            if log_of is None:
                continue
            intruders += 1
            events.extend(log_of.events)
            touched.update(log_of.touched)
        # An intruder that was attached and found nothing to do is still recorded, so that an episode measured under interference is never taken for an undisturbed one.
        if not intruders:
            return {}
        return {"intruders": intruders, "events": events, "touched": sorted(touched)}

    def _load_map(self, map_path: str) -> None:
        self.map_path = map_path
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

    def on_observation(self, body: bytes, number: int = 0) -> None:
        """Decides on one observation and answers it with exactly one action carrying its number, empty when there is nothing to change: a game on the fixed clock waits for that answer before its next period."""
        action = None
        try:
            with deciding():
                action = self._decide(body, number)
        finally:
            self._send(Kind.ACTION, action or b"", number & FLAGS_MASK)

    def _decide(self, body: bytes, number: int = 0) -> Optional[bytes]:
        observation = decode_observation(body)
        self.observations += 1
        self.observation = observation
        # The answer applied at the head of this step is to an earlier observation, so it is counted before this one is.
        self.latency.answered(observation.answered, observation.game_time_ms)
        self.latency.observed(number & FLAGS_MASK, observation.game_time_ms, bool(observation.blocks & BLOCK_REGIONS))
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
            return None
        if self.home is None and observation.unit_states:
            self.home = self._home_region(observation)
        return self.policy.decide(observation)

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
