"""One connected game instance, and the episodes it runs.

The session owns everything that is per instance: which map is loaded, the region table derived from it, the unit catalogue that instance reported, and the policy state. It drives the episode boundaries; the agent only reports what happened.

Regions are derived here rather than in the game process. The rule needs the map file, which is read here anyway, and keeping it on one side means it can be checked without launching the game at all.
"""

from __future__ import annotations

import json
import logging
import os
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
    def __init__(self, connection, address, settings: EpisodeSettings, policy_factory,
                 assets: Optional[AssetPaths] = None, episodes: int = 1):
        self.connection = connection
        self.address = address
        self.settings = settings
        self.policy_factory = policy_factory
        self.assets = assets or AssetPaths.default()
        self.episodes_wanted = episodes

        self.instance = -1
        self.types: List[UnitType] = []
        self.by_lookup: Dict[str, UnitType] = {}
        self.map_content: Optional[MapContent] = None
        self.regions: List[Region] = []
        self.home: Optional[Region] = None
        self.policy = None
        self.records: List[EpisodeRecord] = []
        self.observations = 0
        self.started_at = time.time()

    # ---- outgoing --------------------------------------------------------------------

    def _send(self, kind: Kind, body: bytes) -> None:
        self.connection.sendall(encode(kind, max(0, self.instance), body))

    def _control(self, payload: dict) -> None:
        self._send(Kind.CONTROL, json.dumps(payload).encode("utf-8"))

    def start_episode(self) -> None:
        self._control({
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
        })

    # ---- incoming --------------------------------------------------------------------

    def on_hello(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        self.instance = int(payload.get("instance", 0))
        self.types = [
            UnitType(index=i, name=entry["name"], lookup=entry.get("lookup", entry["name"]),
                     price=int(entry.get("price", 0)), tech=int(entry.get("tech", 1)),
                     building=bool(entry.get("building", False)),
                     builder=bool(entry.get("builder", False)),
                     movement=str(entry.get("movement", "")))
            for i, entry in enumerate(payload.get("unitTypes", []))
        ]
        self.by_lookup = {t.lookup: t for t in self.types}
        log.info("instance %d connected with %d unit types", self.instance, len(self.types))
        self.start_episode()

    def on_episode(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        if payload.get("event") == "started":
            self._on_started(payload)
            return

        record = EpisodeRecord(
            episode=int(payload.get("episode", 0)),
            seconds=int(payload.get("seconds", 0)),
            winner=int(payload.get("winner", -1)),
            alive_teams=int(payload.get("aliveTeams", 0)),
            timeout=bool(payload.get("timeout", False)),
            team=int(payload.get("team", -1)),
            standing=list(payload.get("standing", [])),
        )
        self.records.append(record)
        log.info("instance %d episode %d finished: %ds winner=%d timeout=%s edge=%+.3f standing=%s",
                 self.instance, record.episode, record.seconds, record.winner,
                 record.timeout, record.value_edge, record.standing)
        if len(self.records) < self.episodes_wanted:
            self.start_episode()
        else:
            log.info("instance %d has run its episodes", self.instance)

    def _on_started(self, payload: dict) -> None:
        map_path = payload.get("map", "")
        self.map_content = self._read_map(map_path)
        self.regions = decompose(self.map_content) if self.map_content else []
        self.home = None
        self.policy = self.policy_factory(self)
        self.observations = 0

        rows: List[float] = []
        for region in self.regions:
            rows.extend([round(region.x, 2), round(region.y, 2), float(region.resources), 0.0])
        self._control({"command": "regions", "regions": rows})
        log.info("instance %d episode %d on %s: %d regions, players %s",
                 self.instance, payload.get("episode", 0), os.path.basename(map_path),
                 len(self.regions), payload.get("players", []))

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
