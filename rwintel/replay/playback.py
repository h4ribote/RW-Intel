"""Playing replays back through game instances under observation.

A playback is an episode like any other to everything below the session: the agent loads the replay in place of starting a match, sends observations from the side of the player being studied, and reports the episode's end. What differs is that nothing the control process answers reaches the match -the agent keeps the books of an answer and carries none of it out- and that the agent adds, once per operational period, where every side stands.

Each session takes the next replay from a shared queue when its instance is free, so a run over many replays spreads them over however many instances it was given. The replay is copied into the instance's replays folder, which is where the engine looks for it, under a name the engine and the control frame both carry unchanged.

When the episode the replay recorded is known, from its journal, the playback is set beside it: the engine's own count of checksums it disagreed with, the checksum at the frame the recording side last took one, and the final standing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .. import paths
from ..control.server import Server, ServerSettings
from ..control.session import EpisodeSettings, Session
from ..eval import journal as journals
from .analysis import match_end
from .container import Replay, read_replay
from .timeline import Timeline

log = logging.getLogger(__name__)

#: World steps a frame runs for each step's worth of elapsed time when nothing else is asked for.
DEFAULT_STEPS = 4

#: The engine's checksum interval, for a record that does not carry it.
CHECKSUM_INTERVAL = 300

_UNSAFE = re.compile(r"[^A-Za-z0-9._()\[\] -]")


@dataclass
class Job:
    """One replay to play back, and what is known about the match it recorded."""

    path: str
    replay: Replay
    #: The slot to observe from, or -1 to observe the one playing side not on `other_than_team`.
    viewpoint: int = -1
    #: The team the recording side's own policy played, whose opponent is the side studied when no slot is named.
    other_than_team: int = -1
    #: Game time the playback stops at, or nought to run to the end of the recording.
    until_ms: int = 0
    #: The journal's record of the episode this replay recorded, when there is one.
    reference: Optional[dict] = None

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def placed_name(self) -> str:
        """The file name the replay is placed under in an instance's replays folder."""
        return _UNSAFE.sub("_", os.path.basename(self.path))


def make_job(path: str, viewpoint: int = -1, until_seconds: Optional[float] = None,
             references: Sequence[dict] = ()) -> Job:
    """A job for one replay, with the end of the playback settled: as asked, else just past the end of the recorded episode, else the end of the match as the replay itself records it.

    A playback replays every side from its commands, so no side is known as a computer player and the side to observe has to be named: by slot, or through the episode record, whose team is the policy's and whose opponent is therefore the person.
    """
    replay = read_replay(path)
    reference = find_reference(references, path, replay)
    other_than_team = int(reference.get("team", -1)) if reference is not None else -1
    if viewpoint < 0 and other_than_team < 0:
        raise ValueError(f"{os.path.basename(path)}: give the slot to observe from, or the journal of the match it recorded")
    if until_seconds is not None:
        until_ms = int(until_seconds * 1000)
    elif reference is not None and reference.get("seconds"):
        # The record keeps whole seconds, so the match ended somewhere within the second after the one it says.
        until_ms = (int(reference["seconds"]) + 1) * 1000
    else:
        until_ms = replay.clock.time_ms(match_end(replay)[0])
    return Job(path=path, replay=replay, viewpoint=viewpoint, other_than_team=other_than_team,
               until_ms=until_ms, reference=reference)


def find_reference(records: Sequence[dict], replay_path: str, replay: Replay) -> Optional[dict]:
    """The journal record of the episode a replay recorded: the one that names it, or else the one whose last checksum the replay shows being taken just before the match ended.

    The recording side writes extra checksums at every frame it takes a world checksum, and the record keeps the frame of its last one, which falls within one checksum interval before the end of the match. Checksums are taken on the same grid of frames in every match, so the frame alone is shared by unrelated matches; the end of the match, as the replay records it, is what tells them apart. When more than one record would match, none is taken.
    """
    name = os.path.basename(replay_path)
    for record in records:
        if (record.get("replay") or {}).get("file") == name:
            return record
    frames = {extra.frame for extra in replay.extra_checksums}
    end_frame = match_end(replay)[0]

    def matches(record: dict) -> bool:
        sync = record.get("synchronisation") or {}
        frame = int(sync.get("frame", -1))
        interval = int(sync.get("interval", 0)) or CHECKSUM_INTERVAL
        return frame in frames and 0 <= end_frame - frame <= interval

    matching = [record for record in records if not (record.get("replay") or {}).get("file") and matches(record)]
    return matching[0] if len(matching) == 1 else None


class Jobs:
    """The replays still to be played, handed out one at a time to whichever session is free."""

    def __init__(self, jobs: Sequence[Job]) -> None:
        self._jobs = list(jobs)
        self._lock = threading.Lock()

    def take(self) -> Optional[Job]:
        with self._lock:
            return self._jobs.pop(0) if self._jobs else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)


@dataclass
class PlaybackOptions:
    steps: int = DEFAULT_STEPS
    omniscient: bool = True
    #: Where each playback's timeline and summary are written, under a directory named after the replay.
    output: str = field(default_factory=paths.replays)


def check(job: Job, payload: dict, timeline: Timeline) -> dict:
    """How the playback compares with the episode it recorded: the engine's disagreements, the checksum at the recording side's last checksum frame, and the standing at the end.

    The record keeps the end as whole seconds, so the recorded standing is looked for among the standings the playback reported from that second on: the match was decided somewhere inside it.
    """
    result: Dict[str, object] = {"mismatches": int(payload.get("mismatches", -1))}
    reference = job.reference
    if reference is None:
        return result
    recorded = reference.get("synchronisation") or {}
    frame = int(recorded.get("frame", -1))
    if frame >= 0:
        played = next((value for at, value in timeline.checksums if at == frame), None)
        result["checksum"] = {"frame": frame, "recorded": recorded.get("checksum"), "played": played,
                              "agrees": played is not None and played == recorded.get("checksum")}
    standing = _standing(reference.get("standing", []))
    since = int(reference.get("seconds", 0)) * 1000
    seen = [(row["time_ms"], _standing(row["standing"])) for row in timeline.rows if row["time_ms"] >= since]
    at = next((time_ms for time_ms, played in seen if played == standing), None)
    result["standing"] = {"recorded": standing, "played": _standing(payload.get("standing", [])),
                          "agrees": at is not None, "at_ms": at}
    return result


def _standing(entries: Sequence[dict]) -> List[dict]:
    return sorted(({k: entry.get(k) for k in ("team", "units", "value", "income", "killed", "lost")}
                   for entry in entries), key=lambda entry: entry["team"])


class ReplaySession(Session):
    """One instance, playing replays from the shared queue until there are none left."""

    def __init__(self, connection, address, jobs: Jobs, options: PlaybackOptions,
                 policy_factory: Callable[["ReplaySession"], object], assets=None, journal=None) -> None:
        # No time limit is stated: a recorded match ran to whatever length its players played it, so the time left in it is not known to the layers.
        super().__init__(connection, address, EpisodeSettings(max_seconds=0), arms=[("replay", policy_factory)],
                         assets=assets, episodes=1, journal=journal)
        # How many episodes this session runs is not known in advance: it plays until the queue is empty.
        self.episodes_wanted = sys.maxsize
        self.jobs = jobs
        self.options = options
        self.job: Optional[Job] = None
        self.viewpoint = -1
        self.timeline: Optional[Timeline] = None
        #: What each playback came to, in the order they finished.
        self.results: List[dict] = []

    @property
    def done(self) -> bool:
        return self.job is None and self.episodes_wanted <= len(self.records)

    def start_episode(self) -> None:
        self.job = self.jobs.take()
        if self.job is None:
            self.episodes_wanted = len(self.records)
            log.info("instance %d has no replay left to play", self.instance)
            return
        name = self._place(self.job)
        self.control({"command": "replay", "name": name, "viewpoint": self.job.viewpoint,
                      "otherThanTeam": self.job.other_than_team, "untilMs": self.job.until_ms,
                      "steps": self.options.steps, "omniscient": self.options.omniscient})
        log.info("instance %d plays %s back to %.1fs", self.instance, self.job.stem, self.job.until_ms / 1000.0)

    def _place(self, job: Job) -> str:
        """Copies the replay to where this instance's game looks for replays, and returns the name it is there under."""
        directory = os.path.join(self.directory or paths.instance(max(0, self.instance)), "replays")
        os.makedirs(directory, exist_ok=True)
        target = os.path.join(directory, job.placed_name)
        if os.path.abspath(target) != os.path.abspath(job.path):
            shutil.copyfile(job.path, target)
        return job.placed_name

    def _on_started(self, payload: dict) -> None:
        reported = str(payload.get("map", ""))
        wanted = self.job.replay.map_path if self.job is not None else reported
        if reported and wanted and reported != wanted:
            log.warning("instance %d: the game reports map %s, the replay names %s; using the replay's",
                        self.instance, reported, wanted)
        # Set before the policy is built, since a policy that reads the person's orders needs to know whose they are.
        self.viewpoint = int(payload.get("viewpoint", -1))
        super()._on_started({**payload, "map": wanted or reported})
        self.timeline = Timeline(self.regions, [t.price for t in self.types], omniscient=self.options.omniscient)
        log.info("instance %d observes %s from slot %d", self.instance, self.job.stem if self.job else "?", self.viewpoint)

    def on_episode(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        event = payload.get("event")
        if event == "progress":
            if self.timeline is not None:
                self.timeline.progress(payload)
            return
        if event == "finished" and self.job is not None and self.timeline is not None:
            self._finish(payload)
        elif event == "failed":
            # A replay that did not load is reported and skipped; the rest of the queue is still worth playing.
            log.error("instance %d could not play %s: %s", self.instance, self.job.stem if self.job else "?",
                      payload.get("reason"))
            self.results.append({"replay": self.job.stem if self.job else "", "failed": payload.get("reason")})
            self.start_episode()
            return
        super().on_episode(body)

    def _score_of(self, payload: dict) -> None:
        """A playback ends where its recording or its requested stretch ends, not where a match was scored, so it hands no score and what was collected from it is cut there."""
        return None

    def _replay_of(self, payload: dict) -> dict:
        replay = super()._replay_of(payload)
        for key in ("mismatches", "ended", "exhausted"):
            if key in payload:
                replay[key] = payload[key]
        replay["viewpoint"] = self.viewpoint
        return replay

    def _decide(self, body: bytes, number: int = 0) -> Optional[bytes]:
        answer = super()._decide(body, number)
        if self.timeline is not None and self.observation is not None:
            self.timeline.observe(self.observation)
        return answer

    def _finish(self, payload: dict) -> None:
        job, timeline = self.job, self.timeline
        sync = payload.get("sync") or {}
        timeline.progress({"timeMs": int(payload.get("seconds", 0)) * 1000, "frame": payload.get("frames", 0),
                           "standing": payload.get("standing", []), "checksumFrame": sync.get("frame", -1),
                           "checksum": sync.get("checksum", 0)}, row=False)
        verdict = check(job, payload, timeline)
        directory = os.path.join(self.options.output, job.stem)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"timeline-slot{self.viewpoint}.jsonl"), "w", encoding="utf-8") as out:
            for row in timeline.rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        summary = {"replay": os.path.basename(job.path), "viewpoint": self.viewpoint, "players": self.players,
                   "until_ms": job.until_ms, "seconds": payload.get("seconds"), "frames": payload.get("frames"),
                   "ended": payload.get("ended"), "check": verdict, **timeline.summary()}
        with paths.replacing(os.path.join(directory, f"summary-slot{self.viewpoint}.json")) as out:
            json.dump(summary, out, indent=1)
        self.results.append({"replay": job.stem, "viewpoint": self.viewpoint, "check": verdict,
                             "directory": directory})
        log.info("instance %d finished %s at %ss (%s): %s", self.instance, job.stem, payload.get("seconds"),
                 payload.get("ended"), describe(verdict))
        self.timeline = None


def describe(verdict: dict) -> str:
    """A playback's check as one line."""
    parts = [f"{verdict.get('mismatches')} checksum disagreement(s)"]
    checksum = verdict.get("checksum")
    if checksum:
        parts.append(f"checksum at frame {checksum['frame']} {'agrees' if checksum['agrees'] else 'DIFFERS'} "
                     f"({checksum['played']} played, {checksum['recorded']} recorded)")
    standing = verdict.get("standing")
    if standing:
        parts.append("final standing " + ("agrees" if standing["agrees"] else "DIFFERS"))
    return ", ".join(parts)


class ReplayServer(Server):
    def __init__(self, settings: ServerSettings, jobs: Jobs, options: PlaybackOptions,
                 policy_factory: Callable[[ReplaySession], object]) -> None:
        super().__init__(settings, policy_factory)
        self.jobs = jobs
        self.options = options
        self.policy_factory = policy_factory

    def _finished(self) -> bool:
        with self._lock:
            if len(self.sessions) < self.settings.instances:
                return False
            return all(session.done for session in self.sessions)

    def _session_for(self, instance: int, connection, address) -> Session:
        with self._lock:
            for session in self.sessions:
                if session.instance == instance:
                    session.rebind(connection, address)
                    return session
            session = ReplaySession(connection, address, self.jobs, self.options, self.policy_factory,
                                    assets=self.settings.assets, journal=self.settings.journal)
            self.sessions.append(session)
            return session


def load_references(paths_given: Sequence[str]) -> List[dict]:
    """Every episode record in the journals given."""
    return [record for path in paths_given for record in journals.read(path)]
