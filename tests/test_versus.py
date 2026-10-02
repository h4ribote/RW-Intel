"""Holds the control side of a hosted match to its promises: what the host is told, what plays, and how a match nobody joined or everybody left is reported.

None of this starts the game. The command line is parsed as data, and a session is fed the episode events an agent sends.
"""

from __future__ import annotations

import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel.control.__main__ import DEFAULT_MAX_SECONDS, against_a_person, settle
from rwintel.control.pairing import PEER_WAIT_SECONDS, Pairing
from rwintel.control.server import Server, ServerSettings
from rwintel.control.session import EpisodeRecord, EpisodeSettings, Session


def _refused(argv) -> bool:
    try:
        settle(argv)
    except SystemExit:
        return True
    return False


def _session(episodes: int = 1) -> Session:
    return Session(None, None, EpisodeSettings(), arms=[("script", lambda s: None)], episodes=episodes)


def _event(session: Session, payload: dict) -> None:
    session.on_episode(json.dumps(payload).encode("utf-8"))


def test_a_match_against_a_person_is_hosted_without_a_cutoff_or_a_deadline():
    arguments, arm, pairing = settle(["--versus"])
    assert arm[0] == "script"
    assert arguments.max_seconds == 0
    assert pairing is not None and pairing.peer_wait == 0 and pairing.port == 5123
    host = _session()
    host.instance = 0
    assert pairing.instruction(host) == {"host": True, "port": 5123, "peerWait": 0, "name": "host-0"}


def test_a_paired_match_holds_its_room_for_a_while_and_either_wait_can_be_given():
    arguments, _, pairing = settle(["--paired", "--instances", "2"])
    assert arguments.max_seconds == DEFAULT_MAX_SECONDS and pairing.peer_wait == PEER_WAIT_SECONDS
    _, _, pairing = settle(["--paired", "--instances", "2", "--peer-wait", "20"])
    assert pairing.peer_wait == 20
    arguments, _, pairing = settle(["--versus", "--peer-wait", "300", "--max-seconds", "1800", "--match-port", "5200"])
    assert (pairing.peer_wait, pairing.port, arguments.max_seconds) == (300, 5200, 1800)
    arguments, _, pairing = settle([])
    assert pairing is None and arguments.max_seconds == DEFAULT_MAX_SECONDS


def test_what_plays_is_named_like_an_arm_of_a_comparison():
    assert settle(["--policy", "script:baseline"])[1][0] == "script:baseline"
    assert settle(["--versus", "--policy", "ops-nearest"])[1][0] == "ops-nearest"
    for argv in (["--policy", "nonsense"], ["--policy", "operations:no/such/file.pt"],
                 ["--versus", "--paired"], ["--versus", "--instances", "2"], ["--paired", "--peer-wait", "-1"]):
        assert _refused(argv), argv


def test_a_host_nobody_joined_fails_its_session_and_counts_no_episode():
    session = _session()
    _event(session, {"event": "failed", "episode": 1, "reason": "no other player joined within 20s"})
    assert session.failure == "no other player joined within 20s"
    assert session.records == []


def test_a_match_everybody_left_is_recorded_as_such():
    session = _session()
    _event(session, {"event": "finished", "episode": 1, "seconds": 400, "winner": -1, "aliveTeams": 2, "team": 0,
                     "peerLeft": True})
    record = session.records[0]
    assert record.peer_left and record.as_dict()["peer_left"] is True
    assert against_a_person(record) == "the player left"


def test_how_a_match_against_a_person_ended_is_said_from_the_side_that_played_it():
    def record(**fields) -> EpisodeRecord:
        return EpisodeRecord(**{"episode": 1, "seconds": 600, "winner": -1, "alive_teams": 2, "timeout": False,
                                "team": 0, **fields})

    assert against_a_person(record(winner=0, alive_teams=1)) == "the policy won"
    assert against_a_person(record(winner=1, alive_teams=1)) == "the player won"
    assert against_a_person(record(timeout=True)) == "cut off undecided"
    assert against_a_person(record(winner=0, alive_teams=1, peer_left=True)) == "the player left"


def test_the_replay_a_match_was_recorded_to_is_named_in_its_record_and_kept_out_of_the_instance():
    import tempfile

    from rwintel.control.__main__ import keep_replays

    with tempfile.TemporaryDirectory() as folder:
        name = "Lake (2p) [v1.15] (30 Sep 2026 05.10.55).replay"
        os.makedirs(os.path.join(folder, "instance", "replays"))
        with open(os.path.join(folder, "instance", "replays", name), "wb") as handle:
            handle.write(b"recorded")
        session = _session()
        session.directory = os.path.join(folder, "instance")
        _event(session, {"event": "finished", "episode": 1, "seconds": 874, "winner": 1, "aliveTeams": 1, "team": 0,
                         "replay": name})
        assert session.records[0].replay == {"file": name}
        assert session.records[0].as_dict()["replay"] == {"file": name}
        kept = keep_replays([session], into=os.path.join(folder, "kept"))
        assert kept == [os.path.join(folder, "kept", name)]
        with open(kept[0], "rb") as handle:
            assert handle.read() == b"recorded"
    assert EpisodeRecord(episode=1, seconds=1, winner=-1, alive_teams=2, timeout=False).replay == {}


def test_a_failed_run_releases_every_link_and_names_the_instance():
    server = Server(ServerSettings(arms=[("script", lambda s: None)]))
    ours, theirs = socket.socketpair()
    try:
        session = Session(ours, None, EpisodeSettings(), arms=server.arms)
        session.instance = 0
        session.failure = "no other player joined within 20s"
        server.sessions.append(session)
        server.abandon()
        theirs.settimeout(5)
        assert theirs.recv(1) == b""
        assert server.failures == [(0, "no other player joined within 20s")]
        assert server._finished() is False and server._done.is_set()
    finally:
        ours.close()
        theirs.close()


def test_the_pairing_hosts_on_the_first_instance_by_default():
    pairing = Pairing()
    host, other = _session(), _session()
    host.instance, other.instance = 0, 1
    assert pairing.hosts(host) and not pairing.hosts(other)
    assert pairing.instruction(host)["peerWait"] == PEER_WAIT_SECONDS


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
