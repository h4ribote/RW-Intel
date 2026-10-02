"""Holds the launcher to what the Linux release needs, and its reports to what the probe agent writes.

None of this starts the game. The command line and environment are checked as data, instance directories are built against a stand-in install, and the process handling is exercised on ordinary commands; what the game does with all of it is checked by running it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel import paths
from rwintel.control.server import open_listener
from rwintel.runtime import build, control, instances, ports, reports
from rwintel.runtime.__main__ import DEFAULT_RESTARTS, parser
from rwintel.runtime.display import VirtualDisplay
from rwintel.runtime.game import GAME_ARGUMENTS, MAIN_CLASS, GameInstall, agent_options, command, environment
from rwintel.runtime.launch import CRASH_MARKER, Fleet, Spec, early_failures


@contextlib.contextmanager
def _working_area():
    saved = os.environ.get("RWINTEL_LOCAL")
    with tempfile.TemporaryDirectory() as area:
        os.environ["RWINTEL_LOCAL"] = area
        try:
            yield area
        finally:
            if saved is None:
                os.environ.pop("RWINTEL_LOCAL", None)
            else:
                os.environ["RWINTEL_LOCAL"] = saved


def _fake_install(root: str) -> GameInstall:
    for name in instances.LINKED:
        os.makedirs(os.path.join(root, name))
    os.makedirs(os.path.join(root, "jvm-linux", "bin"))
    java = os.path.join(root, "jvm-linux", "bin", "java")
    with open(java, "w") as handle:
        handle.write("#!/bin/sh\n")
    os.chmod(java, os.stat(java).st_mode | stat.S_IXUSR)
    open(os.path.join(root, "game-lib.jar"), "w").close()
    with open(os.path.join(root, "assets", "marker"), "w") as handle:
        handle.write("install")
    return GameInstall(root)


# ---- the command line ------------------------------------------------------------------------------

def test_command_uses_the_bundled_linux_jvm_and_a_colon_separated_classpath():
    install = GameInstall("/opt/rw")
    argv = command(install, "/repo/agent/rwagent.jar", "host=127.0.0.1,port=8642", heap="1G")
    assert argv[0] == "/opt/rw/jvm-linux/bin/java"
    assert "-Xmx1G" in argv
    assert "-Djava.library.path=/opt/rw" in argv
    assert "-javaagent:/repo/agent/rwagent.jar=host=127.0.0.1,port=8642" in argv
    assert argv[argv.index("-cp") + 1] == "/opt/rw/game-lib.jar:/opt/rw/libs/*"
    assert argv[-len(GAME_ARGUMENTS) - 1:] == [MAIN_CLASS, *GAME_ARGUMENTS]
    assert "-nomods" in GAME_ARGUMENTS and "-nodisplay" in GAME_ARGUMENTS


def test_environment_puts_the_install_first_on_the_linker_path_and_sets_the_display():
    env = environment(GameInstall("/opt/rw"), ":7", base={"LD_LIBRARY_PATH": "/usr/local/lib", "HOME": "/root"},
                      extra={"LP_NUM_THREADS": "1"})
    assert env["LD_LIBRARY_PATH"] == "/opt/rw:/usr/local/lib"
    assert env["DISPLAY"] == ":7"
    assert env["HOME"] == "/root" and env["LP_NUM_THREADS"] == "1"
    assert environment(GameInstall("/opt/rw"), ":0", base={})["LD_LIBRARY_PATH"] == "/opt/rw"


def test_agent_options_keep_their_order_and_append_what_was_typed():
    assert agent_options({"a": 1, "b": "x"}) == "a=1,b=x"
    assert agent_options({"a": 1}, " obs=true,catalog=true, ") == "a=1,obs=true,catalog=true"
    assert agent_options({}, "dump=3") == "dump=3"


def test_an_incomplete_install_is_refused_by_name():
    with tempfile.TemporaryDirectory() as root:
        try:
            GameInstall(root).require()
        except FileNotFoundError as error:
            assert "game-lib.jar" in str(error)
        else:
            raise AssertionError("an empty directory was accepted as an install")


# ---- instance directories -------------------------------------------------------------------------

def test_instances_link_the_read_only_trees_and_own_the_written_ones():
    with _working_area(), tempfile.TemporaryDirectory() as root:
        install = _fake_install(root)
        prepared = instances.prepare(2, install)
        assert (prepared.created, prepared.reused) == (2, 0)
        for index in (0, 1):
            directory = paths.instance(index)
            for name in instances.LINKED:
                assert os.path.islink(os.path.join(directory, name))
                assert os.readlink(os.path.join(directory, name)) == os.path.join(root, name)
            for name in instances.OWN:
                path = os.path.join(directory, name)
                assert os.path.isdir(path) and not os.path.islink(path)
        again = instances.prepare(3, install)
        assert (again.created, again.reused, again.relinked) == (1, 2, 0)


def test_instances_follow_the_install_they_are_prepared_against():
    with _working_area(), tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        instances.prepare(1, _fake_install(first))
        moved = instances.prepare(1, _fake_install(second))
        assert moved.relinked == 1
        assert os.readlink(os.path.join(paths.instance(0), "assets")) == os.path.join(second, "assets")


def test_recreating_an_instance_never_reaches_into_the_install():
    with _working_area(), tempfile.TemporaryDirectory() as root:
        install = _fake_install(root)
        instances.prepare(1, install)
        with open(os.path.join(paths.instance(0), "preferences.ini"), "w") as handle:
            handle.write("[settings]\nleftover:1\n")
        instances.prepare(1, install, force=True)
        with open(os.path.join(paths.instance(0), "preferences.ini")) as handle:
            assert handle.read() == "[settings]\nsendReports:false\n"
        with open(os.path.join(root, "assets", "marker")) as handle:
            assert handle.read() == "install"


def test_every_instance_is_kept_from_uploading_crash_reports():
    with tempfile.TemporaryDirectory() as directory:
        preferences = os.path.join(directory, "preferences.ini")
        instances.disable_crash_reports(directory)
        with open(preferences) as handle:
            assert handle.read() == "[settings]\nsendReports:false\n"
        with open(preferences, "w") as handle:
            handle.write("[settings]\naiDifficulty:1\nsendReports:true\nzoom:2\n")
        instances.disable_crash_reports(directory)
        with open(preferences) as handle:
            assert handle.read() == "[settings]\naiDifficulty:1\nzoom:2\nsendReports:false\n"
        settled = os.path.getmtime(preferences)
        os.utime(preferences, (settled - 100, settled - 100))
        instances.disable_crash_reports(directory)
        assert os.path.getmtime(preferences) == settled - 100
    with _working_area(), tempfile.TemporaryDirectory() as root:
        instances.prepare(2, _fake_install(root))
        for index in (0, 1):
            with open(os.path.join(paths.instance(index), "preferences.ini")) as handle:
                assert "sendReports:false" in handle.read().splitlines()


def test_missing_instances_are_named():
    with _working_area():
        try:
            instances.require([0, 5])
        except FileNotFoundError as error:
            assert "05" in str(error)
        else:
            raise AssertionError("missing instance directories were accepted")


# ---- what the probe agent writes -------------------------------------------------------------------

def test_rate_lines_are_read_and_the_last_of_each_instance_is_summarised():
    lines = ["[rw-probe] fps=164.8 speed=0.57x step=3.5ms objects=249 gameTime=2.9s",
             "noise",
             "[rw-probe] fps=299.5 speed=10.00x step=33.4ms objects=332 gameTime=52.9s"]
    samples = reports.rate_samples(lines)
    assert [s.fps for s in samples] == [164.8, 299.5]
    assert samples[-1].objects == 332
    summary = reports.summarise_rates([samples[-1], reports.RateSample(250.0, 8.0, 40.0)], 10.0)
    assert summary.instances == 2
    assert summary.speed_minimum == 8.0 and abs(summary.speed_aggregate - 18.0) < 1e-9
    assert summary.fps_total == 549.5 and summary.step_maximum_ms == 40.0
    assert reports.summarise_rates([], 10.0) is None


def test_result_lines_become_outcomes_with_the_surviving_value_share():
    lines = [
        "[rw-probe] result: episode=1 seed=1000 seconds=400 frames=9000 winner=0 aliveTeams=1 timeout=false units=80 team0Units=40 team0Value=30000 team1Units=0 team1Value=0",
        "[rw-probe] result: episode=2 seed=1000 seconds=1200 frames=30000 winner=-1 aliveTeams=2 timeout=true units=100 team0Units=50 team0Value=10000 team1Units=50 team1Value=30000",
        "[rw-probe] match: episode 2 finished",
    ]
    found = reports.outcomes(lines)
    assert reports.count_results(lines) == 2
    assert [o.decided for o in found] == [True, False]
    assert found[0].edge == 1.0 and found[1].edge == -0.5
    summary = reports.summarise_outcomes(found)
    assert (summary.episodes, summary.decided, summary.timeouts) == (2, 1, 1)
    assert summary.win_rate == 1.0 and summary.win_rate_error == 0.0
    assert summary.length.mean == 800 and (summary.length_minimum, summary.length_maximum) == (400, 1200)
    text = "\n".join(reports.outcome_lines(summary, rate=80.0, instances=8))
    assert "surviving value share" in text and "per arm" in text and "80 game seconds per second" in text


def test_estimates_divide_by_the_game_time_the_run_actually_played():
    found = [reports.Outcome(seconds=400, frames=0, winner=0, timeout=False, units=0),
             reports.Outcome(seconds=1200, frames=0, winner=-1, timeout=True, units=0)]
    assert reports.game_seconds_per_second(found, 20.0) == 80.0
    assert reports.game_seconds_per_second(found, 0.0) == 0.0
    assert reports.hours_for_both_arms(100, 360.0, 80.0) == 2 * 100 * 360.0 / 80.0 / 3600.0
    assert reports.hours_for_both_arms(100, 360.0, 0.0) == float("inf")


# ---- ports ------------------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_port_someone_listens_on_is_reported_and_a_free_one_is_not():
    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        problem = ports.check_free(port)
        assert problem is not None and f"TCP port {port}" in problem
    finally:
        listener.close()
    assert ports.check_free(port) is None


def test_a_control_port_someone_listens_on_is_reported_before_the_control_side_starts():
    listener = open_listener("127.0.0.1", _free_port())
    port = listener.getsockname()[1]
    try:
        problem = ports.check_listener_free("127.0.0.1", port)
        assert problem is not None and f"TCP port {port}" in problem
    finally:
        listener.close()
    assert ports.check_listener_free("127.0.0.1", port) is None


def test_a_second_control_process_cannot_share_the_port():
    port = _free_port()
    first = open_listener("127.0.0.1", port)
    try:
        try:
            second = open_listener("127.0.0.1", port)
        except OSError:
            pass
        else:
            second.close()
            raise AssertionError("two control processes bound the same port")
    finally:
        first.close()


def test_a_control_process_restarts_on_a_port_whose_connections_are_still_closing():
    port = _free_port()
    listener = open_listener("127.0.0.1", port)
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = listener.accept()
    # The side that closes first holds TIME_WAIT, and here that is the control process's side, which is the case a restart meets.
    accepted.close()
    listener.close()
    time.sleep(0.1)
    client.close()
    again = open_listener("127.0.0.1", port)
    again.close()


# ---- building ---------------------------------------------------------------------------------------

def test_an_agent_is_stale_until_its_jar_is_newer_than_every_source():
    with tempfile.TemporaryDirectory() as directory:
        target = build.Target("t", directory, "*.java", "t.jar")
        for name in ("A.java", "manifest.txt"):
            open(os.path.join(directory, name), "w").close()
        assert build.stale(target)
        open(target.jar, "w").close()
        later = time.time() + 5
        os.utime(target.jar, (later, later))
        assert not build.stale(target)
        os.utime(os.path.join(directory, "A.java"), (later + 5, later + 5))
        assert build.stale(target)


def test_every_agent_is_built_for_the_bundled_java_8():
    assert build.JAVA_RELEASE == "8"
    assert set(build.TARGETS) == {"agent", "probe", "lab"}
    for target in build.TARGETS.values():
        assert target.source_files(), target.name
        assert os.path.isfile(target.manifest), target.name
    # The experiment agent reaches the engine through the control agent's own reflection.
    assert os.path.join(build.AGENT_DIRECTORY, "Engine.java") in build.TARGETS["lab"].source_files()


def test_every_agent_carries_the_frame_layer_and_compiles_against_the_install_slick():
    frame = os.path.join(build.FRAME_DIRECTORY, "Frame.java")
    for target in build.TARGETS.values():
        assert frame in target.source_files(), target.name
    with tempfile.TemporaryDirectory() as root:
        install = _fake_install(root)
        try:
            build.classpath(install)
        except build.BuildError as error:
            assert "slick.jar" in str(error)
        else:
            raise AssertionError("an install without Slick was accepted")
        os.makedirs(os.path.join(root, "libs"))
        open(os.path.join(root, "libs", "slick.jar"), "w").close()
        assert build.classpath(install) == os.path.join(root, "libs", "slick.jar")


def test_a_shared_source_makes_an_agent_stale():
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as shared:
        target = build.Target("t", directory, "*.java", "t.jar", (shared,))
        for path in (os.path.join(directory, "A.java"), os.path.join(directory, "manifest.txt"), os.path.join(shared, "S.java")):
            open(path, "w").close()
        open(target.jar, "w").close()
        later = time.time() + 5
        os.utime(target.jar, (later, later))
        assert not build.stale(target)
        os.utime(os.path.join(shared, "S.java"), (later + 5, later + 5))
        assert build.stale(target)


# ---- processes and display --------------------------------------------------------------------------

def test_a_fleet_stops_its_processes_and_reports_those_that_died():
    with tempfile.TemporaryDirectory() as logs:
        with Fleet(logs) as fleet:
            long = fleet.start(Spec("00", logs, ["sleep", "60"]))
            fleet.start(Spec("01", logs, [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"]))
            assert fleet.wait(0.5, poll=0.1) == "time"
            failures = early_failures(fleet)
            assert failures == ["01 exited with 3: boom"]
            assert fleet.alive() == [long]
        assert long.process.poll() is not None
        assert os.path.isfile(os.path.join(logs, "00.out")) and os.path.isfile(os.path.join(logs, "01.err"))


def test_a_fleet_that_has_ended_by_itself_says_so():
    with tempfile.TemporaryDirectory() as logs, Fleet(logs) as fleet:
        fleet.start(Spec("00", logs, ["true"]))
        assert fleet.wait(0, poll=0.05) == "exited"


def _game(flag: str = "") -> list:
    """A stand-in for a game that, like the engine, writes the crash marker and stays alive. Given a flag file, it crashes only while the flag does not exist yet, and creates it."""
    marker = CRASH_MARKER.decode()
    script = ("import os, sys, time\n"
              f"flag = {flag!r}\n"
              "crash = not flag or not os.path.exists(flag)\n"
              "if flag: open(flag, 'w').close()\n"
              "sys.stdout.write('loading\\n')\n"
              f"if crash: sys.stdout.write({marker!r} + '\\n')\n"
              "sys.stdout.flush()\n"
              "time.sleep(60)\n")
    return [sys.executable, "-c", script]


def _first_report(fleet: Fleet, running, seconds: float = 10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        reason = fleet.crashed(running)
        if reason is not None:
            return reason
        time.sleep(0.05)
    return None


def test_a_game_that_writes_the_crash_marker_is_reported_once_while_it_stays_alive():
    with tempfile.TemporaryDirectory() as logs, Fleet(logs) as fleet:
        running = fleet.start(Spec("00", logs, _game()))
        assert _first_report(fleet, running) == "crashed"
        assert running.process.poll() is None
        assert fleet.crashed(running) is None


def test_a_healthy_game_is_never_reported_and_an_exited_one_always_is():
    with tempfile.TemporaryDirectory() as logs, Fleet(logs) as fleet:
        healthy = fleet.start(Spec("00", logs, [sys.executable, "-c", "print('playing', flush=True); import time; time.sleep(60)"]))
        gone = fleet.start(Spec("01", logs, [sys.executable, "-c", "import sys; sys.exit(3)"]))
        assert _first_report(fleet, gone) == "exited with 3"
        assert fleet.crashed(gone) == "exited with 3"
        assert _first_report(fleet, healthy, seconds=1.0) is None


def test_a_restarted_game_keeps_its_earlier_logs_and_is_judged_on_its_new_ones():
    with tempfile.TemporaryDirectory() as logs, Fleet(logs) as fleet:
        flag = os.path.join(logs, "crashed-once")
        first = fleet.start(Spec("00", logs, _game(flag)))
        assert _first_report(fleet, first) == "crashed"
        second = fleet.restart(first)
        assert first.process.poll() is not None
        assert fleet.running == [second] and second.restarts == 1
        with open(os.path.join(logs, "00.out.1"), "rb") as handle:
            assert CRASH_MARKER in handle.read()
        assert _first_report(fleet, second, seconds=1.0) is None
        assert second.process.poll() is None


def test_run_restarts_a_crashed_game_up_to_its_limit_and_then_says_it_gave_up(capsys):
    import types

    from rwintel.runtime.__main__ import _restart_crashed

    def marked(path: str) -> bool:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with open(path, "rb") as handle:
                if CRASH_MARKER in handle.read():
                    return True
            time.sleep(0.05)
        return False

    with tempfile.TemporaryDirectory() as logs, Fleet(logs) as fleet:
        fleet.start(Spec("00", logs, _game()))
        run = types.SimpleNamespace(fleet=fleet)
        given_up = set()
        assert marked(os.path.join(logs, "00.out"))
        _restart_crashed(run, 1, given_up)
        assert fleet.running[0].restarts == 1 and given_up == set()
        assert marked(os.path.join(logs, "00.out"))
        _restart_crashed(run, 1, given_up)
        _restart_crashed(run, 1, given_up)
        assert given_up == {"00"} and fleet.running[0].restarts == 1
        said = capsys.readouterr().out.splitlines()
        assert said[0].startswith("00 crashed; restarting it (1 of 1)")
        assert said[1:] == ["00 crashed; not restarting it after 1 restart(s)"]


def test_restarts_default_to_a_few_and_can_be_turned_off():
    assert parser().parse_args(["run", "--", "control"]).restarts == DEFAULT_RESTARTS > 0
    assert parser().parse_args(["run", "--restarts", "0", "--", "control"]).restarts == 0


def _hello(running: bool) -> bytes:
    return json.dumps({"instance": 0, "build": "28", "unitTypes": [], "running": running}).encode("utf-8")


def test_the_catalogue_carries_what_each_type_can_do_as_the_engine_samples_it():
    """A transport reports a range without having a weapon, so being armed is the sample's own answer and not a range; what a type carries, makes and how fast it goes arrive with it."""
    from rwintel.control.session import EpisodeSettings, Session

    class _Connection:
        def sendall(self, data):
            pass

    types = [
        {"name": "c_tank", "lookup": "tank", "price": 350, "tech": 1, "building": False, "builder": False,
         "movement": "LAND", "canAttack": True, "range": 130.0, "speed": 66.0, "capacity": -1, "slots": 1,
         "upgradable": False, "menu": []},
        {"name": "hovercraft", "lookup": "hovercraft", "price": 600, "tech": 1, "building": False, "builder": False,
         "movement": "HOVER", "canAttack": False, "range": 30.0, "speed": 54.0, "capacity": 4, "slots": 1,
         "upgradable": False, "menu": [], "carries": [0]},
        {"name": "seaFactory", "lookup": "seaFactory", "price": 1000, "tech": 1, "building": True, "builder": False,
         "movement": "NONE", "canAttack": False, "range": 0.0, "speed": 0.0, "capacity": -1, "slots": 1,
         "upgradable": True, "menu": [1]},
    ]
    session = Session(_Connection(), None, EpisodeSettings(), [("arm", lambda s: None)], episodes=0)
    session.on_hello(json.dumps({"instance": 0, "build": "28", "unitTypes": types}).encode("utf-8"))
    tank, hovercraft, factory = session.types
    assert tank.armed and not hovercraft.armed and hovercraft.range == 30.0
    assert hovercraft.transport and hovercraft.carries == (0,) and not tank.transport
    assert factory.menu == (1,) and factory.upgradable and hovercraft.speed == 54.0


def test_a_game_restarted_mid_episode_has_that_episode_closed_unrecorded_and_played_again():
    from rwintel.control.session import EpisodeSettings, Session

    class _Policy:
        closed = False

        def close(self):
            self.closed = True

    session = Session(None, None, EpisodeSettings(), arms=[("script", lambda s: None)], episodes=2)
    started = []
    session.start_episode = lambda: started.append(len(session.records))
    interrupted = _Policy()
    session.policy = interrupted
    session.history = [{"second": 30.0, "standing": []}]
    session.on_hello(_hello(running=False))

    assert interrupted.closed and session.policy is None
    assert session.records == [] and session.history == []
    assert started == [0]


def test_episodes_are_numbered_by_the_session_across_a_restarted_game():
    from rwintel.control.session import EpisodeSettings, Session

    session = Session(None, None, EpisodeSettings(), arms=[("script", lambda s: None)], episodes=3)
    session.start_episode = lambda: None
    session.on_episode(json.dumps({"event": "finished", "episode": 1, "seconds": 60}).encode("utf-8"))
    session.policy = object()
    session.on_hello(_hello(running=False))
    # The restarted game counts its episodes from one again.
    session.on_episode(json.dumps({"event": "finished", "episode": 1, "seconds": 60}).encode("utf-8"))
    assert [record.episode for record in session.records] == [1, 2]


def test_a_game_that_rejoins_its_running_episode_keeps_it():
    from rwintel.control.session import EpisodeSettings, Session

    session = Session(None, None, EpisodeSettings(), arms=[("script", lambda s: None)], episodes=2)
    resumed = []
    session._resume = lambda map_path: resumed.append(map_path)
    session.start_episode = lambda: resumed.append("started")
    session.policy = object()
    session.on_hello(_hello(running=True))
    assert resumed == [""]


def test_a_private_virtual_display_is_started_and_stopped():
    if shutil.which("Xvfb") is None:
        print("skipped: Xvfb is not installed")
        return
    with tempfile.TemporaryDirectory() as logs:
        display = VirtualDisplay(None, os.path.join(logs, "xvfb.log"))
        name = display.start()
        try:
            assert name.startswith(":") and name[1:].isdigit()
            assert os.path.exists(f"/tmp/.X11-unix/X{name[1:]}")
        finally:
            display.stop()
    assert VirtualDisplay(":5").start() == ":5"


def test_the_software_renderer_is_held_to_one_thread_unless_told_otherwise():
    from rwintel.runtime.__main__ import DEFAULT_RENDER_THREADS, _extra_environment

    assert DEFAULT_RENDER_THREADS == 1
    assert _extra_environment(parser().parse_args(["probe"])) == {"LP_NUM_THREADS": "1"}
    assert _extra_environment(parser().parse_args(["agents", "--render-threads", "4"])) == {"LP_NUM_THREADS": "4"}
    assert _extra_environment(parser().parse_args(["outcomes", "--render-threads", "0"])) == {}


def test_games_run_undrawn_on_the_fixed_clock_unless_told_otherwise():
    from rwintel.runtime.__main__ import _agent_options, frame_options

    fixed = frame_options(parser().parse_args(["probe"]))
    assert fixed == {"draw": "false", "clock": "fixed", "step": 25, "fps": 300, "speed": 0.0}
    paced = frame_options(parser().parse_args(["outcomes", "--speed", "20", "--step-ms", "50"]))
    assert (paced["clock"], paced["step"], paced["speed"]) == ("fixed", 50, 20.0)
    wall = frame_options(parser().parse_args(["agents", "--clock", "wall", "--draw"]))
    assert wall == {"draw": "true", "clock": "wall", "step": 25, "fps": 300, "speed": 10.0}
    options = _agent_options(parser().parse_args(["agents", "--fps", "600", "--clock", "wall", "--speed", "20"]), 3).split(",")
    assert {"instance=3", "clock=wall", "fps=600", "speed=20.0", "draw=false"} <= set(options)
    for argv in (["probe", "--clock", "turbo"], ["probe", "--step-ms", "0"], ["run", "--fps", "0"]):
        try:
            parser().parse_args(argv)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {argv}")


def test_a_paired_match_runs_on_the_wall_clock():
    from rwintel.runtime.__main__ import frame_options, paired_clock

    pair = parser().parse_args(["pair"])
    assert paired_clock(pair) == "" and frame_options(pair)["clock"] == "wall" and frame_options(pair)["speed"] == 10.0
    assert paired_clock(parser().parse_args(["pair", "--clock", "wall"])) == ""
    assert "--clock" in paired_clock(parser().parse_args(["run", "--clock", "fixed"]))


def test_a_match_against_a_person_runs_on_the_wall_clock_at_real_speed():
    from rwintel.runtime.__main__ import frame_options, versus_clock

    run = parser().parse_args(["run"])
    assert versus_clock(run) == "" and frame_options(run)["clock"] == "wall" and frame_options(run)["speed"] == 1.0
    faster = parser().parse_args(["run", "--speed", "2"])
    assert versus_clock(faster) == "" and frame_options(faster)["speed"] == 2.0
    assert "--clock" in versus_clock(parser().parse_args(["run", "--clock", "fixed"]))


def test_a_playback_runs_on_the_replay_clock_and_only_a_playback_does():
    from rwintel.runtime.__main__ import describe_clock, frame_options, replay_clock, replay_problem

    run = parser().parse_args(["run"])
    assert replay_clock(run) == "" and frame_options(run)["clock"] == "replay" and frame_options(run)["speed"] == 0.0
    assert "replay clock" in describe_clock(run)
    assert replay_clock(parser().parse_args(["run", "--clock", "replay"])) == ""
    for clock in ("fixed", "wall"):
        assert "--clock" in replay_clock(parser().parse_args(["run", "--clock", clock]))
    assert replay_problem(parser().parse_args(["run", "--clock", "replay"])) != ""
    assert replay_problem(parser().parse_args(["run"])) == ""


def test_the_replay_control_side_plays_and_leaves_reading_files_to_be_run_directly():
    wanted = control.prepare(["replay", "play", "a.replay", "b.replay"], 2)
    assert wanted.replay and wanted.module == "rwintel.replay" and not wanted.networked
    assert wanted.arguments[-2:] == ["--instances", "2"]
    for words in (["replay", "inspect", "a.replay"], ["replay", "verify", "a.replay", "--log", "x"], ["replay"]):
        try:
            control.prepare(words, 1)
        except control.ControlError:
            continue
        raise AssertionError(f"accepted {words}")
    assert not control.prepare(["control"], 1).replay


def test_a_hosted_match_is_joined_at_the_loopback_first_and_then_at_this_machine():
    assert ports.join_addresses(5123, ["10.0.0.5"]) == ["localhost:5123", "10.0.0.5:5123"]
    assert ports.join_addresses(6000, []) == ["localhost:6000"]
    found = ports.local_addresses()
    assert len(found) == len(set(found)) and not any(address.startswith("127.") for address in found)


def test_the_control_periods_have_to_be_whole_numbers_of_fixed_steps():
    from rwintel.runtime.__main__ import period_problem

    assert period_problem(parser().parse_args(["agents"])) == ""
    assert period_problem(parser().parse_args(["run", "--step-ms", "40"])) == ""
    assert "--tactical-ms" in period_problem(parser().parse_args(["run", "--step-ms", "33"]))
    assert "--operational-ms" in period_problem(parser().parse_args(["pair", "--operational-ms", "2010"]))
    assert period_problem(parser().parse_args(["agents", "--clock", "wall", "--step-ms", "33"])) == ""


# ---- the control side under the launcher ------------------------------------------------------------

def test_the_control_side_is_given_the_instance_count_and_tells_the_games_where_to_dial():
    wanted = control.prepare(["--", "learn", "tactics", "--port=9001", "--host", "0.0.0.0"], 4)
    assert wanted.name == "learn" and wanted.module == "rwintel.learn"
    assert wanted.arguments[-2:] == ["--instances", "4"]
    assert (wanted.port, wanted.connect_host, wanted.instances) == (9001, "127.0.0.1", 4)
    assert wanted.argv[:3] == [sys.executable, "-m", "rwintel.learn"]
    plain = control.prepare(["control", "--episodes", "2", "--instances", "2"], 2)
    assert plain.arguments.count("--instances") == 1 and plain.port == control.DEFAULT_PORT
    assert plain.connect_host == "127.0.0.1" and not plain.paired
    paired = control.prepare(["control", "--paired", "--match-port", "5200"], 2)
    assert paired.paired and paired.networked and not paired.versus and paired.match_port == 5200
    versus = control.prepare(["control", "--versus"], 1)
    assert versus.versus and versus.networked and not versus.paired and versus.match_port == control.DEFAULT_MATCH_PORT
    assert not plain.networked


def test_a_control_side_the_launcher_cannot_serve_is_refused_before_anything_starts():
    for words, count in ((["control", "--instances", "3"], 2),
                         (["control", "--paired"], 3),
                         (["control", "--versus"], 2),
                         (["control", "--versus", "--paired"], 1),
                         (["eval", "--versus"], 1),
                         (["learn", "clone", "--save", "x.pt"], 1),
                         (["learn", "--layer", "tactics", "clone"], 1),
                         (["learn", "tactics", "--port"], 1),
                         (["learn", "tactics", "--instances", "two"], 1),
                         (["train"], 1),
                         (["--"], 1)):
        try:
            control.prepare(words, count)
        except control.ControlError:
            continue
        raise AssertionError(f"accepted {words} with {count} game(s)")
    assert control.prepare(["learn", "--layer", "tactics", "collect"], 1).name == "learn"


#: Stands in for a control process: listens, then waits to be interrupted and says how it ended through its exit code.
_FAKE_CONTROL = """
import socket, sys, time
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", int(sys.argv[1])))
listener.listen(1)
try:
    time.sleep(60)
except KeyboardInterrupt:
    sys.exit(3)
"""


def _fake(port: int, script: str = _FAKE_CONTROL) -> control.ControlProcess:
    wanted = control.prepare(["control", "--port", str(port)], 1)
    return control.ControlProcess(wanted, argv=[sys.executable, "-c", script, str(port)])


def test_a_control_process_is_waited_for_and_stopped_the_way_ctrl_c_stops_it():
    port = _free_port()
    with _fake(port) as process:
        process.start()
        assert process.wait_listening(seconds=10)
        assert process.stop(grace=10) == 3
    assert process.poll() == 3


def test_a_control_process_stops_on_sigint_even_when_the_launcher_ignores_it():
    import signal

    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        port = _free_port()
        with _fake(port) as process:
            process.start()
            assert process.wait_listening(seconds=10)
            assert process.stop(grace=10) == 3
    finally:
        signal.signal(signal.SIGINT, previous)


def test_a_control_process_that_exits_before_listening_is_reported_as_such():
    with _fake(_free_port(), script="import sys; sys.exit(5)") as process:
        process.start()
        assert not process.wait_listening(seconds=10)
        assert process.poll() == 5


def test_the_command_line_refuses_what_the_tools_cannot_honour():
    for argv in (["probe", "--seconds", "5"], ["instances"], ["agents", "--count", "65"], ["pair", "--match-port", "80"]):
        try:
            parser().parse_args(argv)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {argv}")
    parsed = parser().parse_args(["outcomes", "--levels", "1,0"])
    assert parsed.count == max(1, min(64, os.cpu_count() or 1)) and parsed.levels == "1,0"


def test_an_experiment_log_condenses_to_the_moments_each_unit_changed():
    from rwintel.runtime import lablog

    lines = [
        "noise from the game\n",
        "[rw-lab] t=2.0 do loadUp hc1 t1\n",
        "[rw-lab] track t=1.0 hc1 #13 hovercraft (3250,1770) hp=450/450 load=0/4 orders=0\n",
        "[rw-lab] track t=2.0 hc1 #13 hovercraft (3260,1771) hp=440/450 load=0/4 orders=0\n",
        "[rw-lab] track t=3.0 hc1 #13 hovercraft (3270,1772) hp=440/450 load=1/4 orders=0\n",
        "[rw-lab] track t=4.0 hc1 #13 hovercraft (3500,1772) hp=440/450 load=1/4 orders=0\n",
        "[rw-lab] track t=4.0 t1 #20 c_tank (3270,1772) hp=210/210 inside=#13 orders=0\n",
        "[rw-lab] catalog: tank|custom.j|LAND\n",
        "[rw-lab] near: #191 tree (3350,1390) hp=100/100\n",
    ]
    kept = list(lablog.timeline(lines))
    # A small move and a scratch of damage are not news; a load, a long move and a new unit are.
    assert [line.split()[0] for line in kept if not line.startswith("    ")] == ["1.0", "3.0", "4.0", "4.0"]
    assert kept[0] == "    t=2.0 do loadUp hc1 t1"
    assert [line for line in lablog.timeline(lines, {"t1"}) if not line.startswith("    ")][0].split()[1] == "t1"


# ---- a measurement split between control processes ----------------------------------------------------

def test_a_split_run_numbers_its_games_as_the_unsplit_run_would_and_gives_each_part_its_port_and_journal():
    wanted = control.prepare(["eval", "--port=9001", "--arm", "script", "--record", "x.jsonl", "--card-share", "0.8",
                              "--fit-out", "w.json"], 7)
    shards = control.split(wanted, 3, "logs")
    assert [(s.first, s.count) for s in shards] == [(0, 3), (3, 2), (5, 2)]
    assert [s.command.port for s in shards] == [9001, 9002, 9003]
    for index, shard in enumerate(shards):
        words = shard.command.arguments
        assert control._options(words, "--port") == [str(9001 + index)]
        assert control._options(words, "--instances") == [str(shard.count)] and shard.command.instances == shard.count
        assert control._options(words, "--record") == [os.path.join("logs", f"journal-{index}.jsonl")] == [shard.journal]
        assert control._options(words, "--card-share") == ["0.266667"]
        assert "--fit-out" not in words and control._options(words, "--arm") == ["script"]
    assert control.split(control.prepare(["learn", "duel", "--card-share", "0.8"], 4), 2, "d")[1].command.arguments[-2:] \
        == ["--card-share", "0.4"]
    assert control.whole(wanted) == control.Shard(wanted, 0, 7, "x.jsonl")


def test_only_a_measurement_can_be_split_and_never_into_more_parts_than_games():
    for words, count, controls in ((["learn", "tactics"], 4, 2),
                                   (["learn", "--layer", "tactics", "collect"], 4, 2),
                                   (["control", "--episodes", "2"], 4, 2),
                                   (["control", "--paired"], 2, 2),
                                   (["eval", "--from", "a.jsonl"], 4, 2),
                                   (["eval"], 2, 3),
                                   (["eval"], 2, 0)):
        try:
            control.check_split(control.prepare(words, count), controls)
        except control.ControlError:
            continue
        raise AssertionError(f"split {words} with {count} game(s) {controls} ways")
    control.check_split(control.prepare(["learn", "tactics"], 4), 1)
    control.check_split(control.prepare(["learn", "duel", "--load", "a.pt"], 4), 4)


def test_a_split_run_is_reported_together_with_the_options_that_shape_the_report():
    wanted = control.prepare(["eval", "--arm", "script", "--arm", "operations:m/a.pt", "--fit-out", "w.json",
                              "--weights", "opening", "--verbose", "--seed", "5"], 4)
    argv = control.report_argv(wanted, ["j0", "j1"])
    assert argv[:3] == [sys.executable, "-m", "rwintel.eval"]
    assert argv[3:] == ["--from", "j0", "j1", "--arm", "script", "--arm", "operations:m/a.pt", "--weights", "opening",
                        "--fit-out", "w.json", "--verbose"]
    duel = control.prepare(["learn", "duel", "--load", "a.pt"], 4)
    assert control.report_argv(duel, ["j0"])[1:] == ["-m", "rwintel.learn", "duel", "--from", "j0"]
    with _working_area() as area:
        assert control.record_path(duel, "s") == os.path.join(area, "episodes", "duel-s.jsonl")
    assert control.record_path(wanted, "s").endswith(os.path.join("episodes", "eval-s.jsonl"))
    assert control.record_path(control.prepare(["eval", "--record", "r.jsonl"], 2), "s") == "r.jsonl"


def test_a_game_announces_its_place_in_the_run_and_dials_its_part():
    from rwintel.runtime.__main__ import _agent_options

    arguments = parser().parse_args(["run", "--count", "4", "--", "eval"])
    arguments.control_host, arguments.port = "127.0.0.1", 8642
    assert "port=8642" in _agent_options(arguments, 0) and "instance=0" in _agent_options(arguments, 0)
    second = _agent_options(arguments, 5, 8643)
    assert "port=8643" in second and "instance=5" in second
    assert parser().parse_args(["run", "--", "eval"]).controls == 1


def test_a_fleet_stops_only_the_processes_named():
    with tempfile.TemporaryDirectory() as logs, Fleet(logs) as fleet:
        first = fleet.start(Spec("00", logs, ["sleep", "60"]))
        second = fleet.start(Spec("01", logs, ["sleep", "60"]))
        fleet.stop(["00"])
        assert first.process.poll() is not None and second.process.poll() is None


def test_the_parts_journals_are_appended_in_order_after_what_the_journal_held():
    with tempfile.TemporaryDirectory() as area:
        parts = [os.path.join(area, f"journal-{i}.jsonl") for i in range(3)]
        with open(parts[0], "w") as handle:
            handle.write('{"instance": 0, "seconds": 10}\n{"instance": 1, "seconds": 20}\n')
        with open(parts[2], "w") as handle:
            handle.write('{"instance": 2, "seconds": 30}')
        target = os.path.join(area, "out", "pooled.jsonl")
        os.makedirs(os.path.dirname(target))
        with open(target, "w") as handle:
            handle.write('{"instance": 9, "seconds": 1}\n')
        start = control.journal_size(target)
        assert control.pool(parts, target) == 3
        assert [e["instance"] for e in control.journal_entries(target)] == [9, 0, 1, 2]
        assert [e["instance"] for e in control.journal_entries(target, start)] == [0, 1, 2]
        assert list(control.journal_entries(os.path.join(area, "missing.jsonl"))) == []


def test_throughput_is_the_game_seconds_journalled_over_the_wall_clock():
    found = reports.throughput([{"seconds": 300}, {"seconds": 600}, {}], 45.0)
    assert (found.episodes, found.game_seconds, found.rate) == (3, 900, 20.0)
    assert found.line() == "throughput: 3 episode(s), 900 game seconds in 45s, 20.0 game seconds per second"
    assert reports.throughput([], 0.0).rate == 0.0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
