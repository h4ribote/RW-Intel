"""Holds the working area to one layout under local/, independent of the directory a command is run from."""

from __future__ import annotations

import contextlib
import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rwintel import paths
from rwintel.data import AssetPaths
from rwintel.eval.journal import default_path


@contextlib.contextmanager
def _environment(**values):
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_defaults_are_under_the_repository_whatever_the_current_directory():
    with _environment(RWINTEL_LOCAL=None, RWINTEL_GAME=None), tempfile.TemporaryDirectory() as elsewhere:
        before = os.getcwd()
        os.chdir(elsewhere)
        try:
            local = os.path.join(paths.REPOSITORY, "local")
            assert paths.local_root() == local
            assert paths.game() == os.path.join(local, paths.GAME_DIRECTORY)
            assert paths.instance(3) == os.path.join(local, "instances", "03")
            assert paths.logs("probe") == os.path.join(local, "logs", "probe")
            assert os.path.dirname(default_path("duel")) == os.path.join(local, "episodes")
            assert AssetPaths.default().assets == os.path.join(local, paths.GAME_DIRECTORY, "assets")
        finally:
            os.chdir(before)


def test_environment_moves_the_working_area_and_the_game():
    with tempfile.TemporaryDirectory() as area, tempfile.TemporaryDirectory() as game:
        with _environment(RWINTEL_LOCAL=area, RWINTEL_GAME=game):
            assert paths.local_root() == area
            assert paths.episodes() == os.path.join(area, "episodes")
            assert paths.game() == game
            assert AssetPaths.default().assets == os.path.join(game, "assets")
        with _environment(RWINTEL_LOCAL=area, RWINTEL_GAME=None):
            assert paths.game() == os.path.join(area, paths.GAME_DIRECTORY)


def test_journal_default_names_the_run_and_never_reuses_a_file_across_names():
    first, second = default_path("tactics"), default_path("duel")
    assert os.path.basename(first).startswith("tactics-") and first.endswith(".jsonl")
    assert os.path.basename(second).startswith("duel-")
    assert first != second


def test_each_run_gets_a_log_directory_of_its_own_and_latest_follows_the_newest():
    with tempfile.TemporaryDirectory() as area, _environment(RWINTEL_LOCAL=area):
        first = paths.run_log_directory("agents")
        second = paths.run_log_directory("agents")
        assert first != second and os.path.isdir(first) and os.path.isdir(second)
        latest = os.path.join(paths.logs("agents"), "latest")
        assert os.path.realpath(latest) == os.path.realpath(second)
        log_file = paths.run_log_file("control")
        assert os.path.isfile(log_file) and log_file.endswith(".log")
        assert os.path.realpath(os.path.join(paths.logs("control"), "latest")) == os.path.realpath(log_file)


def test_runs_started_together_in_one_second_never_share_a_log_directory():
    """Several launchers started at once from different processes land on the same second; each has to come away with a directory of its own rather than one of them failing on a name another just took."""
    from concurrent.futures import ThreadPoolExecutor

    with tempfile.TemporaryDirectory() as area, _environment(RWINTEL_LOCAL=area):
        with ThreadPoolExecutor(max_workers=8) as pool:
            directories = list(pool.map(lambda _: paths.run_log_directory("run"), range(16)))
            files = list(pool.map(lambda _: paths.run_log_file("learn"), range(16)))
        assert len(set(directories)) == 16 and all(os.path.isdir(d) for d in directories)
        assert len(set(files)) == 16 and all(os.path.isfile(f) for f in files)


def test_ensure_parent_creates_the_directory_and_returns_the_path():
    with tempfile.TemporaryDirectory() as area:
        target = os.path.join(area, "a", "b", "file.pt")
        assert paths.ensure_parent(target) == target
        assert os.path.isdir(os.path.join(area, "a", "b"))


def test_replacing_puts_the_whole_file_in_place_or_leaves_the_old_one():
    with tempfile.TemporaryDirectory() as area:
        target = os.path.join(area, "models", "net.pt")
        with paths.replacing(target, "wb") as handle:
            handle.write(b"first")
            # Nothing is at the destination until the block has finished.
            assert not os.path.exists(target)
        with open(target, "rb") as handle:
            assert handle.read() == b"first"

        try:
            with paths.replacing(target, "wb") as handle:
                handle.write(b"half of the second")
                raise RuntimeError("the writer failed")
        except RuntimeError:
            pass
        with open(target, "rb") as handle:
            assert handle.read() == b"first"
        assert os.listdir(os.path.dirname(target)) == ["net.pt"]

        accented = "{\"run\": \"r" + chr(0xE9) + "sum" + chr(0xE9) + "\"}"
        with paths.replacing(os.path.join(area, "run.json")) as handle:
            handle.write(accented)
        with open(os.path.join(area, "run.json"), encoding="utf-8") as handle:
            assert handle.read() == accented


def test_replacing_syncs_the_whole_new_file_before_it_takes_the_name():
    from unittest import mock

    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(descriptor):
        status = os.fstat(descriptor)
        events.append(("fsync", stat.S_ISDIR(status.st_mode), status.st_ino, status.st_size))
        real_fsync(descriptor)

    def replace(source, destination):
        events.append(("replace", os.path.basename(destination)))
        real_replace(source, destination)

    with tempfile.TemporaryDirectory() as area:
        target = os.path.join(area, "shard-00000.npz")
        with mock.patch.object(os, "fsync", fsync), mock.patch.object(os, "replace", replace):
            with paths.replacing(target, "wb") as handle:
                handle.write(b"the whole shard")
        written = os.stat(target)
        # The file synced is the one that ends up under the name, with every byte the block wrote already in it.
        assert events[0] == ("fsync", False, written.st_ino, len(b"the whole shard"))
        assert events[1] == ("replace", "shard-00000.npz")
        if os.name == "posix":
            assert events[2][:2] == ("fsync", True) and len(events) == 3

        events.clear()
        with mock.patch.object(os, "fsync", fsync), mock.patch.object(os, "replace", replace):
            try:
                with paths.replacing(target, "wb") as handle:
                    handle.write(b"half")
                    raise RuntimeError("the writer failed")
            except RuntimeError:
                pass
        assert events == []


def test_replacing_leaves_the_permissions_a_plain_open_would():
    if os.name != "posix":
        return
    saved = os.umask(0o027)
    try:
        with tempfile.TemporaryDirectory() as area:
            target = os.path.join(area, "weights.json")
            with paths.replacing(target) as handle:
                handle.write("{}")
            assert os.stat(target).st_mode & 0o777 == 0o640
            # A file that is already there keeps its own mode, as it does when open() truncates it.
            os.chmod(target, 0o604)
            with paths.replacing(target) as handle:
                handle.write("{}")
            assert os.stat(target).st_mode & 0o777 == 0o604
    finally:
        os.umask(saved)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
