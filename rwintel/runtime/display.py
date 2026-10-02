"""The X display the game processes draw on.

`-nodisplay` does not make the game headless: it still creates a 10x10 OpenGL window, and loading a map needs that window's context. A machine without a display therefore needs a virtual one. A launcher starts a private Xvfb for its processes and stops it when they are stopped, so no run depends on a display another run left behind; one display serves every process of a run.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import time
from typing import IO, Optional

#: The screen of the virtual display. The game's window is 10x10, so anything will do.
SCREEN = "640x480x24"

#: How long Xvfb is given to report the display it took.
STARTUP_SECONDS = 15.0


class DisplayError(RuntimeError):
    pass


class VirtualDisplay:
    """A private Xvfb, or an existing display when one is asked for by name.

    `requested` is None or "xvfb" for a private Xvfb, "inherit" for the display in the environment, or a display name such as ":0".
    """

    def __init__(self, requested: Optional[str] = None, log_path: Optional[str] = None) -> None:
        self.requested = requested or "xvfb"
        self.log_path = log_path
        self.name: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._log: Optional[IO[bytes]] = None

    def __enter__(self) -> "VirtualDisplay":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self) -> str:
        if self.requested == "inherit":
            name = os.environ.get("DISPLAY")
            if not name:
                raise DisplayError("--display inherit was given but DISPLAY is not set")
            self.name = name
            return name
        if self.requested != "xvfb":
            self.name = self.requested
            return self.name
        executable = shutil.which("Xvfb")
        if executable is None:
            raise DisplayError("Xvfb is not installed; install the xvfb package, or pass --display inherit to use the display in the environment")
        read_end, write_end = os.pipe()
        self._log = open(self.log_path, "ab") if self.log_path else open(os.devnull, "wb")
        try:
            # -displayfd makes the server pick a free display number itself and write it to the pipe once it is accepting connections, which is both the number and the readiness signal.
            self._process = subprocess.Popen(
                [executable, "-displayfd", str(write_end), "-screen", "0", SCREEN, "-nolisten", "tcp", "-noreset"],
                pass_fds=(write_end,), stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            os.close(write_end)
        try:
            number = self._read_number(read_end)
        finally:
            os.close(read_end)
        self.name = f":{number}"
        return self.name

    def _read_number(self, fd: int) -> str:
        deadline = time.monotonic() + STARTUP_SECONDS
        text = b""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                chunk = os.read(fd, 64)
                if not chunk:
                    break
                text += chunk
                if b"\n" in text:
                    return text.split(b"\n", 1)[0].decode().strip()
            if self._process is not None and self._process.poll() is not None:
                break
        self.stop()
        where = f"; see {self.log_path}" if self.log_path else ""
        raise DisplayError(f"Xvfb did not report a display within {STARTUP_SECONDS:.0f}s{where}")

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._log is not None:
            self._log.close()
            self._log = None
