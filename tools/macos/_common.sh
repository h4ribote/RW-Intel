#!/bin/bash
# Shared setup for the macOS host scripts: repository paths, defaults, and the checks every launcher needs. Source this from a script in tools/macos; it defines RW_ROOT, RW_MACOS, RW_GAME_DIR and RW_IMAGE and the helper functions below.
#
# The game install defaults to local/RustedWarfare_Linux, which is the Linux distribution that carries the bundled JVM and the natives. That is deliberately not local/rw: local/rw is the macOS copy with the JVM and natives stripped, and the container cannot run without them.

RW_MACOS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RW_ROOT=$(cd "$RW_MACOS/../.." && pwd)

: "${RW_GAME_DIR:=$RW_ROOT/local/RustedWarfare_Linux}"
: "${RW_IMAGE:=rw-linux:latest}"

# The engine binary and the natives arrive without their executable bit when the distribution is unpacked from a zip. The install is mounted read-only into the container, so the bit has to be set on the host, where the tree is writable.
prepare_game() {
    if [ ! -d "$RW_GAME_DIR" ]; then
        echo "no game install at $RW_GAME_DIR; set RW_GAME_DIR to the Linux distribution (the one with jvm-linux and the .so natives)" >&2
        exit 1
    fi
    if [ ! -x "$RW_GAME_DIR/jvm-linux/bin/java" ]; then
        chmod +x "$RW_GAME_DIR/jvm-linux/bin/"* 2>/dev/null || true
        find "$RW_GAME_DIR" -name '*.so' -o -name '*.so.1' -exec chmod +x {} + 2>/dev/null || true
    fi
}

require_docker() {
    command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH" >&2; exit 1; }
    docker info >/dev/null 2>&1 || { echo "docker daemon is not running" >&2; exit 1; }
}

require_image() {
    docker image inspect "$RW_IMAGE" >/dev/null 2>&1 || {
        echo "no image $RW_IMAGE; build it with tools/macos/build-image.sh" >&2
        exit 1
    }
}

require_probe_agent() {
    [ -f "$RW_ROOT/tools/probe-agent/rwprobe.jar" ] || {
        echo "no probe agent at tools/probe-agent/rwprobe.jar; build it with tools/probe-agent/build.sh" >&2
        exit 1
    }
}

require_control_agent() {
    [ -f "$RW_ROOT/agent/rwagent.jar" ] || {
        echo "no control agent at agent/rwagent.jar; build it with agent/build.sh" >&2
        exit 1
    }
}
