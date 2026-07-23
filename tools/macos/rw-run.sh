#!/bin/bash
# Launches one or more Rusted Warfare instances inside the amd64 Linux container.
#
# This is the in-container half of the macOS tooling. It is driven entirely by environment variables so that the probe, the control agent and the built-in-AI measurement all share one launcher. The host scripts under tools/macos set those variables and mount this file in; nothing here is macOS specific, so the same launcher runs unchanged on a native x86-64 Linux host.
#
# The engine always creates an OpenGL context, even under -nodisplay, so a display is required. LWJGL enumerates display modes by running the xrandr utility, and Xvfb only answers that with the RANDR extension enabled; without either the mode list comes back empty and the display initialiser throws before a context is ever made. LD_LIBRARY_PATH carries the dependants of librocketConnector.so, which the loader resolves relative to the current directory the same way the Windows loader resolves the dependants of rocketConnector64.dll.
set -u

: "${RW_GAME:=/game}"           # the mounted game install (the Linux distribution, with jvm-linux and the natives)
: "${RW_INSTANCES:=1}"          # how many game processes to start in this container
: "${RW_XVFB:=1024x768x24}"     # virtual screen geometry
: "${RW_AGENT:=}"               # path to a -javaagent jar in the container, or empty for none
: "${RW_AGENT_OPTS:=}"          # agent option string; %I expands to the instance index, %S to 1000 plus the index
: "${RW_GAME_ARGS:=-nodisplay -nosound -nomusic -nomods -nologfile}"
: "${RW_HEAP:=800M}"            # -Xmx for each instance
: "${RW_DURATION:=0}"           # seconds to run before stopping; 0 waits for the instances to exit on their own
: "${RW_LOGDIR:=/tmp/rw-logs}"  # where per-instance stdout is written, named 00.log, 01.log and so on

Xvfb :99 -screen 0 "$RW_XVFB" +extension RANDR +extension GLX >/tmp/xvfb.log 2>&1 &
export DISPLAY=:99
for _ in $(seq 40); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.2; done

mkdir -p "$RW_LOGDIR"
java_bin="$RW_GAME/jvm-linux/bin/java"

pids=()
for i in $(seq 0 $((RW_INSTANCES - 1))); do
    name=$(printf '%02d' "$i")
    dir="/rw/$name"
    mkdir -p "$dir"
    (
        cd "$dir" || exit 1
        # Read-only trees are shared with the master copy; the game writes preferences, saves and cache into its own current directory, and resolves its natives relative to it as well.
        for d in assets font res mods; do ln -sfn "$RW_GAME/$d" "$d"; done
        for so in "$RW_GAME"/*.so "$RW_GAME"/*.so.1; do [ -e "$so" ] && ln -sfn "$so" "$(basename "$so")"; done
        mkdir -p saves cache replays
    )

    opts=${RW_AGENT_OPTS//%I/$i}
    opts=${opts//%S/$((1000 + i))}
    agent_flag=()
    [ -n "$RW_AGENT" ] && agent_flag=(-javaagent:"$RW_AGENT=$opts")

    (
        cd "$dir" || exit 1
        LD_LIBRARY_PATH=. "$java_bin" -Xmx"$RW_HEAP" -Dfile.encoding=UTF-8 -Djava.library.path="$dir" \
            "${agent_flag[@]}" \
            -cp "$RW_GAME/game-lib.jar:$RW_GAME/libs/*" com.corrodinggames.rts.java.Main \
            $RW_GAME_ARGS
    ) >"$RW_LOGDIR/$name.log" 2>&1 &
    pids+=($!)
done

echo "[rw-run] started $RW_INSTANCES instance(s); logs in $RW_LOGDIR" >&2

# Wait only on the instance processes, never with a bare wait: Xvfb is a background child of this shell and never exits on its own, so a bare wait would block the container forever.
if [ "$RW_DURATION" -gt 0 ]; then
    sleep "$RW_DURATION"
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
    wait "${pids[@]}" 2>/dev/null || true
else
    wait "${pids[@]}"
fi
