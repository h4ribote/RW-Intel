#!/bin/bash
# Couples a host-side control or learning command with the game container it drives.
#
# The learning and duelling runs are two processes that have to be alive together: the rwintel command listens on the host and holds the policies, and the game instances live in the amd64 Linux container and dial in to it. On Windows those are two console windows a person starts by hand; on macOS one is a host process and the other is a container, and nothing ties their lifetimes together. This does. It starts the host command, brings the container up against it, waits for the host command to finish its episodes, and stops the container the moment it does, so no game is left running to fight the next run for the cores.
#
# The host command is given after a literal --, verbatim, so this file never has to know which rwintel subcommand it is running or which of its flags mean what. It must bind 0.0.0.0, since the container reaches it as a separate host:
#
#   ./tools/macos/learn-run.sh --count 6 --speed 10 -- \
#       .venv/bin/python -m rwintel.learn collect --layer tactics --host 0.0.0.0 --instances 6 --episodes 4 --record local/teacher.jsonl
#
# The container is started detached and named after this shell's pid, so a run cleans up only its own game instances and two runs can never stop each other's.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

count=1; speed=10; tactical=200; operational=2000
while [ $# -gt 0 ]; do
    case "$1" in
        --count) count=$2; shift 2 ;;
        --speed) speed=$2; shift 2 ;;
        --tactical) tactical=$2; shift 2 ;;
        --operational) operational=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { echo "no host command given after --" >&2; exit 2; }

require_docker
require_image
require_control_agent
prepare_game

opts="host=host.docker.internal,port=8642,instance=%I,speed=$speed,tactical=$tactical,operational=$operational"
logdir="$RW_ROOT/local/agent-logs"
rm -rf "$logdir"; mkdir -p "$logdir"

name="rw-agents-$$"
host_pid=""
cleanup() {
    docker rm -f "$name" >/dev/null 2>&1 || true
    [ -n "$host_pid" ] && kill "$host_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Job control, and it is not a convenience. A non-interactive shell sets SIGINT and SIGQUIT to be IGNORED in the processes it starts in the background, and that disposition is inherited: without this, the host command cannot be interrupted at all. Measured the hard way — a run whose game instance had died sat waiting for episodes that would never come, took no notice of an interrupt, and could only be killed, which loses the parameters because the save is past the point where the run would have returned. With job control on, the background job keeps the default disposition and an interrupt reaches it, so it unwinds through its own shutdown and saves what it has.
set -m

# The host command comes up first and binds its port, then the container's agents dial in. They retry until it answers, so the order is not strict, but starting the listener first spares the first connection its back-off.
echo "[learn-run] host: $*" >&2
"$@" &
host_pid=$!

# An interrupt or a termination of this script is passed on to the run rather than only tearing the container down under it. Without this, stopping the launcher would leave the host command listening for agents that no longer exist.
forward() {
    [ -n "$host_pid" ] && kill -INT "$host_pid" 2>/dev/null
}
trap forward INT TERM

echo "[learn-run] starting $count game instance(s) as container $name" >&2
docker run -d --name "$name" --platform linux/amd64 \
    --add-host=host.docker.internal:host-gateway \
    -v "$RW_GAME_DIR":/game:ro \
    -v "$RW_ROOT/agent":/agent:ro \
    -v "$RW_MACOS/rw-run.sh":/rw-run.sh:ro \
    -v "$logdir":/tmp/rw-logs \
    -e RW_INSTANCES="$count" \
    -e RW_AGENT=/agent/rwagent.jar \
    -e RW_AGENT_OPTS="$opts" \
    -e RW_DURATION=0 \
    "$RW_IMAGE" bash /rw-run.sh >/dev/null

# Wait on the host command, not the container: the run is over when the episodes are, and the container only exists to feed them. set -e must not turn the host command's own exit code into this script's early exit before the trap can stop the container, so the wait's status is captured rather than left to trip.
status=0
wait "$host_pid" || status=$?
host_pid=""
echo "[learn-run] host command exited $status; stopping container" >&2
exit "$status"
