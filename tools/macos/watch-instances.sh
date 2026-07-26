#!/bin/bash
# Puts a lost game instance back.
#
# A game process can die outright — twice in one night of continuous running, once on a corrupted native heap and once on a JVM fatal error inside libc — and the control process then waits for its missing episodes for ever, because a run is over when every instance has run its own. On an unattended queue that is a run lost and everything behind it stopped.
#
# This watches a run's log for an instance that has stopped reporting while others go on, and starts one fresh agent under that instance's number. The session it dials into is the one that was left open, so it takes up the remaining episodes and the run finishes and saves.
#
#   ./tools/macos/watch-instances.sh local/strategy-s1.log local/strategy-s1.done
#
# The second argument is the file whose appearance means the run is over; the watch stops there and takes its rescue agents away, because an agent left standing would dial into the NEXT run as that instance and put two agents on one session.
set -uo pipefail
source "$(dirname "$0")/_common.sh"

log=${1:?say which log to watch}
until_file=${2:?say which file means the run is over}
quiet=${3:-240}     # seconds without a line from an instance before it counts as lost
rescued=""

require_docker
require_image
require_control_agent
prepare_game

cleanup() {
    for name in $rescued; do docker rm -f "$name" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT

while [ ! -f "$until_file" ]; do
    sleep 30
    [ -f "$log" ] || continue
    now=$(date +%s)
    for instance in 0 1 2 3 4 5 6 7; do
        name="rw-rescue-$instance"
        echo "$rescued" | grep -q "$name" && continue
        # An instance that has said it is done is not lost.
        grep -q "instance $instance has run its episodes" "$log" && continue
        # When did this instance last say anything, against when anything did?
        seen=$(grep -n "instance $instance " "$log" | tail -1 | cut -d: -f1)
        [ -n "$seen" ] || continue
        total=$(grep -c "" "$log")
        # Lines are cheap and time stamps are not on every line, so quietness is measured in lines other instances have written since this one last did.
        [ $((total - seen)) -gt 200 ] || continue
        echo "[watch] instance $instance has been quiet for $((total - seen)) lines; putting one back" >&2
        docker rm -f "$name" >/dev/null 2>&1 || true
        docker run -d --name "$name" --platform linux/amd64 \
            --add-host=host.docker.internal:host-gateway \
            -v "$RW_GAME_DIR":/game:ro -v "$RW_ROOT/agent":/agent:ro \
            -v "$RW_MACOS/rw-run.sh":/rw-run.sh:ro -v "$RW_ROOT/local/agent-logs-rescue":/tmp/rw-logs \
            -e RW_INSTANCES=1 -e RW_AGENT=/agent/rwagent.jar \
            -e RW_AGENT_OPTS="host=host.docker.internal,port=8642,instance=$instance,speed=10,tactical=200,operational=2000" \
            -e RW_DURATION=0 "$RW_IMAGE" bash /rw-run.sh >/dev/null
        rescued="$rescued $name"
    done
done
