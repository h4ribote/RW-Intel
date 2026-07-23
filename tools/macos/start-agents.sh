#!/bin/bash
# Starts game instances under the control agent, which dial in to the control process.
#
# The macOS analogue of Start-RwAgents.ps1. The control process runs natively on the host; the game runs in the amd64 Linux container and reaches the host at host.docker.internal. Start the control process first, and start it listening on 0.0.0.0 rather than 127.0.0.1, or the container's connection is refused:
#
#   python -m rwintel.control --host 0.0.0.0 --instances 2 --episodes 2 --map Lake --max-seconds 300
#   ./tools/macos/start-agents.sh -Count 2 -Speed 10
#
# The agents drive nothing by themselves. Episodes start when the control process says so, which keeps the settings of a run in one place. This runs attached and streams the per-instance logs; stop it with Ctrl-C, or pass -Seconds to stop after a fixed time.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

count=1; speed=10; seconds=0
control_host="host.docker.internal"; port=8642
tactical=200; operational=2000; extra=""

while [ $# -gt 0 ]; do
    case "$1" in
        -Count|--count) count=$2; shift 2 ;;
        -Speed|--speed) speed=$2; shift 2 ;;
        -Seconds|--seconds) seconds=$2; shift 2 ;;
        -ControlHost|--control-host) control_host=$2; shift 2 ;;
        -Port|--port) port=$2; shift 2 ;;
        -TacticalMs|--tactical) tactical=$2; shift 2 ;;
        -OperationalMs|--operational) operational=$2; shift 2 ;;
        -AgentOptions|--agent-options) extra=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

require_docker
require_image
require_control_agent
prepare_game

# The control process numbers its instances from nought, so %I is the identity it knows an instance by.
opts="host=$control_host,port=$port,instance=%I,speed=$speed,tactical=$tactical,operational=$operational"
[ -n "$extra" ] && opts="$opts,$extra"

logdir="$RW_ROOT/local/agent-logs"
rm -rf "$logdir"; mkdir -p "$logdir"

echo "starting $count instance(s) against $control_host:$port, logs in $logdir"
# --add-host maps host.docker.internal on native Linux, where Docker Desktop's automatic mapping is absent; it is harmless on macOS.
docker run --rm --platform linux/amd64 \
    --add-host=host.docker.internal:host-gateway \
    -v "$RW_GAME_DIR":/game:ro \
    -v "$RW_ROOT/agent":/agent:ro \
    -v "$RW_MACOS/rw-run.sh":/rw-run.sh:ro \
    -v "$logdir":/tmp/rw-logs \
    -e RW_INSTANCES="$count" \
    -e RW_AGENT=/agent/rwagent.jar \
    -e RW_AGENT_OPTS="$opts" \
    -e RW_DURATION="$seconds" \
    "$RW_IMAGE" bash /rw-run.sh
