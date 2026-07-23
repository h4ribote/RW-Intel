#!/bin/bash
# Runs one or more instances under the probe agent and reports the simulation rate each sustained.
#
# The macOS analogue of Start-RwProbe.ps1. The game runs in the amd64 Linux container; everything else is the same measurement. Measurements are only meaningful on an otherwise idle machine, and the first half minute of every run is JIT warmup.
#
#   ./tools/macos/start-probe.sh -Count 8 -Speed 10 -Seconds 90
#   ./tools/macos/start-probe.sh -Count 1 -Speed 10 -Seconds 200 -Map Lake -Difficulty 1
#   ./tools/macos/start-probe.sh -Count 1 -Speed 10 -Seconds 200 -Map Lake -AgentOptions 'obs=true,catalog=true'
#
# The engine advances game time by the frame delta times the speed multiple, and the delta comes from real elapsed time, so the reported speed is the wall-clock multiple. Watch the step size it prints: 1000 times the speed over the frame rate is how much game time each step covers, which is the fidelity cost of running fast.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

count=1; speed=10; seconds=90; interval=15000
map=""; opponents=1; difficulty=1; episodes=20; maxseconds=0; extra=""

while [ $# -gt 0 ]; do
    case "$1" in
        -Count|--count) count=$2; shift 2 ;;
        -Speed|--speed) speed=$2; shift 2 ;;
        -Seconds|--seconds) seconds=$2; shift 2 ;;
        -IntervalMs|--interval) interval=$2; shift 2 ;;
        -Map|--map) map=$2; shift 2 ;;
        -Opponents|--opponents) opponents=$2; shift 2 ;;
        -Difficulty|--difficulty) difficulty=$2; shift 2 ;;
        -Episodes|--episodes) episodes=$2; shift 2 ;;
        -MaxSeconds|--max-seconds) maxseconds=$2; shift 2 ;;
        -AgentOptions|--agent-options) extra=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

require_docker
require_image
require_probe_agent
prepare_game

opts="interval=$interval,speed=$speed"
if [ -n "$map" ]; then
    # Each instance gets its own seed so that concurrent runs are not all the same match.
    opts="$opts,match=$map,ai=$opponents,difficulty=$difficulty,episodes=$episodes,seed=%S,maxSeconds=$maxseconds"
fi
[ -n "$extra" ] && opts="$opts,$extra"

logdir="$RW_ROOT/local/probe-logs"
rm -rf "$logdir"; mkdir -p "$logdir"

echo "running $count instance(s) at speed $speed for ${seconds}s"
docker run --rm --platform linux/amd64 \
    -v "$RW_GAME_DIR":/game:ro \
    -v "$RW_ROOT/tools/probe-agent":/agent:ro \
    -v "$RW_MACOS/rw-run.sh":/rw-run.sh:ro \
    -v "$logdir":/tmp/rw-logs \
    -e RW_INSTANCES="$count" \
    -e RW_AGENT=/agent/rwprobe.jar \
    -e RW_AGENT_OPTS="$opts" \
    -e RW_DURATION="$seconds" \
    "$RW_IMAGE" bash /rw-run.sh

python3 - "$logdir" "$speed" <<'PY'
import re, sys, glob, os, math
logdir, requested = sys.argv[1], float(sys.argv[2])
pat = re.compile(r"fps=([\d.]+) speed=([\d.]+)x step=([\d.]+)ms")
speeds, rates, steps = [], [], []
for path in sorted(glob.glob(os.path.join(logdir, "*.log"))):
    last = None
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if m:
            last = m
    if last:
        rates.append(float(last.group(1)))
        speeds.append(float(last.group(2)))
        steps.append(float(last.group(3)))
    else:
        print(f"  {os.path.basename(path)}: no measurement (see the log)")
if not speeds:
    sys.exit("No instance reported a measurement.")
n = len(speeds)
print()
print(f"  instances        {n}")
print(f"  requested speed  {requested:g}")
print(f"  speed average    {sum(speeds)/n:.2f}x")
print(f"  speed minimum    {min(speeds):.2f}x")
print(f"  speed aggregate  {sum(speeds):.1f}x")
print(f"  fps average      {round(sum(rates)/n)}")
print(f"  fps total        {round(sum(rates))}")
print(f"  step average     {round(sum(steps)/n)}ms")
print(f"  step maximum     {round(max(steps))}ms")
PY
