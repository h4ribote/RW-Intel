#!/bin/bash
# Runs a matchup for many episodes and reports the spread of the outcomes.
#
# The macOS analogue of Measure-MatchOutcomes.ps1. Two policies cannot be compared on the same match, because the same seed and settings do not reproduce one, so every comparison is a difference between two distributions and the only honest way to plan one is to know how wide those distributions are. This runs one matchup repeatedly and reports what a single episode tells you: the win rate and its standard error, the spread of match lengths, and how many episodes a given difference in win rate would take to detect.
#
# Both sides are built-in AI players and the local player only watches. Slots alternate teams, so two opponents land on opposite teams and play one against one; the map therefore needs a start position for the local player's slot as well, which means a four-player map.
#
#   ./tools/macos/measure-match-outcomes.sh -Count 8 -Episodes 6 -Map Islands -Difficulty 1
#   ./tools/macos/measure-match-outcomes.sh -Count 8 -Episodes 5 -Map Islands -Levels 1,0
set -euo pipefail
source "$(dirname "$0")/_common.sh"

count=8; map="Islands"; difficulty=1; levels=""; opponents=2
episodes=6; speed=10; maxseconds=1200; timeout_minutes=40

while [ $# -gt 0 ]; do
    case "$1" in
        -Count|--count) count=$2; shift 2 ;;
        -Map|--map) map=$2; shift 2 ;;
        -Difficulty|--difficulty) difficulty=$2; shift 2 ;;
        -Levels|--levels) levels=$2; shift 2 ;;
        -Opponents|--opponents) opponents=$2; shift 2 ;;
        -Episodes|--episodes) episodes=$2; shift 2 ;;
        -Speed|--speed) speed=$2; shift 2 ;;
        -MaxSeconds|--max-seconds) maxseconds=$2; shift 2 ;;
        -TimeoutMinutes|--timeout-minutes) timeout_minutes=$2; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

require_docker
require_image
require_probe_agent
prepare_game

opts="interval=15000,speed=$speed,match=$map,ai=$opponents,difficulty=$difficulty,contestants=2,episodes=$episodes,seed=%S,maxSeconds=$maxseconds"
# An uneven matchup is what gives a known effect size to size a comparison against; the room can only apply one setting to every AI it adds, so a per-contestant list overrides it.
[ -n "$levels" ] && opts="$opts,levels=$(echo "$levels" | tr ',' ';')"

logdir="$RW_ROOT/local/outcome-logs"
rm -rf "$logdir"; mkdir -p "$logdir"

name="rw-measure-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT

wanted=$((count * episodes))
matchup=$([ -n "$levels" ] && echo "difficulties ${levels//,/ vs }" || echo "both at difficulty $difficulty")
echo "running $count instance(s) x $episodes episode(s) = $wanted episodes, two AI players, $matchup, on $map"

docker run -d --name "$name" --platform linux/amd64 \
    -v "$RW_GAME_DIR":/game:ro \
    -v "$RW_ROOT/tools/probe-agent":/agent:ro \
    -v "$RW_MACOS/rw-run.sh":/rw-run.sh:ro \
    -v "$logdir":/tmp/rw-logs \
    -e RW_INSTANCES="$count" \
    -e RW_AGENT=/agent/rwprobe.jar \
    -e RW_AGENT_OPTS="$opts" \
    -e RW_DURATION=0 \
    "$RW_IMAGE" bash /rw-run.sh >/dev/null

deadline=$(($(date +%s) + timeout_minutes * 60))
while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 20
    # grep exits non-zero until the first result appears; the subshell with || true keeps that from tripping set -e under pipefail.
    done_count=$( (grep -h '^\[rw-probe\] result:' "$logdir"/*.log 2>/dev/null || true) | wc -l | tr -d ' ')
    echo "  $done_count/$wanted episodes finished"
    [ "$done_count" -ge "$wanted" ] && break
done
cleanup
trap - EXIT

python3 - "$logdir" "$speed" "$count" <<'PY'
import re, sys, glob, os, math
logdir, speed, count = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
line_re = re.compile(r"seconds=(\d+) frames=(\d+) winner=(-?\d+) aliveTeams=(\d+) timeout=(\w+) units=(\d+)")
results = []
for path in glob.glob(os.path.join(logdir, "*.log")):
    for line in open(path, errors="replace"):
        if not line.startswith("[rw-probe] result:"):
            continue
        m = line_re.search(line)
        if not m:
            continue
        v0 = int((re.search(r"team0Value=(\d+)", line) or [0, "0"])[1]) if "team0Value=" in line else 0
        v1 = int((re.search(r"team1Value=(\d+)", line) or [0, "0"])[1]) if "team1Value=" in line else 0
        total = v0 + v1
        results.append(dict(
            seconds=int(m.group(1)), winner=int(m.group(3)),
            timeout=(m.group(5).lower() == "true"), units=int(m.group(6)),
            edge=(v0 - v1) / total if total > 0 else None))
if not results:
    sys.exit(f"No episode finished. See {logdir}")

decided = [r for r in results if not r["timeout"] and r["winner"] >= 0]
wins = sum(1 for r in decided if r["winner"] == 0)
timeouts = sum(1 for r in results if r["timeout"])
lengths = [r["seconds"] for r in results]
mean = sum(lengths) / len(lengths)
sd = math.sqrt(sum((x - mean) ** 2 for x in lengths) / (len(lengths) - 1)) if len(lengths) > 1 else 0.0
edges = [r["edge"] for r in results if r["edge"] is not None]
emean = sum(edges) / len(edges) if edges else float("nan")
esd = math.sqrt(sum((x - emean) ** 2 for x in edges) / (len(edges) - 1)) if len(edges) > 1 else float("nan")
rate = wins / len(decided) if decided else float("nan")
se = math.sqrt(rate * (1 - rate) / len(decided)) if decided else float("nan")

print()
print(f"  episodes         {len(results)}")
print(f"  decided          {len(decided)}")
print(f"  timeouts         {timeouts}")
print(f"  win rate team0   {rate:.3f}")
print(f"  win rate stderr  {se:.3f}")
print(f"  length mean sec  {round(mean)}")
print(f"  length sd sec    {round(sd)}")
print(f"  length min/max   {min(lengths)}/{max(lengths)}")
print(f"  units mean       {round(sum(r['units'] for r in results)/len(results))}")
print(f"  edge mean        {emean:.3f}")
print(f"  edge sd          {esd:.3f}")

# Episodes per arm for a two-sided test at 5% with 80% power, comparing two rates around the observed one: n = (1.96+0.84)^2 * 2p(1-p) / delta^2.
print()
print("episodes per arm needed, at the 5% level with 80% power")
print("  on the win rate, if matches decided at all:")
p = 0.5 if math.isnan(rate) else rate
for delta in (0.10, 0.20, 0.30):
    n = math.ceil(7.849 * 2 * p * (1 - p) / (delta * delta))
    hours = round(2 * n * mean / speed / count / 3600, 2)
    print(f"    {delta:4.0%} difference: {n:6d} per arm, {hours} hours for both arms at {count} instances")
if not math.isnan(esd):
    print("  on the surviving value share, which every episode yields:")
    for delta in (0.05, 0.10, 0.20):
        n = math.ceil(7.849 * 2 * esd * esd / (delta * delta))
        hours = round(2 * n * mean / speed / count / 3600, 2)
        print(f"    {delta:5.2f} difference: {n:6d} per arm, {hours} hours for both arms at {count} instances")
PY
