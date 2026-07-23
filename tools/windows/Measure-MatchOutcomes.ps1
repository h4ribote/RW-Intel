<#
.SYNOPSIS
Runs a matchup for many episodes and reports the spread of the outcomes.

.DESCRIPTION
Two policies cannot be compared on the same match, because the same seed and the same settings do not reproduce a match.
Every comparison is therefore a difference between two distributions, and the only honest way to plan one is to know how wide those distributions are.
This runs one matchup repeatedly and reports what a single episode actually tells you: the win rate and its standard error, the spread of match lengths, and how many episodes a given difference in win rate would take to detect.

Both sides are built-in AI players and the local player only watches. Leaving the local player in the match makes every episode end the same way, and the local player cannot be handed to the AI either, because the built-in AI is a separate player class that only the host's add-AI path creates.

Slots alternate teams, so two opponents land on opposite teams and play one against one. The map therefore needs a starting position for the local player's slot as well as for both of them, which means at least three: use a four player map.

Measurements are only meaningful on an otherwise idle machine, and the parallelism should be the one the throughput document settled on.

.PARAMETER Difficulty
Difficulty for both AI players unless Levels overrides it.

.PARAMETER Levels
Difficulty per contestant, for example 1,0. An uneven matchup is what gives a known effect size to size a comparison against; the room itself can only apply one setting to every AI it adds.

.PARAMETER Episodes
Episodes each instance runs. Total episodes are this times Count.

.EXAMPLE
.\Measure-MatchOutcomes.ps1 -Count 8 -Episodes 5 -Map Islands -Difficulty 1
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 64)]
    [int]$Count = 8,

    [string]$Map = 'Islands',

    [ValidateRange(-2, 3)]
    [int]$Difficulty = 1,

    [int[]]$Levels = @(),

    [int]$Opponents = 2,

    [int]$Episodes = 6,

    [double]$Speed = 10,

    [int]$MaxSeconds = 1200,

    [int]$TimeoutMinutes = 40,

    [string]$MasterPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\rw'),

    [string]$InstanceRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\instances'),

    [string]$AgentJar = (Join-Path (Split-Path $PSScriptRoot -Parent) 'probe-agent\rwprobe.jar'),

    [string]$LogRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\outcome-logs')
)

$ErrorActionPreference = 'Stop'

$java = Join-Path $MasterPath 'jvm64\bin\java.exe'
if (-not (Test-Path $java)) { throw "No bundled JVM at $java" }
if (-not (Test-Path $AgentJar)) { throw "No probe agent at $AgentJar. Build it with probe-agent\build.ps1" }

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Get-ChildItem $LogRoot -Filter '*.out' -ErrorAction SilentlyContinue | Remove-Item -Force

$processes = @()
for ($i = 0; $i -lt $Count; $i++) {
    $name = '{0:d2}' -f $i
    $dir = Join-Path $InstanceRoot $name
    if (-not (Test-Path $dir)) { throw "Instance directory missing: $dir" }

    # A different seed per instance only varies the map's own randomisation; the match does not reproduce from a seed anyway.
    $agentOptions = "interval=15000,speed=$Speed,match=$Map,ai=$Opponents,difficulty=$Difficulty," +
        "contestants=2,episodes=$Episodes,seed=$(1000 + $i),maxSeconds=$MaxSeconds"
    if ($Levels.Count -gt 0) { $agentOptions += ",levels=$($Levels -join ';')" }

    $arguments = @(
        '-Xmx800M',
        '-Dfile.encoding=UTF-8',
        "-Djava.library.path=$dir",
        "-javaagent:$AgentJar=$agentOptions",
        '-cp', "$MasterPath\game-lib.jar;$MasterPath\libs\*",
        'com.corrodinggames.rts.java.Main',
        '-nodisplay', '-nosound', '-nomusic', '-nomods'
    )

    $processes += Start-Process -FilePath $java -WorkingDirectory $dir -ArgumentList $arguments `
        -RedirectStandardOutput (Join-Path $LogRoot "$name.out") `
        -RedirectStandardError (Join-Path $LogRoot "$name.err") `
        -PassThru -WindowStyle Hidden
}

$wanted = $Count * $Episodes
$matchup = if ($Levels.Count -gt 0) { "difficulties $($Levels -join ' vs ')" } else { "both at difficulty $Difficulty" }
Write-Host "running $Count instance(s) x $Episodes episode(s) = $wanted episodes, two AI players, $matchup, on $Map"

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    $done = (Select-String -Path (Join-Path $LogRoot '*.out') -Pattern '^\[rw-probe\] result:' -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "  $done/$wanted episodes finished"
    if ($done -ge $wanted) { break }
}

foreach ($process in $processes) {
    try { $process.Kill() } catch { }
}
Start-Sleep -Seconds 3

$results = @()
foreach ($line in Select-String -Path (Join-Path $LogRoot '*.out') -Pattern '^\[rw-probe\] result:' -ErrorAction SilentlyContinue) {
    if ($line.Line -match 'seconds=(\d+) frames=(\d+) winner=(-?\d+) aliveTeams=(\d+) timeout=(\w+) units=(\d+)') {
        $entry = [PSCustomObject]@{
            Seconds = [int]$Matches[1]
            Frames  = [int]$Matches[2]
            Winner  = [int]$Matches[3]
            Timeout = [bool]::Parse($Matches[5])
            Units   = [int]$Matches[6]
            Value0  = 0
            Value1  = 0
            Edge    = [double]::NaN
        }
        if ($line.Line -match 'team0Value=(\d+)') { $entry.Value0 = [int]$Matches[1] }
        if ($line.Line -match 'team1Value=(\d+)') { $entry.Value1 = [int]$Matches[1] }
        $total = $entry.Value0 + $entry.Value1
        # The share of surviving value one side holds, which is the shape a score of an undecided position has to take.
        if ($total -gt 0) { $entry.Edge = ($entry.Value0 - $entry.Value1) / $total }
        $results += $entry
    }
}

if ($results.Count -eq 0) { throw "No episode finished. See $LogRoot" }

$decided = $results | Where-Object { -not $_.Timeout -and $_.Winner -ge 0 }
# Team 0 against team 1 is the whole matchup once the local player is watching, so one side's share is the outcome.
$wins = ($decided | Where-Object { $_.Winner -eq 0 } | Measure-Object).Count
$timeouts = ($results | Where-Object { $_.Timeout } | Measure-Object).Count

$lengths = $results | ForEach-Object { $_.Seconds }
$lengthStats = $lengths | Measure-Object -Average -Minimum -Maximum
$mean = $lengthStats.Average
$sd = if ($lengths.Count -gt 1) {
    [math]::Sqrt((($lengths | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Sum).Sum) / ($lengths.Count - 1))
} else { 0 }

$edges = $results | Where-Object { -not [double]::IsNaN($_.Edge) } | ForEach-Object { $_.Edge }
$edgeMean = if ($edges.Count -gt 0) { ($edges | Measure-Object -Average).Average } else { [double]::NaN }
$edgeSd = if ($edges.Count -gt 1) {
    [math]::Sqrt((($edges | ForEach-Object { ($_ - $edgeMean) * ($_ - $edgeMean) } | Measure-Object -Sum).Sum) / ($edges.Count - 1))
} else { [double]::NaN }

$rate = if ($decided.Count -gt 0) { $wins / $decided.Count } else { [double]::NaN }
# A win rate is a Bernoulli mean, so its standard error is the usual sqrt(p(1-p)/n).
$standardError = if ($decided.Count -gt 0) { [math]::Sqrt($rate * (1 - $rate) / $decided.Count) } else { [double]::NaN }

Write-Host ''
[PSCustomObject]@{
    Episodes        = $results.Count
    Decided         = $decided.Count
    Timeouts        = $timeouts
    WinRateTeam0    = [math]::Round($rate, 3)
    WinRateStdError = [math]::Round($standardError, 3)
    LengthMeanSec   = [math]::Round($mean, 0)
    LengthSdSec     = [math]::Round($sd, 0)
    LengthMinSec    = $lengthStats.Minimum
    LengthMaxSec    = $lengthStats.Maximum
    UnitsMean       = [math]::Round((($results | ForEach-Object { $_.Units } | Measure-Object -Average).Average), 0)
    EdgeMean        = [math]::Round($edgeMean, 3)
    EdgeSd          = [math]::Round($edgeSd, 3)
} | Format-List

# Episodes needed for a two-sided test at 5% with 80% power, comparing two win rates around the observed one.
# The usual normal approximation: n per arm = (1.96+0.84)^2 * 2p(1-p) / delta^2.
Write-Host 'episodes per arm needed, at the 5% level with 80% power'
Write-Host '  on the win rate, if matches decided at all:'
$p = if ([double]::IsNaN($rate)) { 0.5 } else { $rate }
foreach ($delta in 0.10, 0.20, 0.30) {
    $n = [math]::Ceiling(7.849 * 2 * $p * (1 - $p) / ($delta * $delta))
    $hours = [math]::Round(2 * $n * $mean / $Speed / $Count / 3600, 2)
    Write-Host ("    {0,4:P0} difference: {1,6} per arm, {2} hours for both arms at {3} instances" -f $delta, $n, $hours, $Count)
}
if (-not [double]::IsNaN($edgeSd)) {
    Write-Host '  on the surviving value share, which every episode yields:'
    foreach ($delta in 0.05, 0.10, 0.20) {
        $n = [math]::Ceiling(7.849 * 2 * $edgeSd * $edgeSd / ($delta * $delta))
        $hours = [math]::Round(2 * $n * $mean / $Speed / $Count / 3600, 2)
        Write-Host ("    {0,5:N2} difference: {1,6} per arm, {2} hours for both arms at {3} instances" -f $delta, $n, $hours, $Count)
    }
}
