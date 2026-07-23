<#
.SYNOPSIS
Runs one or more Rusted Warfare instances under the probe agent and reports the simulation rate.

.DESCRIPTION
Launches the game headlessly (a 10x10 OpenGL window, which is as close to headless as the engine allows), attaches the probe agent so the simulation runs faster than real time, lets it settle, then kills the processes and summarises what each instance achieved.

The engine advances game time by the frame delta multiplied by the field H, and the delta comes from real elapsed time, so H is the wall-clock speed multiple.
The frame rate is separately pinned at 300, which means each simulation step covers 1000 * H / fps milliseconds of game time.
Watch that step size: it is the fidelity cost of running fast.

Measurements are only meaningful on an otherwise idle machine, and the first half minute of every run is distorted by JIT warmup.

.PARAMETER Count
Number of instances to run concurrently. Instance directories must already exist; create them with
New-RwInstance.ps1.

.PARAMETER Speed
Value written to the engine speed multiplier. Use 1 to leave the game at real time.

.PARAMETER Seconds
How long to let the instances run before stopping them.

.PARAMETER Map
Runs real skirmish episodes instead of leaving the game at the menu, on the first built-in map whose file name contains this text.
Without it the workload is the battle the game runs behind its own menu, which is a real simulation but not a real match.

.PARAMETER AgentOptions
Extra comma separated options appended to the agent's own, for the features this script has no parameter of its own for: catalog, obs, spawn, dump, act.
See the header of probe-agent/RwProbeAgent.java for the full list.

.EXAMPLE
.\New-RwInstance.ps1 -Count 8
.\Start-RwProbe.ps1 -Count 8 -Speed 10 -Seconds 90

.EXAMPLE
.\Start-RwProbe.ps1 -Count 8 -Speed 10 -Seconds 180 -Map Lake -Difficulty 1

.EXAMPLE
.\Start-RwProbe.ps1 -Count 1 -Speed 10 -Seconds 200 -Map Lake -AgentOptions 'obs=true,catalog=true'
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 64)]
    [int]$Count = 1,

    [double]$Speed = 10,

    [ValidateRange(10, 3600)]
    [int]$Seconds = 60,

    [int]$IntervalMs = 15000,

    [string]$Map = '',

    [int]$Opponents = 1,

    [ValidateRange(-2, 3)]
    [int]$Difficulty = 1,

    [int]$Episodes = 20,

    [int]$MaxSeconds = 0,

    [string]$AgentOptions = '',

    [string]$MasterPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\rw'),

    [string]$InstanceRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\instances'),

    [string]$AgentJar = (Join-Path (Split-Path $PSScriptRoot -Parent) 'probe-agent\rwprobe.jar'),

    [string]$LogRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\probe-logs'),

    [string]$HeapSize = '800M'
)

$ErrorActionPreference = 'Stop'

$java = Join-Path $MasterPath 'jvm64\bin\java.exe'
if (-not (Test-Path $java)) { throw "No bundled JVM at $java" }
if (-not (Test-Path $AgentJar)) { throw "No probe agent at $AgentJar. Build it with probe-agent\build.ps1" }

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$running = Get-Process java -ErrorAction SilentlyContinue
if ($running) {
    Write-Warning "$($running.Count) java process(es) already running; results will be distorted."
}

$processes = @()
for ($i = 0; $i -lt $Count; $i++) {
    $name = '{0:d2}' -f $i
    $dir = Join-Path $InstanceRoot $name
    if (-not (Test-Path $dir)) { throw "Instance directory missing: $dir" }

    $agentOptions = "interval=$IntervalMs,speed=$Speed"
    if ($Map -ne '') {
        # Each instance gets its own seed so that concurrent runs are not all the same match.
        $agentOptions += ",match=$Map,ai=$Opponents,difficulty=$Difficulty,episodes=$Episodes,seed=$(1000 + $i),maxSeconds=$MaxSeconds"
    }
    if ($AgentOptions -ne '') { $agentOptions += ",$AgentOptions" }

    # -nomods keeps unit definitions consistent: locally installed mods change them.
    $arguments = @(
        "-Xmx$HeapSize",
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

Write-Host "running $Count instance(s) at speed $Speed for $Seconds s"
Start-Sleep -Seconds $Seconds

foreach ($process in $processes) {
    try { $process.Kill() } catch { }
}
Start-Sleep -Seconds 3

$speeds = @()
$rates = @()
$steps = @()

for ($i = 0; $i -lt $Count; $i++) {
    $name = '{0:d2}' -f $i
    # Only lines with a plain positive rate are usable; anything else is a sample spanning an episode boundary.
    $last = Get-Content (Join-Path $LogRoot "$name.out") -ErrorAction SilentlyContinue |
        Select-String -Pattern 'fps=[\d.]+ speed=[\d.]+x step=[\d.]+ms' | Select-Object -Last 1
    if ($last -match 'fps=([\d.]+) speed=([\d.]+)x step=([\d.]+)ms') {
        $rates += [double]$Matches[1]
        $speeds += [double]$Matches[2]
        $steps += [double]$Matches[3]
    } else {
        Write-Warning "instance $name produced no measurement; see $LogRoot\$name.out"
    }
}

if ($speeds.Count -eq 0) { throw 'No instance reported a measurement.' }

$speedStats = $speeds | Measure-Object -Average -Minimum -Sum
$rateStats = $rates | Measure-Object -Average -Sum
$stepStats = $steps | Measure-Object -Average -Maximum

[PSCustomObject]@{
    Instances        = $speeds.Count
    RequestedSpeed   = $Speed
    SpeedAverage     = [math]::Round($speedStats.Average, 2)
    SpeedMinimum     = [math]::Round($speedStats.Minimum, 2)
    SpeedAggregate   = [math]::Round($speedStats.Sum, 1)
    FpsAverage       = [math]::Round($rateStats.Average, 0)
    FpsTotal         = [math]::Round($rateStats.Sum, 0)
    StepAverageMs    = [math]::Round($stepStats.Average, 0)
    StepMaximumMs    = [math]::Round($stepStats.Maximum, 0)
}
