<#
.SYNOPSIS
Starts game instances under the control agent, which dial in to the control process.

.DESCRIPTION
Start the control process first; the agents wait for it and retry until it is listening.

    python -m rwintel.control --instances 1 --map Lake --max-seconds 300
    .\tools\windows\Start-RwAgents.ps1 -Count 1

The agents drive nothing by themselves. Episodes start when the control process says so, which is what keeps the settings of a run in one place.

.PARAMETER Count
Instances to start. Their directories must already exist; create them with New-RwInstance.ps1.

.PARAMETER Speed
Engine speed multiplier the agent holds. Ten is the measured best on this machine at eight instances.

.PARAMETER Seconds
How long to leave the instances running before stopping them. Zero leaves them running.

.EXAMPLE
.\Start-RwAgents.ps1 -Count 1 -Speed 10 -Seconds 240
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 64)]
    [int]$Count = 1,

    # Which instance directory the run starts numbering at. Nought for an ordinary run; anything else is for
    # putting a second, smaller run on instances a run already under way is not using, which is how a change
    # is smoke tested without stopping a measurement.
    [ValidateRange(0, 63)]
    [int]$Offset = 0,

    [double]$Speed = 10,

    [int]$Seconds = 0,

    [string]$ControlHost = '127.0.0.1',

    [int]$Port = 8642,

    [int]$TacticalMs = 200,

    [int]$OperationalMs = 2000,

    [string]$AgentOptions = '',

    [string]$MasterPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\rw'),

    [string]$InstanceRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\instances'),

    [string]$AgentJar = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'agent\rwagent.jar'),

    [string]$LogRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\agent-logs'),

    [string]$HeapSize = '800M'
)

$ErrorActionPreference = 'Stop'

$java = Join-Path $MasterPath 'jvm64\bin\java.exe'
if (-not (Test-Path $java)) { throw "No bundled JVM at $java" }
if (-not (Test-Path $AgentJar)) { throw "No agent at $AgentJar. Build it with agent\build.ps1" }

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$processes = @()
for ($i = 0; $i -lt $Count; $i++) {
    $name = '{0:d2}' -f ($i + $Offset)
    $dir = Join-Path $InstanceRoot $name
    if (-not (Test-Path $dir)) { throw "Instance directory missing: $dir" }

    # The control process numbers its instances from nought whatever directory they run in, so the offset moves
    # the directory and the log and not the identity the run knows an instance by.
    $options = "host=$ControlHost,port=$Port,instance=$i,speed=$Speed,tactical=$TacticalMs,operational=$OperationalMs"
    if ($AgentOptions -ne '') { $options += ",$AgentOptions" }

    # -nomods keeps unit definitions consistent: locally installed mods change them.
    $arguments = @(
        "-Xmx$HeapSize",
        '-Dfile.encoding=UTF-8',
        "-Djava.library.path=$dir",
        "-javaagent:$AgentJar=$options",
        '-cp', "$MasterPath\game-lib.jar;$MasterPath\libs\*",
        'com.corrodinggames.rts.java.Main',
        '-nodisplay', '-nosound', '-nomusic', '-nomods'
    )

    $processes += Start-Process -FilePath $java -WorkingDirectory $dir -ArgumentList $arguments `
        -RedirectStandardOutput (Join-Path $LogRoot "$name.out") `
        -RedirectStandardError (Join-Path $LogRoot "$name.err") `
        -PassThru -WindowStyle Hidden
}

Write-Host "started $Count instance(s) against ${ControlHost}:$Port, logs in $LogRoot"

if ($Seconds -gt 0) {
    Start-Sleep -Seconds $Seconds
    foreach ($process in $processes) {
        try { $process.Kill() } catch { }
    }
    Write-Host "stopped $Count instance(s)"
}
