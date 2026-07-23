<#
.SYNOPSIS
Starts the two game instances of one paired lockstep match.

.DESCRIPTION
Start the control process first; the agents wait for it and retry until it is listening.

    python -m rwintel.control --instances 2 --map Lake --max-seconds 300
    .\tools\windows\Start-RwPairedMatch.ps1

Both instances are launched identically. Which of them hosts and which joins is not decided here but by the control process, which sends host, port and join alongside the start command, and the port it sends has to be the one checked here.

What this adds over Start-RwAgents.ps1 is that check, because a host that cannot bind its port says so only in the game's own log and returns quietly to the caller; the pair then sits waiting for a match that will never begin. Failing here instead says it at the point where it can still be fixed.

A host compares a checksum of the core unit definitions before admitting anyone and refuses a client whose units differ. Instances junction back to one master copy of the install and are launched with mods off, so the two agree by construction as long as neither directory has been given a copy of its own.

.PARAMETER MatchPort
The TCP and UDP port the hosting instance binds. Two pairs running side by side need different ones, and so does a pair sharing a machine with a real game.

.PARAMETER Speed
Engine speed multiplier both instances hold. They advance in step with each other, so a pair given two different multipliers spends its time waiting rather than simulating.

.PARAMETER Seconds
How long to leave the instances running before stopping them. Zero leaves them running.

.EXAMPLE
.\Start-RwPairedMatch.ps1 -Speed 5 -Seconds 600
#>
[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$MatchPort = 5123,

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

$holder = Get-NetTCPConnection -State Listen -LocalPort $MatchPort -ErrorAction SilentlyContinue
if ($holder) {
    $owners = ($holder | ForEach-Object { $_.OwningProcess } | Sort-Object -Unique) -join ', '
    throw "Port $MatchPort is already being listened on by process $owners. Stop it or pass a different -MatchPort, and send the control process the same one."
}

& (Join-Path $PSScriptRoot 'Start-RwAgents.ps1') -Count 2 -Speed $Speed -Seconds $Seconds `
    -ControlHost $ControlHost -Port $Port -TacticalMs $TacticalMs -OperationalMs $OperationalMs `
    -AgentOptions $AgentOptions -MasterPath $MasterPath -InstanceRoot $InstanceRoot `
    -AgentJar $AgentJar -LogRoot $LogRoot -HeapSize $HeapSize

Write-Host "paired match: instances 00 and 01, match port $MatchPort is free"
