<#
.SYNOPSIS
Creates lightweight working directories for running several Rusted Warfare processes side by side.

.DESCRIPTION
Each instance needs its own current directory, because the game writes preferences.ini, saves, replays and cache relative to it.
The bulk of the install is read-only, so instances reference it instead of copying it: directories are NTFS junctions back to the master copy, and the native DLLs in the install root are hardlinks.
An instance therefore costs no meaningful disk space.

The DLLs cannot be junctioned because junctions only work on directories, and they cannot be left out because the Windows loader resolves the dependencies of rocketConnector64.dll through the current directory.
Without them the game aborts with UnsatisfiedLinkError.

.PARAMETER Count
Number of instance directories to create, named 00, 01, and so on.

.PARAMETER MasterPath
The master copy of the game install.

.PARAMETER InstanceRoot
Directory that will hold the instance directories.

.PARAMETER Force
Recreate instance directories that already exist, discarding their saves and settings.

.EXAMPLE
.\New-RwInstance.ps1 -Count 8
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 64)]
    [int]$Count,

    [string]$MasterPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\rw'),

    [string]$InstanceRoot = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\instances'),

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# Read-only trees shared with the master copy.
$LinkedDirectories = @('assets', 'font', 'res', 'mods')

# Written by the game, so each instance keeps its own.
$OwnDirectories = @('saves', 'cache', 'replays')

if (-not (Test-Path (Join-Path $MasterPath 'game-lib.jar'))) {
    throw "No game-lib.jar under $MasterPath. Point -MasterPath at a copy of the game install."
}

$masterDlls = Get-ChildItem -Path $MasterPath -File -Filter *.dll
if ($masterDlls.Count -eq 0) {
    throw "No DLLs in $MasterPath. The copy looks incomplete."
}

$masterDrive = (Get-Item $MasterPath).PSDrive.Name
$instanceParent = Split-Path $InstanceRoot -Qualifier
if ($instanceParent.TrimEnd(':') -ne $masterDrive) {
    throw "Hardlinks require the same volume: master is on ${masterDrive}: but instances would be on $instanceParent"
}

New-Item -ItemType Directory -Force -Path $InstanceRoot | Out-Null

$created = 0
$reused = 0

for ($i = 0; $i -lt $Count; $i++) {
    $name = '{0:d2}' -f $i
    $path = Join-Path $InstanceRoot $name

    if ((Test-Path $path) -and -not $Force) {
        $reused++
        continue
    }

    if (Test-Path $path) {
        # Remove the junctions first so their targets are never followed into the master copy.
        foreach ($link in $LinkedDirectories) {
            $linkPath = Join-Path $path $link
            if (Test-Path $linkPath) {
                [System.IO.Directory]::Delete($linkPath, $false)
            }
        }
        Remove-Item -Path $path -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $path | Out-Null

    foreach ($link in $LinkedDirectories) {
        $target = Join-Path $MasterPath $link
        if (Test-Path $target) {
            New-Item -ItemType Junction -Path (Join-Path $path $link) -Target $target | Out-Null
        }
    }

    foreach ($own in $OwnDirectories) {
        New-Item -ItemType Directory -Force -Path (Join-Path $path $own) | Out-Null
    }

    foreach ($dll in $masterDlls) {
        New-Item -ItemType HardLink -Path (Join-Path $path $dll.Name) -Target $dll.FullName | Out-Null
    }

    $created++
}

Write-Host "instances: $created created, $reused reused, root $InstanceRoot"
