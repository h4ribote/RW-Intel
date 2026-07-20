<#
.SYNOPSIS
Builds the probe agent jar.

.DESCRIPTION
Compiles against the JDK bundled with the game.
Rusted Warfare ships a complete OpenJDK 13 under jvm64, including javac, jar and javap, so nothing has to be installed separately.
Building with the same JVM that will load the agent also rules out class file version mismatches.

.EXAMPLE
.\build.ps1
#>
[CmdletBinding()]
param(
    [string]$MasterPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'local\rw')
)

$ErrorActionPreference = 'Stop'

$javac = Join-Path $MasterPath 'jvm64\bin\javac.exe'
$jar = Join-Path $MasterPath 'jvm64\bin\jar.exe'
if (-not (Test-Path $javac)) { throw "No bundled JDK at $javac" }

$classes = Join-Path $PSScriptRoot 'classes'
if (Test-Path $classes) { Remove-Item $classes -Recurse -Force }
New-Item -ItemType Directory -Force -Path $classes | Out-Null

& $javac -d $classes (Join-Path $PSScriptRoot 'RwProbeAgent.java')
if ($LASTEXITCODE -ne 0) { throw 'compilation failed' }

$output = Join-Path $PSScriptRoot 'rwprobe.jar'
& $jar --create --file $output --manifest (Join-Path $PSScriptRoot 'manifest.txt') -C $classes .
if ($LASTEXITCODE -ne 0) { throw 'packaging failed' }

Write-Host "built $output"
