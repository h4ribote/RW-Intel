<#
.SYNOPSIS
Builds the control agent jar.

.DESCRIPTION
Compiles against the JDK bundled with the game.
Rusted Warfare ships a complete OpenJDK 13 under jvm64, including javac and jar, so nothing has to be installed separately, and building with the same JVM that will load the agent rules out class file version mismatches.

.EXAMPLE
.\build.ps1
#>
[CmdletBinding()]
param(
    [string]$MasterPath = (Join-Path (Split-Path $PSScriptRoot -Parent) 'local\rw')
)

$ErrorActionPreference = 'Stop'

$javac = Join-Path $MasterPath 'jvm64\bin\javac.exe'
$jar = Join-Path $MasterPath 'jvm64\bin\jar.exe'
if (-not (Test-Path $javac)) { throw "No bundled JDK at $javac" }

$classes = Join-Path $PSScriptRoot 'classes'
if (Test-Path $classes) { Remove-Item $classes -Recurse -Force }
New-Item -ItemType Directory -Force -Path $classes | Out-Null

$sources = Get-ChildItem $PSScriptRoot -Filter '*.java' | ForEach-Object { $_.FullName }
& $javac -Xlint:-options -d $classes $sources
if ($LASTEXITCODE -ne 0) { throw 'compilation failed' }

$output = Join-Path $PSScriptRoot 'rwagent.jar'
& $jar --create --file $output --manifest (Join-Path $PSScriptRoot 'manifest.txt') -C $classes .
if ($LASTEXITCODE -ne 0) { throw 'packaging failed' }

Write-Host "built $output"
