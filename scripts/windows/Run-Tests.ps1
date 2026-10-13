<#
  AutoCut Studio - run the automated tests on Windows (ASCII only on purpose: Windows PowerShell 5.1).

      .\scripts\windows\Run-Tests.ps1              # everything (the end-to-end flows render real videos and take a few minutes)
      .\scripts\windows\Run-Tests.ps1 -Fast        # skip the slow end-to-end flows
      .\scripts\windows\Run-Tests.ps1 -Only tests\test_core.py

  Needs Setup-AutoCut.bat to have run (it creates .venv and downloads ffmpeg).  The tests use the offscreen Qt platform,
  so no window opens, and they use a private AUTOCUT_HOME, so your real settings and projects are never touched.
#>
param([switch]$Fast, [string]$Only = '')
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw 'Run Setup-AutoCut.bat first (.venv was not found).' }

& $py -m pip install -q -r (Join-Path $Root 'requirements-dev.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install of the test requirements failed' }

$ffmpegBin = Join-Path $Root 'tools\ffmpeg\bin'
if (Test-Path -LiteralPath $ffmpegBin) { $env:PATH = "$ffmpegBin;$env:PATH" }
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONPATH = Join-Path $Root 'src'
$testHome = Join-Path ([System.IO.Path]::GetTempPath()) ("autocut-tests-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
[void][System.IO.Directory]::CreateDirectory($testHome)
$env:AUTOCUT_HOME = $testHome

$argsList = @('-m', 'pytest', '-p', 'no:cacheprovider', '-o', 'addopts=', '-q', '--tb=short')
if ($Fast) { $argsList += @('-m', 'not e2e') }
if ($Only) { $argsList += $Only }
& $py @argsList
$code = $LASTEXITCODE
Remove-Item -LiteralPath $testHome -Recurse -Force -ErrorAction SilentlyContinue
exit $code
