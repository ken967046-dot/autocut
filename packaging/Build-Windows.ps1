<#
  AutoCut Studio - build the Windows "one folder" app with PyInstaller (ASCII only on purpose: Windows PowerShell 5.1).

      powershell -NoProfile -ExecutionPolicy Bypass -File packaging\Build-Windows.ps1

  Run Setup-AutoCut.bat first (it creates .venv).
  Result: dist\AutoCutStudio\AutoCutStudio.exe.  STATUS: not verified until it was built and started on Windows.

  ffmpeg is NOT put into the folder by default: the builds the setup downloads are mostly GPL, and whoever gives such a
  folder to other people takes on the GPL duties for it.  The exe finds ffmpeg in ffmpeg\bin next to itself, in
  %LOCALAPPDATA%\AutoCutStudio\ffmpeg or on PATH.  -BundleFfmpeg copies tools\ffmpeg\bin\ffmpeg.exe and ffprobe.exe into
  the folder - for your own use, or after you have checked the licence of your build.
#>
param([switch]$BundleFfmpeg)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
if ($BundleFfmpeg) { $env:AUTOCUT_BUNDLE_FFMPEG = '1' } else { Remove-Item Env:\AUTOCUT_BUNDLE_FFMPEG -ErrorAction SilentlyContinue }
if (-not (Test-Path -LiteralPath $py)) { throw 'Run Setup-AutoCut.bat first (.venv was not found).' }

& $py -m pip install --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) { throw 'pip install pyinstaller failed' }
& $py -m PyInstaller --noconfirm --clean --distpath (Join-Path $Root 'dist') --workpath (Join-Path $Root 'build') (Join-Path $Root 'packaging\autocut.spec')
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }

$exe = Join-Path $Root 'dist\AutoCutStudio\AutoCutStudio.exe'
if (Test-Path -LiteralPath $exe) { Write-Host "Built: $exe" -ForegroundColor Green } else { throw 'The build finished but the exe is missing.' }
