<#
  AutoCut Studio - one-time setup for Windows 10/11 (ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less
  files with the legacy code page).

  Run it by double-clicking Setup-AutoCut.bat in the project folder, or:
      powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\Setup-AutoCut.ps1 [-WithSpeech] [-WithAI] [-SkipFfmpeg] [-NoShortcut] [-RunTests]

  Where things go
    inside the project folder   .venv (the Python packages) and tools\ffmpeg (the ffmpeg download)
    outside the project folder  pip's download cache, the desktop shortcut, the editor's settings (current-user
                                registry), the editor's cache folder %LOCALAPPDATA%\AutoCutStudio, and Python 3.12
                                itself - installed for the current user through winget, and only when no Python
                                3.10+ is found
  What it downloads (the setup needs the internet; the editor itself uploads nothing): the Python packages from
  pypi.org, ffmpeg from gyan.dev or github.com (checked against the SHA-256 the publisher lists), Python through winget.

  Exit code   0 done     1 stopped by an error     2 done, but ffmpeg is missing     3 done, but the tests failed
#>
param(
    [switch]$WithSpeech,     # also install faster-whisper + opencc (captions from speech; models download on first use)
    [switch]$WithAI,         # also install the Claude SDK (optional AI highlight judge; needs an API key, sends subtitle text only)
    [switch]$SkipFfmpeg,     # do not download ffmpeg (use this when ffmpeg is already on PATH)
    [switch]$NoShortcut,     # do not create the desktop shortcut
    [switch]$RunTests        # run the test-suite at the end (needs ffmpeg)
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $Root

function Step([string]$m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Invoke-Checked([string]$what, [scriptblock]$cmd) {
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function Find-Python {
    foreach ($ver in '3.12', '3.13', '3.11', '3.10') {
        try {
            $p = & py "-$ver" -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return ([string]$p).Trim() }
        } catch { }
    }
    foreach ($name in 'python', 'python3') {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notlike '*WindowsApps*') {
            try {
                $ok = & $cmd.Source -c "import sys;print(int(sys.version_info>=(3,10)))" 2>$null
                if ("$ok".Trim() -eq '1') { return $cmd.Source }
            } catch { }
        }
    }
    $base = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    if (Test-Path -LiteralPath $base) {
        foreach ($dir in (Get-ChildItem -LiteralPath $base -Directory | Sort-Object Name -Descending)) {
            $exe = Join-Path $dir.FullName 'python.exe'
            if (Test-Path -LiteralPath $exe) { return $exe }
        }
    }
    return $null
}

$exitCode = 0

Step 'Looking for Python 3.10+'
$py = Find-Python
if (-not $py) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Python 3.10+ was not found and winget is not available. Install Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH") and run this setup again.'
    }
    Step 'Installing Python 3.12 for the current user (winget)'
    & winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    $wingetCode = $LASTEXITCODE
    # winget's exit code is not a reliable "is Python there now" (it is also non-zero when nothing needed doing): look again.
    $py = Find-Python
    if (-not $py) {
        throw "Python 3.12 could not be installed through winget (exit code $wingetCode). Install it from https://www.python.org/downloads/ (tick `"Add python.exe to PATH`"), then run this setup again."
    }
}
Write-Host "Using $py"

Step 'Creating the virtual environment (.venv)'
$venvPy = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) { Invoke-Checked 'venv creation' { & $py -m venv (Join-Path $Root '.venv') } }
Invoke-Checked 'pip upgrade' { & $venvPy -m pip install --upgrade pip }

Step 'Installing AutoCut Studio and its packages (PySide6, numpy, OpenCV) - this takes a few minutes'
Invoke-Checked 'pip install requirements' { & $venvPy -m pip install -r (Join-Path $Root 'requirements.txt') }
Invoke-Checked 'pip install autocut' { & $venvPy -m pip install -e $Root --no-deps }

if ($WithSpeech) {
    Step 'Installing speech recognition (faster-whisper, opencc). The model itself downloads the first time you use it.'
    Invoke-Checked 'pip install speech' { & $venvPy -m pip install -r (Join-Path $Root 'requirements-speech.txt') }
}
if ($WithAI) {
    Step 'Installing the optional AI highlight judge (anthropic). It needs an API key and only ever sends subtitle text.'
    Invoke-Checked 'pip install ai' { & $venvPy -m pip install -r (Join-Path $Root 'requirements-ai.txt') }
}

# ffmpeg is the one step that depends on somebody else's web server: when it fails the editor is still installed, so
# carry on (doctor, shortcut) and say clearly at the end what is missing.
$ffmpegProblem = $null
if (-not $SkipFfmpeg) {
    $localFfmpeg = Join-Path $Root 'tools\ffmpeg\bin\ffmpeg.exe'
    $haveHere = Test-Path -LiteralPath $localFfmpeg
    $onPath = (Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue)
    if ($haveHere -or $onPath) {
        Step 'ffmpeg already available - skipping the download'
    } else {
        Step 'Downloading ffmpeg'
        try {
            & (Join-Path $PSScriptRoot 'Get-Ffmpeg.ps1')
        } catch {
            $ffmpegProblem = "$($_.Exception.Message)"
            Write-Host "ffmpeg was NOT installed: $ffmpegProblem" -ForegroundColor Yellow
        }
    }
}

Step 'Environment check (autocut doctor)'
& $venvPy -m autocut doctor
if ($LASTEXITCODE -ne 0) { Write-Host 'doctor reported problems - see above.' -ForegroundColor Yellow }

if (-not $NoShortcut) {
    Step 'Creating a desktop shortcut'
    try {
        $desktop = [Environment]::GetFolderPath('Desktop')
        $shell = New-Object -ComObject WScript.Shell
        $lnk = $shell.CreateShortcut((Join-Path $desktop 'AutoCut Studio.lnk'))
        $lnk.TargetPath = Join-Path $Root '.venv\Scripts\pythonw.exe'
        $lnk.Arguments = '-m autocut studio'
        $lnk.WorkingDirectory = $Root
        $lnk.Description = 'AutoCut Studio'
        $lnk.Save()
        Write-Host "Shortcut created on the desktop: $desktop"
    } catch {
        Write-Host "Could not create the shortcut ($_). Use Start-AutoCut.bat instead." -ForegroundColor Yellow
    }
}

if ($RunTests) {
    Step 'Running the test-suite (Run-Tests.ps1)'
    try {
        & (Join-Path $PSScriptRoot 'Run-Tests.ps1')
        $testCode = $LASTEXITCODE
    } catch {
        Write-Host "The tests could not be run: $_" -ForegroundColor Yellow
        $testCode = 1
    }
    if ($testCode -ne 0) {
        $exitCode = 3
        Write-Host 'Some tests failed - see above.' -ForegroundColor Yellow
    }
}

Write-Host ""
if ($ffmpegProblem) {
    if ($exitCode -eq 0) { $exitCode = 2 }
    Write-Host 'ffmpeg is NOT installed: the editor opens, but it cannot import, preview or export media until it is.' -ForegroundColor Yellow
    Write-Host 'Run scripts\windows\Get-Ffmpeg.ps1 again later, or download a build from https://ffmpeg.org/download.html' -ForegroundColor Yellow
    Write-Host 'and put ffmpeg.exe and ffprobe.exe into tools\ffmpeg\bin (or add them to PATH).' -ForegroundColor Yellow
}
if ($exitCode -eq 0) {
    Write-Host "Done. Start the editor with the desktop shortcut or Start-AutoCut.bat" -ForegroundColor Green
}
exit $exitCode
