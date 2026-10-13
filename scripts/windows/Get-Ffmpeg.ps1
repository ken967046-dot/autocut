<#
  Downloads a Windows ffmpeg build into tools\ffmpeg\bin (the place AutoCut Studio looks first).
  ffmpeg is a separate program (LGPL/GPL depending on the build); AutoCut Studio only runs it as a child process.
  (ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less files with the legacy code page.)

  Sources, in this order (both are the Windows builds linked from https://ffmpeg.org/download.html).  EVERY download is
  checked against the SHA-256 its publisher puts next to it; a file that does not match is never installed, and a
  mismatch stops the script (it does not quietly fall back to another source):
    1. https://www.gyan.dev/ffmpeg/builds/   release essentials     (ffmpeg-release-essentials.zip + .sha256)
    2. https://github.com/BtbN/FFmpeg-Builds GPL build of the 8.1 release line (checksums.sha256 in the same release)

  Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\Get-Ffmpeg.ps1 [-Dest <folder>]
#>
param([string]$Dest)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Dest) { $Dest = Join-Path $Root 'tools\ffmpeg' }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'      # Invoke-WebRequest is much faster without the progress bar

$sources = @(
    @{ Name = 'gyan.dev essentials'; File = 'ffmpeg-release-essentials.zip';
       Url = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip';
       ShaUrl = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip.sha256' },
    @{ Name = 'BtbN GPL build (8.1 release line)'; File = 'ffmpeg-n8.1-latest-win64-gpl-8.1.zip';
       Url = 'https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-n8.1-latest-win64-gpl-8.1.zip';
       ShaUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/checksums.sha256' }
)

# Windows PowerShell 5.1 returns the body of a file it does not know as bytes (octet-stream): decode it ourselves,
# otherwise "-as [string]" turns it into "104 101 97 ..." and no checksum ever matches.
function Get-Text([string]$Url) {
    $r = Invoke-WebRequest -UseBasicParsing -Uri $Url
    if ($r.Content -is [byte[]]) { return [Text.Encoding]::UTF8.GetString($r.Content) }
    return [string]$r.Content
}

# The SHA-256 for $FileName from a checksum file: either just the hash, or lines like "<hash>  <file name>".
function Find-Hash([string]$Text, [string]$FileName) {
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^\s*([0-9a-fA-F]{64})(?:\s+\*?(.+?))?\s*$') {
            $name = $Matches[2]
            if ((-not $name) -or $name -eq $FileName) { return $Matches[1].ToLower() }
        }
    }
    return $null
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('autocut-ffmpeg-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$done = $false
$tampered = $false
try {
    foreach ($src in $sources) {
        if ($tampered) { break }
        try {
            Write-Host "Downloading $($src.Name) ..."
            $zip = Join-Path $tmp 'ffmpeg.zip'
            Invoke-WebRequest -UseBasicParsing -Uri $src.Url -OutFile $zip
            $expected = Find-Hash (Get-Text $src.ShaUrl) $src.File
            if (-not $expected) { throw "no SHA-256 for $($src.File) was found in $($src.ShaUrl)" }
            $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLower()
            if ($expected -ne $actual) {
                $tampered = $true
                throw "SHA-256 mismatch (published $expected, downloaded file $actual)"
            }
            Write-Host 'SHA-256 verified.'
            $extract = Join-Path $tmp 'x'
            Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
            $ffmpeg = Get-ChildItem -LiteralPath $extract -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1
            $ffprobe = Get-ChildItem -LiteralPath $extract -Recurse -Filter 'ffprobe.exe' | Select-Object -First 1
            if (-not $ffmpeg -or -not $ffprobe) { throw 'ffmpeg.exe / ffprobe.exe were not found in the archive' }
            $bin = Join-Path $Dest 'bin'
            New-Item -ItemType Directory -Path $bin -Force | Out-Null
            Copy-Item -LiteralPath $ffmpeg.FullName -Destination (Join-Path $bin 'ffmpeg.exe') -Force
            Copy-Item -LiteralPath $ffprobe.FullName -Destination (Join-Path $bin 'ffprobe.exe') -Force
            # Run it once.  All of its output is read (Select-Object -First on the pipeline would stop it early and leave
            # $LASTEXITCODE stale), and stderr text must not turn into a terminating error under Windows PowerShell 5.1.
            $old = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try { $out = @(& (Join-Path $bin 'ffmpeg.exe') -version 2>&1); $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
            if ($code -ne 0) { throw "the downloaded ffmpeg.exe does not start (exit code $code)" }
            $ver = [string]($out | Select-Object -First 1)
            Write-Host "Installed: $ver"
            Write-Host "Location : $bin"
            $done = $true
            break
        } catch {
            Write-Host "Source '$($src.Name)' failed: $_" -ForegroundColor Yellow
        }
    }
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
if ($tampered) {
    throw 'A download did not match the SHA-256 its publisher lists, so it was NOT installed and no other source was tried. Run this again later; if it keeps happening, download ffmpeg yourself from https://ffmpeg.org/download.html and put ffmpeg.exe and ffprobe.exe into tools\ffmpeg\bin.'
}
if (-not $done) {
    throw 'Could not download ffmpeg. Download a build from https://ffmpeg.org/download.html, and put ffmpeg.exe and ffprobe.exe into tools\ffmpeg\bin (or add them to PATH).'
}
