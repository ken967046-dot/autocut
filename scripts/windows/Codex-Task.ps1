<#
  AutoCut Studio - hand ONE small task card (docs\CODEX_TASKS.md) to Codex as the implementer.
  ASCII only on purpose (Windows PowerShell 5.1).

  Needs the Codex CLI installed and logged in:   npm install -g @openai/codex    then    codex login

      .\scripts\windows\Codex-Task.ps1 -List                   # show the task cards
      .\scripts\windows\Codex-Task.ps1 -Task CODEX-2           # work on CODEX-2 on branch codex/CODEX-2
      .\scripts\windows\Codex-Task.ps1 -Task CODEX-2 -Extra "the video is D:\talk.mp4, subtitles D:\talk.srt"
      .\scripts\windows\Codex-Task.ps1 -Task CODEX-2 -DryRun   # only write the prompt file

  Codex runs with a workspace-write sandbox on its own git branch (never on main).  It must commit its work and add a
  hand-over note to TASKS.md; Claude reviews and merges.  Output: .codex-runs\task-<id>-<time>.md
  Saving quota: one card per run, the card lists the files to read, and the prompt tells Codex not to explore further.
#>
param(
    [string]$Task = '',
    [string]$Extra = '',
    [switch]$List,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $Root

$utf8 = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $utf8
try { [Console]::OutputEncoding = $utf8 } catch { }

$cardsFile = Join-Path $Root 'docs\CODEX_TASKS.md'
if (-not (Test-Path -LiteralPath $cardsFile)) { throw "Missing $cardsFile" }
$text = [System.IO.File]::ReadAllText($cardsFile, $utf8)

# split into "## <ID>" sections
$cards = [ordered]@{}
$current = $null
$buf = New-Object System.Text.StringBuilder
foreach ($line in ($text -split "`r?`n")) {
    if ($line -match '^##\s+(CODEX-\d+)') {
        if ($current) { $cards[$current] = $buf.ToString() }
        $current = $Matches[1]
        $buf = New-Object System.Text.StringBuilder
    }
    if ($current) { [void]$buf.AppendLine($line) }
}
if ($current) { $cards[$current] = $buf.ToString() }

if ($List -or -not $Task) {
    Write-Host 'Task cards in docs\CODEX_TASKS.md:'
    foreach ($k in $cards.Keys) {
        $title = (($cards[$k] -split "`r?`n") | Where-Object { $_ -match '^\*\*' } | Select-Object -First 1)
        Write-Host ("  {0}  {1}" -f $k, $title)
    }
    if (-not $Task) { return }
}
$id = $Task.ToUpper()
if (-not $cards.Contains($id)) { throw "Unknown task '$Task'. Use -List to see the cards." }
if ($id -eq 'CODEX-1') { throw 'CODEX-1 is a read-only review: run scripts\windows\Codex-Review.ps1 with the command shown in docs\CODEX_TASKS.md.' }

$hasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path -LiteralPath (Join-Path $Root '.git'))
if (-not $hasGit) { throw 'This needs git (Codex works on its own branch).' }
$dirty = @(& git status --porcelain)
if ($dirty.Count -gt 0 -and -not $DryRun) { throw 'The working tree has uncommitted changes. Commit or stash them first.' }

$branch = "codex/$id"
$prompt = @"
You are Codex, the implementer on the AutoCut Studio project (PySide6 desktop video editor, Windows 11 target).
Read AGENTS.md first and obey its rules. You work on git branch '$branch' (already checked out).

Your task is exactly this card (do not start other work, do not explore files the card does not need -- the user's
quota is limited):

$($cards[$id])

When you are done:
1. Run the tests the card names (python -m pytest <files> -q). Fix what you broke.
2. Commit in small steps with messages that say WHY; end each message with:  Co-Authored-By: Codex <noreply@openai.com>
3. In TASKS.md add a hand-over note for $id using the template in COLLABORATION.md (what changed, why, how verified,
   what was NOT verified). Set its status to 'review'. Commit that too.
4. Your final answer: a short summary + the commands you ran and their results. Never claim something was verified if
   you did not run it.
"@
if ($Extra) { $prompt += "`nExtra information from the user: $Extra`n" }

$outDir = Join-Path $Root '.codex-runs'
[void][System.IO.Directory]::CreateDirectory($outDir)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$promptFile = Join-Path $outDir "task-$id-$stamp.prompt.txt"
$outFile = Join-Path $outDir "task-$id-$stamp.md"
[System.IO.File]::WriteAllText($promptFile, $prompt, $utf8)
Write-Host "Prompt written: $promptFile"
if ($DryRun) { return }

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw 'The Codex CLI was not found. Install it with:  npm install -g @openai/codex   then run:  codex login'
}
$exists = @(& git branch --list $branch)
if ($exists.Count -gt 0) { & git switch $branch } else { & git switch -c $branch }
if ($LASTEXITCODE -ne 0) { throw "Could not switch to branch $branch" }

Write-Host "Codex is working on $id (branch $branch). This can take a while..." -ForegroundColor Cyan
$prompt | & codex exec -s workspace-write --skip-git-repo-check -o $outFile -
if ($LASTEXITCODE -ne 0) { throw "codex exec failed (exit code $LASTEXITCODE). If it asks you to sign in, run:  codex login" }
if (Test-Path -LiteralPath $outFile) { Write-Host "Codex summary: $outFile" -ForegroundColor Green }
Write-Host "Next: review the branch (git log main..$branch), then ask Claude to review and merge it." -ForegroundColor Green
