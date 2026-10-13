<#
  AutoCut Studio - ask Codex (OpenAI Codex CLI) for a read-only code review (ASCII only on purpose: Windows PowerShell 5.1).

  Needs the Codex CLI installed and logged in:   npm install -g @openai/codex    then    codex login

      .\scripts\windows\Codex-Review.ps1                          # review the current branch against main (whole project when there is no diff)
      .\scripts\windows\Codex-Review.ps1 -Scope all               # review the whole project
      .\scripts\windows\Codex-Review.ps1 -Base HEAD~3             # review the last 3 commits
      .\scripts\windows\Codex-Review.ps1 -Focus "core/ops.py crossfade edge cases"
      .\scripts\windows\Codex-Review.ps1 -DryRun                  # only write the prompt file, do not call Codex

  Codex runs in a READ-ONLY sandbox (it cannot change files).  Output: .codex-runs\review-<time>.md (its final answer)
  and .codex-runs\review-<time>.prompt.txt (what it was asked).  Paste the result into TASKS.md and hand it to Claude.
#>
param(
    [string]$Base = 'main',
    [ValidateSet('diff', 'all')] [string]$Scope = 'diff',
    [string]$Focus = '',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $Root

$utf8 = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $utf8
try { [Console]::OutputEncoding = $utf8 } catch { }

$promptSrc = Join-Path $Root 'docs\CODEX_REVIEW_PROMPT.md'
if (-not (Test-Path -LiteralPath $promptSrc)) { throw "Missing $promptSrc" }
$prompt = [System.IO.File]::ReadAllText($promptSrc, $utf8)

$hasGit = $null -ne (Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path -LiteralPath (Join-Path $Root '.git'))
if ($Scope -eq 'diff' -and $hasGit) {
    $changed = @(& git diff --name-only "$Base...HEAD" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $changed.Count -eq 0) {
        Write-Host "No diff against '$Base' - reviewing the whole project instead." -ForegroundColor Yellow
        $Scope = 'all'
    }
} elseif ($Scope -eq 'diff') {
    $Scope = 'all'
}
if ($Scope -eq 'diff') {
    $prompt += "`n`n## This run`nReview ONLY the changes made on this branch: run ``git diff $Base...HEAD`` and read the touched files in full where needed.`n"
} else {
    $prompt += "`n`n## This run`nReview the WHOLE project, starting from src/autocut/core, render, autoedit and gui (in that order).`n"
}
if ($Focus) { $prompt += "`nExtra focus requested by the user: $Focus`n" }

$outDir = Join-Path $Root '.codex-runs'
[void][System.IO.Directory]::CreateDirectory($outDir)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$promptFile = Join-Path $outDir "review-$stamp.prompt.txt"
$outFile = Join-Path $outDir "review-$stamp.md"
[System.IO.File]::WriteAllText($promptFile, $prompt, $utf8)
Write-Host "Prompt written: $promptFile"
if ($DryRun) { return }

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw 'The Codex CLI was not found. Install it with:  npm install -g @openai/codex   then run:  codex login'
}
Write-Host 'Asking Codex (read-only sandbox). This can take several minutes...' -ForegroundColor Cyan
$prompt | & codex exec -s read-only --skip-git-repo-check -o $outFile -
if ($LASTEXITCODE -ne 0) { throw "codex exec failed (exit code $LASTEXITCODE). If it asks you to sign in, run:  codex login" }
if (Test-Path -LiteralPath $outFile) { Write-Host "Review saved: $outFile" -ForegroundColor Green } else { Write-Host 'Codex finished but wrote no output file.' -ForegroundColor Yellow }
