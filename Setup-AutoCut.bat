@echo off
rem AutoCut Studio one-time setup (creates .venv, installs packages, downloads ffmpeg, adds a desktop shortcut).
rem Add "-WithSpeech" after the file name to also install speech recognition (captions), e.g. from a terminal:
rem     Setup-AutoCut.bat -WithSpeech
rem Exit code of the setup script: 0 done, 1 stopped by an error, 2 done but ffmpeg is missing, 3 done but tests failed.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\windows\Setup-AutoCut.ps1" %*
set "CODE=%ERRORLEVEL%"
echo.
if %CODE% GEQ 3 (
    echo Setup finished, but some tests failed. Please read the message above.
) else if %CODE% EQU 2 (
    echo Setup finished, but ffmpeg is NOT installed yet. The editor opens, but it cannot import, preview or export.
    echo Please read the message above to add ffmpeg, then use the "AutoCut Studio" shortcut or Start-AutoCut.bat.
) else if %CODE% EQU 1 (
    echo Setup did not finish. Please read the message above.
) else (
    echo Setup finished. Use the "AutoCut Studio" shortcut on your desktop or double-click Start-AutoCut.bat.
)
pause
endlocal & exit /b %CODE%
