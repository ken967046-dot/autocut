@echo off
rem AutoCut Studio launcher: double-click to open the editor. Drag media files onto this file to open them directly.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo AutoCut Studio is not installed yet. Please double-click Setup-AutoCut.bat first.
    pause
    goto :done
)
rem pythonw has no console, so a broken installation (a missing Qt DLL, a package that failed to install) would make
rem the window simply never appear.  Load the program once with the console python first: if that fails, say why.
".venv\Scripts\python.exe" -c "import autocut.gui.app" >nul 2>nul
if errorlevel 1 (
    echo AutoCut Studio cannot start: the installation looks incomplete or damaged.
    echo Please double-click Setup-AutoCut.bat again. What Python reported:
    echo.
    ".venv\Scripts\python.exe" -c "import autocut.gui.app"
    echo.
    pause
    goto :done
)
start "" ".venv\Scripts\pythonw.exe" -m autocut studio %*
:done
endlocal
