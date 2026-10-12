@echo off
rem Startet SHARP Studio per Doppelklick.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-studio.ps1"
if errorlevel 1 (
    echo.
    echo SHARP Studio konnte nicht gestartet werden.
    pause
)
