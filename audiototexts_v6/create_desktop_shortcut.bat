@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "VPY="
if exist ".venv\Scripts\python.exe" set "VPY=%~dp0.venv\Scripts\python.exe"
if not defined VPY if exist ".pyexe" set /p VPY=<".pyexe"
if not defined VPY (
  echo Please run setup_conda.bat or setup_windows.bat first.
  pause
  exit /b 1
)
"%VPY%" stt.py shortcut
pause
