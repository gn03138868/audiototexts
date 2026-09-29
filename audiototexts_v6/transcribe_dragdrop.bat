@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Usagi STT V6 - drag and drop
set "VPY="
if exist ".venv\Scripts\python.exe" set "VPY=%~dp0.venv\Scripts\python.exe"
if not defined VPY if exist ".pyexe" set /p VPY=<".pyexe"
if not defined VPY (
  echo Please run setup_conda.bat or setup_windows.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag one or more audio / video files or folders onto this file.
  echo Results TXT / SRT / MD are saved next to each file.
  pause
  exit /b 1
)
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
"%VPY%" stt.py run -i %* --formats txt,srt,md,json
pause
