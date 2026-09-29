@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Usagi STT V6 - keep this window open while using the program
set "VPY="
if exist ".venv\Scripts\python.exe" set "VPY=%~dp0.venv\Scripts\python.exe"
if not defined VPY if exist ".pyexe" set /p VPY=<".pyexe"
if defined VPY (
  "%VPY%" -c "import gradio, faster_whisper, numpy" >nul 2>&1
  if errorlevel 1 (
    echo The Python environment at %VPY% is incomplete.
    echo Please run setup_conda.bat ^(Anaconda^) or setup_windows.bat again.
    pause
    exit /b 1
  )
)
if not defined VPY (
  echo First run: the environment is not installed yet.
  echo  - If you use Anaconda: close this window and double-click setup_conda.bat
  echo  - Otherwise press any key to install now with setup_windows.bat
  pause
  call "%~dp0setup_windows.bat" nopause
  if errorlevel 1 (
    pause
    exit /b 1
  )
  set "VPY=%~dp0.venv\Scripts\python.exe"
)
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
echo.
echo Python: %VPY%
echo Starting ... the program window will open automatically (http://127.0.0.1:7861).
echo Keep this black window open while you use the program. Close it to quit.
echo.
"%VPY%" app.py
echo.
echo The program has stopped. If there is an error above, please take a screenshot.
pause
