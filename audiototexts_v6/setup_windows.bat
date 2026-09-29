@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Usagi STT setup
echo ============================================================
echo  Usagi STT V6 setup for Windows (Python venv)
echo  PyTorch 2.9.1 + CUDA 12.8  (RTX 5060 / 5080 / all RTX 20-50)
echo  NOTE: RTX 50 series needs NVIDIA driver 570 or newer.
echo  If you use Anaconda, close this window and run setup_conda.bat.
echo ============================================================
echo.

set "PYEXE="
call :findpy
if not defined PYEXE (
  echo Python 3.11 was not found. Installing it automatically ...
  where winget >nul 2>&1
  if not errorlevel 1 (
    winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
  )
  call :findpy
)
if not defined PYEXE (
  echo winget not available or failed - downloading the official installer from python.org ...
  curl.exe -L -o "%TEMP%\python-3.11.9-amd64.exe" https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
  if exist "%TEMP%\python-3.11.9-amd64.exe" (
    "%TEMP%\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
  )
  call :findpy
)
if not defined PYEXE (
  echo [ERROR] Could not install Python automatically.
  echo         Please install Python 3.11 from https://www.python.org/downloads/release/python-3119/
  echo         tick "Add python.exe to PATH", then run this again.
  goto :fail
)
echo Using Python: %PYEXE%

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment .venv ...
  %PYEXE% -m venv .venv
  if errorlevel 1 goto :fail
)
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo [1/4] Installing PyTorch 2.9.1 (CUDA 12.8, supports RTX 50 / Blackwell sm_120) - about 3 GB ...
"%VPY%" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

echo.
echo [2/4] Installing speech recognition packages (torch version locked) ...
"%VPY%" -m pip install -r requirements.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :fail

echo.
echo [3/4] Optional: Japanese word splitter for precise Japanese subtitle timing (nagisa) ...
"%VPY%" -m pip install nagisa -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 echo       nagisa could not be installed - Japanese subtitles will use estimated timing. This is OK.

echo.
echo [4/4] Environment check ...
"%VPY%" stt.py doctor
"%VPY%" stt.py shortcut

echo.
echo Setup finished. Double-click start_ui.bat (or the desktop shortcut) to open the program.
if /i not "%~1"=="nopause" pause
exit /b 0

:findpy
for %%V in (3.11 3.12 3.10) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
  )
)
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%ProgramFiles%\Python312\python.exe") do (
  if not defined PYEXE (
    if exist %%P set PYEXE=%%P
  )
)
if not defined PYEXE (
  python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" >nul 2>&1 && set "PYEXE=python"
)
exit /b 0

:fail
echo.
echo [ERROR] Setup did not finish. Please take a screenshot of this window.
if /i not "%~1"=="nopause" pause
exit /b 1
