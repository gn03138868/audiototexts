@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Usagi STT setup (Anaconda)
echo ============================================================
echo  Usagi STT V6 setup for Anaconda / Miniconda
echo  Creates a separate conda env "stt" (Python 3.11), so your
echo  base env and the "vctts" env are not changed.
echo  PyTorch 2.9.1 + CUDA 12.8 (RTX 5060 / 5080 / all RTX 20-50)
echo ============================================================
echo.

set "CONDA="
for %%P in ("%USERPROFILE%\anaconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\condabin\conda.bat" "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" "%LOCALAPPDATA%\miniconda3\condabin\conda.bat" "%ProgramData%\anaconda3\condabin\conda.bat" "%ProgramData%\miniconda3\condabin\conda.bat" "C:\anaconda3\condabin\conda.bat") do (
  if not defined CONDA (
    if exist %%P set CONDA=%%P
  )
)
if not defined CONDA (
  where conda >nul 2>&1 && set "CONDA=conda"
)
if not defined CONDA (
  echo [ERROR] conda was not found. Open "Anaconda Prompt", cd to this folder and run setup_conda.bat there.
  goto :fail
)
echo Using conda: %CONDA%

call %CONDA% run -n stt python -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo Creating conda env "stt" with Python 3.11 ...
  call %CONDA% create -n stt python=3.11 -y
  if errorlevel 1 goto :fail
)

set "VPY="
for /f "usebackq delims=" %%i in (`call %CONDA% run -n stt python -c "import sys;print(sys.executable)"`) do set "VPY=%%i"
if not exist "%VPY%" (
  echo [ERROR] Could not locate python.exe of env "stt".
  goto :fail
)
echo Env python: %VPY%
> ".pyexe" echo %VPY%

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
echo Setup finished. Double-click start_ui.bat (or the desktop shortcut) to open the program,
echo or in Anaconda Prompt:  conda activate stt  then  python app.py
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup did not finish. Please take a screenshot of this window.
pause
exit /b 1
