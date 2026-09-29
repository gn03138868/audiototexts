@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Usagi STT V6 - download models
set "VPY="
if exist ".venv\Scripts\python.exe" set "VPY=%~dp0.venv\Scripts\python.exe"
if not defined VPY if exist ".pyexe" set /p VPY=<".pyexe"
if not defined VPY (
  echo Please run setup_conda.bat or setup_windows.bat first.
  pause
  exit /b 1
)
echo Pre-downloading models (optional; otherwise they download on first use).
echo Qwen3-ASR-1.7B + aligner about 6 GB, Whisper large-v3 about 3 GB.
"%VPY%" stt.py download --engine qwen3-asr-1.7b
"%VPY%" stt.py download --engine whisper-large-v3
pause
