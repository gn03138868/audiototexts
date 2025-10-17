Audiototexts Tool

Overview

Audiototexts is a user-friendly graphical interface built with Python and Tkinter, designed to transcribe audio files efficiently using the Whisper model by OpenAI. It supports batch processing, automatic segmentation of large files, GPU acceleration, and advanced features like silence detection and repetition removal. The tool is tailored for users who need to convert audio files into text with minimal effort, offering a robust solution for both small and large-scale transcription tasks.

Features

Batch Processing: Transcribe multiple audio files in one go.
Large File Handling: Automatically splits large audio files into manageable chunks to prevent memory issues.
GPU Acceleration: Leverages CUDA-enabled GPUs for faster transcription (falls back to CPU if GPU is unavailable).
Silence Detection (VAD): Optionally skips silent segments to improve transcription quality (use with caution due to potential hallucination issues).
Repetition Detection: Automatically removes repetitive or looping content for cleaner output.
Multilingual Support: Supports transcription in multiple languages, including Chinese, English, Japanese, Korean, and automatic language detection.
User-Friendly Interface: Features a scrollable GUI with progress bars, file lists, and detailed logs.
Customisable Settings: Adjust model size, chunk length, and maximum file size for optimal performance.
Cross-Platform: Runs on Windows, macOS, and Linux (with some platform-specific dependencies).


Installation
Prerequisites

Python: Version 3.8 or higher.
FFmpeg: Required for audio file processing.
CUDA (optional): For GPU acceleration, ensure you have a compatible NVIDIA GPU and CUDA installed.

Steps

Install Python:

Download and install Python from python.org.
Ensure pip is available.


Install FFmpeg:

Download FFmpeg from gyan.dev (Windows) or install via package managers:
Windows: Extract ffmpeg.exe to the same folder as the script or add it to your system PATH.
macOS: brew install ffmpeg
Linux: sudo apt-get install ffmpeg (Ubuntu/Debian) or equivalent for your distribution.




Install Python Dependencies:Run the following command in your terminal or command prompt:
pip install whisper torch tkinter pydub


Ensure torch is installed with CUDA support for GPU acceleration (e.g., pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 for CUDA 11.8).


Download the Script:

Save the provided Python script (e.g., audiototexts_v0.31.py) to your desired directory.




Usage

Run the Application:Execute the script using Python:
python audiototexts_v0.31.py


Interface Overview:

Input Folder: Select a folder containing audio files (supported formats: MP3, WAV, M4A, FLAC, AAC, OGG, WMA, Opus).
Output Folder: Choose where to save the transcribed text files.
Model Selection: Choose from Tiny, Base, Small, Medium, or Large models (Medium is recommended for balance).
Language: Select the audio language or use "Auto" for automatic detection.
Hardware Settings: Enable/disable GPU acceleration (if available).
File Splitting: Set maximum file size (MB) and chunk length (minutes) for large files.
Quality Options: Enable silence detection (VAD) and repetition detection for cleaner results.
Progress and Logs: Monitor transcription progress and view detailed logs in the interface.


Start Transcription:

Click "Start Batch Transcription" to process all audio files in the selected folder.
Results are saved as text files (full transcript) and Markdown files (segmented transcript with timestamps) in the output folder.


Stop or Monitor:

Use the "Stop" button to halt processing.
Check memory status (GPU-enabled systems) to monitor resource usage.




Output Format

Full Transcript: Saved as <filename>_transcript.txt containing the complete transcribed text.
Segmented Transcript: Saved as <filename>_segments.md with timestamps, detected language, and segment details.

Example Markdown output:
# Audio Transcription Result

**File**: example_audio.mp3
**Language**: Chinese
**Size**: 25.50 MB
**Segments**: 10
**Duration**: 00:05:30

---

**[00:00:00 → 00:00:05]**

Hello, this is a test audio.

---

**[00:00:06 → 00:00:10]**

This is the second segment.

---


Troubleshooting

FFmpeg Not Found:
Ensure FFmpeg is installed and accessible in your system PATH or placed in the script's directory.


GPU Not Detected:
Verify that CUDA is installed and compatible with your GPU.
Check if torch.cuda.is_available() returns True in Python.


Memory Issues:
Reduce the model size (e.g., use Tiny or Base) or lower the chunk length for large files.


Slow Transcription:
Enable GPU acceleration if available, or use a smaller model for faster processing.


Hallucinations in Transcription:
Disable VAD (silence detection) if the model produces unexpected or repetitive text.




Notes

Performance: GPU acceleration significantly speeds up transcription, but CPU mode is supported for systems without NVIDIA GPUs.
Large Files: Files exceeding the specified size (default: 200 MB) are automatically split into chunks (default: 10 minutes) to avoid memory issues.
Temporary Files: The tool creates a temp_chunks folder for processing large files, which is automatically cleaned up after transcription.
Memory Management: The tool includes memory cleanup to prevent crashes during batch processing.


License
This project is licensed under the MIT License.

音檔轉文字小兔歐 (audiototexts)

概述

音檔轉文字小兔歐 (audiototexts) 是一個基於 Python 和 Tkinter 開發的圖形化介面工具，旨在使用 OpenAI 的 Whisper 模型高效轉錄音檔。它支援批次處理、大型檔案自動切割、GPU 加速，以及靜音偵測和重複內容移除等進階功能。此工具專為需要將音檔轉為文字的用戶設計，提供簡單且強大的解決方案，適用於小型和大型轉錄任務。

功能

批次處理：一次轉錄多個音檔。
大型檔案處理：自動將大型音檔切割為可管理的片段，以避免記憶體問題。
GPU 加速：支援 CUDA 的 GPU 可大幅加速轉錄（無 GPU 時自動切換至 CPU）。
靜音偵測 (VAD)：可選擇跳過靜音片段以提升轉錄品質（請謹慎使用，因可能導致模型產生幻覺）。
重複偵測：自動移除重複或循環內容，生成更乾淨的輸出。
多語言支援：支援多種語言轉錄，包括中文、英文、日文、韓文及自動語言偵測。
友善介面：提供可滾動的圖形介面，包含進度條、檔案列表和詳細日誌。
自訂設定：可調整模型大小、片段長度和最大檔案大小以最佳化效能。
跨平台：支援 Windows、macOS 和 Linux（部分平台需特定依賴項）。


安裝
前置條件

Python：版本 3.8 或更高。
FFmpeg：音檔處理所需的工具。
CUDA（可選）：若需 GPU 加速，確保有相容的 NVIDIA GPU 和 CUDA。

步驟

安裝 Python：

從 python.org 下載並安裝 Python。
確保 pip 可用。


安裝 FFmpeg：

Windows：從 gyan.dev 下載 FFmpeg，解壓後將 ffmpeg.exe 放入腳本所在資料夾或加入系統 PATH。
macOS：使用 brew install ffmpeg。
Linux：使用 sudo apt-get install ffmpeg（Ubuntu/Debian）或對應的套件管理指令。


安裝 Python 依賴項：在終端機或命令提示字元執行：
pip install whisper torch tkinter pydub


若需 GPU 加速，確保安裝支援 CUDA 的 torch，例如：pip install torch --extra-index-url https://download.pytorch.org/whl/cu118




下載腳本：

將提供的 Python 腳本（例如 audiototexts_v0.31.py）儲存到指定資料夾。




使用方法

啟動應用程式：使用 Python 執行腳本：
python audiototexts_v0.31.py


介面概覽：

輸入資料夾：選擇包含音檔的資料夾（支援格式：MP3、WAV、M4A、FLAC、AAC、OGG、WMA、Opus）。
輸出資料夾：選擇儲存轉錄文字檔的資料夾。
模型選擇：從 Tiny、Base、Small、Medium 或 Large 模型中選擇（推薦 Medium 以平衡速度與準確度）。
語言：選擇音檔語言或使用「自動」進行語言偵測。
硬體設定：啟用/停用 GPU 加速（若可用）。
檔案切割：設定最大檔案大小（MB）和片段長度（分鐘）以處理大型檔案。
品質選項：啟用靜音偵測 (VAD) 和重複偵測以獲得更乾淨的結果。
進度與日誌：監控轉錄進度和查看介面中的詳細日誌。


開始轉錄：

點擊「開始批次轉錄」以處理選擇資料夾中的所有音檔。
結果將儲存為文字檔（完整轉錄）和 Markdown 檔（包含時間戳的分段轉錄）。


停止或監控：

使用「停止」按鈕中止處理。
在支援 GPU 的系統上，檢查記憶體狀態以監控資源使用情況。




輸出格式

完整轉錄：儲存為 <檔名>_transcript.txt，包含完整的轉錄文字。
分段轉錄：儲存為 <檔名>_segments.md，包含時間戳、偵測語言和分段詳細資訊。

範例 Markdown 輸出：
# 音檔轉錄結果

**檔案**：example_audio.mp3
**語言**：中文
**大小**：25.50 MB
**片段**：10
**時長**：00:05:30

---

**[00:00:00 → 00:00:05]**

您好，這是一個測試音檔。

---

**[00:00:06 → 00:00:10]**

這是第二個片段。

---


疑難排解

找不到 FFmpeg：
確保 FFmpeg 已安裝並可透過系統 PATH 或腳本資料夾存取。


未偵測到 GPU：
確認已安裝 CUDA 並與 GPU 相容。
在 Python 中檢查 torch.cuda.is_available() 是否回傳 True。


記憶體問題：
減小模型大小（例如使用 Tiny 或 Base）或縮短大型檔案的片段長度。


轉錄速度慢：
若有 GPU，啟用 GPU 加速；或使用較小的模型以加快處理速度。


轉錄出現幻覺：
若模型產生意外或重複文字，可停用靜音偵測 (VAD)。




注意事項

效能：GPU 加速可顯著提升轉錄速度，但無 NVIDIA GPU 的系統也可使用 CPU 模式。
大型檔案：超過指定大小（預設：200 MB）的檔案將自動切割為片段（預設：10 分鐘）以避免記憶體問題。
暫存檔案：工具會建立 temp_chunks 資料夾處理大型檔案，轉錄完成後會自動清理。
記憶體管理：工具內建記憶體清理功能，以防止批次處理時發生崩潰。


授權
本專案採用 MIT 授權。
