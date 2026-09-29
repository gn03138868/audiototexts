# 🎙️🐰 音檔轉錄小兔歐 V6

音檔／影片轉逐字稿（TXT）、字幕（SRT）、含時間戳記的MD。
全部在本機GPU上執行，不會上傳任何錄音。

和 V5.10前版相比：

- **亂碼與重複詞大幅減少**：先用語音偵測（VAD）切掉靜音與音樂，再逐段檢查「重複迴圈、字數不合理、已知幻覺句、中日文字混亂、提示詞外洩」，有問題就整段重轉。
- **更快**：GPU 批次辨識。同一顆 GPU 上，faster-whisper 批次模式比 openai-whisper 快約 8 倍（官方測試：13 分鐘音檔 2 分 23 秒 → 17 秒）。
- **更準的中文**：新增 Qwen3-ASR-1.7B（2026 年開源，中文錯字率約為 Whisper large-v3 的一半）與 Breeze-ASR-25（聯發科，台灣華語＋中英夾雜）。
- 自動轉成繁體（台灣字形），支援專有名詞表與自動修正規則。
- 不需要自己放 ffmpeg.exe / ffprobe.exe；影片（mp4 / mov）可以直接丟。
- 支援 RTX 5060 / 5080（Blackwell）。
- 但目前還沒開發泰文模組，將來再看看，我沒有足夠好的server做這件事情。

---

## 1. 安裝（一次就好）

1. NVIDIA 驅動更新到 **570 以上**（RTX 50 系列必要）。
2. 用 **Anaconda** 的話：雙擊 **`setup_conda.bat`**。它會建立獨立的 `stt` 環境（Python 3.11），不會動到 base 或 vctts 的環境。
   沒有 Anaconda：雙擊 `setup_windows.bat`（會自動安裝 Python 3.11 並建立 `.venv`）。
3. 安裝完成後，桌面會出現「音檔轉錄小兔歐」捷徑。

第一次轉錄會自動下載模型（Qwen3-ASR-1.7B＋對齊模型約 6 GB；Whisper large-v3 約 3 GB）。
想先下載好，可以雙擊 `download_models.bat`。

## 2. 使用

雙擊桌面捷徑（或 `start_ui.bat`）→ 會開啟一個獨立視窗（黑色視窗是程式本體，使用中請不要關）。

| 分頁 | 做什麼 |
|---|---|
| ① 轉錄 | 上傳檔案，或貼上檔案／資料夾路徑（大檔案建議貼路徑，不用上傳）→ 開始。有進度條，可隨時停止。 |
| ② 專有名詞與修正表 | 課程術語、人名；以及 `炭匯 => 碳匯` 這類自動修正規則 |
| 說明 | 本文件 |

也可以把音檔**拖曳到 `transcribe_dragdrop.bat`** 上，用預設設定直接轉錄。

### 輸出檔案（預設存在原檔旁邊）

| 檔案 | 內容 |
|---|---|
| `名稱_逐字稿.txt` | 分段落的純文字 |
| `名稱_逐字稿.srt` | 字幕（中日文每行最多 20 字，可調） |
| `名稱_逐字稿.md` | 每段附時間；最後列出「⚠ 需要人工確認的片段」 |
| `名稱_逐字稿.json` | 每一段的原始輸出、語言、重轉紀錄（進階用） |

檔名都有 `_逐字稿`，不會覆蓋你原本的 `.srt`。

## 3. 選哪個引擎？

| 情境 | 建議 |
|---|---|
| 中文課堂、會議（台灣華語、中英夾雜） | **Qwen3-ASR-1.7B**（預設）或 **Breeze-ASR-25** |
| 日文、英文、其他語言 | Qwen3-ASR-1.7B 或 Whisper large-v3 |
| 幾個小時的錄音先看大意 | Whisper large-v3-turbo（最快） |
| 顯示卡 8 GB 以下 | Qwen3-ASR-0.6B 或 large-v3-turbo |

| 引擎 | 類型 | 優點 | 注意 |
|---|---|---|---|
| Qwen3-ASR-1.7B | 音訊編碼器＋Qwen3 語言模型 | 中文最準（WenetSpeech CER 4.97 vs Whisper-large-v3 9.86）；30 種語言＋22 種中文方言；會自己判斷每段的語言 | 需要 transformers ≥ 5.13；字幕時間用另一個 0.6B 對齊模型 |
| Whisper large-v3 | 編碼器－解碼器 | 最成熟；99 種語言 | 原生版本最容易幻覺，本程式已加上 VAD 與各種檢查 |
| Breeze-ASR-25 | Whisper large-v2 微調 | 台灣華語、中英夾雜（例如「這個 policy 的 baseline」） | 以中文為主的錄音效果最好 |
| large-v3-turbo | Whisper 精簡解碼器 | 最快 | 準確度略低 |

Qwen3-ASR 載入失敗時會自動改用 Whisper large-v3，不會中斷工作。

### 4. 語言設定

- **自動判斷（建議）**：Qwen 逐段判斷語言；Whisper 用最長的 12 段投票決定主要語言。
  只會在「允許的語言」（預設 中／日／英）中選，避免被判成韓文、粵語等造成亂碼。
- **混語**：中、日、英輪流說的會議。每段各自判斷語言；太短又不確定的段落沿用前後段的語言。
- **指定語言**：確定只有一種語言時最穩。


## 5. 業界目前的最佳做法＋加強

```
音檔 → 解碼 16 kHz → Silero VAD 找出說話區段 → 組成 ≤28 秒的語音段（<1.2 秒的段補上下文）
     → 語言判斷（限定允許清單）→ GPU 批次辨識（beam search 5）
     → 品質檢查 → 有問題的段落整段重轉（溫度遞增、重複懲罰、必要時換語言）
     → 去迴圈、刪幻覺句、相鄰重複段合併 → 簡轉繁、修正表 → 依標點與停頓切字幕
```

| 做法 | 參考 | 效果 |
|---|---|---|
| VAD 前處理（只送說話的部分進模型） | WhisperX、faster-whisper 批次模式 | 幻覺率 47% → 約 5%（WhisperX VAD，2026 研究比較） |
| 幻覺句清單＋去迴圈（Bag of Hallucinations） | Barański et al., ICASSP 2025 | 與 VAD 合用效果最好 |
| 不接上一段文字（condition_on_previous_text=False） | Whisper 官方參數說明 | 避免一段出錯後一路錯下去 |
| 壓縮比、平均對數機率、no-speech 機率 | Whisper 官方的品質門檻 | 抓出重複與沒人說話的段落 |
| 每秒音節數檢查（本程式新增） | — | 3 秒的語音卻輸出 80 個字 → 一定是幻覺 |
| 語言／文字系統檢查（本程式新增） | — | 中文段落出現大量假名、日文段落出現簡體字 → 換語言重轉 |
| 提示詞外洩檢查（本程式新增） | — | 模型把提示詞唸回來時會被抓出來 |

每段檢查的結果都寫在 JSON；重轉後仍有疑慮的段落會列在 MD 最後與介面下方的表格，方便對照原音檔確認。

## 6. 專有名詞與修正表（glossary.txt）

```
# 專有名詞：一行一個或逗號分隔 → 提示給模型
碳匯, 碳權, 自願減量, IPCC, REDD+
# 修正規則：錯誤 => 正確（轉錄後自動取代）
炭匯 => 碳匯
```

專有名詞不要放太多（建議 50 個以內），太長的提示反而可能被模型唸出來（程式會偵測並重轉）。

## 7. 進階設定

| 設定 | 預設 | 什麼時候調 |
|---|---|---|
| 語音偵測門檻 | 0.5 | 背景吵、漏句 → 0.35；雜音被當成說話 → 0.6 |
| 每段最長秒數 | 28 | 一般不用改 |
| 批次大小 | 自動 | 16 GB 顯示卡自動 16、8 GB 自動 8；出現記憶體不足時會自動減半 |
| 精準字幕時間 | 開 | Whisper 用字級時間戳記；Qwen 會多載入 0.6B 對齊模型（日文需要 nagisa，setup 會嘗試安裝） |
| 只轉前 N 分鐘 | 0 | 先試跑 3 分鐘確認語言與專有名詞 |

## 8. 指令列

```bat
conda activate stt
python stt.py run -i "C:\錄音\會議.m4a" "D:\課程錄影" --engine qwen3-asr-1.7b
python stt.py run -i 會議.m4a --engine breeze-asr-25 --language zh --limit-minutes 3
python stt.py run -i 資料夾 --language mixed --allowed zh,ja,en --skip-existing
python stt.py doctor --full        :: 檢查 GPU（會實際載入 Whisper tiny 測試）
python stt.py download --engine whisper-large-v3
```

## 9. 速度參考（估計值，實際視錄音內容而定）

以 2 小時課堂錄音估算：

- RTX 5080（16 GB）：Whisper large-v3 約 2–4 分鐘；Qwen3-ASR-1.7B 約 3–8 分鐘
- RTX 5060（8 GB）：約上面的 1.5–2 倍
- V5.10（openai-whisper、無批次）：同一顆 GPU 約 10–25 分鐘（再加上重轉時間）

## 10. 疑難排解

- **畫面上方顯示 GPU ❌**：更新 NVIDIA 驅動（≥ 570），再按「重新檢查」；或在 Anaconda Prompt 執行 `conda activate stt` → `python stt.py doctor --full`，把結果截圖。
- **`cublas64_12.dll` not found / `CUBLAS_STATUS_NOT_SUPPORTED`**：本程式會自動使用 PyTorch 內附的 CUDA 函式庫，並在 RTX 50 系列上使用 float16（避開 CTranslate2 舊版 int8 在 Blackwell 上的已知問題）。仍出現時重新執行 setup。
- **Qwen3-ASR 無法載入**：確認 `transformers` ≥ 5.13（`python stt.py doctor` 會顯示）。程式會自動改用 Whisper large-v3 繼續。
- **模型下載很慢或失敗**：學校網路擋 Hugging Face 時，換個網路先執行 `download_models.bat`。
- **某段還是怪怪的**：看 MD 最後的「⚠ 需要人工確認」；可以把正確寫法加進修正表，或換另一個引擎再轉一次比較。
- **中文變成簡體**：「中文輸出」選「繁體（台灣字形）」（預設）。日文段落不會被轉換。
- **漏掉很小聲的發言**：語音偵測門檻調到 0.35。
- **不小心關掉黑色視窗**：再點一次桌面捷徑即可。

---

參考資料：
Qwen3-ASR（github.com/QwenLM/Qwen3-ASR、arXiv 2601.21337）·
Breeze-ASR-25（huggingface.co/MediaTek-Research/Breeze-ASR-25）·
faster-whisper（github.com/SYSTRAN/faster-whisper）·
Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio（arXiv 2501.11378）·
Reducing Hallucinated Transcripts in Whisper via Hallucination Space Projection（arXiv 2609.04561）·
CTranslate2 changelog（RTX 50 系列 int8 問題）
