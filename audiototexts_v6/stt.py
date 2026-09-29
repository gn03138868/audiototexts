#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音檔轉錄小兔歐 V6 — 核心引擎與指令列

流程（每個檔案）：
  1. 解碼音訊（PyAV，影片也可以；不需要另外安裝 ffmpeg）
  2. Silero VAD 找出有人說話的區段 → 組成 ≤28 秒的語音段（靜音、音樂不送進模型）
  3. 語言判斷：整檔一種語言，或「混語」逐段判斷（只在允許的語言中選，例如 中/日/英）
  4. GPU 批次辨識（Whisper / Breeze-ASR-25 用 faster-whisper；Qwen3-ASR 用 transformers）
  5. 品質檢查：迴圈重複、每秒字數不合理、壓縮比、已知幻覺句、語言/文字系統不符、提示詞外洩
     → 有問題的語音段整段重跑（溫度遞增、重複懲罰），仍有問題就去除迴圈並標記 ⚠
  6. 後處理：去迴圈、去幻覺句、簡轉繁（OpenCC 台灣正體）、專有名詞修正表
  7. 輸出 TXT / SRT / MD / JSON

用法：
  python stt.py doctor                 檢查 GPU 與套件
  python stt.py run -i 音檔或資料夾 ...  轉錄
  python stt.py download --engine qwen3-asr-1.7b   預先下載模型
"""
from __future__ import annotations

import argparse
import difflib
import gc
import json
import os
import re
import sys
import time
import traceback
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "音檔轉錄小兔歐"
VERSION = "6.0"
APP_DIR = Path(__file__).resolve().parent
SR = 16000
OUT_SUFFIX = "_逐字稿"

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".opus", ".webm",
              ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".3gp", ".amr", ".aiff", ".aif"}

# ════════════════════════════════════════════════════════════════════
# 引擎清單
# ════════════════════════════════════════════════════════════════════
ENGINES: dict[str, dict] = {
    "qwen3-asr-1.7b": dict(kind="qwen", model="Qwen/Qwen3-ASR-1.7B-hf",
                           label="Qwen3-ASR-1.7B（中文最準・含日英與方言・建議）", size_gb=4.7),
    "whisper-large-v3": dict(kind="whisper", model="large-v3",
                             label="Whisper large-v3（多語通用・最成熟）", size_gb=3.1),
    "breeze-asr-25": dict(kind="whisper", model="SoybeanMilk/faster-whisper-Breeze-ASR-25",
                          label="Breeze-ASR-25（台灣華語＋中英夾雜）", size_gb=3.1),
    "whisper-large-v3-turbo": dict(kind="whisper", model="large-v3-turbo",
                                   label="Whisper large-v3-turbo（最快・準確度略低）", size_gb=1.6),
    "qwen3-asr-0.6b": dict(kind="qwen", model="Qwen/Qwen3-ASR-0.6B-hf",
                           label="Qwen3-ASR-0.6B（輕量・適合 8 GB 以下顯示卡）", size_gb=1.9),
    "dummy": dict(kind="dummy", model="", label="測試模式（不載入模型，用來檢查流程）", size_gb=0),
}
DEFAULT_ENGINE = "qwen3-asr-1.7b"
FALLBACK_ENGINE = "whisper-large-v3"
QWEN_ALIGNER = os.environ.get("STT_QWEN_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B-hf")

LANG_NAMES = {"zh": "中文", "ja": "日文", "en": "英文", "yue": "粵語", "ko": "韓文", "fr": "法文",
              "de": "德文", "es": "西班牙文", "th": "泰文", "vi": "越南文", "id": "印尼文",
              "ms": "馬來文", "ru": "俄文", "it": "義大利文", "pt": "葡萄牙文"}
QWEN_NAME_TO_CODE = {"chinese": "zh", "english": "en", "japanese": "ja", "cantonese": "yue", "korean": "ko",
                     "french": "fr", "german": "de", "spanish": "es", "thai": "th", "vietnamese": "vi",
                     "indonesian": "id", "malay": "ms", "russian": "ru", "italian": "it", "portuguese": "pt",
                     "arabic": "ar", "hindi": "hi", "turkish": "tr", "dutch": "nl", "filipino": "fil"}
ALIGNER_LANGS = {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}

# ════════════════════════════════════════════════════════════════════
# 小工具
# ════════════════════════════════════════════════════════════════════
_DLL_HANDLES = []


def setup_stdio():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:  # noqa: BLE001
            pass


def prepare_cuda_dlls():
    """Windows：讓 CTranslate2 找得到 PyTorch 內附的 cuBLAS / cuDNN（cublas64_12.dll 等）。"""
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401  (import 時會把 torch\\lib 加入 DLL 搜尋路徑)
        lib = Path(torch.__file__).parent / "lib"
    except Exception:  # noqa: BLE001
        return
    dirs = [lib] + sorted((lib.parent.parent / "nvidia").glob("*/bin"))  # 另外裝的 nvidia-cublas-cu12 等
    for d in dirs:
        if d.is_dir():
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(d)))
            except Exception:  # noqa: BLE001
                pass
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def emit(kind: str, **kw):
    """給介面解析的進度訊息（一行 JSON）；只有從圖形介面執行時才輸出。"""
    if os.environ.get("STT_UI") != "1":
        return
    kw["kind"] = kind
    print("@@" + json.dumps(kw, ensure_ascii=False), flush=True)


def log(msg: str = ""):
    print(msg, flush=True)


def fmt_ts(t: float, sep: str = ",") -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_hms(t: float) -> str:
    t = int(max(0, t))
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def parse_time(s: str | None) -> float:
    if not s:
        return 0.0
    parts = [float(p) for p in str(s).strip().split(":")]
    v = 0.0
    for p in parts:
        v = v * 60 + p
    return v


# ── 文字分類 ───────────────────────────────────────────────────────
RE_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ebef]")
RE_KANA = re.compile(r"[぀-ゟ゠-ヿㇰ-ㇿｦ-ﾟ]")
RE_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ]")
RE_LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’\-][A-Za-zÀ-ɏ]+)*")
RE_DIGITS = re.compile(r"\d")
CJK_PUNCT = "，。！？、；：「」『』（）《》〈〉【】…—～·"
STRONG_BREAK = set("。！？!?；;")
WEAK_BREAK = set("，,、：:")
# 簡體特有字（日文不會用）：日文段落出現這些字 → 多半是中日混亂
SIMPLIFIED_ONLY = set("这们么对该让给为说时过还进发经问题现无样见吗呢觉认实头边关应热专业东车马门话语读书买卖长开间听页风飞个两从电动气难谁这样怎么")


def is_cjk_char(ch: str) -> bool:
    return bool(RE_HAN.match(ch) or RE_KANA.match(ch) or RE_HANGUL.match(ch))


def syllables(text: str) -> float:
    """估計唸完這段文字需要的音節數（中文字≈1、假名≈1、漢字(日)≈1.5、英文單字≈1.4、數字≈1）。"""
    han = len(RE_HAN.findall(text))
    kana = len(RE_KANA.findall(text))
    hangul = len(RE_HANGUL.findall(text))
    words = len(RE_LATIN_WORD.findall(text))
    digits = len(RE_DIGITS.findall(text))
    han_w = 1.5 if kana > han * 0.3 else 1.0
    return han * han_w + kana + hangul + words * 1.4 + digits


def display_units(text: str) -> float:
    """字幕長度：中日韓字 = 1，其他字元 = 0.5。"""
    return sum(1.0 if is_cjk_char(c) else 0.5 for c in text if not c.isspace()) + 0.5 * text.count(" ")


def compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def norm_key(text: str) -> str:
    """比對用：小寫、去空白與標點。"""
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                   if unicodedata.category(c)[0] in "LN")


def join_texts(a: str, b: str, lang: str | None = None, gap: float = 0.0) -> str:
    """接兩段文字：中日文不加空白；前一段沒有句尾標點時，依停頓長短補「。」或「，／、」。"""
    if not a:
        return b
    if not b:
        return a
    if is_cjk_char(a[-1]) and is_cjk_char(b[0]) and lang in ("zh", "yue", "ja"):
        if gap >= 1.2:
            return a + "。" + b
        return a + ("、" if lang == "ja" else "，") + b
    if is_cjk_char(a[-1]) or a[-1] in CJK_PUNCT or is_cjk_char(b[0]) or b[0] in CJK_PUNCT:
        return a + b
    return a + " " + b


def tidy_spaces(text: str) -> str:
    """整理空白：中日文字之間不留空白（Whisper 常在中文字間插空白）。"""
    text = re.sub(r"\s+", " ", text).strip()
    cjk = r"[぀-ヿ㐀-䶿一-鿿豈-﫿，。！？、；：「」『』（）《》…]"
    text = re.sub(rf"(?<={cjk}) (?={cjk})", "", text)
    return text


# ════════════════════════════════════════════════════════════════════
# 去迴圈（重複詞 / 重複句）
# ════════════════════════════════════════════════════════════════════
def _unit_core_len(unit: str) -> int:
    return len(norm_key(unit))


def find_loop_spans(text: str, max_unit: int = 40):
    """找出 [(start, end, unit, count)]：同一單位連續重複到「不像人話」的程度。

    規則：單一字元連續 ≥6 次；2–3 字的詞連續 ≥4 次；4 字以上的片語/句子連續 ≥3 次。
    例：「對對對」「哈哈哈」保留；「我們我們我們我們」「的的的的的的」視為迴圈。
    """
    spans = []
    n = len(text)
    i = 0
    while i < n:
        best = None
        for L in range(1, min(max_unit, (n - i + 1) // 2) + 1):
            unit = text[i:i + L]
            core = _unit_core_len(unit)
            if core == 0:
                continue
            need = 6 if core == 1 else (4 if core <= 3 else 3)
            if i + L * (need - 1) + len(unit.rstrip()) > n:
                continue
            cnt = 1
            j = i + L
            while text.startswith(unit, j):
                cnt += 1
                j += L
            tail = unit.rstrip()
            if cnt >= need - 1 and tail != unit and text.startswith(tail, j) and not text[j + len(tail):j + len(tail) + 1].isalnum():
                cnt += 1  # 最後一次重複後面沒有空白（例如句尾）
                j += len(tail)
            if cnt >= need:
                cover = j - i
                if best is None or cover > best[1] - best[0]:
                    best = (i, j, unit, cnt)
        if best:
            spans.append(best)
            i = best[1]
        else:
            i += 1
    return spans


def deloop(text: str) -> tuple[str, int]:
    """把迴圈縮成 1 次（單一字元保留 2 次）。回傳 (新文字, 修改處數)。"""
    spans = find_loop_spans(text)
    if not spans:
        text2 = re.sub(r"([，。、！？,.!?；;])\1{2,}", r"\1", text)
        return text2, int(text2 != text)
    out, pos = [], 0
    for s, e, unit, cnt in spans:
        out.append(text[pos:s])
        out.append(unit * (2 if _unit_core_len(unit) == 1 else 1))
        pos = e
    out.append(text[pos:])
    new = "".join(out)
    new = re.sub(r"([，。、！？,.!?；;])\1{2,}", r"\1", new)
    # 句子層級的迴圈縮完後可能又組成新的迴圈（例如 A B A B A B）→ 再跑一次
    new2, more = deloop(new) if new != text else (new, 0)
    return new2, len(spans) + more


# ════════════════════════════════════════════════════════════════════
# 已知幻覺句（Bag of Hallucinations）
# ════════════════════════════════════════════════════════════════════
# 「強・可刪子字串」：這些字幕組/影片平台的句子不可能真的出現在課堂/會議中 → 不論在哪裡出現都刪掉
BOH_STRONG_SUB = [
    "字幕由Amara.org社區提供", "字幕由Amara.org社区提供", "Amara.org社區提供", "Amara.org社区提供",
    "subtitles by the amara.org community", "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目", "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "明鏡與點點欄目", "明镜与点点栏目", "點贊訂閱轉發打賞", "点赞订阅转发打赏", "中文字幕志願者", "中文字幕志愿者",
    "優優獨播劇場", "优优独播剧场", "YoYo Television Series Exclusive", "チャンネル登録よろしくお願いします",
]
# 「強・整段」：只有整段幾乎就是這句時才刪（這些詞在正常內容裡也可能出現）
BOH_STRONG_WHOLE = BOH_STRONG_SUB + [
    "字幕志願者", "字幕志愿者", "詞曲李宗盛", "词曲李宗盛", "請訂閱我的頻道", "请订阅我的频道", "歡迎訂閱", "欢迎订阅",
    "訂閱頻道", "订阅频道", "字幕製作", "字幕制作", "時間軸校對", "ytb字幕", "チャンネル登録", "please subscribe",
    "like and subscribe", "don't forget to subscribe", "subscribe to my channel", "transcribed by", "subtitles by",
    "captions by", "transcription by", "otter.ai", "rev.com",
]
# 「弱」：真的可能有人說，只有在「整段只有這句」且模型信心低/段落很短時才刪
BOH_WEAK = [
    "謝謝觀看", "谢谢观看", "感謝觀看", "感谢观看", "謝謝收看", "谢谢收看", "感謝收看", "感谢收看", "謝謝大家",
    "谢谢大家", "謝謝", "谢谢", "感謝您的觀看", "嗯", "啊", "呃", "我們下期再見", "下次見",
    "ご視聴ありがとうございました", "ありがとうございました", "ありがとうございます", "おやすみなさい",
    "お疲れ様でした", "thank you", "thank you very much", "thanks for watching", "thank you for watching",
    "thanks", "you", "so", "bye", "the end", "music", "applause", "laughter",
]


class Boh:
    def __init__(self):
        self.strong = sorted({norm_key(p) for p in BOH_STRONG_WHOLE if norm_key(p)}, key=len, reverse=True)
        self.sub = sorted({norm_key(p) for p in BOH_STRONG_SUB if norm_key(p)}, key=len, reverse=True)
        self.weak = {norm_key(p) for p in BOH_WEAK if norm_key(p)}
        # 用於刪除子字串：每個字元之間允許空白/標點
        self.sub_res = []
        for p in sorted(BOH_STRONG_SUB, key=len, reverse=True):
            chars = [re.escape(c) for c in p if not c.isspace()]
            if chars:
                self.sub_res.append(re.compile(r"[\s\W]*".join(chars) + r"[\s。．.!！]*", re.I))

    def is_strong_only(self, text: str) -> bool:
        k = norm_key(text)
        return bool(k) and any(p in k and len(k) <= len(p) * 1.5 + 4 for p in self.strong)

    def contains_sub(self, text: str) -> bool:
        k = norm_key(text)
        return any(p in k for p in self.sub)

    def remove_strong(self, text: str) -> tuple[str, int]:
        n = 0
        for r in self.sub_res:
            text, c = r.subn("", text)
            n += c
        return text.strip(" ，,、"), n

    def is_weak_only(self, text: str) -> bool:
        return norm_key(text) in self.weak


BOH = Boh()


# ════════════════════════════════════════════════════════════════════
# 品質檢查
# ════════════════════════════════════════════════════════════════════
MAX_SYL_PER_SEC = {"ja": 13.0, "default": 11.0}


def assess(text: str, lang: str | None, speech_dur: float, *, avg_logprob=None, no_speech_prob=None,
           prompt: str | None = None, hit_cap: bool = False) -> dict:
    """回傳 {"flags": [...], "drop": bool, "score": float}（score 越大越好）。"""
    flags: list[str] = []
    t = text.strip()
    if not t:
        if speech_dur >= 2.5:
            flags.append("empty")
        return {"flags": flags, "drop": False, "score": -1.0 if flags else 0.0}

    # 1) Whisper 標準的「這段其實沒人說話」判斷
    if no_speech_prob is not None and avg_logprob is not None and no_speech_prob > 0.6 and avg_logprob < -1.0:
        return {"flags": ["no_speech"], "drop": True, "score": -9.0}
    # 2) 幻覺句（字幕組/平台句子先刪掉，剩下沒東西 → 整段丟棄）
    t_sub, n_sub = BOH.remove_strong(t)
    if (n_sub and not norm_key(t_sub)) or BOH.is_strong_only(t_sub):
        return {"flags": ["hallucination"], "drop": True, "score": -9.0}
    t = t_sub
    if BOH.is_weak_only(t):
        weak_evidence = (no_speech_prob is not None and no_speech_prob > 0.25) or \
                        (avg_logprob is not None and avg_logprob < -0.75) or speech_dur < 1.0
        if weak_evidence:
            return {"flags": ["hallucination"], "drop": True, "score": -9.0}
    k = norm_key(t)

    # 3) 迴圈、壓縮比
    if find_loop_spans(t):
        flags.append("loop")
    if len(t.encode("utf-8")) > 60 and compression_ratio(t) > 2.4:
        flags.append("compression")
    # 4) 每秒音節數不合理（模型「編」出比實際更多的內容）
    syl = syllables(t)
    limit = MAX_SYL_PER_SEC.get(lang or "", MAX_SYL_PER_SEC["default"])
    if syl > max(limit * max(speech_dur, 0.0), 0) + 12:
        flags.append("too_dense")
    if hit_cap:
        flags.append("too_long")
    # 5) 語言與文字系統不符
    han, kana = len(RE_HAN.findall(t)), len(RE_KANA.findall(t))
    words = len(RE_LATIN_WORD.findall(t))
    if lang == "zh" and kana >= 3 and kana > 0.15 * (han + kana):
        flags.append("script")
    elif lang == "ja" and (((han + kana) >= 12 and kana == 0) or sum(ch in SIMPLIFIED_ONLY for ch in t) >= 2):
        flags.append("script")
    elif lang == "en" and (han + kana) > max(4, words * 0.5):
        flags.append("script")
    # 6) 提示詞外洩（把 initial prompt 唸回來）
    if prompt:
        pk = norm_key(prompt)
        if len(k) >= 8 and (k in pk or (len(pk) >= 8 and pk[:16] in k)):
            flags.append("prompt_echo")
    # 7) 信心度
    if avg_logprob is not None and avg_logprob < -1.0:
        flags.append("low_conf")

    weights = {"loop": 5, "too_dense": 4, "compression": 3, "too_long": 3, "script": 3, "prompt_echo": 5,
               "hallucination": 4, "low_conf": 1, "empty": 2}
    score = -sum(weights.get(f, 1) for f in flags) + (avg_logprob or 0.0) * 0.5
    return {"flags": flags, "drop": False, "score": score}


RETRY_FLAGS = {"loop", "too_dense", "compression", "too_long", "script", "prompt_echo", "hallucination", "empty"}
FLAG_TEXT = {"loop": "重複迴圈", "too_dense": "字數異常多", "compression": "壓縮比異常", "too_long": "輸出過長",
             "script": "語言/文字不符", "prompt_echo": "提示詞外洩", "hallucination": "疑似幻覺句",
             "low_conf": "信心度低", "empty": "有聲音但沒辨識出文字", "no_speech": "非語音"}


# ════════════════════════════════════════════════════════════════════
# 中文字形轉換 & 修正表
# ════════════════════════════════════════════════════════════════════
class ScriptConverter:
    MODES = {"traditional": "s2tw", "traditional_tw": "s2twp", "simplified": "t2s", "none": None}

    def __init__(self, mode: str):
        self.mode = mode
        self.cc = None
        cfg = self.MODES.get(mode)
        if cfg:
            try:
                import opencc
                self.cc = opencc.OpenCC(cfg)
            except Exception as e:  # noqa: BLE001
                log(f"   ⚠ 無法載入 OpenCC（{e}），中文不做繁簡轉換。請執行 pip install opencc")

    def __call__(self, text: str, lang: str | None) -> str:
        if self.cc is None or lang not in ("zh", "yue", None):
            return text
        if lang is None and RE_KANA.search(text):
            return text  # 日文不可轉，會改掉漢字
        return self.cc.convert(text)


@dataclass
class Glossary:
    terms: list[str] = field(default_factory=list)
    replacements: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None) -> "Glossary":
        g = cls()
        if not path or not Path(path).exists():
            return g
        for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                a, b = (x.strip() for x in line.split("=>", 1))
                if a:
                    g.replacements.append((a, b))
            else:
                g.terms.extend(t.strip() for t in re.split(r"[，,、;；]", line) if t.strip())
        return g

    def apply(self, text: str) -> str:
        for a, b in self.replacements:
            if re.fullmatch(r"[A-Za-z0-9 .\-]+", a):
                text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", b, text)
            else:
                text = text.replace(a, b)
        return text


def whisper_prompt(lang: str | None, zh_mode: str, glossary: Glossary) -> str | None:
    terms = "、".join(glossary.terms[:40])
    if lang == "zh":
        head = "以下是繁體中文的課堂或會議錄音逐字稿，使用全形標點符號。" if zh_mode.startswith("traditional") \
            else "以下是普通话的录音逐字稿，使用标点符号。"
        return head + (f"專有名詞：{terms}。" if terms else "")
    if lang == "ja":
        return "以下は日本語の講義・会議の書き起こしです。句読点を使います。" + (f"用語：{terms}。" if terms else "")
    if lang == "en":
        return "The following is a transcript of a lecture or meeting." + (
            f" Terms: {', '.join(glossary.terms[:40])}." if glossary.terms else "")
    return ("Terms: " + ", ".join(glossary.terms[:40]) + ".") if glossary.terms else None


# ════════════════════════════════════════════════════════════════════
# 音訊：解碼、VAD、切段
# ════════════════════════════════════════════════════════════════════
def load_audio(path: str | Path):
    import numpy as np
    try:
        from faster_whisper.audio import decode_audio
        a = decode_audio(str(path), sampling_rate=SR)
        return np.ascontiguousarray(a, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        log(f"   ⚠ PyAV 解碼失敗（{e}），改用 ffmpeg …")
    import subprocess
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        ff = "ffmpeg"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    p = subprocess.run([ff, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, creationflags=flags)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"無法讀取音訊：{p.stderr.decode('utf-8', 'replace')[-400:]}")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


@dataclass
class Chunk:
    idx: int
    start: float
    end: float
    regions: list[tuple[float, float]]
    lang: str | None = None
    lang_probs: dict = field(default_factory=dict)
    text: str = ""
    raw_text: str = ""
    pieces: list[tuple[str, float, float]] = field(default_factory=list)  # 對應 raw_text 的時間片段
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    hit_cap: bool = False
    flags: list[str] = field(default_factory=list)
    dropped: bool = False
    retried: int = 0
    fixed: list[str] = field(default_factory=list)
    text_for_align: str = ""
    align_words: list | None = None

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def speech_dur(self) -> float:
        return sum(e - s for s, e in self.regions)


def vad_regions(audio, threshold: float = 0.5, max_chunk: float = 28.0) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    opts = VadOptions(threshold=threshold, min_speech_duration_ms=200, max_speech_duration_s=max_chunk,
                      min_silence_duration_ms=300, speech_pad_ms=200)
    ts = get_speech_timestamps(audio, opts)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def build_chunks(regions, total_dur: float, max_chunk: float = 28.0, max_gap: float = 1.5,
                 min_len: float = 1.2) -> list[Chunk]:
    """把 VAD 區段組成語音段：相鄰且停頓 ≤max_gap 秒的合併，總長 ≤max_chunk；太短的段往前後補上下文。"""
    groups: list[list[tuple[float, float]]] = []
    for s, e in regions:
        if groups:
            gs = groups[-1][0][0]
            ge = groups[-1][-1][1]
            if s - ge <= max_gap and e - gs <= max_chunk:
                groups[-1].append((s, e))
                continue
        groups.append([(s, e)])
    chunks: list[Chunk] = []
    for g in groups:
        s, e = g[0][0], g[-1][1]
        chunks.append(Chunk(idx=len(chunks), start=s, end=e, regions=g))
    # 太短（<1.2 秒）的段：Whisper 對 <1 秒的片段特別容易幻覺 → 用前後的實際音訊補到 min_len
    for i, c in enumerate(chunks):
        if c.dur >= min_len:
            continue
        lo = chunks[i - 1].end if i > 0 else 0.0
        hi = chunks[i + 1].start if i + 1 < len(chunks) else total_dur
        need = min_len - c.dur
        ns, ne = max(lo, c.start - need / 2), min(hi, c.end + need / 2)
        short = min_len - (ne - ns)
        if short > 0:
            ns = max(lo, ns - short)
            short = min_len - (ne - ns)
            if short > 0:
                ne = min(hi, ne + short)
        c.start, c.end = ns, ne
    return chunks


# ════════════════════════════════════════════════════════════════════
# 時間對應：把「文字中的每個字」對到時間 → 切字幕
# ════════════════════════════════════════════════════════════════════
def char_times_from_pieces(text: str, pieces: list[tuple[str, float, float]]):
    """pieces 串起來 == text（或大致相同）。回傳每個字元的 (start, end)。"""
    times = []
    for ptxt, t0, t1 in pieces:
        n = len(ptxt)
        if n == 0:
            continue
        w = [0.0 if c.isspace() else (1.0 if is_cjk_char(c) else 0.4) for c in ptxt]
        tot = sum(w) or 1.0
        acc = 0.0
        for k in range(n):
            a = t0 + (t1 - t0) * acc / tot
            acc += w[k]
            b = t0 + (t1 - t0) * acc / tot
            times.append((a, b))
    joined = "".join(p[0] for p in pieces)
    if joined == text:
        return times
    return remap_char_times(joined, times, text)


def remap_char_times(src: str, src_times, dst: str):
    """dst 是 src 修改後的文字（去迴圈、刪幻覺句…），用 difflib 對齊把時間搬過去。"""
    out: list[tuple[float, float] | None] = [None] * len(dst)
    sm = difflib.SequenceMatcher(None, src, dst, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            if a + k < len(src_times):
                out[b + k] = src_times[a + k]
    # 內插沒有對到的字元
    known = [i for i, v in enumerate(out) if v is not None]
    if not known:
        if src_times:
            t0, t1 = src_times[0][0], src_times[-1][1]
        else:
            t0 = t1 = 0.0
        n = max(1, len(dst))
        return [(t0 + (t1 - t0) * i / n, t0 + (t1 - t0) * (i + 1) / n) for i in range(len(dst))]
    for i in range(len(dst)):
        if out[i] is None:
            prev = max((k for k in known if k < i), default=None)
            nxt = min((k for k in known if k > i), default=None)
            if prev is not None and nxt is not None:
                a = out[prev][1]
                b = out[nxt][0]
                span = nxt - prev
                out[i] = (a + (b - a) * (i - prev - 1) / span, a + (b - a) * (i - prev) / span)
            elif prev is not None:
                out[i] = (out[prev][1], out[prev][1])
            else:
                out[i] = (out[nxt][0], out[nxt][0])
    return out


def pieces_from_regions(text: str, regions: list[tuple[float, float]]):
    """沒有字級時間時：把字依比例分配到 VAD 的說話區段（跳過停頓）。"""
    if not text:
        return []
    total = sum(e - s for s, e in regions) or 1e-6
    w = [0.0 if c.isspace() else (1.0 if is_cjk_char(c) else 0.4) for c in text]
    tot = sum(w) or 1.0
    bounds = []
    acc = 0.0
    for s, e in regions:
        bounds.append((acc, acc + (e - s), s))
        acc += e - s

    def at(x):  # x: 0..total speech-time → 實際時間
        for a, b, s in bounds:
            if x <= b + 1e-9:
                return s + (x - a)
        return regions[-1][1]

    times = []
    acc = 0.0
    for k in range(len(text)):
        a = at(total * acc / tot)
        acc += w[k]
        times.append((a, at(total * acc / tot)))
    return times


def _latin(ch: str) -> bool:
    return ch.isalnum() and not is_cjk_char(ch)


def _cu(ch: str) -> float:
    return 0.0 if ch.isspace() else (1.0 if is_cjk_char(ch) else 0.5)


def split_cues(text: str, times, max_units: float, max_dur: float = 7.0, min_units: float = 4.0):
    """依標點、長度、停頓切字幕。回傳 [(start, end, text)]。"""
    n = len(text)
    spans = []
    start = 0
    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        units = 0.0
        last_strong = last_weak = last_space = -1
        cut = None
        i = start
        while i < n:
            ch = text[i]
            units += _cu(ch)
            if ch in STRONG_BREAK:
                last_strong = i
            elif ch in WEAK_BREAK:
                last_weak = i
            elif ch.isspace():
                last_space = i
            if ch in STRONG_BREAK and units >= min_units:
                cut = i
                break
            if i + 1 < n and times[i + 1][0] - times[i][1] >= 0.8 and units >= 2:  # 停頓
                cut = i
                break
            if units > max_units or (times[i][1] - times[start][0] > max_dur and units >= min_units):
                for pos in (last_strong, last_weak, last_space):
                    if pos > start and display_units(text[start:pos + 1]) >= max_units * 0.35:
                        cut = pos
                        break
                if cut is None:
                    k = i
                    if _latin(text[k]) and k + 1 < n and _latin(text[k + 1]):
                        while k > start and _latin(text[k]):
                            k -= 1
                    cut = k if k > start else i
                break
            i += 1
        if cut is None:
            cut = n - 1
        spans.append((start, cut + 1))
        start = cut + 1
    out = []
    for a, b in spans:
        seg = text[a:b]
        seg_s = seg.strip()
        if not seg_s or not norm_key(seg_s):
            continue
        a2 = a + (len(seg) - len(seg.lstrip()))
        b2 = a2 + len(seg_s) - 1
        out.append((times[a2][0], times[b2][1], seg_s))
    return out


def finalize_cues(cues: list[tuple[float, float, str]], min_dur: float = 0.6, target: float = 1.2):
    """整理字幕：不重疊、最短 0.6 秒、短字幕在不撞到下一句的前提下延長到 1.2 秒、去掉句尾逗號句號。"""
    cues = sorted(cues, key=lambda x: x[0])
    res = []
    for k, (s0, e0, t) in enumerate(cues):
        t = re.sub(r"[，,、。．；;：:]+$", "", t).strip()
        if not t:
            continue
        s_, e_ = s0, e0
        if res and s_ < res[-1][1] + 0.01:
            s_ = res[-1][1] + 0.01
        nxt = cues[k + 1][0] if k + 1 < len(cues) else float("inf")
        e_ = max(e_, s_ + min_dur)
        if e_ - s_ < target:
            e_ = min(s_ + target, max(e_, nxt - 0.02))
        if e_ > nxt - 0.01:
            e_ = max(s_ + 0.3, nxt - 0.01)
        res.append((s_, e_, t))
    return res


# ════════════════════════════════════════════════════════════════════
# 引擎：faster-whisper（Whisper / Breeze-ASR-25）
# ════════════════════════════════════════════════════════════════════
def pick_device(device: str):
    """回傳 (kind, index, vram_gb, cc)。"""
    kind, index = "cpu", 0
    vram, cc = 0.0, (0, 0)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        torch = None
        has_cuda = False
    if device.startswith("cuda") or device == "auto":
        if has_cuda:
            kind = "cuda"
            index = int(device.split(":")[1]) if ":" in device else 0
            p = torch.cuda.get_device_properties(index)
            vram = p.total_memory / 1024 ** 3
            cc = (p.major, p.minor)
        elif device.startswith("cuda"):
            log("   ⚠ 找不到可用的 CUDA GPU，改用 CPU（會很慢）")
    return kind, index, vram, cc


def auto_batch(kind: str, vram: float, engine: str) -> int:
    if kind != "cuda":
        return 2
    base = 16 if vram >= 15 else (12 if vram >= 11 else (8 if vram >= 7 else 4))
    if engine == "whisper-large-v3-turbo" or engine == "qwen3-asr-0.6b":
        base *= 2
    return base


class WhisperEngine:
    kind = "whisper"

    def __init__(self, model_id: str, device: str, batch_size: int, word_ts: bool):
        dev, idx, vram, cc = pick_device(device)
        prepare_cuda_dlls()
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        self.device = dev
        self.batch_size = batch_size or auto_batch(dev, vram, "")
        self.word_ts = word_ts
        tries = ["float16", "int8_float16", "float32"] if dev == "cuda" else ["int8", "float32"]
        last = None
        for ct in tries:
            try:
                log(f"   載入 {model_id}（{dev}{':' + str(idx) if dev == 'cuda' else ''}，{ct}）… 第一次使用會自動下載")
                self.model = WhisperModel(model_id, device=dev, device_index=idx, compute_type=ct,
                                          cpu_threads=(os.cpu_count() or 4) if dev == "cpu" else 0)
                self.compute_type = ct
                if dev == "cuda":
                    self._smoke_test()
                break
            except Exception as e:  # noqa: BLE001
                last = e
                log(f"   ⚠ {ct} 失敗：{str(e)[:300]}")
                self.model = None
                gc.collect()
        if self.model is None:
            raise RuntimeError(f"無法載入 Whisper 模型：{last}")
        self.pipe = BatchedInferencePipeline(self.model)
        self.multilingual = self.model.model.is_multilingual
        log(f"   ✅ 模型就緒（batch={self.batch_size}）")

    def _smoke_test(self):
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        f = self.model.feature_extractor(np.zeros(SR, dtype=np.float32))[..., :-1]
        enc = self.model.encode(pad_or_trim(f))
        self.model.model.detect_language(enc)

    # ── 語言判斷（限定在 allowed 內）───────────────────────────────
    def detect_langs(self, audio, chunks: list[Chunk], allowed: list[str] | None) -> list[dict]:
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        if not self.multilingual:
            return [{"en": 1.0} for _ in chunks]
        fe = self.model.feature_extractor
        res = []
        bs = max(1, self.batch_size)
        for i in range(0, len(chunks), bs):
            batch = chunks[i:i + bs]
            feats = np.stack([pad_or_trim(fe(audio[int(c.start * SR):int(c.end * SR)])[..., :-1]) for c in batch])
            enc = self.model.encode(feats)
            out = self.model.model.detect_language(enc)
            for item in out:
                probs = {tok[2:-2]: p for tok, p in item}
                if allowed:
                    probs = {k: v for k, v in probs.items() if k in allowed}
                tot = sum(probs.values()) or 1.0
                res.append({k: v / tot for k, v in sorted(probs.items(), key=lambda x: -x[1])[:5]})
        return res

    # ── 批次辨識 ───────────────────────────────────────────────────
    def transcribe(self, audio, chunks: list[Chunk], lang: str | None, prompt: str | None, progress=None):
        clips = [{"start": c.start, "end": c.end} for c in chunks]
        seek_map = {int((int(c.start * SR) / SR) * 100): c for c in chunks}
        for c in chunks:
            c.raw_text, c.pieces = "", []
        segs, _info = self.pipe.transcribe(
            audio, language=lang, task="transcribe", clip_timestamps=clips, batch_size=self.batch_size,
            beam_size=5, temperature=0.0, without_timestamps=False, word_timestamps=self.word_ts,
            initial_prompt=prompt, vad_filter=False, log_progress=False)
        done = set()
        for s in segs:
            c = seek_map.get(s.seek)
            if c is None:  # 保險：用時間找
                mid = (s.start + s.end) / 2
                c = min(chunks, key=lambda x: 0 if x.start <= mid <= x.end else min(abs(mid - x.start), abs(mid - x.end)))
            self._add_segment(c, s)
            if c.idx not in done:
                done.add(c.idx)
                if progress:
                    progress(len(done))
        for c in chunks:
            c.lang = lang
            c.raw_text = c.raw_text.strip()

    def _add_segment(self, c: Chunk, s):
        text = s.text
        lo, hi = c.start, c.end + 0.3
        clamp = lambda t: min(max(float(t), lo), hi)  # noqa: E731
        if s.words:
            for w in s.words:
                c.pieces.append((w.word, clamp(w.start), clamp(w.end)))
        elif text:
            c.pieces.append((text, clamp(s.start), clamp(s.end)))
        c.raw_text += text
        c.avg_logprob = float(s.avg_logprob)
        c.no_speech_prob = float(s.no_speech_prob)

    def retry(self, audio, c: Chunk, lang: str | None, prompt: str | None, attempt: int) -> Chunk:
        """整段重跑：溫度遞增＋重複懲罰（這時才用 faster-whisper 的循序模式，才有 temperature fallback）。"""
        seg_audio = audio[int(c.start * SR):int(c.end * SR)]
        opts = dict(language=lang, task="transcribe", beam_size=5, best_of=5,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8] if attempt == 0 else [0.2, 0.4, 0.6, 0.8, 1.0],
                    compression_ratio_threshold=2.2, log_prob_threshold=-1.0, no_speech_threshold=0.6,
                    condition_on_previous_text=False, repetition_penalty=1.1 if attempt == 0 else 1.25,
                    no_repeat_ngram_size=0 if attempt == 0 else 6, without_timestamps=False,
                    word_timestamps=self.word_ts, vad_filter=False,
                    initial_prompt=prompt if attempt == 0 else None,
                    hallucination_silence_threshold=2.0 if self.word_ts else None)
        segs, _ = self.model.transcribe(seg_audio, **opts)
        segs = list(segs)
        nc = Chunk(idx=c.idx, start=c.start, end=c.end, regions=c.regions, lang=lang, lang_probs=c.lang_probs)
        lps, nsp, toks = [], [], []
        for s in segs:
            text = s.text
            clamp = lambda t: min(max(c.start + float(t), c.start), c.end + 0.3)  # noqa: E731
            if s.words:
                for w in s.words:
                    nc.pieces.append((w.word, clamp(w.start), clamp(w.end)))
            elif text:
                nc.pieces.append((text, clamp(s.start), clamp(s.end)))
            nc.raw_text += text
            n = max(1, len(s.tokens))
            lps.append(s.avg_logprob * n)
            toks.append(n)
            nsp.append(s.no_speech_prob)
        nc.raw_text = nc.raw_text.strip()
        nc.avg_logprob = sum(lps) / sum(toks) if toks else None
        nc.no_speech_prob = sum(nsp) / len(nsp) if nsp else None
        return nc

    def close(self):
        self.pipe = None
        self.model = None
        gc.collect()


# ════════════════════════════════════════════════════════════════════
# 引擎：Qwen3-ASR（transformers 原生支援，需 transformers ≥ 5.13）
# ════════════════════════════════════════════════════════════════════
class QwenEngine:
    kind = "qwen"

    def __init__(self, model_id: str, device: str, batch_size: int, use_aligner: bool):
        import torch
        import transformers
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3ASRForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(f"transformers {transformers.__version__} 不支援 Qwen3-ASR，"
                               "請執行 pip install -U \"transformers>=5.13\"") from e
        dev, idx, vram, cc = pick_device(device)
        self.torch = torch
        self.device = torch.device(f"cuda:{idx}" if dev == "cuda" else "cpu")
        if dev == "cuda":
            self.dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        else:
            self.dtype = torch.float32
        self.batch_size = batch_size or auto_batch(dev, vram, "qwen3-asr-0.6b" if "0.6B" in model_id else "")
        log(f"   載入 {model_id}（{self.device}，{str(self.dtype).replace('torch.', '')}）… 第一次使用會自動下載")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.aligner = None
        if use_aligner:
            try:
                from transformers import AutoModelForTokenClassification
                log(f"   載入字幕時間對齊模型 {QWEN_ALIGNER} …")
                self.aligner_proc = AutoProcessor.from_pretrained(QWEN_ALIGNER)
                self.aligner = AutoModelForTokenClassification.from_pretrained(
                    QWEN_ALIGNER, dtype=self.dtype).to(self.device).eval()
            except Exception as e:  # noqa: BLE001
                log(f"   ⚠ 對齊模型載入失敗（{str(e)[:200]}），字幕時間改用比例估計")
                self.aligner = None
        log(f"   ✅ 模型就緒（batch={self.batch_size}）")

    def _generate(self, arrays, langs, prompt, max_new, **gen):
        torch = self.torch
        kw = {}
        if prompt:
            kw["prompt"] = [prompt] * len(arrays)
        inputs = self.processor.apply_transcription_request(audio=arrays, language=langs, **kw)
        inputs = inputs.to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new, **gen)
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(gen_ids, return_format="parsed")
        if isinstance(parsed, dict):
            parsed = [parsed]
        pad = self.processor.tokenizer.pad_token_id
        eos = set(self.model.generation_config.eos_token_id if isinstance(
            self.model.generation_config.eos_token_id, (list, tuple)) else [self.model.generation_config.eos_token_id])
        caps = []
        for row in gen_ids.tolist():
            n = sum(1 for t in row if t != pad and t not in eos)
            caps.append(n >= max_new - 1)
        return parsed, caps

    def _run(self, audio, chunks: list[Chunk], langs: list[str | None], prompt, progress=None, max_new_override=None,
             **gen):
        order = sorted(range(len(chunks)), key=lambda i: chunks[i].dur)
        results: dict[int, tuple[str | None, str, bool]] = {}
        done = 0
        bs = max(1, self.batch_size)
        k = 0
        while k < len(order):
            ids = order[k:k + bs]
            arrays = [audio[int(chunks[i].start * SR):int(chunks[i].end * SR)] for i in ids]
            max_new = max_new_override or int(max(chunks[i].dur for i in ids) * 9) + 32
            try:
                parsed, caps = self._generate(arrays, [langs[i] for i in ids], prompt, max_new, **gen)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    bs = max(1, bs // 2)
                    self.batch_size = bs
                    self.torch.cuda.empty_cache()
                    log(f"   ⚠ GPU 記憶體不足，batch 降為 {bs}")
                    continue
                raise
            for i, p, cap in zip(ids, parsed, caps):
                lang_name = (p.get("language") or "").strip().lower()
                text = (p.get("transcription") or "").strip()
                code = QWEN_NAME_TO_CODE.get(lang_name) if lang_name else None
                if lang_name and code is None and text:  # 方言等：依文字判斷
                    code = "ja" if RE_KANA.search(text) else ("zh" if RE_HAN.search(text) else "en")
                results[i] = (code, text, cap)
            k += len(ids)
            done += len(ids)
            if progress:
                progress(done)
        return [results[i] for i in range(len(chunks))]

    def detect_langs(self, audio, chunks: list[Chunk], allowed):
        res = self._run(audio, chunks, [None] * len(chunks), None, max_new_override=24, do_sample=False)
        out = []
        for code, _t, _c in res:
            out.append({code: 1.0} if code else {})
        return out

    def transcribe(self, audio, chunks: list[Chunk], langs: list[str | None], prompt, progress=None):
        res = self._run(audio, chunks, langs, prompt, progress, do_sample=False, num_beams=1)
        for c, (code, text, cap), forced in zip(chunks, res, langs):
            c.lang = forced or code
            c.raw_text = text
            c.hit_cap = cap
            c.pieces = []

    def retry(self, audio, c: Chunk, lang, prompt, attempt: int) -> Chunk:
        gen = dict(do_sample=False, num_beams=1, repetition_penalty=1.15) if attempt == 0 else \
            dict(do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.1, no_repeat_ngram_size=8)
        (code, text, cap), = self._run(audio, [c], [lang], None if attempt else prompt, None, **gen)
        return Chunk(idx=c.idx, start=c.start, end=c.end, regions=c.regions, lang=lang or code,
                     lang_probs=c.lang_probs, raw_text=text, hit_cap=cap)

    def align(self, audio, chunks: list[Chunk], progress=None):
        """用 Qwen3-ForcedAligner 取得字級時間（每段 ≤28 秒，遠低於 5 分鐘上限）。"""
        if self.aligner is None:
            return
        import importlib.util
        skip = set()
        if importlib.util.find_spec("nagisa") is None:
            skip.add("ja")  # 日文對齊需要 nagisa 斷詞
        if importlib.util.find_spec("soynlp") is None:
            skip.add("ko")
        todo = [c for c in chunks if c.text_for_align and c.lang in ALIGNER_LANGS and c.lang not in skip]
        todo.sort(key=lambda c: (c.lang, c.idx))  # 同一批同一語言，一批失敗不會拖累別的語言
        bs = max(1, self.batch_size)
        k = 0
        while k < len(todo):
            j = k
            while j < len(todo) and j - k < bs and todo[j].lang == todo[k].lang:
                j += 1
            batch = todo[k:j]
            k = j
            try:
                arrays = [audio[int(c.start * SR):int(c.end * SR)] for c in batch]
                inputs, word_lists = self.aligner_proc.prepare_forced_aligner_inputs(
                    audio=arrays, transcript=[c.text_for_align for c in batch], language=[c.lang for c in batch])
                inputs = inputs.to(self.device, self.dtype)
                with self.torch.inference_mode():
                    logits = self.aligner(**inputs).logits
                stamps = self.aligner_proc.decode_forced_alignment(
                    logits=logits, input_ids=inputs["input_ids"], word_lists=word_lists,
                    timestamp_token_id=self.aligner.config.timestamp_token_id)
            except Exception as e:  # noqa: BLE001
                log(f"   ⚠ 對齊失敗（{str(e)[:160]}），這批字幕時間改用比例估計")
                continue
            for c, words in zip(batch, stamps):
                c.align_words = [(w["text"], c.start + w["start_time"], c.start + w["end_time"]) for w in words]
            if progress:
                progress(k, len(todo))

    def close(self):
        self.model = None
        self.aligner = None
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ════════════════════════════════════════════════════════════════════
# 引擎：測試模式（不需模型，會故意產生幻覺與迴圈來測試過濾器）
# ════════════════════════════════════════════════════════════════════
class DummyEngine:
    kind = "dummy"
    batch_size = 8
    SAMPLES = {
        "zh": ["我们今天要讨论森林碳汇的计算方法", "这个部分跟碳权交易的规则有关", "大家可以看一下这张投影片"],
        "ja": ["今日は森林の炭素吸収について話します", "この図を見てください"],
        "en": ["Today we will talk about forest carbon sinks", "Please have a look at this slide"],
    }

    def __init__(self, *a, **k):
        log("   ✅ 測試模式（不載入模型）")

    def detect_langs(self, audio, chunks, allowed):
        langs = allowed or ["zh"]
        return [{langs[(c.idx // 7) % len(langs)]: 0.9} for c in chunks]

    def _text(self, c: Chunk, lang: str, clean: bool = False) -> str:
        pool = self.SAMPLES.get(lang, self.SAMPLES["en"])
        t = pool[c.idx % len(pool)]
        if clean:
            return t
        if c.idx % 5 == 3:
            t = t + ("我们我们我们我们我们" if lang == "zh" else " the the the the the the")
        if c.idx % 9 == 4:
            t = "字幕由Amara.org社区提供"
        if c.idx % 11 == 6:
            t = (t + "。") * 6
        return t

    def transcribe(self, audio, chunks, langs, prompt, progress=None):
        for n, (c, lang) in enumerate(zip(chunks, langs), 1):
            c.lang = lang or "zh"
            c.raw_text = self._text(c, c.lang)
            c.avg_logprob, c.no_speech_prob = -0.3, 0.05
            if progress:
                progress(n)

    def retry(self, audio, c, lang, prompt, attempt):
        return Chunk(idx=c.idx, start=c.start, end=c.end, regions=c.regions, lang=lang or c.lang,
                     raw_text=self._text(c, lang or c.lang or "zh", clean=True), avg_logprob=-0.2, no_speech_prob=0.02)

    def close(self):
        pass


def load_engine(name: str, device: str, batch_size: int, word_ts: bool, use_aligner: bool):
    spec = ENGINES[name]
    if spec["kind"] == "whisper":
        return WhisperEngine(spec["model"], device, batch_size, word_ts)
    if spec["kind"] == "qwen":
        return QwenEngine(spec["model"], device, batch_size, use_aligner)
    return DummyEngine()


# ════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════
@dataclass
class Settings:
    engine: str = DEFAULT_ENGINE
    language: str = "auto"          # auto / mixed / zh / ja / en / ...
    allowed: list[str] = field(default_factory=lambda: ["zh", "ja", "en"])
    zh_script: str = "traditional"  # traditional / traditional_tw / simplified / none
    formats: list[str] = field(default_factory=lambda: ["txt", "srt", "md", "json"])
    device: str = "auto"
    batch_size: int = 0
    vad_threshold: float = 0.5
    max_chunk: float = 28.0
    retry: bool = True
    word_ts: bool = True
    aligner: bool = True
    glossary: str | None = None
    out_dir: str | None = None
    skip_existing: bool = False
    start_at: float = 0.0
    limit_minutes: float = 0.0
    max_units_cjk: float = 20.0
    max_units_latin: float = 42.0


def choose_single_lang(prob_list: list[dict], allowed: list[str] | None) -> tuple[str | None, dict]:
    score: dict[str, float] = {}
    for d in prob_list:
        for k, v in d.items():
            if k and (not allowed or k in allowed):
                score[k] = score.get(k, 0.0) + v
    if not score:
        return (allowed[0] if allowed else None), {}
    tot = sum(score.values())
    score = {k: v / tot for k, v in sorted(score.items(), key=lambda x: -x[1])}
    return next(iter(score)), score


def smooth_langs(chunks: list[Chunk]):
    """混語模式：短又不確定的段落沿用前後段的語言。"""
    for i, c in enumerate(chunks):
        if not c.lang_probs:
            continue
        top = max(c.lang_probs.values())
        if top >= 0.6 or c.speech_dur >= 4:
            continue
        cands = [chunks[j] for j in (i - 1, i + 1) if 0 <= j < len(chunks) and chunks[j].lang]
        cands = [n for n in cands if c.lang_probs.get(n.lang, 0) >= 0.1]
        if cands:
            best = max(cands, key=lambda n: max(n.lang_probs.values()) if n.lang_probs else 0)
            c.lang = best.lang


def clean_chunk(c: Chunk):
    """去幻覺句、去迴圈、整理空白 → c.text（還沒做繁簡轉換）。"""
    t = c.raw_text or ""
    t, n_boh = BOH.remove_strong(t)
    if n_boh:
        c.fixed.append(f"刪除幻覺句×{n_boh}")
    t2, n_loop = deloop(t)
    if n_loop:
        c.fixed.append(f"去除迴圈×{n_loop}")
    t2 = tidy_spaces(t2)
    c.text = t2


def output_paths(src: Path, s: Settings) -> dict[str, Path]:
    out_dir = Path(s.out_dir) if s.out_dir else src.parent
    stem = src.stem
    return {"txt": out_dir / f"{stem}{OUT_SUFFIX}.txt", "srt": out_dir / f"{stem}{OUT_SUFFIX}.srt",
            "md": out_dir / f"{stem}{OUT_SUFFIX}.md", "json": out_dir / f"{stem}{OUT_SUFFIX}.json"}


def process_file(path: Path, engine, s: Settings, conv: ScriptConverter, glossary: Glossary,
                 file_no: int, n_files: int) -> dict:
    t_start = time.time()
    log("-" * 60)
    log(f"📄 [{file_no}/{n_files}] {path.name}")
    emit("file", file=file_no, files=n_files, name=path.name)
    emit("stage", stage="decode", text="讀取音訊")
    audio = load_audio(path)
    total = len(audio) / SR
    off = 0.0
    if s.start_at > 0 or s.limit_minutes > 0:
        a = int(s.start_at * SR)
        b = int((s.start_at + s.limit_minutes * 60) * SR) if s.limit_minutes > 0 else len(audio)
        audio = audio[a:b]
        off = s.start_at
        log(f"   只處理 {fmt_hms(off)} 起 {len(audio) / SR / 60:.1f} 分鐘")
    dur = len(audio) / SR
    log(f"   長度：{fmt_hms(total)}")

    # ── VAD ──
    emit("stage", stage="vad", text="偵測語音區段")
    t0 = time.time()
    regions = vad_regions(audio, s.vad_threshold, s.max_chunk)
    chunks = build_chunks(regions, dur, max_chunk=s.max_chunk)
    speech = sum(e - b for b, e in regions)
    log(f"   語音偵測：{len(regions)} 個說話區段 → {len(chunks)} 個語音段，語音 {fmt_hms(speech)}"
        f"（略過 {fmt_hms(dur - speech)} 的靜音/音樂），{time.time() - t0:.1f} 秒")
    if not chunks:
        log("   ⚠ 沒有偵測到語音")

    # ── 語言 ──
    allowed = [x for x in s.allowed if x] or None
    is_whisper = engine.kind == "whisper"

    def prompt_for(lg):
        if is_whisper:
            return whisper_prompt(lg, s.zh_script, glossary)
        return ("Vocabulary: " + ", ".join(glossary.terms[:60])) if glossary.terms else None

    fixed_lang = s.language not in ("auto", "mixed")
    if chunks:
        emit("stage", stage="lang", text="判斷語言")
        if fixed_lang:
            for c in chunks:
                c.lang = s.language
            log(f"   語言：{LANG_NAMES.get(s.language, s.language)}（指定）")
        elif not is_whisper:
            # Qwen3-ASR：語言判斷是解碼的一部分（不額外花時間）→ 逐段判斷，之後再限制在允許清單
            for c in chunks:
                c.lang = None
            log("   語言：由模型逐段判斷" + (
                "（限定 " + "／".join(LANG_NAMES.get(x, x) for x in allowed) + "）" if allowed else ""))
        else:
            sample = chunks if s.language == "mixed" else \
                sorted(sorted(chunks, key=lambda c: -c.speech_dur)[:12], key=lambda c: c.idx)
            probs = engine.detect_langs(audio, sample, allowed)
            for c, p in zip(sample, probs):
                c.lang_probs = p
            main, dist = choose_single_lang(probs, allowed)
            if s.language == "mixed":
                for c in chunks:
                    top = next(iter(c.lang_probs), None) if c.lang_probs else None
                    c.lang = top if top and (c.lang_probs[top] >= 0.5 or c.speech_dur >= 4) else main
                smooth_langs(chunks)
                cnt: dict[str, int] = {}
                for c in chunks:
                    cnt[c.lang or "?"] = cnt.get(c.lang or "?", 0) + 1
                log("   語言（逐段）：" + "、".join(f"{LANG_NAMES.get(k, k)} {v} 段" for k, v in cnt.items()))
            else:
                for c in chunks:
                    c.lang = main
                log("   語言：" + "、".join(f"{LANG_NAMES.get(k, k)} {v:.0%}" for k, v in list(dist.items())[:3])
                    + f" → 使用 {LANG_NAMES.get(main, main)}")

    # ── 辨識 ──
    t0 = time.time()
    n_chunks = len(chunks)
    emit("stage", stage="asr", text="辨識中", done=0, total=n_chunks)

    def prog_factory(base):
        def _p(k):
            emit("progress", done=base + k, total=n_chunks)
        return _p

    if chunks and is_whisper:
        groups: dict[str | None, list[Chunk]] = {}
        for c in chunks:
            groups.setdefault(c.lang, []).append(c)
        base = 0
        for lg, grp in groups.items():
            engine.transcribe(audio, grp, lg, prompt_for(lg), prog_factory(base))
            base += len(grp)
    elif chunks:
        engine.transcribe(audio, chunks, [c.lang for c in chunks], prompt_for(None), prog_factory(0))
        if not fixed_lang and allowed:
            main, _ = choose_single_lang([{c.lang: 1.0} for c in chunks if c.lang in allowed], allowed)
            bad = [c for c in chunks if c.raw_text and c.lang not in allowed]
            if bad:
                log(f"   {len(bad)} 段被判成允許清單以外的語言 → 以 {LANG_NAMES.get(main, main)} 重跑")
                engine.transcribe(audio, bad, [main] * len(bad), prompt_for(None))
            for c in chunks:
                if c.lang is None:
                    c.lang = main
    asr_time = time.time() - t0
    log(f"   辨識完成：{asr_time:.1f} 秒")

    # ── 品質檢查與重轉 ──
    emit("stage", stage="check", text="品質檢查")
    n_retry = n_fixed = n_drop = 0
    for pos in range(len(chunks)):
        c = chunks[pos]
        q = assess(c.raw_text, c.lang, max(c.speech_dur, c.dur * 0.8), avg_logprob=c.avg_logprob,
                   no_speech_prob=c.no_speech_prob, prompt=prompt_for(c.lang), hit_cap=c.hit_cap)
        c.flags = q["flags"]
        if q["drop"]:
            c.dropped = True
            continue
        if not (s.retry and set(c.flags) & RETRY_FLAGS):
            continue
        # 要試哪些語言：原本的語言；若這段其實是別的語言（語言判斷 ≥50%）就先試那個
        cand_langs = [c.lang]
        if not fixed_lang and allowed and len(allowed) > 1:
            if not c.lang_probs:
                try:
                    c.lang_probs = engine.detect_langs(audio, [c], allowed)[0]
                except Exception:  # noqa: BLE001
                    c.lang_probs = {}
            if c.lang_probs:
                top = next(iter(c.lang_probs))
                if top != c.lang and c.lang_probs[top] >= 0.5:
                    cand_langs = [top, c.lang]
                elif "script" in c.flags:
                    cand_langs += [k for k in c.lang_probs if k != c.lang][:1]
        best, best_q, attempts = c, q, 0
        for lg in cand_langs:
            for att in range(2):
                try:
                    nc = engine.retry(audio, c, lg, prompt_for(lg), att)
                except Exception as e:  # noqa: BLE001
                    log(f"   ⚠ 重轉失敗：{str(e)[:200]}")
                    break
                attempts += 1
                nq = assess(nc.raw_text, nc.lang, max(nc.speech_dur, nc.dur * 0.8), avg_logprob=nc.avg_logprob,
                            no_speech_prob=nc.no_speech_prob, prompt=prompt_for(lg), hit_cap=nc.hit_cap)
                if nq["drop"] or nq["score"] > best_q["score"]:
                    best, best_q = nc, nq
                if nq["drop"] or not (set(nq["flags"]) & RETRY_FLAGS):
                    break
            if best_q["drop"] or (best is not c and not (set(best_q["flags"]) & RETRY_FLAGS)):
                break
        n_retry += 1
        before = c.raw_text[:40]
        if best is not c:
            best.retried = attempts
            best.flags = best_q["flags"]
            best.dropped = best_q["drop"]
            chunks[pos] = best
        c = chunks[pos]
        ok = c.dropped or not (set(c.flags) & RETRY_FLAGS)
        after = "（刪除）" if c.dropped else repr(c.raw_text[:40])
        log(f"   {'✅' if ok else '⚠'} 重轉 {fmt_hms(off + c.start)} "
            f"[{'、'.join(FLAG_TEXT.get(f, f) for f in q['flags'])}] {before!r} → {after}")
    # 清理
    for c in chunks:
        if c.dropped:
            n_drop += 1
            c.text = ""
            continue
        clean_chunk(c)
        if c.fixed:
            n_fixed += 1
        if not c.text or not norm_key(c.text):
            c.dropped = True
            n_drop += 1
    # 相鄰段落完全相同 → 典型幻覺（重複上一句），保留第一個
    prev = None
    for c in chunks:
        if c.dropped:
            continue
        k = norm_key(c.text)
        if prev is not None and k and k == norm_key(prev.text) and len(k) >= 4 and c.start - prev.end < 5:
            c.dropped = True
            c.fixed.append("與上一段重複")
            n_drop += 1
            continue
        prev = c
    kept = [c for c in chunks if not c.dropped]

    # ── 字幕時間 ──
    if isinstance(engine, QwenEngine) and engine.aligner is not None and "srt" in s.formats:
        emit("stage", stage="align", text="對齊字幕時間")
        for c in kept:
            c.text_for_align = c.text
        t0 = time.time()
        engine.align(audio, kept)
        log(f"   字幕時間對齊：{time.time() - t0:.1f} 秒")

    cues = []
    for c in kept:
        aw = c.align_words
        if aw and (aw[-1][2] - aw[0][1]) >= 0.3 * c.speech_dur:
            times = char_times_from_words(c.text, aw)
        elif c.pieces:
            times = char_times_from_pieces(c.text, c.pieces)
        else:
            times = pieces_from_regions(c.text, c.regions)
        latin_heavy = len(RE_LATIN_WORD.findall(c.text)) * 3 > len(RE_HAN.findall(c.text) + RE_KANA.findall(c.text))
        mu = s.max_units_latin / 2 if latin_heavy else s.max_units_cjk  # display_units：英文字元算 0.5
        for a, b, t in split_cues(c.text, times, mu):
            cues.append((off + a, off + b, glossary.apply(conv(t, c.lang)), c))
    cues_final = finalize_cues([(a, b, t) for a, b, t, _ in cues])

    # ── 輸出 ──
    for c in kept:
        c.text = glossary.apply(conv(c.text, c.lang))
    elapsed = time.time() - t_start
    info = dict(file=str(path), name=path.name, duration=total, processed_from=off, processed_dur=dur,
                speech=speech, engine=s.engine, engine_label=ENGINES[s.engine]["label"], language_mode=s.language,
                chunks=len(chunks), kept=len(kept), retried=n_retry, fixed=n_fixed, dropped=n_drop,
                elapsed=elapsed, asr_time=asr_time, zh_script=s.zh_script, version=VERSION)
    paths = write_outputs(path, s, info, kept, chunks, cues_final, off)
    speed = dur / max(elapsed, 1e-6)
    log(f"   🔄 重轉 {n_retry} 段、🧹 修正 {n_fixed} 段、🗑 刪除 {n_drop} 段（非語音/幻覺/重複）")
    log(f"   ⏱ 共 {elapsed:.1f} 秒（{speed:.0f}× 即時速度）")
    for k, p in paths.items():
        log(f"   💾 {p}")
    emit("file_done", file=file_no, files=n_files, name=path.name, outputs=[str(p) for p in paths.values()],
         flagged=sum(1 for c in kept if set(c.flags) & RETRY_FLAGS), elapsed=elapsed)
    return {"info": info, "outputs": paths}


def char_times_from_words(text: str, words):
    """對齊模型給的字詞（不含標點）→ 對回含標點的文字。"""
    pieces = []
    pos = 0
    for w, a, b in words:
        k = text.find(w, pos)
        if k < 0:
            continue
        if k > pos:
            pieces.append((text[pos:k], a, a))  # 標點/空白：時間放在下一個字的開頭
        pieces.append((w, a, b))
        pos = k + len(w)
    if pos < len(text):
        last = pieces[-1][2] if pieces else 0.0
        pieces.append((text[pos:], last, last))
    return char_times_from_pieces(text, pieces)


def build_paragraphs(kept: list[Chunk], off: float, gap: float = 3.0, max_units: float = 350):
    paras, cur = [], None
    for c in kept:
        if cur and (c.start - cur["end"] < gap) and c.lang == cur["lang"] and display_units(cur["text"]) < max_units:
            cur["text"] = join_texts(cur["text"], c.text, c.lang, c.start - cur["end"])
            cur["end"] = c.end
            cur["flags"] |= set(c.flags) & RETRY_FLAGS
            cur["chunks"].append(c)
        else:
            cur = {"start": c.start, "end": c.end, "text": c.text, "lang": c.lang,
                   "flags": set(c.flags) & RETRY_FLAGS, "chunks": [c]}
            paras.append(cur)
    for p in paras:
        p["start"] += off
        p["end"] += off
    return paras


def write_outputs(src: Path, s: Settings, info: dict, kept, chunks, cues, off) -> dict[str, Path]:
    paths = output_paths(src, s)
    paths["txt"].parent.mkdir(parents=True, exist_ok=True)
    paras = build_paragraphs(kept, off)
    written = {}
    if "txt" in s.formats:
        paths["txt"].write_text("\n\n".join(p["text"] for p in paras) + "\n", encoding="utf-8")
        written["txt"] = paths["txt"]
    if "srt" in s.formats:
        with open(paths["srt"], "w", encoding="utf-8") as f:
            for i, (a, b, t) in enumerate(cues, 1):
                f.write(f"{i}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{t}\n\n")
        written["srt"] = paths["srt"]
    if "md" in s.formats:
        flagged = [c for c in kept if set(c.flags) & RETRY_FLAGS]
        langs: dict[str, int] = {}
        for c in kept:
            langs[c.lang or "?"] = langs.get(c.lang or "?", 0) + 1
        lines = [f"# {src.stem}", "",
                 "| 項目 | 內容 |", "|---|---|",
                 f"| 原始檔案 | {src.name} |",
                 f"| 總長度 | {fmt_hms(info['duration'])}" + (f"（處理 {fmt_hms(off)} 起 {fmt_hms(info['processed_dur'])}）" if off or
                                                               info['processed_dur'] < info['duration'] - 1 else "") + " |",
                 f"| 語音長度 | {fmt_hms(info['speech'])} |",
                 f"| 引擎 | {info['engine_label']} |",
                 "| 語言 | " + "、".join(f"{LANG_NAMES.get(k, k)}（{v} 段）" for k, v in langs.items()) + " |",
                 f"| 品質處理 | 重轉 {info['retried']} 段、修正 {info['fixed']} 段、刪除 {info['dropped']} 段 |",
                 f"| 需人工確認 | {len(flagged)} 段（標記 ⚠） |",
                 f"| 處理時間 | {info['elapsed']:.0f} 秒 |",
                 f"| 產生時間 | {time.strftime('%Y-%m-%d %H:%M')} · {APP_NAME} V{VERSION} |", "", "---", ""]
        for p in paras:
            mark = " ⚠" if p["flags"] else ""
            lines.append(f"**[{fmt_hms(p['start'])} → {fmt_hms(p['end'])}]**{mark}")
            lines.append("")
            lines.append(p["text"])
            lines.append("")
        if flagged:
            lines += ["---", "", "## ⚠ 需要人工確認的片段", "",
                      "這些段落重轉後仍有疑慮（已自動去除重複），建議對照原音檔確認。", "",
                      "| 時間 | 原因 | 內容 |", "|---|---|---|"]
            for c in flagged:
                why = "、".join(FLAG_TEXT.get(f, f) for f in c.flags if f in RETRY_FLAGS)
                lines.append(f"| {fmt_hms(off + c.start)} | {why} | {c.text[:80].replace('|', '｜')} |")
            lines.append("")
        paths["md"].write_text("\n".join(lines), encoding="utf-8")
        written["md"] = paths["md"]
    if "json" in s.formats:
        data = dict(info=info, paragraphs=[dict(start=p["start"], end=p["end"], text=p["text"], lang=p["lang"],
                                                flags=sorted(p["flags"])) for p in paras],
                    chunks=[dict(i=c.idx, start=off + c.start, end=off + c.end, lang=c.lang, text=c.text,
                                 raw=c.raw_text, flags=c.flags, dropped=c.dropped, retried=c.retried, fixed=c.fixed,
                                 avg_logprob=c.avg_logprob, no_speech_prob=c.no_speech_prob,
                                 lang_probs=c.lang_probs) for c in chunks],
                    cues=[dict(start=a, end=b, text=t) for a, b, t in cues])
        paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        written["json"] = paths["json"]
    return written


def collect_inputs(items: list[str], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    for it in items:
        p = Path(it.strip().strip('"'))
        if p.is_dir():
            it2 = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in it2 if f.is_file() and f.suffix.lower() in AUDIO_EXTS
                            and not f.stem.endswith(OUT_SUFFIX))
        elif p.is_file():
            files.append(p)
        elif it.strip():
            log(f"⚠ 找不到：{it}")
    seen, out = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def run(inputs: list[str], s: Settings, recursive: bool = False) -> int:
    prepare_cuda_dlls()  # 必須在 import faster_whisper / ctranslate2 之前
    files = collect_inputs(inputs, recursive)
    if not files:
        log("❌ 沒有可處理的音檔/影片")
        return 2
    glossary = Glossary.load(s.glossary)
    conv = ScriptConverter(s.zh_script)
    log("=" * 60)
    log(f"🎙️🐰 {APP_NAME} V{VERSION}")
    log(f"   檔案：{len(files)} 個 · 引擎：{ENGINES[s.engine]['label']}")
    if glossary.terms or glossary.replacements:
        log(f"   專有名詞 {len(glossary.terms)} 個、修正規則 {len(glossary.replacements)} 條")
    log("=" * 60)
    todo = []
    for f in files:
        if s.skip_existing and output_paths(f, s)["json"].exists():
            log(f"⏭ 已有結果，略過：{f.name}")
        else:
            todo.append(f)
    if not todo:
        emit("all_done", ok=0, failed=0)
        return 0
    emit("stage", stage="load", text="載入模型")
    t0 = time.time()
    try:
        engine = load_engine(s.engine, s.device, s.batch_size, s.word_ts, s.aligner and "srt" in s.formats)
    except Exception as e:  # noqa: BLE001
        log(f"❌ 無法載入 {s.engine}：{e}")
        if ENGINES[s.engine]["kind"] == "qwen":
            log(f"   → 改用 {ENGINES[FALLBACK_ENGINE]['label']}")
            s.engine = FALLBACK_ENGINE
            engine = load_engine(s.engine, s.device, s.batch_size, s.word_ts, False)
        else:
            raise
    log(f"   模型載入 {time.time() - t0:.1f} 秒")
    ok = failed = 0
    for i, f in enumerate(todo, 1):
        try:
            process_file(f, engine, s, conv, glossary, i, len(todo))
            ok += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            failed += 1
            log(f"   ❌ 失敗：{e}")
            log(traceback.format_exc())
            emit("file_error", file=i, files=len(todo), name=f.name, error=str(e))
        gc.collect()
    engine.close()
    log("=" * 60)
    log(f"🎉 完成：成功 {ok} 個" + (f"、失敗 {failed} 個" if failed else ""))
    emit("all_done", ok=ok, failed=failed)
    return 0 if failed == 0 else 1


# ════════════════════════════════════════════════════════════════════
# doctor：環境檢查
# ════════════════════════════════════════════════════════════════════
def doctor(as_json: bool = False, full: bool = False) -> int:
    import importlib
    import platform
    import shutil
    r = {"python": platform.python_version(), "exe": sys.executable, "os": platform.platform(), "torch": None,
         "torch_cuda": None, "gpus": [], "nvidia_smi": bool(shutil.which("nvidia-smi")), "packages": {},
         "ct2_cuda": 0, "ct2_types": [], "advice": []}
    for mod, name in [("faster_whisper", "faster-whisper"), ("ctranslate2", "ctranslate2"),
                      ("transformers", "transformers"), ("gradio", "gradio"), ("opencc", "opencc"),
                      ("av", "av"), ("onnxruntime", "onnxruntime"), ("huggingface_hub", "huggingface_hub"),
                      ("nagisa", "nagisa")]:
        try:
            m = importlib.import_module(mod)
            r["packages"][name] = getattr(m, "__version__", "ok")
        except Exception:  # noqa: BLE001
            r["packages"][name] = None
    try:
        import torch
        r["torch"] = torch.__version__
        r["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                g = {"index": i, "name": p.name, "capability": f"sm_{p.major}{p.minor}",
                     "vram_gb": round(p.total_memory / 1024 ** 3, 1), "kernels_ok": False}
                try:
                    x = torch.randn(256, 256, device=f"cuda:{i}")
                    (x @ x).sum().item()
                    g["kernels_ok"] = True
                except Exception as e:  # noqa: BLE001
                    g["error"] = str(e)[:200]
                    r["advice"].append(f"GPU {p.name} 無法執行 PyTorch 運算（{str(e)[:80]}）：RTX 50 系列需要 PyTorch 2.7+ 與 "
                                       "CUDA 12.8，請重新執行 setup")
                r["gpus"].append(g)
    except Exception as e:  # noqa: BLE001
        r["advice"].append(f"PyTorch 無法載入：{e}")
    try:
        prepare_cuda_dlls()
        import ctranslate2
        r["ct2_cuda"] = ctranslate2.get_cuda_device_count()
        if r["ct2_cuda"]:
            r["ct2_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as e:  # noqa: BLE001
        r["advice"].append(f"CTranslate2 無法使用 GPU：{str(e)[:200]}")
    if r["nvidia_smi"] and not r["gpus"]:
        r["advice"].append("偵測到 NVIDIA 驅動，但 PyTorch 看不到 GPU：請確認裝的是 CUDA 版 PyTorch（setup 會安裝 cu128），"
                           "並把 NVIDIA 驅動更新到 570 以上（RTX 50 系列必要）")
    if r["gpus"] and not r["ct2_cuda"]:
        r["advice"].append("PyTorch 可用 GPU，但 CTranslate2 看不到 → Whisper 引擎會用 CPU。Qwen3-ASR 不受影響。")
    if not r["packages"].get("transformers"):
        r["advice"].append("缺少 transformers（Qwen3-ASR 需要）")
    else:
        try:
            import transformers as _tf
            r["qwen3_asr"] = hasattr(_tf, "Qwen3ASRForConditionalGeneration")
            if not r["qwen3_asr"]:
                raise ImportError
        except Exception:  # noqa: BLE001
            r["qwen3_asr"] = False
            r["advice"].append("目前的 transformers 不支援 Qwen3-ASR，請 pip install -U \"transformers>=5.13\"")
    if not r["packages"].get("opencc"):
        r["advice"].append("缺少 opencc：中文不會自動轉繁體（pip install opencc）")
    if full:
        try:
            prepare_cuda_dlls()
            from faster_whisper import WhisperModel
            import numpy as np
            dev = "cuda" if r["ct2_cuda"] else "cpu"
            m = WhisperModel("tiny", device=dev, compute_type="float16" if dev == "cuda" else "int8")
            list(m.transcribe(np.zeros(SR * 2, dtype=np.float32), language="en")[0])
            r["whisper_test"] = f"ok ({dev})"
        except Exception as e:  # noqa: BLE001
            r["whisper_test"] = f"failed: {str(e)[:300]}"
            r["advice"].append(f"Whisper 測試失敗：{str(e)[:200]}")
    if as_json:
        print(json.dumps(r, ensure_ascii=False))
        return 0
    print(f"Python {r['python']}  ({r['exe']})")
    print(f"PyTorch {r['torch']}  CUDA {r['torch_cuda']}")
    for g in r["gpus"]:
        print(f"GPU {g['index']}: {g['name']} {g['capability']} {g['vram_gb']} GB  運算 {'OK' if g['kernels_ok'] else '失敗'}")
    if not r["gpus"]:
        print("GPU：無（將使用 CPU）")
    print(f"CTranslate2 GPU 數：{r['ct2_cuda']}  支援：{', '.join(r['ct2_types'])}")
    for k, v in r["packages"].items():
        print(f"  {k:16s} {v or '— 未安裝'}")
    if "whisper_test" in r:
        print(f"Whisper tiny 測試：{r['whisper_test']}")
    for a in r["advice"]:
        print(f"⚠ {a}")
    if not r["advice"]:
        print("✅ 環境正常")
    return 0


def make_shortcut() -> int:
    """在桌面建立「音檔轉錄小兔歐」捷徑（指向 start_ui.bat，黑色視窗最小化）。"""
    if os.name != "nt":
        log("只有 Windows 需要建立捷徑。")
        return 0
    import subprocess
    q = lambda p: str(p).replace("'", "''")  # noqa: E731  PowerShell 單引號跳脫
    bat = APP_DIR / "start_ui.bat"
    ico = APP_DIR / "app.ico"
    ps = (
        "$d=[Environment]::GetFolderPath('Desktop');"
        f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d '{APP_NAME}.lnk'));"
        f"$s.TargetPath='{q(bat)}';$s.WorkingDirectory='{q(APP_DIR)}';$s.WindowStyle=7;"
        f"$s.Description='{APP_NAME} V{VERSION}';"
        + (f"$s.IconLocation='{q(ico)},0';" if ico.exists() else "") + "$s.Save()")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        log(f"✅ 已在桌面建立捷徑「{APP_NAME}」")
        return 0
    log(f"⚠ 建立捷徑失敗：{r.stderr.strip()[:300]}")
    return 1


def download(engine: str, aligner: bool) -> int:
    spec = ENGINES[engine]
    from huggingface_hub import snapshot_download
    if spec["kind"] == "whisper":
        from faster_whisper.utils import download_model
        log(f"下載 {spec['model']} …")
        log(download_model(spec["model"]))
    elif spec["kind"] == "qwen":
        for repo in [spec["model"]] + ([QWEN_ALIGNER] if aligner else []):
            log(f"下載 {repo} …")
            log(snapshot_download(repo))
    return 0


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    setup_stdio()
    ap = argparse.ArgumentParser(prog="stt.py", description=f"{APP_NAME} V{VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="檢查 GPU 與套件")
    d.add_argument("--json", action="store_true")
    d.add_argument("--full", action="store_true", help="實際載入 Whisper tiny 測試 GPU")
    sub.add_parser("shortcut", help="在桌面建立捷徑")
    dl = sub.add_parser("download", help="預先下載模型")
    dl.add_argument("--engine", default=DEFAULT_ENGINE, choices=list(ENGINES))
    dl.add_argument("--no-aligner", action="store_true")
    r = sub.add_parser("run", help="轉錄")
    r.add_argument("-i", "--input", nargs="+", required=True, help="音檔/影片/資料夾（可多個）")
    r.add_argument("-o", "--out", default=None, help="輸出資料夾（預設：原檔旁邊）")
    r.add_argument("--engine", default=DEFAULT_ENGINE, choices=list(ENGINES))
    r.add_argument("--language", default="auto", help="auto / mixed / zh / ja / en / ...")
    r.add_argument("--allowed", default="zh,ja,en", help="auto / mixed 時允許的語言（逗號分隔，空白＝不限）")
    r.add_argument("--zh-script", default="traditional", choices=list(ScriptConverter.MODES))
    r.add_argument("--formats", default="txt,srt,md,json")
    r.add_argument("--device", default="auto", help="auto / cuda / cuda:0 / cuda:1 / cpu")
    r.add_argument("--batch-size", type=int, default=0, help="0＝依顯示卡記憶體自動")
    r.add_argument("--vad-threshold", type=float, default=0.5)
    r.add_argument("--max-chunk", type=float, default=28.0)
    r.add_argument("--no-retry", action="store_true")
    r.add_argument("--no-word-timestamps", action="store_true")
    r.add_argument("--no-aligner", action="store_true")
    r.add_argument("--glossary", default=str(APP_DIR / "glossary.txt"))
    r.add_argument("--skip-existing", action="store_true")
    r.add_argument("--recursive", action="store_true")
    r.add_argument("--start-at", default="0", help="從某時間開始，例如 00:19:00")
    r.add_argument("--limit-minutes", type=float, default=0.0, help="只處理 N 分鐘（試跑用）")
    r.add_argument("--max-chars", type=float, default=20.0, help="中日文字幕每行最多字數")
    r.add_argument("--model", default=None, help="（進階）用自訂模型路徑或 Hugging Face ID 取代引擎預設模型")
    a = ap.parse_args(argv)
    if a.cmd == "doctor":
        return doctor(a.json, a.full)
    if a.cmd == "download":
        return download(a.engine, not a.no_aligner)
    if a.cmd == "shortcut":
        return make_shortcut()
    if a.model:
        ENGINES[a.engine] = dict(ENGINES[a.engine], model=a.model)
    s = Settings(engine=a.engine, language=a.language,
                 allowed=[x.strip() for x in a.allowed.split(",") if x.strip()],
                 zh_script=a.zh_script, formats=[x.strip() for x in a.formats.split(",") if x.strip()],
                 device=a.device, batch_size=a.batch_size, vad_threshold=a.vad_threshold, max_chunk=a.max_chunk,
                 retry=not a.no_retry, word_ts=not a.no_word_timestamps, aligner=not a.no_aligner,
                 glossary=a.glossary, out_dir=a.out, skip_existing=a.skip_existing,
                 start_at=parse_time(a.start_at), limit_minutes=a.limit_minutes, max_units_cjk=a.max_chars)
    return run(a.input, s, a.recursive)


if __name__ == "__main__":
    sys.exit(main())
