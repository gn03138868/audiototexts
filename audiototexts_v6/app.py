#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音檔轉錄小兔歐 V6 — 圖形介面

啟動：python app.py（或雙擊 start_ui.bat）→ 自動開啟獨立視窗（Edge/Chrome app 模式）或瀏覽器
所有運算都在本機；轉錄在獨立子程序中執行，可隨時停止，結束後 GPU 記憶體會完全釋放。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# 學校/公司 Proxy 不可以攔截本機介面
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = ",".join(x for x in (os.environ.get(_k, ""), "localhost,127.0.0.1,::1") if x)

try:
    import gradio as gr
except Exception as _e:  # noqa: BLE001
    print(f"\n[錯誤] 這個 Python 環境無法載入 gradio：{_e}\n  Python：{sys.executable}\n"
          "  請重新執行 setup_conda.bat（Anaconda）或 setup_windows.bat。\n")
    sys.exit(1)

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import stt  # noqa: E402  (只用到常數與清單，不會載入模型)

CLI = APP_DIR / "stt.py"
OUTPUTS = APP_DIR / "outputs"
GLOSSARY = APP_DIR / "glossary.txt"
OUTPUTS.mkdir(exist_ok=True)

ENGINE_LABELS = {v["label"]: k for k, v in stt.ENGINES.items()}
LANG_MODES = {"自動判斷（整份錄音一種主要語言，建議）": "auto", "混語（逐段判斷，適合中日英輪流說）": "mixed",
              "中文": "zh", "日文": "ja", "英文": "en", "粵語": "yue", "韓文": "ko"}
ALLOWED = {"中文": "zh", "日文": "ja", "英文": "en", "粵語": "yue", "韓文": "ko"}
ZH_MODES = {"繁體（台灣字形）": "traditional", "繁體＋台灣用語（軟件→軟體、視頻→影片）": "traditional_tw",
            "簡體": "simplified", "不轉換": "none"}
FORMATS = {"TXT 純文字": "txt", "SRT 字幕": "srt", "MD 逐字稿（含時間與待確認清單）": "md", "JSON（完整資料）": "json"}

# ════════════════════════════════════════════════════════════════════
# 子程序（一次一個工作，可停止）
# ════════════════════════════════════════════════════════════════════
_job_lock = threading.Lock()
_current: subprocess.Popen | None = None


def _popen(args: list[str]) -> subprocess.Popen:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", STT_UI="1",
               HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen([sys.executable, "-u", str(CLI), *args], cwd=str(APP_DIR), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)


def run_cli(args: list[str]):
    """逐行產生輸出；最後一個是 ('__rc__', returncode)。"""
    global _current
    if not _job_lock.acquire(blocking=False):
        yield "⚠ 已有工作在執行中，請等它完成或按「停止」。"
        yield ("__rc__", -1)
        return
    try:
        _current = _popen(args)
        assert _current.stdout is not None
        for line in _current.stdout:
            yield line.rstrip("\r\n")
        _current.wait()
        yield ("__rc__", _current.returncode)
    finally:
        _current = None
        _job_lock.release()


def stop_job():
    p = _current
    if p and p.poll() is None:
        p.terminate()
        return "⏹ 已停止。已完成的檔案都保留在輸出資料夾。"
    return "目前沒有執行中的工作。"


def run_quick(args: list[str], timeout: int = 180) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    p = subprocess.run([sys.executable, str(CLI), *args], cwd=str(APP_DIR), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout, creationflags=flags)
    return p.stdout + p.stderr


# ════════════════════════════════════════════════════════════════════
# 環境狀態
# ════════════════════════════════════════════════════════════════════
def gpu_status_md() -> str:
    out = ""
    try:
        out = run_quick(["doctor", "--json"])
        r = json.loads(out[out.index("{"): out.rindex("}") + 1])
    except Exception as e:  # noqa: BLE001
        tail_txt = "\n".join(out.strip().splitlines()[-12:]) if out else str(e)
        return (f"⚠ 無法檢查環境。Python：`{sys.executable}`\n\n```\n{tail_txt}\n```\n"
                "請在同一個環境執行 `python stt.py doctor`，把結果截圖。")
    parts = []
    if r["gpus"]:
        for g in r["gpus"]:
            parts.append(f"{'✅' if g['kernels_ok'] else '❌'} **{g['name']}** · {g['capability']} · {g['vram_gb']} GB")
    elif r["nvidia_smi"]:
        parts.append("❌ 偵測到 NVIDIA GPU，但 PyTorch 無法使用它")
    else:
        parts.append("⚠ 沒有可用的 NVIDIA GPU（將使用 CPU，速度很慢）")
    parts.append(f"PyTorch {r['torch'] or '未安裝'}" + (f" · CUDA {r['torch_cuda']}" if r["torch_cuda"] else ""))
    parts.append("Whisper 加速 " + ("✅" if r["ct2_cuda"] else "CPU"))
    parts.append("Qwen3-ASR " + ("✅" if r.get("qwen3_asr") else "❌"))
    parts.append("繁體轉換 " + ("✅" if r["packages"].get("opencc") else "❌"))
    md = " ｜ ".join(parts)
    if r["advice"]:
        md += "\n\n" + "\n".join(f"- ⚠ {a}" for a in r["advice"])
    return md


# ════════════════════════════════════════════════════════════════════
# 轉錄
# ════════════════════════════════════════════════════════════════════
STAGE_FRAC = {"decode": 0.02, "vad": 0.05, "lang": 0.08, "asr": 0.10, "check": 0.86, "align": 0.92}


def _bar(frac: float) -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    return (f"<div style='height:10px;border-radius:5px;background:var(--neutral-200);overflow:hidden'>"
            f"<div style='width:{pct:.1f}%;height:100%;background:var(--color-accent);transition:width .3s'></div></div>")


def _status(st: dict) -> str:
    files = max(1, st.get("files", 1))
    fi = st.get("file", 0)
    frac = st.get("frac", 0.0)
    overall = ((max(fi, 1) - 1) + frac) / files if fi else 0.0
    head = st.get("head", "⏳ 準備中…")
    line = f"**{head}**"
    if fi:
        line += f"　檔案 {fi}/{files}：`{st.get('name', '')}`"
    if st.get("stage_text"):
        line += f"　·　{st['stage_text']}"
        if st.get("total"):
            line += f" {st.get('done', 0)}/{st['total']} 段"
    return line + "\n\n" + _bar(overall) + f"\n\n<small>整體 {overall * 100:.0f}%</small>"


def _flag_rows(json_paths: list[Path]) -> list[list[str]]:
    rows = []
    for jp in json_paths:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for c in d.get("chunks", []):
            fl = [f for f in c.get("flags", []) if f in stt.RETRY_FLAGS]
            if fl and not c.get("dropped"):
                rows.append([Path(d["info"]["name"]).name, stt.fmt_hms(c["start"]),
                             "、".join(stt.FLAG_TEXT.get(f, f) for f in fl), c.get("text", "")[:120]])
    return rows


def _download_copies(paths: list[Path]) -> list[str]:
    """把結果複製到暫存資料夾再給「下載」元件（輸出資料夾可能在 D: 槽等 Gradio 不允許直接存取的位置）。"""
    import shutil
    import tempfile
    d = Path(tempfile.gettempdir()) / "usagi_stt_downloads" / time.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        if p.exists():
            q = d / p.name
            shutil.copy2(p, q)
            out.append(str(q))
    return out


def do_transcribe(files, paths_text, engine_label, lang_label, allowed_labels, zh_label, fmt_labels, device, batch,
                  vad_thr, max_chunk, retry, precise, limit_min, start_at, skip_existing, max_chars, out_dir,
                  progress=gr.Progress()):
    empty = (gr.update(), gr.update(), gr.update(), gr.update())
    inputs: list[str] = []
    uploaded = False
    for f in files or []:
        inputs.append(f if isinstance(f, str) else f.name)
        uploaded = True
    for line in (paths_text or "").splitlines():
        if line.strip():
            inputs.append(line.strip().strip('"'))
    if not inputs:
        yield ("請先上傳檔案，或貼上檔案／資料夾路徑。", "", *empty)
        return
    fmts = [FORMATS[x] for x in fmt_labels] or ["txt", "srt", "md", "json"]
    if "json" not in fmts:
        fmts.append("json")  # 介面需要 JSON 來列出待確認片段（很小）
    out = (out_dir or "").strip().strip('"')
    if not out and uploaded:
        out = str(OUTPUTS)
    args = ["run", "-i", *inputs, "--engine", ENGINE_LABELS[engine_label], "--language", LANG_MODES[lang_label],
            "--allowed", ",".join(ALLOWED[x] for x in allowed_labels), "--zh-script", ZH_MODES[zh_label],
            "--formats", ",".join(fmts), "--device", device, "--batch-size", str(int(batch or 0)),
            "--vad-threshold", str(vad_thr), "--max-chunk", str(max_chunk), "--max-chars", str(max_chars),
            "--glossary", str(GLOSSARY)]
    if out:
        args += ["-o", out]
    if not retry:
        args.append("--no-retry")
    if not precise:
        args += ["--no-word-timestamps", "--no-aligner"]
    if limit_min and float(limit_min) > 0:
        args += ["--limit-minutes", str(float(limit_min))]
    if start_at and start_at.strip():
        args += ["--start-at", start_at.strip()]
    if skip_existing:
        args.append("--skip-existing")

    log: list[str] = []
    st = {"head": "⏳ 載入模型中…（第一次使用會自動下載模型，約 2–5 GB）"}
    outputs: list[Path] = []
    preview = ""
    progress(0, desc="準備中")
    last_yield = 0.0
    for item in run_cli(args):
        if isinstance(item, tuple):
            rc = item[1]
            jsons = [p for p in outputs if p.suffix == ".json"]
            rows = _flag_rows(jsons)
            shown = _download_copies([p for p in outputs
                                      if p.suffix != ".json" or "json" in [FORMATS[x] for x in fmt_labels]])
            if rc == 0:
                st["head"] = "✅ 完成"
                st["frac"] = 1.0
                st["stage_text"] = ""
                st["total"] = 0
                if not st.get("file"):
                    st["file"] = st["files"] = 1
                msg = _status(st)
                if outputs:
                    msg += f"\n\n輸出資料夾：`{outputs[0].parent}`"
                if rows:
                    msg += f"\n\n⚠ 有 **{len(rows)} 段**重轉後仍需人工確認（見下表；MD 檔最後也有清單）。"
            else:
                msg = "❌ 未完成（失敗或已停止）。請展開「詳細紀錄」查看原因。已完成的檔案都保留在輸出資料夾。"
            yield (msg, "\n".join(log[-400:]), gr.update(value=preview), gr.update(value=shown or None),
                   gr.update(value=rows or [["", "", "沒有需要確認的片段 🎉", ""]]),
                   str(outputs[0].parent) if outputs else (out or ""))
            return
        if item.startswith("@@"):
            try:
                ev = json.loads(item[2:])
            except Exception:  # noqa: BLE001
                continue
            k = ev.get("kind")
            if k == "file":
                st.update(file=ev["file"], files=ev["files"], name=ev["name"], frac=0.0, head="⏳ 轉錄中",
                          stage_text="", total=0, done=0)
            elif k == "stage":
                st.update(stage_text=ev.get("text", ""), frac=STAGE_FRAC.get(ev.get("stage"), st.get("frac", 0)),
                          total=ev.get("total", 0), done=ev.get("done", 0))
                if ev.get("stage") == "load":
                    st["head"] = "⏳ 載入模型中…（第一次使用會自動下載模型，約 2–5 GB）"
            elif k == "progress":
                st.update(done=ev["done"], total=ev["total"], frac=0.10 + 0.75 * ev["done"] / max(1, ev["total"]))
            elif k == "file_done":
                st["frac"] = 1.0
                outputs += [Path(p) for p in ev.get("outputs", [])]
                txt = next((Path(p) for p in ev.get("outputs", []) if p.endswith(".txt")), None) or \
                    next((Path(p) for p in ev.get("outputs", []) if p.endswith(".md")), None)
                if txt and txt.exists():
                    preview = txt.read_text(encoding="utf-8")[:20000]
            files_n = max(1, st.get("files", 1))
            overall = ((max(st.get("file", 1), 1) - 1) + st.get("frac", 0)) / files_n
            progress(overall, desc=f"{st.get('stage_text', '')}")
        else:
            log.append(item)
        now = time.time()
        if now - last_yield > 0.25:
            last_yield = now
            yield (_status(st), "\n".join(log[-400:]), gr.update(value=preview) if preview else gr.update(),
                   gr.update(), gr.update(), gr.update())


def open_folder(path: str):
    p = Path(path.strip().strip('"')) if path and path.strip() else OUTPUTS
    if not p.exists():
        return f"找不到資料夾：{p}"
    try:
        if os.name == "nt":
            os.startfile(str(p))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return f"📂 已開啟：`{p}`"
    except Exception as e:  # noqa: BLE001
        return f"無法開啟：{e}"


def load_glossary():
    return GLOSSARY.read_text(encoding="utf-8") if GLOSSARY.exists() else ""


def save_glossary(t):
    GLOSSARY.write_text(t, encoding="utf-8")
    g = stt.Glossary.load(GLOSSARY)
    return f"✅ 已儲存：專有名詞 {len(g.terms)} 個、修正規則 {len(g.replacements)} 條。下次轉錄時生效。"


# ════════════════════════════════════════════════════════════════════
# 介面
# ════════════════════════════════════════════════════════════════════
CSS = """
.gradio-container {max-width: 1150px !important; margin: auto}
#title h1 {margin-bottom: 0}
#gpu {border-left: 4px solid var(--color-accent); padding: 6px 12px; background: var(--block-background-fill)}
.log textarea {font-family: ui-monospace, Consolas, monospace !important; font-size: 12px !important}
#preview textarea {font-size: 15px !important; line-height: 1.7 !important}
"""

ENGINE_HELP = """
| 情境 | 建議引擎 |
|---|---|
| 中文課堂、會議（台灣華語、中英夾雜、方言） | **Qwen3-ASR-1.7B**（預設）或 Breeze-ASR-25 |
| 日文、英文或其他語言 | Qwen3-ASR-1.7B 或 Whisper large-v3 |
| 趕時間、長時間錄音先看大意 | Whisper large-v3-turbo |
| 顯示卡 8 GB 以下 | Qwen3-ASR-0.6B 或 large-v3-turbo |

同一份錄音可以用兩個引擎各轉一次，比較 MD 檔最後的「⚠ 需要人工確認」清單。
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"{stt.APP_NAME} V{stt.VERSION}") as demo:
        gr.Markdown(f"# 🎙️🐰 {stt.APP_NAME} V{stt.VERSION}\n"
                    "音檔／影片 → 逐字稿（TXT）、字幕（SRT）、含時間戳記的 MD。自動略過靜音與音樂、偵測並重轉亂碼與重複詞。",
                    elem_id="title")
        with gr.Row():
            gpu_md = gr.Markdown("⏳ 檢查 GPU 環境中…", elem_id="gpu")
            gpu_btn = gr.Button("重新檢查", size="sm", scale=0, min_width=100)
        last_out = gr.State("")

        with gr.Tabs():
            with gr.Tab("① 轉錄"):
                with gr.Row():
                    with gr.Column(scale=3):
                        in_files = gr.File(label="音檔或影片（可多個；mp3 / m4a / wav / mp4 / mov …）",
                                           file_count="multiple", file_types=["audio", "video"])
                        in_paths = gr.Textbox(label="或貼上檔案／資料夾路徑（一行一個；大檔案建議用這個，不用上傳）",
                                              lines=3, placeholder=r"C:\Users\...\錄音\2026-09-30 會議.m4a")
                    with gr.Column(scale=2):
                        engine = gr.Dropdown(list(ENGINE_LABELS), value=stt.ENGINES[stt.DEFAULT_ENGINE]["label"],
                                             label="辨識引擎")
                        lang = gr.Dropdown(list(LANG_MODES), value=list(LANG_MODES)[0], label="語言")
                        allowed = gr.CheckboxGroup(list(ALLOWED), value=["中文", "日文", "英文"],
                                                   label="自動判斷時只在這些語言中選（避免中日亂碼）")
                        zh = gr.Radio(list(ZH_MODES), value=list(ZH_MODES)[0], label="中文輸出")
                fmts = gr.CheckboxGroup(list(FORMATS), value=list(FORMATS)[:3], label="輸出格式")
                with gr.Accordion("進階設定", open=False):
                    with gr.Row():
                        device = gr.Dropdown(["auto", "cuda", "cuda:0", "cuda:1", "cpu"], value="auto", label="運算裝置",
                                             info="auto 會自動使用 NVIDIA GPU")
                        batch = gr.Number(0, precision=0, label="批次大小（0＝依顯示卡記憶體自動）")
                        max_chars = gr.Slider(12, 30, value=20, step=1, label="中日文字幕每行最多字數")
                    with gr.Row():
                        vad_thr = gr.Slider(0.2, 0.8, value=0.5, step=0.05, label="語音偵測門檻（VAD）",
                                            info="背景很吵、漏句時調低（0.35）；雜音被當成說話時調高（0.6）")
                        max_chunk = gr.Slider(10, 29, value=28, step=1, label="每段最長秒數",
                                              info="模型一次看的長度；越長上下文越多")
                    with gr.Row():
                        retry = gr.Checkbox(True, label="自動重轉有問題的段落（亂碼、重複詞、幻覺句）")
                        precise = gr.Checkbox(True, label="精準字幕時間（字級對齊；Qwen 會多載入 0.6B 對齊模型）")
                        skip = gr.Checkbox(False, label="略過已經轉過的檔案")
                    with gr.Row():
                        limit_min = gr.Number(0, label="只轉前 N 分鐘（試跑用；0＝全部）")
                        start_at = gr.Textbox("", label="從某時間開始（選填，例如 00:19:00）")
                    out_dir = gr.Textbox("", label="輸出資料夾（空白＝存在原檔旁邊；上傳的檔案存到程式的 outputs 資料夾）")
                with gr.Row():
                    go = gr.Button("▶ 開始轉錄", variant="primary", scale=3)
                    stop = gr.Button("⏹ 停止", variant="stop", scale=1)
                    open_btn = gr.Button("📂 開啟輸出資料夾", scale=1)
                status = gr.Markdown("")
                with gr.Row():
                    preview = gr.Textbox(label="逐字稿預覽", lines=16, max_lines=30, elem_id="preview",
                                         interactive=False)
                    out_files = gr.File(label="下載（TXT / SRT / MD）", file_count="multiple", interactive=False)
                flagged = gr.Dataframe(headers=["檔案", "時間", "原因", "內容"], label="⚠ 需要人工確認的片段",
                                       interactive=False, wrap=True, value=[["", "", "（轉錄完成後顯示）", ""]])
                with gr.Accordion("詳細紀錄", open=False):
                    log_box = gr.Textbox(lines=16, max_lines=40, elem_classes="log", show_label=False, autoscroll=True)
                with gr.Accordion("該選哪個引擎？", open=False):
                    gr.Markdown(ENGINE_HELP)

            with gr.Tab("② 專有名詞與修正表"):
                gr.Markdown("**專有名詞**：一行一個（或用逗號分隔），會提示給模型，提高課程術語、人名、地名的辨識率。\n\n"
                            "**修正規則**：`錯誤 => 正確`，轉錄完成後自動取代（例如 `炭匯 => 碳匯`）。以 `#` 開頭的行是註解。")
                gl_text = gr.Code(load_glossary(), language=None, lines=24, label="glossary.txt")
                with gr.Row():
                    gl_save = gr.Button("儲存", variant="primary")
                    gl_reload = gr.Button("重新載入")
                gl_msg = gr.Markdown("")

            with gr.Tab("說明"):
                gr.Markdown((APP_DIR / "README.md").read_text(encoding="utf-8") if (APP_DIR / "README.md").exists() else "")

        demo.load(gpu_status_md, outputs=gpu_md)
        gpu_btn.click(gpu_status_md, outputs=gpu_md)
        go.click(do_transcribe,
                 [in_files, in_paths, engine, lang, allowed, zh, fmts, device, batch, vad_thr, max_chunk, retry,
                  precise, limit_min, start_at, skip, max_chars, out_dir],
                 [status, log_box, preview, out_files, flagged, last_out])
        stop.click(stop_job, None, status)
        open_btn.click(lambda a, b: open_folder(a or b), [last_out, out_dir], status)
        gl_save.click(save_glossary, gl_text, gl_msg)
        gl_reload.click(load_glossary, None, gl_text)
    return demo


def open_window(url: str):
    """優先用 Edge / Chrome 的 app 模式開成獨立視窗（看起來像一般 Windows 軟體），否則用預設瀏覽器。"""
    if os.name == "nt":
        cands = [os.path.expandvars(p) for p in (
            r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
            r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
            r"%ProgramFiles%\Google\Chrome\Application\chrome.exe")]
        for exe in cands:
            if os.path.exists(exe):
                try:
                    subprocess.Popen([exe, f"--app={url}", "--window-size=1220,980"])
                    return
                except Exception:  # noqa: BLE001
                    pass
    import webbrowser
    webbrowser.open(url)


def main():
    import argparse
    import socket
    import traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    stt.setup_stdio()

    def free_port(start: int) -> int:
        for p in range(start, start + 40):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return start

    port = a.port or free_port(7861)
    url = f"http://127.0.0.1:{port}"
    print(f"Python  : {sys.executable}")
    print(f"Gradio  : {gr.__version__}")
    print(f"網址    : {url}   （視窗沒自動打開的話，請手動在瀏覽器貼上這個網址）")
    print("使用中請不要關閉這個黑色視窗；要結束程式時關掉它即可。\n", flush=True)
    try:
        demo = build_ui()
        demo.queue(default_concurrency_limit=1)
        kw = dict(server_name="127.0.0.1", server_port=port, inbrowser=False, share=False,
                  prevent_thread_lock=True, allowed_paths=[str(APP_DIR), str(Path.home())],
                  favicon_path=str(APP_DIR / "app.ico") if (APP_DIR / "app.ico").exists() else None)
        try:  # Gradio 6：theme / css 放在 launch()
            demo.launch(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"), css=CSS, **kw)
        except TypeError:
            demo.css = CSS
            demo.launch(**kw)
        if not a.no_browser:
            open_window(url)
        demo.block_thread()
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("\n[錯誤] 介面無法啟動（見上方訊息）。請把這個視窗截圖。\n"
              "常見原因：防毒/防火牆封鎖本機連線、VPN/Proxy、或套件版本不符。")
        sys.exit(1)


if __name__ == "__main__":
    main()
