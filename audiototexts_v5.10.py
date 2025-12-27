# ==========================================
# 🎙️🐰 音檔轉錄小兔歐 V5.10（智慧重轉版）
# 
# 繁體中文介面
# 
# 功能特色：
# 1. 自動偵測語意不明片段（信心度低、異常文字）
# 2. 自動重新轉錄問題片段
# 3. 用不同參數多次嘗試
# 4. 支援日中英多語言混合
# 5. 修復 ffprobe 問題
# ==========================================

import whisper
import os
import sys
import threading
import subprocess
import gc
import torch
import queue
import time
import re
from datetime import timedelta
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# 嘗試載入 pydub（可選）
PYDUB_AVAILABLE = False
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    pass


class WhisperTranscriberV5:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️🐰 音檔轉錄小兔歐 V5.10（智慧重轉版）")
        self.root.geometry("880x900")
        self.root.resizable(True, True)
        
        # ========== 變數初始化 ==========
        self.input_folder = StringVar()
        self.output_folder = StringVar(value=os.getcwd())
        self.model_size = StringVar(value="large-v3")
        self.transcribe_mode = StringVar(value="balanced")
        self.use_gpu = BooleanVar(value=True)
        
        # 輸出格式
        self.output_txt = BooleanVar(value=True)
        self.output_srt = BooleanVar(value=True)
        self.output_md = BooleanVar(value=True)
        
        # 智慧重轉設定
        self.auto_retry_unclear = BooleanVar(value=True)
        self.confidence_threshold = DoubleVar(value=-0.8)
        self.max_retry_attempts = IntVar(value=3)
        
        # 後處理設定
        self.merge_short_segments = BooleanVar(value=True)
        self.remove_duplicates = BooleanVar(value=True)
        self.min_segment_length = DoubleVar(value=2.0)
        
        # 大檔案處理
        self.max_file_size = IntVar(value=100)
        self.chunk_length = IntVar(value=5)
        
        # 狀態變數
        self.is_processing = False
        self.model = None
        self.audio_files = []
        self.full_audio = None
        self.temp_dir = os.path.join(os.getcwd(), "temp_chunks")
        
        # 執行緒安全佇列
        self.gui_queue = queue.Queue()
        
        # 支援的音訊格式
        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus', '.webm'}
        
        # 檢查 GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_info = f"✅ {gpu_name} ({gpu_mem:.1f} GB)"
        else:
            self.gpu_info = "❌ 無可用 GPU，將使用 CPU（速度較慢）"
            self.use_gpu.set(False)
        
        # 檢查 ffmpeg/ffprobe
        self.ffmpeg_ok, self.ffprobe_ok = self.check_ffmpeg_components()
        
        # 語意不明的判斷模式
        self.unclear_patterns = [
            # 中日文不自然混合（日文語法 + 簡體中文）
            r'[ぁ-んァ-ン][们这那什么怎样][ぁ-んァ-ン]',
            # 重複字元過多
            r'(.)\1{4,}',
            # 奇怪的標點組合
            r'[。、]{3,}',
            # 純數字或符號（可能是亂碼）
            r'^[\d\s\.\,\-]+$',
            # 日文助詞後接簡體中文
            r'[はがをにでと][们这那什]',
        ]
        
        # 設定介面樣式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 建立介面
        self.setup_ui()
        self.process_gui_queue()
    
    def check_ffmpeg_components(self):
        """檢查 ffmpeg 和 ffprobe"""
        ffmpeg_ok = False
        ffprobe_ok = False
        
        # 檢查當前目錄
        current_dir = os.getcwd()
        
        if os.path.exists(os.path.join(current_dir, "ffmpeg.exe")):
            ffmpeg_ok = True
            os.environ['PATH'] = current_dir + os.pathsep + os.environ.get('PATH', '')
        
        if os.path.exists(os.path.join(current_dir, "ffprobe.exe")):
            ffprobe_ok = True
        
        # 檢查系統 PATH
        try:
            result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_ok = True
                ffmpeg_path = result.stdout.strip().split('\n')[0]
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                
                # 檢查同目錄是否有 ffprobe
                if os.path.exists(os.path.join(ffmpeg_dir, 'ffprobe.exe')):
                    ffprobe_ok = True
        except:
            pass
        
        try:
            result = subprocess.run(['where', 'ffprobe'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ffprobe_ok = True
        except:
            pass
        
        return ffmpeg_ok, ffprobe_ok
    
    def setup_ui(self):
        """建立使用者介面"""
        # 主容器（含捲軸）
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient=VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # 滑鼠滾輪
        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind("<MouseWheel>", on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        row = 0
        
        # ==================== 1. 系統狀態 ====================
        status_frame = ttk.LabelFrame(self.scrollable_frame, text="🔧 系統狀態", padding="10")
        status_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        row += 1
        
        # GPU 狀態
        gpu_color = "green" if self.gpu_available else "red"
        ttk.Label(status_frame, text=f"GPU：{self.gpu_info}", 
                 foreground=gpu_color).grid(row=0, column=0, sticky=W)
        
        # FFmpeg 狀態
        if self.ffmpeg_ok:
            ffmpeg_text = "✅ FFmpeg：已找到"
            ffmpeg_color = "green"
        else:
            ffmpeg_text = "❌ FFmpeg：未找到"
            ffmpeg_color = "red"
        ttk.Label(status_frame, text=ffmpeg_text, foreground=ffmpeg_color).grid(row=1, column=0, sticky=W)
        
        # FFprobe 狀態
        if self.ffprobe_ok:
            ffprobe_text = "✅ FFprobe：已找到"
            ffprobe_color = "green"
        else:
            ffprobe_text = "⚠️ FFprobe：未找到（大檔案分段功能可能受限）"
            ffprobe_color = "orange"
        ttk.Label(status_frame, text=ffprobe_text, foreground=ffprobe_color).grid(row=2, column=0, sticky=W)
        
        # pydub 狀態
        if PYDUB_AVAILABLE:
            pydub_text = "✅ PyDub：已安裝"
            pydub_color = "green"
        else:
            pydub_text = "⚠️ PyDub：未安裝（智慧重轉功能受限）"
            pydub_color = "orange"
        ttk.Label(status_frame, text=pydub_text, foreground=pydub_color).grid(row=3, column=0, sticky=W)
        
        if not self.ffprobe_ok or not PYDUB_AVAILABLE:
            ttk.Label(status_frame, 
                     text="   💡 提示：請將 ffmpeg.exe 和 ffprobe.exe 放到程式資料夾", 
                     foreground="gray", font=('', 8)).grid(row=4, column=0, sticky=W)
        
        # ==================== 2. 檔案選擇 ====================
        file_frame = ttk.LabelFrame(self.scrollable_frame, text="📁 檔案設定", padding="10")
        file_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        row += 1
        
        ttk.Label(file_frame, text="音檔資料夾：").grid(row=0, column=0, sticky=W)
        ttk.Entry(file_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, sticky=(W, E), padx=5)
        ttk.Button(file_frame, text="瀏覽", command=self.browse_input).grid(row=0, column=2)
        
        self.file_info_label = ttk.Label(file_frame, text="尚未選擇資料夾", foreground="gray")
        self.file_info_label.grid(row=1, column=0, columnspan=3, sticky=W, pady=(5, 0))
        
        ttk.Label(file_frame, text="輸出資料夾：").grid(row=2, column=0, sticky=W, pady=(10, 0))
        ttk.Entry(file_frame, textvariable=self.output_folder, width=50).grid(row=2, column=1, sticky=(W, E), padx=5, pady=(10, 0))
        ttk.Button(file_frame, text="瀏覽", command=self.browse_output).grid(row=2, column=2, pady=(10, 0))
        
        # ==================== 3. 模型設定 ====================
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="🤖 模型設定", padding="10")
        model_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        row += 1
        
        # 左側：模型大小
        left_col = ttk.Frame(model_frame)
        left_col.grid(row=0, column=0, sticky=NW)
        
        ttk.Label(left_col, text="模型大小：", font=('', 9, 'bold')).grid(row=0, column=0, sticky=W)
        
        models = [
            ("medium — 速度較快", "medium"),
            ("large-v2 — 準確度高", "large-v2"),
            ("large-v3 — 最新版本（推薦）", "large-v3"),
        ]
        for i, (text, value) in enumerate(models):
            ttk.Radiobutton(left_col, text=text, variable=self.model_size, 
                           value=value).grid(row=i+1, column=0, sticky=W, padx=15)
        
        # 右側：轉錄模式
        right_col = ttk.Frame(model_frame)
        right_col.grid(row=0, column=1, sticky=NW, padx=(40, 0))
        
        ttk.Label(right_col, text="轉錄模式：", font=('', 9, 'bold')).grid(row=0, column=0, sticky=W)
        
        modes = [
            ("🛡️ 保守 — 減少幻覺", "conservative"),
            ("⚖️ 平衡 — 推薦使用", "balanced"),
            ("🔥 積極 — 減少漏句", "aggressive"),
        ]
        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(right_col, text=text, variable=self.transcribe_mode, 
                           value=value).grid(row=i+1, column=0, sticky=W, padx=15)
        
        # GPU 選項
        gpu_frame = ttk.Frame(model_frame)
        gpu_frame.grid(row=1, column=0, columnspan=2, sticky=W, pady=(15, 0))
        
        ttk.Checkbutton(gpu_frame, text="使用 GPU 加速（大幅提升速度）", 
                       variable=self.use_gpu,
                       state="normal" if self.gpu_available else "disabled").grid(row=0, column=0, sticky=W)
        
        # ==================== 4. 智慧重轉設定 ====================
        retry_frame = ttk.LabelFrame(self.scrollable_frame, text="🔄 智慧重轉設定（語意不明自動重試）", padding="10")
        retry_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        row += 1
        
        ttk.Checkbutton(retry_frame, 
                       text="✅ 啟用語意不明自動重轉",
                       variable=self.auto_retry_unclear).grid(row=0, column=0, sticky=W)
        
        # 說明
        desc_text = (
            "當偵測到以下情況時，會自動用不同參數重新轉錄：\n"
            "• 信心度過低（模型不確定）\n"
            "• 中日文不自然混合（如「彼は专做」）\n"
            "• 重複字元過多\n"
            "• 壓縮比異常（可能是幻覺）"
        )
        ttk.Label(retry_frame, text=desc_text, foreground="gray", 
                 font=('', 9)).grid(row=1, column=0, sticky=W, pady=(5, 10))
        
        # 參數設定
        param_frame = ttk.Frame(retry_frame)
        param_frame.grid(row=2, column=0, sticky=W)
        
        ttk.Label(param_frame, text="信心度閾值：").grid(row=0, column=0)
        ttk.Spinbox(param_frame, from_=-1.5, to=-0.3, increment=0.1,
                   textvariable=self.confidence_threshold, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="（越低越嚴格，建議 -0.8）").grid(row=0, column=2)
        
        ttk.Label(param_frame, text="最大重試次數：").grid(row=1, column=0, pady=(5, 0))
        ttk.Spinbox(param_frame, from_=1, to=5, increment=1,
                   textvariable=self.max_retry_attempts, width=6).grid(row=1, column=1, padx=5, pady=(5, 0))
        ttk.Label(param_frame, text="次").grid(row=1, column=2, pady=(5, 0))
        
        # ==================== 5. 後處理與輸出設定 ====================
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="📄 後處理與輸出設定", padding="10")
        output_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        row += 1
        
        # 後處理
        post_frame = ttk.Frame(output_frame)
        post_frame.grid(row=0, column=0, sticky=W)
        
        ttk.Checkbutton(post_frame, text="合併過短片段（< 2 秒）",
                       variable=self.merge_short_segments).grid(row=0, column=0, sticky=W)
        ttk.Checkbutton(post_frame, text="移除重複內容",
                       variable=self.remove_duplicates).grid(row=0, column=1, sticky=W, padx=(20, 0))
        
        # 輸出格式
        format_frame = ttk.Frame(output_frame)
        format_frame.grid(row=1, column=0, sticky=W, pady=(10, 0))
        
        ttk.Label(format_frame, text="輸出格式：").grid(row=0, column=0, sticky=W)
        ttk.Checkbutton(format_frame, text="TXT（純文字）", 
                       variable=self.output_txt).grid(row=0, column=1, sticky=W, padx=(10, 0))
        ttk.Checkbutton(format_frame, text="SRT（字幕檔）", 
                       variable=self.output_srt).grid(row=0, column=2, sticky=W, padx=(10, 0))
        ttk.Checkbutton(format_frame, text="MD（含時間戳記）", 
                       variable=self.output_md).grid(row=0, column=3, sticky=W, padx=(10, 0))
        
        # 大檔案分段
        chunk_frame = ttk.Frame(output_frame)
        chunk_frame.grid(row=2, column=0, sticky=W, pady=(10, 0))
        
        ttk.Label(chunk_frame, text="大檔案分段：超過").grid(row=0, column=0)
        ttk.Spinbox(chunk_frame, from_=50, to=500, increment=50,
                   textvariable=self.max_file_size, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(chunk_frame, text="MB 時，每").grid(row=0, column=2)
        ttk.Spinbox(chunk_frame, from_=3, to=10, increment=1,
                   textvariable=self.chunk_length, width=5).grid(row=0, column=3, padx=5)
        ttk.Label(chunk_frame, text="分鐘切一段").grid(row=0, column=4)
        
        # ==================== 6. 進度顯示 ====================
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="⏳ 處理進度", padding="10")
        progress_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        row += 1
        
        self.overall_label = ttk.Label(progress_frame, text="整體進度：0 / 0")
        self.overall_label.grid(row=0, column=0, sticky=W)
        
        self.overall_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.overall_bar.grid(row=1, column=0, sticky=(W, E), pady=5)
        
        self.current_label = ttk.Label(progress_frame, text="目前檔案：尚未開始")
        self.current_label.grid(row=2, column=0, sticky=W, pady=(10, 0))
        
        self.current_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=400)
        self.current_bar.grid(row=3, column=0, sticky=(W, E), pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="等待開始...", foreground="blue")
        self.status_label.grid(row=4, column=0, sticky=W)
        
        # 重轉統計
        self.retry_stats_label = ttk.Label(progress_frame, text="", foreground="orange")
        self.retry_stats_label.grid(row=5, column=0, sticky=W)
        
        # ==================== 7. 執行日誌 ====================
        log_frame = ttk.LabelFrame(self.scrollable_frame, text="📋 執行日誌", padding="10")
        log_frame.grid(row=row, column=0, sticky=(W, E), pady=5)
        log_frame.columnconfigure(0, weight=1)
        row += 1
        
        self.log_text = ScrolledText(log_frame, height=10, wrap=WORD, 
                                     state='disabled', bg='#f8f8f8', font=('Consolas', 9))
        self.log_text.grid(row=0, column=0, sticky=(W, E), pady=5)
        
        # ==================== 8. 控制按鈕 ====================
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.grid(row=row, column=0, pady=15)
        row += 1
        
        self.start_btn = ttk.Button(btn_frame, text="🚀 開始轉錄", 
                                    command=self.start_transcription, width=18)
        self.start_btn.grid(row=0, column=0, padx=8)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ 停止", 
                                   command=self.stop_transcription, 
                                   state='disabled', width=18)
        self.stop_btn.grid(row=0, column=1, padx=8)
        
        ttk.Button(btn_frame, text="📂 開啟輸出資料夾", 
                  command=self.open_output_folder, width=18).grid(row=0, column=2, padx=8)
        
        # 更新捲軸區域
        self.root.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # 顯示初始日誌
        self.log("=" * 55)
        self.log("🎙️🐰 音檔轉錄小兔歐 V5.1 已啟動")
        self.log("=" * 55)
        self.log(f"GPU：{'可用 ✅' if self.gpu_available else '不可用 ❌'}")
        self.log(f"FFmpeg：{'已找到 ✅' if self.ffmpeg_ok else '未找到 ❌'}")
        self.log(f"FFprobe：{'已找到 ✅' if self.ffprobe_ok else '未找到 ⚠️'}")
        self.log(f"PyDub：{'已安裝 ✅' if PYDUB_AVAILABLE else '未安裝 ⚠️'}")
        self.log("")

    # ==================== GUI 輔助方法 ====================
    
    def process_gui_queue(self):
        """處理 GUI 更新佇列"""
        try:
            while True:
                task = self.gui_queue.get_nowait()
                task_type = task.get('type')
                
                if task_type == 'log':
                    self.log_text.config(state='normal')
                    self.log_text.insert(END, task['msg'] + "\n")
                    self.log_text.see(END)
                    self.log_text.config(state='disabled')
                    
                elif task_type == 'status':
                    self.status_label.config(text=task['msg'], foreground=task.get('color', 'blue'))
                    
                elif task_type == 'retry_stats':
                    self.retry_stats_label.config(text=task['msg'])
                    
                elif task_type == 'progress':
                    self.overall_label.config(text=f"整體進度：{task['current']} / {task['total']}")
                    if task['total'] > 0:
                        self.overall_bar['value'] = (task['current'] / task['total']) * 100
                        
                elif task_type == 'current_file':
                    self.current_label.config(text=f"目前檔案：{task['filename']}")
                    
                elif task_type == 'msgbox':
                    if task['box'] == 'info':
                        messagebox.showinfo(task['title'], task['msg'])
                    elif task['box'] == 'error':
                        messagebox.showerror(task['title'], task['msg'])
                    elif task['box'] == 'warning':
                        messagebox.showwarning(task['title'], task['msg'])
                    elif task['box'] == 'askyesno':
                        result = messagebox.askyesno(task['title'], task['msg'])
                        if task.get('callback'):
                            task['callback'](result)
                            
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_gui_queue)
    
    def log(self, msg):
        """寫入日誌"""
        self.gui_queue.put({'type': 'log', 'msg': msg})
    
    def status(self, msg, color="blue"):
        """更新狀態"""
        self.gui_queue.put({'type': 'status', 'msg': msg, 'color': color})
    
    def retry_stats(self, msg):
        """更新重轉統計"""
        self.gui_queue.put({'type': 'retry_stats', 'msg': msg})
    
    def progress(self, current, total):
        """更新進度"""
        self.gui_queue.put({'type': 'progress', 'current': current, 'total': total})
    
    def current_file(self, filename):
        """更新目前檔案"""
        self.gui_queue.put({'type': 'current_file', 'filename': filename})
    
    def msgbox(self, box_type, title, msg, callback=None):
        """顯示訊息框"""
        self.gui_queue.put({'type': 'msgbox', 'box': box_type, 'title': title, 'msg': msg, 'callback': callback})
    
    def browse_input(self):
        """選擇輸入資料夾"""
        folder = filedialog.askdirectory(title="選擇音檔資料夾")
        if folder:
            self.input_folder.set(folder)
            self.scan_audio_files(folder)
    
    def browse_output(self):
        """選擇輸出資料夾"""
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.output_folder.set(folder)
    
    def scan_audio_files(self, folder):
        """掃描音檔"""
        self.audio_files = []
        
        try:
            for filename in os.listdir(folder):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.audio_extensions:
                    filepath = os.path.join(folder, filename)
                    self.audio_files.append(filepath)
            
            self.audio_files.sort()
            
            if self.audio_files:
                total_size = sum(os.path.getsize(f) for f in self.audio_files) / (1024 * 1024)
                info_text = f"✅ 找到 {len(self.audio_files)} 個音檔，共 {total_size:.1f} MB"
                self.file_info_label.config(text=info_text, foreground="green")
                
                self.log(f"📁 掃描資料夾：{folder}")
                self.log(f"   找到 {len(self.audio_files)} 個音檔")
            else:
                self.file_info_label.config(text="❌ 此資料夾沒有找到音檔", foreground="red")
                
        except Exception as e:
            self.file_info_label.config(text=f"錯誤：{str(e)}", foreground="red")
    
    def open_output_folder(self):
        """開啟輸出資料夾"""
        folder = self.output_folder.get()
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            self.msgbox('warning', "提示", "輸出資料夾不存在")
    
    def clear_memory(self):
        """清理記憶體"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================== 語意不明偵測 ====================
    
    def is_unclear_segment(self, segment):
        """判斷片段是否語意不明"""
        text = segment.get("text", "").strip()
        avg_logprob = segment.get("avg_logprob", 0)
        no_speech_prob = segment.get("no_speech_prob", 0)
        compression_ratio = segment.get("compression_ratio", 1)
        
        reasons = []
        
        # 1. 信心度過低
        if avg_logprob < self.confidence_threshold.get():
            reasons.append(f"信心度低({avg_logprob:.2f})")
        
        # 2. 靜音機率過高
        if no_speech_prob > 0.7:
            reasons.append(f"可能是靜音({no_speech_prob:.2f})")
        
        # 3. 壓縮比異常（可能是重複/幻覺）
        if compression_ratio > 2.5:
            reasons.append(f"壓縮比高({compression_ratio:.2f})")
        
        # 4. 文字模式異常
        for pattern in self.unclear_patterns:
            if re.search(pattern, text):
                reasons.append("文字模式異常")
                break
        
        # 5. 中日文不自然混合檢測
        if self.has_unnatural_mixing(text):
            reasons.append("中日混合不自然")
        
        # 6. 太短且信心度不高
        if len(text) < 3 and avg_logprob < -0.5:
            reasons.append("內容過短")
        
        return reasons
    
    def has_unnatural_mixing(self, text):
        """檢測中日文是否不自然混合"""
        # 日文假名
        hiragana = set('ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖ')
        katakana = set('ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ')
        
        # 簡體中文特有字（不在日文中使用）
        simplified_only = set('这那里么们该让给对为')
        
        has_kana = any(c in hiragana or c in katakana for c in text)
        has_simplified = any(c in simplified_only for c in text)
        
        # 如果同時有假名和簡體中文特有字，很可能是錯誤
        if has_kana and has_simplified:
            return True
        
        return False

    # ==================== 核心轉錄邏輯 ====================
    
    def get_transcribe_options(self, device, attempt=0):
        """取得轉錄參數，根據重試次數和模式調整"""
        fp16 = (device == "cuda")
        mode = self.transcribe_mode.get()
        
        # 基礎參數
        options = {
            "task": "transcribe",
            "verbose": False,
            "fp16": fp16,
            "language": None,  # 自動偵測
            "condition_on_previous_text": False,  # 避免錯誤累積
        }
        
        # 根據模式設定基礎閾值
        if mode == "conservative":
            base_no_speech = 0.6
            base_logprob = -0.8
            base_compression = 2.2
        elif mode == "balanced":
            base_no_speech = 0.5
            base_logprob = -1.0
            base_compression = 2.4
        else:  # aggressive
            base_no_speech = 0.3
            base_logprob = -1.5
            base_compression = 2.8
        
        # 根據重試次數調整參數
        if attempt == 0:
            options["temperature"] = 0.0
            options["no_speech_threshold"] = base_no_speech
            options["logprob_threshold"] = base_logprob
            options["compression_ratio_threshold"] = base_compression
        elif attempt == 1:
            options["temperature"] = 0.2
            options["no_speech_threshold"] = base_no_speech - 0.1
            options["logprob_threshold"] = base_logprob - 0.2
            options["compression_ratio_threshold"] = base_compression + 0.2
        elif attempt == 2:
            options["temperature"] = (0.0, 0.2, 0.4)
            options["no_speech_threshold"] = base_no_speech - 0.2
            options["logprob_threshold"] = base_logprob - 0.5
            options["compression_ratio_threshold"] = base_compression + 0.4
            options["beam_size"] = 5
            options["best_of"] = 5
        else:
            options["temperature"] = (0.0, 0.2, 0.4, 0.6)
            options["no_speech_threshold"] = 0.2
            options["logprob_threshold"] = -2.0
            options["compression_ratio_threshold"] = 3.0
            options["beam_size"] = 5
            options["best_of"] = 5
        
        return options
    
    def get_retry_options_for_language(self, device, lang):
        """針對特定語言的重試參數"""
        fp16 = (device == "cuda")
        
        options = {
            "task": "transcribe",
            "verbose": False,
            "fp16": fp16,
            "language": lang,
            "condition_on_previous_text": False,
            "temperature": 0.0,
            "no_speech_threshold": 0.3,
            "logprob_threshold": -1.5,
            "compression_ratio_threshold": 2.8,
        }
        
        # 針對不同語言的提示
        if lang == "zh":
            options["initial_prompt"] = "這是中文對話。會議討論內容。繁體中文。"
        elif lang == "ja":
            options["initial_prompt"] = "これは日本語の会話です。会議の内容です。"
        elif lang == "en":
            options["initial_prompt"] = "This is English conversation. Meeting discussion."
        
        return options
    
    def start_transcription(self):
        """開始轉錄"""
        # 驗證
        if not self.audio_files:
            self.msgbox('error', "錯誤", "請先選擇包含音檔的資料夾！")
            return
        
        if not self.ffmpeg_ok:
            self.msgbox('error', "錯誤", 
                       "找不到 FFmpeg！\n\n"
                       "請下載 FFmpeg：\n"
                       "1. 前往 https://www.gyan.dev/ffmpeg/builds/\n"
                       "2. 下載 ffmpeg-release-essentials.zip\n"
                       "3. 解壓縮後，將 bin 資料夾內的\n"
                       "   ffmpeg.exe 和 ffprobe.exe\n"
                       "   複製到程式同一資料夾")
            return
        
        # 確認設定
        device = "GPU" if self.use_gpu.get() and self.gpu_available else "CPU"
        mode_names = {"conservative": "保守", "balanced": "平衡", "aggressive": "積極"}
        retry_status = "開啟" if self.auto_retry_unclear.get() else "關閉"
        
        confirm_msg = (
            f"準備開始轉錄\n\n"
            f"📁 檔案數量：{len(self.audio_files)} 個\n"
            f"🤖 模型：{self.model_size.get()}\n"
            f"🎚️ 模式：{mode_names.get(self.transcribe_mode.get())}\n"
            f"🔄 智慧重轉：{retry_status}\n"
            f"💻 裝置：{device}\n\n"
            f"確定要開始嗎？"
        )
        
        def on_confirm(yes):
            if yes:
                self.is_processing = True
                self.start_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
                self.overall_bar['value'] = 0
                
                thread = threading.Thread(target=self.run_transcription, daemon=True)
                thread.start()
        
        self.msgbox('askyesno', "確認", confirm_msg, on_confirm)
    
    def run_transcription(self):
        """執行轉錄（背景執行緒）"""
        success_count = 0
        fail_count = 0
        total_retries = 0
        
        try:
            self.log("=" * 55)
            self.log("🚀 開始批次轉錄")
            self.log("=" * 55)
            
            self.clear_memory()
            
            device = "cuda" if self.use_gpu.get() and self.gpu_available else "cpu"
            self.log(f"💻 使用裝置：{device.upper()}")
            self.log(f"🎚️ 轉錄模式：{self.transcribe_mode.get()}")
            self.log(f"🔄 智慧重轉：{'開啟' if self.auto_retry_unclear.get() else '關閉'}")
            
            # 載入模型
            self.status("正在載入模型...", "blue")
            self.log(f"🤖 載入模型：{self.model_size.get()}...")
            
            load_start = time.time()
            self.model = whisper.load_model(self.model_size.get(), device=device)
            load_time = time.time() - load_start
            
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
            
            self.log(f"   載入完成（耗時 {load_time:.1f} 秒）")
            self.log("")
            
            total_files = len(self.audio_files)
            
            for index, audio_file in enumerate(self.audio_files, 1):
                if not self.is_processing:
                    self.log("⚠️ 使用者中止處理")
                    break
                
                filename = os.path.basename(audio_file)
                
                self.log("-" * 55)
                self.log(f"📄 [{index}/{total_files}] {filename}")
                
                self.current_file(filename)
                self.progress(index - 1, total_files)
                
                try:
                    retries = self.transcribe_single_file(audio_file, device)
                    total_retries += retries
                    success_count += 1
                    self.log(f"   ✅ 完成")
                    
                except Exception as e:
                    fail_count += 1
                    self.log(f"   ❌ 錯誤：{e}")
                    import traceback
                    self.log(f"   {traceback.format_exc()}")
                
                self.clear_memory()
                self.progress(index, total_files)
                self.retry_stats(f"累計重轉：{total_retries} 個片段")
            
            # 完成
            self.log("")
            self.log("=" * 55)
            self.log("🎉 批次處理完成！")
            self.log(f"   ✅ 成功：{success_count} 個")
            if fail_count > 0:
                self.log(f"   ❌ 失敗：{fail_count} 個")
            self.log(f"   🔄 重轉片段：{total_retries} 個")
            self.log("=" * 55)
            
            self.status("✅ 轉錄完成！", "green")
            self.current_file("全部完成")
            
            self.msgbox('info', "完成", 
                       f"批次轉錄完成！\n\n"
                       f"成功：{success_count} 個\n"
                       f"失敗：{fail_count} 個\n"
                       f"重轉片段：{total_retries} 個\n\n"
                       f"結果已儲存至輸出資料夾")
            
        except Exception as e:
            self.log(f"❌ 嚴重錯誤：{e}")
            import traceback
            self.log(traceback.format_exc())
            self.status("❌ 發生錯誤", "red")
            self.msgbox('error', "錯誤", f"轉錄過程發生錯誤：\n\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.model = None
            self.full_audio = None
            self.clear_memory()
            
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            self.root.after(0, lambda: self.current_bar.stop())
    
    def transcribe_single_file(self, audio_file, device):
        """轉錄單一檔案，回傳重轉次數"""
        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        self.log(f"   大小：{size_mb:.1f} MB")
        
        # 嘗試載入音檔（用於智慧重轉）
        self.full_audio = None
        if PYDUB_AVAILABLE and self.ffprobe_ok:
            try:
                self.full_audio = AudioSegment.from_file(audio_file)
            except Exception as e:
                self.log(f"   ⚠️ 無法載入音檔供重轉使用：{e}")
        
        # 轉錄
        if size_mb > self.max_file_size.get() and self.full_audio:
            self.log(f"   ✂️ 檔案較大，分段處理...")
            result = self.transcribe_chunked(audio_file, device)
        else:
            result = self.transcribe_direct(audio_file, device)
        
        # 智慧重轉
        retry_count = 0
        if self.auto_retry_unclear.get() and self.full_audio:
            result, retry_count = self.retry_unclear_segments(result, device)
        
        # 後處理
        result = self.post_process(result)
        
        # 儲存
        self.save_results(audio_file, result)
        
        return retry_count
    
    def transcribe_direct(self, audio_file, device):
        """直接轉錄"""
        self.status(f"轉錄中：{os.path.basename(audio_file)}", "orange")
        self.current_bar.start()
        
        options = self.get_transcribe_options(device, attempt=0)
        
        start_time = time.time()
        
        try:
            result = self.model.transcribe(audio_file, **options)
        except Exception as e:
            self.current_bar.stop()
            raise e
        
        elapsed = time.time() - start_time
        
        self.current_bar.stop()
        
        segments = result.get("segments", [])
        detected_lang = result.get("language", "?")
        
        self.log(f"   耗時：{elapsed:.1f} 秒")
        self.log(f"   偵測語言：{detected_lang}")
        self.log(f"   片段數：{len(segments)}")
        
        return result
    
    def transcribe_chunked(self, audio_file, device):
        """分段轉錄"""
        chunk_ms = self.chunk_length.get() * 60 * 1000
        audio = self.full_audio
        
        duration_min = len(audio) / (1000 * 60)
        self.log(f"   時長：{duration_min:.1f} 分鐘")
        
        # 切割（含重疊）
        overlap_ms = 2000  # 2 秒重疊
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_ms, len(audio))
            chunks.append((start, audio[start:end]))
            start = end - overlap_ms
        
        self.log(f"   分為 {len(chunks)} 段")
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        all_segments = []
        options = self.get_transcribe_options(device, attempt=0)
        result = None
        
        for i, (offset_ms, chunk) in enumerate(chunks, 1):
            if not self.is_processing:
                break
            
            self.status(f"轉錄片段 {i}/{len(chunks)}...", "orange")
            self.current_bar.start()
            
            temp_file = os.path.join(self.temp_dir, f"chunk_{i}.wav")
            chunk.export(temp_file, format="wav")
            
            try:
                result = self.model.transcribe(temp_file, **options)
                
                offset_sec = offset_ms / 1000.0
                for seg in result.get("segments", []):
                    # 跳過重疊區的重複內容
                    if i > 1 and seg["start"] < (overlap_ms / 1000):
                        continue
                    
                    new_seg = {
                        "start": seg["start"] + offset_sec,
                        "end": seg["end"] + offset_sec,
                        "text": seg["text"].strip(),
                        "avg_logprob": seg.get("avg_logprob", 0),
                        "no_speech_prob": seg.get("no_speech_prob", 0),
                        "compression_ratio": seg.get("compression_ratio", 1),
                    }
                    all_segments.append(new_seg)
                
            finally:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            self.clear_memory()
        
        self.current_bar.stop()
        
        full_text = " ".join([s["text"] for s in all_segments if s["text"]])
        
        return {
            "text": full_text,
            "segments": all_segments,
            "language": result.get("language", "unknown") if result else "unknown"
        }
    
    def retry_unclear_segments(self, result, device):
        """重新轉錄語意不明的片段"""
        segments = result.get("segments", [])
        if not segments:
            return result, 0
        
        retry_count = 0
        max_attempts = self.max_retry_attempts.get()
        improved_segments = []
        
        for i, seg in enumerate(segments):
            reasons = self.is_unclear_segment(seg)
            
            if reasons:
                self.log(f"      ⚠️ 片段 {i+1} 語意不明：{', '.join(reasons)}")
                self.log(f"         原文：{seg['text'][:50]}...")
                
                # 嘗試重新轉錄
                best_seg = seg
                best_score = seg.get("avg_logprob", -999)
                
                for attempt in range(max_attempts):
                    new_seg = self.retry_single_segment(seg, device, attempt)
                    
                    if new_seg:
                        new_score = new_seg.get("avg_logprob", -999)
                        new_reasons = self.is_unclear_segment(new_seg)
                        
                        # 如果新結果更好
                        if new_score > best_score and len(new_reasons) < len(reasons):
                            best_seg = new_seg
                            best_score = new_score
                            self.log(f"         ✅ 重轉 {attempt+1}：{new_seg['text'][:50]}...")
                
                if best_seg != seg:
                    retry_count += 1
                
                improved_segments.append(best_seg)
            else:
                improved_segments.append(seg)
        
        if retry_count > 0:
            self.log(f"   🔄 共改善 {retry_count} 個片段")
        
        full_text = " ".join([s["text"] for s in improved_segments if s.get("text")])
        
        return {
            "text": full_text,
            "segments": improved_segments,
            "language": result.get("language", "unknown")
        }, retry_count
    
    def retry_single_segment(self, segment, device, attempt):
        """重新轉錄單個片段"""
        if not self.full_audio:
            return None
        
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        
        # 擴展一點範圍
        start_ms = max(0, start_ms - 500)
        end_ms = min(len(self.full_audio), end_ms + 500)
        
        # 擷取片段
        segment_audio = self.full_audio[start_ms:end_ms]
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        temp_file = os.path.join(self.temp_dir, f"retry_{start_ms}.wav")
        segment_audio.export(temp_file, format="wav")
        
        try:
            # 根據嘗試次數選擇不同策略
            if attempt == 0:
                options = self.get_transcribe_options(device, attempt=1)
            elif attempt == 1:
                options = self.get_retry_options_for_language(device, "zh")
            elif attempt == 2:
                options = self.get_retry_options_for_language(device, "ja")
            else:
                options = self.get_transcribe_options(device, attempt=3)
            
            result = self.model.transcribe(temp_file, **options)
            
            if result.get("segments"):
                seg = result["segments"][0]
                return {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": seg["text"].strip(),
                    "avg_logprob": seg.get("avg_logprob", 0),
                    "no_speech_prob": seg.get("no_speech_prob", 0),
                    "compression_ratio": seg.get("compression_ratio", 1),
                }
            
        except Exception as e:
            self.log(f"         ❌ 重轉失敗：{e}")
        
        finally:
            try:
                os.remove(temp_file)
            except:
                pass
        
        return None
    
    def post_process(self, result):
        """後處理"""
        segments = result.get("segments", [])
        if not segments:
            return result
        
        original_count = len(segments)
        
        # 移除重複
        if self.remove_duplicates.get():
            segments = self.remove_duplicate_segments(segments)
            removed = original_count - len(segments)
            if removed > 0:
                self.log(f"   🧹 移除 {removed} 個重複")
        
        # 合併短片段
        if self.merge_short_segments.get():
            before = len(segments)
            segments = self.merge_short(segments)
            merged = before - len(segments)
            if merged > 0:
                self.log(f"   📎 合併 {merged} 個短片段")
        
        full_text = " ".join([s["text"] for s in segments if s.get("text")])
        
        return {
            "text": full_text,
            "segments": segments,
            "language": result.get("language", "unknown")
        }
    
    def remove_duplicate_segments(self, segments):
        """移除重複片段"""
        if not segments:
            return segments
        
        cleaned = []
        prev_text = ""
        repeat_count = 0
        
        for seg in segments:
            text = seg.get("text", "").strip()
            normalized = re.sub(r'[^\w]', '', text.lower())
            prev_normalized = re.sub(r'[^\w]', '', prev_text.lower())
            
            if normalized == prev_normalized and normalized:
                repeat_count += 1
                if repeat_count > 2:
                    continue
            else:
                repeat_count = 0
            
            if len(normalized) < 2:
                continue
            
            cleaned.append(seg)
            prev_text = text
        
        return cleaned
    
    def merge_short(self, segments):
        """合併短片段"""
        if not segments:
            return segments
        
        min_len = self.min_segment_length.get()
        merged = []
        
        i = 0
        while i < len(segments):
            seg = segments[i].copy()
            duration = seg["end"] - seg["start"]
            
            while duration < min_len and i + 1 < len(segments):
                next_seg = segments[i + 1]
                gap = next_seg["start"] - seg["end"]
                if gap > 2:
                    break
                
                seg["end"] = next_seg["end"]
                seg["text"] = seg["text"] + " " + next_seg.get("text", "")
                duration = seg["end"] - seg["start"]
                i += 1
            
            merged.append(seg)
            i += 1
        
        return merged
    
    def save_results(self, audio_file, result):
        """儲存轉錄結果"""
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = self.output_folder.get()
        
        segments = result.get("segments", [])
        full_text = result.get("text", "")
        detected_lang = result.get("language", "unknown")
        
        saved_files = []
        
        # TXT 格式
        if self.output_txt.get():
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            saved_files.append("TXT")
        
        # SRT 格式
        if self.output_srt.get():
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, 1):
                    start_time = self.format_srt_time(seg["start"])
                    end_time = self.format_srt_time(seg["end"])
                    text = seg.get("text", "").strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            saved_files.append("SRT")
        
        # MD 格式
        if self.output_md.get():
            md_path = os.path.join(output_dir, f"{base_name}_逐字稿.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {base_name}\n\n")
                f.write(f"**原始檔案**：{os.path.basename(audio_file)}\n")
                f.write(f"**偵測語言**：{detected_lang}\n")
                f.write(f"**片段數量**：{len(segments)}\n")
                f.write(f"**轉錄模式**：{self.transcribe_mode.get()}\n")
                f.write(f"**智慧重轉**：{'開啟' if self.auto_retry_unclear.get() else '關閉'}\n")
                
                if segments:
                    total_duration = segments[-1]["end"]
                    duration_str = str(timedelta(seconds=int(total_duration)))
                    f.write(f"**總時長**：{duration_str}\n")
                
                f.write(f"\n---\n\n")
                
                for seg in segments:
                    start = str(timedelta(seconds=int(seg["start"])))
                    end = str(timedelta(seconds=int(seg["end"])))
                    text = seg.get("text", "").strip()
                    
                    f.write(f"**[{start} → {end}]**\n\n")
                    f.write(f"{text}\n\n")
            saved_files.append("MD")
        
        self.log(f"   💾 已儲存：{', '.join(saved_files)}")
    
    def format_srt_time(self, seconds):
        """格式化 SRT 時間"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def stop_transcription(self):
        """停止轉錄"""
        def on_confirm(yes):
            if yes:
                self.is_processing = False
                self.status("⏹️ 已停止", "red")
                self.log("⚠️ 使用者停止轉錄")
        
        self.msgbox('askyesno', "確認", "確定要停止轉錄嗎？\n\n已完成的檔案會保留。", on_confirm)


# ==========================================
# 主程式進入點
# ==========================================
if __name__ == "__main__":
    root = Tk()
    app = WhisperTranscriberV5(root)
    root.mainloop()