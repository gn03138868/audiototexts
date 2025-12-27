# ==========================================
# 批次音檔轉錄 V5（智慧重轉版）
# 新增功能：
# 1. 自動偵測語意不明片段（信心度低、異常文字）
# 2. 自動重新轉錄問題片段
# 3. 用不同參數多次嘗試
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
from pydub import AudioSegment
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️🐰 音檔轉錄小兔歐 V5（智慧重轉版）")
        self.root.geometry("850x850")
        self.root.resizable(True, True)
        
        # 變數
        self.input_folder = StringVar()
        self.model_size = StringVar(value="large-v3")
        self.output_folder = StringVar(value=os.getcwd())
        self.max_file_size = IntVar(value=100)
        self.chunk_length = IntVar(value=5)
        self.use_gpu = BooleanVar(value=True)
        self.min_segment_length = DoubleVar(value=2.0)
        self.merge_short_segments = BooleanVar(value=True)
        self.remove_duplicates = BooleanVar(value=True)
        
        # 新增：語意不明重轉設定
        self.auto_retry_unclear = BooleanVar(value=True)
        self.confidence_threshold = DoubleVar(value=-0.8)  # 信心度閾值
        self.max_retry_attempts = IntVar(value=3)  # 最大重試次數
        
        self.is_processing = False
        self.model = None
        self.audio_files = []
        self.temp_dir = os.path.join(os.getcwd(), "temp_chunks")
        
        self.gui_queue = queue.Queue()
        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus'}
        
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_info = f"✅ GPU: {torch.cuda.get_device_name(0)}"
        else:
            self.gpu_info = "❌ GPU 不可用"
            self.use_gpu.set(False)
        
        # 語意不明的判斷模式
        self.unclear_patterns = [
            # 中日文不自然混合（日文語法 + 簡體中文）
            r'[ぁ-んァ-ン][们这那什么怎样][ぁ-んァ-ン]',
            # 重複字符過多
            r'(.)\1{4,}',
            # 奇怪的標點組合
            r'[。、]{3,}',
            # 純數字或符號（可能是亂碼）
            r'^[\d\s\.\,\-]+$',
            # 日文助詞後接簡體中文
            r'[はがをにでと][们这那什]',
        ]
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        self.check_ffmpeg()
        self.process_gui_queue()
    
    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = Canvas(main_container, bg='white')
        scrollbar = ttk.Scrollbar(main_container, orient=VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # ========== 1. 資料夾選擇 ==========
        folder_frame = ttk.LabelFrame(self.scrollable_frame, text="📁 資料夾", padding="10")
        folder_frame.grid(row=0, column=0, sticky=(W, E), pady=5)
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="音檔：").grid(row=0, column=0, sticky=W)
        ttk.Entry(folder_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, sticky=(W, E), padx=5)
        ttk.Button(folder_frame, text="瀏覽", command=self.browse_folder_input).grid(row=0, column=2)
        
        self.folder_info = ttk.Label(folder_frame, text="", foreground="gray")
        self.folder_info.grid(row=1, column=0, columnspan=3, sticky=W)
        
        ttk.Label(folder_frame, text="輸出：").grid(row=2, column=0, sticky=W, pady=(5,0))
        ttk.Entry(folder_frame, textvariable=self.output_folder, width=50).grid(row=2, column=1, sticky=(W, E), padx=5, pady=(5,0))
        ttk.Button(folder_frame, text="瀏覽", command=self.browse_folder_output).grid(row=2, column=2, pady=(5,0))
        
        # ========== 2. 模型設定 ==========
        model_frame = ttk.LabelFrame(self.scrollable_frame, text="⚙️ 模型設定", padding="10")
        model_frame.grid(row=1, column=0, sticky=(W, E), pady=5)
        
        left_frame = ttk.Frame(model_frame)
        left_frame.grid(row=0, column=0, sticky=W)
        
        ttk.Label(left_frame, text="模型：", font=('', 9, 'bold')).grid(row=0, column=0, sticky=W)
        for i, (text, value) in enumerate([
            ("medium（快）", "medium"),
            ("large-v2（準）", "large-v2"),
            ("large-v3（最準，推薦）", "large-v3")
        ]):
            ttk.Radiobutton(left_frame, text=text, variable=self.model_size, 
                           value=value).grid(row=i+1, column=0, sticky=W, padx=10)
        
        gpu_frame = ttk.Frame(model_frame)
        gpu_frame.grid(row=1, column=0, sticky=W, pady=(10,0))
        ttk.Label(gpu_frame, text=self.gpu_info, 
                 foreground="green" if self.gpu_available else "red").grid(row=0, column=0)
        ttk.Checkbutton(gpu_frame, text="使用 GPU", variable=self.use_gpu,
                       state="normal" if self.gpu_available else "disabled").grid(row=0, column=1, padx=10)
        
        # ========== 3. 智慧重轉設定（新增！）==========
        retry_frame = ttk.LabelFrame(self.scrollable_frame, text="🔄 智慧重轉設定（語意不明自動重試）", padding="10")
        retry_frame.grid(row=2, column=0, sticky=(W, E), pady=5)
        
        ttk.Checkbutton(retry_frame, 
                       text="✅ 啟用語意不明自動重轉",
                       variable=self.auto_retry_unclear).grid(row=0, column=0, sticky=W)
        
        # 說明
        desc_label = ttk.Label(retry_frame, 
            text="當偵測到以下情況時，會自動用不同參數重新轉錄：\n"
                 "• 信心度過低（模型不確定）\n"
                 "• 中日文不自然混合（如「彼は专做」）\n"
                 "• 重複字符過多\n"
                 "• 壓縮比異常（可能是幻覺）",
            foreground="gray", font=('', 9))
        desc_label.grid(row=1, column=0, sticky=W, pady=(5,10))
        
        # 信心度閾值
        conf_frame = ttk.Frame(retry_frame)
        conf_frame.grid(row=2, column=0, sticky=W)
        ttk.Label(conf_frame, text="信心度閾值：").grid(row=0, column=0)
        ttk.Spinbox(conf_frame, from_=-1.5, to=-0.3, increment=0.1,
                   textvariable=self.confidence_threshold, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(conf_frame, text="（越低越嚴格，建議 -0.8）").grid(row=0, column=2)
        
        # 最大重試次數
        retry_count_frame = ttk.Frame(retry_frame)
        retry_count_frame.grid(row=3, column=0, sticky=W, pady=(5,0))
        ttk.Label(retry_count_frame, text="最大重試次數：").grid(row=0, column=0)
        ttk.Spinbox(retry_count_frame, from_=1, to=5, increment=1,
                   textvariable=self.max_retry_attempts, width=5).grid(row=0, column=1, padx=5)
        ttk.Label(retry_count_frame, text="次").grid(row=0, column=2)
        
        # ========== 4. 後處理設定 ==========
        post_frame = ttk.LabelFrame(self.scrollable_frame, text="🔧 後處理設定", padding="10")
        post_frame.grid(row=3, column=0, sticky=(W, E), pady=5)
        
        ttk.Checkbutton(post_frame, text="合併過短片段（< 2 秒）",
                       variable=self.merge_short_segments).grid(row=0, column=0, sticky=W)
        
        ttk.Checkbutton(post_frame, text="移除重複內容",
                       variable=self.remove_duplicates).grid(row=0, column=1, sticky=W, padx=(20,0))
        
        chunk_frame = ttk.Frame(post_frame)
        chunk_frame.grid(row=1, column=0, columnspan=2, sticky=W, pady=(10,0))
        ttk.Label(chunk_frame, text="大檔案分段：每").grid(row=0, column=0)
        ttk.Spinbox(chunk_frame, from_=3, to=10, textvariable=self.chunk_length, width=5).grid(row=0, column=1, padx=5)
        ttk.Label(chunk_frame, text="分鐘").grid(row=0, column=2)
        
        # ========== 5. 進度 ==========
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="⏳ 進度", padding="10")
        progress_frame.grid(row=4, column=0, sticky=(W, E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.overall_label = ttk.Label(progress_frame, text="整體：0/0")
        self.overall_label.grid(row=0, column=0, sticky=W)
        self.overall_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_bar.grid(row=1, column=0, sticky=(W, E), pady=2)
        
        self.current_label = ttk.Label(progress_frame, text="當前：-")
        self.current_label.grid(row=2, column=0, sticky=W, pady=(5,0))
        self.current_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.current_bar.grid(row=3, column=0, sticky=(W, E), pady=2)
        
        self.status_label = ttk.Label(progress_frame, text="等待開始", foreground="blue")
        self.status_label.grid(row=4, column=0, sticky=W)
        
        # 重轉統計
        self.retry_stats_label = ttk.Label(progress_frame, text="", foreground="orange")
        self.retry_stats_label.grid(row=5, column=0, sticky=W)
        
        # ========== 6. 日誌 ==========
        log_frame = ttk.LabelFrame(self.scrollable_frame, text="📋 日誌", padding="10")
        log_frame.grid(row=5, column=0, sticky=(W, E), pady=5)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=10, wrap=WORD, state='disabled', bg='#f5f5f5')
        self.log_text.grid(row=0, column=0, sticky=(W, E), padx=5, pady=5)
        
        # ========== 7. 按鈕 ==========
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.grid(row=6, column=0, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="🚀 開始", command=self.start_transcription, width=15)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹️ 停止", command=self.stop_transcription, 
                                   state='disabled', width=15)
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        ttk.Button(btn_frame, text="📂 開啟輸出", command=self.open_output_folder, width=15).grid(row=0, column=2, padx=5)
        
        self.root.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # ==================== GUI 輔助方法 ====================
    
    def process_gui_queue(self):
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
                    self.overall_label.config(text=f"整體：{task['current']}/{task['total']}")
                    if task['total'] > 0:
                        self.overall_bar['value'] = (task['current'] / task['total']) * 100
                elif task_type == 'current':
                    self.current_label.config(text=f"當前：{task['filename']}")
                elif task_type == 'msgbox':
                    if task['box'] == 'info':
                        messagebox.showinfo(task['title'], task['msg'])
                    elif task['box'] == 'error':
                        messagebox.showerror(task['title'], task['msg'])
                    elif task['box'] == 'askyesno':
                        result = messagebox.askyesno(task['title'], task['msg'])
                        if task.get('callback'):
                            task['callback'](result)
        except queue.Empty:
            pass
        self.root.after(100, self.process_gui_queue)
    
    def log(self, msg):
        self.gui_queue.put({'type': 'log', 'msg': msg})
    
    def status(self, msg, color="blue"):
        self.gui_queue.put({'type': 'status', 'msg': msg, 'color': color})
    
    def retry_stats(self, msg):
        self.gui_queue.put({'type': 'retry_stats', 'msg': msg})
    
    def progress(self, current, total):
        self.gui_queue.put({'type': 'progress', 'current': current, 'total': total})
    
    def current_file(self, filename):
        self.gui_queue.put({'type': 'current', 'filename': filename})
    
    def msgbox(self, box, title, msg, callback=None):
        self.gui_queue.put({'type': 'msgbox', 'box': box, 'title': title, 'msg': msg, 'callback': callback})

    def check_ffmpeg(self):
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=3)
            if result.returncode == 0:
                self.log("✅ ffmpeg OK")
                return
        except:
            pass
        self.log("⚠️ 找不到 ffmpeg")
    
    def browse_folder_input(self):
        folder = filedialog.askdirectory(title="選擇音檔資料夾")
        if folder:
            self.input_folder.set(folder)
            self.scan_files(folder)
    
    def browse_folder_output(self):
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.output_folder.set(folder)
    
    def scan_files(self, folder):
        self.audio_files = []
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in self.audio_extensions:
                self.audio_files.append(os.path.join(folder, f))
        self.audio_files.sort()
        
        if self.audio_files:
            total = sum(os.path.getsize(f) for f in self.audio_files) / (1024*1024)
            self.folder_info.config(text=f"✅ {len(self.audio_files)} 個音檔，共 {total:.1f} MB", foreground="green")
        else:
            self.folder_info.config(text="❌ 沒有找到音檔", foreground="red")

    def open_output_folder(self):
        folder = self.output_folder.get()
        if os.path.exists(folder):
            os.startfile(folder)

    def clear_memory(self):
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
        
        # 2. 靜音概率過高
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
        """取得轉錄參數，根據重試次數調整"""
        fp16 = (device == "cuda")
        
        # 基礎參數
        options = {
            "task": "transcribe",
            "verbose": False,
            "fp16": fp16,
            "language": None,  # 自動偵測
            "condition_on_previous_text": False,
            "word_timestamps": True,
        }
        
        # 根據重試次數調整參數
        if attempt == 0:
            # 第一次：標準參數
            options["temperature"] = 0.0
            options["no_speech_threshold"] = 0.5
            options["logprob_threshold"] = -1.0
            options["compression_ratio_threshold"] = 2.4
        elif attempt == 1:
            # 第二次：稍微放寬
            options["temperature"] = 0.2
            options["no_speech_threshold"] = 0.4
            options["logprob_threshold"] = -1.2
            options["compression_ratio_threshold"] = 2.6
        elif attempt == 2:
            # 第三次：更積極
            options["temperature"] = (0.0, 0.2, 0.4)
            options["no_speech_threshold"] = 0.3
            options["logprob_threshold"] = -1.5
            options["compression_ratio_threshold"] = 2.8
            options["beam_size"] = 5
            options["best_of"] = 5
        else:
            # 第四次以上：最積極
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
            options["initial_prompt"] = "這是中文對話。會議討論內容。"
        elif lang == "ja":
            options["initial_prompt"] = "これは日本語の会話です。会議の内容です。"
        elif lang == "en":
            options["initial_prompt"] = "This is English conversation. Meeting discussion."
        
        return options
    
    def start_transcription(self):
        if not self.audio_files:
            self.msgbox('error', "錯誤", "請先選擇音檔資料夾！")
            return
        
        device = "GPU" if self.use_gpu.get() and self.gpu_available else "CPU"
        retry_status = "開啟" if self.auto_retry_unclear.get() else "關閉"
        
        msg = (f"準備處理 {len(self.audio_files)} 個音檔\n\n"
               f"模型：{self.model_size.get()}\n"
               f"設備：{device}\n"
               f"智慧重轉：{retry_status}\n\n"
               f"開始？")
        
        def on_confirm(yes):
            if yes:
                self.is_processing = True
                self.start_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
                threading.Thread(target=self.run_transcription, daemon=True).start()
        
        self.msgbox('askyesno', "確認", msg, on_confirm)
    
    def run_transcription(self):
        success = 0
        fail = 0
        total_retries = 0
        
        try:
            self.log("=" * 50)
            self.log("🌐 多語言模式 + 智慧重轉")
            self.clear_memory()
            
            device = "cuda" if self.use_gpu.get() and self.gpu_available else "cpu"
            self.log(f"🖥️ 設備：{device}")
            
            self.status("載入模型...", "blue")
            self.log(f"🤖 載入 {self.model_size.get()}...")
            
            self.model = whisper.load_model(self.model_size.get(), device=device)
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
            
            self.log("✅ 模型載入完成")
            
            total = len(self.audio_files)
            
            for i, audio_file in enumerate(self.audio_files, 1):
                if not self.is_processing:
                    break
                
                filename = os.path.basename(audio_file)
                self.log(f"\n{'='*50}")
                self.log(f"📄 [{i}/{total}] {filename}")
                
                self.current_file(filename)
                self.progress(i-1, total)
                
                try:
                    retries = self.transcribe_file(audio_file, device)
                    total_retries += retries
                    success += 1
                except Exception as e:
                    fail += 1
                    self.log(f"❌ 錯誤：{e}")
                    import traceback
                    self.log(traceback.format_exc())
                
                self.clear_memory()
                self.progress(i, total)
                self.retry_stats(f"累計重轉：{total_retries} 個片段")
            
            self.status("✅ 完成！", "green")
            self.log(f"\n{'='*50}")
            self.log(f"🎉 完成！成功 {success}，失敗 {fail}")
            self.log(f"🔄 共重轉 {total_retries} 個語意不明片段")
            
            self.msgbox('info', "完成", 
                       f"成功：{success}\n失敗：{fail}\n\n重轉片段：{total_retries} 個")
            
        except Exception as e:
            self.log(f"❌ 錯誤：{e}")
            self.msgbox('error', "錯誤", str(e))
        
        finally:
            self.is_processing = False
            self.model = None
            self.clear_memory()
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            self.root.after(0, lambda: self.current_bar.stop())
    
    def transcribe_file(self, audio_file, device):
        """轉錄單個檔案，返回重轉次數"""
        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        self.log(f"  📦 {size_mb:.1f} MB")
        
        # 載入音檔（用於重轉片段）
        try:
            self.full_audio = AudioSegment.from_file(audio_file)
        except Exception as e:
            self.log(f"  ⚠️ 無法載入音檔：{e}")
            raise
        
        # 轉錄
        if size_mb > self.max_file_size.get():
            self.log(f"  ✂️ 分段處理...")
            result = self.transcribe_chunked(audio_file, device)
        else:
            result = self.transcribe_direct(audio_file, device)
        
        # 智慧重轉
        retry_count = 0
        if self.auto_retry_unclear.get():
            result, retry_count = self.retry_unclear_segments(result, device)
        
        # 後處理
        result = self.post_process(result)
        
        # 儲存
        self.save_result(audio_file, result)
        self.log(f"  ✅ 完成，{len(result.get('segments', []))} 個片段")
        
        return retry_count
    
    def transcribe_direct(self, audio_file, device):
        """直接轉錄"""
        self.status("轉錄中...", "orange")
        self.current_bar.start()
        
        options = self.get_transcribe_options(device, attempt=0)
        
        start = time.time()
        result = self.model.transcribe(audio_file, **options)
        elapsed = time.time() - start
        
        self.log(f"  🌐 語言：{result.get('language', '?')}")
        self.log(f"  ⏱️ {elapsed:.1f} 秒")
        self.current_bar.stop()
        
        return result
    
    def transcribe_chunked(self, audio_file, device):
        """分段轉錄"""
        chunk_ms = self.chunk_length.get() * 60 * 1000
        audio = self.full_audio
        
        duration_min = len(audio) / (1000 * 60)
        self.log(f"  ⏱️ 時長：{duration_min:.1f} 分鐘")
        
        overlap_ms = 2000
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_ms, len(audio))
            chunks.append((start, audio[start:end]))
            start = end - overlap_ms
        
        self.log(f"  📊 分為 {len(chunks)} 段")
        
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
                self.log(f"    ⚠️ 片段 {i+1} 語意不明：{', '.join(reasons)}")
                self.log(f"       原文：{seg['text'][:50]}...")
                
                # 嘗試重新轉錄
                best_seg = seg
                best_score = seg.get("avg_logprob", -999)
                
                for attempt in range(max_attempts):
                    new_seg = self.retry_single_segment(seg, device, attempt)
                    
                    if new_seg:
                        new_score = new_seg.get("avg_logprob", -999)
                        new_reasons = self.is_unclear_segment(new_seg)
                        
                        # 如果新結果更好（信心度更高且問題更少）
                        if new_score > best_score and len(new_reasons) < len(reasons):
                            best_seg = new_seg
                            best_score = new_score
                            self.log(f"       ✅ 重轉 {attempt+1}：{new_seg['text'][:50]}...")
                
                if best_seg != seg:
                    retry_count += 1
                
                improved_segments.append(best_seg)
            else:
                improved_segments.append(seg)
        
        if retry_count > 0:
            self.log(f"  🔄 共改善 {retry_count} 個片段")
        
        full_text = " ".join([s["text"] for s in improved_segments if s.get("text")])
        
        return {
            "text": full_text,
            "segments": improved_segments,
            "language": result.get("language", "unknown")
        }, retry_count
    
    def retry_single_segment(self, segment, device, attempt):
        """重新轉錄單個片段"""
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
                # 第一次：用不同的通用參數
                options = self.get_transcribe_options(device, attempt=1)
            elif attempt == 1:
                # 第二次：強制中文
                options = self.get_retry_options_for_language(device, "zh")
            elif attempt == 2:
                # 第三次：強制日文
                options = self.get_retry_options_for_language(device, "ja")
            else:
                # 其他：更積極的通用參數
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
            self.log(f"       ❌ 重轉失敗：{e}")
        
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
        
        if self.remove_duplicates.get():
            segments = self.remove_duplicate_segments(segments)
            removed = original_count - len(segments)
            if removed > 0:
                self.log(f"  🧹 移除 {removed} 個重複")
        
        if self.merge_short_segments.get():
            before = len(segments)
            segments = self.merge_short(segments)
            merged = before - len(segments)
            if merged > 0:
                self.log(f"  📎 合併 {merged} 個短片段")
        
        full_text = " ".join([s["text"] for s in segments if s.get("text")])
        
        return {
            "text": full_text,
            "segments": segments,
            "language": result.get("language", "unknown")
        }
    
    def remove_duplicate_segments(self, segments):
        """移除重複"""
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
    
    def save_result(self, audio_file, result):
        """儲存結果"""
        base = os.path.splitext(os.path.basename(audio_file))[0]
        out_dir = self.output_folder.get()
        
        # TXT
        txt_path = os.path.join(out_dir, f"{base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", ""))
        
        # MD
        md_path = os.path.join(out_dir, f"{base}_時間戳.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {base}\n\n")
            f.write(f"**模式**：多語言自動偵測 + 智慧重轉\n")
            f.write(f"**片段數**：{len(result.get('segments', []))}\n\n")
            f.write("---\n\n")
            
            for seg in result.get("segments", []):
                start = str(timedelta(seconds=int(seg["start"])))
                end = str(timedelta(seconds=int(seg["end"])))
                text = seg.get("text", "").strip()
                
                f.write(f"**[{start} → {end}]**\n\n{text}\n\n")
        
        self.log(f"  💾 已儲存")
    
    def stop_transcription(self):
        def on_confirm(yes):
            if yes:
                self.is_processing = False
                self.status("已停止", "red")
        self.msgbox('askyesno', "確認", "確定要停止？", on_confirm)


if __name__ == "__main__":
    root = Tk()
    app = WhisperGUI(root)
    root.mainloop()