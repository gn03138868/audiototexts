# ==========================================
# 批次音檔轉錄 GUI 版本（支援大檔案自動切割 + GPU 加速 + 滾動條 + 線程安全）
# ==========================================

import whisper
import os
import sys
import threading
import subprocess
import gc
import torch
import queue
from datetime import timedelta
from pydub import AudioSegment
from pydub.utils import make_chunks
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ 音檔轉錄小兔歐 (audiototexts) (丟丟丟 GPU 加速版)")
        self.root.geometry("773x773")
        self.root.resizable(True, True)
        
        # 變數
        self.input_folder = StringVar()
        self.model_size = StringVar(value="medium")
        self.language = StringVar(value="zh")
        self.output_folder = StringVar(value=os.getcwd())
        self.max_file_size = IntVar(value=200)  # 預設 200 MB
        self.chunk_length = IntVar(value=10)  # 預設每段 10 分鐘
        self.enable_vad = BooleanVar(value=True)  # 啟用靜音偵測
        self.repetition_detection = BooleanVar(value=True)  # 啟用重複偵測
        self.use_gpu = BooleanVar(value=True)  # 預設使用 GPU
        self.is_processing = False
        self.model = None
        self.audio_files = []
        self.temp_dir = os.path.join(os.getcwd(), "temp_chunks")  # 暫存資料夾
        
        # 線程安全隊列
        self.gui_queue = queue.Queue()
        
        # 支援的音訊格式
        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus'}
        
        # 檢查 GPU 可用性
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_info = f"✅ GPU 可用: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)"
        else:
            self.gpu_info = "❌ GPU 不可用，將使用 CPU，哭了，會很久"
        
        # 設定樣式
        style = ttk.Style()
        style.theme_use('clam')
        
        self.setup_ui()
        self.check_ffmpeg()
        
        # 開始處理 GUI 隊列
        self.process_gui_queue()
    
    def setup_ui(self):
        """建立使用者介面"""
        # 創建主框架和滾動條
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # 創建 Canvas 和滾動條
        self.canvas = Canvas(main_container, bg='white')
        scrollbar = ttk.Scrollbar(main_container, orient=VERTICAL, command=self.canvas.yview)
        
        # 可滾動的框架
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # 配置 Canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # 綁定鼠標滾輪事件
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        # 佈局 Canvas 和滾動條
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # 配置主滾動框架
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        # ========== 1. 資料夾選擇區 ==========
        folder_frame = ttk.LabelFrame(self.scrollable_frame, text="📁 資料夾選擇", padding="10")
        folder_frame.grid(row=0, column=0, sticky=(W, E), pady=5)
        folder_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_frame, text="音檔資料夾：").grid(row=0, column=0, sticky=W)
        ttk.Entry(folder_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, sticky=(W, E), padx=5)
        ttk.Button(folder_frame, text="瀏覽...", command=self.browse_folder_input).grid(row=0, column=2)
        
        self.folder_info = ttk.Label(folder_frame, text="尚未選擇資料夾", foreground="gray")
        self.folder_info.grid(row=1, column=0, columnspan=3, sticky=W, pady=(5,0))
        
        # 檔案列表
        self.file_list_frame = ttk.Frame(folder_frame)
        self.file_list_frame.grid(row=2, column=0, columnspan=3, sticky=(W, E), pady=(5,0))
        self.file_list_frame.columnconfigure(0, weight=1)
        
        # ========== 2. 硬體設定區 ==========
        hardware_frame = ttk.LabelFrame(self.scrollable_frame, text="⚙️ 硬體設定", padding="10")
        hardware_frame.grid(row=1, column=0, sticky=(W, E), pady=5)
        
        # GPU 資訊
        gpu_status_frame = ttk.Frame(hardware_frame)
        gpu_status_frame.grid(row=0, column=0, columnspan=2, sticky=W)
        
        gpu_icon = "✅" if self.gpu_available else "❌"
        gpu_text = f"{gpu_icon} {self.gpu_info}"
        ttk.Label(gpu_status_frame, text=gpu_text, 
                 foreground="green" if self.gpu_available else "red").grid(row=0, column=0, sticky=W)
        
        # GPU 開關
        gpu_switch_frame = ttk.Frame(hardware_frame)
        gpu_switch_frame.grid(row=1, column=0, columnspan=2, sticky=W, pady=(5,0))
        
        self.gpu_checkbox = ttk.Checkbutton(gpu_switch_frame, text="使用 GPU 加速（大幅提升速度）",
                                           variable=self.use_gpu, 
                                           state="normal" if self.gpu_available else "disabled")
        self.gpu_checkbox.grid(row=0, column=0, sticky=W)
        
        if not self.gpu_available:
            self.use_gpu.set(False)
            ttk.Label(gpu_switch_frame, text="（GPU 不可用）", 
                     foreground="gray", font=('', 8)).grid(row=0, column=1, padx=5)
        
        # 模型選擇
        ttk.Label(hardware_frame, text="選擇模型：").grid(row=2, column=0, sticky=W, pady=(10,0))
        models = [
            ("tiny - 最快速，準確度最低", "tiny"),
            ("base - 快速，準確度低", "base"),
            ("small - 較快，準確度中等", "small"),
            ("medium - 平衡", "medium"),
            ("large - 最慢，但準確度最高（推薦）", "large")
        ]
        
        for i, (text, value) in enumerate(models):
            ttk.Radiobutton(hardware_frame, text=text, variable=self.model_size, 
                           value=value).grid(row=3+i, column=0, sticky=W, padx=20)
        
        # 語言選擇
        ttk.Label(hardware_frame, text="音檔語言：").grid(row=2, column=1, sticky=W, padx=(30,0), pady=(10,0))
        languages = [
            ("中文（繁體/簡體）", "zh"),
            ("英文", "en"),
            ("日文", "ja"),
            ("韓文", "ko"),
            ("自動偵測", "auto")
        ]
        
        for i, (text, value) in enumerate(languages):
            ttk.Radiobutton(hardware_frame, text=text, variable=self.language, 
                           value=value).grid(row=3+i, column=1, sticky=W, padx=10)
        
        # ========== 3. 輸出設定區 ==========
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="💾 輸出設定", padding="10")
        output_frame.grid(row=2, column=0, sticky=(W, E), pady=5)
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="輸出資料夾：").grid(row=0, column=0, sticky=W)
        ttk.Entry(output_frame, textvariable=self.output_folder, width=50).grid(row=0, column=1, sticky=(W, E), padx=5)
        ttk.Button(output_frame, text="選擇...", command=self.browse_folder_output).grid(row=0, column=2)
        
        # 切割設定
        ttk.Label(output_frame, text="檔案切割設定：", font=('', 9, 'bold')).grid(row=1, column=0, sticky=W, pady=(10,5))
        
        size_frame = ttk.Frame(output_frame)
        size_frame.grid(row=2, column=0, columnspan=3, sticky=W)
        ttk.Label(size_frame, text="  當檔案超過").grid(row=0, column=0)
        size_spinbox = ttk.Spinbox(size_frame, from_=50, to=1000, increment=50, 
                                    textvariable=self.max_file_size, width=8)
        size_spinbox.grid(row=0, column=1, padx=5)
        ttk.Label(size_frame, text="MB 時，自動切割為每段").grid(row=0, column=2)
        chunk_spinbox = ttk.Spinbox(size_frame, from_=5, to=30, increment=5,
                                     textvariable=self.chunk_length, width=8)
        chunk_spinbox.grid(row=0, column=3, padx=5)
        ttk.Label(size_frame, text="分鐘").grid(row=0, column=4)
        
        ttk.Label(output_frame, text="  💡 提示：切割可避免記憶體不足，但會增加處理時間", 
                 foreground="gray", font=('', 8)).grid(row=3, column=0, columnspan=3, sticky=W, pady=(2,0))
        
        # 品質改善設定
        ttk.Label(output_frame, text="品質改善設定：", font=('', 9, 'bold')).grid(row=4, column=0, sticky=W, pady=(10,5))
        
        ttk.Checkbutton(output_frame, text="啟用靜音偵測VAD-自動跳過靜音片段，但通常不建議用，不知道為什麼會讓模型容易產生幻覺，這一定是幻覺，嚇不倒我的",
                       variable=self.enable_vad).grid(row=5, column=0, columnspan=3, sticky=W, padx=20)
        
        ttk.Checkbutton(output_frame, text="啟用重複偵測-自動移除重複迴圈內容（強烈建議）",
                       variable=self.repetition_detection).grid(row=6, column=0, columnspan=3, sticky=W, padx=20)
        
        ttk.Label(output_frame, text="  💡 提示：這兩個選項可大幅改善輸出品質，避免出現重複文字，但靜音那個有時候怪怪的，斟酌使用喔", 
                 foreground="green", font=('', 8)).grid(row=7, column=0, columnspan=3, sticky=W, pady=(2,0))
        
        # ========== 4. 進度顯示區 ==========
        progress_frame = ttk.LabelFrame(self.scrollable_frame, text="⏳ 處理進度", padding="10")
        progress_frame.grid(row=3, column=0, sticky=(W, E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        # 整體進度
        self.overall_progress_label = ttk.Label(progress_frame, text="整體進度：0/0")
        self.overall_progress_label.grid(row=0, column=0, sticky=W)
        
        self.overall_progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_progress_bar.grid(row=1, column=0, sticky=(W, E), pady=5)
        
        # 當前檔案進度
        self.current_file_label = ttk.Label(progress_frame, text="當前檔案：無")
        self.current_file_label.grid(row=2, column=0, sticky=W, pady=(10,0))
        
        self.current_progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.current_progress_bar.grid(row=3, column=0, sticky=(W, E), pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="等待開跑...", foreground="blue")
        self.status_label.grid(row=4, column=0, sticky=W)
        
        # ========== 5. 日誌顯示區 ==========
        log_frame = ttk.LabelFrame(self.scrollable_frame, text="📋 處理日誌", padding="10")
        log_frame.grid(row=4, column=0, sticky=(W, E), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = ScrolledText(log_frame, height=15, wrap=WORD, 
                                     state='disabled', bg='#f5f5f5')
        self.log_text.grid(row=0, column=0, sticky=(W, E, N, S), padx=5, pady=5)
        
        # ========== 6. 控制按鈕區 ==========
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=5, column=0, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="🚀 開始批次轉錄", 
                                    command=self.start_transcription, width=20)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ 停止", 
                                   command=self.stop_transcription, 
                                   state='disabled', width=20)
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="📂 開啟輸出資料夾", 
                  command=self.open_output_folder, width=20).grid(row=0, column=2, padx=5)
        
        # 添加記憶體監控按鈕（GPU 版本專用）
        if self.gpu_available:
            ttk.Button(button_frame, text="📊 記憶體狀態", 
                      command=self.show_memory_status, width=20).grid(row=0, column=3, padx=5)
        
        # 更新滾動區域
        self.root.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def process_gui_queue(self):
        """處理 GUI 隊列中的任務（線程安全）"""
        try:
            while True:
                # 非阻塞方式獲取隊列中的任務
                task = self.gui_queue.get_nowait()
                if task is None:
                    break
                
                # 執行任務
                task_type = task.get('type')
                if task_type == 'log':
                    self._log(task['message'])
                elif task_type == 'update_status':
                    self._update_status(task['message'], task.get('color', 'blue'))
                elif task_type == 'update_progress':
                    self._update_progress(task['current'], task['total'])
                elif task_type == 'update_current_file':
                    self._update_current_file(task['filename'])
                elif task_type == 'messagebox':
                    if task['msg_type'] == 'error':
                        messagebox.showerror(task['title'], task['message'])
                    elif task['msg_type'] == 'info':
                        messagebox.showinfo(task['title'], task['message'])
                    elif task['msg_type'] == 'warning':
                        messagebox.showwarning(task['title'], task['message'])
                    elif task['msg_type'] == 'askyesno':
                        result = messagebox.askyesno(task['title'], task['message'])
                        if 'callback' in task:
                            task['callback'](result)
                
        except queue.Empty:
            pass
        
        # 每100毫秒檢查一次隊列
        self.root.after(100, self.process_gui_queue)

    def queue_log(self, message):
        """將日誌訊息加入隊列"""
        self.gui_queue.put({'type': 'log', 'message': message})

    def _log(self, message):
        """寫入日誌（線程安全版本）"""
        self.log_text.config(state='normal')
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self.log_text.config(state='disabled')

    def log(self, message):
        """寫入日誌（兼容舊版本）"""
        self.queue_log(message)

    def queue_update_status(self, message, color="blue"):
        """將狀態更新加入隊列"""
        self.gui_queue.put({'type': 'update_status', 'message': message, 'color': color})

    def _update_status(self, message, color="blue"):
        """更新狀態（線程安全版本）"""
        self.status_label.config(text=message, foreground=color)

    def update_status(self, message, color="blue"):
        """更新狀態（兼容舊版本）"""
        self.queue_update_status(message, color)

    def queue_update_progress(self, current, total):
        """將進度更新加入隊列"""
        self.gui_queue.put({'type': 'update_progress', 'current': current, 'total': total})

    def _update_progress(self, current, total):
        """更新整體進度（線程安全版本）"""
        self.overall_progress_label.config(text=f"整體進度：{current}/{total}")
        if total > 0:
            progress = (current / total) * 100
            self.overall_progress_bar['value'] = progress

    def update_progress(self, current, total):
        """更新整體進度（兼容舊版本）"""
        self.queue_update_progress(current, total)

    def queue_update_current_file(self, filename):
        """將當前檔案更新加入隊列"""
        self.gui_queue.put({'type': 'update_current_file', 'filename': filename})

    def _update_current_file(self, filename):
        """更新當前處理的檔案（線程安全版本）"""
        self.current_file_label.config(text=f"當前檔案：{filename}")

    def update_current_file(self, filename):
        """更新當前處理的檔案（兼容舊版本）"""
        self.queue_update_current_file(filename)

    def queue_messagebox(self, msg_type, title, message, callback=None):
        """將訊息框加入隊列"""
        self.gui_queue.put({
            'type': 'messagebox', 
            'msg_type': msg_type, 
            'title': title, 
            'message': message,
            'callback': callback
        })

    def check_ffmpeg(self):
        """檢查 ffmpeg"""
        self.log("🔍 檢查 ffmpeg...")
        ffmpeg_path = self.find_ffmpeg()
        
        if ffmpeg_path:
            self.log(f"✅ 找到 ffmpeg：{ffmpeg_path}")
            os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
        else:
            msg = ("❌ 找不到 ffmpeg！\n\n"
                   "請下載 ffmpeg：\n"
                   "1. 前往 https://www.gyan.dev/ffmpeg/builds/\n"
                   "2. 下載 ffmpeg-release-essentials.zip\n"
                   "3. 解壓後將 bin/ffmpeg.exe 放到程式同一資料夾")
            self.log(msg)
            self.queue_messagebox('warning', "需要 ffmpeg", msg)
    
    def find_ffmpeg(self):
        """尋找 ffmpeg 執行檔"""
        if os.path.exists("ffmpeg.exe"):
            return os.path.abspath("ffmpeg.exe")
        
        try:
            result = subprocess.run(['where', 'ffmpeg'], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=3)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe")
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None

    def get_device(self):
        """取得計算設備（GPU 或 CPU）- 線程安全版本"""
        # 在主線程中讀取變數值
        use_gpu = self.use_gpu.get() if self.gpu_available else False
        
        if use_gpu and self.gpu_available:
            # 檢查 GPU 記憶體是否足夠
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            free_memory = gpu_memory - allocated
            
            # 如果可用記憶體少於 1GB，警告用戶
            if free_memory < 1 * 1024**3:  # 1 GB
                self.log(f"⚠️ GPU 記憶體緊張囉：{free_memory/1024**3:.1f} GB 可用")
                
                # 使用隊列顯示確認對話框
                result_queue = queue.Queue()
                
                def show_dialog():
                    result = messagebox.askyesno(
                        "記憶體警告", 
                        f"GPU 記憶體僅剩 {free_memory/1024**3:.1f} GB，可能不足。\n\n是否繼續使用 GPU？"
                    )
                    result_queue.put(result)
                
                # 在主線程中顯示對話框
                self.root.after(0, show_dialog)
                
                # 等待用戶回應
                try:
                    result = result_queue.get(timeout=30)  # 30秒超時
                    if result:
                        return "cuda"
                    else:
                        return "cpu"
                except queue.Empty:
                    self.log("⚠️ 等待用戶回應超時，使用 CPU")
                    return "cpu"
            return "cuda"
        else:
            return "cpu"
    
    def show_memory_status(self):
        """顯示記憶體狀態"""
        if self.gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            free_memory = gpu_memory - allocated - cached
            
            status_msg = (f"GPU 記憶體狀態：\n"
                         f"總記憶體：{gpu_memory/1024**3:.1f} GB\n"
                         f"已分配：{allocated/1024**3:.1f} GB\n"
                         f"已快取：{cached/1024**3:.1f} GB\n"
                         f"可用：{free_memory/1024**3:.1f} GB")
            
            self.log(f"📊 {status_msg}")
            self.queue_messagebox('info', "記憶體狀態", status_msg)
        else:
            self.queue_messagebox('info', "記憶體狀態", "GPU 不可用，幫哭")

    def browse_folder_input(self):
        """選擇輸入資料夾"""
        folder = filedialog.askdirectory(title="選擇包含音檔的資料夾")
        if folder:
            self.input_folder.set(folder)
            self.scan_audio_files(folder)
    
    def browse_folder_output(self):
        """選擇輸出資料夾"""
        folder = filedialog.askdirectory(title="選擇輸出資料夾")
        if folder:
            self.output_folder.set(folder)

    def scan_audio_files(self, folder):
        """掃描資料夾內的音訊檔"""
        self.audio_files = []
        
        try:
            for filename in os.listdir(folder):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.audio_extensions:
                    filepath = os.path.join(folder, filename)
                    self.audio_files.append(filepath)
            
            self.audio_files.sort()  # 按檔名排序
            
            if self.audio_files:
                total_size = sum(os.path.getsize(f) for f in self.audio_files) / (1024 * 1024)
                info = f"✅ 找到 {len(self.audio_files)} 個音檔  |  📦 總大小：{total_size:.2f} MB"
                self.folder_info.config(text=info, foreground="green")
                
                # 顯示檔案列表
                self.display_file_list()
                
                self.log(f"\n📁 掃描資料夾：{folder}")
                self.log(f"✅ 找到 {len(self.audio_files)} 個音檔")
                for i, f in enumerate(self.audio_files, 1):
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    self.log(f"  {i}. {os.path.basename(f)} ({size_mb:.2f} MB)")
            else:
                self.folder_info.config(text="❌ 資料夾中沒有找到音檔", foreground="red")
                self.log(f"❌ 在 {folder} 中沒有找到音檔")
                self.queue_messagebox('warning', "沒有音檔", "選擇的資料夾中沒有找到支援的音檔格式，幫哭！")
                
        except Exception as e:
            self.folder_info.config(text=f"錯誤：{str(e)}", foreground="red")
            self.log(f"❌ 掃描資料夾時發生錯誤：{e}")

    def display_file_list(self):
        """顯示檔案列表"""
        # 清除舊的列表
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        if not self.audio_files:
            return
        
        # 建立列表框
        list_label = ttk.Label(self.file_list_frame, text=f"將處理以下 {len(self.audio_files)} 個檔案：")
        list_label.grid(row=0, column=0, sticky=W, pady=(5,2))
        
        list_frame = ttk.Frame(self.file_list_frame)
        list_frame.grid(row=1, column=0, sticky=(W, E))
        
        listbox = Listbox(list_frame, height=min(5, len(self.audio_files)), width=70)
        scrollbar = ttk.Scrollbar(list_frame, orient=VERTICAL, command=listbox.yview)
        listbox.config(yscrollcommand=scrollbar.set)
        
        for f in self.audio_files:
            listbox.insert(END, os.path.basename(f))
        
        listbox.grid(row=0, column=0, sticky=(W, E))
        scrollbar.grid(row=0, column=1, sticky=(N, S))

    def start_transcription(self):
        """開始批次轉錄"""
        # 驗證輸入
        if not self.input_folder.get():
            self.queue_messagebox('error', "錯誤", "請先選擇音檔資料夾！")
            return
        
        if not self.audio_files:
            self.queue_messagebox('error', "錯誤", "沒有找到音檔！")
            return
        
        # 使用隊列顯示確認對話框
        device = "GPU" if self.use_gpu.get() and self.gpu_available else "CPU"
        msg = (f"準備處理 {len(self.audio_files)} 個音檔\n\n"
               f"使用設備：{device}\n"
               f"模型：{self.model_size.get()}\n\n"
               f"這可能需要很長時間，請確保電腦不會進入睡眠模式。\n\n確定要開始嗎？要確喔")
        
        def handle_confirmation(result):
            if result:
                # 更新 UI
                self.is_processing = True
                self.start_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
                self.overall_progress_bar['value'] = 0
                
                # 在新執行緒中處理
                thread = threading.Thread(target=self.process_batch_transcription, daemon=True)
                thread.start()
        
        self.queue_messagebox('askyesno', "確認", msg, handle_confirmation)

    def process_batch_transcription(self):
        """批次處理轉錄（在背景執行緒）"""
        success_count = 0
        fail_count = 0
        
        try:
            # 清理記憶體
            self.log(f"{'='*60}")
            self.log("🧹 清理記憶體...")
            self.clear_memory()
            
            # 決定使用的設備
            device = self.get_device()
            self.log(f"🖥️ 使用設備：{device}")
            
            # 載入模型（只載入一次）
            self.update_status("🤖 正在載入模型...", "blue")
            self.log(f"🤖 載入 {self.model_size.get()} 模型到 {device}...")
            
            # 根據設備選擇適當的載入選項
            if device == "cuda":
                self.model = whisper.load_model(self.model_size.get(), device="cuda")
                # 啟用 GPU 優化
                torch.backends.cudnn.benchmark = True
            else:
                self.model = whisper.load_model(self.model_size.get(), device="cpu")
            
            self.log("✅ 模型載入完成！")
            
            # 顯示模型參數數量（僅供參考）
            if hasattr(self.model, 'parameters'):
                param_count = sum(p.numel() for p in self.model.parameters())
                self.log(f"📊 模型參數：{param_count:,} 個")
            
            # 逐個處理檔案
            total_files = len(self.audio_files)
            
            for index, audio_file in enumerate(self.audio_files, 1):
                if not self.is_processing:
                    self.log("⚠️ 使用者已停止批次處理")
                    break
                
                self.log(f"\n{'='*60}")
                self.log(f"📄 處理檔案 {index}/{total_files}")
                self.log(f"{'='*60}")
                
                filename = os.path.basename(audio_file)
                self.update_current_file(filename)
                self.update_progress(index - 1, total_files)
                
                try:
                    # 轉錄單個檔案
                    self.process_single_file(audio_file, index, total_files, device)
                    success_count += 1
                    
                    # 每處理完一個檔案就清理一次記憶體
                    self.log("  🧹 清理記憶體...")
                    self.clear_memory()
                    
                except Exception as e:
                    fail_count += 1
                    self.log(f"❌ 處理失敗：{e}")
                    self.log(f"⚠️ 跳過此檔案，繼續處理下一個...")
                    # 失敗時也清理記憶體
                    self.clear_memory()
                
                self.update_progress(index, total_files)
            
            # 完成
            self.current_progress_bar.stop()
            self.update_status("✅ 批次轉錄完成！", "green")
            self.update_current_file("全部完成")
            
            self.log(f"\n{'='*60}")
            self.log("🎉 批次處理完成！")
            self.log(f"✅ 成功：{success_count} 個")
            if fail_count > 0:
                self.log(f"❌ 失敗：{fail_count} 個")
            self.log(f"🖥️ 使用設備：{device}")
            self.log(f"{'='*60}")
            
            # 最後再清理一次
            self.log("🧹 最終記憶體清理...")
            self.clear_memory()
            
            self.queue_messagebox('info', "完成", 
                                f"批次轉錄完成！\n\n"
                                f"成功：{success_count} 個\n"
                                f"失敗：{fail_count} 個\n\n"
                                f"使用設備：{device}\n"
                                f"結果已儲存到輸出資料夾。")
            
        except Exception as e:
            self.log(f"❌ 批次處理失敗：{e}")
            self.queue_messagebox('error', "錯誤", f"批次處理失敗：\n\n{str(e)}")
            
        finally:
            self.is_processing = False
            # 使用隊列更新 UI 狀態
            def update_ui():
                self.start_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
                self.current_progress_bar.stop()
            
            self.root.after(0, update_ui)
            
            # 釋放模型記憶體
            self.model = None
            self.clear_memory()

    # 以下方法保持不變，但會使用線程安全的日誌方法
    def process_single_file(self, audio_file, current_num, total_num, device):
        """處理單個音檔"""
        filename = os.path.basename(audio_file)
        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        
        self.log(f"📁 檔案：{filename}")
        self.log(f"📦 大小：{size_mb:.2f} MB")
        self.log(f"🖥️ 設備：{device}")
        
        # 檢查是否需要切割
        max_size = self.max_file_size.get()
        if size_mb > max_size:
            self.log(f"⚠️ 檔案超過 {max_size} MB，將自動切割處理...")
            self.process_large_file(audio_file, current_num, total_num, device)
        else:
            self.log("✅ 檔案大小適中，直接處理")
            self.process_normal_file(audio_file, current_num, total_num, device)

    def process_normal_file(self, audio_file, current_num, total_num, device):
        """處理一般大小的音檔"""
        filename = os.path.basename(audio_file)
        
        # 開始轉錄
        self.update_status(f"🎙️ 正在轉錄 ({current_num}/{total_num})...", "orange")
        self.current_progress_bar.start()
        
        lang = self.language.get()
        if lang == "auto":
            lang = None
        
        self.log("🎙️ 開始轉錄...")
        
        # 根據設備調整參數
        fp16 = (device == "cuda")  # GPU 使用 fp16 加速
        
        # 改善後的參數設定
        transcribe_options = {
            "language": lang,
            "task": "transcribe",
            "verbose": False,
            "fp16": fp16,  # 根據設備調整
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),  # 多溫度嘗試，避免幻覺
            "compression_ratio_threshold": 2.4,  # 壓縮率閾值
            "logprob_threshold": -1.0,  # 對數概率閾值
            "no_speech_threshold": 0.6,  # 提高靜音偵測閾值
            "condition_on_previous_text": False,  # 不依賴前文，減少重複
        }
        
        # 如果啟用 VAD
        if self.enable_vad.get():
            transcribe_options["vad_filter"] = True
            self.log("  ✓ 已啟用靜音偵測")
        
        # 記錄開始時間
        import time
        start_time = time.time()
        
        result = self.model.transcribe(audio_file, **transcribe_options)
        
        # 計算處理時間
        processing_time = time.time() - start_time
        self.log(f"  ⏱️ 轉錄用時：{processing_time:.1f} 秒")
        
        if not self.is_processing:
            return
        
        # 重複偵測與清理
        if self.repetition_detection.get():
            self.log("  🔍 檢查重複內容...")
            result = self.remove_repetitions(result)
        
        # 儲存結果
        self.current_progress_bar.stop()
        self.update_status(f"💾 正在儲存結果 ({current_num}/{total_num})...", "blue")
        self.save_results(audio_file, result)
        
        self.log(f"✅ 完成：{filename}")

    def process_large_file(self, audio_file, current_num, total_num, device):
        """處理大型音檔（自動切割）"""
        filename = os.path.basename(audio_file)
        chunk_length_ms = self.chunk_length.get() * 60 * 1000  # 轉換為毫秒
        
        try:
            # 建立暫存資料夾
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            
            # 載入音檔
            self.log("📥 載入音檔...")
            self.update_status(f"📥 載入大型音檔 ({current_num}/{total_num})...", "orange")
            
            audio = AudioSegment.from_file(audio_file)
            duration_minutes = len(audio) / (1000 * 60)
            
            self.log(f"⏱️ 音檔時長：{duration_minutes:.1f} 分鐘")
            
            # 切割音檔
            self.log(f"✂️ 切割為每段 {self.chunk_length.get()} 分鐘...")
            chunks = make_chunks(audio, chunk_length_ms)
            total_chunks = len(chunks)
            
            self.log(f"📊 共切割為 {total_chunks} 個片段")
            
            # 根據設備調整參數
            fp16 = (device == "cuda")  # GPU 使用 fp16 加速
            
            # 處理每個片段
            all_segments = []
            time_offset = 0
            
            for i, chunk in enumerate(chunks, 1):
                if not self.is_processing:
                    self.log("⚠️ 使用者已停止處理")
                    break
                
                self.log(f"  處理片段 {i}/{total_chunks}...")
                self.update_status(f"🎙️ 轉錄片段 {i}/{total_chunks} ({current_num}/{total_num})...", "orange")
                
                # 儲存暫存檔
                chunk_filename = os.path.join(self.temp_dir, f"chunk_{i}.wav")
                chunk.export(chunk_filename, format="wav")
                
                # 轉錄片段
                lang = self.language.get()
                if lang == "auto":
                    lang = None
                
                # 改善後的參數設定
                transcribe_options = {
                    "language": lang,
                    "task": "transcribe",
                    "verbose": False,
                    "fp16": fp16,
                    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),
                    "compression_ratio_threshold": 2.4,
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "condition_on_previous_text": False,
                }
                
                if self.enable_vad.get():
                    transcribe_options["vad_filter"] = True
                
                # 記錄開始時間
                import time
                start_time = time.time()
                
                result = self.model.transcribe(chunk_filename, **transcribe_options)
                
                # 計算處理時間
                processing_time = time.time() - start_time
                self.log(f"    ⏱️ 片段用時：{processing_time:.1f} 秒")
                
                # 調整時間戳
                for seg in result["segments"]:
                    seg["start"] += time_offset
                    seg["end"] += time_offset
                    all_segments.append(seg)
                
                # 更新時間偏移
                time_offset += len(chunk) / 1000.0
                
                # 刪除暫存檔
                try:
                    os.remove(chunk_filename)
                except:
                    pass
                
                # 清理記憶體
                self.clear_memory()
            
            if not self.is_processing:
                return
            
            # 合併所有文字
            full_text = " ".join([seg["text"].strip() for seg in all_segments])
            
            # 建立完整結果
            combined_result = {
                "text": full_text,
                "segments": all_segments,
                "language": result.get("language", "unknown")
            }
            
            # 重複偵測與清理
            if self.repetition_detection.get():
                self.log("  🔍 檢查重複內容...")
                combined_result = self.remove_repetitions(combined_result)
            
            # 儲存結果
            self.current_progress_bar.stop()
            self.update_status(f"💾 正在儲存結果 ({current_num}/{total_num})...", "blue")
            self.save_results(audio_file, combined_result)
            
            self.log(f"✅ 完成：{filename}（共 {total_chunks} 個片段）")
            
        except Exception as e:
            self.log(f"❌ 處理大型檔案時發生錯誤：{e}")
            raise
        
        finally:
            # 清理暫存資料夾
            self.cleanup_temp_files()

    def save_results(self, audio_file, result):
        """儲存轉錄結果"""
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = self.output_folder.get()
        
        # 完整文字稿
        txt_file = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        self.log(f"  💾 完整文字：{os.path.basename(txt_file)}")
        
        # 分段文字稿
        md_file = os.path.join(output_dir, f"{base_name}_segments.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# 音檔轉錄結果\n\n")
            f.write(f"**檔案**：{os.path.basename(audio_file)}\n")
            
            # 顯示偵測到的語言
            detected_lang = result.get('language', 'unknown')
            lang_names = {
                'zh': '中文', 'en': '英文', 'ja': '日文',
                'ko': '韓文', 'es': '西班牙文', 'fr': '法文',
                'de': '德文', 'it': '義大利文'
            }
            f.write(f"**語言**：{lang_names.get(detected_lang, detected_lang)}\n")
            
            size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            f.write(f"**大小**：{size_mb:.2f} MB\n")
            f.write(f"**片段**：{len(result['segments'])} 個\n")
            
            if result['segments']:
                total_time = str(timedelta(seconds=int(result['segments'][-1]['end'])))
                f.write(f"**時長**：{total_time}\n\n")
            else:
                f.write(f"**時長**：N/A\n\n")
            
            f.write("---\n\n")
            
            for seg in result["segments"]:
                start = str(timedelta(seconds=int(seg["start"])))
                end = str(timedelta(seconds=int(seg["end"])))
                text = seg["text"].strip()
                
                f.write(f"**[{start} → {end}]**\n\n{text}\n\n---\n\n")
        
        self.log(f"  💾 分段文字：{os.path.basename(md_file)}")
        
        # 顯示偵測到的語言
        detected_lang = result.get('language', 'unknown')
        lang_name = lang_names.get(detected_lang, detected_lang)
        self.log(f"  🌐 偵測語言：{lang_name}")
        self.log(f"  🎯 片段數：{len(result['segments'])} 個")

    def stop_transcription(self):
        """停止轉錄"""
        def handle_stop(result):
            if result:
                self.is_processing = False
                self.update_status("⏹️ 已停止", "red")
                self.log("\n⚠️ 使用者停止批次處理")
        
        self.queue_messagebox('askyesno', "確認", 
                            "確定要停止批次轉錄嗎？\n\n已處理的檔案會保留，未處理的會跳過。", 
                            handle_stop)

    def open_output_folder(self):
        """開啟輸出資料夾"""
        folder = self.output_folder.get()
        if os.path.exists(folder):
            os.startfile(folder)
        else:
            self.queue_messagebox('error', "錯誤", "輸出資料夾不存在！")

    def clear_memory(self):
        """清理記憶體"""
        # 執行垃圾回收
        gc.collect()
        
        # 如果有使用 CUDA，清理 GPU 快取
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 稍微暫停讓系統處理
        import time
        time.sleep(0.1)

    def cleanup_temp_files(self):
        """清理暫存檔案"""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    try:
                        os.remove(file_path)
                    except:
                        pass
                # 嘗試刪除暫存資料夾
                try:
                    os.rmdir(self.temp_dir)
                except:
                    pass
        except Exception as e:
            self.log(f"⚠️ 清理暫存檔案時發生錯誤：{e}")

    def remove_repetitions(self, result):
        """偵測並移除重複的內容"""
        segments = result["segments"]
        if not segments:
            return result
        
        # 偵測重複片段
        cleaned_segments = []
        seen_texts = set()
        repetition_count = 0
        max_repetition = 3  # 允許最多重複3次
        
        for seg in segments:
            text = seg["text"].strip()
            
            # 跳過空白或太短的片段
            if not text or len(text) < 5:
                continue
            
            # 檢查是否為重複
            if text in seen_texts:
                repetition_count += 1
                if repetition_count >= max_repetition:
                    self.log(f"  ⚠️ 偵測到重複內容，已截斷：「{text[:50]}...」")
                    break  # 停止處理後續內容
            else:
                repetition_count = 0
                seen_texts.add(text)
            
            cleaned_segments.append(seg)
        
        # 偵測長文字中的重複模式
        full_text = " ".join([seg["text"].strip() for seg in cleaned_segments])
        full_text = self.detect_text_loops(full_text)
        
        # 更新結果
        result["segments"] = cleaned_segments
        result["text"] = full_text
        
        removed = len(segments) - len(cleaned_segments)
        if removed > 0:
            self.log(f"  ✓ 已移除 {removed} 個重複片段")
        
        return result

    def detect_text_loops(self, text):
        """偵測文字中的重複循環模式"""
        words = text.split()
        
        if len(words) < 50:
            return text
        
        # 檢查後半部分是否有明顯的重複模式
        half_point = len(words) // 2
        first_half = " ".join(words[:half_point])
        second_half = " ".join(words[half_point:])
        
        # 檢查長重複序列（10個字以上）
        for window_size in range(50, 10, -5):
            if len(words) < window_size * 3:
                continue
            
            # 從後面往前檢查
            for i in range(len(words) - window_size * 2, max(0, len(words) - window_size * 10), -1):
                pattern = " ".join(words[i:i+window_size])
                rest_text = " ".join(words[i+window_size:])
                
                # 計算模式出現次數
                count = rest_text.count(pattern)
                if count >= 3:  # 如果重複3次以上
                    self.log(f"  ⚠️ 偵測到循環重複模式（長度：{window_size}字），已截斷")
                    return " ".join(words[:i+window_size])
        
        return text

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    root = Tk()
    app = WhisperGUI(root)

    root.mainloop()
