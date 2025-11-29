

class VideoDisplayWindow:
    """视频显示窗口类"""

    def __init__(self, title="闯红灯检测系统", width=1280, height=720):
        if not HAS_TKINTER:
            raise RuntimeError("tkinter 未安装，无法创建 GUI 窗口")

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg="#2C2C2C")

        # 创建视频显示区域
        self.video_frame = tk.Frame(self.root, bg="#1C1C1C")
        self.video_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.video_label = tk.Label(self.video_frame, bg="#1C1C1C")
        self.video_label.pack(expand=True)

        # 创建控制面板
        self.control_frame = tk.Frame(self.root, bg="#2C2C2C", height=100)
        self.control_frame.pack(fill="x", padx=10, pady=5)
        self.control_frame.pack_propagate(False)

        # 控制按钮
        self.pause_btn = tk.Button(
            self.control_frame,
            text="暂停",
            command=self.toggle_pause,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12),
            width=10,
        )
        self.pause_btn.pack(side="left", padx=5, pady=5)

        self.speed_btn = tk.Button(
            self.control_frame,
            text="速度: 1.0x",
            command=self.change_speed,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12),
            width=10,
        )
        self.speed_btn.pack(side="left", padx=5, pady=5)

        self.screenshot_btn = tk.Button(
            self.control_frame,
            text="截图",
            command=self.take_screenshot,
            bg="#FF9800",
            fg="white",
            font=("Arial", 12),
            width=10,
        )
        self.screenshot_btn.pack(side="left", padx=5, pady=5)

        # 状态信息
        self.status_frame = tk.Frame(self.root, bg="#2C2C2C", height=50)
        self.status_frame.pack(fill="x", padx=10, pady=5)
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_frame,
            text="就绪",
            bg="#2C2C2C",
            fg="white",
            font=("Arial", 12),
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=5, pady=5)

        # 状态变量
        self.is_paused = False
        self.speed = 1.0
        self.current_frame = None
        self.on_close = None

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_pause(self):
        """切换暂停/继续"""
        self.is_paused = not self.is_paused
        self.pause_btn.config(text="继续" if self.is_paused else "暂停")

    def change_speed(self):
        """调整播放速度"""
        speeds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        idx = speeds.index(self.speed) if self.speed in speeds else 1
        idx = (idx + 1) % len(speeds)
        self.speed = speeds[idx]
        self.speed_btn.config(text=f"速度: {self.speed}x")

    def take_screenshot(self):
        """截取当前帧"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{timestamp}.jpg", self.current_frame)
            self.update_status(f"截图已保存: screenshot_{timestamp}.jpg", "#4CAF50")

    def update_frame(self, frame):
        """更新显示帧"""
        self.current_frame = frame.copy()

        # 转换颜色空间 BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 调整大小
        h, w = rgb_frame.shape[:2]
        if h > 0 and w > 0:
            # 计算缩放比例
            display_w = min(self.root.winfo_width() - 40, w)
            display_h = int(h * (display_w / w))
            if display_h > self.root.winfo_height() - 200:
                display_h = self.root.winfo_height() - 200
                display_w = int(w * (display_h / h))

            # 缩放图像
            resized = cv2.resize(rgb_frame, (display_w, display_h))

            # 转换为 PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(resized))

            # 更新显示
            self.video_label.config(image=photo, width=display_w, height=display_h)
            self.video_label.image = photo  # 保持引用

    def update_status(self, text, color="#FFFFFF"):
        """更新状态信息"""
        self.status_label.config(text=text, fg=color)

    def on_closing(self):
        """窗口关闭事件"""
        if self.on_close:
            self.on_close()
        self.root.destroy()

    def update(self):
        """更新窗口"""
        self.root.update()

    def is_closed(self):
        """检查窗口是否已关闭"""
        return not self.root.winfo_exists()

    def get_speed(self):
        """获取当前播放速度"""
        return self.speed

    def is_paused(self):
        """检查是否暂停"""
        return self.is_paused

