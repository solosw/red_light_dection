"""
闯红灯检测系统主脚本 - 支持实时GUI显示
整合YOLO检测、红绿灯分类、目标跟踪和违规判断
"""

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from detection_zone import DetectionZone, ZoneManager, interactive_zone_selection
from infer_classifier import TrafficLightPredictor
from sort.SimpleSort import Sort

# GUI支持
try:
    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import Image, ImageTk

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False


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


class TrafficViolationDetector:
    """闯红灯检测系统"""

    def __init__(
        self,
        yolo_model_path="yolov8s.pt",
        classifier_model_path="models/traffic_light_classifier.pth",
        detection_zone=None,
        polygon_points=None,
        realtime_display=True,
        window_name="闯红灯检测系统 - 实时监控",
    ):
        """
        初始化检测系统
        Args:
            yolo_model_path: YOLO模型路径
            classifier_model_path: 红绿灯分类模型路径
            detection_zone: 检测区域 (x1, y1, x2, y2)，只有在此区域内的行人才会被判定为闯红灯
            polygon_points: 4个点坐标列表 [x1, y1, x2, y2, x3, y3, x4, y4]
            realtime_display: 是否开启实时显示
            window_name: 实时显示窗口名称
        """
        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✓ YOLO模型加载成功: {yolo_model_path}")

        # 加载红绿灯分类模型
        if os.path.exists(classifier_model_path):
            self.traffic_light_predictor = TrafficLightPredictor(classifier_model_path)
            print(f"✓ 红绿灯分类模型加载成功: {classifier_model_path}")
        else:
            print(f"⚠ 红绿灯分类模型不存在: {classifier_model_path}")
            self.traffic_light_predictor = None

        # 初始化 Sort 跟踪器
        # 参数: max_age=最大无更新帧数, min_hits=最小命中次数, iou_threshold=IoU阈值
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        print("✓ Sort 跟踪器初始化完成")

        # 初始化区域管理器
        self.zone_manager = ZoneManager()

        # 支持多种区域类型
        self.detection_zones: List[DetectionZone] = []

        # 优先使用多边形区域（4个点）
        if polygon_points is not None and len(polygon_points) == 8:
            from detection_zone import PolygonZone

            polygon_zone = PolygonZone("custom_polygon")
            # 解析8个坐标为4个点
            points = [
                (polygon_points[i], polygon_points[i + 1]) for i in range(0, 8, 2)
            ]
            polygon_zone.points = points
            self.detection_zones.append(polygon_zone)
            print(f"✓ 加载4点定义的多边形检测区域")
            print(f"  4个点坐标: {points}")
        # 兼容旧的检测区域格式 (x1, y1, x2, y2)
        elif detection_zone is not None:
            from detection_zone import RectZone

            rect_zone = RectZone("default")
            x1, y1, x2, y2 = detection_zone
            rect_zone.set_rect(x1, y1, x2, y2)
            self.detection_zones.append(rect_zone)
            print(f"✓ 加载检测区域: {detection_zone}")
        else:
            print("✓ 未设置检测区域，将启动交互式选择")

        # 统计信息
        self.stats = {
            "total_frames": 0,
            "total_violations": 0,
            "detected_vehicles": 0,
            "detected_traffic_lights": 0,
            "red_light_frames": 0,
            "green_light_frames": 0,
            "yellow_light_frames": 0,
        }

        # 违规记录
        self.violations = []

        # 实时显示配置
        self.realtime_display = realtime_display
        self.window_name = window_name
        self.paused = False
        self.speed_factor = 1.0  # 播放速度倍数 (1.0=正常速度, 2.0=2倍速, 0.5=0.5倍速)

        # 初始化显示窗口
        self.gui_window = None
        self.window_available = False
        if self.realtime_display:
            if HAS_TKINTER:
                try:
                    self.gui_window = VideoDisplayWindow(title=window_name)
                    self.gui_window.on_close = self.on_window_close
                    print(f"✓ GUI窗口已创建: {window_name}")
                    self.window_available = True
                except Exception as e:
                    print(f"⚠️ 无法创建GUI窗口: {e}")
                    self.window_available = False
            else:
                print("⚠️ tkinter 未安装，无法创建 GUI 窗口")
                self.window_available = False

        if not self.window_available:
            print("⚠️ 将自动切换到非实时模式 (仅处理数据，不显示视频)")

    def on_window_close(self):
        """窗口关闭回调"""
        self.window_available = False
        print("\n⚠️ GUI窗口已关闭，检测将继续在后台运行...")

    def set_detection_zone(self, frame_width, frame_height):
        """设置检测区域（基于视频尺寸自动设置）"""
        if not self.detection_zones:
            # 默认设置为画面中心区域（宽度40%-60%，高度50%-90%）
            from detection_zone import RectZone

            rect_zone = RectZone("default")
            x1 = int(frame_width * 0.3)
            y1 = int(frame_height * 0.4)
            x2 = int(frame_width * 0.7)
            y2 = int(frame_height * 0.9)
            rect_zone.set_rect(x1, y1, x2, y2)
            self.detection_zones.append(rect_zone)

            # 保留向后兼容
            self.detection_zone = (x1, y1, x2, y2)

        print(f"✓ 检测区域已设置，共 {len(self.detection_zones)} 个区域")
        for i, zone in enumerate(self.detection_zones):
            print(f"  区域 {i + 1}: {zone.name} ({zone.__class__.__name__})")

    def is_in_detection_zone(self, bbox):
        """
        判断目标是否在检测区域内（支持多个区域）
        Args:
            bbox: 目标边界框 (x1, y1, x2, y2)
        Returns:
            bool: 如果目标中心点在任意检测区域内返回True，否则返回False
        """
        if not self.detection_zones:
            return True  # 如果没有设置检测区域，默认所有目标都在区域内

        # 计算目标中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # 检查中心点是否在任意检测区域内
        for zone in self.detection_zones:
            if zone.is_point_inside(int(center_x), int(center_y)):
                return True
        return False

    def detect_traffic_lights(self, frame):
        """检测红绿灯并进行分类"""
        traffic_lights = []

        # YOLO检测所有目标
        results = self.yolo_model(frame, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = self.yolo_model.names[class_id]

                if class_name == "traffic light":
                    # 获取红绿灯bbox
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox

                    # 使用分类模型判断颜色
                    color = "unknown"
                    confidence = 0.0

                    if self.traffic_light_predictor:
                        # 裁剪红绿灯区域
                        roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

                        if roi.size > 0:
                            try:
                                pred = self.traffic_light_predictor.predict_from_array(
                                    roi
                                )
                                color = pred["class"]
                                confidence = pred["confidence"]
                            except:
                                color = "unknown"
                                confidence = 0.0

                    traffic_lights.append(
                        {"bbox": bbox, "color": color, "confidence": confidence}
                    )

        return traffic_lights

    def detect_vehicles(self, frame):
        """检测车辆和行人"""
        vehicles = []

        results = self.yolo_model(frame, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = self.yolo_model.names[class_id]

                # 只检测车辆和行人
                if class_name in [
                    "car",
                    "truck",
                    "bus",
                    "motorcycle",
                    "person",
                    "bicycle",
                ]:
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf.cpu().numpy()[0])

                    vehicles.append(
                        {"bbox": bbox, "class": class_name, "confidence": confidence}
                    )

        return vehicles

    def select_detection_zone_interactive(self, video_path):
        """交互式选择检测区域

        Args:
            video_path: 视频文件路径
        Returns:
            DetectionZone: 选中的区域对象或 None
        """
        print("\n" + "=" * 80)
        print("🎯 交互式检测区域选择")
        print("=" * 80)

        # 使用新的区域选择系统
        zone = interactive_zone_selection(video_path, zone_type="rect")

        if zone:
            print(f"\n✓ 区域选择成功！")
            return zone
        else:
            print(f"\n⚠️ 区域选择失败或被取消")
            print("将使用默认区域")
            return None

    def check_violations(self, vehicles, traffic_lights):
        """检查闯红灯违规"""
        violations_found = []

        # 获取当前交通灯状态
        current_light_color = "unknown"
        max_confidence = 0.0

        for tl in traffic_lights:
            if tl["confidence"] > max_confidence:
                current_light_color = tl["color"]
                max_confidence = tl["confidence"]

        # 更新统计信息
        if current_light_color == "red":
            self.stats["red_light_frames"] += 1
        elif current_light_color == "green":
            self.stats["green_light_frames"] += 1
        elif current_light_color == "yellow":
            self.stats["yellow_light_frames"] += 1

        # 只有在红灯时才检查违规
        if current_light_color != "red":
            return violations_found

        # 检查每个行人是否在红灯下移动
        for obj_id, vehicle in self.tracker.objects.items():
            # 只检查行人（人行红灯违规）
            if vehicle["class"] != "person":
                continue

            # 只要是行人在红灯下就算违规（不需要越过停止线）
            # 检查是否已经记录为违规（第一次检测到红灯时记录）
            already_violated = False
            for v in self.violations:
                if v["object_id"] == obj_id:
                    already_violated = True
                    break

            # 如果是红灯且还没有记录为违规，则记录
            if current_light_color == "red" and not already_violated:
                # 检查是否在移动（需要至少2帧的轨迹数据）
                trajectory = vehicle.get("trajectory", [])
                is_moving = False

                if len(trajectory) >= 2:
                    # 计算最近两帧的中心点移动距离
                    prev_bbox = trajectory[-2]
                    curr_bbox = trajectory[-1]

                    # 计算中心点
                    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
                    curr_center_x = (curr_bbox[0] + curr_bbox[2]) / 2
                    curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2

                    # 计算欧几里得距离
                    movement_distance = (
                        (curr_center_x - prev_center_x) ** 2
                        + (curr_center_y - prev_center_y) ** 2
                    ) ** 0.5

                    # 如果移动距离超过阈值（5像素），认为是移动
                    if movement_distance > 5.0:
                        is_moving = True

                # 只有在移动时才记录违规
                if not is_moving:
                    continue

                # 获取行人bbox
                bbox = vehicle["bbox"]

                # 检查行人是否在检测区域内，只有在区域内才算违规
                if not self.is_in_detection_zone(bbox):
                    continue

                # 获取红绿灯位置（用于绘制违规指示）
                traffic_light_bbox = None
                if traffic_lights:
                    # 选择置信度最高的红绿灯
                    traffic_light_bbox = max(
                        traffic_lights, key=lambda x: x["confidence"]
                    )["bbox"]

                violation = {
                    "object_id": obj_id,
                    "vehicle_class": vehicle["class"],
                    "timestamp": time.time(),
                    "frame": self.stats["total_frames"],
                    "vehicle_bbox": bbox.copy(),
                    "traffic_light_bbox": traffic_light_bbox,
                    "light_color": current_light_color,
                }

                violations_found.append(violation)

        return violations_found

    def check_violations_with_sort(self, tracks_dict, traffic_lights):
        """检查闯红灯违规（适配 Sort 跟踪器）"""
        violations_found = []

        # 获取当前交通灯状态
        current_light_color = "unknown"
        max_confidence = 0.0

        for tl in traffic_lights:
            if tl["confidence"] > max_confidence:
                current_light_color = tl["color"]
                max_confidence = tl["confidence"]

        # 更新统计信息
        if current_light_color == "red":
            self.stats["red_light_frames"] += 1
        elif current_light_color == "green":
            self.stats["green_light_frames"] += 1
        elif current_light_color == "yellow":
            self.stats["yellow_light_frames"] += 1

        # 只有在红灯时才检查违规
        if current_light_color != "red":
            return violations_found

        # 检查每个行人是否在红灯下移动
        for obj_id, vehicle in tracks_dict.items():
            # 只检查行人（人行红灯违规）
            if vehicle["class"] != "person":
                continue

            # 检查是否已经记录为违规
            already_violated = False
            for v in self.violations:
                if v["object_id"] == obj_id:
                    already_violated = True
                    break

            # 如果是红灯且还没有记录为违规，则记录
            if current_light_color == "red" and not already_violated:
                # 简化处理：只要检测到行人在红灯下就记录违规
                # 检查行人是否在检测区域内
                bbox = vehicle["bbox"]
                if not self.is_in_detection_zone(bbox):
                    continue

                # 获取红绿灯位置（用于绘制违规指示）
                traffic_light_bbox = None
                if traffic_lights:
                    traffic_light_bbox = max(
                        traffic_lights, key=lambda x: x["confidence"]
                    )["bbox"]

                violation = {
                    "object_id": obj_id,
                    "vehicle_class": vehicle["class"],
                    "timestamp": time.time(),
                    "frame": self.stats["total_frames"],
                    "vehicle_bbox": bbox.copy(),
                    "traffic_light_bbox": traffic_light_bbox,
                    "light_color": current_light_color,
                }

                violations_found.append(violation)

        return violations_found

    def draw_tracks_with_sort(
        self, frame, tracked_objects, violation_ids, vehicles=None
    ):
        """绘制跟踪结果（适配 Sort 跟踪器）

        Args:
            frame: 输入帧
            tracked_objects: Sort 跟踪器输出的数组 [[x1, y1, x2, y2, track_id], ...]
            violation_ids: 违规对象ID列表
            vehicles: 车辆检测结果列表（用于获取类别信息）
        """
        if len(tracked_objects) == 0:
            return frame

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            # 绘制bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 根据是否是违规对象选择颜色
            if track_id in violation_ids:
                color = (0, 0, 255)  # 红色 - 违规对象
                thickness = 3
            else:
                color = (0, 255, 0)  # 绿色 - 正常跟踪对象
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 绘制ID和类别标签
            # 先尝试从 vehicles 中获取类别信息
            class_name = "unknown"
            if vehicles:
                for vehicle in vehicles:
                    bbox = vehicle["bbox"]
                    if (
                        abs(bbox[0] - x1) < 10
                        and abs(bbox[1] - y1) < 10
                        and abs(bbox[2] - x2) < 10
                        and abs(bbox[3] - y2) < 10
                    ):
                        class_name = vehicle["class"]
                        break

            label = f"ID:{track_id}"
            if track_id in violation_ids:
                label += " VIOLATION!"

            # 绘制文本
            text_color = (0, 0, 255) if track_id in violation_ids else (0, 255, 0)
            text_y = y1 - 10 if (y1 - 10) > 0 else y1 + 20
            cv2.putText(
                frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
            )

            # 为违规对象添加额外标识
            if track_id in violation_ids:
                warning_text = "!!!"
                warning_y = y1 - 40 if y1 - 40 > 10 else y1 - 10
                cv2.putText(
                    frame,
                    warning_text,
                    (x2 - 30, warning_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

        return frame

    def draw_ui(self, frame, vehicles, traffic_lights, violations):
        """绘制UI界面"""
        height, width = frame.shape[:2]

        # 绘制检测区域（支持多种形状）
        if self.detection_zones:
            for i, zone in enumerate(self.detection_zones):
                # 绘制区域
                frame = zone.draw(
                    frame, color=(0, 255, 0), thickness=2, fill_alpha=0.15
                )

                # 添加区域编号
                if zone.points:
                    x, y = zone.points[0]
                    cv2.putText(
                        frame,
                        f"Zone {i + 1}: {zone.name}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

        # 绘制交通灯状态
        if traffic_lights:
            # 选择置信度最高的红绿灯
            best_tl = max(traffic_lights, key=lambda x: x["confidence"])
            tl_bbox = best_tl["bbox"]
            tl_color = best_tl["color"]

            # 绘制红绿灯框
            x1, y1, x2, y2 = map(int, tl_bbox)
            color_map = {
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
            }
            tl_display_color = color_map.get(tl_color, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), tl_display_color, 2)
            cv2.putText(
                frame,
                f"TL: {tl_color.upper()} ({best_tl['confidence']:.0f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                tl_display_color,
                2,
            )

        # 绘制违规警告
        if violations:
            cv2.putText(
                frame,
                "!!! VIOLATION DETECTED !!!",
                (width // 4, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Violation Count: {len(self.violations) + len(violations)}",
                (width // 4, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # 绘制统计信息
        self.draw_stats(frame)

        return frame

    def draw_stats(self, frame):
        """绘制统计信息"""
        # 添加实时显示状态信息
        display_text = [
            f"Frame: {self.stats['total_frames']}",
            f"Violations: {self.stats['total_violations']}",
            f"Vehicles: {self.stats['detected_vehicles']}",
            f"TL Detected: {self.stats['detected_traffic_lights']}",
        ]

        if self.window_available:
            status = "PAUSED" if self.gui_window.is_paused else "RUNNING"
            speed_text = f"{self.speed_factor:.1f}x"
            display_text.insert(0, f"Status: {status} | Speed: {speed_text}")

        for i, text in enumerate(display_text):
            cv2.putText(
                frame,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    def process_video(self, input_video, output_video=None):
        """处理视频并进行闯红灯检测"""
        print("\n" + "=" * 80)
        print("开始闯红灯检测")
        print("=" * 80)

        # 打开视频
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {input_video}")
            return

        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {total_frames}")

        # 检查并设置检测区域
        if not self.detection_zones:
            print("\n检测区域未设置，将启动交互式选择...")
            selected_zone = self.select_detection_zone_interactive(input_video)
            if selected_zone:
                self.detection_zones.append(selected_zone)
                print(f"\n✓ 使用交互式选择的检测区域")
                print(f"  类型: {selected_zone.__class__.__name__}")
                print(f"  名称: {selected_zone.name}")
            else:
                print("\n⚠️ 区域选择被取消或失败，使用默认区域")
                self.set_detection_zone(width, height)
        else:
            # 已设置检测区域
            self.set_detection_zone(width, height)

        # 创建输出视频写入器
        output_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            print(f"  输出视频: {output_video}")

        # 实时显示提示
        if self.window_available:
            print(f"\n" + "=" * 80)
            print("实时监控模式已开启!")
            print("=" * 80)
            print("GUI控制:")
            print("  点击[暂停/继续]按钮: 暂停/继续播放")
            print("  点击[速度]按钮: 调整播放速度")
            print("  点击[截图]按钮: 保存当前帧")
            print("  关闭窗口: 退出程序")
            print("=" * 80)
            self.gui_window.update_status("正在初始化...", "#2196F3")
        elif self.realtime_display:
            print(f"\n" + "=" * 80)
            print("⚠️ 实时监控模式提示:")
            print("=" * 80)
            print("检测将在后台运行，生成报告和输出视频")
            print("=" * 80)

        # 开始处理
        print(f"\n开始处理视频...")
        if not self.window_available and not self.realtime_display:
            print("进度: ", end="", flush=True)

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # GUI模式：检查窗口状态
                if self.window_available:
                    self.gui_window.update()
                    if self.gui_window.is_closed():
                        print("\n⚠️ 用户关闭了窗口")
                        break

                    # 如果暂停，只更新GUI状态，不处理新帧
                    if self.gui_window.is_paused:
                        # 更新状态显示
                        status = f"帧: {frame_count} | 违规: {self.stats['total_violations']}"
                        self.gui_window.update_status(status, "#FFA500")
                        time.sleep(0.03)
                        continue

                    # 更新播放速度
                    self.speed_factor = self.gui_window.get_speed()

                self.stats["total_frames"] = frame_count

                # 非实时模式显示进度
                if not self.window_available and not self.realtime_display:
                    if frame_count % (total_frames // 10) == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"{progress:.0f}% ", end="", flush=True)

                # 检测红绿灯
                traffic_lights = self.detect_traffic_lights(frame)
                self.stats["detected_traffic_lights"] += len(traffic_lights)

                # 检测车辆
                vehicles = self.detect_vehicles(frame)
                self.stats["detected_vehicles"] += len(vehicles)

                # 转换检测结果为 Sort 格式 [x1, y1, x2, y2, score]
                if len(vehicles) > 0:
                    dets = np.array(
                        [
                            [
                                v["bbox"][0],
                                v["bbox"][1],
                                v["bbox"][2],
                                v["bbox"][3],
                                v["confidence"],
                            ]
                            for v in vehicles
                        ]
                    )
                else:
                    dets = np.empty((0, 5))

                # 更新 Sort 跟踪器
                tracked_objects = self.tracker.update(dets)
                # tracked_objects: [[x1, y1, x2, y2, track_id], ...]

                # 构建跟踪结果的字典格式，便于后续处理
                tracks_dict = {}
                if len(tracked_objects) > 0:
                    for track in tracked_objects:
                        x1, y1, x2, y2, track_id = track
                        track_id = int(track_id)

                        # 找到对应的车辆信息
                        vehicle_info = None
                        for v in vehicles:
                            bbox = v["bbox"]
                            # 检查是否匹配（简单IoU匹配）
                            if (
                                abs(bbox[0] - x1) < 5
                                and abs(bbox[1] - y1) < 5
                                and abs(bbox[2] - x2) < 5
                                and abs(bbox[3] - y2) < 5
                            ):
                                vehicle_info = v
                                break

                        if vehicle_info:
                            tracks_dict[track_id] = {
                                "bbox": [x1, y1, x2, y2],
                                "class": vehicle_info["class"],
                                "confidence": vehicle_info["confidence"],
                                "trajectory": [],  # 简化处理
                                "violation_frames": [],
                                "violation_flag": False,
                                "entry_time": None,
                                "exit_time": None,
                                "disappeared": 0,
                            }

                # 检查违规
                violations = self.check_violations_with_sort(
                    tracks_dict, traffic_lights
                )

                # 记录违规
                for violation in violations:
                    self.violations.append(violation)
                    self.stats["total_violations"] += 1
                    print(
                        f"\n⚠️  闯红灯违规! ID: {violation['object_id']}, 车型: {violation['vehicle_class']}"
                    )

                # 绘制跟踪结果（显示所有跟踪对象）
                # 获取所有曾经违规的对象ID
                all_violation_ids = [v["object_id"] for v in self.violations]
                # 也包括当前帧新检测到的违规
                current_violation_ids = [v["object_id"] for v in violations]
                all_violation_ids.extend(current_violation_ids)

                # 绘制所有跟踪对象（基于 Sort 输出）
                frame = self.draw_tracks_with_sort(
                    frame, tracked_objects, all_violation_ids, vehicles
                )

                # 绘制UI
                frame = self.draw_ui(frame, vehicles, traffic_lights, violations)

                # GUI实时显示
                if self.window_available:
                    self.gui_window.update_frame(frame)
                    status = f"帧: {frame_count} | 车辆: {len(vehicles)} | 违规: {self.stats['total_violations']}"
                    self.gui_window.update_status(status, "#4CAF50")
                else:
                    # 非实时模式或后台模式
                    # 写入输出视频
                    if output_writer:
                        output_writer.write(frame)

                # 控制播放速度（GUI模式）
                if self.window_available and self.speed_factor > 1.0:
                    # 加速播放时跳过一些帧
                    skip_frames = int(self.speed_factor) - 1
                    for _ in range(skip_frames):
                        cap.grab()

                frame_count += 1

        except KeyboardInterrupt:
            print("\n⚠️ 检测到键盘中断")

        finally:
            # 释放资源
            cap.release()
            if output_writer:
                output_writer.release()

            # 关闭GUI窗口
            if self.window_available and self.gui_window:
                self.gui_window.root.destroy()

            elapsed_time = time.time() - start_time

            if not self.window_available and not self.realtime_display:
                print("100%")

            print(f"\n✓ 视频处理完成!")
            print(f"  处理时间: {elapsed_time:.2f} 秒")
            print(f"  平均FPS: {frame_count / elapsed_time:.2f}")

            # 保存报告
            self.save_report(input_video, output_video, elapsed_time)

    def save_report(self, input_video, output_video, elapsed_time):
        """保存检测报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "input_video": input_video,
            "output_video": output_video,
            "statistics": self.stats,
            "violations": self.violations,
            "processing_time": elapsed_time,
        }

        report_file = "traffic_violation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ 报告已保存: {report_file}")

        # 打印总结
        print("\n" + "=" * 80)
        print("检测结果总结")
        print("=" * 80)
        print(f"总处理帧数: {self.stats['total_frames']}")
        print(f"检测车辆次数: {self.stats['detected_vehicles']}")
        print(f"检测红绿灯次数: {self.stats['detected_traffic_lights']}")
        print(f"红灯帧数: {self.stats['red_light_frames']}")
        print(f"绿灯帧数: {self.stats['green_light_frames']}")
        print(f"黄灯帧数: {self.stats['yellow_light_frames']}")
        print(f"闯红灯违规次数: {self.stats['total_violations']}")
        print(f"违规详情:")
        for i, violation in enumerate(self.violations, 1):
            print(
                f"  {i}. ID: {violation['object_id']}, 车型: {violation['vehicle_class']}, "
                f"时间: {datetime.fromtimestamp(violation['timestamp']):%H:%M:%S}"
            )
        print("=" * 80)


def main():
    """主函数"""
    # 输入输出视频路径
    input_video = "input.mp4"
    output_video = "output_traffic_violation.mp4"

    # 创建检测器
    detector = TrafficViolationDetector(
        yolo_model_path="yolov8s.pt",
        classifier_model_path="models/traffic_light_classifier.pth",
    )

    # 处理视频
    detector.process_video(input_video, output_video)


if __name__ == "__main__":
    main()
