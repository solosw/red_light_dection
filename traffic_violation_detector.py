"""
闯红灯检测系统主脚本
整合YOLO检测、红绿灯分类、目标跟踪和违规判断
"""

import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from infer_classifier import TrafficLightPredictor
from tracker import SimpleTracker


class TrafficViolationDetector:
    """闯红灯检测系统"""

    def __init__(
        self,
        yolo_model_path="yolov8s.pt",
        classifier_model_path="models/traffic_light_classifier.pth",
        stop_line_y=None,
    ):
        """
        初始化检测系统
        Args:
            yolo_model_path: YOLO模型路径
            classifier_model_path: 红绿灯分类模型路径
            stop_line_y: 停止线Y坐标（None则自动设置）
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

        # 初始化跟踪器
        self.tracker = SimpleTracker(max_disappeared=30, min_iou=0.3)
        print("✓ 跟踪器初始化完成")

        # 检测区域配置
        self.stop_line_y = stop_line_y
        self.violation_cooldown = {}  # 违规冷却期：防止重复记录
        self.cooldown_frames = 60  # 60帧冷却期

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

    def set_stop_line(self, frame_height):
        """设置停止线（基于视频高度自动设置）"""
        if self.stop_line_y is None:
            # 在视频下方20%处设置停止线
            self.stop_line_y = int(frame_height * 0.8)
        print(f"✓ 停止线设置: Y = {self.stop_line_y}")

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

        # 检查每个车辆是否越线
        for obj_id, vehicle in self.tracker.objects.items():
            # 跳过非车辆目标
            if vehicle["class"] not in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                continue

            bbox = vehicle["bbox"]
            vehicle_bottom_y = bbox[3]  # 车辆底部Y坐标

            # 检查是否越过停止线（车辆底部越过停止线）
            crossed_line = vehicle_bottom_y > self.stop_line_y

            # 检查是否在冷却期内
            if obj_id in self.violation_cooldown:
                if self.violation_cooldown[obj_id] > 0:
                    self.violation_cooldown[obj_id] -= 1
                    continue

            # 如果越线且不在冷却期，则记录违规
            if crossed_line and self.violation_cooldown.get(obj_id, 0) == 0:
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
                self.violation_cooldown[obj_id] = self.cooldown_frames

        return violations_found

    def draw_ui(self, frame, vehicles, traffic_lights, violations):
        """绘制UI界面"""
        height, width = frame.shape[:2]

        # 绘制停止线
        cv2.line(
            frame, (0, self.stop_line_y), (width, self.stop_line_y), (0, 0, 255), 3
        )
        cv2.putText(
            frame,
            "STOP LINE",
            (10, self.stop_line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
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
        stats_text = [
            f"Frame: {self.stats['total_frames']}",
            f"Violations: {self.stats['total_violations']}",
            f"Vehicles: {self.stats['detected_vehicles']}",
            f"TL Detected: {self.stats['detected_traffic_lights']}",
        ]

        for i, text in enumerate(stats_text):
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

        # 设置停止线
        self.set_stop_line(height)

        # 创建输出视频写入器
        output_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            print(f"  输出视频: {output_video}")

        # 开始处理
        print(f"\n开始处理视频...")
        print("进度: ", end="", flush=True)

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每10%进度显示一次
            if frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                print(f"{progress:.0f}% ", end="", flush=True)

            self.stats["total_frames"] = frame_count

            # 检测红绿灯
            traffic_lights = self.detect_traffic_lights(frame)
            self.stats["detected_traffic_lights"] += len(traffic_lights)

            # 检测车辆
            vehicles = self.detect_vehicles(frame)
            self.stats["detected_vehicles"] += len(vehicles)

            # 转换检测结果为跟踪器格式
            detections = vehicles

            # 更新跟踪器
            tracks = self.tracker.update(detections)

            # 检查违规
            violations = self.check_violations(vehicles, traffic_lights)

            # 记录违规
            for violation in violations:
                self.violations.append(violation)
                self.stats["total_violations"] += 1
                print(
                    f"\n⚠️  闯红灯违规! ID: {violation['object_id']}, 车型: {violation['vehicle_class']}"
                )

            # 绘制跟踪轨迹
            frame = self.tracker.draw_tracks(frame)

            # 绘制UI
            frame = self.draw_ui(frame, vehicles, traffic_lights, violations)

            # 写入输出视频
            if output_writer:
                output_writer.write(frame)

            frame_count += 1

        # 释放资源
        cap.release()
        if output_writer:
            output_writer.release()

        elapsed_time = time.time() - start_time
        print("100%")
        print(f"\n✓ 视频处理完成!")
        print(f"  处理时间: {elapsed_time:.2f} 秒")
        print(f"  平均FPS: {frame_count / elapsed_time:.2f}")

        # 保存报告
        #self.save_report(input_video, output_video, elapsed_time)

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
