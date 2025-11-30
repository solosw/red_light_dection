#!/usr/bin/env python
"""
检测区域管理模块
支持多种形状：矩形、多边形、圆形、椭圆、自由绘制
"""

import json
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


class DetectionZone:
    """检测区域基类"""

    def __init__(self, name: str = "zone"):
        self.name = name
        self.points: List[Tuple[int, int]] = []

    def add_point(self, x: int, y: int):
        """添加点"""
        self.points.append((x, y))

    def clear(self):
        """清空点"""
        self.points = []

    def is_point_inside(self, x: int, y: int) -> bool:
        """检查点是否在区域内（子类实现）"""
        raise NotImplementedError

    def draw(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制区域（子类实现）"""
        raise NotImplementedError

    def save_to_file(self, filepath: str):
        """保存区域到文件"""
        data = {
            "name": self.name,
            "type": self.__class__.__name__,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str):
        """从文件加载区域"""
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            data = json.load(f)

        zone_type = data.get("type", "RectZone")
        name = data.get("name", "zone")
        points = data.get("points", [])

        if zone_type == "RectZone":
            zone = RectZone(name)
        elif zone_type == "PolygonZone":
            zone = PolygonZone(name)
        elif zone_type == "CircleZone":
            zone = CircleZone(name)
        elif zone_type == "EllipseZone":
            zone = EllipseZone(name)
        elif zone_type == "FreeDrawZone":
            zone = FreeDrawZone(name)
        else:
            zone = RectZone(name)

        zone.points = points
        return zone


class RectZone(DetectionZone):
    """矩形区域"""

    def __init__(self, name: str = "rect"):
        super().__init__(name)
        self.p1: Optional[Tuple[int, int]] = None
        self.p2: Optional[Tuple[int, int]] = None

    def set_rect(self, x1: int, y1: int, x2: int, y2: int):
        """设置矩形"""
        self.p1 = (min(x1, x2), min(y1, y2))
        self.p2 = (max(x1, x2), max(y1, y2))

    def is_point_inside(self, x: int, y: int) -> bool:
        """检查点是否在矩形内"""
        if not self.p1 or not self.p2:
            return False
        x1, y1 = self.p1
        x2, y2 = self.p2
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制矩形"""
        if not self.p1 or not self.p2:
            return frame

        x1, y1 = self.p1
        x2, y2 = self.p2

        # 填充
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # 边框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 文字
        cv2.putText(
            frame,
            f"Rect Zone: {self.name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        return frame


class PolygonZone(DetectionZone):
    """多边形区域"""

    def __init__(self, name: str = "polygon"):
        super().__init__(name)

    def close_polygon(self):
        """闭合多边形"""
        if len(self.points) > 2:
            if self.points[0] != self.points[-1]:
                self.points.append(self.points[0])

    def is_point_inside(self, x: int, y: int) -> bool:
        """检查点是否在多边形内（射线法）"""
        if len(self.points) < 3:
            return False

        # 闭合多边形（不影响原数据）
        pts = self.points[:]
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        # 射线法
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]

            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
            ):
                inside = not inside
            j = i

        return inside

    def draw(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制多边形"""
        if len(self.points) < 2:
            return frame

        # 填充
        overlay = frame.copy()
        pts = np.array(self.points, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # 边框
        if len(self.points) > 1:
            cv2.polylines(frame, [pts], False, color, thickness)

        # 绘制顶点
        for i, (x, y) in enumerate(self.points):
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.putText(
                frame,
                str(i + 1),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # 文字
        if self.points:
            x, y = self.points[0]
            cv2.putText(
                frame,
                f"Polygon Zone: {self.name} ({len(self.points)} points)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return frame


class CircleZone(DetectionZone):
    """圆形区域"""

    def __init__(self, name: str = "circle"):
        super().__init__(name)
        self.center: Optional[Tuple[int, int]] = None
        self.radius: float = 0

    def set_circle(self, x: int, y: int, radius: int):
        """设置圆形"""
        self.center = (x, y)
        self.radius = radius

    def is_point_inside(self, x: int, y: int) -> bool:
        """检查点是否在圆形内"""
        if not self.center:
            return False
        cx, cy = self.center
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance <= self.radius

    def draw(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制圆形"""
        if not self.center:
            return frame

        cx, cy = self.center

        # 填充
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), int(self.radius), color, -1)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # 边框
        cv2.circle(frame, (cx, cy), int(self.radius), color, thickness)

        # 圆心
        cv2.circle(frame, (cx, cy), 4, color, -1)

        # 文字
        cv2.putText(
            frame,
            f"Circle Zone: {self.name} (r={int(self.radius)})",
            (cx, cy - int(self.radius) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        return frame


class EllipseZone(DetectionZone):
    """椭圆区域"""

    def __init__(self, name: str = "ellipse"):
        super().__init__(name)
        self.center: Optional[Tuple[int, int]] = None
        self.axes: Tuple[int, int] = (0, 0)  # (major_axis, minor_axis)
        self.angle: float = 0

    def set_ellipse(self, x: int, y: int, width: int, height: int, angle: float = 0):
        """设置椭圆"""
        self.center = (x, y)
        self.axes = (width // 2, height // 2)
        self.angle = angle

    def is_point_inside(self, x: int, y: int) -> bool:
        """检查点是否在椭圆内"""
        if not self.center or self.axes[0] <= 0 or self.axes[1] <= 0:
            return False

        cx, cy = self.center
        a, b = self.axes

        # 旋转坐标
        cos_angle = np.cos(np.radians(self.angle))
        sin_angle = np.sin(np.radians(self.angle))

        dx = x - cx
        dy = y - cy

        # 旋转到椭圆坐标系
        x_rot = dx * cos_angle + dy * sin_angle
        y_rot = -dx * sin_angle + dy * cos_angle

        # 椭圆方程: (x/a)^2 + (y/b)^2 <= 1
        return (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1

    def draw(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制椭圆"""
        if not self.center or self.axes[0] <= 0 or self.axes[1] <= 0:
            return frame

        cx, cy = self.center

        # 填充
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, cy), self.axes, self.angle, 0, 360, color, -1)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # 边框
        cv2.ellipse(frame, (cx, cy), self.axes, self.angle, 0, 360, color, thickness)

        # 文字
        cv2.putText(
            frame,
            f"Ellipse Zone: {self.name}",
            (cx, cy - self.axes[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        return frame


class FreeDrawZone(DetectionZone):
    """自由绘制区域"""

    def __init__(self, name: str = "free"):
        super().__init__(name)

    def is_point_inside(self, x: int, y: int) -> bool:
        """检查点是否在自由绘制区域内"""
        if len(self.points) < 3:
            return False

        # 转换为多边形判断
        poly = PolygonZone(name)
        poly.points = self.points[:]
        return poly.is_point_inside(x, y)

    def draw(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制自由区域"""
        if len(self.points) < 2:
            return frame

        # 转换为多边形绘制
        poly = PolygonZone(self.name)
        poly.points = self.points[:]
        return poly.draw(frame, color, thickness, fill_alpha)


class ZoneManager:
    """区域管理器"""

    def __init__(self):
        self.zones: List[DetectionZone] = []
        self.selected_zone: Optional[DetectionZone] = None
        self.drawing = False
        self.temp_points: List[Tuple[int, int]] = []

    def add_zone(self, zone: DetectionZone):
        """添加区域"""
        self.zones.append(zone)
        self.selected_zone = zone

    def remove_zone(self, index: int):
        """删除区域"""
        if 0 <= index < len(self.zones):
            self.zones.pop(index)
            if self.selected_zone in self.zones:
                self.selected_zone = None

    def clear_all(self):
        """清空所有区域"""
        self.zones = []
        self.selected_zone = None

    def is_point_in_any_zone(self, x: int, y: int) -> bool:
        """检查点是否在任意区域内"""
        for zone in self.zones:
            if zone.is_point_inside(x, y):
                return True
        return False

    def get_zone_at_point(self, x: int, y: int) -> Optional[DetectionZone]:
        """获取包含指定点的区域"""
        for zone in self.zones:
            if zone.is_point_inside(x, y):
                return zone
        return None

    def draw_all(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.3):
        """绘制所有区域"""
        for zone in self.zones:
            zone.draw(frame, color, thickness, fill_alpha)
        return frame

    def save_all(self, directory: str):
        """保存所有区域"""
        os.makedirs(directory, exist_ok=True)
        for i, zone in enumerate(self.zones):
            filepath = os.path.join(directory, f"zone_{i}_{zone.name}.json")
            zone.save_to_file(filepath)

    def load_all(self, directory: str) -> bool:
        """加载所有区域"""
        self.clear_all()

        if not os.path.exists(directory):
            return False

        for filename in os.listdir(directory):
            if filename.endswith(".json") and filename.startswith("zone_"):
                filepath = os.path.join(directory, filename)
                zone = DetectionZone.load_from_file(filepath)
                if zone:
                    self.zones.append(zone)

        return len(self.zones) > 0


def interactive_zone_selection(
    video_path: str, zone_type: str = "polygon"
) -> Optional[DetectionZone]:
    """交互式区域选择

    Args:
        video_path: 视频路径
        zone_type: 区域类型 ("rect", "polygon", "circle", "ellipse", "free")

    Returns:
        选中的区域或None
    """
    # 检测GUI支持
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        gui_supported = True
    except Exception:
        gui_supported = False

    if not gui_supported:
        print("\n⚠️  当前环境不支持GUI")
        print("请使用命令行指定区域:")
        print(f"  python realtime_demo.py --zone X1 Y1 X2 Y2")
        return None

    # 打开视频获取第一帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ 无法打开视频文件")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("✗ 无法读取视频帧")
        return None

    height, width = frame.shape[:2]
    display_frame = frame.copy()

    # 状态变量
    drawing = False
    points = []
    center = None
    temp_shape = None

    # 鼠标事件回调
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, points, center, temp_shape

        if zone_type == "rect":
            # 矩形绘制
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                points = [(x, y)]
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                display_frame[:] = frame.copy()
                temp_shape = ("rect", (points[0], (x, y)))
                cv2.rectangle(display_frame, points[0], (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                points.append((x, y))
                temp_shape = None
                return points

        elif zone_type == "polygon":
            # 多边形绘制
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) > 1:
                    cv2.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONDBLCLK:
                if len(points) >= 3:
                    return "__FINISH__"

        elif zone_type == "circle":
            # 圆形绘制
            if event == cv2.EVENT_LBUTTONDOWN:
                if center is None:
                    center = (x, y)
                    drawing = True
                else:
                    drawing = False
                    radius = int(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))
                    temp_shape = None
                    return ("circle", center, radius)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                display_frame[:] = frame.copy()
                radius = int(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))
                temp_shape = ("circle", center, radius)
                cv2.circle(display_frame, center, radius, (0, 255, 0), 2)

        elif zone_type == "free":
            # 自由绘制
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                points = [(x, y)]
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                points.append((x, y))
                if len(points) > 1:
                    cv2.line(display_frame, points[-2], points[-1], (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                if len(points) > 10:  # 至少10个点
                    drawing = False
                    return points

        return None

    # 设置鼠标回调
    cv2.namedWindow("选择检测区域", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("选择检测区域", mouse_callback)

    # 显示说明
    instructions = {
        "rect": "点击拖拽绘制矩形，双击确认",
        "polygon": "点击添加顶点，双击结束",
        "circle": "点击确定圆心，再次点击确定半径",
        "ellipse": "点击确定中心，拖拽确定大小",
        "free": "按下鼠标拖拽绘制，松开结束",
    }

    cv2.putText(
        display_frame,
        f"选择区域类型: {zone_type}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display_frame,
        instructions.get(zone_type, ""),
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display_frame,
        "按 ESC 退出, Enter 确认",
        (50, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )

    print("\n" + "=" * 80)
    print("🎨 交互式区域绘制")
    print("=" * 80)
    print(f"区域类型: {zone_type}")
    print(f"说明: {instructions.get(zone_type, '')}")
    print("操作:")
    print("  鼠标: 绘制区域")
    print("  Enter: 确认选择")
    print("  ESC: 取消")
    print("=" * 80)

    # 主循环
    result = None
    while True:
        cv2.imshow("选择检测区域", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\n⚠️ 用户取消选择")
            break
        elif key == 13:  # Enter
            if zone_type == "polygon" and len(points) >= 3:
                result = PolygonZone()
                result.points = points
                print(f"\n✓ 区域选择完成: {len(points)} 个顶点")
                break
            elif zone_type == "rect" and len(points) == 2:
                result = RectZone()
                x1, y1 = points[0]
                x2, y2 = points[1]
                result.set_rect(x1, y1, x2, y2)
                print(f"\n✓ 矩形区域选择完成: ({x1}, {y1}) -> ({x2}, {y2})")
                break
        elif key == ord("q"):

            print("\n⚠️ 用户取消选择")
            break

    cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    # 测试
    zone = interactive_zone_selection("input.mp4", "polygon")
    if zone:
        print(f"选择的区域: {zone.name}")
