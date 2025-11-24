"""
轻量级多目标跟踪器（修复版）
基于IoU匹配，无需训练，适用于车辆和行人跟踪
修复了索引越界问题
"""

from collections import defaultdict

import cv2
import numpy as np


class SimpleTracker:
    """简单多目标跟踪器"""

    def __init__(self, max_disappeared=30, min_iou=0.3):
        """
        初始化跟踪器
        Args:
            max_disappeared: 最大消失帧数（超过则删除跟踪对象）
            min_iou: 最小IoU阈值（用于匹配检测框）
        """
        self.next_object_id = 0
        self.objects = {}  # id -> track info
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou

    def register(self, bbox, class_name):
        """注册新目标"""
        object_id = self.next_object_id
        self.next_object_id += 1

        self.objects[object_id] = {
            "bbox": bbox,
            "class": class_name,
            "disappeared": 0,
            "trajectory": [bbox],  # 轨迹
            "violation_frames": [],  # 违规帧列表
            "violation_flag": False,  # 是否违规
            "entry_time": None,  # 进入检测区域时间
            "exit_time": None,  # 离开检测区域时间
        }
        return object_id

    def deregister(self, object_id):
        """删除目标"""
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections):
        """
        更新跟踪器
        Args:
            detections: YOLO检测结果列表，每个包含 (bbox, class_name, confidence)
        Returns:
            跟踪后的目标列表
        """
        if len(detections) == 0:
            # 标记所有对象为消失
            for object_id in list(self.objects.keys()):
                self.objects[object_id]["disappeared"] += 1
                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 当前帧检测框
        input_bboxes = np.array([d["bbox"] for d in detections])
        input_classes = [d["class"] for d in detections]

        # 如果没有跟踪对象，直接注册所有检测结果
        if len(self.objects) == 0:
            for i, (bbox, class_name) in enumerate(zip(input_bboxes, input_classes)):
                self.register(bbox, class_name)
        else:
            # 获取已有对象的bbox和id
            object_ids = list(self.objects.keys())
            object_bboxes = np.array([self.objects[id]["bbox"] for id in object_ids])

            # 计算IoU矩阵
            iou_matrix = self._iou_matrix(input_bboxes, object_bboxes)

            # 贪心匹配：找到每个检测框的最佳匹配
            matched_indices = []
            if iou_matrix.size > 0:
                # 使用改进的匹配算法，确保索引不越界
                for i in range(len(input_bboxes)):
                    if i >= iou_matrix.shape[0]:
                        break

                    # 找到当前行的最大值
                    row = iou_matrix[i, :]
                    if len(row) == 0:
                        continue

                    max_val = np.max(row)
                    max_col = np.argmax(row)

                    # 检查最大值是否超过阈值且列索引有效
                    if max_val > self.min_iou and max_col < len(object_ids):
                        matched_indices.append((i, max_col))
                        # 将当前行和匹配列设为-1，避免重复匹配
                        if i < iou_matrix.shape[0]:
                            iou_matrix[i, :] = -1
                        if max_col < iou_matrix.shape[1]:
                            iou_matrix[:, max_col] = -1

            # 标记已匹配和未匹配的检测框
            matched_detections = set([i for i, _ in matched_indices])
            unmatched_detections = set(range(len(input_bboxes))) - matched_detections

            # 标记已匹配和未匹配的对象
            matched_objects = set([j for _, j in matched_indices])
            unmatched_objects = (
                set([i for i in range(len(object_ids))]) - matched_objects
            )

            # 更新匹配的对象
            for detection_idx, object_idx in matched_indices:
                # 添加边界检查
                if detection_idx >= len(input_bboxes) or object_idx >= len(object_ids):
                    continue

                object_id = object_ids[object_idx]

                # 更新bbox和class
                self.objects[object_id]["bbox"] = input_bboxes[detection_idx]
                self.objects[object_id]["class"] = input_classes[detection_idx]
                self.objects[object_id]["disappeared"] = 0

                # 添加轨迹点
                self.objects[object_id]["trajectory"].append(
                    input_bboxes[detection_idx]
                )
                if len(self.objects[object_id]["trajectory"]) > 50:  # 限制轨迹长度
                    self.objects[object_id]["trajectory"] = self.objects[object_id][
                        "trajectory"
                    ][-50:]

            # 处理未匹配的检测框（注册新目标）
            for detection_idx in unmatched_detections:
                if detection_idx < len(input_bboxes):
                    self.register(
                        input_bboxes[detection_idx], input_classes[detection_idx]
                    )

            # 处理未匹配的对象（标记为消失）
            for object_idx in unmatched_objects:
                if object_idx < len(object_ids):
                    object_id = object_ids[object_idx]
                    self.objects[object_id]["disappeared"] += 1

                    if self.objects[object_id]["disappeared"] > self.max_disappeared:
                        self.deregister(object_id)

        return self.objects

    def _iou_matrix(self, bboxes1, bboxes2):
        """计算两组bbox之间的IoU矩阵"""
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.array([])

        # 转换bbox格式 (x1, y1, x2, y2) -> (x, y, w, h)
        def bbox_to_xywh(bbox):
            return np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])

        def xywh_to_bbox(xywh):
            return np.array([xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]])

        # 转换为xywh格式
        bboxes1_xywh = np.array([bbox_to_xywh(b) for b in bboxes1])
        bboxes2_xywh = np.array([bbox_to_xywh(b) for b in bboxes2])

        # 计算IoU矩阵
        iou_matrix = np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

        for i in range(len(bboxes1)):
            for j in range(len(bboxes2)):
                iou_matrix[i, j] = self._iou(bboxes1_xywh[i], bboxes2_xywh[j])

        return iou_matrix

    def _iou(self, box1, box2):
        """计算两个bbox的IoU"""
        # box1, box2: (x, y, w, h)
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2

        # 计算交集
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def draw_tracks(self, frame):
        """在帧上绘制跟踪轨迹和ID"""
        for object_id, track_info in self.objects.items():
            bbox = track_info["bbox"]
            class_name = track_info["class"]
            trajectory = track_info["trajectory"]

            # 绘制轨迹
            if len(trajectory) > 1:
                points = [
                    (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                    for bbox in trajectory[-10:]
                ]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], (255, 255, 0), 2)

            # 绘制bbox
            x1, y1, x2, y2 = map(int, bbox)
            color = self._get_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制ID和类别
            label = f"ID:{object_id} {class_name}"
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return frame

    def _get_color(self, class_name):
        """获取类别对应的颜色"""
        color_map = {
            "car": (0, 0, 255),
            "truck": (0, 0, 255),
            "bus": (0, 0, 255),
            "motorcycle": (255, 0, 0),
            "bicycle": (255, 0, 0),
            "person": (0, 255, 0),
            "traffic light": (0, 255, 255),
            "stop sign": (255, 255, 0),
        }
        return color_map.get(class_name, (255, 255, 255))


if __name__ == "__main__":
    # 测试跟踪器
    tracker = SimpleTracker()

    # 模拟YOLO检测结果
    detections = [
        {"bbox": [100, 100, 200, 200], "class": "car", "confidence": 0.9},
        {"bbox": [300, 300, 400, 400], "class": "person", "confidence": 0.8},
    ]

    print("测试跟踪器...")
    tracks = tracker.update(detections)
    print(f"跟踪目标数: {len(tracks)}")
    for obj_id, info in tracks.items():
        print(f"  ID {obj_id}: {info['class']} at {info['bbox']}")
