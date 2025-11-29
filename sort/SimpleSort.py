import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    """
    使用恒定速度模型的卡尔曼滤波器
    状态: [u, v, s, r, u_dot, v_dot, s_dot]
    u,v: bbox中心; s:面积; r:宽高比（固定）;
    """
    count = 0

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        # 初始化状态向量 (7维)
        self.kf = np.zeros((7, 1))
        self.kf[0] = (bbox[0] + bbox[2]) / 2.0  # u
        self.kf[1] = (bbox[1] + bbox[3]) / 2.0  # v
        self.kf[2] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # s
        self.kf[3] = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)  # r

        # 协方差矩阵 P (7x7)
        self.P = np.eye(7) * 1000.0
        self.P[4:, 4:] *= 1000.0  # 速度项初始不确定性大

        # 状态转移矩阵 F (7x7)
        self.F = np.eye(7)
        dt = 1.0  # 假设帧间隔为1
        self.F[0, 4] = dt  # u += u_dot * dt
        self.F[1, 5] = dt  # v += v_dot * dt
        self.F[2, 6] = dt  # s += s_dot * dt

        # 观测矩阵 H (4x7): 只能观测 [u, v, s, r]
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        # 过程噪声协方差 Q (7x7)
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01  # 速度噪声小些

        # 观测噪声协方差 R (4x4)
        self.R = np.eye(4) * 0.1

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0

    def predict(self):
        """预测下一状态"""
        self.kf = np.dot(self.F, self.kf)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        """用观测值更新状态"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # 构造观测向量 z = [u, v, s, r]
        z = np.zeros((4, 1))
        z[0] = (bbox[0] + bbox[2]) / 2.0
        z[1] = (bbox[1] + bbox[3]) / 2.0
        z[2] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        z[3] = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1] + 1e-6)

        # 卡尔曼增益
        y = z - np.dot(self.H, self.kf)  # 残差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态
        self.kf = self.kf + np.dot(K, y)
        I = np.eye(7)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        """返回 [x1, y1, x2, y2] 格式的 bbox"""
        u, v, s, r = self.kf[:4].flatten()
        w = np.sqrt(s * r)
        h = s / w
        x1 = u - w / 2.0
        y1 = v - h / 2.0
        x2 = u + w / 2.0
        y2 = v + h / 2.0
        return [x1, y1, x2, y2]


def iou_batch(bb_test, bb_gt):
    """
    计算两组 bbox 的 IoU 矩阵
    bb_test: (N, 4)  [x1,y1,x2,y2]
    bb_gt:   (M, 4)
    返回: (N, M) IoU 矩阵
    """
    bb_test = np.array(bb_test)
    bb_gt = np.array(bb_gt)

    N = bb_test.shape[0]
    M = bb_gt.shape[0]

    if N == 0 or M == 0:
        return np.zeros((N, M))

    # 计算交集
    ixmin = np.maximum(bb_test[:, 0].reshape(-1, 1), bb_gt[:, 0].reshape(1, -1))
    iymin = np.maximum(bb_test[:, 1].reshape(-1, 1), bb_gt[:, 1].reshape(1, -1))
    ixmax = np.minimum(bb_test[:, 2].reshape(-1, 1), bb_gt[:, 2].reshape(1, -1))
    iymax = np.minimum(bb_test[:, 3].reshape(-1, 1), bb_gt[:, 3].reshape(1, -1))

    iw = np.maximum(ixmax - ixmin + 1, 0)
    ih = np.maximum(iymax - iymin + 1, 0)
    inters = iw * ih

    # 计算并集
    area_test = (bb_test[:, 2] - bb_test[:, 0] + 1) * (bb_test[:, 3] - bb_test[:, 1] + 1)
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0] + 1) * (bb_gt[:, 3] - bb_gt[:, 1] + 1)

    union = area_test.reshape(-1, 1) + area_gt.reshape(1, -1) - inters
    iou = inters / union

    return iou

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: [[x1, y1, x2, y2, score], ...]
        返回: [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        # 预测所有轨迹
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [*pos, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 匹配检测与轨迹
        if len(dets) > 0 and len(trks) > 0:
            iou_matrix = iou_batch(dets[:, :4], trks[:, :4])
            if min(iou_matrix.shape) > 0:
                a = (iou_matrix > self.iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis=1)
                else:
                    # 匈牙利算法（最小化 1 - IoU）
                    cost_matrix = 1 - iou_matrix
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    matched_indices = np.column_stack((row_ind, col_ind))
            else:
                matched_indices = np.empty((0, 2))
        else:
            matched_indices = np.empty((0, 2))

        # 标记未匹配的检测和轨迹
        unmatched_dets = []
        unmatched_trks = list(range(len(trks)))
        matched_det_indices = set()

        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
            else:
                matched_det_indices.add(m[0])
                self.trackers[m[1]].update(dets[m[0], :4])

        for d in range(len(dets)):
            if d not in matched_det_indices:
                unmatched_dets.append(d)

        # 删除已匹配的轨迹索引
        for m in matched_indices:
            if m[1] in unmatched_trks:
                unmatched_trks.remove(m[1])

        # 创建新轨迹
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # 收集输出结果
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append([*d, trk.id + 1])  # ID 从1开始
            i -= 1
            # 删除过期轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.array(ret)