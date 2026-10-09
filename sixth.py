"""
文件：sixth.py
地图定位 + 纸上形状识别（黑三角 / 黑矩形 / 黑圆形、黄圆）+ 4×3 网格坐标
只保留一个实时窗口

流程：
摄像头 -> 纸张黑色外框(二值化/Canny + 轮廓筛选) -> 四角排序 -> 透视变换(内部 800×600)
-> 拉正图上识别形状与中心点 -> 固定 4×3 网格算行列坐标 -> 逆透视画回原图
-> 单一 "Original" 窗口实时显示

稳定性设计：
1. 外框跨帧平滑：角点平滑、跳变需连续多帧确认才切换、短暂漏检保留旧角点、
   连续漏检超过设定帧数后清除；纸张移动时也能逐步跟上，不会永久锁定旧位置；
2. 形状按中心点位置跨帧匹配跟踪：连续多帧确认后才显示，短暂消失保留，
   减少标签与坐标闪烁；
3. 网格为等距 4 行 3 列，用固定布局直接计算坐标，不依赖逐根检测细网格线。

运行：python sixth.py   （q 键退出）
"""

import cv2
import numpy as np

# ===== 可调参数（根据实际摄像头画面调整） =====
CAMERA_INDEX = 1                       # 摄像头编号（保持 1）
OUT_W, OUT_H = 800, 600                # 透视校正后输出尺寸（4:3）

# ---- 纸张外框定位 ----
MIN_AREA_RATIO = 0.01                  # 外框面积下限（占画面比例），排除小图形/网格
BOARD_MAX_RATIO = 0.8                  # 外框面积上限（占画面比例），排除整幅画面误检
ASPECT_MIN, ASPECT_MAX = 1.0, 3.5      # 外框长宽比范围（纸张约 4:3 ≈ 1.33）
BOARD_SMOOTH = 0.5                     # 外框角点平滑系数（0~1，越大跟随越快）
BOARD_JUMP_MAX = 40                    # 角点允许最大跳动（px），超过视为异常/跳变
BOARD_CONFIRM = 3                      # 新位置需连续几帧确认才切换（拒绝瞬时误检）
BOARD_MISS_MAX = 30                    # 连续漏检多少帧后清除外框（约 0.5~1 秒）

# ---- 形状识别（warped 800×600 坐标） ----
SHAPE_MARGIN = 50                      # 跳过纸张边框区域宽度，避免把外框当形状
MIN_SHAPE_AREA = 300                   # 形状最小面积（过滤噪声/网格碎片/阴影）
BLACK_MAX_GRAY = 90                    # 黑色形状内部平均灰度上限（排除阴影、黄色等）
TRI_ANGLE_MIN = 40                     # 三角形最小内角（度）
RECT_ANGLE_TOL = 20                    # 矩形内角与 90° 的允许偏差（度）
MAX_RECT_ASPECT = 3.0                  # 矩形最大长宽比（排除细长网格线）
CIRCULARITY_THRESHOLD = 0.75           # 圆形度阈值：4π·面积/周长²
YELLOW_LOWER = (15, 60, 90)            # 黄色 HSV 下界 (H, S, V)
YELLOW_UPPER = (40, 255, 255)          # 黄色 HSV 上界

# ---- 形状跨帧跟踪 ----
SHAPE_CONFIRM = 3                      # 连续几帧确认后才显示
SHAPE_LOST_MAX = 10                    # 消失多少帧后移除
SHAPE_MATCH_RADIUS = 45                # 位置匹配半径（warped px）

# ---- 网格坐标（等距 4 行 3 列） ----
GRID_ROWS = 4
GRID_COLS = 3
COORD_BASE = 0                         # 行列从 0 开始编号
# 网格区域在拉正图(800×600)中的 左/上/右/下 边界（像素）。
# 说明：网格不一定铺满整张纸。先运行程序观察拉正图（内部）里网格的实际范围，
# 再把四边设成网格的边界；默认取黑色外框内侧。边界不准确时，
# 靠近边界的格子的坐标会判错，请务必按实际纸张校准这 4 个值。
GRID_LEFT, GRID_TOP = 60, 60
GRID_RIGHT, GRID_BOTTOM = 740, 540

# 绘制颜色：外框=绿，三角形=红，矩形=蓝，圆形=黄
SHAPE_COLORS = {"tri": (0, 0, 255),
                "rect": (255, 0, 0),
                "cir": (0, 255, 255)}


def grid_coord(cx, cy):
    """把拉正图中的中心点换算成网格坐标 (row, col)；越界返回 None。

    row 由 y（上下）计算、col 由 x（左右）计算，行列不会颠倒。
    """
    if not (GRID_LEFT <= cx <= GRID_RIGHT and GRID_TOP <= cy <= GRID_BOTTOM):
        return None
    col = int((cx - GRID_LEFT) * GRID_COLS / (GRID_RIGHT - GRID_LEFT))
    row = int((cy - GRID_TOP) * GRID_ROWS / (GRID_BOTTOM - GRID_TOP))
    col = min(max(col, 0), GRID_COLS - 1)
    row = min(max(row, 0), GRID_ROWS - 1)
    return COORD_BASE + row, COORD_BASE + col

class BoardTracker:
    """跨帧稳定纸张外框：平滑抖动、拒绝跳变、短暂漏检保留、连续漏检清除。"""

    def __init__(self):
        self.stable = None       # 当前生效的有序角点
        self.pending = None      # 跳变候选角点
        self.pending_count = 0
        self.miss = 0            # 连续漏检帧数

    @staticmethod
    def _dist(a, b):
        """四个角点对应位置的最大距离（px）。"""
        return float(np.linalg.norm(a - b, axis=1).max())

    def update(self, corners):
        """输入本帧检测到的有序角点（或 None），返回应生效的稳定角点（或 None）。"""
        if corners is None:
            self.miss += 1
            if self.miss > BOARD_MISS_MAX:
                # 连续漏检超时：纸张可能已移走，清除旧位置
                self.stable = None
                self.pending = None
                self.pending_count = 0
            return self.stable

        corners = corners.astype(np.float32)

        if self.stable is None:
            self.stable = corners
            self.miss = 0
            return self.stable

        if self._dist(corners, self.stable) <= BOARD_JUMP_MAX:
            # 与当前位置一致：逐帧平滑，抑制抖动；刷新漏检计数
            a = BOARD_SMOOTH
            self.stable = self.stable * (1 - a) + corners * a
            self.miss = 0
            self.pending = None
            self.pending_count = 0
        else:
            # 出现明显跳变：先记为候选，连续多帧一致才切换，
            # 避免把一帧误检当成纸张真的移动
            if self.pending is None:
                self.pending = corners
                self.pending_count = 1
            elif self._dist(corners, self.pending) <= BOARD_JUMP_MAX:
                a = BOARD_SMOOTH
                self.pending = self.pending * (1 - a) + corners * a
                self.pending_count += 1
            else:
                self.pending = corners
                self.pending_count = 1
            if self.pending_count >= BOARD_CONFIRM:
                self.stable = self.pending
                self.pending = None
                self.pending_count = 0
                self.miss = 0
        return self.stable


class ShapeTracker:
    """按中心点位置跨帧匹配跟踪形状：确认后显示、短暂消失保留、防止标签闪烁。"""

    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """detections: [(名称, warped轮廓, warped中心点)]，返回应显示的跟踪项列表。

        返回的每个跟踪项是 dict：{"name", "contour", "centroid"}。
        """
        unmatched = list(detections)
        for t in self.tracks:
            best, bd = None, SHAPE_MATCH_RADIUS
            for i, d in enumerate(unmatched):
                dd = float(np.linalg.norm(
                    np.asarray(d[2], np.float32) - t["centroid"]))
                if dd < bd:
                    bd, best = dd, i
            if best is not None:
                name, contour, centroid = unmatched.pop(best)
                t["name"] = name
                t["contour"] = np.asarray(contour, np.float32)
                # 中心点平滑，避免坐标逐帧跳动
                t["centroid"] = t["centroid"] * 0.7 + np.asarray(centroid, np.float32) * 0.3
                t["streak"] += 1
                t["lost"] = 0
            else:
                t["lost"] += 1  # 本帧没匹配上：短暂保留，超过上限再移除

        for name, contour, centroid in unmatched:
            self.tracks.append({"name": name,
                                "contour": np.asarray(contour, np.float32),
                                "centroid": np.asarray(centroid, np.float32),
                                "streak": 1, "lost": 0})

        self.tracks = [t for t in self.tracks if t["lost"] <= SHAPE_LOST_MAX]
        return [t for t in self.tracks if t["streak"] >= SHAPE_CONFIRM]

def _pick_board(mask, img_size):
    """在二值图/边缘图上筛选最像纸张外框的四边形角点。

    筛选条件：面积在合理范围、凸四边形、长宽比接近纸张比例，优先面积较大。
    """
    # 闭运算把断开的黑色外框连起来（Canny 边缘也一起闭合）
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img_size * MIN_AREA_RATIO
    max_area = img_size * BOARD_MAX_RATIO
    best, best_area = None, 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # 用四边形围成的面积判断大小（对二值填充块和 Canny 细线条都成立）
        area = cv2.contourArea(approx)
        if not (min_area <= area <= max_area):
            continue
        # 长宽比大致符合纸张的 4:3
        (_, (rw, rh), _) = cv2.minAreaRect(approx)
        if min(rw, rh) <= 0:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not (ASPECT_MIN <= ratio <= ASPECT_MAX):
            continue
        if area > best_area:               # 优先面积较大的外框
            best_area, best = area, approx.reshape(4, 2).astype(np.float32)
    return best


def detect_board(frame):
    """在画面中找到纸张的黑色外框，返回四个角点（无序）；没找到返回 None。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 方案一：Otsu 反向二值化（黑色外框在亮纸上最干净）
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best = _pick_board(mask, gray.size)
    if best is not None:
        return best

    # 方案二：光照不均时用自适应阈值
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 10)
    best = _pick_board(mask, gray.size)
    if best is not None:
        return best

    # 方案三：Canny 边缘兜底
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    return _pick_board(edges, gray.size)


def order_points(pts):
    """将四个角点排序为：左上、右上、右下、左下。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    # 必须有四个有效且不同的点
    if len(pts) != 4 or not np.isfinite(pts).all():
        return None

    if len(np.unique(np.round(pts, 3), axis=0)) != 4:
        return None

    # 按照相对中心点的角度排序，形成连续的四边形
    center = np.mean(pts, axis=0)
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )
    cyclic = pts[np.argsort(angles)]

    if not cv2.isContourConvex(cyclic.reshape(4, 1, 2)):
        return None

    def edge_len(i):
        p = cyclic[i]
        q = cyclic[(i + 1) % 4]
        return float(np.linalg.norm(q - p))

    def horizontal_score(i):
        length = edge_len(i)
        if length <= 0:
            return 0.0
        dx = cyclic[(i + 1) % 4][0] - cyclic[i][0]
        return abs(float(dx)) / length

    lengths = [edge_len(i) for i in range(4)]
    scores = [horizontal_score(i) for i in range(4)]
    mean_y = [
        (cyclic[i][1] + cyclic[(i + 1) % 4][1]) / 2
        for i in range(4)
    ]

    # 选择较长的一组对边作为纸张的宽边
    pair_a = lengths[0] + lengths[2]
    pair_b = lengths[1] + lengths[3]

    if pair_a > pair_b * 1.1:
        pair = (0, 2)
    elif pair_b > pair_a * 1.1:
        pair = (1, 3)
    else:
        pair = (
            (0, 2)
            if scores[0] + scores[2] >= scores[1] + scores[3]
            else (1, 3)
        )

    # 两条候选宽边中，选择图像位置更靠上的一条
    top_i = min(pair, key=lambda i: mean_y[i])

    a = cyclic[top_i]
    b = cyclic[(top_i + 1) % 4]

    if a[0] <= b[0]:
        ordered = np.array([
            a, b,
            cyclic[(top_i + 2) % 4],
            cyclic[(top_i + 3) % 4]
        ], dtype=np.float32)
    else:
        ordered = np.array([
            b, a,
            cyclic[(top_i + 3) % 4],
            cyclic[(top_i + 2) % 4]
        ], dtype=np.float32)

    # 排除明显异常的四边形
    if not cv2.isContourConvex(ordered.reshape(4, 1, 2)):
        return None

    interior = polygon_angles(ordered)
    if min(interior) < 20 or max(interior) > 160:
        return None

    if abs(cv2.contourArea(ordered.reshape(4, 1, 2))) < 10:
        return None

    return ordered


def warp_board(frame, corners):
    """四个角点(左上,右上,右下,左下) -> 透视变换到 OUT_W×OUT_H（仅内部使用）。"""
    dst = np.array([[0, 0], [OUT_W - 1, 0],
                    [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (OUT_W, OUT_H))
    return warped, matrix


def draw_board(frame, corners):
    """在原图上画出纸张外框（绿线）和四个角点（红点 + 编号）。"""
    pts = np.round(corners).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def polygon_angles(pts):
    """计算多边形各顶点内角（度）。"""
    pts = pts.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    angles = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1, v2 = a - b, c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        cos = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos)))
    return angles


def classify_shape(cnt):
    """用顶点数、内角、长宽比、圆形度分类为 tri/rect/cir；不明确返回 None。

    矩形不只数顶点：还要求四个内角都接近 90° 且长宽比不能太离谱，
    网格线、细长碎片等不会被误判成矩形。
    """
    area = cv2.contourArea(cnt)
    if area < MIN_SHAPE_AREA:
        return None
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if not cv2.isContourConvex(approx):      # 凸性筛选
        return None

    if len(approx) == 3:
        if min(polygon_angles(approx)) >= TRI_ANGLE_MIN:
            return "tri"
    elif len(approx) == 4:
        angles = polygon_angles(approx)
        if all(abs(a - 90.0) <= RECT_ANGLE_TOL for a in angles):
            (_, (rw, rh), _) = cv2.minAreaRect(approx)
            if min(rw, rh) > 0:
                ratio = max(rw, rh) / min(rw, rh)
                if ratio <= MAX_RECT_ASPECT:     # 排除细长线条
                    return "rect"
    else:                                       # 5 个以上顶点：圆形度判断
        circ = 4 * np.pi * area / (peri * peri)
        if circ >= CIRCULARITY_THRESHOLD:
            return "cir"
    return None

def _centroid(cnt):
    """返回轮廓质心 (cx, cy)。"""
    m = cv2.moments(cnt)
    if m["m00"] > 0:
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
    pts = cnt.reshape(-1, 2)
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def detect_shapes(warped):
    """在拉正图内部识别形状，返回 [(名称, warped轮廓, warped中心点), ...]。

    黑色形状走灰度+Otsu；黄色圆形走 HSV 掩码（避免灰度处理丢失黄色特征）。
    """
    h, w = warped.shape[:2]
    x0 = y0 = SHAPE_MARGIN
    roi = warped[y0:h - y0, x0:w - x0]      # 跳过外框区域，避免把外框当形状
    kernel = np.ones((5, 5), np.uint8)
    shapes = []

    # ---- 黑色形状：灰度 + Otsu 反向二值化 ----
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # 去掉细网格线/噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # 补形状内部断点
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_SHAPE_AREA:
            continue
        # 内部平均灰度必须足够暗：过滤阴影、浅色网格、黄色等
        fill = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(fill, [cnt], -1, 255, -1)
        if cv2.mean(gray, mask=fill)[0] > BLACK_MAX_GRAY:
            continue
        name = classify_shape(cnt)
        if name is None:
            continue
        cnt2 = cnt.reshape(-1, 2).astype(np.float32)
        cnt2[:, 0] += x0                     # 加回 ROI 偏移，回到 warped 坐标
        cnt2[:, 1] += y0
        shapes.append((name, cnt2, _centroid(cnt2)))

# 函数外：统一定义颜色
SHAPE_COLORS = {
    "tri": (0, 0, 255),       # 红色：三角形
    "rect": (255, 0, 0),      # 蓝色：矩形
    "cir": (0, 255, 255),     # 黄色：黑色圆形
    "nailong": (0, 165, 255)  # 橙色：黄色圆形
}


def draw_tracks(frame, matrix, tracks):
    # 函数内：根据类别取颜色
    for t in tracks:
        name = t["name"]
        color = tuple(
            int(v) for v in SHAPE_COLORS.get(name, (255, 255, 255))
        )

        # 后面继续保留原来的绘制代码

def draw_tracks(frame, matrix, tracks):
    """把跟踪到的形状轮廓、中心点和网格坐标用逆透视变换画回原图。"""
    if not tracks:
        return
    inv = np.linalg.inv(matrix)              # warped -> 原图
    for t in tracks:
        name = t["name"]
    color = tuple(
    int(v) for v in SHAPE_COLORS.get(name, (255, 255, 255))
)


        # 轮廓映射回原图
    back = cv2.perspectiveTransform(t["contour"].reshape(1, -1, 2), inv).reshape(-1, 2)
    back_int = np.round(back).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [back_int], True, color, 2)

    # 中心点映射回原图
    cen = np.asarray(t["centroid"], np.float32).reshape(1, 1, 2)
    p = cv2.perspectiveTransform(cen, inv).reshape(-1)
    px, py = int(p[0]), int(p[1])
    cv2.circle(frame, (px, py), 4, color, -1)

    # 网格坐标：在 warped 坐标中计算（行列不颠倒）；越界则不标坐标
    g = grid_coord(float(t["centroid"][0]), float(t["centroid"][1]))
    text = f"{name} (row={g[0]}, col={g[1]})" if g is not None else name
    cv2.putText(frame, text, (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def open_camera():
    """打开摄像头：默认后端（Windows 下通常为 MSMF），失败则改 DirectShow。"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"摄像头 {CAMERA_INDEX} 默认后端打开失败，改用 DirectShow...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    return cap


def main():
    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败：编号 {CAMERA_INDEX}。")

    board_tracker = BoardTracker()
    shape_tracker = ShapeTracker()
    print("地图定位 + 形状识别 + 网格坐标 已启动：把纸张放进画面，按 q 退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            # 偶尔取帧失败不崩溃：用 DirectShow 重新打开一次
            print("读取画面失败，改用 DirectShow 重新打开...")
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("摄像头重开失败，程序结束。")
                break
            continue

        # 1. 纸张外框：检测 -> 排序 -> 跨帧稳定
        detected = detect_board(frame)
        ordered = None
        if detected is not None:
            ordered = order_points(detected)   # 角点无效时返回 None，不进入有效更新
        if ordered is None:
            # 本帧没有可靠外框/角点无效：跳过有效更新，
            # 由跟踪器保留上一帧有效位置（连续无效超过时限才清除）
            corners = board_tracker.update(None)
        else:
            corners = board_tracker.update(ordered)

        # 2. 内部透视变换 + 形状识别 + 网格坐标，统一画回原图
        if corners is not None:
            draw_board(frame, corners)                          # 外框绿色 + 角点
            warped, matrix = warp_board(frame, corners)
            detections = detect_shapes(warped)
            tracks = shape_tracker.update(detections)
            draw_tracks(frame, matrix, tracks)

        # 3. 只保留一个实时窗口
        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()