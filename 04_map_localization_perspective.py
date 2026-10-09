"""
文件：04_map_localization_perspective.py
第一阶段：地图定位 + 透视变换 + 内部物体检测（精简版）

流程：摄像头 -> 找到地图黑色外框 -> 四个角点 -> 角点排序 -> 透视变换(内部)
      -> 在拉正后的地图上检测 黑色/黄色 物体 -> 轮廓逆变换画回原图 -> 单窗口实时显示

方法说明（都是最传统的 OpenCV 手段，简单稳定）：
1. 灰度 + 高斯模糊去掉噪点；
2. Canny 边缘检测：基于梯度，天然抗光照整体变化，不需要复杂的自适应阈值；
3. 轮廓提取 + 多边形近似成四边形 + 简单几何筛选（面积、凸性、4:3 长宽比）；
4. 角点排序为 左上 / 右上 / 右下 / 左下；
5. getPerspectiveTransform + warpPerspective 把地图拉正为 800×600（仅内部处理）；
6. 黑色物体：自适应阈值；黄色物体：HSV 颜色范围——都在拉正后的地图上检测，
   得到的是地图坐标系下的轮廓，下一阶段可直接做形状分类和网格定位。

运行：python 04_map_localization_perspective.py   （q 键退出）
"""

import cv2
import numpy as np

# ===== 基础配置 =====
CAMERA_INDEX = 0          # 摄像头编号，外接摄像头可能需改成 1
OUT_W, OUT_H = 800, 600   # 透视校正后输出尺寸（4:3，与地图比例一致）
MIN_AREA_RATIO = 0.01     # 外框面积至少占画面的比例，排除小噪声

# 地图内部物体检测配置
MIN_OBJECT_AREA = 100            # 物体轮廓最小面积（warped 坐标），过滤小噪声
BORDER_MARGIN = 50               # 跳过地图四周边框区域的宽度（px），避免把外框当黑色物体
BLACK_BLOCK_SIZE = 25            # 黑色检测：自适应阈值局部窗口（必须是奇数）
BLACK_C = 8                      # 黑色检测：比局部均值暗多少才算黑色
YELLOW_LOWER = (20, 60, 90)      # 黄色检测：HSV 下界 (H, S, V)
YELLOW_UPPER = (35, 255, 255)    # 黄色检测：HSV 上界


def detect_board(frame):
    """在画面中找到地图的黑色外框，返回四个角点（无序）；没找到返回 None。"""
    # 1. 灰度 + 模糊
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny 找边缘，再膨胀把断裂的边连起来
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # 3. 提取轮廓，逐个拟合成四边形，留下最像地图外框的一个
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = gray.size * MIN_AREA_RATIO
    max_area = gray.size * 0.8            # 排除几乎铺满画面的误检
    best, best_area = None, 0
    for cnt in contours:
        # 多边形近似；若点太多，放宽误差再试一次
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # 注意：Canny 得到的轮廓是“线条”，contourArea(cnt) 只是描边面积；
        # 用四边形围成的面积来判断大小才符合直觉，也不会误伤细线条。
        area = cv2.contourArea(approx)
        if not (min_area <= area <= max_area):
            continue
        # 长宽比大致符合 4:3 的地图
        (_, (rw, rh), _) = cv2.minAreaRect(approx)
        if min(rw, rh) <= 0:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not (1.0 < ratio < 3.5):
            continue
        if area > best_area:              # 取面积最大的候选
            best_area, best = area, approx.reshape(4, 2).astype(np.float32)
    return best


def order_points(pts):
    """把四个角点整理成顺序：左上、右上、右下、左下。"""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)          # x+y：最小的是左上，最大的是右下
    d = np.diff(pts, axis=1).ravel()  # y-x：最小的是右上，最大的是左下
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)


def warp_board(frame, corners):
    """用四个角点做透视变换，把地图拉正为 OUT_W×OUT_H。

    返回 (拉正后的地图, 3×3 变换矩阵)。
    matrix 可把 warped 坐标变换回原图，第二阶段做角色定位时可直接复用。
    """
    dst = np.array([[0, 0], [OUT_W - 1, 0],
                    [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (OUT_W, OUT_H))
    return warped, matrix


def draw_board(frame, corners):
    """在原图上画出地图外框（绿线）和四个角点（红点 + 编号）。"""
    pts = np.round(corners).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(frame, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def detect_black_objects(warped):
    """在拉正后的地图上检测黑色物体（自适应阈值），返回轮廓（warped 坐标）。

    只检测地图内部（跳过四周 BORDER_MARGIN 宽的边框带），
    黑色外框从范围上就被排除了，不会误当成普通黑色物体。
    """
    h, w = warped.shape[:2]
    x0 = y0 = BORDER_MARGIN
    roi = warped[y0:h - y0, x0:w - x0]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, BLACK_BLOCK_SIZE, BLACK_C)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去掉噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 把碎片连成整块

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_OBJECT_AREA:
            cnt = cnt.reshape(-1, 2).astype(np.float32)
            cnt[:, 0] += x0                    # 加回 ROI 偏移，回到 warped 坐标
            cnt[:, 1] += y0
            objects.append(cnt)
    return objects


def detect_yellow_objects(warped):
    """在拉正后的地图上检测黄色物体（HSV 颜色范围），返回轮廓（warped 坐标）。"""
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.reshape(-1, 2).astype(np.float32) for cnt in contours
            if cv2.contourArea(cnt) >= MIN_OBJECT_AREA]


def draw_on_original(frame, matrix, contours, color):
    """把透视校正图上检测到的轮廓（warped 坐标）用逆透视变换画回原始画面。"""
    if not contours:
        return
    inv = np.linalg.inv(matrix)
    for cnt in contours:
        back = cv2.perspectiveTransform(cnt.reshape(1, -1, 2), inv).reshape(-1, 2)
        back = np.round(back).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [back], True, color, 2)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败：编号 {CAMERA_INDEX}。")

    print("地图定位 + 物体检测已启动：把地图放进画面，按 q 退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 找地图外框并透视拉正（仅内部处理，不额外显示拉正图）
        corners = detect_board(frame)
        if corners is not None:
            corners = order_points(corners)
            draw_board(frame, corners)                       # 原图画外框 + 角点
            warped, matrix = warp_board(frame, corners)

            # 2. 在拉正后的地图里检测黑色（自适应阈值）和黄色（HSV）物体
            black = detect_black_objects(warped)
            yellow = detect_yellow_objects(warped)

            # 3. 轮廓映射回原图：黑色=红色，黄色=蓝色，外框=绿色
            draw_on_original(frame, matrix, black, (0, 0, 255))
            draw_on_original(frame, matrix, yellow, (255, 0, 0))
            cv2.putText(frame, "board:green  black:red  yellow:blue",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        # 4. 只保留一个实时窗口显示最终结果
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()