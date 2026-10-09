# 临时验证脚本：验证 order_points 修复后即删除
import importlib.util

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("sx", r"d:\文件\OpenCV代码库\sixth.py")
sx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sx)

DST = np.array([[0, 0], [799, 0], [799, 599], [0, 599]], np.float32)


def make_quad(theta_deg, tilt=(1., 1., 1., 1.), center=(640., 360.)):
    """返回 [TL, TR, BR, BL] 顺序的四边形角点。"""
    theta = np.deg2rad(theta_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    half = np.array([[-240, -180], [240, -180], [240, 180], [-240, 180]], np.float32)
    q = half @ R.T + np.array(center, np.float32)
    q[:, 1] *= np.array(tilt, np.float32)
    return q


# 1. 各种旋转/倾斜下，无序输入也能还原成 [TL,TR,BR,BL]
cases = [(0, (1, 1, 1, 1)), (15, (0.92, 0.88, 1.06, 1.04)), (30, (1, 1, 1, 1)),
         (40, (0.9, 0.85, 1.05, 1.1)), (45, (1, 1, 1, 1))]
for deg, t in cases:
    truth = make_quad(deg, t)
    rng = np.random.default_rng(deg)
    out = sx.order_points(truth[rng.permutation(4)])
    assert out is not None, f"theta={deg} 返回 None"
    err = np.linalg.norm(out - truth, axis=1).max()
    print(f"theta={deg:3d} 排序偏差 {err:.2f} px")
    assert err < 5, f"theta={deg} 排序错误"
print("旋转/倾斜排序 OK")

# 2. 无效输入安全拒绝（返回 None 且不抛异常）
bad = [None,
       np.array([[0, 0], [10, 0], [5, 5]], np.float32),                       # 只有3点
       np.array([[0, 0], [10, 0], [5, 10], [3, np.nan]], np.float32),         # NaN
       np.array([[0, 0], [10, 0], [5, 10], [0, 0]], np.float32),              # 重复点/退化
       np.array([[0, 0], [100, 0], [50, 10], [25, 5]], np.float32),           # 共线退化
       np.array([[0, 0], [100, 0], [30, 30], [100, 100]], np.float32)]        # 凹四边形
for i, b in enumerate(bad):
    try:
        r = sx.order_points(b)
        assert r is None, f"无效输入[{i}] 未被拒绝"
    except Exception as e:
        print(f"无效输入[{i}] 抛异常: {e}")
        raise
print("无效输入全部安全拒绝 OK")

# 3. 角点顺序 与 透视矩阵 匹配（正/逆方向都验证）
q = make_quad(15, (0.92, 0.88, 1.06, 1.04))
out = sx.order_points(q)
M = cv2.getPerspectiveTransform(out, DST)
mapped = cv2.perspectiveTransform(out.reshape(1, 4, 2), M).reshape(4, 2)
err = np.linalg.norm(mapped - DST, axis=1).max()
print("正向透视映射误差: %.6f px" % err)
assert err < 1e-3
back = cv2.perspectiveTransform(DST.reshape(1, 4, 2), np.linalg.inv(M)).reshape(4, 2)
assert np.linalg.norm(back - out, axis=1).max() < 1e-3
print("角点顺序 <-> 透视矩阵 匹配 OK")

# 4. 完整流程冒烟：倾斜地图 + 左上角黑块，验证拉正后朝向正确（不在错误拉伸）
w, h = 1280, 720
xx, yy = np.meshgrid(np.arange(w), np.arange(h))
bg = (210 + (xx / w) * 30).astype(np.uint8)
frame = cv2.merge([bg, bg, bg])
quad = make_quad(15, (0.92, 0.88, 1.06, 1.04))
M_room = np.linalg.inv(cv2.getPerspectiveTransform(quad.astype(np.float32), DST))


def draw_poly(pts, color):
    p = cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(1, -1, 2), M_room).reshape(-1, 2)
    cv2.fillPoly(frame, [np.round(p).astype(np.int32)], color)


draw_poly([[0, 0], [799, 0], [799, 599], [0, 599]], (0, 0, 0))
draw_poly([[24, 24], [775, 24], [775, 575], [24, 575]], (200, 200, 200))
draw_poly([[100, 100], [140, 100], [140, 140], [100, 140]], (0, 0, 0))  # 靠近地图左上角
detected = sx.detect_board(frame)
assert detected is not None
corners = sx.order_points(detected)
assert corners is not None
warped, matrix = sx.warp_board(frame, corners)
shapes = sx.detect_shapes(warped)
assert shapes, "未识别到黑色方块"
cen = shapes[0][2]
print("拉正后黑块中心:", tuple(round(c, 1) for c in cen), "（应位于左上区域 x<400,y<300）")
assert cen[0] < 400 and cen[1] < 300, "拉正方向错误（黑块不在左上）"
print("完整流程冒烟 OK")

# 5. fuzz：200 组随机凸四边形，绝不抛异常，且与透视矩阵一致
rng = np.random.default_rng(7)
for k in range(200):
    ang = float(rng.uniform(0, 45))
    cx0, cy0 = float(rng.uniform(300, 900)), float(rng.uniform(100, 500))
    tilt = rng.uniform(0.85, 1.15, 4)
    quad = make_quad(ang, tilt, (cx0, cy0))
    res = sx.order_points(quad[rng.permutation(4)])
    if res is not None:
        Mt = cv2.getPerspectiveTransform(res, DST)
        mm = cv2.perspectiveTransform(res.reshape(1, 4, 2), Mt).reshape(4, 2)
        assert np.linalg.norm(mm - DST, axis=1).max() < 1e-3
print("fuzz 200 组无异常 OK")
print("全部验证通过 OK")