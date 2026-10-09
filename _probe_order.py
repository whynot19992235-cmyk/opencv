# 临时探针：逐步检查 order_points 内部每步结果
import numpy as np
import cv2

import importlib.util
spec = importlib.util.spec_from_file_location("sx", r"d:\文件\OpenCV代码库\sixth.py")
sx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sx)

pts = np.array([[400, 180], [880, 180], [880, 540], [400, 540]], np.float32)  # TL,TR,BR,BL 正立
pts = pts[np.array([2, 0, 3, 1])]  # 打乱
print("输入:", pts.tolist())

p = np.asarray(pts, np.float32)
print("shape:", p.shape, "finite:", np.all(np.isfinite(p)))
contour = p.reshape(4, 1, 2)
print("convex:", cv2.isContourConvex(contour))
area = cv2.contourArea(contour)
print("area:", area)
(bx, by, bw, bh) = cv2.boundingRect(contour)
print("bbox:", bx, by, bw, bh, "ratio:", area / (bw * bh))

cx, cy = p[:, 0].mean(), p[:, 1].mean()
angles = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
cyclic = p[np.argsort(angles)]
print("cyclic:", cyclic.tolist())

def _horiz_score(i):
    a, b = cyclic[i], cyclic[(i + 1) % 4]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = np.hypot(dx, dy)
    return abs(dx) / length if length > 0 else 0.0

scores = [_horiz_score(i) for i in range(4)]
mean_y = [0.5 * (cyclic[i][1] + cyclic[(i + 1) % 4][1]) for i in range(4)]
print("scores:", scores)
print("mean_y:", mean_y, "cy:", cy)
best = max(scores)
candidates = [i for i in range(4) if scores[i] >= best * 0.6]
above = [i for i in candidates if mean_y[i] < cy]
top_edges = above if above else candidates
print("candidates:", candidates, "above:", above, "top_edges:", top_edges)
top_i = sorted(top_edges, key=lambda i: (-scores[i], mean_y[i]))[0]
print("top_i:", top_i)

a, b = cyclic[top_i], cyclic[(top_i + 1) % 4]
print("a,b:", a, b)
if a[0] <= b[0]:
    tl, tr, br, bl = a, b, cyclic[(top_i + 2) % 4], cyclic[(top_i + 3) % 4]
else:
    tl, tr, br, bl = b, a, cyclic[(top_i + 3) % 4], cyclic[(top_i + 2) % 4]
ordered = np.array([tl, tr, br, bl], np.float32)
print("ordered:", ordered.tolist())
print("convex2:", cv2.isContourConvex(ordered.reshape(4, 1, 2)))
interior = sx.polygon_angles(ordered)
print("interior angles:", interior)