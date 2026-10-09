"""
YOLO 实时摄像头目标识别教程

你将学到什么：
1. 摄像头会不断产生新的画面帧。
2. 程序会持续读取摄像头画面，并立即交给 YOLO 检测。
3. 检测结果会马上显示在窗口中，所以你会看到实时识别效果。

运行环境：
1. 建议使用 Python 3.10、3.11 或 3.12。
2. 需要安装第三方库：ultralytics、opencv-python。
3. 安装命令：
       python3 -m pip install ultralytics opencv-python
   或者在本目录运行：
       python3 -m pip install -r requirements.txt
4. 运行摄像头版本时，系统可能会要求你允许 Python 或终端访问摄像头。

运行示例：
    python3 03_yolo_webcam_realtime.py

如果摄像头打不开，可以修改下面“配置区”里的 CAMERA_INDEX，例如从 0 改成 1。

窗口打开后：
    按 q 退出
    按 s 保存当前画面截图
"""

from pathlib import Path
import time

import cv2
from ultralytics import YOLO


# =========================
# 配置区：初学者主要修改这里
# =========================

# 摄像头编号。
# 通常笔记本内置摄像头是 0；如果你使用外接摄像头，可能需要改成 1 或 2。
CAMERA_INDEX = 0

# YOLO 模型名称。
# yolov8n.pt 速度较快，适合先体验实时识别。
MODEL_PATH = "yolov8n.pt"

# 置信度阈值。
# 数字越大，检测结果越严格；数字越小，可能显示更多检测框。
CONFIDENCE_THRESHOLD = 0.25

# 摄像头画面宽度和高度。
# 分辨率越高，画面越清晰，但识别速度可能越慢。
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 按 s 保存截图时使用的目录。
SNAPSHOT_DIR = "./outputs/webcam_snapshots"


def main():
    snapshot_dir = Path(SNAPSHOT_DIR)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 YOLO 模型。
    # yolov8n.pt 速度较快，适合先体验实时识别。
    # 如果你的电脑性能更强，也可以把配置区的 MODEL_PATH 改成 yolov8s.pt 等更大的模型。
    model = YOLO(MODEL_PATH)

    # 2. 打开摄像头。
    # camera=0 表示第一个摄像头，通常是笔记本内置摄像头。
    # 如果打不开，可以把配置区的 CAMERA_INDEX 改成 1 或 2。
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"摄像头打开失败：编号 {CAMERA_INDEX}。"
            "请检查摄像头权限，或尝试把 CAMERA_INDEX 改成 1。"
        )

    # 3. 设置摄像头分辨率。
    # 分辨率越高，画面越清晰，但 YOLO 需要处理的像素越多，推理速度通常会变慢。
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    print("实时识别已启动。按 q 退出，按 s 保存当前画面截图。")

    last_time = time.time()
    fps = 0.0
    snapshot_count = 0

    while True:
        # 4. 从摄像头读取当前画面。
        # 每循环一次，程序就会读取一张最新画面，这张画面也可以理解为一帧。
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头画面失败，程序结束。")
            break

        # 5. 执行 YOLO 检测。
        # 这里每次只检测当前这一帧，然后马上进入下一帧，所以看起来是连续实时识别。
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        # 6. 绘制检测框。
        # annotated_frame 是已经画好类别、置信度和矩形框的新画面。
        annotated_frame = result.plot()

        # 7. 计算并显示实时 FPS。
        # FPS 表示每秒能处理多少帧。FPS 越高，画面越流畅。
        # 模型越大、分辨率越高，FPS 通常越低。
        now = time.time()
        time_diff = now - last_time
        last_time = now
        if time_diff > 0:
            fps = 1.0 / time_diff

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 8. 显示结果窗口。
        # imshow 会把当前这一帧显示出来；下一次循环会显示下一帧。
        cv2.imshow("YOLO Realtime Webcam Detection", annotated_frame)

        # 9. 读取键盘操作。
        # waitKey(1) 表示等待 1 毫秒，既能让窗口刷新，也能读取你按下的键。
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("你已退出实时识别。")
            break
        if key == ord("s"):
            snapshot_count += 1
            snapshot_path = snapshot_dir / f"snapshot_{snapshot_count:03d}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)
            print(f"已保存截图：{snapshot_path.resolve()}")

    # 10. 释放摄像头并关闭窗口。
    # 程序结束时要释放摄像头，否则其他软件可能暂时无法使用摄像头。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
