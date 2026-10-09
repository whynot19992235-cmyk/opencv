"""
YOLO 单张照片目标识别教程

你将学到什么：
1. 如何把一张图片交给 YOLO 模型。
2. 如何读取 YOLO 预测出的类别、置信度和目标位置。
3. 如何把识别框画回图片并保存结果。

运行环境：
1. 建议使用 Python 3.10、3.11 或 3.12。
2. 需要安装第三方库：ultralytics、opencv-python。
3. 安装命令：
       python3 -m pip install ultralytics opencv-python
   或者在本目录运行：
       python3 -m pip install -r requirements.txt

使用方法：
1. 先把要识别的图片放到本目录，例如 test.jpg。
2. 修改下面“配置区”里的 IMAGE_PATH。
3. 运行：
       python3 01_yolo_image_detect.py

运行示例：
    python3 01_yolo_image_detect.py

注意：
1. 第一次运行时，ultralytics 会自动下载默认模型 yolov8n.pt。
2. 如果检测框太少，可以把 CONFIDENCE_THRESHOLD 调低。
3. 如果误识别太多，可以把 CONFIDENCE_THRESHOLD 调高。
"""

from pathlib import Path

import cv2
from ultralytics import YOLO


# =========================
# 配置区：初学者主要修改这里
# =========================

# 要识别的图片路径。
# 你可以把图片放在 yolo 目录下，然后把这里改成 "./你的图片名.jpg"。
IMAGE_PATH = "./image2.jpg"

# YOLO 模型名称。
# yolov8n.pt 中的 n 表示 nano：模型较小，下载快、运行快，适合先跑通流程。
MODEL_PATH = "yolov8n.pt"

# 置信度阈值。
# 取值范围通常是 0 到 1。数字越大，模型越“严格”；数字越小，显示的框可能越多。
CONFIDENCE_THRESHOLD = 0.15

# 结果图片保存路径。
OUTPUT_PATH = "./outputs/image_result.jpg"

# 是否弹窗显示结果图片。
# True 表示显示窗口；False 表示只保存图片，不弹窗。
SHOW_WINDOW = True


def main():
    image_path = Path(IMAGE_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片文件：{image_path}")
    # 1. 加载 YOLO 模型。
    # yolov8n.pt 中的 n 表示 nano：模型较小，下载快、运行快，适合初学者先跑通流程。
    model = YOLO(MODEL_PATH)

    # 2. 用 OpenCV 读取图片。
    # cv2.imread() 会把图片读成一个三维数组，可以理解为“高 x 宽 x 颜色通道”的数字表格。
    # OpenCV 读取出来的颜色顺序是 BGR，ultralytics 可以直接处理，不需要你手动转换。
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"图片读取失败，请检查文件是否为有效图片：{image_path}")

    # 3. 执行目标检测。
    # conf 是置信度阈值。只有模型足够“有把握”的目标，才会被保留下来。
    # result.boxes 里保存每一个检测框，包括类别编号、置信度、矩形框坐标。
    results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    # 4. 把检测结果画到图片上。
    # plot() 会自动画矩形框、类别名称和置信度，返回一张已经标注好的新图片。
    annotated_image = result.plot()

    # 5. 在终端打印结构化结果。
    # 你可以通过这些文字结果理解：YOLO 不只是画框，它内部实际输出了类别、分数和坐标。
    print("\n识别结果：")
    if result.boxes is None or len(result.boxes) == 0:
        print("  没有检测到置信度足够高的目标。可以尝试降低 CONFIDENCE_THRESHOLD。")
    else:
        for index, box in enumerate(result.boxes, start=1):
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"  {index}. 类别={class_name}, "
                f"置信度={confidence:.2f}, "
                f"位置=({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})"
            )

    # 6. 保存结果图片。
    # 保存后你可以打开输出图片，对比原图和标注图的区别。
    cv2.imwrite(str(output_path), annotated_image)
    print(f"\n结果图片已保存：{output_path.resolve()}")

    # 7. 可选弹窗显示。按任意键关闭窗口。
    if SHOW_WINDOW:
        cv2.imshow("YOLO Image Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
