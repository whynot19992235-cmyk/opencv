"""
OpenCV 教程版综合案例：图像处理实验台

这个案例适合初学者第一次系统认识 OpenCV。
程序会在一个窗口里同时展示多个常见图像处理结果：
    1. 原图
    2. BGR 三通道拆分
    3. 灰度图
    4. 高斯模糊
    5. Canny 边缘检测
    6. 阈值分割
    7. 形态学处理
    8. 轮廓检测和计数
    9. HSV 颜色提取
    10. 直方图均衡化

运行方式：
    1. 先把终端 / 命令提示符切换到本文件所在的文件夹。
       Windows 示例：cd 你的课程文件夹\\opencv
       macOS 示例：cd 你的课程文件夹/opencv

    2. 安装依赖库。可以直接安装到当前 Python，也可以安装到虚拟环境。
       Windows 常用命令：py -m pip install -r requirements.txt
       macOS 常用命令：python3 -m pip install -r requirements.txt

    3. 运行程序。
       Windows 常用命令：py opencv_rich_teaching_case.py
       macOS 常用命令：python3 opencv_rich_teaching_case.py

常见问题：
    - 如果提示 No module named 'cv2'，说明 OpenCV 没有安装到当前 Python 环境。
    - 如果窗口没有显示，确认不是在不支持图形界面的远程终端里运行。
    - 如果打开摄像头失败，检查系统相机权限，并确认摄像头没有被其它软件占用。
    - 如果画面太卡，把 FRAME_WIDTH / FRAME_HEIGHT 调小，例如 960 / 540。
"""

from pathlib import Path

import cv2
import numpy as np


# =========================
# 1. 可修改配置
# =========================
# 如果要调整程序，建议先只改这里。

# 是否使用摄像头：
#   False：不使用摄像头，程序自动生成一张教学测试图，最稳定。
#   True：使用摄像头，把摄像头画面实时做各种 OpenCV 处理。
USE_CAMERA = True

# 摄像头编号。大多数电脑内置摄像头是 0。
CAMERA_INDEX = int(input("请输入摄像头编号："))

# 摄像头画面大小。分辨率越高，画面越清楚，但计算也越慢。
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 如果想用自己的图片，把图片路径写在这里。
# 例如 Windows: IMAGE_PATH = r"C:\Users\student\Desktop\test.jpg"
# 例如 macOS:   IMAGE_PATH = "/Users/student/Desktop/test.jpg"
# 留空字符串表示不用外部图片，自动生成教学测试图。
IMAGE_PATH = ""

# 是否自动保存一张实验台截图，方便交作业或写课堂笔记。
SAVE_GALLERY_IMAGE = True
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# OpenCV 窗口名称。
WINDOW_NAME = "OpenCV Teaching Lab"


# =========================
# 2. 生成或读取原始图像
# =========================
def create_teaching_image(width=1280, height=720):
    """
    生成一张专门用于 OpenCV 教学的测试图。

    为什么不直接依赖外部图片？
        因为学生拿到代码后，可能没有图片文件。
        自动生成测试图可以保证每个人第一次运行都能看到效果。

    这张图故意包含：
        - 不同颜色：方便讲 BGR、HSV、颜色提取。
        - 几何图形：方便讲轮廓检测。
        - 黑白文字：方便讲灰度、阈值、边缘。
        - 噪声点：方便讲模糊降噪。
        - 倾斜矩形：方便讲边缘和轮廓。
    """
    image = np.full((height, width, 3), (245, 245, 245), dtype=np.uint8)

    # 画一个浅色背景网格，帮助学生观察坐标和画面位置。
    for x in range(0, width, 80):
        cv2.line(image, (x, 0), (x, height), (225, 225, 225), 1)
    for y in range(0, height, 80):
        cv2.line(image, (0, y), (width, y), (225, 225, 225), 1)

    # OpenCV 颜色顺序是 BGR，不是 RGB。
    # 下面的 (255, 0, 0) 在 OpenCV 里是蓝色，不是红色。
    cv2.rectangle(image, (60, 80), (260, 280), (255, 0, 0), -1)     # 蓝色矩形
    cv2.circle(image, (410, 180), 105, (0, 255, 0), -1)             # 绿色圆
    cv2.ellipse(image, (650, 180), (130, 80), 0, 0, 360, (0, 0, 255), -1)  # 红色椭圆

    # 画一个倾斜四边形，用来观察边缘和轮廓。
    paper = np.array([[820, 70], [1130, 130], [1080, 350], [760, 300]], dtype=np.int32)
    cv2.fillPoly(image, [paper], (235, 235, 255))
    cv2.polylines(image, [paper], True, (60, 60, 60), 4)
    cv2.putText(image, "document", (835, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 3)

    # 画几个类似硬币的圆形，用来演示轮廓计数。
    coin_centers = [(160, 470), (270, 470), (380, 470), (490, 470), (600, 470)]
    for index, center in enumerate(coin_centers, start=1):
        cv2.circle(image, center, 45, (40, 180, 220), -1)
        cv2.circle(image, center, 45, (30, 120, 160), 3)
        cv2.putText(image, str(index), (center[0] - 13, center[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 60, 80), 3)

    # 画黑白文字和线条，用来观察灰度、阈值、边缘效果。
    cv2.putText(image, "OpenCV", (760, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (20, 20, 20), 6)
    cv2.putText(image, "BGR  Gray  Blur  Edge  Contour", (760, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.line(image, (760, 610), (1160, 650), (0, 0, 0), 4)

    # 加一些随机噪声点。模糊处理后，这些噪声会变得不明显。
    rng = np.random.default_rng(8)
    for _ in range(700):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        color = int(rng.integers(0, 255))
        image[y, x] = (color, color, color)

    return image


def load_source_image():
    """
    读取教学输入图像。

    优先级：
        1. 如果 USE_CAMERA=True，主循环会从摄像头读取画面。
        2. 如果 IMAGE_PATH 不为空，读取学生指定的图片。
        3. 如果没有指定图片，自动生成教学测试图。
    """
    if IMAGE_PATH:
        image_path = Path(IMAGE_PATH)
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"无法读取图片：{image_path}")
            print("请检查路径是否正确。现在改用程序自动生成的教学测试图。")
            return create_teaching_image(FRAME_WIDTH, FRAME_HEIGHT)

        return image

    return create_teaching_image(FRAME_WIDTH, FRAME_HEIGHT)


# =========================
# 3. 通用显示工具
# =========================
def to_bgr(image):
    """
    把图像统一转成 BGR 三通道。

    OpenCV 的 imshow 可以显示灰度图，也可以显示彩色图。
    但我们要把很多小图拼成一个大图，如果有的是单通道、有的是三通道，
    np.hstack / np.vstack 会因为维度不一致而报错。
    所以这里统一转成三通道。
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_tile(image, tile_width=256, tile_height=160):
    """把每个教学效果图缩放到统一大小，方便拼成九宫格。"""
    return cv2.resize(to_bgr(image), (tile_width, tile_height))


def put_tile_label(image, title, note):
    """
    给每个小图加标题和一句教学提示。

    这里使用英文标签，是因为 cv2.putText 对中文支持不好。
    中文解释写在代码注释和 README 里。
    """
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(result, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(result, note, (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 255), 1)
    return result


def make_tile(image, title, note):
    """生成一张带标题的小教学图。"""
    tile = resize_tile(image)
    return put_tile_label(tile, title, note)


# =========================
# 4. 各种 OpenCV 教学效果
# =========================
def demo_bgr_channels(image):
    """
    演示 BGR 三通道拆分。

    OpenCV 读入的彩色图不是 RGB，而是 BGR：
        B = Blue  蓝色通道
        G = Green 绿色通道
        R = Red   红色通道

    cv2.split(image) 可以把一张彩色图拆成三个单通道灰度图。
    通道越亮，说明这个位置的对应颜色成分越强。
    """
    blue, green, red = cv2.split(image)

    blue_view = cv2.merge([blue, np.zeros_like(blue), np.zeros_like(blue)])
    green_view = cv2.merge([np.zeros_like(green), green, np.zeros_like(green)])
    red_view = cv2.merge([np.zeros_like(red), np.zeros_like(red), red])

    top = np.hstack([
        cv2.resize(blue_view, (180, 220)),
        cv2.resize(green_view, (180, 220)),
    ])
    bottom = cv2.resize(red_view, (360, 220))
    return cv2.addWeighted(top, 0.55, bottom, 0.45, 0)


def demo_gray(image):
    """演示彩色图转灰度图。灰度图只有亮暗，没有颜色。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def demo_blur(image):
    """
    演示高斯模糊。

    模糊的作用：
        - 降低噪声
        - 让边缘检测更稳定
        - 但也会让细节变少
    """
    return cv2.GaussianBlur(image, (15, 15), 0)


def demo_canny(image):
    """
    演示 Canny 边缘检测。

    边缘一般出现在颜色或亮度变化很明显的地方。
    例如图形边界、文字边界、物体轮廓。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)
    return edges


def demo_threshold(image):
    """
    演示二值阈值分割。

    阈值分割会把灰度图变成黑白图：
        灰度值大于阈值 -> 白色
        灰度值小于阈值 -> 黑色

    这常用于把文字、形状、前景从背景中分离出来。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _threshold_value, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary


def demo_morphology(image):
    """
    演示形态学处理。

    形态学常用于处理二值图。
    本例使用“闭运算”：先膨胀再腐蚀。
    它可以连接一些小断裂，让目标区域更完整。
    """
    binary = demo_threshold(image)
    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def demo_contours(image):
    """
    演示轮廓检测。

    轮廓可以理解为“白色区域的边界”。
    常见应用：
        - 数硬币
        - 找纸张边缘
        - 找零件外形
        - 统计图像里的目标数量
    """
    binary = demo_morphology(image)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    valid_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # 面积太小的轮廓大多是噪声，所以跳过。
        if area < 1200:
            continue

        valid_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

    cv2.putText(result, f"count: {valid_count}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    return result


def demo_hsv_color_mask(image):
    """
    演示 HSV 颜色空间和颜色提取。

    BGR 适合显示图像，HSV 更适合按颜色筛选。
        H = Hue 色相，可以理解为“是什么颜色”
        S = Saturation 饱和度，可以理解为“颜色有多鲜艳”
        V = Value 明度，可以理解为“有多亮”

    本例提取绿色区域，所以绿色圆会保留下来，其它区域变暗。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def demo_histogram_equalization(image):
    """
    演示直方图均衡化。

    直方图均衡化会重新分配亮度，让暗处和亮处的层次更明显。
    本例只对灰度图做均衡化，方便观察亮度对比变化。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


def build_gallery(image):
    """
    把多个 OpenCV 处理结果拼成一个大画面。

    课堂讲解建议：
        先看左上角 Original，再按顺序比较每个处理结果和原图的差异。
    """
    tiles = [
        make_tile(image, "1 Original", "source image / camera frame"),
        make_tile(demo_bgr_channels(image), "2 BGR Channels", "OpenCV color order is BGR"),
        make_tile(demo_gray(image), "3 Gray", "remove color, keep brightness"),
        make_tile(demo_blur(image), "4 Gaussian Blur", "reduce noise, lose details"),
        make_tile(demo_canny(image), "5 Canny Edge", "find strong boundaries"),
        make_tile(demo_threshold(image), "6 Threshold", "turn gray into black/white"),
        make_tile(demo_morphology(image), "7 Morphology", "connect broken white regions"),
        make_tile(demo_contours(image), "8 Contours", "find object outlines"),
        make_tile(demo_hsv_color_mask(image), "9 HSV Color Mask", "extract green color"),
        make_tile(demo_histogram_equalization(image), "10 EqualizeHist", "enhance brightness contrast"),
    ]

    # 10 张图不好直接做九宫格，所以做成 2 行 x 5 列。
    first_row = np.hstack(tiles[:5])
    second_row = np.hstack(tiles[5:])
    gallery = np.vstack([first_row, second_row])

    footer = np.full((46, gallery.shape[1], 3), (35, 35, 35), dtype=np.uint8)
    cv2.putText(
        footer,
        "q: quit    s: save current gallery    set USE_CAMERA=True for webcam",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return np.vstack([gallery, footer])


def save_gallery(gallery):
    """保存当前实验台截图。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "opencv_teaching_gallery.jpg"
    cv2.imwrite(str(output_path), gallery)
    print(f"已保存实验台截图：{output_path}")


# =========================
# 5. 主程序入口
# =========================
def main():
    """
    主程序逻辑：
        - 如果 USE_CAMERA=True，循环读取摄像头画面并实时处理。
        - 如果 USE_CAMERA=False，只处理一张图片或自动生成的教学图。
    """
    cap = None

    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print("无法打开摄像头。请检查摄像头编号、系统相机权限，或是否被其它软件占用。")
            return

        print("摄像头模式已启动。按 q 退出，按 s 保存当前实验台截图。")
    else:
        print("图片模式已启动。程序会显示一张自动生成或指定图片的 OpenCV 实验台。")

    saved_once = False

    try:
        while True:
            if USE_CAMERA:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面。")
                    break
                frame = cv2.flip(frame, 1)
            else:
                frame = load_source_image()

            gallery = build_gallery(frame)

            if SAVE_GALLERY_IMAGE and not USE_CAMERA and not saved_once:
                save_gallery(gallery)
                saved_once = True

            cv2.imshow(WINDOW_NAME, gallery)
            key = cv2.waitKey(30 if USE_CAMERA else 0) & 0xFF

            if key == ord("s"):
                save_gallery(gallery)
            if key == ord("q") or not USE_CAMERA:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
