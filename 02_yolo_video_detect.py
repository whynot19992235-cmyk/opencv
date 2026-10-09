"""
YOLO 动态视频目标识别教程

你将学到什么：
1. 视频本质上是很多张连续播放的图片，每一张图片叫一帧。
2. 程序会循环读取视频的每一帧，并把每一帧交给 YOLO 检测。
3. 检测完成后，程序会把带识别框的新帧写入一个新视频文件。

运行环境：
1. 建议使用 Python 3.10、3.11 或 3.12。
2. 需要安装第三方库：ultralytics、opencv-python。
3. 安装命令：
       python3 -m pip install ultralytics opencv-python
   或者在本目录运行：
       python3 -m pip install -r requirements.txt
4. 如果你希望输出视频保留原视频声音，还需要安装 ffmpeg。
   ffmpeg 是一个常用的视频/音频处理工具，这里用它把“检测后的视频画面”和“原视频声音”合并起来。

   Mac 安装方式：
   - 如果你已经安装了 Homebrew，可以运行：
       brew install ffmpeg
   - Homebrew 是 Mac 上常用的软件包管理工具，可以理解为“用命令安装软件的工具”。
   - 如果你的电脑没有 brew，可以先搜索并安装 Homebrew，或者请有经验的人协助安装。

   Windows 安装方式：
   - 方法 1：如果你的 Windows 支持 winget，可以在 PowerShell 中运行：
       winget install Gyan.FFmpeg
   - 方法 2：下载 ffmpeg 的 Windows 版本，解压后把 bin 目录添加到系统 PATH。
   - 安装完成后，重新打开终端，运行 ffmpeg -version。如果能看到版本号，说明安装成功。

使用方法：
1. 先把要识别的视频放到本目录，例如 street.mp4。
2. 修改下面“配置区”里的 VIDEO_PATH。
3. 运行：
       python3 02_yolo_video_detect.py

运行示例：
    python3 02_yolo_video_detect.py

窗口打开后，按 q 可以提前结束实时预览。
"""

from pathlib import Path
import shutil
import subprocess
import time

import cv2
from ultralytics import YOLO


# =========================
# 配置区：初学者主要修改这里
# =========================

# 要识别的视频路径。
# 你可以把视频放在 yolo 目录下，然后把这里改成 "./你的视频名.mp4"。
VIDEO_PATH = "./messi.mp4"

# YOLO 模型名称。
# yolov8n.pt 速度较快，适合先跑通视频识别流程。
MODEL_PATH = "yolov8n.pt"

# 置信度阈值。
# 数字越大，检测结果越严格；数字越小，可能显示更多检测框。
CONFIDENCE_THRESHOLD = 0.35

# 识别后的视频保存路径。
OUTPUT_PATH = "./outputs/video_result.mp4"

# 是否一边识别一边弹窗预览。
# True 表示显示预览窗口；False 表示只保存输出视频，不弹窗。
SHOW_WINDOW = False

# 是否保留原视频的音频。
# OpenCV 的 VideoWriter 只能写入画面，不能直接写入声音。
# 所以这里会先生成一个无声的检测结果视频，再用 ffmpeg 把原视频声音合并回来。
PRESERVE_AUDIO = True

# 是否保留中间生成的无声视频。
# False 表示音频合并成功后自动删除中间文件，让输出目录更干净。
KEEP_SILENT_VIDEO = False


def merge_original_audio(original_video_path, silent_video_path, final_video_path):
    """
    把原视频的音频合并到 YOLO 处理后的视频中。

    你可以把这个函数理解为：
    1. silent_video_path 提供“已经画好检测框的画面”。
    2. original_video_path 提供“原视频里的声音”。
    3. ffmpeg 把两者合并成 final_video_path。
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("\n没有找到 ffmpeg，无法自动合并音频。")
        print("当前已生成无声视频：", silent_video_path.resolve())
        print("如需保留音频，请先安装 ffmpeg。")
        print("Mac：如果已经安装 Homebrew，可以运行 brew install ffmpeg。")
        print("Windows：可以在 PowerShell 运行 winget install Gyan.FFmpeg，或手动安装后加入 PATH。")
        print("安装后重新打开终端，运行 ffmpeg -version 检查是否成功。")
        return False

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(original_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_video_path),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        print("\nffmpeg 合并音频失败，当前保留无声视频。")
        print("无声视频路径：", silent_video_path.resolve())
        print("错误信息：")
        print(error.stderr.decode("utf-8", errors="ignore"))
        return False

    return True


def main():
    video_path = Path(VIDEO_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果需要保留音频，OpenCV 先写入临时无声视频，最后再用 ffmpeg 合成最终视频。
    # 如果不需要保留音频，OpenCV 直接写入 OUTPUT_PATH。
    if PRESERVE_AUDIO:
        silent_output_path = output_path.with_name(
            f"{output_path.stem}_silent{output_path.suffix}"
        )
    else:
        silent_output_path = output_path

    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")

    # 1. 加载 YOLO 模型。
    # 第一次使用 yolov8n.pt 时，ultralytics 会自动从网络下载这个模型文件。
    model = YOLO(MODEL_PATH)

    # 2. 打开视频文件。
    # VideoCapture 是 OpenCV 提供的视频读取工具，既可以读取本地视频，也可以读取摄像头。
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"视频打开失败，请检查文件格式：{video_path}")

    # 3. 读取原视频的基本属性，保证输出视频尺寸和帧率与原视频一致。
    # fps 表示每秒播放多少帧；width 和 height 表示视频画面的宽和高。
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 有些视频文件可能读不到 fps，这里给一个保底值，避免后面创建输出视频失败。
    if fps <= 0:
        fps = 25

    # 4. 创建视频写入器。
    # VideoWriter 负责把每一帧标注后的图片重新组合成一个新视频。
    # mp4v 是常见的 mp4 编码方式，大多数电脑都可以播放。
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent_output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"输出视频创建失败：{silent_output_path}")

    frame_index = 0
    start_time = time.time()

    print("开始视频识别。按 q 可以提前结束预览窗口。")

    while True:
        # 5. 读取一帧。
        # ret=True 表示成功读到画面；ret=False 表示视频已经结束或读取失败。
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # 6. 对当前帧进行 YOLO 检测。
        # 这里和单张图片识别非常像，因为视频的一帧本质上就是一张图片。
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]

        # 7. 把检测框画回当前帧。
        # result.plot() 返回的是已经画好检测框、类别名和置信度的画面。
        annotated_frame = result.plot()

        # 8. 写入输出视频。
        # 每处理完一帧，就把这一帧追加到输出视频中。
        writer.write(annotated_frame)

        # 9. 打印进度。
        # 通过帧数和速度，你可以观察电脑处理视频识别任务的快慢。
        if frame_index == 1 or frame_index % 30 == 0:
            elapsed = time.time() - start_time
            speed = frame_index / elapsed if elapsed > 0 else 0
            print(
                f"已处理 {frame_index}/{total_frames or '?'} 帧，"
                f"平均速度 {speed:.1f} 帧/秒"
            )

        # 10. 可选实时预览。
        # 如果 SHOW_WINDOW=True，就会看到正在生成的检测画面；按 q 可以提前退出。
        if SHOW_WINDOW:
            cv2.imshow("YOLO Video Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("你已提前结束识别。")
                break

    # 11. 释放资源。
    # 程序结束前要关闭视频文件和窗口，否则输出视频可能没有正确写完。
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if PRESERVE_AUDIO:
        print("\n画面识别完成，正在把原视频音频合并到结果视频中...")
        audio_merged = merge_original_audio(video_path, silent_output_path, output_path)

        if audio_merged:
            if not KEEP_SILENT_VIDEO and silent_output_path != output_path:
                silent_output_path.unlink(missing_ok=True)
            print(f"视频识别完成，带音频结果已保存：{output_path.resolve()}")
        else:
            print(f"视频识别完成，但结果不含音频：{silent_output_path.resolve()}")
    else:
        print(f"\n视频识别完成，结果已保存：{output_path.resolve()}")


if __name__ == "__main__":
    main()
