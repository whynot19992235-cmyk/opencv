import cv2
from tkinter import filedialog, Tk

# 创建文件选择窗口，但不显示主窗口
root = Tk()
root.withdraw()

# 让你自己选择图片
path = filedialog.askopenfilename(
    title="请选择一张图片",
    filetypes=[
        ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
        ("所有文件", "*.*")
    ]
)

# 读取你选择的图片
img = cv2.imread(path)

# 显示图片
cv2.imshow("我的图片", img)

# 等待按键
cv2.waitKey(0)

# 关闭窗口
cv2.destroyAllWindows()