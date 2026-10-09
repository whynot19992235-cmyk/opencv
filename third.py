import cv2              # 导入 OpenCV 库
import numpy as np      # 用来把文件读成字节数组（解决中文路径问题）
import tkinter as tk    # 导入 Tkinter（Python 自带的 GUI 库）
from tkinter import filedialog  # 导入文件选择对话框

root = tk.Tk()      # 创建一个窗口
root.withdraw()     # 隐藏窗口，只保留后面的文件选择框

path = filedialog.askopenfilename(          # 弹出窗口让用户选择图片
    filetypes=[('图片文件', '*.jpg;*.jpeg;*.png'), ('所有文件', '*.*')]
)

if not path:                     # 用户点了“取消”时，path 是空字符串
    print('未选择图片，程序退出')   # 打印提示信息
    exit()                       # 正常结束程序

data = np.fromfile(path, dtype=np.uint8)   # 用二进制方式读取文件，支持中文路径
img = cv2.imdecode(data, cv2.IMREAD_COLOR) # 把字节数据解码成图片

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 把彩色图片转换成灰度图

cv2.imshow('我的图片', img)   # 显示图片
cv2.imshow('灰度图', gray)    # 再打开一个窗口显示灰度图
cv2.waitKey(0)               # 等待按键，按任意键继续
cv2.destroyAllWindows()      # 关闭所有窗口