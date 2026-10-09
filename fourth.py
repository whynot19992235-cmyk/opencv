import cv2  # 导入 OpenCV 库

cap = cv2.VideoCapture(int(input("你要调用什么摄像头：")))  # 打开摄像头，参数 0 表示第一个摄像头设备

while True:                      # 不断重复地获取画面
    ret, frame = cap.read()      # 读取一帧画面：ret 表示是否成功，frame 是画面数据
    if not ret:                  # 如果读取失败（例如摄像头被占用或拔掉）
        break                    # 跳出循环
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # 转成灰度图
    blur = cv2.GaussianBlur(gray, (5, 5), 0)             # 高斯模糊：去掉相机噪点，边缘更干净
    edges = cv2.Canny(blur, 60, 180)                     # 边缘检测：找明暗交界线，阈值调高抗背景纹理
    edges = cv2.dilate(edges, None, iterations=2)        # 膨胀：把断开的边缘连成闭合的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出所有轮廓

    for c in contours:                                    # 逐个检查每个轮廓
        area = cv2.contourArea(c)                         # 算轮廓的面积
        if area < 800:                                    # 面积太小的多半是噪点，跳过
            continue
        x, y, w, h = cv2.boundingRect(c)                  # 取轮廓的外接矩形
        if x < 2 or y < 2 or x + w > frame.shape[1] - 2 or y + h > frame.shape[0] - 2:
            continue                                      # 碰到画面边缘的轮廓不完整，跳过减少误判
        peri = cv2.arcLength(c, True)                     # 算轮廓的周长
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)   # 把轮廓近似成多边形
        sides = len(approx)                               # 看多边形有几个顶点
        if sides == 3:                                    # 3 个顶点 → 三角形
            name = '三角形'
        elif sides == 4:                                  # 4 个顶点 → 矩形
            name = '矩形'
        else:                                             # 顶点更多 → 圆形
            name = '圆形'
        cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)  # 用蓝色线把轮廓画在原图上
        x, y = approx[0][0]                               # 取第一个顶点的坐标
        cv2.putText(frame, name, (x, y - 10),              # 在轮廓旁边写上形状的名字
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # (0,0,255)是红色，方便看清
    cv2.imshow('摄像头画面', frame)   # 在窗口里显示这一帧
    if cv2.waitKey(1) & 0xFF == 27:  # 等 1 毫秒刷新画面；按 ESC(键码27) 退出
        break                    # 跳出循环

cap.release()            # 释放摄像头资源.
cv2.destroyAllWindows()  # 关闭显示窗口

