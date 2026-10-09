import cv2

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("摄像头读取失败")
        break

    # 1. 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊，减少噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 二值化
    _, binary = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # 4. 提取轮廓
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = frame.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

    # 5. 增加 HSV 黄色识别：生成掩码（黄色区域=白，其他=黑）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = (20, 80, 80)
    upper_yellow = (40, 255, 255)

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    # 6. 提取黄色区域的轮廓
    yellow_contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 7. 在原摄像头画面上用蓝色线标出黄色区域的轮廓
    cv2.drawContours(
        result,
        yellow_contours,
        -1,
        (255, 0, 0),
        2
    )

    cv2.imshow("摄像头", result)
    cv2.imshow("二值图", binary)
    cv2.imshow("黄色区域", yellow_mask)

    

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break







cap.release()
cv2.destroyAllWindows()