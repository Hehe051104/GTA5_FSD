# tune_minimap.py
# 这是一个专门用来校准小地图颜色和区域的工具

import cv2
import numpy as np
from utils.screen_grab import ScreenGrabber
from config import MINIMAP_ROI_RATIO

def nothing(x):
    pass

def main():
    print("🎛️ 小地图校准工具启动...")
    print("1. 拖动滑动条，直到只有【紫色导航线】变成白色，背景全黑。")
    print("2. 尤其注意要把蓝色的护甲条、绿色的生命条过滤掉。")
    print("3. 满意后按 'q' 退出，复制控制台打印的代码。")

    grabber = ScreenGrabber()
    cv2.namedWindow("Minimap Tuner")

    # 初始值 (根据你现在的 config)
    cv2.createTrackbar("H Min", "Minimap Tuner", 125, 179, nothing)
    cv2.createTrackbar("S Min", "Minimap Tuner", 50, 255, nothing)
    cv2.createTrackbar("V Min", "Minimap Tuner", 100, 255, nothing)

    cv2.createTrackbar("H Max", "Minimap Tuner", 155, 179, nothing)
    cv2.createTrackbar("S Max", "Minimap Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Max", "Minimap Tuner", 255, 255, nothing)

    while True:
        frame = grabber.get_frame()
        if frame is None: continue

        h, w = frame.shape[:2]

        # 截取小地图
        y1 = int(h * MINIMAP_ROI_RATIO[0])
        y2 = int(h * MINIMAP_ROI_RATIO[1])
        x1 = int(w * MINIMAP_ROI_RATIO[2])
        x2 = int(w * MINIMAP_ROI_RATIO[3])
        minimap = frame[y1:y2, x1:x2]

        # 转 HSV
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # 获取滑动条的值
        h_min = cv2.getTrackbarPos("H Min", "Minimap Tuner")
        s_min = cv2.getTrackbarPos("S Min", "Minimap Tuner")
        v_min = cv2.getTrackbarPos("V Min", "Minimap Tuner")
        h_max = cv2.getTrackbarPos("H Max", "Minimap Tuner")
        s_max = cv2.getTrackbarPos("S Max", "Minimap Tuner")
        v_max = cv2.getTrackbarPos("V Max", "Minimap Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # 生成掩膜 (Mask)
        mask = cv2.inRange(hsv, lower, upper)

        # 显示效果：原图 | 掩膜(黑白) | 提取结果(彩色)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(minimap, minimap, mask=mask)

        # 拼接显示，方便对比
        # 如果屏幕太小放不下，可以只显示 mask_bgr
        stacked = np.hstack((minimap, mask_bgr, result))

        cv2.imshow("Minimap Tuner", stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n" + "="*40)
            print("✅ 最佳参数如下 (请复制到 config.py):")
            print(f"NAV_COLOR_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"NAV_COLOR_UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            print("="*40 + "\n")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()