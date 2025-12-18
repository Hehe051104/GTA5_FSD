import cv2
import numpy as np
import mss
import win32gui
import ctypes
from config import GTA_WINDOW_TITLE, WINDOW_OFFSET

# [重要] 告诉 Windows 我们需要真实的物理坐标 (解决 DPI 缩放导致的偏移/截取不全问题)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

class ScreenGrabber:
    def __init__(self):
        # 初始化截屏对象
        self.sct = mss.mss()
        self.hwnd = None # 缓存窗口句柄

    def get_frame(self):
        """
        获取屏幕帧，经过优化，缓存句柄
        """
        try:
            # 1. 如果没有缓存句柄，或者句柄失效，则重新查找
            if self.hwnd is None or not win32gui.IsWindow(self.hwnd):
                self.hwnd = win32gui.FindWindow(None, GTA_WINDOW_TITLE)
                if not self.hwnd:
                    return None

            # 2. [核心逻辑] 实时获取窗口的最新坐标 (GetWindowRect 非常快)
            # rect 返回的是 (left, top, right, bottom)
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y, r, b = rect

            # 计算宽度和高度
            w = r - x
            h = b - y

            # 3. 过滤无效状态 (比如窗口最小化时，坐标通常是负数或极小)
            if x < -1000 or w < 10 or h < 10:
                # print("窗口最小化中...")
                return None

            # 4. 构建 mss 需要的抓取区域 (应用 config.py 里的去边框偏移)
            # 这一步是动态的，x 和 y 变了，monitor 就变了
            monitor = {
                "top": y + WINDOW_OFFSET['border_top'],
                "left": x + WINDOW_OFFSET['border_side'],
                "width": w - (WINDOW_OFFSET['border_side'] * 2),
                "height": h - WINDOW_OFFSET['border_top'] - WINDOW_OFFSET['border_side']
            }

            # 5. 抓取这个动态区域
            screenshot = self.sct.grab(monitor)

            # 6. 转成 OpenCV 图像
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return img

        except Exception as e:
            # 比如拖动窗口的过程中可能会有一瞬间抓取失败，容错处理
            # print(f"抓取错位: {e}")
            return None