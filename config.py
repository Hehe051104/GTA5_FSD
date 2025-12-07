# config.py

import numpy as np

# ================= Basic Window Config =================
GTA_WINDOW_TITLE = "Grand Theft Auto V"
DEBUG_MODE = True
WINDOW_OFFSET = {
    'border_top': 30,
    'border_side': 8
}

# ================= Resolution Config =================
LANE_VIEW_WIDTH = 1280
LANE_VIEW_HEIGHT = 720
VIEW_WIDTH = LANE_VIEW_WIDTH
VIEW_HEIGHT = LANE_VIEW_HEIGHT

# ================= Lane Detection ROI (Trapezoid) =================
ROI_POINTS_SRC = np.float32([
    [0.10, 1.00],  # Bottom Left
    [0.90, 1.00],  # Bottom Right
    [0.56, 0.46],  # Top Right
    [0.44, 0.46]   # Top Left
])

<<<<<<< HEAD
# 单车道检测时的默认偏移量 (当只检测到一侧车道线时，假设另一侧的距离)
SINGLE_LANE_OFFSET = 250

# 之前的矩形配置作废，保留此变量防止报错，但逻辑会改用上面的 POLY
CAR_HOOD_MASK_RATIO = [0.0, 0.0, 0.0, 0.0]
=======
# ================= Car Body Mask (Trapezoid) =================
CAR_MASK_POLY = np.float32([
    [0.35, 0.55],  # Top Left
    [0.65, 0.55],  # Top Right
    [0.95, 1.00],  # Bottom Right
    [0.05, 1.00]   # Bottom Left
])
# Legacy support
CAR_HOOD_MASK_RATIO = [0.60, 1.00, 0.20, 0.80]

# ================= Minimap Config (Calibrated) =================
# ROI [Top, Bottom, Left, Right]
MINIMAP_ROI_RATIO = [0.815, 1.045, 0.001, 0.171]

# Purple Navigation Line HSV Thresholds
NAV_COLOR_LOWER = np.array([125, 50, 100])
NAV_COLOR_UPPER = np.array([155, 255, 255])

# Fusion Weights: 70% Map (Direction), 30% Lane (Centering)
W_MAP = 0.7
W_LANE = 0.3

# ================= PID Control Config =================
PID_Kp = 0.45      # Proportional
PID_Ki = 0.002     # Integral
PID_Kd = 0.15      # Derivative
PID_DEADBAND = 20  # Deadband (pixels)
MAX_KEY_DURATION = 0.3

# ================= Smoothing Config =================
CENTER_THRESHOLD = 50
SMOOTH_WINDOW = 5
>>>>>>> ace3bcae962f6d66a4e118fb5839768d5f6681f4
