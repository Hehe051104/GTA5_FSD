# modules/perception/map_reader.py

import cv2
import numpy as np
from config import MINIMAP_ROI_RATIO, NAV_COLOR_LOWER, NAV_COLOR_UPPER

class MapReader:
    def __init__(self):
        pass

    def process(self, frame):
        h, w = frame.shape[:2]

        y1 = int(h * MINIMAP_ROI_RATIO[0])
        y2 = int(h * MINIMAP_ROI_RATIO[1])
        x1 = int(w * MINIMAP_ROI_RATIO[2])
        x2 = int(w * MINIMAP_ROI_RATIO[3])

        if y1 >= y2 or x1 >= x2: return 0, None

        minimap = frame[y1:y2, x1:x2]
        map_h, map_w = minimap.shape[:2]

        car_center_x = map_w // 2
        car_center_y = map_h // 2

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, NAV_COLOR_LOWER, NAV_COLOR_UPPER)

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Scan-Line Logic
        scan_line_y = int(map_h * 0.55)
        scan_strip = mask[scan_line_y : scan_line_y+5, :]
        nonzero = np.where(scan_strip > 0)
        nonzero_x = nonzero[1]

        nav_error = 0

        # Visual Debug Elements
        cv2.line(minimap, (car_center_x, 0), (car_center_x, map_h), (255, 0, 0), 2)
        cv2.line(minimap, (0, scan_line_y), (map_w, scan_line_y), (100, 100, 100), 1)

        if len(nonzero_x) > 0:
            target_x = int(np.mean(nonzero_x))

            # Direction Logic: (Center - Target)
            # Left turn = Positive, Right turn = Negative
            nav_error = (car_center_x - target_x) * 3.0

            cv2.circle(minimap, (target_x, scan_line_y), 6, (0, 255, 0), -1)
            cv2.line(minimap, (car_center_x, map_h), (target_x, scan_line_y), (0, 255, 255), 2)

        return nav_error, minimap