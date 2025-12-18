# modules/perception/map_reader.py

import cv2
import numpy as np
from config import MINIMAP_ROI_RATIO, NAV_COLOR_LOWER, NAV_COLOR_UPPER, ROAD_COLOR_LOWER, ROAD_COLOR_UPPER

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
        
        # 1. Detect Navigation Route (Purple)
        mask_nav = cv2.inRange(hsv, NAV_COLOR_LOWER, NAV_COLOR_UPPER)
        
        # 2. Detect Road (White/Grey)
        mask_road = cv2.inRange(hsv, ROAD_COLOR_LOWER, ROAD_COLOR_UPPER)

        kernel = np.ones((3,3), np.uint8)
        mask_nav = cv2.erode(mask_nav, kernel, iterations=1)
        mask_nav = cv2.dilate(mask_nav, kernel, iterations=2)
        
        mask_road = cv2.erode(mask_road, kernel, iterations=1)
        mask_road = cv2.dilate(mask_road, kernel, iterations=2)

        # --- Logic: Prioritize Navigation, Fallback to Road ---
        nav_error = 0
        target_x = car_center_x
        target_y = car_center_y
        
        # Check for Navigation Route first
        nonzero_nav = cv2.findNonZero(mask_nav)
        nonzero_road = cv2.findNonZero(mask_road)
        
        active_mask_points = None
        mode = "None"
        
        if nonzero_nav is not None:
            active_mask_points = nonzero_nav
            mode = "Nav (Purple)"
            # Visualization: Highlight Nav
            minimap[mask_nav > 0] = [0, 255, 0] # Green overlay for Nav
        elif nonzero_road is not None:
            active_mask_points = nonzero_road
            mode = "Road (White)"
            # Visualization: Highlight Road
            minimap[mask_road > 0] = [0, 100, 255] # Orange overlay for Road

        # Visual Debug Elements
        cv2.line(minimap, (car_center_x, 0), (car_center_x, map_h), (255, 0, 0), 2) # Vertical Center
        cv2.line(minimap, (0, car_center_y), (map_w, car_center_y), (255, 0, 0), 2) # Horizontal Center

        if active_mask_points is not None:
            # Calculate centroid of the path
            M = cv2.moments(active_mask_points)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                target_x = cx
                target_y = cy

                # Calculate vector from Car to Target
                # Car is at (car_center_x, car_center_y)
                # Target is at (cx, cy)
                
                # Coordinate transformation to Car Frame:
                # X axis: Right (Positive)
                # Y axis: Forward (Positive) -> Image Up is Negative Y, so we invert dy
                dx = target_x - car_center_x
                dy = car_center_y - target_y # Invert Y so Up is Positive
                
                # Calculate Angle in degrees
                # 0 deg = Forward
                # 90 deg = Right
                # -90 deg = Left
                # 180 deg = Behind
                angle = np.degrees(np.arctan2(dx, dy))
                
                # Map Angle to Steering Offset
                # Heuristic: 1 degree ~= 4 pixels offset
                # 90 degrees -> 360 offset (Strong Turn)
                nav_error = angle * 5.0 
                
                # Visualization
                cv2.circle(minimap, (cx, cy), 8, (0, 255, 255), -1) # Yellow Target
                cv2.line(minimap, (car_center_x, car_center_y), (cx, cy), (0, 255, 255), 2) # Yellow Vector
                
                # Display Angle
                cv2.putText(minimap, f"{mode}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(minimap, f"Ang: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return nav_error, minimap