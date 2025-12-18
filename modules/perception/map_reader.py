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
        # Adjust car center Y to align with the player arrow (slightly lower than image center)
        car_center_y = int(map_h * 0.58)

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        # 1. Detect Navigation Route (Purple)
        mask_nav = cv2.inRange(hsv, NAV_COLOR_LOWER, NAV_COLOR_UPPER)
        
        # --- Optimization: Limit Lookahead (Circular ROI) ---
        # Only consider the path within a certain radius of the car to avoid distant turns affecting steering
        # Radius set to 25% of the smaller dimension (Reduced from 35% to focus on immediate path)
        lookahead_radius = int(min(map_h, map_w) * 0.25) 
        mask_roi = np.zeros_like(mask_nav)
        cv2.circle(mask_roi, (car_center_x, car_center_y), lookahead_radius, 255, -1)
        mask_nav = cv2.bitwise_and(mask_nav, mask_roi)
        
        kernel = np.ones((3,3), np.uint8)
        mask_nav = cv2.erode(mask_nav, kernel, iterations=1)
        mask_nav = cv2.dilate(mask_nav, kernel, iterations=2)
        
        # --- Logic: Only Navigation ---
        nav_error = None
        target_x = car_center_x
        target_y = car_center_y
        
        # Check for Navigation Route
        nonzero_nav = cv2.findNonZero(mask_nav)
        
        active_mask_points = None
        mode = "None"
        
        if nonzero_nav is not None:
            active_mask_points = nonzero_nav
            mode = "Nav (Purple)"
            # Visualization: Highlight Nav
            minimap[mask_nav > 0] = [0, 255, 0] # Green overlay for Nav

        # Visual Debug Elements
        cv2.line(minimap, (car_center_x, 0), (car_center_x, map_h), (255, 0, 0), 1) # Vertical Center
        cv2.line(minimap, (0, car_center_y), (map_w, car_center_y), (255, 0, 0), 1) # Horizontal Center
        cv2.circle(minimap, (car_center_x, car_center_y), lookahead_radius, (200, 200, 200), 1) # Lookahead Limit Circle

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