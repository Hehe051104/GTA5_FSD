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

        if y1 >= y2 or x1 >= x2: return 0, False, "None", None

        minimap = frame[y1:y2, x1:x2]
        map_h, map_w = minimap.shape[:2]

        car_center_x = map_w // 2
        # Adjust car center Y to align with the player arrow
        # Moved up to 0.50 (Center) based on user feedback
        car_center_y = int(map_h * 0.50)

        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        # 1. Detect Navigation Route (Purple)
        mask_nav_full = cv2.inRange(hsv, NAV_COLOR_LOWER, NAV_COLOR_UPPER)
        
        # --- Check if Car is ON Route ---
        # Check a small radius around the car center to see if it overlaps with the purple line
        mask_car_check = np.zeros_like(mask_nav_full)
        cv2.circle(mask_car_check, (car_center_x, car_center_y), 6, 255, -1) # 6px radius
        on_route_overlap = cv2.bitwise_and(mask_nav_full, mask_car_check)
        is_on_route = cv2.countNonZero(on_route_overlap) > 0
        
        # --- Optimization: Smart Turn (Forward Priority) ---
        # Strategy: "Go Straight First, Then Turn"
        # 1. Check a narrow strip directly ahead of the car.
        # 2. If path exists there, follow it (keep straight).
        # 3. If not, look for the turn using the Pure Pursuit Ring.
        
        pursuit_radius = int(min(map_h, map_w) * 0.20)
        
        # A. Check Forward Path
        forward_width = 8 # +/- 8 pixels (Total 16px width)
        mask_forward = np.zeros_like(mask_nav_full)
        # Rectangle from car center upwards to pursuit radius
        cv2.rectangle(mask_forward, 
                      (car_center_x - forward_width, car_center_y - pursuit_radius),
                      (car_center_x + forward_width, car_center_y),
                      255, -1)
        
        mask_forward_check = cv2.bitwise_and(mask_nav_full, mask_forward)
        forward_pixels = cv2.countNonZero(mask_forward_check)
        
        nav_error = None
        active_mask_points = None
        mode = "None"
        target_x = car_center_x
        target_y = car_center_y
        
        # Threshold: If enough purple pixels are directly ahead
        if forward_pixels > 20:
             # --- Mode: Forward Priority ---
             active_mask_points = cv2.findNonZero(mask_forward_check)
             mode = "Nav (Forward)"
             minimap[mask_forward_check > 0] = [0, 255, 0] # Green for Forward
             
             # Visual Debug: Show Forward ROI
             cv2.rectangle(minimap, 
                      (car_center_x - forward_width, car_center_y - pursuit_radius),
                      (car_center_x + forward_width, car_center_y),
                      (100, 255, 100), 1)
        else:
             # --- Mode: Turn (Pure Pursuit) ---
             # No path ahead, so we look for the turn on the ring
             mask_ring = np.zeros_like(mask_nav_full)
             cv2.circle(mask_ring, (car_center_x, car_center_y), pursuit_radius, 255, thickness=4)
             cv2.rectangle(mask_ring, (0, car_center_y + 5), (map_w, map_h), 0, -1) # Upper half only
             
             intersect_nav = cv2.bitwise_and(mask_nav_full, mask_ring)
             points_pursuit = cv2.findNonZero(intersect_nav)
             
             if points_pursuit is not None:
                 active_mask_points = points_pursuit
                 mode = "Nav (Turn)"
                 minimap[intersect_nav > 0] = [0, 255, 255] # Yellow for Turn Target
             else:
                 # --- Mode: Fallback (Close Range) ---
                 fallback_radius = int(min(map_h, map_w) * 0.12)
                 mask_roi = np.zeros_like(mask_nav_full)
                 cv2.circle(mask_roi, (car_center_x, car_center_y), fallback_radius, 255, -1)
                 mask_nav_fallback = cv2.bitwise_and(mask_nav_full, mask_roi)
                 
                 points_fallback = cv2.findNonZero(mask_nav_fallback)
                 if points_fallback is not None:
                     active_mask_points = points_fallback
                     mode = "Nav (Fallback)"
                     minimap[mask_nav_fallback > 0] = [0, 0, 255] # Red for Fallback

        # Visual Debug Elements
        cv2.line(minimap, (car_center_x, 0), (car_center_x, map_h), (255, 0, 0), 1) # Vertical Center
        cv2.line(minimap, (0, car_center_y), (map_w, car_center_y), (255, 0, 0), 1) # Horizontal Center
        cv2.circle(minimap, (car_center_x, car_center_y), pursuit_radius, (200, 200, 200), 1) # Pursuit Circle

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
        
        # Visual feedback for "On Route" status
        status_color = (0, 255, 0) if is_on_route else (0, 0, 255)
        cv2.circle(minimap, (car_center_x, car_center_y), 4, status_color, -1)

        return nav_error, is_on_route, mode, minimap