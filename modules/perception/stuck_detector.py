import cv2
import numpy as np
import time

class StuckDetector:
    def __init__(self, check_interval=0.5, stuck_threshold=2.0):
        self.last_frame = None
        self.last_check_time = time.time()
        self.check_interval = check_interval  # Check every 0.5 seconds
        
        self.stuck_start_time = None
        self.stuck_threshold = stuck_threshold # Must be stuck for 2 seconds to trigger
        self.is_stuck = False
        
        # Recovery logic (Reverse)
        self.recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_duration = 1.5 # Reverse for 1.5 seconds

    def update(self, frame, current_key):
        """
        Returns: (is_stuck, is_recovering)
        """
        current_time = time.time()
        
        # If currently recovering (reversing)
        if self.recovery_mode:
            if current_time - self.recovery_start_time > self.recovery_duration:
                self.recovery_mode = False # Recovery done
                self.stuck_start_time = None # Reset stuck timer
                return False, False
            return True, True # Still recovering

        # Only check for stuck if we are pressing 'W' (Forward)
        if current_key != 'W':
            self.stuck_start_time = None
            return False, False

        # Limit check frequency to save CPU
        if current_time - self.last_check_time < self.check_interval:
            return False, False

        # 1. Image Preprocessing (Resize + Grayscale)
        # Downscaling significantly reduces computation and ignores minor jitters
        small_frame = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Image Differencing
        if self.last_frame is not None:
            # Calculate absolute difference between current and last frame
            diff = cv2.absdiff(gray, self.last_frame)
            
            # Threshold: only pixels with enough change count as "motion"
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            
            # Calculate ratio of non-zero pixels (motion amount)
            non_zero_count = cv2.countNonZero(thresh)
            total_pixels = gray.shape[0] * gray.shape[1]
            motion_ratio = non_zero_count / total_pixels
            
            # 3. Decision Logic
            # If motion is extremely low (e.g., < 1%), the screen is static
            if motion_ratio < 0.01: 
                if self.stuck_start_time is None:
                    self.stuck_start_time = current_time
                elif current_time - self.stuck_start_time > self.stuck_threshold:
                    # Confirmed stuck! Trigger recovery
                    self.recovery_mode = True
                    self.recovery_start_time = current_time
                    print("⚠️ Stuck detected! Initiating recovery maneuver...")
                    return True, True
            else:
                # Screen is moving, reset timer
                self.stuck_start_time = None
        
        self.last_frame = gray
        self.last_check_time = current_time
        
        return False, False
