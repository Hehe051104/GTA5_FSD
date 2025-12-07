# modules/perception/lane_detector.py

import cv2
import numpy as np
from collections import deque
from config import ROI_POINTS_SRC, CAR_MASK_POLY, CENTER_THRESHOLD, SMOOTH_WINDOW

class LaneDetector:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        # Using 640x360 as in your good version
        self.proc_w = 640
        self.proc_h = 360

        self.src_pts = np.float32(ROI_POINTS_SRC) * np.float32([self.proc_w, self.proc_h])
        offset = self.proc_w * 0.25
        self.dst_pts = np.float32([
            [offset, self.proc_h],
            [self.proc_w-offset, self.proc_h],
            [self.proc_w-offset, 0],
            [offset, 0]
        ])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.Minv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

        self.nwindows = 9
        self.margin = 50
        self.minpix = 25

        self.best_left_fit = None
        self.best_right_fit = None
        self.lane_width = 300
        self.offset_history = deque(maxlen=SMOOTH_WINDOW)

    def process(self, frame, blocking_boxes=None):
        masked_frame = frame.copy()
        if blocking_boxes:
            for box in blocking_boxes:
                x1, y1, x2, y2 = box
                box_w = x2 - x1
                safe_padding = int(box_w * 0.1)
                cv2.rectangle(masked_frame, (x1 + safe_padding, y1), (x2 - safe_padding, y2), (0, 0, 0), -1)

        masked_frame = self._apply_car_mask(masked_frame)

        frame_small = cv2.resize(masked_frame, (self.proc_w, self.proc_h))
        warped_img = cv2.warpPerspective(frame_small, self.M, (self.proc_w, self.proc_h))

        binary_warped = self._combined_threshold(warped_img)
        leftx, lefty, rightx, righty = self._find_lane_pixels(binary_warped)

        left_found = len(leftx) > 50
        right_found = len(rightx) > 50

        lane_info = {'offset': 0, 'status': 'No Lane'}
        result_frame = frame

        left_fit = None
        right_fit = None

        if left_found:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.best_left_fit = left_fit
        if right_found:
            right_fit = np.polyfit(righty, rightx, 2)
            self.best_right_fit = right_fit

        if not left_found and self.best_left_fit is not None:
            left_fit = self.best_left_fit
            left_found = True
        if not right_found and self.best_right_fit is not None:
            right_fit = self.best_right_fit
            right_found = True

        if left_found or right_found:
            lane_info = self._calculate_position(left_fit, right_fit, left_found, right_found)
            result_frame = self._draw_lane_lines(frame, binary_warped, left_fit, right_fit, left_found, right_found)
        else:
            self.offset_history.clear()

        # RETURN 3 VALUES: frame, info, binary_debug
        return result_frame, lane_info, binary_warped

    def _combined_threshold(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= 200) & (l_channel <= 255)] = 1

        combined = np.zeros_like(sxbinary)
        combined[(s_binary == 1) | (l_binary == 1) | (sxbinary == 1)] = 255
        return combined

    def _find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)

        dead_zone_w = int(self.proc_w * 0.15)
        histogram[midpoint - dead_zone_w : midpoint + dead_zone_w] = 0

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if histogram[leftx_base] < 5: leftx_base = -1
        if histogram[rightx_base] < 5: rightx_base = -1

        window_height = int(binary_warped.shape[0] // self.nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        if leftx_base != -1:
            leftx_current = leftx_base
            for window in range(self.nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - self.margin
                win_xleft_high = leftx_current + self.margin
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                if len(good_left_inds) > self.minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))

        if rightx_base != -1:
            rightx_current = rightx_base
            for window in range(self.nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xright_low = rightx_current - self.margin
                win_xright_high = rightx_current + self.margin
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                if len(good_right_inds) > self.minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))

        leftx, lefty, rightx, righty = [], [], [], []
        if len(left_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
        if len(right_lane_inds) > 0:
            right_lane_inds = np.concatenate(right_lane_inds)
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty

    def _calculate_position(self, left_fit, right_fit, left_found, right_found):
        y_eval = self.proc_h
        car_center = self.proc_w / 2
        left_x, right_x = 0, 0
        if left_found: left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        if right_found: right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

        lane_center = 0
        if left_found and right_found:
            lane_center = (left_x + right_x) / 2
            current_width = right_x - left_x
            if 150 < current_width < 600:
                self.lane_width = 0.9 * self.lane_width + 0.1 * current_width
        elif left_found: lane_center = left_x + (self.lane_width / 2)
        elif right_found: lane_center = right_x - (self.lane_width / 2)
        else: return {'offset': 0, 'status': 'Lost'}

        offset_pixel = car_center - lane_center
        scale_factor = self.width / self.proc_w
        raw_offset_real = offset_pixel * scale_factor

        self.offset_history.append(raw_offset_real)
        avg_offset = sum(self.offset_history) / len(self.offset_history)

        status = "Center"
        if avg_offset > CENTER_THRESHOLD: status = "Right ->"
        elif avg_offset < -CENTER_THRESHOLD: status = "<- Left"
        return {'offset': avg_offset, 'status': status}

    def _draw_lane_lines(self, frame_hd, binary_warped, left_fit, right_fit, left_found, right_found):
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        if left_found: left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        else:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            left_fitx = right_fitx - self.lane_width

        if right_found: right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        else:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = left_fitx + self.lane_width

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]).astype(np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int32)
        pts_poly = np.hstack((pts_left, np.flipud(pts_right)))

        cv2.fillPoly(color_warp, [pts_poly], (0, 255, 0))
        if left_found:
            cv2.polylines(color_warp, [pts_left], False, (0, 0, 255), thickness=15)
        if right_found:
            cv2.polylines(color_warp, [pts_right], False, (255, 0, 0), thickness=15)

        newwarp_small = cv2.warpPerspective(color_warp, self.Minv, (self.proc_w, self.proc_h))
        newwarp_hd = cv2.resize(newwarp_small, (self.width, self.height))
        result = cv2.addWeighted(frame_hd, 1, newwarp_hd, 0.4, 0)
        return result

    def _apply_car_mask(self, img):
        h, w = img.shape[:2]
        pts = np.array(CAR_MASK_POLY * [w, h], dtype=np.int32)
        masked = img.copy()
        cv2.fillPoly(masked, [pts], (0, 0, 0))
        return masked