# main.py

import cv2
import time
import numpy as np
from utils.screen_grab import ScreenGrabber
<<<<<<< HEAD
# from modules.perception.yolo_service import YoloService # Deprecated
# from modules.perception.segmentation_lane_detector import SegmentationLaneDetector # Deprecated
from modules.perception.yolop_service import YOLOPService # New SOTA YOLOPv2
from modules.control.pid_controller import PIDController

# ================= 配置区域 =================
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
STEERING_THRESHOLD = 150  # 再次增大阈值：只有偏移超过 150 像素才转向
# --- PID 控制器参数 ---
KP = 0.005  # 增加 KP：单次修正力度加大
TURN_COOLDOWN = 0.25 # 增加冷却：每次转向后强制等待 0.25s，防止连续微调导致画龙
ENABLE_AUTOPILOT = True # 开关：启用自动驾驶控制 (LKA模式)
# ===========================================

def main():
    # 倒计时
    for i in list(range(3))[::-1]:
        print(f"将在 {i+1} 秒后启动...")
        time.sleep(1)

    print(f"🚀 GTA5_FSD (YOLOPv2 Mode) 启动...")

    try:
        grabber = ScreenGrabber()
        
        # 初始化 YOLOPv2 服务 (全能模型：检测+路面+车道线)
        # 首次运行会自动下载模型 (约 180MB)
        yolop_bot = YOLOPService(model_path='models/yolopv2.onnx', device='cuda')
        
        # 控制器
        controller = PIDController(Kp=KP, steering_threshold=STEERING_THRESHOLD, turn_cooldown=TURN_COOLDOWN)
            
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
=======
from modules.perception.yolo_service import YoloService
from modules.perception.lane_detector import LaneDetector
from modules.perception.map_reader import MapReader
from modules.control.car_controller import CarController
from config import VIEW_WIDTH, VIEW_HEIGHT, CAR_MASK_POLY, W_MAP, W_LANE

def main():
    print(f"🚀 GTA5_FSD (Final Fusion) Starting...")
    try:
        grabber = ScreenGrabber()
        yolo_bot = YoloService(model_path='models/yolov8n.pt')
        lane_bot = LaneDetector(width=VIEW_WIDTH, height=VIEW_HEIGHT)
        map_bot = MapReader()
        controller = CarController()
    except Exception as e:
        print(f"❌ Init Failed: {e}")
>>>>>>> ace3bcae962f6d66a4e118fb5839768d5f6681f4
        return

    print("✅ Ready. Press W.")
    last_time = time.time()

    while True:
        # 1. 获取屏幕
        raw_frame = grabber.get_frame()
        if raw_frame is None: continue

        frame = cv2.resize(raw_frame, (VIEW_WIDTH, VIEW_HEIGHT))
<<<<<<< HEAD
        
        # 2. YOLOPv2 全能处理
        # process 返回: 叠加了分割图的帧, 导航信息
        result_frame, lane_info = yolop_bot.process(frame)
        
        offset = lane_info['offset']
        status = lane_info['status']

        # 3. 控制逻辑
        action = "Straight"
        # 只要状态包含 "Tracking" (无论是 AreaPrimary 还是 Lines)，都启用控制
        if ENABLE_AUTOPILOT and "Tracking" in status:
            action = controller.get_action(offset)
        else:
            action = "Manual Mode" if not ENABLE_AUTOPILOT else "Straight"

        # 4. 显示信息
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Offset: {offset}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Action: {action}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Status: {status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        if ENABLE_AUTOPILOT:
             cv2.putText(result_frame, "MODE: LKA (You drive W)", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
             cv2.putText(result_frame, "MODE: MANUAL", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示主窗口
        cv2.imshow('GTA5 FSD - YOLOPv2', result_frame)
=======

        detections = yolo_bot.detect(frame)
        ego_car_box = []
        img_h, img_w = frame.shape[:2]
        center_x = img_w // 2
        for d in detections:
            if d['label'] in ['car', 'truck', 'bus']:
                x1, y1, x2, y2 = d['bbox']
                box_center_x = (x1 + x2) // 2
                if y2 > img_h * 0.90 and abs(box_center_x - center_x) < 200:
                    ego_car_box.append([x1, y1, x2, y2])

        # Lane Detection
        result_frame, lane_info, debug_binary = lane_bot.process(frame, blocking_boxes=ego_car_box)
        lane_offset = lane_info['offset']
        lane_status = lane_info['status']

        # Map Reading
        nav_error, debug_map = map_bot.process(frame)

        # Fusion
        final_steering = 0
        if lane_status == "No Lane" or lane_status == "Lost":
            final_steering = nav_error * 1.5
            mode = "MAP_ONLY"
        else:
            final_steering = (nav_error * W_MAP) + (lane_offset * W_LANE)
            mode = "FUSION"

        # Control
        controller.update_steering(final_steering)

        # Drawing
        mask_overlay = result_frame.copy()
        pts = np.array(CAR_MASK_POLY * [VIEW_WIDTH, VIEW_HEIGHT], dtype=np.int32)
        cv2.fillPoly(mask_overlay, [pts], (100, 100, 100))
        cv2.addWeighted(mask_overlay, 0.4, result_frame, 0.6, 0, result_frame)

        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            label = f"{d['label']} {d['conf']:.2f}"
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        loop_time = time.time()
        fps = 1 / (loop_time - last_time)
        last_time = loop_time

        cv2.putText(result_frame, f"FPS: {int(fps)} | Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        info_text = f"Map({W_MAP}):{int(nav_error)} + Lane({W_LANE}):{int(lane_offset)} = {int(final_steering)}"
        center_x = (VIEW_WIDTH - 400) // 2
        cv2.putText(result_frame, info_text, (center_x, VIEW_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('GTA5 FSD - Pilot View', result_frame)
        if debug_map is not None:
            cv2.imshow('DEBUG: Minimap', cv2.resize(debug_map, (300, 300)))
        if debug_binary is not None:
            cv2.imshow('DEBUG: Lane Binary', cv2.resize(debug_binary, (640, 360)))
>>>>>>> ace3bcae962f6d66a4e118fb5839768d5f6681f4

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
