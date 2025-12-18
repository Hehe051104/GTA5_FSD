# main.py

import cv2
import time
import numpy as np
from utils.screen_grab import ScreenGrabber
from modules.perception.yolop_service import YOLOPService # New SOTA YOLOPv2
from modules.perception.yolo_service import YoloService   # YOLOv8 Object Detection
from modules.control.pid_controller import PIDController
from utils.directkeys import press_key_by_name, release_key_by_name

# ================= 配置区域 =================
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
STEERING_THRESHOLD = 150  # 再次增大阈值：只有偏移超过 150 像素才转向
# --- PID 控制器参数 ---
KP = 0.005  # 增加 KP：单次修正力度加大
TURN_COOLDOWN = 0.25 # 增加冷却：每次转向后强制等待 0.25s，防止连续微调导致画龙
ENABLE_AUTOPILOT = True # 开关：启用自动驾驶控制 (LKA模式)

# --- 危险区域配置 (ROI) ---
# 动态 ROI 配置
ROI_CONFIG = {
    'lane_width': 350,              # 车道宽度 (像素)
    'y_min': VIEW_HEIGHT // 2 - 50, # 地平线附近 (远端)
    'y_max': VIEW_HEIGHT - 150,     # 仪表盘上方 (近端)
    'brake_threshold': 0.55         # 刹车距离阈值 (0.0-1.0): 物体底部超过屏幕高度的 55% 时刹车
}
# ===========================================

def check_danger(detections, lane_center_x):
    """
    检查是否有物体进入基于车道中心的动态危险区域，并判断距离
    """
    # 动态计算 ROI X 范围
    roi_x_min = lane_center_x - (ROI_CONFIG['lane_width'] // 2)
    roi_x_max = lane_center_x + (ROI_CONFIG['lane_width'] // 2)
    
    # 边界检查
    roi_x_min = max(0, roi_x_min)
    roi_x_max = min(VIEW_WIDTH, roi_x_max)

    # 计算刹车线的 Y 坐标 (像素)
    brake_line_y = int(VIEW_HEIGHT * ROI_CONFIG['brake_threshold'])

    current_roi = {
        'x_min': int(roi_x_min),
        'x_max': int(roi_x_max),
        'y_min': ROI_CONFIG['y_min'],
        'y_max': ROI_CONFIG['y_max'],
        'brake_line': brake_line_y
    }

    for det in detections:
        # 只关心车辆、行人等障碍物
        if det['label'] in ['car', 'truck', 'bus', 'motorcycle', 'person']:
            x1, y1, x2, y2 = det['bbox']
            # 计算物体底部中心点
            cx = (x1 + x2) // 2
            cy = y2 
            
            # 1. X轴判断: 物体中心是否在本车道内
            in_lane = (current_roi['x_min'] < cx < current_roi['x_max'])
            
            # 2. Y轴判断: 物体是否在地平线以下 (有效视野内)
            in_view = (current_roi['y_min'] < cy < current_roi['y_max'])
            
            if in_lane and in_view:
                # 3. 距离判断: 只有当物体底部超过刹车线 (离我们足够近) 时才刹车
                # y 坐标越大，代表物体在屏幕越下方，离我们越近
                if cy > brake_line_y:
                    return True, det, current_roi # [危险] 距离过近，刹车！
                else:
                    # 在车道内但距离尚远，可以返回一个 "预警" 状态 (这里暂不处理，视为安全)
                    pass
                
    return False, None, current_roi

def main():
    # 倒计时
    for i in list(range(3))[::-1]:
        print(f'将在 {i+1} 秒后启动...')
        time.sleep(1)

    print(f' GTA5_FSD (Turbo Mode) 启动...')

    try:
        grabber = ScreenGrabber()
        
        # 初始化 YOLOPv2 服务 (全能模型：检测+路面+车道线)
        # 首次运行会自动下载模型 (约 180MB)
        yolop_bot = YOLOPService(model_path='models/yolopv2.pt', device='cuda')
        
        # 初始化 YOLOv8 服务 (物体检测：车辆、行人等)
        yolo_bot = YoloService(model_path='models/yolov8n.pt')
        
        # 控制器
        controller = PIDController(Kp=KP, steering_threshold=STEERING_THRESHOLD, turn_cooldown=TURN_COOLDOWN)
            
    except Exception as e:
        print(f' 初始化失败: {e}')
        import traceback
        traceback.print_exc()
        return

    print(' 视觉系统就绪 - 自动驾驶已启用')
    
    # 速度控制状态
    current_speed_key = None 
    
    # 自动驾驶开关状态 (本地变量)
    autopilot_on = ENABLE_AUTOPILOT

    while True:
        t0 = time.time()
        
        # 1. 获取屏幕
        raw_frame = grabber.get_frame()
        if raw_frame is None: continue
        t1 = time.time()

        frame = cv2.resize(raw_frame, (VIEW_WIDTH, VIEW_HEIGHT))
        t2 = time.time()
        
        # 2. YOLOPv2 全能处理
        # process 返回: 叠加了分割图的帧, 导航信息
        result_frame, lane_info = yolop_bot.process(frame)
        
        # 获取当前车道中心 (用于动态 ROI)
        # offset = lane_center - screen_center
        # 所以 lane_center = screen_center + offset
        current_lane_center_x = (VIEW_WIDTH // 2) + lane_info['offset']
        
        # 3. YOLOv8 物体检测
        # 在 YOLOPv2 的结果上叠加检测框
        detections = yolo_bot.detect(frame)
        
        # --- 碰撞风险检测 (使用动态 ROI) ---
        is_danger, danger_obj, current_roi = check_danger(detections, current_lane_center_x)
        
        # 可视化危险区域 (动态跟随车道)
        roi_color = (0, 0, 255) if is_danger else (0, 255, 0) # 红/绿
        cv2.rectangle(result_frame, 
                      (current_roi['x_min'], current_roi['y_min']), 
                      (current_roi['x_max'], current_roi['y_max']), 
                      roi_color, 2)
        
        # 可视化刹车线 (蓝色虚线)
        brake_y = current_roi['brake_line']
        cv2.line(result_frame, (current_roi['x_min'], brake_y), (current_roi['x_max'], brake_y), (255, 255, 0), 2)
        cv2.putText(result_frame, "BRAKE LINE", (current_roi['x_min'], brake_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if is_danger:
             cv2.putText(result_frame, f"WARNING: {danger_obj['label']} TOO CLOSE!", (current_roi['x_min'], current_roi['y_min']-10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['conf']:.2f}"
            # 如果是导致危险的物体，用红色高亮
            color = (0, 0, 255) if (is_danger and det == danger_obj) else (0, 165, 255)
            
            # 画框
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            # 画标签背景
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            # 画文字
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        t3 = time.time()
        
        offset = lane_info['offset']
        status = lane_info['status']

        # 4. 控制逻辑
        action_steer = 'Straight'
        action_speed = 'Idle'

        if autopilot_on:
            # --- 转向控制 ---
            if 'Tracking' in status:
                action_steer = controller.get_action(offset)
            
            # --- 速度控制 (ACC) ---
            target_key = 'W' # 默认巡航
            
            if is_danger:
                target_key = 'SPACE' # 危险则刹车 (使用空格键)
                action_speed = 'BRAKE (Danger)'
            else:
                action_speed = 'CRUISE'
            
            # 执行按键切换
            if target_key != current_speed_key:
                if current_speed_key:
                    release_key_by_name(current_speed_key)
                press_key_by_name(target_key)
                current_speed_key = target_key
        else:
            action_steer = 'Manual'
            action_speed = 'Manual'
            # 确保释放按键
            if current_speed_key:
                release_key_by_name(current_speed_key)
                current_speed_key = None
        
        # 5. 显示信息
        fps = 1 / (time.time() - t0)
        
        cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f'Offset: {offset}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f'Steer: {action_steer}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Speed: {action_speed}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Status: {status}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        if autopilot_on:
             cv2.putText(result_frame, 'MODE: AUTOPILOT (ACC + LKA)', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
             cv2.putText(result_frame, '[L] to Disable', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
             cv2.putText(result_frame, 'MODE: MANUAL', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
             cv2.putText(result_frame, '[L] to Enable', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示主窗口
        cv2.imshow('GTA5 FSD - Turbo', result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            autopilot_on = not autopilot_on
            print(f"Autopilot toggled: {autopilot_on}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
