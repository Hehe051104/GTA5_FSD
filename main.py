# main.py

import cv2
import time
import numpy as np
import win32api # For global key state
from utils.screen_grab import ScreenGrabber
from modules.perception.yolop_service import YOLOPService # New SOTA YOLOPv2
from modules.perception.yolo_service import YoloService   # YOLOv8 Object Detection
from modules.perception.map_reader import MapReader       # Minimap Navigation
from modules.perception.collision_monitor import CollisionMonitor # Collision Warning
from modules.control.pid_controller import PIDController
from utils.directkeys import press_key_by_name, release_key_by_name

# ================= 配置区域 =================
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
STEERING_THRESHOLD = 140  # 再次增大阈值：只有偏移超过 150 像素才转向
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
        
        # 初始化小地图导航
        map_bot = MapReader()
        
        # 初始化碰撞预警系统
        collision_monitor = CollisionMonitor(expansion_threshold=1.05, history_len=5)
        
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
    l_key_pressed = False # 用于 L 键防抖

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
        
        # --- 小地图导航处理 ---
        # 注意：map_bot 需要原始分辨率的帧来截取小地图区域
        nav_offset, minimap_vis = map_bot.process(raw_frame)
        
        # 3. YOLOv8 物体检测
        # 在 YOLOPv2 的结果上叠加检测框
        detections = yolo_bot.detect(frame)
        
        # --- 碰撞风险检测 (使用动态 ROI) ---
        # 动态计算 ROI X 范围
        roi_x_min = current_lane_center_x - (ROI_CONFIG['lane_width'] // 2)
        roi_x_max = current_lane_center_x + (ROI_CONFIG['lane_width'] // 2)
        roi_x_min = max(0, roi_x_min)
        roi_x_max = min(VIEW_WIDTH, roi_x_max)
        
        current_roi = {
            'x_min': int(roi_x_min),
            'x_max': int(roi_x_max),
            'y_min': ROI_CONFIG['y_min'],
            'y_max': ROI_CONFIG['y_max']
        }
        
        is_danger, danger_obj, debug_info = collision_monitor.update(detections, current_roi)
        
        # 可视化危险区域 (动态跟随车道)
        roi_color = (0, 0, 255) if is_danger else (0, 255, 0) # 红/绿
        cv2.rectangle(result_frame, 
                      (current_roi['x_min'], current_roi['y_min']), 
                      (current_roi['x_max'], current_roi['y_max']), 
                      roi_color, 2)
        
        if is_danger:
             cv2.putText(result_frame, f"WARNING: COLLISION RISK!", (current_roi['x_min'], current_roi['y_min']-10), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示调试信息 (膨胀率等)
        for i, msg in enumerate(debug_info):
             cv2.putText(result_frame, msg, (10, 350 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['conf']:.2f}"
            
            # 如果是导致危险的物体，用红色高亮
            is_danger_target = False
            if is_danger and danger_obj:
                 d_bbox = danger_obj['bbox']
                 if x1 == d_bbox[0] and y1 == d_bbox[1]:
                     is_danger_target = True
            
            color = (0, 0, 255) if is_danger_target else (0, 165, 255)
            
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
        
        # 决策逻辑：
        # 1. 视觉车道保持 (最高优先级，精度最高)
        # 2. 小地图道路保持 (中等优先级，用于视觉丢失时)
        # 3. 小地图导航 (用于路口转向或辅助)
        
        final_offset = 0
        nav_source = "None"
        
        # 检查小地图是否有有效信号 (导航或道路)
        has_map_signal = (nav_offset is not None and nav_offset != 0)
        
        if 'Tracking' in status:
            # 视觉系统正常工作
            final_offset = offset
            nav_source = "Vision (Lane)"
            
            # 如果小地图指示大幅度转向 (例如路口)，可以尝试融合
            # 但目前为了稳定性，视觉优先
        elif has_map_signal:
            # 视觉丢失，使用小地图 (导航优先，道路其次，由 MapReader 内部处理)
            final_offset = nav_offset
            nav_source = "Map (GPS/Road)"
        else:
            nav_source = "Lost"

        if autopilot_on:
            # --- 转向控制 ---
            if nav_source != "Lost":
                action_steer = controller.get_action(final_offset)
            else:
                # 丢失所有信号，保持直行
                action_steer = 'Straight'
            
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
        
        # 叠加小地图可视化
        if minimap_vis is not None:
            try:
                # 将小地图可视化缩放并放置在右上角
                h, w = minimap_vis.shape[:2]
                target_h = 150
                scale = target_h / h
                target_w = int(w * scale)
                minimap_small = cv2.resize(minimap_vis, (target_w, target_h))
                
                # 放置在右上角
                result_frame[10:10+target_h, VIEW_WIDTH-10-target_w:VIEW_WIDTH-10] = minimap_small
                cv2.putText(result_frame, "GPS Nav", (VIEW_WIDTH-10-target_w, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            except Exception as e:
                print(f"Minimap overlay error: {e}")

        cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f'Offset: {final_offset:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f'Steer: {action_steer}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Speed: {action_speed}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Status: {status}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(result_frame, f'Nav: {nav_source}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if autopilot_on:
             cv2.putText(result_frame, 'MODE: AUTOPILOT (ACC + LKA)', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
             cv2.putText(result_frame, '[L] to Disable', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
             cv2.putText(result_frame, 'MODE: MANUAL', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
             cv2.putText(result_frame, '[L] to Enable', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示主窗口
        cv2.imshow('GTA5 FSD - Turbo', result_frame)

        # --- 全局按键检测 (win32api) ---
        # 检测 'L' 键 (0x4C)
        if win32api.GetAsyncKeyState(0x4C):
            if not l_key_pressed:
                autopilot_on = not autopilot_on
                print(f"Autopilot toggled: {autopilot_on}")
                l_key_pressed = True
        else:
            l_key_pressed = False

        # OpenCV 窗口按键 (仅用于退出)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
