# main.py

import cv2
import time
import numpy as np
import win32api # For global key state
from utils.screen_grab import ScreenGrabber
from modules.perception.yolop_service import YOLOPService # New SOTA YOLOPv2
from modules.perception.yolo_service import YoloService   # YOLOv8 Object Detection
from modules.perception.map_reader import MapReader       # Minimap Navigation
from modules.perception.stuck_detector import StuckDetector # Visual Stuck Detection
from modules.perception.collision_monitor import CollisionMonitor # Collision Warning
from modules.control.pid_controller import PIDController
from utils.directkeys import press_key_by_name, release_key_by_name

# ================= 配置区域 =================
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720

# --- 转向阈值配置 ---
# 视觉车道保持阈值: 降低到 60 以提供更灵敏的修正，防止大幅度画龙
STEERING_THRESHOLD_LANE = 140
# 小地图导航阈值: 保持灵敏
STEERING_THRESHOLD_MAP = 30

# --- 偏移量偏置 (Lane Bias) ---
# 正值 = 强制车身向右靠 (单位: 像素)
# 解决 YOLOP 识别整条路导致压黄线的问题
LANE_OFFSET_BIAS = 40

# --- PID 控制器参数 ---
KP = 0.005  
# 冷却时间: 降低到 0.2s，允许更频繁的微调
TURN_COOLDOWN = 0.3
ENABLE_AUTOPILOT = True 

# --- 危险区域配置 (ROI) ---
# 动态 ROI 配置
ROI_CONFIG = {
    'lane_width': 400,              # [修改] 适度加宽 (350 -> 400)，避免误检对向车道
    'y_min': VIEW_HEIGHT // 2 - 50, # 地平线附近 (远端)
    'y_max': VIEW_HEIGHT - 150,     # 仪表盘上方 (近端)
    'brake_threshold': 0.65         # [修改] 刹车距离阈值 (更远就开始刹车)
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
        
        # 初始化卡死检测器
        stuck_bot = StuckDetector(stuck_threshold=2.0)
        
        # 初始化碰撞预警系统
        collision_monitor = CollisionMonitor(expansion_threshold=1.05, history_len=5)
        
        # 控制器 (默认使用车道保持阈值)
        controller = PIDController(Kp=KP, steering_threshold=STEERING_THRESHOLD_LANE, turn_cooldown=TURN_COOLDOWN)
            
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
    
    # 无尽模式状态 (K键控制)
    endless_mode = False
    k_key_pressed = False

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
        nav_offset, is_on_route, nav_mode, minimap_vis = map_bot.process(raw_frame)
        
        # 3. YOLOv8 物体检测
        # 在 YOLOPv2 的结果上叠加检测框
        detections = yolo_bot.detect(frame)
        
        # --- 卡死检测 (机器视觉) ---
        # 传入当前帧和当前按下的按键
        is_stuck, is_recovering = stuck_bot.update(frame, current_speed_key)
        
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
        
        final_offset = 0
        nav_source = "None"
        
        # 检查信号状态
        has_map_signal = (nav_offset is not None)
        has_lane_signal = ('Tracking' in status)
        
        # --- 决策状态机 ---
        should_cruise = False
        
        if has_map_signal:
            # 检查是否处于直行模式 (Forward Mode)
            # 只有在明确检测到前方有直行路径时，才允许使用视觉车道保持
            is_forward = "Forward" in nav_mode
            
            if not is_forward:
                 # 场景0: 非直行模式 (Turn / Fallback) -> 强制地图导航
                 # 此时前方矩形区域无路径，说明即将转向或到达尽头，必须忽略视觉车道保持
                 final_offset = nav_offset
                 nav_source = f"Map ({nav_mode})"
                 should_cruise = True
                 controller.steering_threshold = STEERING_THRESHOLD_MAP
            elif has_lane_signal:
                if is_on_route:
                    # 场景1.1: 在路线上且前方直行 (On Route + Forward) -> 信任视觉车道保持
                    # 车辆位于紫色导航线上，且前方有路，优先使用平滑的视觉车道保持
                    # [修改] 加上偏移量偏置，强制靠右行驶
                    final_offset = offset + LANE_OFFSET_BIAS
                    nav_source = "Vision (On Route)"
                    should_cruise = True
                    controller.steering_threshold = STEERING_THRESHOLD_LANE # 低灵敏度，防画龙
                else:
                    # 场景1.2: 偏离路线 (Off Route) -> 导航强力介入
                    # 车辆不在紫色线上（可能在路口或偏离），使用导航修正
                    final_offset = (offset * 0.3) + (nav_offset * 0.7)
                    nav_source = "Map Intervention"
                    should_cruise = True
                    controller.steering_threshold = STEERING_THRESHOLD_MAP # 高灵敏度，快速修正
            else:
                # 场景2: 无车道信号 (十字路口/模糊路段)
                # 只要能看到紫色导航线 (has_map_signal=True)，就应该尝试去追，而不是停车
                # 无论是否在路线上 (is_on_route)，都应该信任 nav_offset
                final_offset = nav_offset
                nav_source = "Map Only (Recovery)"
                should_cruise = True
                controller.steering_threshold = STEERING_THRESHOLD_MAP
        else:
            # 无紫线 (到达目的地 或 丢失)
            if endless_mode and has_lane_signal:
                # 场景3: 无尽模式 (无紫线 + 有车道 + K键激活)
                # [修改] 加上偏移量偏置
                final_offset = offset + LANE_OFFSET_BIAS
                nav_source = "Vision (Endless)"
                should_cruise = True
                # 切换回低灵敏度阈值 (防画龙)
                controller.steering_threshold = STEERING_THRESHOLD_LANE
            else:
                # 场景4: 到达目的地/完全丢失 -> 刹车
                nav_source = "Dest/Lost"
                should_cruise = False # Brake

        if autopilot_on:
            # --- 优先级处理 ---
            
            # 优先级 0: 倒车脱困 (最高优先级)
            if is_recovering:
                # 倒车 + 向左打方向 (试图倒出死胡同)
                target_key = 'S'
                action_speed = 'RECOVER (Reverse)'
                # 强制修改转向逻辑
                action_steer = 'Left (Reverse)'
                # 简单粗暴：直接按 A，并确保 D 松开
                release_key_by_name('D')
                press_key_by_name('A') 
            
            else:
                # 正常转向逻辑
                # 只有在巡航状态下才进行转向修正，刹车时保持直行以防乱转
                if should_cruise:
                    # 动态调整最大转向时长 (针对十字路口急转弯)
                    # 如果偏移量巨大 (>200)，说明是急弯，允许更长的按键时间 (0.6s)
                    # 否则使用默认的微调时间 (0.15s)
                    max_turn_duration = 0.6 if abs(final_offset) > 200 else 0.15
                    action_steer = controller.get_action(final_offset, max_duration=max_turn_duration)
                else:
                    action_steer = 'Straight'
                
                # 正常速度逻辑
                if is_danger:
                    # 碰撞预警优先级最高
                    target_key = 'SPACE'
                    action_speed = 'BRAKE (Danger)'
                elif should_cruise:
                    # 始终保持动力 (按住 W)，即使是急转弯
                    target_key = 'W'
                    action_speed = 'CRUISE'
                else:
                    # 导航逻辑要求的停车
                    target_key = 'SPACE'
                    action_speed = 'BRAKE (Nav Logic)'
            
            # 执行按键切换 (速度控制)
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
        
        # 显示无尽模式状态
        endless_color = (0, 255, 0) if endless_mode else (100, 100, 100)
        cv2.putText(result_frame, f'Endless Mode [K]: {"ON" if endless_mode else "OFF"}', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, endless_color, 2)

        # 显示卡死状态
        if is_recovering:
            cv2.putText(result_frame, "STUCK DETECTED! REVERSING...", (VIEW_WIDTH//2 - 200, VIEW_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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
            
        # 检测 'K' 键 (0x4B) - 无尽模式切换
        if win32api.GetAsyncKeyState(0x4B):
            if not k_key_pressed:
                endless_mode = not endless_mode
                print(f"Endless Mode toggled: {endless_mode}")
                k_key_pressed = True
        else:
            k_key_pressed = False

        # OpenCV 窗口按键 (仅用于退出)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
