# main.py

import cv2
import time
import numpy as np
from utils.screen_grab import ScreenGrabber
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
        print(f'将在 {i+1} 秒后启动...')
        time.sleep(1)

    print(f' GTA5_FSD (YOLOPv2 Mode) 启动...')

    try:
        grabber = ScreenGrabber()
        
        # 初始化 YOLOPv2 服务 (全能模型：检测+路面+车道线)
        # 首次运行会自动下载模型 (约 180MB)
        yolop_bot = YOLOPService(model_path='models/yolopv2.onnx', device='cuda')
        
        # 控制器
        controller = PIDController(Kp=KP, steering_threshold=STEERING_THRESHOLD, turn_cooldown=TURN_COOLDOWN)
            
    except Exception as e:
        print(f' 初始化失败: {e}')
        import traceback
        traceback.print_exc()
        return

    print(' 视觉系统就绪')
    last_time = time.time()

    while True:
        # 1. 获取屏幕
        raw_frame = grabber.get_frame()
        if raw_frame is None: continue

        frame = cv2.resize(raw_frame, (VIEW_WIDTH, VIEW_HEIGHT))
        
        # 2. YOLOPv2 全能处理
        # process 返回: 叠加了分割图的帧, 导航信息
        result_frame, lane_info = yolop_bot.process(frame)
        
        offset = lane_info['offset']
        status = lane_info['status']

        # 3. 控制逻辑
        action = 'Straight'
        # 只要状态包含 'Tracking' (无论是 AreaPrimary 还是 Lines)，都启用控制
        if ENABLE_AUTOPILOT and 'Tracking' in status:
            action = controller.get_action(offset)
        else:
            action = 'Manual Mode' if not ENABLE_AUTOPILOT else 'Straight'

        # 4. 显示信息
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        
        cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f'Offset: {offset}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(result_frame, f'Action: {action}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Status: {status}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        if ENABLE_AUTOPILOT:
             cv2.putText(result_frame, 'MODE: LKA (You drive W)', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
             cv2.putText(result_frame, 'MODE: MANUAL', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示主窗口
        cv2.imshow('GTA5 FSD - YOLOPv2', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
