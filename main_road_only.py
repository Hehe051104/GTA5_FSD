import cv2
import time
import numpy as np
import torch
from utils.screen_grab import ScreenGrabber
from modules.perception.yolop_service import YOLOPService

# ================= 配置 =================
# 为了极致速度，我们不再把结果放大回 1280x720
# 而是直接在 640x640 的推理尺寸上显示
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 640
# =======================================

def main():
    print("🚀 启动极速道路检测模式 (Pure Road Detection)...")
    
    # 1. 初始化
    try:
        grabber = ScreenGrabber()
        # 强制使用 CUDA
        yolop_service = YOLOPService(model_path='models/yolopv2.pt', device='cuda')
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print("✅ 系统就绪，按 'q' 退出")
    
    # 用于计算 FPS
    prev_time = time.time()
    fps_avg = 0
    
    while True:
        loop_start = time.time()
        
        # 2. 抓取屏幕
        # 注意：grabber 返回的是原始分辨率 (比如 1280x720 或 1920x1080)
        frame_large = grabber.get_frame()
        if frame_large is None:
            time.sleep(0.1)
            continue
            
        # 3. 预处理 (Resize)
        # 我们直接把原图缩放到 640x640，用于推理和显示
        # 这样避免了后续把 mask 放大回原图的巨大开销
        frame_input = cv2.resize(frame_large, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # 4. 推理 (Infer)
        # 直接调用 infer，跳过 service.process 中繁重的后处理
        # infer 返回: (det, da_seg, ll_seg)
        # det: 目标检测 (车辆)
        # da_seg: 可行驶区域 (Road)
        # ll_seg: 车道线 (Lane Lines)
        _, da_seg, _ = yolop_service.infer(frame_input)
        
        # 5. 极速后处理
        # da_seg shape: (1, 2, 640, 640) -> 取 channel 1
        road_mask = da_seg[0][1]
        
        # 创建可视化图层
        # 只有当置信度 > 0.5 时才认为是路
        # 使用布尔索引快速赋值，比 cv2.threshold + cv2.addWeighted 快
        
        # 这里的逻辑是：直接修改 frame_input 的像素
        # 绿色通道 (B, G, R) -> Index 1
        # 将认为是路的地方，绿色通道设为 255 (最亮)
        # 这种 "破坏性" 修改比创建新图层叠加要快得多
        frame_input[:, :, 1][road_mask > 0.5] = 255
        
        # 6. 计算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_avg = 0.9 * fps_avg + 0.1 * fps # 平滑 FPS
        
        # 7. 显示
        cv2.putText(frame_input, f"FPS: {int(fps_avg)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("GTA5 FSD - Fast Road Mode", frame_input)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
