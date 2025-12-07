# modules/perception/segmentation_lane_detector.py

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import urllib.request

class SegmentationLaneDetector:
    """
    基于深度学习语义分割的道路检测器
    使用 YOLOv8-Seg 模型直接分割出"可行驶区域" (Drivable Area)
    """
    def __init__(self, model_path='models/yolov8n-seg.pt', width=1280, height=720):
        self.width = width
        self.height = height
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Deep Lane Detector ... Device: {self.device}")

        # 检查模型是否存在，不存在则下载
        # 注意：标准的 yolov8n-seg 是 COCO 数据集 (只有车，没有路)
        # 为了效果最好，我们需要一个在 BDD100K 或 Cityscapes 上训练的模型
        # 这里我们先加载用户提供的模型，如果用户有专门的道路分割模型，替换路径即可
        if not os.path.exists(model_path):
            print(f"⚠️ 模型 {model_path} 不存在，尝试下载标准 YOLOv8n-seg...")
            try:
                # 下载标准模型作为占位 (实际建议用户替换为道路分割专用模型)
                self.model = YOLO('yolov8n-seg.pt') 
            except Exception as e:
                print(f"❌ 下载失败: {e}")
        else:
            self.model = YOLO(model_path)

        # 预热模型
        print("🔥 正在预热模型...")
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=self.device, verbose=False, half=True)

    def process(self, frame):
        """
        返回:
        - result_frame: 绘制了分割掩码的图像
        - lane_info: {'offset': float, 'status': str}
        - mask_vis: 分割掩码的可视化图 (用于 debug)
        """
        # 1. 推理
        # conf=0.25, iou=0.7, imgsz=640, half=True (半精度加速)
        # classes=[0, 1, ...] 如果我们知道"路"是哪个类，可以指定
        # 对于标准 COCO，没有"路"。
        # 假设用户使用的是 BDD100K 训练的模型，通常 class 0 是 road 或 drivable area
        results = self.model.predict(frame, 
                                     device=self.device, 
                                     imgsz=640, 
                                     half=True, 
                                     verbose=False, 
                                     conf=0.3,
                                     retina_masks=True) # retina_masks=True 掩码更精细

        result = results[0]
        lane_info = {'offset': 0, 'status': 'No Road'}
        
        # 创建可视化图
        mask_vis = np.zeros_like(frame)
        overlay = frame.copy()

        if result.masks is not None:
            # 获取所有掩码 (N, H, W)
            masks = result.masks.data
            
            # 寻找"路"的掩码
            # 如果是标准 COCO 模型，我们没有"路"的类。
            # 作为一个聪明的 fallback，我们假设画面下方最大的那个掩码块就是路 (如果它不是车)
            # 或者，如果用户真的去下载了 BDD 模型，我们直接找 class 0
            
            # 策略：合并所有非车辆的掩码，或者寻找位于画面底部的最大掩码
            combined_mask = torch.zeros_like(masks[0], device=self.device)
            
            found_road = False
            
            # 遍历检测到的对象
            for i, cls_id in enumerate(result.boxes.cls):
                class_name = self.model.names[int(cls_id)]
                
                # 排除车辆和人 (COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)
                if int(cls_id) in [0, 2, 3, 5, 7]:
                    continue
                
                # 如果是专门的道路模型，通常 road 是 class 0 或 1
                # 这里我们假设所有"非障碍物"的大面积区域都可能是路
                # 简单起见，我们将所有掩码合并 (假设模型只输出了路，或者我们过滤掉了车)
                combined_mask = torch.logical_or(combined_mask, masks[i])
                found_road = True

            # 如果没找到特定的路，但有掩码，且模型是专门的道路模型(通常只输出路)，那就全用
            if not found_road and len(masks) > 0:
                 # 简单的启发式：取最大的那个掩码
                 combined_mask = torch.sum(masks, dim=0) > 0
                 found_road = True

            if found_road:
                # 转回 CPU numpy
                road_mask = combined_mask.cpu().numpy().astype(np.uint8) * 255
                
                # 调整大小回原图 (retina_masks=True 时已经是原图大小，否则需要 resize)
                if road_mask.shape[:2] != frame.shape[:2]:
                    road_mask = cv2.resize(road_mask, (self.width, self.height))

                # 计算重心 (Centroid)
                M = cv2.moments(road_mask)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 绘制
                    # 绿色覆盖路面
                    color_mask = np.zeros_like(frame)
                    color_mask[road_mask > 0] = [0, 255, 0]
                    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
                    
                    # 绘制重心
                    cv2.circle(overlay, (cx, cy), 10, (0, 0, 255), -1)
                    
                    # 计算前视点 (Look Ahead)
                    # 我们取重心作为引导点，或者取掩码在 y=0.7*h 处的中心
                    # 重心通常比较稳
                    
                    # 计算偏移
                    screen_center = self.width // 2
                    offset = screen_center - cx
                    
                    lane_info = {
                        'offset': offset,
                        'status': 'Tracking'
                    }
                    
                    mask_vis = color_mask

        return overlay, lane_info, mask_vis
