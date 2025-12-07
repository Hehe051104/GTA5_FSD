import cv2
import numpy as np
import onnxruntime as ort
import os
import urllib.request
import torch

class YOLOPService:
    def __init__(self, model_path='models/yolopv2.onnx', device='cuda'):
        self.model_path = model_path
        self.device = device
        self.input_width = 640
        self.input_height = 640
        
        # 核心修复：根据文件扩展名决定加载方式
        if model_path.endswith('.pt'):
            self.use_torch = True
        else:
            self.use_torch = False

        # 如果使用 PyTorch
        if self.use_torch:
            print(f"ℹ️ 检测到 .pt 文件，将使用 PyTorch (TorchScript) 模式。")
            if not os.path.exists(model_path):
                 print(f"❌ 错误: 指定的 TorchScript 模型文件不存在: {model_path}")
                 raise FileNotFoundError(f"Model file not found: {model_path}")
            try:
                self.model = torch.jit.load(model_path)
                if device == 'cuda' and torch.cuda.is_available():
                    self.model = self.model.cuda()
                    print("🧠 YOLOPv2 Service (TorchScript) ... Device: CUDA")
                else:
                    self.model = self.model.cpu()
                    print("🧠 YOLOPv2 Service (TorchScript) ... Device: CPU")
                self.model.eval()
                return # PyTorch 初始化完成
            except Exception as e:
                print(f"❌ 加载 TorchScript 模型失败: {e}")
                raise e

        # 如果使用 ONNX (原逻辑)
        if not os.path.exists(model_path):
            print(f"⚠️ 模型 {model_path} 不存在，正在尝试自动下载 ONNX 版本...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            urls = [
                "https://github.com/ibaiGorordo/ONNX-YOLOP-v2-Lane-Detection/raw/main/models/yolopv2.onnx",
                "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.onnx"
            ]
            
            downloaded = False
            for url in urls:
                try:
                    print(f"正在尝试从 {url} 下载...")
                    urllib.request.urlretrieve(url, model_path)
                    print("✅ 下载完成！")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"❌ 从该源下载失败: {e}")
            
            if not downloaded:
                print("❌ 所有下载源均失败。")
                print("请手动下载 'yolopv2.onnx' 并放入 'models/' 文件夹。")
                raise FileNotFoundError("Model file not found")

        # 初始化 ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"🧠 YOLOPv2 Service (ONNX) ... Device: {device} (Providers: {self.session.get_providers()})")
        except Exception as e:
            print(f"⚠️ 无法加载 CUDA 提供程序或模型加载失败，回退到 CPU: {e}")
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            except Exception as e2:
                print(f"❌ 严重错误: 无法加载模型。请确保 models/yolopv2.onnx 存在且完整。")
                raise e2
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        self.img_h, self.img_w = img.shape[:2]
        
        # Resize
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # Normalize & Transpose (HWC -> CHW)
        # 回退到最原始的预处理逻辑 (仅除以 255)
        # 既然之前的效果好，说明这个 TorchScript 模型可能并没有使用标准的 ImageNet 归一化
        # 或者它内部已经处理了颜色转换
        img_data = img_resized.astype(np.float32) / 255.0
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0) # Add batch dim
        
        return img_data

    def infer(self, img):
        input_tensor = self.preprocess(img)
        
        if self.use_torch:
            # PyTorch 推理
            with torch.no_grad():
                tensor = torch.from_numpy(input_tensor)
                if self.device == 'cuda' and torch.cuda.is_available():
                    tensor = tensor.cuda()
                
                # YOLOPv2 TorchScript 输出通常也是 tuple
                outputs = self.model(tensor)
                
                # 转换回 numpy
                # 注意：TorchScript 模型的输出可能是一个 tuple，也可能是一个 list
                # 甚至有时候输出还在 GPU 上，需要先 detach() 再 cpu()
                
                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    det_out = outputs[0]
                    da_seg_out = outputs[1]
                    ll_seg_out = outputs[2]
                else:
                    # 如果输出不是 tuple/list，那可能结构不对，打印一下类型
                    print(f"⚠️ 模型输出类型异常: {type(outputs)}")
                    return None, None, None

                # 确保转回 CPU numpy
                if isinstance(det_out, torch.Tensor): det_out = det_out.detach().cpu().numpy()
                if isinstance(da_seg_out, torch.Tensor): da_seg_out = da_seg_out.detach().cpu().numpy()
                if isinstance(ll_seg_out, torch.Tensor): ll_seg_out = ll_seg_out.detach().cpu().numpy()
                
                return det_out, da_seg_out, ll_seg_out
        else:
            # ONNX 推理
            outputs = self.session.run(None, {self.input_name: input_tensor})
            det_out = outputs[0]
            da_seg_out = outputs[1]
            ll_seg_out = outputs[2]
            return det_out, da_seg_out, ll_seg_out

    def process(self, img):
        """
        主处理函数：执行推理并返回可视化结果和导航信息
        """
        try:
            det, da_seg, ll_seg = self.infer(img)
        except Exception as e:
            print(f"推理错误: {e}")
            return img, {'offset': 0, 'status': 'Error'}
        
        # --- 后处理分割掩码 ---
        # da_seg shape: (1, 2, 640, 640) -> 取 channel 1 (前景)
        # ll_seg shape: (1, 2, 640, 640) -> 取 channel 1 (前景)
        # 注意：某些模型导出可能只有 1 个 channel (1, 1, 640, 640)
        
        def get_mask(seg_out):
            if seg_out.shape[1] == 2:
                return seg_out[0][1] # 取前景
            else:
                return seg_out[0][0] # 只有一个通道，直接取

        da_mask = get_mask(da_seg)
        ll_mask = get_mask(ll_seg)
        
        # 二值化
        # da_mask (可行驶区域) 保持 0.5
        _, da_mask = cv2.threshold(da_mask, 0.5, 1, cv2.THRESH_BINARY)
        
        # ll_mask (车道线) 阈值从 0.5 降到 0.25
        _, ll_mask = cv2.threshold(ll_mask, 0.25, 1, cv2.THRESH_BINARY)
        
        # 关键修复：必须先 Resize 回原图尺寸，再进行后续计算和可视化
        # 之前的代码在计算重心时使用了未 Resize 的 mask (640x640)，导致和原图 (1280x720) 尺寸不匹配
        da_mask = cv2.resize(da_mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        ll_mask = cv2.resize(ll_mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # --- 计算导航偏移量 (优先基于可行驶区域重心，即绿色区域) ---
        offset = 0
        status = "Searching"
        screen_center = self.img_w // 2

        # 优先使用可行驶区域(da_mask)的重心作为车道中心
        # ROI：下半区域到接近底部，这样重心代表车辆应当前往的位置
        scan_h_start = int(self.img_h * 0.50)
        scan_roi = da_mask[scan_h_start:, :]
        M = cv2.moments(scan_roi)

        scan_y = int(self.img_h * 0.75)  # 默认用于可视化的高度
        cx = None
        cy = None

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"]) + scan_h_start
            offset = cx - screen_center
            status = "Tracking (AreaPrimary)"

            # 可视化：区域重心 (绿色大点) 和中心到重心的连线
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), -1)  # 绿色重心
            cv2.line(img, (screen_center, cy), (cx, cy), (0, 255, 255), 2)
        else:
            # 回退策略：如果区域无法计算重心（例如完全丢失），使用车道线扫描作为退路
            scan_y = int(self.img_h * 0.5)
            scan_line = ll_mask[scan_y, :]
            line_indices = np.where(scan_line > 0)[0]

            left_line_x = None
            right_line_x = None
            if len(line_indices) > 0:
                left_candidates = line_indices[line_indices < screen_center]
                if len(left_candidates) > 0:
                    left_line_x = left_candidates.max()
                right_candidates = line_indices[line_indices > screen_center]
                if len(right_candidates) > 0:
                    right_line_x = right_candidates.min()

            if left_line_x is not None and right_line_x is not None:
                lane_center = (left_line_x + right_line_x) // 2
                offset = lane_center - screen_center
                status = "Tracking (Lines)"
                cv2.line(img, (left_line_x, scan_y), (right_line_x, scan_y), (255, 0, 255), 2)
                cv2.circle(img, (lane_center, scan_y), 8, (0, 255, 255), -1)
            else:
                status = "Lost"

        # 绘制最终偏移量（以当前用于计算的 scan_y 为准）
        if status != "Lost":
            vis_y = cy if cy is not None else scan_y
            cv2.line(img, (screen_center, vis_y), (screen_center + int(offset), vis_y), (0, 255, 255), 2)

        # --- 可视化 ---

        # --- 可视化 ---
        # 创建彩色遮罩
        color_mask = np.zeros_like(img)
        color_mask[da_mask == 1] = [0, 255, 0]  # 绿色：可行驶区域
        color_mask[ll_mask == 1] = [0, 0, 255]  # 红色：车道线
        
        # 叠加到原图
        result_img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)
        
        return result_img, {'offset': offset, 'status': status}
