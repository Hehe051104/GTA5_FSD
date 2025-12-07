# modules/perception/yolo_service.py

from ultralytics import YOLO
import torch

class YoloService:
    def __init__(self, model_path='models/yolov8n.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 YOLO Service (Turbo Mode) ... Device: {self.device}")

        try:
            self.model = YOLO(model_path)
        except Exception:
            self.model = YOLO('yolov8n.pt')

    def detect(self, frame):
        """
        极速推理模式
        """
        # [优化] half=True: 开启半精度 (FP16)，RTX 4060 提速神器
        # [优化] imgsz=640: 强制模型只看 640x640 的大小，不管输入图多大
        results = self.model.predict(frame,
                                     device=self.device,
                                     imgsz=640,
                                     half=True,  # <--- 开启半精度
                                     verbose=False,
                                     conf=0.5)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = self.model.names[cls_id]

                # 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light, 0:person
                if cls_id in [2, 3, 5, 7, 9, 0]:
                    detections.append({
                        'label': label,
                        'conf': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    })
        return detections