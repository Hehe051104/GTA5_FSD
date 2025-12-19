import numpy as np

class CollisionMonitor:
    def __init__(self, expansion_threshold=1.05, history_len=5, min_width=20):
        """
        :param expansion_threshold: 宽度增长阈值 (例如 1.05 表示 5 帧内增长 5%)
        :param history_len: 历史记录长度 (帧数)
        :param min_width: 最小目标宽度 (过滤噪点)
        """
        self.tracked_objects = {} # {id: {'bbox': [x1,y1,x2,y2], 'history': [...], 'label': str}}
        self.next_id = 0
        self.expansion_threshold = expansion_threshold
        self.history_len = history_len
        self.min_width = min_width

    def update(self, detections, lane_roi):
        """
        更新追踪并检测危险
        :param detections: YOLO 检测结果列表
        :param lane_roi: 当前车道 ROI {'x_min', 'x_max', 'y_min', 'y_max'}
        :return: (is_danger, danger_obj, debug_info)
        """
        # 1. 筛选车道内的目标 (只关注前方车辆)
        in_lane_dets = []
        for det in detections:
            if det['label'] not in ['car', 'truck', 'bus', 'motorcycle']:
                continue
                
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = y2
            w = x2 - x1
            
            if w < self.min_width:
                continue
            
            # 判定中心点是否在车道 ROI 内
            # 注意：这里只关心横向 (X轴) 是否在车道内，纵向 (Y轴) 只要在视野内即可
            if (lane_roi['x_min'] < cx < lane_roi['x_max']):
                in_lane_dets.append(det)

        # 2. 简单的目标关联 (基于中心点距离)
        current_objects = {}
        used_dets = set()
        
        # 尝试匹配已有目标
        for obj_id, obj_data in self.tracked_objects.items():
            prev_bbox = obj_data['bbox']
            prev_cx = (prev_bbox[0] + prev_bbox[2]) // 2
            prev_cy = (prev_bbox[1] + prev_bbox[3]) // 2
            
            best_match_idx = -1
            min_dist = float('inf')
            
            for i, det in enumerate(in_lane_dets):
                if i in used_dets: continue
                
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                
                # 匹配阈值 (例如 50 像素)
                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                # 更新目标
                det = in_lane_dets[best_match_idx]
                used_dets.add(best_match_idx)
                
                new_history = obj_data['history'] + [det['bbox']]
                if len(new_history) > self.history_len:
                    new_history.pop(0)
                    
                current_objects[obj_id] = {
                    'bbox': det['bbox'],
                    'history': new_history,
                    'label': det['label']
                }

        # 添加新目标
        for i, det in enumerate(in_lane_dets):
            if i not in used_dets:
                current_objects[self.next_id] = {
                    'bbox': det['bbox'],
                    'history': [det['bbox']],
                    'label': det['label']
                }
                self.next_id += 1
        
        self.tracked_objects = current_objects
        
        # 3. 危险分析 (基于宽度膨胀率)
        is_danger = False
        danger_obj = None
        debug_info = []
        
        for obj_id, obj_data in self.tracked_objects.items():
            history = obj_data['history']
            if len(history) < 3: # 至少需要 3 帧数据
                continue
                
            # 计算宽度变化
            # 比较当前帧和历史第一帧
            curr_w = history[-1][2] - history[-1][0]
            prev_w = history[0][2] - history[0][0]
            
            if prev_w == 0: continue
            
            # 膨胀率
            expansion_rate = curr_w / prev_w
            
            # 绝对距离检查 (作为兜底)
            # 假设 y_max 越大越近 (720 是底部)
            curr_y = history[-1][3]
            
            # 危险判定逻辑：
            # 1. 正在快速变大 (expansion_rate > 阈值) 且距离较近
            # 2. 或者距离已经非常近 (无论是否变大)
            
            is_expanding = expansion_rate > self.expansion_threshold
            
            # 距离阈值定义
            # y_max 通常是 ROI 的底部 (例如 570 左右)
            # 0.6 * 570 = 342 (屏幕中部)
            # 0.8 * 570 = 456 (屏幕中下部)
            
            # 警告距离：进入中距离范围 (0.55 = 55% 高度)
            is_in_warning_zone = curr_y > (lane_roi['y_max'] * 0.55) 
            
            # 刹车距离：进入近距离范围 (强制刹车)
            # [修改] 提高灵敏度：只要物体底部超过 ROI 高度的 65%，就视为必须刹车
            is_in_brake_zone = curr_y > (lane_roi['y_max'] * 0.65)
            
            debug_msg = f"ID:{obj_id} R:{expansion_rate:.2f} Y:{curr_y}"
            debug_info.append(debug_msg)
            
            # 逻辑修改：
            # 如果在刹车距离内，直接危险 (忽略膨胀率)
            # 如果在警告距离内，且正在快速接近 (膨胀)，也危险
            if is_in_brake_zone or (is_in_warning_zone and is_expanding):
                is_danger = True
                danger_obj = obj_data
                # 找到一个危险目标就足够了
                break
                
        return is_danger, danger_obj, debug_info
