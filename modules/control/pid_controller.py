# modules/control/pid_controller.py

import time
import threading
from utils.directkeys import press_key_by_name, release_key_by_name

class PIDController:
    def __init__(self, Kp, steering_threshold, turn_cooldown):
        self.Kp = Kp
        self.steering_threshold = steering_threshold
        self.turn_cooldown = turn_cooldown
        self.last_action_time = 0

    def _execute_turn(self, key, duration):
        """在一个单独的线程中执行点按操作以避免阻塞主循环"""
        def turn_action():
            press_key_by_name(key)
            time.sleep(duration)
            release_key_by_name(key)
            # print(f"Action: Tapped '{key}' for {duration:.3f}s")

        # 使用线程来执行非阻塞的按键操作
        turn_thread = threading.Thread(target=turn_action)
        turn_thread.start()

    def update(self, offset):
        """
        根据偏移量更新转向控制 (点按模式)
        :param offset: 车辆与车道中心的偏移量 (像素)
                       - 负数: 车在车道线左边 (需要向右转 -> D)
                       - 正数: 车在车道线右边 (需要向左转 -> A)
        """
        current_time = time.time()

        # 1. 检查是否在冷却时间内
        if current_time - self.last_action_time < self.turn_cooldown:
            return # 冷却中，不执行任何操作

        error = offset

        # 2. 检查是否在死区内
        if abs(error) < self.steering_threshold:
            return # 偏移量很小，无需转向

        # 3. 计算点按时长和方向
        # 点按时长与误差成正比，但设置一个上限和下限，防止过长或过短
        press_duration = abs(error) * self.Kp
        # 调整最大按键时长，配合冷却时间。单次最长按 0.15s
        press_duration = max(0.02, min(press_duration, 0.15)) 

        # 修正转向逻辑：
        # offset > 0 表示目标在右边 (车偏左)，需要向右转 (D)
        # offset < 0 表示目标在左边 (车偏右)，需要向左转 (A)
        target_key = 'D' if error > 0 else 'A'
        
        self._execute_turn(target_key, press_duration)
        self.last_action_time = current_time
        
        return f"Turn {target_key} ({press_duration:.2f}s)"

    def get_action(self, offset):
        """
        兼容旧接口的别名
        """
        result = self.update(offset)
        return result if result else "Straight"

        # 4. 执行转向并更新时间戳
        self._execute_turn(target_key, press_duration)
        self.last_action_time = current_time

    def stop(self):
        """停止时确保所有按键都被释放 (虽然点按模式下理论上已经释放)"""
        release_key_by_name('A')
        release_key_by_name('D')
        print("Controller stopped, all keys released.")
