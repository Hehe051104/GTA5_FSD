# modules/control/car_controller.py

import time
from pynput.keyboard import Controller
from modules.control.pid import PIDController
from config import PID_Kp, PID_Ki, PID_Kd, PID_DEADBAND, MAX_KEY_DURATION

class CarController:
    def __init__(self):
        self.keyboard = Controller()
        self.pid = PIDController(PID_Kp, PID_Ki, PID_Kd, PID_DEADBAND)

    def update_steering(self, offset):
        control_value = self.pid.update(offset)

        if control_value == 0:
            return

        duration = abs(control_value) / 1000.0
        duration = max(0.02, min(duration, MAX_KEY_DURATION))

        if control_value > 0:
            self.keyboard.press('a')
            time.sleep(duration)
            self.keyboard.release('a')

        elif control_value < 0:
            self.keyboard.press('d')
            time.sleep(duration)
            self.keyboard.release('d')