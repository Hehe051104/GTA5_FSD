# modules/control/pid.py

import time

class PIDController:
    def __init__(self, kp, ki, kd, deadband=0):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.deadband = deadband
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.03

        if abs(error) < self.deadband:
            self.integral = 0
            self.last_error = 0
            self.last_time = current_time
            return 0

        self.integral += error * dt
        self.integral = max(min(self.integral, 500), -500)

        derivative = (error - self.last_error) / dt
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.last_error = error
        self.last_time = current_time

        return output