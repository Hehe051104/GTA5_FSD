# collect_data.py# collect_data.py

# 数据采集脚本 - 用于训练 PilotNet# 数据采集脚本 - 记录屏幕画面和按键操作

# 运行此脚本，然后在游戏中手动驾驶。脚本会记录屏幕画面和你的按键操作。

import cv2

import cv2import time

import numpy as npimport os

import timeimport numpy as np

import osimport win32api

import pandas as pdfrom utils.screen_grab import ScreenGrabber

from utils.screen_grab import ScreenGrabberfrom datetime import datetime

import win32api

# 配置

# 采样率 (每秒保存多少帧)DATA_DIR = 'training_data'

SAMPLE_RATE = 10 VIEW_WIDTH = 320  # 缩小尺寸以节省空间和加快训练 (PilotNet标准输入)

DATA_DIR = 'training_data'VIEW_HEIGHT = 160

FPS_TARGET = 10

def key_check(key_code):

    return win32api.GetAsyncKeyState(key_code)def key_check():

    """检查按键状态 (A/D)"""

def get_steering():    keys = []

    # 简单的按键映射    # Win32 API key codes: A=0x41, D=0x44

    # A = -1 (Left), D = 1 (Right), None = 0    if win32api.GetAsyncKeyState(0x41): keys.append('A')

    # 这种离散数据训练出来的模型会比较生硬，最好是用手柄采集连续值    if win32api.GetAsyncKeyState(0x44): keys.append('D')

    # 但为了演示，我们先用键盘    return keys

    if key_check(0x41): # A

        return -1.0def main():

    elif key_check(0x44): # D    if not os.path.exists(DATA_DIR):

        return 1.0        os.makedirs(DATA_DIR)

    return 0.0        

    print("🎥 数据采集模式启动...")

def main():    print("="*40)

    if not os.path.exists(DATA_DIR):    print(f"数据将保存在: {DATA_DIR}")

        os.makedirs(DATA_DIR)    print("按 'T' 开始/暂停 录制")

        os.makedirs(os.path.join(DATA_DIR, 'images'))    print("按 'Q' 退出")

    print("="*40)

    print("🎥 数据采集模式启动...")    

    print("请在 GTA5 中手动驾驶。")    grabber = ScreenGrabber()

    print("按 'T' 开始/暂停记录")    paused = True

    print("按 'Q' 退出")    file_name = f'{DATA_DIR}/training_data_{int(time.time())}.npy'

    training_data = []

    grabber = ScreenGrabber()    

    recording = False    last_time = time.time()

    frame_count = 0    

    data_log = []    while True:

            # 1. 捕获屏幕

    # 避免覆盖旧数据        screen = grabber.get_frame()

    existing_csv = os.path.join(DATA_DIR, 'driving_log.csv')        if screen is None: continue

    if os.path.exists(existing_csv):        

        try:        # 2. 预处理 (缩小 + 灰度化可选，这里保留彩色)

            df = pd.read_csv(existing_csv)        screen = cv2.resize(screen, (VIEW_WIDTH, VIEW_HEIGHT))

            frame_count = len(df)        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB) # 转RGB适配PyTorch

            print(f"发现已有数据，从帧号 {frame_count} 继续...")        

        except:        # 3. 获取按键

            pass        keys = key_check()

        

    last_time = time.time()        # 编码输出: [A, D] -> One-hot or Scalar

        # [1, 0, 0] = Left (A)

    while True:        # [0, 1, 0] = Straight (None)

        # 控制逻辑        # [0, 0, 1] = Right (D)

        if key_check(0x51): # Q        output = [0, 1, 0] # 默认直行

            break        

        if key_check(0x54): # T        if 'A' in keys:

            recording = not recording            output = [1, 0, 0]

            print(f"记录状态: {'🔴 录制中' if recording else '⏸️ 暂停'}")        elif 'D' in keys:

            time.sleep(0.3)            output = [0, 0, 1]

            

        if not recording:        # 4. 录制逻辑

            time.sleep(0.1)        if win32api.GetAsyncKeyState(ord('T')) & 0x0001: # Toggle

            continue            paused = not paused

            if paused:

        # 限制采样率                print(f"⏸️ 暂停录制. 当前数据量: {len(training_data)}")

        if time.time() - last_time < (1.0 / SAMPLE_RATE):                # 暂停时保存一次，防止丢失

            continue                if len(training_data) > 0:

        last_time = time.time()                    np.save(file_name, training_data)

                    print(f"💾 已保存 {len(training_data)} 条数据到 {file_name}")

        # 1. 抓图                    training_data = [] # 清空内存，准备存下一个文件

        frame = grabber.get_frame()                    file_name = f'{DATA_DIR}/training_data_{int(time.time())}.npy'

        if frame is None: continue            else:

                        print("🔴 开始录制! 请开始驾驶...")

        # 2. 获取转向标签                

        steering = get_steering()        if not paused:

                    training_data.append([screen, output])

        # 3. 保存            

        # 调整大小以节省空间 (PilotNet 输入是 200x66，我们存稍微大一点方便后续处理)            # 每 1000 帧自动保存一次

        # 只保留下半部分            if len(training_data) % 1000 == 0:

        h, w = frame.shape[:2]                print(f"📊 已采集 {len(training_data)} 帧...")

        roi = frame[int(h*0.3):, :]                 np.save(file_name, training_data)

        resized = cv2.resize(roi, (320, 160))                

                # 5. 显示预览

        filename = f"images/frame_{frame_count}_{int(time.time())}.jpg"        # 转回 BGR 显示

        full_path = os.path.join(DATA_DIR, filename)        preview = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

                

        cv2.imwrite(full_path, resized)        # 在画面上画出当前记录的操作

                if not paused:

        # 记录日志: 图片路径, 转向角度            cv2.circle(preview, (20, 20), 10, (0, 0, 255), -1) # 红点录制中

        data_log.append([filename, steering])            if output == [1, 0, 0]: cv2.putText(preview, "LEFT", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    elif output == [0, 0, 1]: cv2.putText(preview, "RIGHT", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_count += 1            else: cv2.putText(preview, "STRAIGHT", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if frame_count % 100 == 0:        else:

            print(f"已采集 {frame_count} 帧数据...")            cv2.putText(preview, "PAUSED (Press T)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 定期保存 CSV 防止丢失            

            df = pd.DataFrame(data_log, columns=['image', 'steering'])        cv2.imshow('Data Collector', preview)

            # 追加模式写入有点麻烦，这里简单覆盖        

            # 实际生产中建议 append        # 控制帧率

            if os.path.exists(existing_csv):        loop_time = time.time() - last_time

                df.to_csv(existing_csv, mode='a', header=False, index=False)        if loop_time < 1/FPS_TARGET:

            else:            time.sleep(1/FPS_TARGET - loop_time)

                df.to_csv(existing_csv, index=False)        last_time = time.time()

            data_log = [] # 清空缓存        

        if cv2.waitKey(1) & 0xFF == ord('q'):

    # 最后保存剩余数据            break

    if data_log:            

        df = pd.DataFrame(data_log, columns=['image', 'steering'])    # 退出前保存

        if os.path.exists(existing_csv):    if len(training_data) > 0:

            df.to_csv(existing_csv, mode='a', header=False, index=False)        np.save(file_name, training_data)

        else:        print(f"💾 最终保存 {len(training_data)} 条数据")

            df.to_csv(existing_csv, index=False)        

    cv2.destroyAllWindows()

    print(f"采集结束。总计 {frame_count} 帧。")

if __name__ == "__main__":

if __name__ == "__main__":    main()

    main()
