# utils/directkeys.py
# From https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# (Modified for our use case)

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Key scancodes
KEY_W = 0x11
KEY_A = 0x1E
KEY_S = 0x1F
KEY_D = 0x20
KEY_SPACE = 0x39

KEY_MAP = {
    'W': KEY_W,
    'A': KEY_A,
    'S': KEY_S,
    'D': KEY_D,
    'SPACE': KEY_SPACE,
}

# Actual functions to press and release keys
def press_key(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def press_key_by_name(key_name):
    """
    Presses a key given its string name (e.g., 'W', 'A').
    """
    if key_name.upper() in KEY_MAP:
        press_key(KEY_MAP[key_name.upper()])
    else:
        print(f"Warning: Key '{key_name}' not found in KEY_MAP.")

def release_key_by_name(key_name):
    """
    Releases a key given its string name (e.g., 'W', 'A').
    """
    if key_name.upper() in KEY_MAP:
        release_key(KEY_MAP[key_name.upper()])
    else:
        print(f"Warning: Key '{key_name}' not found in KEY_MAP.")

if __name__ == '__main__':
    # Example usage:
    print("Testing key presses... Pressing 'W' for 2 seconds.")
    press_key_by_name('W')
    time.sleep(2)
    release_key_by_name('W')
    print("Test complete.")
    time.sleep(1)

    print("Testing steering... Left for 1s, then Right for 1s.")
    press_key_by_name('A')
    time.sleep(1)
    release_key_by_name('A')
    time.sleep(0.5)
    press_key_by_name('D')
    time.sleep(1)
    release_key_by_name('D')
    print("Test complete.")
