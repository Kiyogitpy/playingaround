import random
import threading
import time

import interception
import keyboard
import numpy as np
import win32gui
from interception import left_click
from scipy.stats import truncnorm

tick_lock = threading.Lock()
tick_count = 0
last_action = None
interception.auto_capture_devices(keyboard=True, mouse=True)

BOX1 = (26, 1033)
BOX2 = (72, 1033)

active_event = threading.Event()  # Controls whether ticks are processed
active_event.clear()  # Start paused


def get_pixel_color(hwnd, x, y):
    hdc = win32gui.GetWindowDC(hwnd)
    color = win32gui.GetPixel(hdc, x, y)
    win32gui.ReleaseDC(hwnd, hdc)
    r = color & 0xff
    g = (color >> 8) & 0xff
    b = (color >> 16) & 0xff
    return (r, g, b)


def color_diff(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))


def hybrid_delay(mean=0.15, std_dev=0.03, min_val=0.10, max_val=0.18, uniform_chance=0.10):
    if random.random() < uniform_chance:
        delay = random.uniform(min_val, max_val)
    else:
        # Truncated normal to avoid sharp cutoffs
        a, b = (min_val - mean) / std_dev, (max_val - mean) / std_dev
        delay = truncnorm.rvs(a, b, loc=mean, scale=std_dev)
    print(f"[DELAY] {delay:.3f}")
    return delay


def tick_listener():
    global tick_count
    hwnd = win32gui.FindWindow(None, "RuneLite - Kiyoqt")
    if not hwnd:
        print("[ERROR] RuneLite window not found.")
        return

    last_state = None
    while True:
        c1 = get_pixel_color(hwnd, *BOX1)
        c2 = get_pixel_color(hwnd, *BOX2)
        state = (c1, c2)
        if last_state and state != last_state:
            if color_diff(c1, c2) > 100:
                if active_event.is_set():  # Only act if unpaused
                    with tick_lock:
                        print(f"[TICK] Tick {tick_count} at {time.time():.2f}")
                        handle_tick(tick_count)
                        tick_count = (tick_count + 1) % 3
        last_state = state
        time.sleep(0.01)


def handle_tick(tick_id):
    global last_action
    left_click(clicks=2, interval=hybrid_delay())


def toggle_active():
    while True:
        keyboard.wait("p")
        if active_event.is_set():
            active_event.clear()
            print("[STATUS] Paused")
        else:
            active_event.set()
            print("[STATUS] Running")


# ----- Main Execution -----
if __name__ == '__main__':
    threading.Thread(target=toggle_active, daemon=True).start()
    tick_listener()
