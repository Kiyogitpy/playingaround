import math
import random
import threading
import time

import cv2
import interception
import keyboard
import numpy as np
import pyautogui
import pygetwindow as gw
import pytweening
import win32con
import win32gui
import win32ui
from interception import beziercurve, left_click, move_to
from pywinauto import Desktop
from scipy.stats import truncnorm

# ----- Global Variables -----
shutdown_event = threading.Event()
notification_event = threading.Event()
inventory_event = threading.Event()
# Global flag to signal that the high priority deposit routine is active.
priority_deposit_active = threading.Event()
# ----- Interception Setup -----
interception.auto_capture_devices(keyboard=True, mouse=True)
beziercurve.set_default_params(beziercurve.BezierCurveParams(
    knots=1,
    distortion_mean=1,
    distortion_stdev=0.5,
    distortion_frequency=0.1,
    tween=pytweening.easeOutQuad,
    target_points=100
))


# ----- Utility Functions -----
def get_game_window():
    windows = gw.getWindowsWithTitle("RuneLite - Kiyoqt")
    return windows[0]._hWnd if windows else None


def get_pixel_color(hwnd, x, y):
    hdc = win32gui.GetWindowDC(hwnd)
    color = win32gui.GetPixel(hdc, x, y)
    win32gui.ReleaseDC(hwnd, hdc)
    return (color & 0xff, (color >> 8) & 0xff, (color >> 16) & 0xff)


def is_inventory_full(hwnd):
    return get_pixel_color(hwnd, 2413, 1288) != (62, 53, 41)


def hybrid_delay(mean=0.15, std_dev=0.03, min_val=0.10, max_val=0.18, uniform_chance=0.10):
    if random.random() < uniform_chance:
        delay = random.uniform(min_val, max_val)
    else:
        a, b = (min_val - mean) / std_dev, (max_val - mean) / std_dev
        delay = truncnorm.rvs(a, b, loc=mean, scale=std_dev)
    print(f"[DELAY] {delay:.3f}")
    return delay


def capture_window(hwnd):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w, h = right - left, bottom - top
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, w, h)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (w, h), srcdc, (0, 0), win32con.SRCCOPY)
    img = np.frombuffer(bmp.GetBitmapBits(
        True), dtype=np.uint8).reshape((h, w, 4))
    win32gui.DeleteObject(bmp.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def detect_color(img, target_color):
    # Convert the target RGB to BGR (OpenCV uses BGR by default)
    target_bgr = np.array(
        [target_color[2], target_color[1], target_color[0]], dtype=np.uint8)

    # Create a mask where all pixels that match the target color are set to 255 (white), others are 0 (black)
    mask = cv2.inRange(img, target_bgr, target_bgr)

    # Find contours of the matching regions
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours, mask


def draw_bounding_boxes_and_return_centers(img, contours):
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            centers.append((x + w // 2, y + h // 2))
    return img, centers


def compute_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_custom_curve_params(start, end):
    dist = compute_distance(start, end)
    if dist < 50:
        return beziercurve.BezierCurveParams(1, 0.05, 0.05, 0.02, pytweening.easeOutQuad, 15)
    elif dist < 150:
        return beziercurve.BezierCurveParams(1, 0.2, 0.1, 0.05, pytweening.easeOutQuad, 25)
    else:
        return beziercurve.BezierCurveParams(1, 0.5, 0.3, 0.1, pytweening.easeOutQuad, 40)


def humanized_move_to(target):
    start = pyautogui.position()
    params = get_custom_curve_params(start, target)
    move_to(target, curve_params=params, allow_global_params=False)


def jittered_position(pos, amount=5):
    return (pos[0] + random.randint(-amount, amount), pos[1] + random.randint(-amount, amount))


def template_matching(template_path):
    hwnd = get_game_window()
    if hwnd:
        img = capture_window(hwnd)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.91  # Lowered threshold slightly
        locations = np.where(result >= threshold)

        w, h = template.shape[::-1]
        match_coords = []

        for i, pt in enumerate(zip(*locations[::-1])):
            center_x = pt[0] + w // 2
            center_y = pt[1] + h // 2
            match_coords.append((int(center_x), int(center_y)))
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

        return img, match_coords

# ----- Threads -----


def shutdown_listener():
    print("[INFO] Press F10 to stop the script.")
    keyboard.wait('F10')
    print("[SHUTDOWN] F10 pressed. Exiting...")
    shutdown_event.set()


# ----- Inventory Listener -----
def inventory_listener():
    while not shutdown_event.is_set():
        # Do nothing if the high-priority deposit procedure is active.
        if priority_deposit_active.is_set():
            time.sleep(0.5)
            continue

        hwnd = get_game_window()
        if hwnd:
            full = is_inventory_full(hwnd)
            if full and not inventory_event.is_set():
                inventory_event.set()
                print("[INFO] Inventory full – event set")
            elif not full and inventory_event.is_set():
                inventory_event.clear()
                print("[INFO] Inventory empty – event cleared")
                notification_event.set()
        time.sleep(0.5)

# ----- Inventory Click Loop (Handles Teal/Green steps) -----


def inventory_click_loop():
    while not shutdown_event.is_set():
        # Do nothing if high-priority deposit routine is active.
        if priority_deposit_active.is_set():
            time.sleep(0.2)
            continue

        inventory_event.wait()
        hwnd = get_game_window()
        if hwnd:
            teal_clicked = False
            teal_wait_start = None
            green_clicked = False
            green_click_time = None
            green_click_done = False

            while inventory_event.is_set() and not shutdown_event.is_set():
                # Priority check inside the inner loop too:
                if priority_deposit_active.is_set():
                    break  # Yield control to the higher-priority routine

                img = capture_window(hwnd)
                teal_color = (0, 255, 255)
                teal_contours, _ = detect_color(img, teal_color)
                _, teal_centers = draw_bounding_boxes_and_return_centers(
                    img, teal_contours)

                if not teal_clicked:
                    if len(teal_centers) >= 2:
                        teal_center = teal_centers[0]
                        humanized_move_to(jittered_position(teal_center))
                        time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                        left_click()
                        print(f"[CLICK] Clicked on teal at: {teal_center}")
                        teal_clicked = True
                        teal_wait_start = time.time()
                    else:
                        print(
                            "[INFO] Less than 2 teal objects — skipping teal step")
                        teal_clicked = True
                        green_clicked = True
                        green_click_time = time.time()

                # New retry condition for teal fix
                elif teal_clicked and len(teal_centers) >= 2 and (time.time() - teal_wait_start > 5):
                    print("[RETRY] Teal fix was not successful, retrying teal click")
                    teal_clicked = False  # Reset teal click flag to force retry

                elif teal_clicked and not green_clicked and len(teal_centers) < 2:
                    green_clicked = True
                    green_click_time = time.time()
                    print("[INFO] Teal condition complete — moving to green")

                elif teal_clicked and not green_clicked and (time.time() - teal_wait_start > 15):
                    print("[FAILSAFE] Teal step timed out — retrying teal click")
                    teal_clicked = False  # Retry the teal process

                # ----- Green Processing (Empty Inventory) -----
                if green_clicked and not green_click_done:
                    green_color = (0, 255, 0)
                    green_contours, _ = detect_color(img, green_color)
                    _, green_centers = draw_bounding_boxes_and_return_centers(
                        img, green_contours)
                    if green_centers:
                        green_center = green_centers[0]
                        humanized_move_to(jittered_position(green_center))
                        time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                        left_click()
                        green_click_done = True
                        print(
                            f"[CLICK] Clicked on green to empty inventory at: {green_center}")
                    elif time.time() - green_click_time > 15:
                        print(
                            "[FAILSAFE] Green step timed out — retrying teal detection")
                        teal_clicked = False
                        green_clicked = False
                        green_click_done = False

                # Check if inventory is empty and exit if done.
                if green_click_done and not is_inventory_full(hwnd):
                    print(
                        "[INFO] Inventory is now empty — setting notification_event")
                    inventory_event.clear()
                    notification_event.set()
                    break

                time.sleep(0.5)
        else:
            print("[WARNING] No valid game window handle detected.")
        inventory_event.clear()


# ----- Anti‐Idle Listener -----


def anti_idle_listener():
    last_handle = None
    cooldown = 8
    last_time = 0
    while not shutdown_event.is_set():
        # Yield if a high-priority deposit routine is active.
        if priority_deposit_active.is_set():
            time.sleep(0.5)
            continue

        hwnd = get_game_window()
        if hwnd and is_inventory_full(hwnd):
            time.sleep(0.5)
            continue

        try:
            for win in Desktop(backend="uia").windows():
                if priority_deposit_active.is_set():
                    break  # If high priority routine starts, break out.
                title = win.window_text().strip().lower()
                if "notification" in title and "mozilla firefox" not in title and "toasts" not in title:
                    hwnd_notify = win.handle
                    now = time.time()
                    if hwnd_notify != last_handle or now - last_time > cooldown:
                        print(f"[NOTIFY] {win.window_text()}")
                        notification_event.set()
                        last_handle = hwnd_notify
                        last_time = now
                        break
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(0.5)

# ----- Notification Click Loop (Minig ore with Magenta marker) -----


def notification_click_loop():
    while not shutdown_event.is_set():
        # Only run when no high-priority deposit is active.
        if priority_deposit_active.is_set():
            time.sleep(0.2)
            continue
        if inventory_event.is_set():
            time.sleep(0.2)
            continue

        notification_event.wait()
        if shutdown_event.is_set() or inventory_event.is_set() or priority_deposit_active.is_set():
            continue

        hwnd = get_game_window()
        if hwnd:
            while not shutdown_event.is_set() and not inventory_event.is_set():
                if priority_deposit_active.is_set():
                    break
                time.sleep(hybrid_delay(1.0, min_val=0.5, max_val=2))
                img = capture_window(hwnd)
                # Assume magenta (255, 0, 255) is used for ore mining.
                contours, _ = detect_color(img, (255, 0, 255))
                _, centers = draw_bounding_boxes_and_return_centers(
                    img, contours)
                if centers:
                    target = jittered_position(centers[0])
                    humanized_move_to(target)
                    time.sleep(hybrid_delay(0.075, min_val=0.05, max_val=0.1))
                    left_click()
                    print(
                        f"[CLICK] Clicked on magenta ore target at: {centers[0]}")
                    break
                else:
                    print("[INFO] Waiting for magenta target to appear...")
        notification_event.clear()

# ----- High-Priority Deposit Procedure triggered by Red Detection -----


def red_to_blue_yellow_deposit_loop():
    while not shutdown_event.is_set():
        hwnd = get_game_window()
        if hwnd is None:
            time.sleep(0.5)
            continue

        # Yield if another high-priority operation is in progress.
        if priority_deposit_active.is_set():
            time.sleep(0.5)
            continue

        # Check for red trigger at (144,127).
        red_target = (255, 0, 0)
        current_color = get_pixel_color(hwnd, 144, 127)
        if current_color == red_target and not is_inventory_full(hwnd):
            priority_deposit_active.set()
            print(
                "[PRIORITY] Red detected at (144,127). Initiating high-priority deposit procedure.")

            # ----- Step 1: Pickup Ore from Blue Marker -----
            blue_marker_rgb = (0, 0, 255)
            pickup_success = False
            pickup_timeout = 10
            start_pickup = time.time()
            while not shutdown_event.is_set() and (time.time() - start_pickup < pickup_timeout):
                img = capture_window(hwnd)
                blue_contours, _ = detect_color(img, blue_marker_rgb)
                _, blue_centers = draw_bounding_boxes_and_return_centers(
                    img, blue_contours)
                if blue_centers:
                    blue_center = blue_centers[0]
                    humanized_move_to(jittered_position(blue_center))
                    time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                    left_click()
                    print(
                        f"[PRIORITY CLICK] Picked up ore from blue marker at {blue_center}.")
                    pickup_success = True
                    break
                time.sleep(0.5)

            # ----- Step 2: Wait Until Inventory Is Full -----
            if pickup_success:
                print("[PRIORITY] Waiting for inventory to be full...")
                while not shutdown_event.is_set() and not is_inventory_full(hwnd):
                    time.sleep(0.5)
                print("[PRIORITY] Inventory full: Ore successfully picked up.")

                # ----- Step 3: Deposit Ore at Yellow Marker -----
                yellow_marker_rgb = (255, 255, 0)
                deposit_success = False
                deposit_timeout = 10
                start_deposit = time.time()
                while not shutdown_event.is_set() and (time.time() - start_deposit < deposit_timeout):
                    img = capture_window(hwnd)
                    yellow_contours, _ = detect_color(img, yellow_marker_rgb)
                    _, yellow_centers = draw_bounding_boxes_and_return_centers(
                        img, yellow_contours)
                    if yellow_centers:
                        yellow_center = yellow_centers[0]
                        humanized_move_to(jittered_position(yellow_center))
                        time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                        left_click()
                        print(
                            f"[PRIORITY CLICK] Deposited ore at yellow marker at {yellow_center}.")
                        deposit_success = True
                        break
                    else:
                        print("[PRIORITY] Waiting for yellow deposit marker...")
                        time.sleep(0.5)

                # ----- Step 4: Detect UI and Click Deposit Button -----
                if deposit_success:
                    print("[PRIORITY] Waiting for deposit UI to appear...")
                    ui_timeout = 10
                    start_ui = time.time()
                    deposit_clicked = False

                    while not shutdown_event.is_set() and (time.time() - start_ui < ui_timeout):
                        img, deposit_coords = template_matching(
                            "needle_deposit.png")
                        if deposit_coords:
                            deposit_button = deposit_coords[0]
                            humanized_move_to(
                                jittered_position(deposit_button))
                            time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                            left_click()
                            print(
                                f"[PRIORITY CLICK] Clicked deposit button at {deposit_button}.")
                            deposit_clicked = True
                            break
                        else:
                            print(
                                "[PRIORITY] Waiting for deposit button to appear...")
                            time.sleep(0.5)

                    if deposit_clicked:
                        # Step 5: Wait Until Inventory Is Empty After Deposit
                        print(
                            "[PRIORITY] Waiting for inventory to be empty after deposit...")
                        while not shutdown_event.is_set() and is_inventory_full(hwnd):
                            time.sleep(0.5)
                        print(
                            "[PRIORITY] Deposit complete, inventory is now empty.")
                    else:
                        print(
                            "[PRIORITY FAILSAFE] Deposit button not detected within timeout.")

            # Clear the high-priority flag so that normal operations may resume.
            priority_deposit_active.clear()
            notification_event.set()
        else:
            time.sleep(0.5)


# ----- Main -----
if __name__ == '__main__':
    # ----- Thread Startup Section -----
    # Start each thread as a daemon so that they exit when the main thread does.
    threading.Thread(target=inventory_listener, daemon=True).start()
    threading.Thread(target=inventory_click_loop, daemon=True).start()
    threading.Thread(target=anti_idle_listener, daemon=True).start()
    threading.Thread(target=notification_click_loop, daemon=True).start()
    threading.Thread(target=red_to_blue_yellow_deposit_loop,
                     daemon=True).start()
    shutdown_thread = threading.Thread(target=shutdown_listener, daemon=True)
    shutdown_thread.start()
    shutdown_thread.join()
