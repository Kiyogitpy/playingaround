from scipy.stats import truncnorm
import math
import random
import threading
import time

import cv2
import interception
import keyboard
import numpy as np
import pyautogui  # For getting the current mouse position
import pygetwindow as gw
import pytweening
import win32con
import win32gui
import win32ui
from interception import beziercurve, left_click, move_to, press
from pywinauto import Desktop

# ----- Global Variables and Setup -----
BLACK_RGB = (0, 0, 0)
BOX1 = (26, 1033)
BOX2 = (72, 1033)

shutdown_event = threading.Event()
notification_event = threading.Event()
tick_lock = threading.Lock()
tick_count = 0
last_action = None  # Global variable to track tick 0 action

# Setup interception with default Bezier parameters for longer movements.
interception.auto_capture_devices(keyboard=True, mouse=True)
custom_params = beziercurve.BezierCurveParams(
    knots=1,
    distortion_mean=1,
    distortion_stdev=0.5,
    distortion_frequency=0.1,
    tween=pytweening.easeOutQuad,
    target_points=100
)
beziercurve.set_default_params(custom_params)

# ----- Helper Functions -----


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

# --- Dynamic Humanization Helpers ---


def compute_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_custom_curve_params(start, end, default_params):
    distance = compute_distance(start, end)
    if distance < 50:
        # For very short moves: very smooth and nearly straight.
        return beziercurve.BezierCurveParams(
            knots=1,
            distortion_mean=0.05,       # Further reduced distortion
            distortion_stdev=0.05,      # Less variability – keeps the curve mild
            distortion_frequency=0.02,  # Low frequency of distortion
            tween=pytweening.easeOutQuad,
            target_points=15            # Increased target points for smooth interpolation
        )
    elif distance < 150:
        # For moderate moves: subtle distortion while maintaining smoothness.
        return beziercurve.BezierCurveParams(
            knots=1,
            distortion_mean=0.2,        # Lower mean for gentle curvature
            distortion_stdev=0.1,       # Lower stdev to avoid steep changes
            distortion_frequency=0.05,  # Subtle disturbance along the curve
            tween=pytweening.easeOutQuad,
            target_points=25            # Sufficient points for a smooth but gradual curve
        )
    else:
        # For longer moves, use a slightly modified set for consistency.
        return beziercurve.BezierCurveParams(
            knots=1,
            distortion_mean=0.5,        # Moderate distortion for longer distances
            distortion_stdev=0.3,
            distortion_frequency=0.1,
            tween=pytweening.easeOutQuad,
            target_points=40            # More points to maintain smoothness over longer paths
        )


def humanized_move_to(target):
    """Moves the mouse to the target location using dynamically adjusted human-like parameters."""
    current = pyautogui.position()  # Get current mouse position
    default_params = beziercurve.BezierCurveParams(
        knots=1,
        distortion_mean=1,
        distortion_stdev=0.5,
        distortion_frequency=0.1,
        tween=pytweening.easeOutQuad,
        target_points=100
    )
    custom_params = get_custom_curve_params(current, target, default_params)
    move_to(target, curve_params=custom_params, allow_global_params=False)


def jittered_position(position, jitter_amount=5):
    """Returns a slightly randomized target based on the original 'position'."""
    return (
        position[0] + random.randint(-jitter_amount, jitter_amount),
        position[1] + random.randint(-jitter_amount, jitter_amount)
    )

# ----- Image Capture & Processing Functions -----


def get_game_window():
    windows = gw.getWindowsWithTitle("RuneLite - Kiyoqt")
    if windows:
        return windows[0]._hWnd
    return None


def capture_window(hwnd):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)
    bmp_info = bmp.GetInfo()
    img = np.frombuffer(bmp.GetBitmapBits(True), dtype=np.uint8).reshape(
        (bmp_info['bmHeight'], bmp_info['bmWidth'], 4)
    )
    win32gui.DeleteObject(bmp.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def template_matching(template_path):
    hwnd = get_game_window()
    if hwnd:
        img = capture_window(hwnd)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.92  # Adjust this threshold if needed
        locations = np.where(result >= threshold)
        w, h = template.shape[::-1]
        match_coords = []
        for pt in zip(*locations[::-1]):
            center_x = pt[0] + w // 2
            center_y = pt[1] + h // 2
            match_coords.append((int(center_x), int(center_y)))
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        return img, match_coords
    return None, []


def detect_color(img, target_color, tolerance=0):
    target_bgr = np.array(
        [target_color[2], target_color[1], target_color[0]], dtype=np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(
        np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    lower_bound = np.array([target_hsv[0] - tolerance, 50, 50], dtype=np.uint8)
    upper_bound = np.array(
        [target_hsv[0] + tolerance, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask


def draw_bounding_boxes_and_return_centers(img, contours):
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
            cv2.putText(img, f"Center: ({center_x}, {center_y})", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img, centers


def get_grid_coordinates():
    x0, y0 = 2146, 835
    x5, y6 = 2408, 1279
    cols = 4
    rows = 7
    x_step = (x5 - x0) // (cols - 1)
    y_step = (y6 - y0) // (rows - 1)
    coords = []
    for row in range(rows):
        for col in range(cols):
            x = x0 + col * x_step
            y = y0 + row * y_step
            coords.append((x, y))
    return coords


def draw_grid_coordinates(img):
    coords = get_grid_coordinates()
    for (x, y) in coords:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"({x},{y})", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return img


def show_window():
    hwnd = get_game_window()
    if hwnd:
        img = capture_window(hwnd)
        target_color = (255, 0, 255)
        contours, mask = detect_color(img, target_color, tolerance=0)
        img_with_boxes, centers = draw_bounding_boxes_and_return_centers(
            img, contours)
        img_with_grid = draw_grid_coordinates(img_with_boxes)
        cv2.imshow('Detected Color Areas + Grid', img_with_grid)
        if centers:
            print(f"Detected centers: {centers}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ----- Tick Handling Functions -----


def handle_tick(tick_id):
    global last_action
    grid = get_grid_coordinates()

    if tick_id == 0:
        # Tick 0: Check first for food in inventory.
        _, eat_matches = template_matching("needle_eat.png")
        # Also check for fish (salmon/trout) to decide on the branch.
        _, salmon_matches = template_matching("needle_salmon.png")
        _, trout_matches = template_matching("needle_trout.png")

        if eat_matches:
            # Scenario B: Food is available.
            target_pos = random.choice(eat_matches)
            humanized_move_to(jittered_position(target_pos))
            time.sleep(hybrid_delay(min_val=0.02, max_val=0.076))
            left_click()
            last_action = "food"
            print(f"[ACTION] (Scenario B) Clicked food at {target_pos}")
        else:
            # Scenario A: No food – use herb.
            humanized_move_to(jittered_position(grid[0]))  # Herb slot.
            time.sleep(hybrid_delay(min_val=0.02, max_val=0.076))
            left_click()
            last_action = "herb"
            print("[ACTION] (Scenario A) Clicked herb (grid[0])")

        time.sleep(hybrid_delay(min_val=0.01, max_val=0.066))

    elif tick_id == 1:
        if last_action == "food":
            # Scenario B: Knife fish action.
            _, salmon_matches = template_matching("needle_salmon.png")
            _, trout_matches = template_matching("needle_trout.png")
            combined_matches = salmon_matches + trout_matches
            if combined_matches:
                target_pos = random.choice(combined_matches)
                humanized_move_to(jittered_position(target_pos))
                time.sleep(hybrid_delay(min_val=0.01, max_val=0.066))
                left_click()
                print(
                    f"[ACTION] (Scenario B) Clicked fish with knife at {target_pos}")
            else:
                print("[INFO] (Scenario B) No fish found on tick 1.")
        elif last_action == "herb":
            # Scenario A: First click swamp tar, then knife fish.
            humanized_move_to(jittered_position(grid[1]))  # Swamp Tar.
            time.sleep(hybrid_delay(min_val=0.01, max_val=0.066))
            left_click()
            print("[ACTION] (Scenario A) Clicked swamp tar (grid[1])")

            time.sleep(hybrid_delay(min_val=0.01, max_val=0.066))

            _, salmon_matches = template_matching("needle_salmon.png")
            _, trout_matches = template_matching("needle_trout.png")
            combined_matches = salmon_matches + trout_matches
            if combined_matches:
                target_pos = random.choice(combined_matches)
                humanized_move_to(jittered_position(target_pos))
                time.sleep(hybrid_delay(min_val=0.01, max_val=0.066))
                left_click()
                print(
                    f"[ACTION] (Scenario A) Clicked fish with knife at {target_pos}")
            else:
                print("[INFO] (Scenario A) No fish found on tick 1.")

    elif tick_id == 2:
        # Tick 2: Move to the water spot.
        hwnd = get_game_window()
        if hwnd:
            img = capture_window(hwnd)
            target_color = (255, 0, 255)
            contours, _ = detect_color(img, target_color, tolerance=0)
            _, centers = draw_bounding_boxes_and_return_centers(img, contours)
            if centers:
                center = centers[0]
                humanized_move_to(jittered_position(center))
                time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                left_click()
                print(f"[ACTION] Clicked water spot at {center}")
            else:
                print("[INFO] No water spot found on tick 2.")


def tick_listener():
    global tick_count
    hwnd = win32gui.FindWindow(None, "RuneLite - Kiyoqt")
    if not hwnd:
        print("[ERROR] RuneLite window not found.")
        return

    last_state = None
    while not shutdown_event.is_set():
        c1 = get_pixel_color(hwnd, *BOX1)
        c2 = get_pixel_color(hwnd, *BOX2)
        state = (c1, c2)
        if last_state and state != last_state:
            if color_diff(c1, c2) > 100:
                with tick_lock:
                    print(f"[TICK] Tick {tick_count} at {time.time():.2f}")
                    handle_tick(tick_count)
                    tick_count = (tick_count + 1) % 3
        last_state = state
        time.sleep(0.01)


def anti_idle_listener():
    last_handle = None
    cooldown_seconds = 2
    last_trigger_time = 0
    while not shutdown_event.is_set():
        try:
            windows = Desktop(backend="uia").windows()
            for win in windows:
                title = win.window_text().strip().lower()
                if "notification" in title and "mozilla firefox" not in title and "toasts" not in title:
                    hwnd = win.handle
                    current_time = time.time()
                    if hwnd != last_handle or (current_time - last_trigger_time > cooldown_seconds):
                        print(f"[NOTIFY] {win.window_text()}")
                        notification_event.set()
                        last_handle = hwnd
                        last_trigger_time = current_time
                        break
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(0.5)


def notification_click_loop():
    while not shutdown_event.is_set():
        notification_event.wait()
        hwnd = get_game_window()
        if hwnd:
            time.sleep(hybrid_delay(min_val=0.5, max_val=2))
            img = capture_window(hwnd)
            target_color = (255, 0, 255)
            contours, _ = detect_color(img, target_color, tolerance=0)
            _, centers = draw_bounding_boxes_and_return_centers(img, contours)
            if centers:
                center = centers[0]
                humanized_move_to(jittered_position(center))
                time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                left_click()
                print(f"[CLICK] Clicked on jittered center: {center}")
                notification_event.clear()
            else:
                print("[INFO] No target center detected.")


def shutdown_listener():
    print("[INFO] Press F10 at any time to stop the script.")
    keyboard.wait('F10')
    print("[SHUTDOWN] F10 pressed. Exiting...")
    shutdown_event.set()


# ----- Main Execution -----
if __name__ == '__main__':
    # tick_thread = threading.Thread(target=tick_listener, daemon=True)
    shutdown_thread = threading.Thread(target=shutdown_listener, daemon=True)
    threading.Thread(target=anti_idle_listener, daemon=True).start()
    threading.Thread(target=notification_click_loop, daemon=True).start()

   # tick_thread.start()
    shutdown_thread.start()

    # tick_thread.join()
    shutdown_thread.join()
