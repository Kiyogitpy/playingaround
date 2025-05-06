import cv2
import numpy as np
import pygetwindow as gw
import win32api
import win32con
import win32gui
import win32ui


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
        (bmp_info['bmHeight'], bmp_info['bmWidth'], 4))

    win32gui.DeleteObject(bmp.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def handle_tick(tick_id):
    grid = get_grid_coordinates()
    if tick_id == 0:
        move_to(grid[0])
        time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
        left_click()
        print("[ACTION] Clicked grid[0]")

    elif tick_id == 1:
        move_to(grid[4])
        time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
        left_click()
        time.sleep(hybrid_delay(min_val=0.01, max_val=0.09))
        move_to(grid[5])
        time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
        left_click()
        print("[ACTION] Clicked grid[4]")

    elif tick_id == 2:
        hwnd = get_game_window()
        if hwnd:
            img = capture_window(hwnd)
            target_color = (255, 0, 255)
            contours, _ = detect_color(img, target_color, tolerance=0)
            _, centers = draw_bounding_boxes_and_return_centers(img, contours)
            if centers:
                center = centers[0]
                jittered = (
                    center[0] + random.randint(-5, 5),
                    center[1] + random.randint(-5, 5)
                )
                move_to(jittered)
                time.sleep(hybrid_delay(min_val=0.05, max_val=0.1))
                left_click()
                print(f"[ACTION] Clicked on detected center: {jittered}")
            else:
                print("[INFO] No target color found on tick 2.")


def template_matching(template_path):
    hwnd = get_game_window()
    if hwnd:
        img = capture_window(hwnd)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.85  # Lowered threshold slightly
        locations = np.where(result >= threshold)

        w, h = template.shape[::-1]
        match_coords = []

        for i, pt in enumerate(zip(*locations[::-1])):
            center_x = pt[0] + w // 2
            center_y = pt[1] + h // 2
            match_coords.append((int(center_x), int(center_y)))
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

        return img, match_coords


def show_window(template_path):
    hwnd = get_game_window()
    if hwnd:
        img = capture_window(hwnd)
        img_with_matches, match_coords = template_matching(img, template_path)

        cv2.imshow('Template Matching', img_with_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if match_coords:
            print(f"Match Coordinates: {match_coords}")
            return match_coords
        else:
            print("No matches found.")
            return None


if __name__ == "__main__":
    _, match_coords = template_matching("needle_eat.png")

    for i in range(len(match_coords)):  # Fixed iteration
        print(match_coords[i])  # Prints each coordinate separately
