import time
import cv2
import numpy as np

def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    """Sleep most of ``dt``, then spin for the remainder to reduce jitter."""
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    """Wait until ``t_end`` using a hybrid sleep/spin."""
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


def preprocess_image(img_bgr: np.ndarray, out_size=(128, 128)) -> np.ndarray:
    """Resize with padding"""
    h, w = img_bgr.shape[:2]
    target_h, target_w = out_size
    ratio = max(w / target_w, h / target_h)
    new_h = int(h / ratio)
    new_w = int(w / ratio)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas
