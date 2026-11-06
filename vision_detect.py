"""
vision_detect.py

Отдельный от робота пайплайн: RealSense D435 -> детекция цветных кубиков по HSV ->
3D-координата центроида в системе координат камеры (мм).

Ничего не знает про робота и Astorino - тестируется само по себе (шаг 2 роадмапа).
Только после того, как детекция здесь работает стабильно, переходим к шагу 3
(калибровка камера -> робот).

pip install pyrealsense2 opencv-python numpy
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

# ---- Настройки -------------------------------------------------------------

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

MIN_CONTOUR_AREA = 800  # px^2 - отсекаем шум/мелкие засветы, подберите под размер кубика в кадре
MAX_ASPECT_RATIO = 2.2  # кубик в кадре примерно квадратный/круглый - отсекаем вытянутые блики/отражения
MIN_SOLIDITY = 0.6      # area / area(minAreaRect) - отсекаем рваные, не-кубические пятна

# HSV-диапазоны цветов кубиков. Подбираются под конкретное освещение -
# используйте CALIBRATE_HSV=True ниже, чтобы подстроить через трекбары.
# Красный разбит на два диапазона из-за "разрыва" оттенка hue на границе 0/180 в OpenCV.
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [
        ((0, 90, 60), (8, 255, 255)),
        ((165, 90, 60), (180, 255, 255)),
    ],
    "green": [((40, 60, 50), (90, 255, 255))],
    "blue": [((95, 100, 33), (130, 255, 255))],
    "yellow": [((10, 90, 90), (32, 255, 255))],
}

DRAW_COLOR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}

# Включите, чтобы покрутить HSV-пороги вживую под текущее освещение стола.
CALIBRATE_HSV = False
CALIBRATE_COLOR = "yellow"  # какой цвет из COLOR_RANGES крутим (берётся первый диапазон)


@dataclass
class Detection:
    color: str
    pixel: tuple[int, int]      # центроид в пикселях кадра
    point_mm: tuple[float, float, float]  # X, Y, Z в системе координат камеры, мм
    area_px: float


def _mask_for_ranges(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    # Только смыкание дыр внутри пятна (dilate+erode), без отдельного erode в начале -
    # тот стирал тонкие/рваные пятна (например, жёлтый при узком H-диапазоне) целиком.
    # Шум по-прежнему отсекается фильтром по площади контура (MIN_CONTOUR_AREA).
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _looks_like_cube(c, area: float) -> bool:
    """Грубый фильтр формы: кубик в кадре примерно квадратный, а не вытянутая полоска
    блика/отражения. Считается по minAreaRect - без этого HSV-маска на бликах даёт
    ложные срабатывания того же цвета, что и кубик."""
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    if w <= 0 or h <= 0:
        return False
    aspect = max(w, h) / min(w, h)
    solidity = area / (w * h)
    return aspect <= MAX_ASPECT_RATIO and solidity >= MIN_SOLIDITY


def detect_cubes(color_bgr: np.ndarray, depth_frame, intrinsics) -> list[Detection]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    detections: list[Detection] = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = _clean_mask(_mask_for_ranges(hsv, ranges))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            if not _looks_like_cube(c, area):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            px = int(M["m10"] / M["m00"])
            py = int(M["m01"] / M["m00"])

            depth_m = depth_frame.get_distance(px, py)
            if depth_m <= 0:
                continue  # нет данных глубины в этой точке (дыра/за пределом камеры)

            point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)
            point_mm = (point[0] * 1000.0, point[1] * 1000.0, point[2] * 1000.0)

            detections.append(Detection(color_name, (px, py), point_mm, area))

    return detections


def pick_target(detections: list[Detection], color: str | None = None) -> Detection | None:
    """Выбирает, какой кубик брать: нужного цвета (или самый крупный по площади в кадре,
    если color=None). Используется шагом 4 (pick_and_place.py), чтобы не хватать первый
    попавшийся кубик, если их в кадре несколько."""
    candidates = [d for d in detections if color is None or d.color == color]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.area_px)


def _nothing(_):
    pass


def _make_calibration_window():
    cv2.namedWindow("HSV calibrate")
    lower, upper = COLOR_RANGES[CALIBRATE_COLOR][0]
    cv2.createTrackbar("H low", "HSV calibrate", lower[0], 180, _nothing)
    cv2.createTrackbar("S low", "HSV calibrate", lower[1], 255, _nothing)
    cv2.createTrackbar("V low", "HSV calibrate", lower[2], 255, _nothing)
    cv2.createTrackbar("H high", "HSV calibrate", upper[0], 180, _nothing)
    cv2.createTrackbar("S high", "HSV calibrate", upper[1], 255, _nothing)
    cv2.createTrackbar("V high", "HSV calibrate", upper[2], 255, _nothing)


def _read_calibration_trackbars():
    lower = (
        cv2.getTrackbarPos("H low", "HSV calibrate"),
        cv2.getTrackbarPos("S low", "HSV calibrate"),
        cv2.getTrackbarPos("V low", "HSV calibrate"),
    )
    upper = (
        cv2.getTrackbarPos("H high", "HSV calibrate"),
        cv2.getTrackbarPos("S high", "HSV calibrate"),
        cv2.getTrackbarPos("V high", "HSV calibrate"),
    )
    return lower, upper


def start_pipeline():
    """Общий старт RealSense для vision_detect.py и calibrate.py.
    Возвращает (pipeline, align, intrinsics)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    print(f"Intrinsics: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
          f"ppx={intrinsics.ppx:.2f} ppy={intrinsics.ppy:.2f}")
    return pipeline, align, intrinsics


def main():
    pipeline, align, intrinsics = start_pipeline()

    if CALIBRATE_HSV:
        _make_calibration_window()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())

            if CALIBRATE_HSV:
                lower, upper = _read_calibration_trackbars()
                COLOR_RANGES[CALIBRATE_COLOR][0] = (lower, upper)
                hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
                mask = _clean_mask(cv2.inRange(hsv, np.array(lower), np.array(upper)))
                cv2.imshow("HSV calibrate", mask)

            detections = detect_cubes(color_bgr, depth_frame, intrinsics)

            display = color_bgr.copy()
            for d in detections:
                x, y = d.pixel
                color = DRAW_COLOR.get(d.color, (255, 255, 255))
                cv2.circle(display, (x, y), 6, color, -1)
                label = f"{d.color} ({d.point_mm[0]:.0f},{d.point_mm[1]:.0f},{d.point_mm[2]:.0f})mm"
                cv2.putText(display, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            cv2.imshow("astorino vision", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
