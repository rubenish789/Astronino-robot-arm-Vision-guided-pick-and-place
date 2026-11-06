# -*- coding: utf-8 -*-
"""
Подбор HSV-порогов цвета кубиков под текущее освещение (робот не нужен).

Запуск из папки asto: python -m Alua.hsv_tune yellow
Ползунками добиться, чтобы в окне маски белыми были только кубики этого цвета (целиком), потом Q:
итоговый диапазон печатается в консоль - вписать его в config.COLOR_RANGES.
"""
import argparse

import cv2

import Alua.config as C
from Alua.vision import RealSenseCamera

WIN = "HSV tune"
NAMES = ("H low", "S low", "V low", "H high", "S high", "V high")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("color", choices=list(C.COLOR_RANGES))
    ap.add_argument("--range", type=int, default=0, help="какой из диапазонов цвета крутить (у красного их два)")
    a = ap.parse_args()
    lower, upper = C.COLOR_RANGES[a.color][a.range]
    cam = RealSenseCamera(C.COLOR_RES, C.DEPTH_RES, C.FPS, C.DEPTH_PRESET).start()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, val in zip(NAMES, (*lower, *upper)):
        cv2.createTrackbar(name, WIN, val, 180 if name.startswith("H") else 255, lambda _: None)
    try:
        while True:
            color = cam.capture(1).color
            v = [cv2.getTrackbarPos(n, WIN) for n in NAMES]
            lower, upper = tuple(v[:3]), tuple(v[3:])
            mask = cv2.inRange(cv2.cvtColor(color, cv2.COLOR_BGR2HSV), lower, upper)
            cv2.imshow(WIN, cv2.hconcat([color, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
            if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    print(f'"{a.color}" диапазон {a.range}: ({lower}, {upper})')


if __name__ == "__main__":
    main()
