# -*- coding: utf-8 -*-
"""Цвет кубиков по HSV (диапазоны - config.COLOR_RANGES)."""
import cv2
import numpy as np

import Alua.config as C

DRAW_COLOR = {          # BGR для отрисовки
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


def color_mask(hsv, name):
    """Маска пикселей цвета name (255 - цвет есть)."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in C.COLOR_RANGES[name]:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask


def classify_color(color_bgr, box_px, min_frac=C.COLOR_MIN_FRAC, shrink=C.COLOR_SHRINK):
    """Определяет цвет кубика по HSV внутри его рамки box_px (4 угла, пиксели).
    Берётся только середина рамки (shrink от размера): края рамки захватывают стол, тень, блик
    и соседние кубики другого цвета. None, если ни один цвет не покрывает хотя бы min_frac."""
    box = np.asarray(box_px, np.float32)
    c = box.mean(axis=0)
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(c + shrink * (box - c))), 255)
    area = int((mask > 0).sum())
    if area < 10:
        return None
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    best_name, best_frac = None, 0.0
    for name in C.COLOR_RANGES:
        frac = int(((color_mask(hsv, name) & mask) > 0).sum()) / area
        if frac > best_frac:
            best_name, best_frac = name, frac
    return best_name if best_frac >= min_frac else None
