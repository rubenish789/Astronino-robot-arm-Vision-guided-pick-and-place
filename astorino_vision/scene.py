# -*- coding: utf-8 -*-
"""
Разбор снимка стола: какие кубики лежат вне башен (с делением слипшихся), какого они цвета,
какой высоты башни. Ничего не двигает и не планирует - только смотрит.
"""
from collections import namedtuple

import cv2
import numpy as np

import astorino_vision.config as C
from astorino_vision.colors import classify_color, color_mask
from astorino_vision.vision import Detection

CUBE_MM = C.CUBE_SIZE_M * 1000.0

# Кубик россыпи: det - Detection, color - цвет (None = не распознан), p - точка захвата в базе робота (мм),
# in_cluster - получен делением слипшейся кучки
Cube = namedtuple("Cube", "det color p in_cluster")


class Scene:
    def __init__(self, calib, det):
        self.calib, self.det = calib, det

    def to_pixel(self, frame, p_robot_mm):
        pc = self.calib.to_cam(p_robot_mm)
        return int(round(pc[0] / pc[2] * frame.fx + frame.ppx)), int(round(pc[1] / pc[2] * frame.fy + frame.ppy))

    # ─────────── башни ───────────
    def measure_tower(self, frame, x, y):
        """Число кубиков в башне по карте высот (после det.detect). None - не удалось измерить."""
        h = self.det.last_height
        if h is None:
            return None
        H, W = h.shape
        win = C.TOWER_WIN_PX
        base_top = self.calib.z_at(x, y)          # Z калибровки = верхняя грань одного куба на столе
        # Из-за перспективы верх башни из n кубов виден в своём пикселе - проверяем от высокой к низкой
        for n in range(C.TOWER_MAX, 0, -1):
            u, v = self.to_pixel(frame, [x, y, base_top + (n - 1) * CUBE_MM])
            if not (win <= u < W - win and win <= v < H - win):
                return None
            patch = h[v - win:v + win + 1, u - win:u + win + 1]
            if not np.isfinite(patch).any():
                continue
            h_mm = float(np.nanmedian(patch)) * 1000.0
            if abs(h_mm - n * CUBE_MM) < 0.4 * CUBE_MM:
                return n
            if n == 1 and abs(h_mm) < 0.4 * CUBE_MM:
                return 0                           # на точке башни голый стол
        return None

    def is_tower_cube(self, p, color):
        """Куб - часть башни (не трогать), а не упавший: стоит точно на точке башни своего цвета
        (упавший куб лежит в стороне или не того цвета - его берём и ставим заново)."""
        stacked = p[2] > self.calib.z_at(p[0], p[1]) + 0.5 * CUBE_MM   # стоит на другом кубе, а не на столе
        for c, (x, y) in C.TOWER_SPOTS.items():
            dist = np.hypot(p[0] - x, p[1] - y)
            if dist <= C.TOWER_ON_SPOT_MM and color in (c, None):
                return True
            # верх башни из 2+ кубов: стопку у точки башни не разбираем, даже если она чуть съехала
            # или цвет верха прочитан неверно (иначе робот разбирал уже собранную башню)
            if stacked and dist <= C.TOWER_EXCLUDE_MM:
                return True
        return False

    # ─────────── россыпь ───────────
    def analyze(self, frame, dets):
        """Все кубики россыпи (вне башен), включая разделённые слипшиеся группы.
        Возвращает (список Cube, число кубов в слипшихся группах)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(self.det.last_top)
        members, bad = [], set()
        for k in range(1, n):
            sel = labels == k
            z = float(np.nanmedian(frame.depth_m[sel])) if np.isfinite(frame.depth_m[sel]).any() else 0.0
            if z <= 0:
                continue
            one = (C.CUBE_SIZE_M * frame.fx / z) * (C.CUBE_SIZE_M * frame.fy / z)   # площадь 1 грани, px
            ratio = stats[k, cv2.CC_STAT_AREA] / one
            if ratio < C.CLUSTER_RATIO:
                continue
            if max(self.footprint(self.blob_xy(frame, sel))) <= 1.35 * CUBE_MM:
                continue                   # пятно большое из-за краёв глубины, а на столе - один куб
            for i, d in enumerate(dets):   # детекции внутри группы (например, ложный «куб» между двумя) - долой
                cu, cv_ = np.int32(np.round(d.center_px))
                if 0 <= cv_ < labels.shape[0] and 0 <= cu < labels.shape[1] and labels[cv_, cu] == k:
                    bad.add(i)
            for part in self.split_necks(frame, sel):
                r = cv2.countNonZero(part) / one
                if r < C.CLUSTER_RATIO or max(self.footprint(self.blob_xy(frame, part))) <= 1.35 * CUBE_MM:
                    geo = self.det.faces(frame, part, strict=False)   # отдельный куб (касался углом)
                    if geo:
                        members.append(max(geo, key=lambda g: g.extra["area"]))
                else:
                    members += self.split_best(frame, part > 0, r)
        found = [(d, False) for i, d in enumerate(dets) if i not in bad] + [(d, True) for d in members]
        res, stuck = [], 0
        for d, in_cluster in found:
            p = self.calib.to_robot(d.center_cam)
            color = classify_color(frame.color, d.box_px)
            if self.is_tower_cube(p, color):
                continue
            stuck += in_cluster
            res.append(Cube(d, color, p, in_cluster))
        return res, stuck

    def blob_xy(self, frame, sel_mask):
        """Точки пятна в плоскости стола (мм, база робота)."""
        pts = frame.points[sel_mask > 0]
        pts = pts[np.isfinite(pts).all(1)]
        return ((pts * 1000.0) @ self.calib.R.T + self.calib.t)[:, :2]

    @staticmethod
    def footprint(xy):
        """Большая и меньшая сторона пятна на столе, мм. Точки дальше 1.5 ребра от медианы - выбросы
        глубины (из-за одного такого часть кучки «была» 400 мм и деление врало)."""
        xy = np.asarray(xy, np.float32)
        xy = xy[np.linalg.norm(xy - np.median(xy, axis=0), axis=1) < 1.5 * CUBE_MM]
        if len(xy) < 5:
            return 0.0, 0.0
        w, h = cv2.minAreaRect(xy)[1]
        return float(max(w, h)), float(min(w, h))

    @staticmethod
    def split_necks(frame, sel_mask):
        """Кубики, касающиеся углами («шахматка»), сливаются в одно пятно через тонкую перемычку.
        Эрозия на ~треть грани рвёт перемычку, а сами грани остаются - делим по ней.
        Возвращает маски частей (одна - если перемычек нет)."""
        z = float(np.nanmedian(frame.depth_m[sel_mask > 0]))
        k = max(3, int(round(C.NECK_FRAC * C.CUBE_SIZE_M * frame.fx / z)))
        m = sel_mask.astype(np.uint8) * 255
        er = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab = cv2.connectedComponents(er)
        if n <= 2:
            return [m]
        parts = []
        for j in range(1, n):
            part = cv2.dilate((lab == j).astype(np.uint8) * 255,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 2, k + 2))) & m
            if cv2.countNonZero(part) > 30:
                parts.append(part)
        return parts or [m]

    def split_best(self, frame, sel_mask, ratio):
        """Сколько кубиков в кучке (2..PILE_MAX): площадь даёт оценку, но края глубины «съедены» и
        оценка гуляет на ±1. Берём наименьшее число из оценка-1..оценка+1, при котором каждая часть
        не больше одного куба (иначе в части два куба и её не взять)."""
        est = int(round(ratio))
        ks = [k for k in range(est - 1, est + 2) if 2 <= k <= C.PILE_MAX] or [min(max(est, 2), C.PILE_MAX)]
        parts = []
        for k in ks:
            parts = self.split_cluster(frame, sel_mask, k)
            if len(parts) == k and all(d.extra["side_mm"] <= 1.35 * CUBE_MM and d.extra["side_min"] >= 0.55 * CUBE_MM
                                       for d in parts):
                return parts
        return parts

    def split_cluster(self, frame, sel_mask, n):
        """Делит пятно слипшихся кубов на n кубов: k-means по точкам верхних граней в плоскости стола.
        Одинаковые квадраты делятся ровно по линии касания, поэтому каждая часть - отдельная грань."""
        ys, xs = np.nonzero(sel_mask)
        pts = frame.points[ys, xs]
        ok = np.isfinite(pts).all(1)
        ys, xs, pts = ys[ok], xs[ok], pts[ok]
        if len(pts) < 15 * n:
            return []
        rob = (pts * 1000.0) @ self.calib.R.T + self.calib.t
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, lab, _ = cv2.kmeans(rob[:, :2].astype(np.float32), n, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        out = []
        for j in range(n):
            s = lab.ravel() == j
            if s.sum() < 15:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(np.stack([xs[s], ys[s]], 1).astype(np.float32)))
            side_mm, side_min = self.footprint(rob[s, :2])   # размер части на столе
            center = np.median(pts[s], axis=0).astype(np.float64)
            p0, p1 = frame.deproject(*box[0], center[2]), frame.deproject(*box[1], center[2])
            edge = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
            out.append(Detection(center, edge, box, 0.5, "cube",
                                 {"area": float(s.sum()), "cluster": True, "side_mm": side_mm, "side_min": side_min}))
        return out

    # ─────────── угол куба ───────────
    def refine_edge(self, frame, d, color):
        """Уточнить направление ребра по цветному кадру: верхняя грань = (высота куба) ∩ (цвет куба).
        Цвет даёт чёткие края там, где глубина размыта. None - если уточнить не удалось."""
        if color not in C.COLOR_RANGES or self.det.last_height is None:
            return None
        x0, y0 = np.int32(np.floor(d.box_px.min(0))) - 12
        x1, y1 = np.int32(np.ceil(d.box_px.max(0))) + 12
        H, W = frame.color.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        cmask = color_mask(cv2.cvtColor(frame.color[y0:y1, x0:x1], cv2.COLOR_BGR2HSV), color)
        with np.errstate(invalid="ignore"):
            top = (self.det.last_height[y0:y1, x0:x1] > C.CUBE_SIZE_M * 0.85).astype(np.uint8) * 255
        top = cv2.dilate(top, np.ones((5, 5), np.uint8))   # глубина у краёв «съедена» - чуть расширяем
        mask = cv2.morphologyEx(cmask & top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if min(w, h) < 5 or max(w, h) / min(w, h) > 1.3 or cv2.contourArea(c) < 0.75 * w * h:
            return None                                    # не квадрат (например, сосед того же цвета) - не уточняем
        box = cv2.boxPoints(rect) + np.float32([x0, y0])
        z = float(d.center_cam[2])
        p0, p1 = frame.deproject(*box[0], z), frame.deproject(*box[1], z)
        return (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
