# -*- coding: utf-8 -*-
"""
Зрение: Intel RealSense D435i + поиск кубиков по глубине.

Стол - плоскость (RANSAC), кубик - всё, что выступает над столом на высоту куба.
Глубина даёт 3D-центр верхней грани и направление рёбер.
"""
import time
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass
class Frame:
    color: np.ndarray          # HxWx3 BGR
    depth_m: np.ndarray        # HxW, метры, NaN = нет данных
    points: np.ndarray         # HxWx3, точки в системе камеры, метры
    fx: float
    fy: float
    ppx: float
    ppy: float

    def deproject(self, u, v, z):
        return np.array([(u - self.ppx) / self.fx * z, (v - self.ppy) / self.fy * z, z])


@dataclass
class Detection:
    center_cam: np.ndarray                 # центр верхней грани, камера, метры
    edge_dir_cam: np.ndarray               # направление ребра, камера
    box_px: np.ndarray                     # 4 угла грани в пикселях
    score: float = 1.0
    label: str = "cube"                    # подпись в окне (цвет)
    extra: dict = field(default_factory=dict)

    @property
    def center_px(self):
        return self.box_px.mean(0)


# ───────────────────────────── камера ─────────────────────────────
class RealSenseCamera:
    """D435i: цвет + глубина, выровненная по цвету. При обрыве потока перезапускается сама."""

    def __init__(self, color_res=(1280, 720), depth_res=(848, 480), fps=30, preset=None):
        if rs is None:
            raise RuntimeError("Не установлен pyrealsense2: pip install pyrealsense2")
        self.color_res, self.depth_res, self.fps, self.preset = color_res, depth_res, fps, preset
        self.pipeline = None
        self.info = {}

    def _start_pipeline(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, *self.depth_res, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, *self.color_res, rs.format.bgr8, self.fps)
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            self.pipeline = None
            raise RuntimeError(f"Камера не запустилась: {e}\nПодключите D435i в порт USB 3 (синий) "
                               "качественным кабелем USB 3; проверьте в RealSense Viewer.") from e
        dev = profile.get_device()
        get = lambda k: dev.get_info(k) if dev.supports(k) else "?"
        self.info = {"name": get(rs.camera_info.name), "serial": get(rs.camera_info.serial_number),
                     "firmware": get(rs.camera_info.firmware_version),
                     "usb": get(rs.camera_info.usb_type_descriptor)}
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        if self.preset is not None:
            depth_sensor.set_option(rs.option.visual_preset, float(self.preset))
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.ppx, self.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
        h, w = self.color_res[1], self.color_res[0]
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self._xn = (u - self.ppx) / self.fx
        self._yn = (v - self.ppy) / self.fy
        for _ in range(30):                          # прогрев автоэкспозиции
            self.pipeline.wait_for_frames(5000)

    def start(self):
        self._start_pipeline()
        if not str(self.info.get("usb", "")).startswith("3"):
            print(f"[camera] ВНИМАНИЕ: камера работает по USB {self.info.get('usb')}. "
                  "Нужен порт и кабель USB 3.x, иначе поток будет обрываться.")
        return self

    def _recover(self, err):
        print(f"[camera] поток прервался ({err}) — перезапуск...")
        self.stop()
        try:
            self._start_pipeline()
        except RuntimeError:
            devs = rs.context().query_devices()
            if len(devs):
                print("[camera] аппаратный сброс камеры...")
                devs[0].hardware_reset()
            time.sleep(5)
            self._start_pipeline()
        print("[camera] камера снова работает")

    def _frames(self):
        for attempt in range(3):
            try:
                return self.align.process(self.pipeline.wait_for_frames(3000))
            except RuntimeError as e:
                if attempt == 2:
                    raise
                self._recover(e)

    def capture(self, n_avg=1):
        depths, color = [], None
        for _ in range(max(1, n_avg)):
            fs = self._frames()
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            color = np.asanyarray(c.get_data()).copy()
            dm = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
            dm[dm <= 0] = np.nan
            depths.append(dm)
        if color is None:
            raise RuntimeError("Нет кадров от камеры")
        if len(depths) == 1:
            depth = depths[0]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # пиксели без глубины во всех кадрах
                depth = np.nanmedian(np.stack(depths), axis=0).astype(np.float32)
        pts = np.dstack([self._xn * depth, self._yn * depth, depth])
        return Frame(color, depth, pts, self.fx, self.fy, self.ppx, self.ppy)

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None


# ───────────────────────────── стол ─────────────────────────────
def fit_table_plane(points, iters=200, thr=0.004, sample=4000, rng=np.random.default_rng(0)):
    """RANSAC-плоскость стола. Возвращает (нормаль к камере, точка на плоскости)."""
    P = points.reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 500:
        raise RuntimeError("Слишком мало точек глубины — камера слишком близко к столу или закрыта")
    if len(P) > sample:
        P = P[rng.choice(len(P), sample, replace=False)]
    best, best_cnt = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        cnt = int((np.abs((P - a) @ n) < thr).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, a), cnt
    n, a = best
    inl = P[np.abs((P - a) @ n) < thr]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c)[2][2]
    if n @ c > 0:            # нормаль смотрит на камеру (камера в начале координат)
        n = -n
    return n, c


def height_map(frame, plane):
    n, c = plane
    return (frame.points - c) @ n            # метры над столом, NaN где нет глубины


def table_info(plane):
    """Расстояние камера→стол (м) и наклон оптической оси от вертикали (град)."""
    n, c = plane
    return float(abs(n @ c)), float(np.degrees(np.arccos(np.clip(-n[2], -1, 1))))


# ───────────────────────────── детектор ─────────────────────────────
class CubeDetector:
    """После detect(): last_plane - плоскость стола, last_height - карта высот над столом (м),
    last_top - маска «что-то высотой с куб» (верхние грани)."""

    def __init__(self, cube_size=0.025, top_frac=0.75):
        self.S, self.top_frac = cube_size, top_frac
        self.last_plane = self.last_height = self.last_top = None

    def detect(self, frame):
        plane = fit_table_plane(frame.points)
        h = height_map(frame, plane)
        with np.errstate(invalid="ignore"):
            top = (h > self.S * self.top_frac) & (h < self.S * 2.5)
        top = top.astype(np.uint8) * 255
        top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.last_plane, self.last_height, self.last_top = plane, h, top
        dets = self.faces(frame, top)
        dets.sort(key=lambda d: -d.score)
        return dets

    def faces(self, frame, mask, strict=True):
        """Квадратные грани кубиков в маске. strict - отбрасывать пятна не того размера."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 30:
                continue
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if min(w, h) < 5 or max(w, h) / min(w, h) > 1.6:
                continue
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [c], -1, 255, -1)
            inner = cv2.erode(blob, np.ones((3, 3), np.uint8), iterations=2)
            pts = frame.points[inner > 0]
            pts = pts[np.isfinite(pts).all(1)]
            if len(pts) < 15:
                continue
            z = float(np.median(pts[:, 2]))
            expected = (self.S * frame.fx / z) * (self.S * frame.fy / z)
            ratio = area / expected
            if strict and not (0.35 < ratio < 2.5):
                continue
            box = cv2.boxPoints(rect)
            p0, p1 = frame.deproject(*box[0], z), frame.deproject(*box[1], z)
            edge = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
            center = np.median(pts, axis=0).astype(np.float64)
            score = float(max(0.0, 1.0 - abs(np.log(max(ratio, 1e-3)))))
            out.append(Detection(center, edge, box, score, "cube", {"area": area}))
        return out


# ───────────────────────────── отрисовка ─────────────────────────────
def put_lines(img, lines, y0=28, dy=26, scale=0.7):
    """Строки текста в левом верхнем углу (OpenCV рисует только латиницу, кириллица - «???»)."""
    for k, t in enumerate(lines):
        cv2.putText(img, t, (10, y0 + dy * k), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3)
        cv2.putText(img, t, (10, y0 + dy * k), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1)


def draw(frame_bgr, dets, texts=None):
    """Рамки кубиков (первый - зелёный) и строки текста."""
    img = frame_bgr.copy()
    for i, d in enumerate(dets):
        col = (0, 255, 0) if i == 0 else (0, 200, 255)
        cv2.polylines(img, [np.int32(d.box_px)], True, col, 2)
        cx, cy = map(int, d.center_px)
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        tx, ty = int(d.box_px[:, 0].max()) + 8, int(d.box_px[:, 1].min()) + 14
        t = f"#{i + 1} {d.label} {d.score:.2f}"
        cv2.putText(img, t, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, t, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    put_lines(img, texts or [])
    return img
