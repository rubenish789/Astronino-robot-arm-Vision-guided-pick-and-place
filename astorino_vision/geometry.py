# -*- coding: utf-8 -*-
"""Геометрия: калибровка камера→робот, углы O-A-T (Эйлер Z-Y-Z у Kawasaki), поворот захвата,
прямоугольники и кучки кубов в плоскости стола."""
import json
import math

import numpy as np

import astorino_vision.config as C


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_oat(o, a, t):
    """Матрица ориентации инструмента в базе робота: R = Rz(O)·Ry(A)·Rz(T)."""
    return rot_z(o) @ rot_y(a) @ rot_z(t)


def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def fit_rigid(src, dst):
    """Жёсткое преобразование dst ≈ R·src + t (метод Кабша). Возвращает R, t, ошибки по точкам."""
    A, B = np.asarray(src, float), np.asarray(dst, float)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    residuals = np.linalg.norm((A @ R.T + t) - B, axis=1)
    return R, t, residuals


def grasp_yaw_delta(edge_dir_base, o, a, t):
    """
    На сколько градусов добавить к O, чтобы губки захвата встали параллельно граням куба.
    Прибавка δ к O = поворот инструмента вокруг вертикальной оси базы на δ (Rz(O+δ)=Rz(δ)Rz(O)).
    Куб симметричен на 90°, поэтому результат в диапазоне [-45; 45).
    """
    jaw = rot_oat(o, a, t)[:, 0]
    if math.hypot(jaw[0], jaw[1]) < 0.3 or math.hypot(edge_dir_base[0], edge_dir_base[1]) < 0.3:
        return 0.0
    phi_jaw = math.degrees(math.atan2(jaw[1], jaw[0]))
    phi_edge = math.degrees(math.atan2(edge_dir_base[1], edge_dir_base[0]))
    return (phi_edge - phi_jaw + 45.0) % 90.0 - 45.0


class Calibration:
    """Хранит позу съёмки и преобразование «точка камеры (мм) → точка захвата в базе робота (мм)»."""

    def __init__(self, data):
        self.data = data
        self.R = np.array(data["R"], float)
        self.t = np.array(data["t"], float)
        # ручная поправка XY поверх калибровки (config.CALIB_SHIFT_MM), Z не трогаем
        self.shift = np.array([*C.CALIB_SHIFT_MM, 0.0], float)
        self.photo_pose = list(data["photo_pose"])
        self.photo_joints = list(data["photo_joints"])
        pr = np.array(data["points_robot"], float)
        self.z_floor = float(pr[:, 2].min())
        self.z_ref = float(np.median(pr[:, 2]))
        # Z захвата по точкам калибровки лежит на наклонной плоскости (2026-09-25: +5 мм от Y 335 к Y 485,
        # отклонения от плоскости <= 2 мм) - стол/рука наклонены, одно число z_ref для всего стола врёт.
        A = np.c_[pr[:, :2], np.ones(len(pr))]
        self.z_plane = np.linalg.lstsq(A, pr[:, 2], rcond=None)[0]
        self.xy_min, self.xy_max = pr[:, :2].min(0), pr[:, :2].max(0)
        self.xy_center = pr[:, :2].mean(0)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    def save(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def z_at(self, x, y):
        """Z захвата куба, стоящего на столе в точке (x, y) (как брал оператор при калибровке)."""
        (y0, dz0), (y1, dz1) = C.Z_FIX_Y
        return float(self.z_plane @ [x, y, 1.0]) + dz0 + (dz1 - dz0) * (y - y0) / (y1 - y0)

    def to_robot(self, p_cam_m):
        return self.R @ (np.asarray(p_cam_m, float) * 1000.0) + self.t + self.shift

    def to_cam(self, p_robot_mm):
        """Обратно к to_robot: точка базы робота (мм) → точка камеры (м)."""
        return self.R.T @ (np.asarray(p_robot_mm, float) - self.t - self.shift) / 1000.0

    def dir_to_robot(self, v_cam):
        return self.R @ np.asarray(v_cam, float)

    def in_bounds(self, x, y, margin):
        return bool(self.xy_min[0] - margin <= x <= self.xy_max[0] + margin and
                    self.xy_min[1] - margin <= y <= self.xy_max[1] + margin)


# ───────────── плоскость стола: повёрнутые прямоугольники, кучки кубов (всё в мм, XY базы) ─────────────
def rect(center, u, half_u, half_v):
    """Прямоугольник: центр, ось u, ось v ⟂ u, полуразмеры вдоль u и v."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    return np.asarray(center, float)[:2], u, np.array([-u[1], u[0]]), float(half_u), float(half_v)


def rects_overlap(r1, r2, tol=0.0):
    """Пересекаются ли два повёрнутых прямоугольника (теорема о разделяющей оси); tol - допустимое касание."""
    c1, u1, v1, a1, b1 = r1
    c2, u2, v2, a2, b2 = r2
    dc = c2 - c1
    for ax in (u1, v1, u2, v2):
        p1 = a1 * abs(u1 @ ax) + b1 * abs(v1 @ ax)
        p2 = a2 * abs(u2 @ ax) + b2 * abs(v2 @ ax)
        if abs(dc @ ax) >= p1 + p2 - tol:
            return False
    return True


def spread_apart(points, anchored, edges, cube, iters=20):
    """Кубы не могут пересекаться. Камера (калибровка ~4 мм, деление слипшихся k-means) даёт центры
    соседей на несколько мм ближе - тогда любой толчок выглядит как «въезд в соседа», а палец -
    как «на соседе», и зажатые кубы «некуда отодвинуть». Раздвигаем пары до касания: два квадрата
    не ближе cube / max(|cos|, |sin|) в направлении между центрами (грань к грани 25 мм, по
    диагонали 35); берём меньшее по граням обоих кубов - на случай разного поворота.
    anchored[i] - куб, который не двигаем (тот, что берём/толкаем).
    Пары ближе половины ребра не трогаем - это скорее двойная детекция одного куба."""
    pts = [np.asarray(p, float).copy() for p in points]
    for _ in range(iters):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dv = pts[j] - pts[i]
                dist = float(np.linalg.norm(dv))
                if dist <= cube / 2 or (anchored[i] and anchored[j]):
                    continue
                u = dv / dist
                need = min(cube / max(abs(u @ ed), abs(u[0] * ed[1] - u[1] * ed[0])) for ed in (edges[i], edges[j]))
                if dist >= need - 0.1:
                    continue
                push = u * (need - dist)
                wi = 0.0 if anchored[i] else (1.0 if anchored[j] else 0.5)
                pts[i] -= push * wi
                pts[j] += push * (1.0 - wi)
                moved = True
        if not moved:
            break
    return pts


def pile_of(k, pts, cube):
    """Номера кубиков одной кучки с кубиком k (соседи ближе 1.6 ребра, по цепочке)."""
    pile, todo = {k}, [k]
    while todo:
        i = todo.pop()
        for j, q in enumerate(pts):
            if j not in pile and np.linalg.norm(q - pts[i]) < 1.6 * cube:
                pile.add(j)
                todo.append(j)
    return sorted(pile)


def predict_push(pts, k, n, dist, cube):
    """Где окажутся кубики после толчка куба k на dist по n: кто стоит на пути, едет впереди
    (цепочкой). Грубо, но достаточно, чтобы понять, освободится ли куб. Возвращает (новые точки, кто сдвинулся)."""
    n = np.asarray(n, float)
    perp = np.array([-n[1], n[0]])
    new = [q.copy() for q in pts]
    new[k] = pts[k] + n * dist
    moved = [k]
    for j in sorted(range(len(pts)), key=lambda j: (pts[j] - pts[k]) @ n):
        if j == k:
            continue
        for m in list(moved):
            rel = pts[j] - pts[m]
            gap = (new[j] - new[m]) @ n
            if abs(rel @ perp) < cube - 2.0 and rel @ n > 0 and gap < cube:
                new[j] = new[j] + n * (cube - gap)
        if np.linalg.norm(new[j] - pts[j]) > 0.5:
            moved.append(j)
    return new, moved
