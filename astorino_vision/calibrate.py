# -*- coding: utf-8 -*-
"""
calibrate.py

Калибровка камера→робот. Запуск из папки asto: python -m astorino_vision.calibrate [--manual]

Камера закреплена на руке (eye-in-hand): система координат камеры одна и та же
ТОЛЬКО пока рука стоит в одной и той же позе съёмки. В качестве этой позы
используем C.OBSERVE_JOINTS (уже сохранена вручную в config.py).

Порядок:
  1. Едем в позу съёмки (OBSERVE_JOINTS). Проверяем расстояние до стола (300-450 мм)
     и наклон (<3°). ENTER/C — подтвердить позу и запомнить photo_joints/photo_pose.
  2. Раскладываем 4-6 кубиков по всему кадру (не на одной линии). C — запомнить их
     координаты в камере (кадр замораживается — дальше рука уедет от стола, и живое
     видео потеряет смысл).
  3. На замороженном снимке кубики пронумерованы. Teach Lock -> TEACH, пультом
     подводим захват точно к кубику #1 (как он должен быть взят), не сдвигая кубики,
     SPACE — записать координаты робота для этой точки. Переходим к следующей.
     Z — отменить последнюю точку и повторить.
  4. Метод Кабша считает (R, t): camera_mm -> robot_mm. Сохраняется в calibration.json
     вместе с photo_pose/photo_joints (их читает pick_place.py). После перекалибровки
     заново подобрать GRASP_DOWN_MM в config.py.

Клавиши везде: Q/Esc — выйти без сохранения.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

import astorino_vision.config as C
from astorino_vision.astorino_usb import AstorinoError, connect_from_config
from astorino_vision.geometry import Calibration, fit_rigid
from astorino_vision.vision import CubeDetector, RealSenseCamera, table_info

MIN_POINTS = 4
WIN = "calibrate"


def _stable_order(dets):
    """Сортировка кубиков по положению в кадре (слева направо, сверху вниз), а не по
    score - иначе из-за шума глубины номера кубиков скачут между кадрами."""
    return sorted(dets, key=lambda d: (round(d.center_px[0] / 40.0), d.center_px[1]))


def _put_lines(img, lines, y0=28, dy=26, color=(255, 255, 255)):
    for i, t in enumerate(lines):
        y = y0 + dy * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def step1_photo_pose(cam, det, r):
    """Едем в позу съёмки, проверяем дистанцию/наклон, подтверждаем ENTER/C."""
    print("Шаг 1/3: еду в позу съёмки (OBSERVE_JOINTS)...")
    status = r.prepare()
    print("Статус:", status)
    r.set_speed(C.SPEED_OVERRIDE)
    r.jmove_joints(C.OBSERVE_JOINTS, speed_pct=C.SPEED_OVERRIDE)

    while True:
        f = cam.capture(C.FRAMES_TO_AVERAGE)
        try:
            dets = _stable_order(det.detect(f))
        except RuntimeError as e:
            dets = []
            print("[detect]", e)
        img = f.color.copy()
        lines = ["Шаг 1/3: проверьте дистанцию/наклон. C/ENTER - подтвердить позу съёмки"]
        if det.last_plane is not None:
            dist, tilt = table_info(det.last_plane)
            ok = 0.30 <= dist <= 0.45 and tilt < 3.0
            lines.append(f"table {dist * 1000:.0f} mm  tilt {tilt:.1f} deg" + ("  OK" if ok else "  <-- вне нормы 300-450мм/<3deg"))
        lines.append(f"кубиков в кадре: {len(dets)}")
        _put_lines(img, lines)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return None
        if k in (ord("c"), 13):
            photo_joints = r.joints()
            photo_pose = r.pose()
            print("Поза съёмки:", [round(v, 2) for v in photo_joints])
            return photo_joints, photo_pose


def step2_layout(cam, det):
    """Раскладываем 4-6 кубиков, C - заморозить кадр и список детекций."""
    while True:
        f = cam.capture(C.FRAMES_TO_AVERAGE)
        try:
            dets = _stable_order(det.detect(f))
        except RuntimeError as e:
            dets = []
            print("[detect]", e)
        img = f.color.copy()
        for i, d in enumerate(dets):
            cv2.polylines(img, [np.int32(d.box_px)], True, (0, 200, 255), 2)
            cx, cy = map(int, d.center_px)
            cv2.putText(img, f"#{i + 1}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        _put_lines(img, [f"Шаг 2/3: разложите {MIN_POINTS}-6 кубиков (не на одной линии), не закрывайте их рукой",
                         f"найдено: {len(dets)}   C - запомнить, Q - выход"])
        cv2.imshow(WIN, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return None
        if k == ord("c"):
            if len(dets) < MIN_POINTS:
                print(f"Видно только {len(dets)} кубик(ов), нужно минимум {MIN_POINTS}. Поправьте раскладку.")
                continue
            print(f"Запомнено {len(dets)} кубик(ов) в кадре.")
            return f.color.copy(), dets


def step3_teach(snapshot, dets, r, manual: bool):
    """На замороженном снимке подводим захват к каждому кубику по очереди."""
    points_robot: list[list[float]] = []
    idx = 0
    print("Шаг 3/3: Teach Lock -> TEACH. Подводите захват к кубику #N, не сдвигая кубики.")
    if manual:
        print("(режим --manual: координаты вводятся с клавиатуры терминала, взяты с пульта)")

    while idx < len(dets):
        img = snapshot.copy()
        for i, d in enumerate(dets):
            done = i < idx
            current = i == idx
            col = (0, 255, 0) if done else ((0, 255, 255) if current else (0, 140, 255))
            cv2.polylines(img, [np.int32(d.box_px)], True, col, 2)
            cx, cy = map(int, d.center_px)
            cv2.putText(img, f"#{i + 1}", (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        _put_lines(img, [f"Точка {idx + 1}/{len(dets)}: подведите захват к кубику #{idx + 1}",
                         "SPACE - записать   Z - отменить последнюю   Q - выход"])
        cv2.imshow(WIN, img)
        k = cv2.waitKey(30) & 0xFF

        if k in (ord("q"), 27):
            return None
        if k == ord("z"):
            if points_robot:
                points_robot.pop()
                idx -= 1
                print(f"Точка {idx + 1} отменена, повторите её.")
            continue
        if k == ord(" "):
            if manual:
                try:
                    raw = input(f"  Точка {idx + 1}: X Y Z с пульта (через пробел) > ")
                    x, y, z = (float(v.replace(",", ".")) for v in raw.split())
                except (ValueError, StopIteration):
                    print("  Не разобрал ввод, повторите (например: 300.5 -20 45.2)")
                    continue
                point = [x, y, z]
            else:
                pose = r.pose()
                point = list(pose[:3])
            points_robot.append(point)
            print(f"Точка {idx + 1}: робот {[round(v, 1) for v in point]} мм")
            idx += 1

    return points_robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", action="store_true",
                    help="координаты робота на шаге 3 вводятся вручную с пульта (если чтение по USB "
                         "в TEACH не работает)")
    a = ap.parse_args()

    cam = RealSenseCamera(C.COLOR_RES, C.DEPTH_RES, C.FPS, C.DEPTH_PRESET).start()
    det = CubeDetector(C.CUBE_SIZE_M, C.TOP_FACE_FRAC)
    r = connect_from_config(verbose=False)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    try:
        step1 = step1_photo_pose(cam, det, r)
        if step1 is None:
            print("Отменено на шаге 1.")
            return
        photo_joints, photo_pose = step1

        step2 = step2_layout(cam, det)
        if step2 is None:
            print("Отменено на шаге 2.")
            return
        snapshot, dets = step2

        points_robot = step3_teach(snapshot, dets, r, a.manual)
        if points_robot is None:
            print("Отменено на шаге 3.")
            return

        points_camera_mm = np.array([d.center_cam for d in dets], dtype=float) * 1000.0
        points_robot = np.array(points_robot, dtype=float)

        R, t, residuals = fit_rigid(points_camera_mm, points_robot)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        print("\n--- Результат калибровки ---")
        for i, e in enumerate(residuals, 1):
            print(f"  Точка {i}: ошибка {e:.2f} мм")
        print(f"  RMS: {rms:.2f} мм, максимальная: {residuals.max():.2f} мм")
        if residuals.max() > 15:
            print("ВНИМАНИЕ: ошибка калибровки большая (>15 мм) - проверьте точность жоговки "
                  "и что кубики не сдвинулись между шагом 2 и 3.")

        data = {
            "R": R.tolist(),
            "t": t.tolist(),
            "photo_joints": list(photo_joints),
            "photo_pose": list(photo_pose),
            "points_camera": points_camera_mm.tolist(),
            "points_robot": points_robot.tolist(),
            "rms_mm": rms,
            "max_mm": float(residuals.max()),
        }
        Calibration.save(C.CALIB_FILE, data)
        print(f"Сохранено в {C.CALIB_FILE}")
    except AstorinoError as e:
        print("ОШИБКА робота:", e)
    finally:
        cam.stop()
        r.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
