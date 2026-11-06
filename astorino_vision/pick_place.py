# -*- coding: utf-8 -*-
"""
pick_place.py - главная программа: башни из кубиков по цветам.

Кубики россыпи (левая часть стола) ставятся на башню своего цвета в точках config.TOWER_SPOTS.
Каждый цикл - новый снимок из позы съёмки: scene.py разбирает стол, planner.py выбирает действие
(взять / толкнуть / убрать с места под башню), здесь оно выполняется роботом.
Высота башен измеряется по глубине при каждом снимке, так что после перезапуска программы
или ручной правки башни счётчики восстанавливаются сами.

Запуск (из папки asto): python -m astorino_vision.pick_place [--watch] [--step] [--color red]
Окно:   A - собрать все башни, SPACE - один цикл, R - пересчитать башни, W - наблюдение вкл/выкл,
        S - стоп после текущего действия, Q/Esc - выход.
Робот:  режим REPEAT, Zeroing выполнен, astorino software/RoboDK отключены от порта.
"""
from __future__ import annotations

import argparse
import collections
import time

import cv2
import numpy as np

import astorino_vision.config as C
from astorino_vision.astorino_usb import AstorinoError, connect_from_config
from astorino_vision.colors import DRAW_COLOR
from astorino_vision.geometry import Calibration
from astorino_vision.planner import Planner
from astorino_vision.scene import CUBE_MM, Scene
from astorino_vision.vision import CubeDetector, RealSenseCamera, draw

WIN = "pick & place"


class Watch:
    """Наблюдение: на столе появились кубики (башни сломали) и рука ушла - пора собирать заново.
    Старт, когда подсчёт кубиков не меняется и в кадре нет движения/руки WATCH_STABLE_S секунд."""

    def __init__(self, on):
        self.on = on
        self.sig, self.sig_since, self.done_sig = None, 0.0, None   # done_sig - после сбора осталось это: не повторять
        self.prev_gray, self.calm_since = None, time.time()         # calm_since - когда последний раз было движение
        self.hist = collections.deque()                              # (время, подсчёт) за последние WATCH_SMOOTH_S

    def toggle(self):
        self.on, self.done_sig = not self.on, None

    def rebuilt(self, sig):
        """Сбор закончен, на столе осталось sig (что робот не смог взять) - пока это не изменится, не собирать."""
        self.done_sig = self.sig = sig
        self.hist.clear()
        self.sig_since = self.calm_since = time.time()
        self.prev_gray = None

    def active(self, color, height):
        """Есть ли в кадре рука/движение: то, что заметно меняется между кадрами, или что-то выше самой высокой башни."""
        gray = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(color, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        moving = (self.prev_gray is not None
                  and np.mean(cv2.absdiff(gray, self.prev_gray) > C.WATCH_MOTION_DIFF) > C.WATCH_MOTION_FRAC)
        self.prev_gray = gray
        tall = False
        if height is not None:
            with np.errstate(invalid="ignore"):
                tall = int(np.count_nonzero(height > (C.TOWER_MAX + 0.8) * CUBE_MM / 1000.0)) > C.WATCH_TALL_PX
        return moving or tall

    def update(self, color, height, sig):
        """Очередной кадр. sig - сколько кубиков каждого цвета вне башен. Возвращает (пора собирать, строка для окна)."""
        active = self.active(color, height)
        now = time.time()
        if active:
            self.calm_since = now
        go, wait = False, ""
        if self.on:
            # подсчёт в одном кадре иногда врёт (куб за два) - берём самый частый за WATCH_SMOOTH_S,
            # иначе один «лишний» кадр сбрасывал отсчёт
            self.hist.append((now, sig))
            while self.hist and now - self.hist[0][0] > C.WATCH_SMOOTH_S:
                self.hist.popleft()
            s = collections.Counter(x for _, x in self.hist).most_common(1)[0][0]
            if s != self.sig:
                self.sig, self.sig_since = s, now
            if sum(s) and s != self.done_sig:
                left = C.WATCH_STABLE_S - (now - max(self.sig_since, self.calm_since))
                go = left <= 0
                wait = "   HAND/MOTION - waiting" if active else f"   rebuild in {max(left, 0):.0f}s"
        return go, f"WATCH {'ON' if self.on else 'off'}   W - вкл/выкл" + wait


class PickPlace:
    def __init__(self, args):
        self.args = args
        try:
            self.calib = Calibration.load(C.CALIB_FILE)
        except FileNotFoundError:
            raise SystemExit(f"{C.CALIB_FILE} не найден - сначала калибровка: python -m astorino_vision.calibrate")
        if self.calib.data.get("max_mm", 0) > 15:
            print(f"[pick] ВНИМАНИЕ: ошибка калибровки {self.calib.data['max_mm']:.1f} мм - возможны промахи")
        self.det = CubeDetector(C.CUBE_SIZE_M, C.TOP_FACE_FRAC)
        self.scene = Scene(self.calib, self.det)
        self.planner = Planner(self.calib, self.scene, args.color)
        self.cam = RealSenseCamera(C.COLOR_RES, C.DEPTH_RES, C.FPS, C.DEPTH_PRESET).start()
        self.robot = None
        self.pushes = 0                                  # сколько раз отодвигали зажатые кубы за один сбор
        self.stop_req = False                            # S/Q в окне камеры: остановиться после текущего действия
        self.quit_req = False                            # Q/Esc: после остановки выйти из программы

    @property
    def tower_n(self):
        return self.planner.tower_n

    # ─────────── робот ───────────
    def connect_robot(self):
        self.robot = connect_from_config(verbose=self.args.verbose)
        print("[robot]", self.robot.prepare())
        self.robot.set_speed(C.SPEED_OVERRIDE)
        self.gripper(open_=True)
        self.to_photo()

    def confirm(self, text):
        print(f"  -> {text}")
        if self.args.step and input("    Enter - выполнить, s - стоп: ").strip().lower() == "s":
            raise KeyboardInterrupt

    def move(self, kind, target, text, speed=None, joints=False):
        self.confirm(f"{text}: {np.round(target[:6], 1).tolist()}")
        if kind == "J" and joints:
            self.robot.jmove_joints(target, speed or C.JOINT_SPEED_PCT)
        elif kind == "J":
            self.robot.jmove_pose(target, speed or C.JOINT_SPEED_PCT)
        else:
            self.robot.lmove(target, speed or C.LIN_SPEED_FAST)
        self.poll_keys()                     # клавиши, нажатые в окне камеры, пока рука ехала

    def to_photo(self):
        self.move("J", self.calib.photo_joints, "в позу съёмки", joints=True)

    def poll_keys(self):
        """S - остановиться, когда закончится текущее действие (куб поставлен / толчок сделан, рука в LOOK);
        Q/Esc - то же и потом выйти. Движение не прерывается - куб не останется в захвате."""
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("s"), ord("q"), 27) and not self.stop_req:
            self.stop_req = True
            self.quit_req = k != ord("s")
            print("\n[stop] остановлюсь, когда закончу текущее действие"
                  + (" - потом выход" if self.quit_req else ""))

    def gripper(self, open_):
        self.robot.set_signals(*(C.GRIPPER_OPEN if open_ else C.GRIPPER_CLOSE))
        time.sleep(C.GRIPPER_WAIT_S)

    def gripped_ok(self):
        if not C.GRIP_CHECK_INPUT:
            return True
        return self.robot.sig(C.GRIP_CHECK_INPUT) != C.GRIP_EMPTY_STATE

    # ─────────── действия ───────────
    def pick(self, target):
        x, y, z, o, a, t, j7 = target
        up = lambda dz: [x, y, z + dz, o, a, t, j7]
        print(f"\n[pick] Беру куб: X={x:.1f} Y={y:.1f} Z={z:.1f} O={o:.1f}")
        self.gripper(open_=True)
        self.move("J", up(C.APPROACH_MM), "подход над кубом")
        self.move("L", up(0), "опускание", C.LIN_SPEED_SLOW)
        self.confirm("закрыть захват")
        self.gripper(open_=False)
        if not self.gripped_ok():
            print("[pick] датчик: куб не захвачен")
            self.gripper(open_=True)
            self.move("L", up(C.APPROACH_MM), "отъезд вверх", C.LIN_SPEED_SLOW)
            return False
        self.move("L", up(C.LIFT_MM), "подъём", C.LIN_SPEED_SLOW)
        return True

    def lift_over_towers(self, target):
        """Куб в захвате над точкой взятия: поднять выше всех башен. Возвращает высоту проезда."""
        z_up = self.planner.travel_z()
        if z_up > target[2] + C.LIFT_MM:
            self.move("L", [target[0], target[1], z_up, *target[3:7]], "подъём выше башен", C.LIN_SPEED_SLOW)
        return max(z_up, target[2] + C.LIFT_MM)

    def put_down(self, pose, z_up, text):
        """Опустить куб в pose (Z разжатия), разжать и отъехать вверх на z_up."""
        x, y, z_rel, o, a, t, j7 = pose
        self.move("L", [x, y, z_rel + 20, o, a, t, j7], "опускание", C.LIN_SPEED_SLOW)
        self.move("L", [x, y, z_rel, o, a, t, j7], text, C.PLACE_SPEED_MM_S)
        time.sleep(C.PLACE_SETTLE_S)            # рука успокоилась - иначе куб уходит по инерции
        self.confirm("разжать захват")
        self.gripper(open_=True)
        time.sleep(0.3)
        self.move("L", [x, y, z_up, o, a, t, j7], "отъезд вверх", C.LIN_SPEED_SLOW)

    def place(self, target, color):
        """Поставить взятый куб на башню color. Ориентация захвата - та же, что при взятии (почти вертикально)."""
        pose = self.planner.place_pose(color)
        print(f"[tower] {color}: ставлю куб #{self.tower_n[color] + 1}, разжатие на Z={pose[2]:.1f}")
        z_up = self.lift_over_towers(target)
        self.move("L", [pose[0], pose[1], z_up, *pose[3:]], f"над башней {color}")   # прямая, без провисания JMOVE
        self.put_down(pose, z_up, "постановка")
        self.tower_n[color] += 1

    def park(self, target, xy):
        """Отложить взятый куб на стол в точку xy (уборка места под башню)."""
        x, y = xy
        print(f"[clean] откладываю куб в сторону: X={x:.1f} Y={y:.1f}")
        z_up = self.lift_over_towers(target)
        self.move("L", [x, y, z_up, *target[3:7]], "над свободным местом")
        self.put_down([x, y, self.planner.release_z(x, y, 0), *target[3:7]], z_up, "постановка на стол")

    def push(self, plan):
        """Отодвинуть куб: раскрытый захват опускается рядом, один палец толкает куб торцом."""
        (sx, sy), (ex, ey), o, z = plan
        a, t, j7 = self.planner.a0, self.planner.t0, self.planner.j7
        print(f"\n[push] Отодвигаю куб: от X={sx:.1f} Y={sy:.1f} к X={ex:.1f} Y={ey:.1f}, O={o:.1f}")
        self.gripper(open_=True)
        self.move("J", [sx, sy, z + C.APPROACH_MM, o, a, t, j7], "над точкой толчка")
        self.move("L", [sx, sy, z, o, a, t, j7], "опускание захвата рядом с кубом", C.LIN_SPEED_SLOW)
        self.move("L", [ex, ey, z, o, a, t, j7], "толчок", C.PUSH_SPEED_MM_S)
        self.move("L", [ex, ey, z + C.APPROACH_MM, o, a, t, j7], "отъезд вверх", C.LIN_SPEED_SLOW)
        self.to_photo()
        self.pushes += 1

    def execute(self, act):
        """Выполнить действие планировщика; рука возвращается в позу съёмки. False - куб не захвачен."""
        if act.kind == "push":
            self.push(act.push)
            self.planner.last_push = act.pushed
            return True
        ok = self.pick(act.target)
        if ok:
            if act.color:
                self.place(act.target, act.color)
            else:
                self.park(act.target, act.park_xy)
        self.to_photo()
        return ok or act.clean

    # ─────────── циклы ───────────
    def snapshot(self):
        """Свежий усреднённый кадр после движения руки (старые кадры из очереди выброшены)."""
        for _ in range(C.FLUSH_FRAMES):
            self.cam.capture(1)
        return self.cam.capture(C.FRAMES_TO_AVERAGE)

    def look(self):
        """Снимок из позы съёмки: сверить башни, разобрать россыпь. Возвращает (кадр, кубики, детекции, слипшиеся)."""
        frame = self.snapshot()
        dets = self.det.detect(frame)
        self.planner.update_towers(frame)
        loose, stuck = self.scene.analyze(frame, dets)
        return frame, loose, dets, stuck

    def run_once(self):
        """Один цикл: снимок → убрать мешающий куб с места под башню, иначе взять → поставить, иначе
        отодвинуть зажатый куб → в позу съёмки. True, если что-то сделано."""
        frame, loose, dets, stuck = self.look()
        blocked = self.planner.check_spots(loose)
        self.show(frame, loose, f"LOOK: {len(loose)} cubes" + (f", stuck {stuck}" if stuck else ""))
        act = self.planner.clean_action(frame, loose, blocked, self.pushes)    # сначала чистим место под башни
        why = None
        if act is None:
            act, why = self.planner.next_action(frame, loose, blocked, self.pushes)
        if act is None:
            label = self.args.color or "любой"
            print(f"[pick] кубик ({label}) не выбран: {why}. В кадре {len(dets)} кубик(ов)"
                  + (f", слиплось {stuck}" if stuck else ""))
            return False
        self.show(frame, loose, act.label, first=act.det)
        return self.execute(act)

    def survey(self):
        """Снимок стола: сверить башни, посчитать кубики россыпи по цветам и слипшиеся группы."""
        frame, loose, _, stuck = self.look()
        self.show(frame, loose, "LOOK: survey")
        self.planner.check_spots(loose)
        return self.count_colors(loose), stuck

    def run_all(self):
        """Посчитать кубики по цветам, составить план и собрать башни.
        Кубики берутся по порядку обхода: от верхнего левого к нижнему правому."""
        self.pushes = 0
        self.planner.last_push = None
        self.stop_req = False
        counts, stuck = self.survey()
        plan = {c: min(self.tower_n[c] + counts[c], C.TOWER_MAX) for c in C.TOWER_SPOTS}
        print("\n[tower] на столе:", ", ".join(f"{c} {counts[c]}" for c in C.TOWER_SPOTS)
              + (f", не распознано {counts[None]}" if counts.get(None) else ""))
        print("[tower] план башен:", ", ".join(f"{c} {self.tower_n[c]}→{plan[c]}" for c in C.TOWER_SPOTS))
        for c in C.TOWER_SPOTS:
            extra = self.tower_n[c] + counts[c] - C.TOWER_MAX
            if extra > 0:
                print(f"[tower] {c}: {extra} куб. лишние (башня максимум {C.TOWER_MAX}) - останутся на столе")
        if stuck:
            print(f"[tower] {stuck} куб. стоят вплотную друг к другу - возьму поперёк или сначала отодвину")

        # каждый цикл - новый снимок: после толчка/захвата всё пересчитывается. S/Q - стоп между циклами
        while not self.stop_req and self.run_once():
            self.poll_keys()
        if self.stop_req:
            print("[stop] остановлено по клавише: рука в позе съёмки. A - продолжить сбор")
            return

        counts, stuck = self.survey()         # итоговая проверка камерой
        print("[tower] итог:", ", ".join(
            f"{c} {self.tower_n[c]}/{plan[c]}" + ("" if self.tower_n[c] >= plan[c] else " (!)")
            for c in C.TOWER_SPOTS))
        left = sum(n for k, n in counts.items() if k is not None and self.tower_n.get(k, 0) < C.TOWER_MAX)
        if left:
            print(f"[tower] на столе осталось {left} куб. - робот не смог их безопасно взять/отодвинуть: "
                  "раздвиньте руками и нажмите A")

    @staticmethod
    def count_colors(loose):
        """Сколько кубиков каждого цвета в россыпи; нераспознанные - под ключом None."""
        counts = {c: 0 for c in C.TOWER_SPOTS}
        for cube in loose:
            counts[cube.color] = counts.get(cube.color, 0) + 1
        return counts

    def table_sig(self, loose):
        """Что лежит на столе вне башен: число кубиков по цветам башен (для наблюдения). Счётчики башен тут
        не смотрим - в простое они не обновляются, а упавшая полная башня тоже должна запустить сбор."""
        counts = self.count_colors(loose)
        return tuple(counts[c] for c in C.TOWER_SPOTS)

    def table_sig_now(self):
        frame = self.snapshot()
        loose, _ = self.scene.analyze(frame, self.det.detect(frame))
        return self.table_sig(loose)

    # ─────────── окно ───────────
    def overlay(self, frame, img):
        """Кружки на точках башен, в кружке - сколько кубов в башне."""
        for color, (x, y) in C.TOWER_SPOTS.items():
            u, v = self.scene.to_pixel(frame, [x, y, self.calib.z_at(x, y)])
            col = DRAW_COLOR.get(color, (255, 255, 255))
            cv2.circle(img, (u, v), 22, col, 2)
            cv2.putText(img, str(self.tower_n[color]), (u - 7, v + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return img

    def show(self, frame, loose, text, first=None):
        """Показать снимок из позы съёмки в окне (во время сбора по A окно иначе стоит).
        first - куб, который сейчас берём/толкаем: рисуется первым (зелёным)."""
        items = sorted(loose, key=lambda c: c.det is not first)
        for c in items:
            c.det.label = c.color or "?"
        texts = [text, "TOWERS " + " ".join(f"{c[0].upper()}:{n}" for c, n in self.tower_n.items()),
                 "STOP REQUESTED" if self.stop_req else "S - stop after this step   Q - stop and quit"]
        cv2.imshow(WIN, self.overlay(frame, draw(frame.color, [c.det for c in items], texts)))
        self.poll_keys()

    # ─────────── главный цикл ───────────
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        watch = Watch(self.args.watch)
        try:
            self.connect_robot()
            frame = self.snapshot()
            self.det.detect(frame)
            self.planner.update_towers(frame, verbose=True)
            while True:
                frame = self.cam.capture(1)
                dets = self.det.detect(frame)
                loose, stuck = self.scene.analyze(frame, dets)
                counts = self.count_colors(loose)
                texts = [f"цель: {self.args.color or 'любой'}   SPACE - взять и поставить   Q - выход",
                         "TABLE  " + " ".join(f"{c[0].upper()}:{counts[c]}" for c in C.TOWER_SPOTS)
                         + (f"  ?:{counts[None]}" if counts.get(None) else "")
                         + (f"   STUCK: {stuck}" if stuck else ""),
                         "TOWERS " + " ".join(f"{c[0].upper()}:{n}" for c, n in self.tower_n.items())
                         + f" /{C.TOWER_MAX}   A - собрать все   R - пересчитать"]
                go, text = watch.update(frame.color, self.det.last_height, self.table_sig(loose))
                texts.append(text)
                if go:
                    print(f"\n[watch] на столе кубики для башен ({', '.join(f'{c} {n}' for c, n in zip(C.TOWER_SPOTS, watch.sig) if n)})"
                          " - собираю башни заново")
                    self.run_all()
                    if self.stop_req:
                        watch.on = False
                        print("[watch] наблюдение выключено (стоп). W - включить")
                        if self.quit_req:
                            break
                    watch.rebuilt(self.table_sig_now())
                    continue
                for c in loose:                                   # вместе с разделёнными слипшимися
                    c.det.label = c.color or "?"
                cv2.imshow(WIN, self.overlay(frame, draw(frame.color, [c.det for c in loose], texts)))
                k = cv2.waitKey(30) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    frame = self.snapshot()
                    self.det.detect(frame)
                    self.planner.update_towers(frame, verbose=True)
                if k == ord("a"):
                    self.run_all()
                    if self.stop_req:
                        watch.on = False
                        if self.quit_req:
                            break
                    else:
                        # башни собраны - дальше просто наблюдаем и сами собираем, если их сломают
                        watch.rebuilt(self.table_sig_now())
                        if not watch.on:
                            watch.on = True
                            print("[watch] башни собраны - наблюдаю. Сломаете - соберу сам через "
                                  f"{C.WATCH_STABLE_S:.0f} с после того, как рука уйдёт. W - выключить")
                if k == ord("w"):
                    watch.toggle()
                    print(f"[watch] наблюдение {'включено: робот сам соберёт упавшие башни' if watch.on else 'выключено'}")
                if k == ord(" "):
                    self.stop_req = False
                    self.run_once()
                    if self.quit_req:
                        break
        except AstorinoError as e:
            print(f"\n[robot] ОШИБКА: {e}")
        except KeyboardInterrupt:
            print("\n[pick] остановлено")
            if self.robot:
                try:
                    self.robot.cancel_motion()
                except AstorinoError:
                    pass
        finally:
            self.cam.stop()
            cv2.destroyAllWindows()
            if self.robot:
                self.robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default=None, choices=list(C.TOWER_SPOTS),
                    help="собирать только башню этого цвета (по умолчанию - все)")
    ap.add_argument("--tower", action="store_true", help="не нужен (башни - единственный режим), оставлен для совместимости")
    ap.add_argument("--step", action="store_true", help="подтверждать каждое движение Enter в консоли")
    ap.add_argument("--watch", action="store_true",
                    help="следить за столом сразу и самому собирать башни заново, если они упали")
    ap.add_argument("--verbose", action="store_true", help="печатать байты обмена с роботом")
    PickPlace(ap.parse_args()).run()


if __name__ == "__main__":
    main()
