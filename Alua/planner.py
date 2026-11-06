# -*- coding: utf-8 -*-
"""
Планирование: что делать роботу по снимку стола - взять куб (под каким углом), отодвинуть зажатый
толчком, убрать мешающий куб с места под башню. Сам робот не двигается - результат в Action.

Всё в плоскости стола (мм, база робота). Камера висит на руке сбоку от захвата: при каждом движении
проверяется, что её корпус не над стоящими башнями, а раскрытые пальцы не на соседних кубах.
"""
from dataclasses import dataclass

import numpy as np

import Alua.config as C
from Alua.geometry import grasp_yaw_delta, pile_of, predict_push, rect, rects_overlap, rot_oat, spread_apart, wrap180
from Alua.scene import CUBE_MM


@dataclass
class Action:
    kind: str                  # "pick" - взять det и поставить на башню color (или отложить в park_xy); "push" - толкнуть
    det: object                # куб, с которым работаем (в окне - зелёная рамка)
    label: str                 # подпись в окне камеры
    target: list = None        # pick: [X Y Z O A T JT7] захвата
    color: str = None          # pick: на какую башню ставить; None - отложить в park_xy
    park_xy: object = None
    push: tuple = None         # push: (старт XY центра захвата, конец XY, O, Z)
    pushed: tuple = None       # push: (куда приедет куб, направление, все точки) - чтобы не толкать обратно
    clean: bool = False        # уборка места под башню


class Planner:
    def __init__(self, calib, scene, target_color=None):
        self.calib, self.scene, self.target_color = calib, scene, target_color
        self.tower_n = {c: 0 for c in C.TOWER_SPOTS}   # сколько кубиков в башне каждого цвета
        self.last_push = None                          # последний толчок (не толкать тот же куб обратно)
        self.push_why = ""                             # почему толчок не нашёлся - для лога
        self.o0, self.a0, self.t0 = self.calib.photo_pose[3:6]
        self.j7 = self.calib.photo_pose[6] if len(self.calib.photo_pose) > 6 else 0.0
        # Камера на руке: её положение и длинная ось в системе инструмента (из калибровки в позе съёмки)
        r_photo = rot_oat(self.o0, self.a0, self.t0)
        self.cam_d_tool = r_photo.T @ (self.calib.t - np.array(self.calib.photo_pose[:3]))
        self.cam_ax_tool = r_photo.T @ self.calib.R[:, 0]
        # Порядок обхода кубиков: сначала те, что дальше всего в сторону камеры (в кадре - верхний левый)
        off = (r_photo @ self.cam_d_tool)[:2]
        self.sweep_dir = off / np.linalg.norm(off)
        self.place_yaw = {}                            # поворот захвата при постановке на каждую башню
        for color, (x, y) in C.TOWER_SPOTS.items():
            if not self.calib.in_bounds(x, y, C.XY_MARGIN_MM):
                print(f"[tower] ВНИМАНИЕ: точка башни {color} ({x:.0f}, {y:.0f}) вне зоны калибровки")
            others = [s for c, s in C.TOWER_SPOTS.items() if c != color]
            yaw = next((dy for dy in (0.0, 90.0, -90.0, 180.0) if not self.cam_hits(x, y, self.o0 + dy, others)), None)
            if yaw is None:
                raise SystemExit(f"[tower] башня {color}: камера при любом повороте задевает соседние "
                                 "башни - разнесите TOWER_SPOTS в config.py")
            self.place_yaw[color] = yaw

    # ─────────── башни ───────────
    def update_towers(self, frame, verbose=False):
        """Сверяет счётчики башен с замером по глубине (после det.detect); замер важнее счётчика."""
        for color, (x, y) in C.TOWER_SPOTS.items():
            n = self.scene.measure_tower(frame, x, y)
            if n is None:
                if verbose:
                    print(f"[tower] {color}: высоту измерить не удалось, считаю {self.tower_n[color]}")
                continue
            if n != self.tower_n[color]:
                print(f"[tower] {color}: по камере {n} куб., в счётчике было {self.tower_n[color]} - беру {n}")
                self.tower_n[color] = n

    def tower_spots_up(self):
        """Точки башен, где башня реально стоит (для проверки камеры и пальцев)."""
        return [C.TOWER_SPOTS[c] for c in C.TOWER_SPOTS if self.tower_n[c] > 0]

    def travel_z(self):
        """Высота проезда над всеми башнями: низ куба в захвате выше самой высокой башни + APPROACH_MM."""
        return C.TOWER_BASE_Z + max(self.tower_n.values()) * CUBE_MM + C.APPROACH_MM

    def place_pose(self, color):
        """Поза разжатия на башне color: один и тот же поворот для всех кубиков (грани параллельны губкам,
        значит грани кубиков в башне совпадут), при котором камера не над соседними башнями."""
        x, y = C.TOWER_SPOTS[color]
        return [x, y, self.release_z(x, y, self.tower_n[color]),
                wrap180(self.o0 + self.place_yaw[color]), self.a0, self.t0, self.j7]

    def grasp_z(self, p):
        """Высота захвата. Камера видит верхнюю грань, и её Z гуляет на несколько мм (шум глубины +
        погрешность калибровки) - робот брал куб за самый верх и промахивался. Куб стоит либо на столе,
        либо на другом кубе, поэтому Z привязываем к «уровню» (z_at = захват куба на столе по плоскости
        точек калибровки, +25 мм на каждый куб ниже) и опускаем ещё на GRASP_DOWN_MM - ближе к середине куба."""
        base = self.calib.z_at(p[0], p[1])
        level = max(0, int(round((float(p[2]) - base) / CUBE_MM)))
        return base + level * CUBE_MM - min(C.GRASP_DOWN_MM, C.GRASP_DOWN_MAX_MM)

    def release_z(self, x, y, n):
        """Z разжатия куба, который ставим на n кубов в точке (x, y): куб держим так же, как брали
        (grasp_z), значит низ куба ровно на опоре; +TOWER_GAP_MM, чтобы не давить."""
        return self.calib.z_at(x, y) + n * CUBE_MM - min(C.GRASP_DOWN_MM, C.GRASP_DOWN_MAX_MM) + C.TOWER_GAP_MM

    # ─────────── геометрия пальцев, камеры и соседей ───────────
    def cam_hits(self, x, y, o, obstacles):
        """Точки obstacles (X, Y), которые накрывает корпус камеры, когда захват в (x, y) с углом O=o."""
        r = rot_oat(o, self.a0, self.t0)
        center = np.array([x, y]) + (r @ self.cam_d_tool)[:2]
        u = (r @ self.cam_ax_tool)[:2]
        u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        return [ob for ob in obstacles
                if abs((np.asarray(ob) - center) @ u) <= C.CAM_HALF_LEN_MM
                and abs((np.asarray(ob) - center) @ v) <= C.CAM_HALF_WID_MM]

    def jaw_axis(self, o):
        """Направление хода пальцев в базе при угле O=o (пальцы ходят вдоль длинной стороны камеры)."""
        u = (rot_oat(o, self.a0, self.t0) @ self.cam_ax_tool)[:2]
        return u / np.linalg.norm(u)

    def edge_xy(self, d):
        e = self.calib.dir_to_robot(d.edge_dir_cam)[:2]
        n = np.linalg.norm(e)
        return e / n if n > 1e-6 else np.array([1.0, 0.0])

    def obstacles(self, loose, skip=None, extra=0.0):
        """Прямоугольники соседних кубов и башен (с запасом) - всё, куда нельзя класть пальцы.
        extra - дополнительный запас вокруг кубов (предсказание после толчка)."""
        h = CUBE_MM / 2 + C.OBSTACLE_MARGIN_MM + extra
        edges = [self.edge_xy(c.det) for c in loose]
        pts = spread_apart([c.p[:2] for c in loose], [c.det is skip for c in loose], edges, CUBE_MM)
        obs = [rect(q, e, h, h) for c, q, e in zip(loose, pts, edges) if c.det is not skip]
        for color, (x, y) in C.TOWER_SPOTS.items():
            if self.tower_n[color] == 0:
                continue                          # башни нет (например, упала) - место свободно
            u = self.jaw_axis(self.o0 + self.place_yaw[color])
            obs.append(rect((x, y), u, h + 1.0, h + 1.0))
        return obs

    def fingers_clear(self, x, y, o, obs):
        """Раскрытые пальцы при захвате в (x, y) с углом O=o не ложатся ни на одно препятствие.
        По ходу пальцев от грани куба должно быть свободно GRASP_SIDE_FREE_MM (запас OBSTACLE_MARGIN_MM
        уже в препятствиях), поперёк - GRASP_FREE_WIDTH_MM (середина пальца; края могут лечь на углы
        соседей по диагонали - «шахматка»)."""
        u = self.jaw_axis(o)
        depth = max(C.GRASP_SIDE_FREE_MM - C.OBSTACLE_MARGIN_MM - 0.5, 1.0)   # 0.5 - ровно 12 мм тоже проходит
        off = CUBE_MM / 2 + depth / 2
        for s in (1.0, -1.0):
            f = rect(np.array([x, y]) + s * off * u, u, depth / 2, C.GRASP_FREE_WIDTH_MM / 2)
            if any(rects_overlap(f, ob) for ob in obs):
                return False
        return True

    def push_blocked(self, p):
        """Куда толкать нельзя: на стоящую башню. Пустые точки башен (башня упала) - обычный стол."""
        return any(np.hypot(p[0] - x, p[1] - y) < CUBE_MM + 6.0 for x, y in self.tower_spots_up())

    def loose_center(self):
        """Центр зоны россыпи: из config или середина зоны калибровки левее башен."""
        if C.LOOSE_CENTER is not None:
            return np.asarray(C.LOOSE_CENTER, float)
        x0, y0 = self.calib.xy_min
        x1, y1 = self.calib.xy_max
        x1 = min(x1, min(x for x, _ in C.TOWER_SPOTS.values()) - 2 * C.TOWER_EXCLUDE_MM)
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    # ─────────── место под башни ───────────
    def spot_blockers(self, color, loose):
        """Кубики россыпи, которые мешают поставить куб на башню color: лежат на её точке или там,
        куда опустятся раскрытые пальцы при постановке."""
        x, y = C.TOWER_SPOTS[color]
        o = self.o0 + self.place_yaw[color]
        h = CUBE_MM / 2 + C.OBSTACLE_MARGIN_MM
        me = rect((x, y), self.jaw_axis(o), h, h)
        out = []
        for c in loose:
            r = rect(c.p, self.edge_xy(c.det), h, h)
            if rects_overlap(me, r) or not self.fingers_clear(x, y, o, [r]):
                out.append(c.det)
        return out

    def check_spots(self, loose, verbose=True):
        """Бардак у точек башен. Башню строим только на чистом месте: кубики ближе CLEAR_RADIUS_MM к точке
        или там, куда лягут пальцы при постановке, сначала убираются. Проверяются точки, где башня
        не полная и на столе есть кубики её цвета (т.е. строить будем).
        Упавшие башни: если на точке лежит упавший куб, замер высоты видит «1», хотя башни нет -
        такую башню считаем пустой. Возвращает {цвет: [мешающие кубики]}."""
        blocked = {}
        for col, (x, y) in C.TOWER_SPOTS.items():
            if self.tower_n[col] >= C.TOWER_MAX or not any(c.color == col for c in loose):
                continue
            bl = self.spot_blockers(col, loose)
            bl += [c.det for c in loose
                   if np.hypot(c.p[0] - x, c.p[1] - y) < C.CLEAR_RADIUS_MM and not any(c.det is b for b in bl)]
            if not bl:
                continue
            on_spot = any(np.hypot(c.p[0] - x, c.p[1] - y) < 0.8 * CUBE_MM for c in loose
                          if any(c.det is b for b in bl))       # is, не in: Detection сравнивает массивы через ==
            if on_spot and self.tower_n[col] == 1:
                if verbose:
                    print(f"[tower] {col}: на точке башни лежит упавший куб - башни нет, соберу заново")
                self.tower_n[col] = 0
            blocked[col] = bl
        return blocked

    def park_spot(self, loose, skip, o):
        """Свободное место в зоне россыпи, куда отложить мешающий куб: далеко от других кубиков
        (чтобы его потом можно было взять) и от точек башен; камера над стоящими башнями не проходит.
        Из подходящих - ближайшее к центру зоны россыпи. None - места нет."""
        (x0, y0), (x1, y1) = self.calib.xy_min, self.calib.xy_max
        others = [np.asarray(c.p[:2], float) for c in loose if c.det is not skip]
        towers = self.tower_spots_up()
        center = self.loose_center()
        best = None
        for x in np.arange(x0, x1 + 0.1, 10.0):
            for y in np.arange(y0, y1 + 0.1, 10.0):
                q = np.array([x, y])
                if any(np.linalg.norm(q - np.asarray(s)) < C.CLEAR_RADIUS_MM + 30.0 for s in C.TOWER_SPOTS.values()):
                    continue
                if any(np.linalg.norm(q - r) < C.PARK_CLEAR_MM for r in others):
                    continue
                if towers and self.cam_hits(x, y, o, towers):
                    continue
                dist = np.linalg.norm(q - center)
                if best is None or dist < best[0]:
                    best = (dist, q)
        return None if best is None else best[1]

    # ─────────── выбор действия ───────────
    def clean_action(self, frame, loose, blocked, pushes):
        """Убрать один мешающий куб с места будущей башни: на свою башню, отложить в сторону или,
        если взять нельзя, толкнуть от точки башни. None - убирать нечего или нельзя."""
        mess = []
        for col, bl in blocked.items():
            for d in bl:
                if not any(d is m for m, _ in mess):
                    mess.append((d, col))
        if not mess:
            return None
        info = {id(c.det): c for c in loose}
        print("[clean] место под башню занято: " + ", ".join(
            f"{col} ({len(bl)} куб.)" for col, bl in blocked.items()))
        mess.sort(key=lambda m: -float(info[id(m[0])].p[:2] @ self.sweep_dir))
        for d, spot_c in mess:
            c = info[id(d)]
            target, why = self.plan(d, frame, c.color, self.obstacles(loose, skip=d))
            if target is None:
                print("[clean]", why)
                continue
            # на свою башню - если её место чистое (или мешает только этот самый куб), иначе в сторону
            own_ok = (c.color in C.TOWER_SPOTS and self.tower_n[c.color] < C.TOWER_MAX
                      and all(b is d for b in blocked.get(c.color, [])))
            park_xy = None if own_ok else self.park_spot(loose, d, target[3])
            if not own_ok and park_xy is None:
                print(f"[clean] куб X{c.p[0]:.0f} Y{c.p[1]:.0f}: некуда отложить")
                continue
            return Action("pick", d, f"CLEAN {spot_c} spot: {'to tower' if own_ok else 'park'}", target=target,
                          color=c.color if own_ok else None, park_xy=park_xy, clean=True)
        if pushes >= C.PUSH_MAX:
            return None
        best, best_d = None, None             # взять нельзя - толкаем от точки башни
        for d, spot_c in mess:
            k = next(i for i, c in enumerate(loose) if c.det is d)
            away = np.asarray(loose[k].p[:2], float) - np.asarray(C.TOWER_SPOTS[spot_c], float)
            pl = self.plan_push(k, loose, goal=away)
            if pl is None:
                print(f"[clean] куб X{loose[k].p[0]:.0f} Y{loose[k].p[1]:.0f}: толкнуть нельзя ({self.push_why})")
            elif best is None or pl[0] > best[0]:
                best, best_d = pl, d
        if best is None:
            return None
        return Action("push", best_d, "CLEAN: PUSH", push=best[1], pushed=best[2], clean=True)

    def candidates(self, loose, blocked):
        """Кубики для захвата по порядку обхода: сначала верхний левый (со стороны камеры),
        потом к нижнему правому - камера тогда висит над уже пустым местом."""
        cands = [c for c in loose
                 if (not self.target_color or c.color == self.target_color)
                 and c.color in C.TOWER_SPOTS and self.tower_n[c.color] < C.TOWER_MAX
                 and c.color not in blocked]               # на грязное место не строим (его сначала чистят)
        cands.sort(key=lambda c: -float(c.p[:2] @ self.sweep_dir))
        return cands

    def next_action(self, frame, loose, blocked, pushes):
        """Взять первый по порядку обхода куб, который можно безопасно взять, и поставить на его башню.
        Взять нечего, но есть зажатые кубы - лучший толчок. Возвращает (Action или None, почему ничего)."""
        cands = self.candidates(loose, blocked)
        for c in cands:
            target, why = self.plan(c.det, frame, c.color, self.obstacles(loose, skip=c.det))
            if target is None:
                print("[pick]", why)
                continue
            return Action("pick", c.det, f"PICK {c.color or '?'} X{target[0]:.0f} Y{target[1]:.0f}",
                          target=target, color=c.color), None
        if not cands:
            return None, "подходящих нет"
        if pushes >= C.PUSH_MAX:
            return None, f"уже {pushes} толчков - раздвиньте кубы руками"
        best, best_d = None, None             # лучший толчок среди всех зажатых кубов
        for c in cands:
            pl = self.plan_push(next(i for i, l in enumerate(loose) if l.det is c.det), loose)
            if pl is None:
                print(f"[push] куб X{c.p[0]:.0f} Y{c.p[1]:.0f}: толкнуть нельзя ({self.push_why})")
            elif best is None or pl[0] > best[0]:
                best, best_d = pl, c.det
        if best is None:
            return None, "зажатые кубы некуда отодвинуть"
        if not best[0][0]:
            print("[push] ни один толчок сразу не освобождает куб - отодвигаю от кучи, дальше посмотрим")
        return Action("push", best_d, "PUSH", push=best[1], pushed=best[2]), None

    # ─────────── захват ───────────
    def plan(self, d, frame=None, color=None, obs=()):
        """Поза захвата куба d: [X Y Z O A T JT7] или (None, почему нельзя)."""
        p = self.calib.to_robot(d.center_cam)
        o, a, t = self.o0, self.a0, self.t0
        delta = 0.0
        if C.ROTATE_GRIPPER:
            edge = self.scene.refine_edge(frame, d, color) if frame is not None else None
            if edge is not None:
                e0 = self.calib.dir_to_robot(d.edge_dir_cam)
                e1 = self.calib.dir_to_robot(edge)
                diff = (np.degrees(np.arctan2(e1[1], e1[0]) - np.arctan2(e0[1], e0[0])) + 45.0) % 90.0 - 45.0
                if abs(diff) > 5.0:
                    print(f"[pick] угол по цвету отличается от угла по глубине на {diff:+.0f}° - беру по цвету")
            else:
                edge = d.edge_dir_cam
            delta = grasp_yaw_delta(self.calib.dir_to_robot(edge), o, a, t)
        z = self.grasp_z(p)
        if not self.calib.in_bounds(p[0], p[1], C.XY_MARGIN_MM):
            return None, f"куб вне зоны калибровки (X{p[0]:.0f} Y{p[1]:.0f})"
        # Куб симметричен на 90°: из поворотов δ, δ±90, δ+180 берём ближайший, где пальцы не ложатся
        # на соседей, а камера не над башнями
        towers = self.tower_spots_up()
        why = "пальцы заденут соседний куб"
        for dd in sorted((delta, delta - 90.0, delta + 90.0, delta + 180.0), key=abs):
            if not self.fingers_clear(p[0], p[1], o + dd, obs):
                continue
            if towers and self.cam_hits(p[0], p[1], o + dd, towers):
                why = "камера заденет башню"
                continue
            return [float(p[0]), float(p[1]), z, wrap180(o + dd), a, t, self.j7], None
        return None, f"куб (X{p[0]:.0f} Y{p[1]:.0f}): {why}"

    # ─────────── толчок ───────────
    def plan_push(self, k, loose, goal=None):
        """Как отодвинуть зажатый куб loose[k]: захват раскрыт, один палец встаёт с одной стороны куба
        и толкает его торцом; второй палец идёт в стороне. Строго только одно: палец не должен опуститься
        НА куб. Задевать и толкать другие кубики можно - их новое место предсказывается (predict_push).
        Кучку у края зоны толкаем к центру зоны россыпи, кучку в центре - от её середины в стороны.
        Для каждого направления пробуем длины PUSH_STEPS_MM и проверяем, получится ли потом взять куб.
        Возвращает (оценка, (старт XY центра захвата, конец XY, O, z), (конец XY куба, n, все точки)) или None."""
        d, p = loose[k].det, loose[k].p
        obs = self.obstacles(loose, skip=d)
        o0 = self.o0
        z = self.grasp_z(p)
        e = self.edge_xy(d)
        f = np.array([-e[1], e[0]])
        A = np.asarray(p[:2], float)
        h = CUBE_MM / 2
        ft, fw = C.FINGER_THICK_MM, C.FINGER_WIDTH_MM
        off = C.FINGER_OPEN_MM + ft / 2               # центр пальца от центра захвата (захват раскрыт)
        towers = self.tower_spots_up()
        ja = self.jaw_axis(o0)
        pts = [np.asarray(c.p[:2], float) for c in loose]
        pile = pile_of(k, pts, CUBE_MM)
        pc = np.mean([pts[i] for i in pile], axis=0)
        center = self.loose_center()
        if goal is not None:
            goal = np.asarray(goal, float)            # задано снаружи (уборка: от точки башни)
        elif np.linalg.norm(center - pc) > C.PILE_MIDDLE_MM:
            goal = center - pc                        # кучка у края - тащим к центру зоны
        else:
            goal = A - pc                             # кучка в центре - разгоняем в стороны
        goal = goal / (np.linalg.norm(goal) + 1e-9)

        def pickable_count(new):
            """Сколько кубиков кучки можно будет взять после толчка."""
            lz = [c._replace(p=np.r_[q, c.p[2]]) for c, q in zip(loose, new)]
            cnt = 0
            for i in pile:
                di = lz[i].det
                ei = self.edge_xy(di)
                b = np.degrees(np.arctan2(ei[1], ei[0]) - np.arctan2(ja[1], ja[0]))
                ob = self.obstacles(lz, skip=di, extra=C.PUSH_FREE_MARGIN_MM)
                if any(self.fingers_clear(new[i][0], new[i][1], o0 + b + r * 90.0, ob)
                       and not (towers and self.cam_hits(new[i][0], new[i][1], o0 + b + r * 90.0, towers))
                       for r in range(4)):
                    cnt += 1
            return cnt

        # чем больше кубов в кучке, тем дальше толкаем: кучка едет медленнее предсказанного
        steps = [dist + C.PUSH_PER_CUBE_MM * max(0, len(pile) - 2) for dist in C.PUSH_STEPS_MM]
        last = self.last_push
        best = None
        why = {}                                      # почему отброшены варианты - для лога
        normals = [e, -e, f, -f]                      # по нормали граней куб едет ровно
        diagonals = [(e + f) / np.sqrt(2), (e - f) / np.sqrt(2), (-e + f) / np.sqrt(2), (-e - f) / np.sqrt(2)]
        for prio, n in [(1, n) for n in normals] + [(0, n) for n in diagonals]:
            # не толкать обратно куб, который только что сюда приехал (иначе туда-сюда без конца)
            if last is not None and np.linalg.norm(A - last[0]) < 20.0 and n @ last[1] < -0.5:
                why["обратно"] = why.get("обратно", 0) + 1
                continue
            reach = h * (abs(n @ e) + abs(n @ f))    # насколько куб выступает в сторону пальца (угол - дальше грани)
            finger0 = A - n * (reach + fw / 2 + C.PUSH_GAP_MM)  # центр толкающего пальца в начале
            perp = np.array([-n[1], n[0]])            # пальцы ходят поперёк толчка
            for dist in steps:
                new, moved = predict_push(pts, k, n, dist - C.PUSH_GAP_MM, CUBE_MM)   # куб едет меньше руки на зазор
                if any(not self.calib.in_bounds(new[i][0], new[i][1], C.XY_MARGIN_MM) or self.push_blocked(new[i])
                       for i in moved):
                    why["край зоны/башня"] = why.get("край зоны/башня", 0) + 1
                    break                             # дальше - тем более
                free = pickable_count(new) if prio == 1 else 0
                for side in (1.0, -1.0):              # с какой стороны от линии толчка идёт второй палец
                    tcp0 = finger0 + side * perp * off
                    fingers = (finger0, finger0 + side * perp * 2 * off)
                    if any(rects_overlap(rect(c, n, fw / 2, ft / 2), ob) for c in fingers for ob in obs):
                        why["палец на куб"] = why.get("палец на куб", 0) + 1
                        continue
                    base = (np.degrees(np.arctan2(perp[1], perp[0]) - np.arctan2(ja[1], ja[0])) + 90.0) % 180.0 - 90.0
                    o = None
                    for dd in sorted((base, base + 180.0, base - 180.0), key=abs):
                        if towers and (self.cam_hits(*tcp0, o0 + dd, towers)
                                       or self.cam_hits(*(tcp0 + n * dist), o0 + dd, towers)):
                            continue
                        o = wrap180(o0 + dd)
                        break
                    if o is None:
                        why["камера над башней"] = why.get("камера над башней", 0) + 1
                        continue
                    # сначала - после толчка можно что-то взять, потом - в нужную сторону (к центру/от кучи),
                    # потом - короче и ровнее
                    toward = round(float(n @ goal), 1)
                    score = ((1, toward, free, -dist, prio) if free
                             else (0, toward + dist / 200.0, prio, -len(moved), 0))
                    if best is None or score > best[0]:
                        tcp1 = tcp0 + n * dist
                        # палец идёт по прямой на одной Z: берём выше из начала/конца (стол неровный) + PUSH_UP_MM
                        zp = z + max(0.0, self.calib.z_at(*tcp1) - self.calib.z_at(*A)) + C.PUSH_UP_MM
                        best = (score, (tcp0, tcp1, o, zp), (new[k], n, new))
        self.push_why = ", ".join(f"{k_} {v}" for k_, v in why.items())
        return best
