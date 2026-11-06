# -*- coding: utf-8 -*-
"""
Драйвер робота Kawasaki astorino по USB (виртуальный COM-порт) на Python.

Реализует родной протокол из «astorino Communication Protocol Manual» (2024):
  кадр  = 01 02 | ID | данные | CRC
  CRC   = (сумма всех предыдущих байт) & 0xFF
  float → int32 (×1000), big-endian; строки завершаются байтом 0x03.
Движения выполняются только в режиме REPEAT, при включённых моторах и без ошибок.

Защита от обрывов связи:
  • keepalive — фоновый поток раз в несколько секунд опрашивает статус, если канал простаивает;
  • автопереподключение — при обрыве порт открывается заново и сессия стартует снова.
    Обычные команды после этого повторяются автоматически; движение — нет (робот мог уже поехать).
"""
import struct
import threading
import time
from typing import Any

import serial
from serial.tools import list_ports

HEAD = bytes([0x01, 0x02])
ACK, FAIL, MOTION_DONE = 0x06, 0xCC, 0xAA
TEENSY_VID = 0x16C0   # контроллер astorino — Teensy 4.1

# Длина поля данных в ответах (без заголовка, ID и CRC). None = строка до 0x03.
RESP_LEN = {
    ACK: 0, MOTION_DONE: 0, FAIL: 1,
    0x27: 5,    # статус
    0x26: 18,   # входы/выходы
    0x28: 28,   # суставы JT1..JT7
    0x29: 28,   # X Y Z O A T JT7
    0x4A: 28,   # HOME
    0x54: 28, 0x55: 28,   # прямая / обратная кинематика
    0x58: 12, 0x59: 12,   # RPY <-> OAT
    0x56: 25,   # данные инструмента
    0x5B: 1, 0x5E: 1, 0x62: 1,   # температура CPU, скорость, код ошибки
    0x66: 28, 0x67: 28, 0x6E: 12,
    0x2A: None, 0x38: None, 0x47: None, 0x57: None,   # строки
}

ERRORS = {
    0x01: "ошибка CRC", 0x02: "E-stop или ошибка робота", 0x03: "запущена программа (Cycle ON)",
    0x04: "ошибка записи на SD", 0x05: "режим TEACH: не нажат deadman на пульте",
    0x06: "Cycle OFF", 0x07: "робот не готов", 0x08: "значение вне диапазона",
    0x09: "AS-команда не выполнена", 0x10: "неизвестная команда", 0x11: "ошибка кадра данных",
    0x12: "движение вне диапазона", 0x13: "резкое изменение команды JT",
    0x14: "точка вне рабочей зоны", 0x15: "робот уже движется", 0x16: "не выполнен Zeroing",
    0x17: "нет данных мастеринга", 0x18: "запрещено в режиме TEACH", 0x19: "Zeroing уже выполнен",
    0x20: "таймаут ответа", 0x21: "точка не существует", 0x22: "неверные данные",
    0x23: "программа не выбрана", 0x24: "превышена макс. скорость сустава", 0x25: "RTC выключен",
    0x26: "активен HOLD", 0x27: "движение прервано",
    0x28: "уже подключён другой клиент (закройте astorino software / RoboDK)",
}


class AstorinoError(Exception):
    pass


class LinkLost(AstorinoError):
    """Обрыв USB-связи (порт пропал или робот перестал отвечать)."""
    sent = True   # False — команда не успела уйти в порт, её можно безопасно повторить


class CommandFailed(AstorinoError):
    def __init__(self, cmd_id, code):
        self.cmd_id, self.code = cmd_id, code
        super().__init__(f"Команда 0x{cmd_id:02X} отклонена: код 0x{code:02X} — {ERRORS.get(code, 'неизвестный код')}")


def _i32(value):
    return struct.pack(">i", int(round(float(value) * 1000.0)))


def _floats(data):
    n = len(data) // 4
    return [v / 1000.0 for v in struct.unpack(">" + "i" * n, data[:4 * n])]


class Status:
    """Разбор 5 байт статуса (Communication Protocol Manual, п. 6.4.1)."""

    def __init__(self, b):
        s1, s2, s3, s4, s5 = b
        self.raw = bytes(b)
        self.in_home, self.motor_on, self.repeat_mode, self.hold = [bool(s1 & m) for m in (0x80, 0x40, 0x20, 0x10)]
        self.cycle_on, self.estop, self.error, self.ready = [bool(s1 & m) for m in (0x08, 0x04, 0x02, 0x01)]
        self.ext_hold, self.safety_fence = bool(s2 & 0x80), bool(s2 & 0x40)
        self.dry_run, self.zeroing_done, self.in_motion = bool(s2 & 0x04), bool(s2 & 0x02), bool(s2 & 0x01)
        self.io_module = bool(s3 & 0x80)
        self.tool = (s4 >> 5) & 0x07

    def __str__(self):
        f = lambda v: "да" if v else "нет"
        return (f"REPEAT={f(self.repeat_mode)} моторы={f(self.motor_on)} готов={f(self.ready)} "
                f"ошибка={f(self.error)} E-stop={f(self.estop)} HOLD={f(self.hold)} "
                f"Zeroing={f(self.zeroing_done)} движется={f(self.in_motion)} DryRun={f(self.dry_run)} "
                f"IO-модуль={f(self.io_module)} TOOL={self.tool}")


class IOState:
    """
    Разбор ответа Read IO (0x26), номера сигналов как в AS-языке:
      входы IO-модуля 1001–1008, входы на руке 1057–1058,
      выходы IO-модуля 1–8, выходы на руке 57–58, внутренние 2001–2016.
    Предполагается, что младший бит = первый сигнал (иначе config.IO_LSB_FIRST = False).
    """

    def __init__(self, d, lsb_first=True):
        self.raw = bytes(d)
        bit = (lambda byte, i: bool(byte >> i & 1)) if lsb_first else (lambda byte, i: bool(byte >> (7 - i) & 1))
        internal = int.from_bytes(d[16:18], "big")
        self.state = {}
        for i in range(8):
            self.state[1001 + i] = bit(d[0], i)
            self.state[1 + i] = bit(d[8], i)
        for i in range(2):
            self.state[1057 + i] = bit(d[7], i)
            self.state[57 + i] = bit(d[15], i)
        for i in range(16):
            self.state[2001 + i] = bool(internal >> i & 1) if lsb_first else bool(internal >> (15 - i) & 1)

    def __getitem__(self, n):
        return self.state[abs(int(n))]


class AstorinoUSB:
    def __init__(self, port=None, baud=256000, timeout=2.0, motion_timeout=90.0, verbose=False,
                 keepalive_s=2.0, auto_reconnect=True, io_lsb_first=True):
        self.port_fixed, self.port, self.baud = port, port, baud
        self.timeout, self.motion_timeout = timeout, motion_timeout
        self.verbose = verbose
        self.keepalive_s, self.auto_reconnect, self.io_lsb_first = keepalive_s, auto_reconnect, io_lsb_first
        self.ser: Any = None          # serial.Serial
        self.reconnects = 0
        self._lock = threading.RLock()
        self._last_io = time.monotonic()
        self._stop = threading.Event()
        self._ka_thread = None

    # ─────────── подключение ───────────
    @staticmethod
    def find_ports():
        return [p.device for p in list_ports.comports() if p.vid == TEENSY_VID]

    def _open(self):
        port = self.port_fixed
        if port is None:
            found = self.find_ports()
            if not found:
                raise LinkLost("Робот не найден по USB. Проверьте кабель USB-B и укажите порт вручную "
                               "(config.ROBOT_PORT). Доступные порты: "
                               + (", ".join(p.device for p in list_ports.comports()) or "нет"))
            port = found[0]
        s = serial.Serial()
        s.port, s.baudrate = port, self.baud
        s.bytesize, s.parity, s.stopbits = serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE
        s.timeout, s.write_timeout = 0.05, 2.0
        s.dtr = True
        try:
            s.open()
        except serial.SerialException as e:
            raise LinkLost(f"Не удалось открыть {port}: {e}\n"
                           "Если порт занят — закройте astorino software, RoboDK и другие скрипты.") from e
        self.ser, self.port = s, port
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def _start_session(self):
        try:
            self._transact(0x24)                          # Communication start (TCP/COM)
        except CommandFailed as e:
            if e.code != 0x28:
                raise
            try:                                          # «висящая» старая сессия — закрыть и открыть снова
                self._transact(0x25)
            except AstorinoError:
                pass
            time.sleep(0.2)
            self._transact(0x24)

    def connect(self):
        with self._lock:
            self._open()
            self._start_session()
        if self.verbose:
            print(f"[robot] подключено: {self.port}")
        if self.keepalive_s:
            self._stop.clear()
            self._ka_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
            self._ka_thread.start()
        return self

    def close(self):
        self._stop.set()
        with self._lock:
            if self.ser is None:
                return
            try:
                self._transact(0x25, timeout=1.0)          # Communication end
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def _reconnect(self, reason):
        print(f"[robot] связь потеряна ({reason}) — переподключаюсь...")
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        last = None
        for attempt in range(1, 11):
            time.sleep(1.0)
            try:
                self._open()
                self._start_session()
                self.reconnects += 1
                print(f"[robot] связь восстановлена ({self.port}), попытка {attempt}")
                return
            except AstorinoError as e:
                last = e
        raise LinkLost(f"Не удалось восстановить связь: {last}")

    def _keepalive_loop(self):
        while not self._stop.wait(0.5):
            if self.ser is None or time.monotonic() - self._last_io < self.keepalive_s:
                continue
            if not self._lock.acquire(blocking=False):     # идёт команда/движение — канал и так занят
                continue
            try:
                self._command(0x27, expect=0x27)
            except Exception as e:
                print(f"[robot] keepalive: {e}")
            finally:
                self._lock.release()

    # ─────────── низкий уровень ───────────
    @staticmethod
    def frame(cmd_id, payload=b""):
        body = HEAD + bytes([cmd_id]) + bytes(payload)
        return body + bytes([sum(body) & 0xFF])

    def _read_exact(self, n, deadline):
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.ser.read(n - len(buf))
            except (serial.SerialException, OSError) as e:
                raise LinkLost(f"ошибка чтения порта: {e}") from e
            if chunk:
                buf += chunk
            elif time.monotonic() > deadline:
                raise LinkLost("робот не отвечает (таймаут)")
        return bytes(buf)

    def _read_frame(self, timeout):
        deadline = time.monotonic() + timeout
        prev = None
        while True:                              # синхронизация по заголовку 01 02
            b = self._read_exact(1, deadline)[0]
            if prev == 0x01 and b == 0x02:
                break
            prev = b
        cid = self._read_exact(1, deadline)[0]
        if cid not in RESP_LEN:
            raise AstorinoError(f"Неизвестный ID ответа 0x{cid:02X}")
        n = RESP_LEN[cid]
        if n is None:
            data = bytearray()
            while not data or data[-1] != 0x03:
                data += self._read_exact(1, deadline)
            data = bytes(data)
        else:
            data = self._read_exact(n, deadline)
        crc = self._read_exact(1, deadline)[0]
        calc = (0x01 + 0x02 + cid + sum(data)) & 0xFF
        if crc != calc:
            raise AstorinoError(f"CRC не совпадает в ответе 0x{cid:02X}: {crc:02X} != {calc:02X}")
        if self.verbose:
            print(f"[robot] <- {cid:02X} {data.hex(' ')}")
        return cid, data

    def _transact(self, cmd_id, payload=b"", expect=ACK, timeout=None):
        """Одна попытка: отправить кадр и дождаться ответа с ID=expect (промежуточные ACK пропускаются)."""
        if self.ser is None:
            raise LinkLost("порт не открыт")
        timeout = self.timeout if timeout is None else timeout
        pkt = self.frame(cmd_id, payload)
        try:
            self.ser.reset_input_buffer()
            self.ser.write(pkt)
            self.ser.flush()
        except (serial.SerialException, OSError) as e:
            e2 = LinkLost(f"ошибка записи в порт: {e}")
            e2.sent = False
            raise e2 from e
        if self.verbose:
            print(f"[robot] -> {pkt.hex(' ')}")
        self._last_io = time.monotonic()
        deadline = time.monotonic() + timeout
        try:
            while True:
                cid, data = self._read_frame(max(0.05, deadline - time.monotonic()))
                if cid == FAIL:
                    raise CommandFailed(cmd_id, data[0])
                if cid == expect:
                    return data
        except LinkLost as e:
            e.sent = True
            raise
        finally:
            self._last_io = time.monotonic()

    def _command(self, cmd_id, payload=b"", expect=ACK, timeout=None):
        with self._lock:
            try:
                return self._transact(cmd_id, payload, expect, timeout)
            except LinkLost as e:
                if not self.auto_reconnect:
                    raise
                self._reconnect(e)
                if expect == MOTION_DONE and getattr(e, "sent", True):
                    raise LinkLost("Связь оборвалась во время движения. Соединение восстановлено, "
                                   "но движение не повторяется — проверьте положение робота.") from e
                return self._transact(cmd_id, payload, expect, timeout)

    # ─────────── чтение состояния ───────────
    def status(self):
        return Status(self._command(0x27, expect=0x27))

    def pose(self):
        """[X, Y, Z, O, A, T, JT7] текущего TCP (мм, град)."""
        return _floats(self._command(0x29, expect=0x29))

    def joints(self):
        """[JT1..JT7] (град)."""
        return _floats(self._command(0x28, expect=0x28))

    def error_code(self):
        return self._command(0x62, expect=0x62)[0]

    def read_io(self):
        return IOState(self._command(0x26, expect=0x26), self.io_lsb_first)

    def sig(self, number):
        """Состояние сигнала (как SIG() в AS): 1001..1008, 1057, 1058, 1..8, 57, 58, 2001..2016."""
        on = self.read_io()[number]
        return (not on) if number < 0 else on

    # ─────────── действия ───────────
    def motor_on(self):
        self._command(0x20)

    def reset_error(self):
        self._command(0x22)

    def hold_off(self):
        self._command(0x31)

    def cancel_motion(self):
        self._command(0x45)

    def set_speed(self, percent):
        self._command(0x39, bytes([max(1, min(100, int(percent)))]))

    def set_output(self, io_number, state):
        """Прямая запись выхода (Communication Protocol Manual, аналог Write Output) -
        не через интерпретатор AS. io_number: например 57/58 (рука) или 1..8 (IO-модуль)."""
        self._command(0x3A, bytes([int(io_number) & 0xFF, 1 if state else 0]))

    def set_signals(self, *numbers):
        """Как SIGNAL в AS: положительный номер — включить, отрицательный — выключить. Через прямую
        запись выхода (0x3A), а не AS-команду SIGNAL (0x4B) - та отклоняется контроллером с кодом 0x09."""
        for n in numbers:
            self.set_output(abs(int(n)), int(n) > 0)

    # ─────────── движения (блокирующие: ждут «Motion instruction completed») ───────────
    def _motion(self, cmd_id, payload):
        self._command(cmd_id, payload, expect=MOTION_DONE, timeout=self.motion_timeout)

    @staticmethod
    def _vals7(values):
        v = list(values)
        if len(v) == 6:
            v.append(0.0)
        if len(v) != 7:
            raise ValueError("Нужно 6 или 7 значений")
        return b"".join(_i32(x) for x in v)

    def lmove(self, pose, speed_mm_s=50, acc=50, dec=50):
        """Линейное движение к [X,Y,Z,O,A,T(,JT7)]. Скорость 1..250 мм/с."""
        spd = max(1, min(250, int(speed_mm_s)))
        self._motion(0x4E, bytes([0x01, spd, acc, dec]) + self._vals7(pose))

    def jmove_pose(self, pose, speed_pct=30, acc=50, dec=50):
        """Движение в суставной интерполяции к декартовой точке [X,Y,Z,O,A,T(,JT7)]."""
        spd = max(1, min(100, int(speed_pct)))
        self._motion(0x50, bytes([0x01, spd, acc, dec]) + self._vals7(pose))

    def jmove_joints(self, joints, speed_pct=30, acc=50, dec=50):
        """Движение к углам суставов [JT1..JT6(,JT7)]."""
        spd = max(1, min(100, int(speed_pct)))
        self._motion(0x50, bytes([0x02, spd, acc, dec]) + self._vals7(joints))

    # ─────────── высокий уровень ───────────
    def prepare(self):
        """Проверить готовность к автоматическим движениям и включить моторы."""
        st = self.status()
        if st.estop:
            raise AstorinoError("Нажата аварийная кнопка (E-stop). Отпустите её.")
        if st.error:
            self.reset_error()
            time.sleep(0.5)
            st = self.status()
            if st.error:
                code = self.error_code()
                raise AstorinoError(f"Ошибка робота не сбрасывается (код {code}). Посмотрите на пульте.")
        if not st.repeat_mode:
            raise AstorinoError("Робот в режиме TEACH. Переключите пульт/ключ в режим REPEAT.")
        if not st.zeroing_done:
            raise AstorinoError("Не выполнен Zeroing. Сделайте его в astorino software или с пульта.")
        if st.cycle_on:
            raise AstorinoError("Выполняется AS-программа (Cycle ON). Остановите её.")
        if st.hold:
            self.hold_off()
        if st.dry_run:
            print("[robot] ВНИМАНИЕ: включён DryRun — робот не будет физически двигаться.")
        if not st.motor_on:
            self.motor_on()
            time.sleep(0.5)
        return self.status()


def connect_from_config(verbose=False, port=None):
    """Подключение с параметрами из config.py."""
    import astorino_vision.config as C
    return AstorinoUSB(port or C.ROBOT_PORT, C.ROBOT_BAUD, verbose=verbose,
                       keepalive_s=C.KEEPALIVE_S, auto_reconnect=C.AUTO_RECONNECT,
                       io_lsb_first=C.IO_LSB_FIRST).connect()
