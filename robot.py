#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Робот‑симулятор с автоматически генерируемой картой ресторана.

Новые возможности:
  • Карта больше экрана, робот может перемещаться по всей её территории.
  • Камера следует за роботом.
  • Если файла commands.txt нет – не завершается работа, а используется
    пустой набор команд (упрощённый режим управления через сеть).
  • Добавлена очередь `command_queue` (thread‑safe) для команд,
    получаемых по сети.
  • Класс `NetworkClient` (отдельный поток) подключается к
    `path_planner` и посылает текущие координаты + угол робота,
    получая от него команды ROTATE/FORWARD и кладя их в `command_queue`.
"""

import math
import sys
import threading
import queue
import socket
import json
import time
import pygame

# ----------------------------------------------------------------------
#  Константы карты
# ----------------------------------------------------------------------
MAP_UNITS = 100                # количество условных единиц по стороне
MAP_SCALE = 15                 # 1 условная единица = 15 пикселей
MAP_WIDTH = MAP_UNITS * MAP_SCALE
MAP_HEIGHT = MAP_UNITS * MAP_SCALE

# ----------------------------------------------------------------------
#  Чтение команд из файла
# ----------------------------------------------------------------------
def load_commands(file_path: str):
    """
    Возвращает список кортежей (cmd, value).
    Поддерживаемые команды: FORWARD, BACKWARD, ROTATE, WAIT.
    Строки, начинающиеся с '#', игнорируются.
    Если файл не найден – возвращаем пустой список (но выводим предупреждение).
    """
    cmds = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cmd = parts[0].upper()
                if cmd not in ("FORWARD", "BACKWARD", "ROTATE", "WAIT"):
                    print(f"[warning] неизвестная команда «{cmd}» – будет пропущена")
                    continue
                value = float(parts[1]) if len(parts) > 1 else 0.0
                cmds.append((cmd, value))
    except FileNotFoundError:
        print(f"[warning] файл «{file_path}» не найден – переход в сетевой режим.")
    return cmds
# ----------------------------------------------------------------------
#  Генерация карты ресторана (мировые координаты)
# ----------------------------------------------------------------------
def generate_restaurant_map():
    """
    Возвращает:
        walls       – список pygame.Rect, участвующих в коллизиях (мировые координаты);
        draw_items  – список кортежей (world_rect, colour) для отрисовки;
        start_pos   – координаты (x, y) начального положения робота в мировых координатах.
    Карта полностью покрывает область 0 … MAP_WIDTH / MAP_HEIGHT независимо от размеров окна.
    """
    # Преобразователь «логические единицы → мир‑пиксели»
    def to_rect(gx, gy, gw, gh):
        """Преобразовать координаты в условных единицах в pygame.Rect (мировые)."""
        return pygame.Rect(
            int(gx * MAP_SCALE),
            int(gy * MAP_SCALE),
            int(gw * MAP_SCALE),
            int(gh * MAP_SCALE),
        )

    draw_items = []

    # ------------------------------------------------------------------
    #  Внешняя рамка (толщиной 2 условные единицы)
    # ------------------------------------------------------------------
    border = 2
    draw_items.append((to_rect(0, 0, 100, border), (100, 100, 100)))                     # верх
    draw_items.append((to_rect(0, 100 - border, 100, border), (100, 100, 100)))         # низ
    draw_items.append((to_rect(0, border, border, 100 - 2 * border), (100, 100, 100))) # левый
    draw_items.append((to_rect(100 - border, border,
                               border, 100 - 2 * border), (100, 100, 100)))        # правый

    # ------------------------------------------------------------------
    #  Кухня – закрытая зона (красноватый оттенок)
    # ------------------------------------------------------------------
    kitchen_rect = to_rect(border, border, 25 - border, 25 - border)
    draw_items.append((kitchen_rect, (200, 150, 150)))

    # ------------------------------------------------------------------
    #  Бар (зелёный оттенок)
    # ------------------------------------------------------------------
    bar_rect = to_rect(25, border, 50, 8)
    draw_items.append((bar_rect, (150, 200, 150)))

    # ------------------------------------------------------------------
    #  Центральный стол‑опора (коричневый оттенок)
    # ------------------------------------------------------------------
    pillar_rect = to_rect(70, 70, 8, 8)
    draw_items.append((pillar_rect, (180, 160, 120)))

    # ------------------------------------------------------------------
    #  Нова́я зона хранения (коричнево‑золотой оттенок)
    # ------------------------------------------------------------------
    storage_rect = to_rect(80, 20, 15, 15)
    draw_items.append((storage_rect, (180, 140, 100)))

    # ------------------------------------------------------------------
    #  Садовый участок (зелёный)
    # ------------------------------------------------------------------
    garden_rect = to_rect(10, 70, 30, 20)
    draw_items.append((garden_rect, (120, 190, 120)))

    # ------------------------------------------------------------------
    #  Столы в зале (препятствия)
    # ------------------------------------------------------------------
    TABLE_SIZE = 5      # логические единицы (≈ 75 px)
    tables = [
        to_rect(33, 28, TABLE_SIZE, TABLE_SIZE),  # Стол 1
        to_rect(53, 28, TABLE_SIZE, TABLE_SIZE),  # Стол 2
        to_rect(33, 48, TABLE_SIZE, TABLE_SIZE),  # Стол 3
        to_rect(53, 48, TABLE_SIZE, TABLE_SIZE),  # Стол 4
    ]
    for tbl in tables:
        draw_items.append((tbl, (200, 200, 120)))   # светло‑коричневый стол

    # ------------------------------------------------------------------
    #  Список стен (препятствия). Убираем кухню и столы – они не вызывают коллизий.
    # ------------------------------------------------------------------
    walls = [rect for rect, _ in draw_items
             if rect is not kitchen_rect and rect not in tables]

    # ------------------------------------------------------------------
    #  Начальная позиция робота – центр «зала».
    # ------------------------------------------------------------------
    start_pos = (50 * MAP_SCALE, 85 * MAP_SCALE)

    return walls, draw_items, start_pos


# ----------------------------------------------------------------------
#  Класс робота
# ----------------------------------------------------------------------
class Robot(pygame.sprite.Sprite):
    """
    Спрайт‑робот, умеющий линейно двигаться и вращаться.
    Добавлена очередь `command_queue` (thread‑safe) для команд из сети.
    """

    def __init__(self, pos, command_list):
        super().__init__()

        # ---------- Визуальная часть ----------
        w, h = 40, 60               # исходный размер спрайта
        self.base_image = pygame.Surface((w, h), pygame.SRCALPHA)
        # Треугольник, указывающий «вперёд», направлен вверх
        pygame.draw.polygon(
            self.base_image,
            (0, 255, 0),
            [(w // 2, 0), (w, h), (0, h)]
        )

        # ---------- Физика ----------
        self.pos = pygame.math.Vector2(pos)   # позиция в мировых координатах
        self.angle = 0.0                     # 0° – «вверх», + по‑часовой
        self.speed = 200.0                   # пикселей в секунду (линейно)
        self.rot_speed = 180.0               # градусов в секунду (вращение)

        # ---------- Команды ----------
        # Список, загруженный из файла (может быть пустым)
        self.commands = command_list[:]      # копия списка кортежей (cmd, val)

        # Очередь, куда будут помещаться команды из сети
        self.command_queue = queue.Queue()

        self.current_cmd = None
        self.current_val = None

        self.move_remaining = 0.0
        self.move_dir = 1                     # 1 – вперёд, -1 – назад

        self.rotate_remaining = 0.0
        self.rotate_sign = 1

        self.wait_remaining = 0.0

        # ------- след робота -------
        self.path_points = [self.pos.copy()]  # первая точка – старт

        # ---------- Начальная отрисовка ----------
        self.image = pygame.transform.rotate(self.base_image, -self.angle)
        self.rect = self.image.get_rect(center=self.pos)

    # ------------------------------------------------------------------
    def heading(self):
        """Единичный вектор направления «вперёд» (в координатах экрана)."""
        rad = math.radians(self.angle)
        return pygame.math.Vector2(math.sin(rad), -math.cos(rad))

    # ------------------------------------------------------------------
    def _next_command(self):
        """
        Берём следующую команду.
        Приоритет – команды из сети (`command_queue`), затем – из списка файла.
        """
        if not self.command_queue.empty():
            cmd, val = self.command_queue.get_nowait()
        elif self.commands:
            cmd, val = self.commands.pop(0)
        else:
            self.current_cmd = None
            return

        self.current_cmd = cmd
        self.current_val = val

        if cmd in ("FORWARD", "BACKWARD"):
            signed = val if cmd == "FORWARD" else -val
            self.move_remaining = abs(signed)
            self.move_dir = 1 if signed >= 0 else -1
        elif cmd == "ROTATE":
            self.rotate_remaining = abs(val)
            self.rotate_sign = 1 if val >= 0 else -1
        elif cmd == "WAIT":
            self.wait_remaining = val

    # ------------------------------------------------------------------
    def update(self, dt, walls):
        """Основной метод update – вызывается каждый кадр."""
        if self.current_cmd is None:
            self._next_command()
            return

        # -------------------- Движение --------------------
        if self.current_cmd in ("FORWARD", "BACKWARD"):
            if self.move_remaining <= 0:
                self.current_cmd = None
                return

            step = self.speed * dt
            step = min(step, self.move_remaining)

            delta = self.heading() * step * self.move_dir
            new_pos = self.pos + delta

            # Проверка столкновения со стеной
            temp_image = pygame.transform.rotate(self.base_image, -self.angle)
            temp_rect = temp_image.get_rect(center=new_pos)

            if any(temp_rect.colliderect(w) for w in walls):
                # Останавливаем текущую команду
                self.current_cmd = None
                self.move_remaining = 0.0
                return

            # Перемещаем без столкновений
            self.pos = new_pos
            self.move_remaining -= step

            self.image = temp_image
            self.rect = self.image.get_rect(center=self.pos)

        # -------------------- Поворот --------------------
        elif self.current_cmd == "ROTATE":
            if self.rotate_remaining <= 0:
                self.current_cmd = None
                return

            step_angle = self.rot_speed * dt
            step_angle = min(step_angle, self.rotate_remaining)

            self.angle = (self.angle + self.rotate_sign * step_angle) % 360
            self.rotate_remaining -= step_angle

            self.image = pygame.transform.rotate(self.base_image, -self.angle)
            self.rect = self.image.get_rect(center=self.pos)

        # -------------------- Ожидание --------------------
        elif self.current_cmd == "WAIT":
            self.wait_remaining -= dt
            if self.wait_remaining <= 0:
                self.current_cmd = None
        # (Других команд пока нет)


# ----------------------------------------------------------------------
#  Сетевой клиент (подключается к path_planner‑серверу)
# ----------------------------------------------------------------------
class NetworkClient(threading.Thread):
    """
    Отдельный поток‑клиент, подключающийся к серверу
    (localhost:5555). Каждые ~0.1 сек отправляет JSON‑сообщение:
        {"pos": [x, y], "angle": a}
    Приём команд от сервера – строки ROTATE 30, FORWARD 120 и т.д.
    Каждая полученная команда кладётся в robot.command_queue.
    """

    def __init__(self,
                 robot: Robot,
                 host: str = "127.0.0.1",
                 port: int = 5555):
        super().__init__(daemon=True)
        self.robot = robot
        self.host = host
        self.port = port
        self.sock = None
        self.running = True

    # --------------------------------------------------------------
    def _connect(self):
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(0.5)
                print(f"[network] Подключён к {self.host}:{self.port}")
                return
            except Exception as e:
                print("[network] Не удалось подключиться, повтор через 2 сек…", e)
                time.sleep(2)

    # --------------------------------------------------------------
    def run(self):
        self._connect()
        buffer = b""
        while self.running:
            # ---------- Отправляем состояние ----------
            try:
                state = {
                    "pos": [self.robot.pos.x, self.robot.pos.y],
                    "angle": self.robot.angle,
                }
                line = json.dumps(state) + "\n"
                self.sock.sendall(line.encode())
            except Exception as e:
                print("[network] Ошибка отправки состояния:", e)
                self._reconnect()
                continue

            # ---------- Приём команд ----------
            try:
                data = self.sock.recv(1024)
                if not data:
                    print("[network] Сервер разорвал соединение.")
                    self._reconnect()
                    continue

                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    line = line.strip().decode()
                    if not line:
                        continue
                    parts = line.split()
                    cmd = parts[0].upper()
                    val = float(parts[1]) if len(parts) > 1 else 0.0
                    if cmd in ("FORWARD", "BACKWARD", "ROTATE", "WAIT"):
                        self.robot.command_queue.put((cmd, val))
                    else:
                        print(f"[network] Неизвестная команда от сервера: {cmd}")

            except socket.timeout:
                # таймаут – просто продолжаем цикл
                pass
            except Exception as e:
                print("[network] Ошибка при получении команд:", e)
                self._reconnect()
                continue

            time.sleep(0.05)   # небольшая пауза, чтобы не «забивать» процессор

    # --------------------------------------------------------------
    def _reconnect(self):
        """Закрывает текущий сокет и пытается заново подключиться."""
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self._connect()

    # --------------------------------------------------------------
    def stop(self):
        self.running = False
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
#  Камера – следит за роботом, ограничивая область просмотра границами карты
# ----------------------------------------------------------------------
class Camera:
    """
    Простой «окно просмотра». Хранит смещение (offset), которое
    применяется к мировым объектам перед их отрисовкой.
    """

    def __init__(self, screen_w: int, screen_h: int, map_w: int, map_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.map_w = map_w
        self.map_h = map_h
        self.offset = pygame.math.Vector2(0, 0)

    def update(self, target_pos: pygame.math.Vector2):
        """
        Выравнивает центр камеры по целевой позиции (обычно – роботу).
        При этом смещение ограничивается пределами карты.
        """
        desired_x = target_pos.x - self.screen_w // 2
        desired_y = target_pos.y - self.screen_h // 2

        max_x = max(0, self.map_w - self.screen_w)
        max_y = max(0, self.map_h - self.screen_h)

        self.offset.x = max(0, min(desired_x, max_x))
        self.offset.y = max(0, min(desired_y, max_y))

    def apply(self, rect: pygame.Rect) -> pygame.Rect:
        """Возвращает прямоугольник, смещённый относительно камеры."""
        return rect.move(-int(self.offset.x), -int(self.offset.y))

    def world_to_screen(self, pos: pygame.math.Vector2) -> pygame.math.Vector2:
        """Переводит точку из мировых координат в координаты экрана."""
        return pos - self.offset


# ----------------------------------------------------------------------
#  Основной цикл программы
# ----------------------------------------------------------------------
def main():
    pygame.init()

    # --------------------------------------------------------------
    # Параметры окна (полноэкранный, но можно изменить)
    # --------------------------------------------------------------
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Робот‑симулятор: ресторан")
    clock = pygame.time.Clock()
    screen_w, screen_h = screen.get_width(), screen.get_height()

    # --------------------------------------------------------------
    # Генерация карты ресторана (мировые координаты)
    # --------------------------------------------------------------
    walls, draw_items, start_pos = generate_restaurant_map()

    # --------------------------------------------------------------
    # Загрузка предварительных команд (может быть пустым файлом)
    # --------------------------------------------------------------
    commands = load_commands("commands.txt")
    robot = Robot(start_pos, commands)

    # --------------------------------------------------------------
    # Сетевой клиент (подключаемся к path_planner‑серверу)
    # --------------------------------------------------------------
    net_client = NetworkClient(robot)
    net_client.start()

    # --------------------------------------------------------------
    # Камера
    # --------------------------------------------------------------
    camera = Camera(screen_w, screen_h, MAP_WIDTH, MAP_HEIGHT)

    # --------------------------------------------------------------
    # Группа спрайтов (единственный спрайт)
    # --------------------------------------------------------------
    all_sprites = pygame.sprite.Group(robot)

    # --------------------------------------------------------------
    # Игровой цикл
    # --------------------------------------------------------------
    running = True
    while running:
        dt = clock.tick(60) / 1000.0   # прошедшее время в секундах

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Обновление робота (внутри берёт команды из очереди и списка)
        robot.update(dt, walls)

        # Центрирование камеры на роботе
        camera.update(robot.pos)

        # Приведение позиции робота к координатам экрана
        robot_screen_pos = camera.world_to_screen(robot.pos)
        robot.rect.center = (int(robot_screen_pos.x), int(robot_screen_pos.y))

        # ---------- Отрисовка ----------
        screen.fill((30, 30, 30))           # фон

        # Статические объекты (мебель, стены) – смещаем на экран
        for world_rect, colour in draw_items:
            screen_rect = camera.apply(world_rect)
            pygame.draw.rect(screen, colour, screen_rect)

        # Спрайты (робот)
        all_sprites.draw(screen)

        pygame.display.flip()

    # --------------------------------------------------------------
    # Чистка
    # --------------------------------------------------------------
    net_client.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
