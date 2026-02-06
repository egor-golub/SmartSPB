#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pixel‑based path planner for the restaurant‑robot simulator.

* Координаты цели задаются **в пикселях** и считаются от **левого нижнего
  угла карты** (не от угла окна!).
* Координаты вводятся интерактивно в консоли (print + input).
* После построения путь записывается в файл commands.txt (по умолчанию)
  в формате, понимаемом robot.py.
* Опции командной строки позволяют задать размер окна, размер ячейки
  сетки планировщика, запас от стен и имя файла‑вывода.

Запуск:

    python path_planner_pixels.py [--screen W H] [--resolution N]
                                 [--margin M] [--outfile FILE]

"""

import argparse
import heapq
import math
import sys
from typing import List, Tuple

import pygame
from robot import generate_restaurant_map


# ----------------------------------------------------------------------
#  Вспомогательные функции для проверки линии видимости и сглаживания
# ----------------------------------------------------------------------
def line_of_sight(p1: Tuple[float, float],
                 p2: Tuple[float, float],
                 walls: List[pygame.Rect]) -> bool:
    """Возвращает True, если отрезок p1‑p2 НЕ пересекает ни одну стену."""
    # Шаг 5 px – достаточно точно и быстро
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(int(dist // 5), 1)
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        if any(wall.collidepoint(x, y) for wall in walls):
            return False
    return True


def smooth_path(path: List[Tuple[float, float]],
                walls: List[pygame.Rect]) -> List[Tuple[float, float]]:
    """
    Убирает лишние вершины: от текущей точки пытаемся «увидеть» как можно
    дальше по пути, пока не встретится препятствие.
    """
    if not path:
        return []

    smooth = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Самая дальняя достижимая точка
        j = len(path) - 1
        while j > i + 1:
            if line_of_sight(smooth[-1], path[j], walls):
                break
            j -= 1
        smooth.append(path[j])
        i = j
    return smooth


# ----------------------------------------------------------------------
#  Преобразование готового пути в команды робота
# ----------------------------------------------------------------------
def path_to_commands(path: List[Tuple[float, float]],
                    start_angle: float = 0.0) -> List[Tuple[str, float]]:
    """
    По упрощённому пути (списку точек в пикселях) формирует команды
    ROTATE <угол> и FORWARD <расстояние>.  Угол измеряется от «вверх»,
    0° – вверх (по экранному –y), + по часовой стрелке.
    """
    if len(path) < 2:
        return []

    cmds: List[Tuple[str, float]] = []
    cur_angle = start_angle % 360.0

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)

        # Угол в диапазоне 0‑360, 0 – вверх (–y)
        target_angle = math.degrees(math.atan2(dx, -dy)) % 360.0

        # Наименьший поворот (от -180 до +180)
        delta = ((target_angle - cur_angle + 180) % 360) - 180

        if abs(delta) > 1e-4:
            cmds.append(("ROTATE", round(delta, 2)))
        cmds.append(("FORWARD", round(dist, 2)))

        cur_angle = target_angle

    return cmds


def write_commands(commands: List[Tuple[str, float]], filename: str) -> None:
    """Записывает список команд в файл формата, понятного robot.py."""
    with open(filename, "w", encoding="utf-8") as f:
        for cmd, val in commands:
            f.write(f"{cmd} {val}\n")


# ----------------------------------------------------------------------
#  Класс и функции планировщика A*
# ----------------------------------------------------------------------
class AStarGrid:
    """
    Дискретная сетка (размер ячейки задаётся в пикселях) для поиска
    пути методом A*.
    """

    def __init__(self,
                 width_px: int,
                 height_px: int,
                 res: int,
                 walls: List[pygame.Rect],
                 robot_radius: float = 10.0):
        self.res = float(res)
        self.cols = int(math.ceil(width_px / self.res))
        self.rows = int(math.ceil(height_px / self.res))
        self.walls = walls
        self.robot_radius = robot_radius

        # Свободные клетки: True – свободна, False – занята стеной
        self.grid = [[True for _ in range(self.cols)] for _ in range(self.rows)]
        self._build_grid()

    # --------------------------------------------------------------
    def _cell_center(self, gx: int, gy: int) -> Tuple[float, float]:
        """Центр ячейки в пикселях."""
        return (gx * self.res + self.res / 2.0,
                gy * self.res + self.res / 2.0)

    # --------------------------------------------------------------
    def _build_grid(self) -> None:
        """Помечаем клетки, которые пересекаются со стенами (с учётом запаса)."""
        for gy in range(self.rows):
            for gx in range(self.cols):
                cx, cy = self._cell_center(gx, gy)

                robot_rect = pygame.Rect(
                    cx - self.robot_radius,
                    cy - self.robot_radius,
                    self.robot_radius * 2,
                    self.robot_radius * 2,
                )
                if any(wall.colliderect(robot_rect) for wall in self.walls):
                    self.grid[gy][gx] = False

    # --------------------------------------------------------------
    def world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Пиксель → индексы ячейки."""
        x, y = pos
        gx = int(x // self.res)
        gy = int(y // self.res)
        gx = max(0, min(self.cols - 1, gx))
        gy = max(0, min(self.rows - 1, gy))
        return gx, gy

    def grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """Индексы ячейки → центр в пикселях."""
        return self._cell_center(*cell)

    # --------------------------------------------------------------
    def neighbors(self,
                  cell: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """8‑направленные соседи + стоимость (1 или √2)."""
        gx, gy = cell
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.grid[ny][nx]:
                        cost = math.hypot(dx, dy)
                        result.append(((nx, ny), cost))
        return result

    # --------------------------------------------------------------
    @staticmethod
    def heuristic(a: Tuple[int, int],
                  b: Tuple[int, int],
                  scale: float) -> float:
        """Эвристика – евклидово расстояние (в пикселях)."""
        ax, ay = a
        bx, by = b
        return math.hypot((ax - bx) * scale, (ay - by) * scale)

    # --------------------------------------------------------------
    def a_star(self,
               start_world: Tuple[float, float],
               goal_world: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Возвращает путь в виде списка точек (в пикселях)."""

        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)

        if not self.grid[start[1]][start[0]]:
            raise ValueError("Стартовая позиция находится внутри препятствия")
        if not self.grid[goal[1]][goal[0]]:
            raise ValueError("Целевая позиция находится внутри препятствия")

        open_set = []
        heapq.heappush(open_set, (0.0, 0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal, self.res)}
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Восстановление пути
                path_cells = [current]
                while current in came_from:
                    current = came_from[current]
                    path_cells.append(current)
                path_cells.reverse()
                return [self.grid_to_world(c) for c in path_cells]

            for neighbor, move_cost in self.neighbors(current):
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal, self.res)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        raise RuntimeError("Путь не найден")


# ----------------------------------------------------------------------
#  Запрос координат цели у пользователя
# ----------------------------------------------------------------------
def get_target_pixel_from_user(scale: float,
                              offset_x: float,
                              offset_y: float) -> Tuple[float, float]:
    """
    Пользователь вводит X и Y в пикселях, измеряемых от
    *левого нижнего угла карты* (не от окна).
    Функция возвращает координаты, уже преобразованные к системе
    pygame (origin – левый верхний угол).
    """
    map_pix_w = 100 * scale
    map_pix_h = 100 * scale

    print("\n=== Ввод цели (пиксели, отсчёт от левого нижнего угла карты) ===")
    print(f"Размер карты: {int(map_pix_w)} × {int(map_pix_h)} пикселей")
    print("Пример ввода: 150 80  (x = 150 px от левого края, y = 80 px от нижнего)")

    while True:
        raw = input("Введите X Y через пробел → ").strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 2:
            print("[warning] Требуется ровно два числа.")
            continue
        try:
            x_user = float(parts[0])
            y_user = float(parts[1])
        except ValueError:
            print("[warning] Некорректные числа, попробуйте ещё раз.")
            continue

        if not (0 <= x_user <= map_pix_w and 0 <= y_user <= map_pix_h):
            print("[warning] Координаты выходят за пределы карты, введите новые.")
            continue

        # Преобразуем в координаты pygame (origin – левый верхний)
        x_screen = offset_x + x_user
        y_screen = offset_y + map_pix_h - y_user
        return x_screen, y_screen


# ----------------------------------------------------------------------
#  Основная часть программы
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pixel‑based планировщик пути для restaurant‑robot."
    )
    parser.add_argument(
        "--screen", nargs=2, type=int, metavar=("W", "H"),
        help="Размер окна (по умолчанию – текущий размер дисплея)."
    )
    parser.add_argument(
        "--resolution", type=int, default=5,
        help="Размер ячейки сетки планировщика в пикселях (по умолчанию 5)."
    )
    parser.add_argument(
        "--outfile", type=str, default="commands.txt",
        help="Файл, в который будет записан найденный план (по умолчанию commands.txt)."
    )
    parser.add_argument(
        "--margin", type=float, default=5.0,
        help="Дополнительный запас от стен в пикселях (по умолчанию 5)."
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Инициализация pygame (скрытый экран – нам не нужен рендер)
    # --------------------------------------------------------------
    pygame.init()
    if args.screen:
        scr_w, scr_h = args.screen
        screen = pygame.display.set_mode((scr_w, scr_h), pygame.HIDDEN)
    else:
        info = pygame.display.Info()
        scr_w, scr_h = info.current_w, info.current_h
        screen = pygame.display.set_mode((scr_w, scr_h), pygame.HIDDEN)

    # --------------------------------------------------------------
    # Генерация карты (стены, отрисовка, стартовая позиция)
    # --------------------------------------------------------------
    walls, draw_items, start_pos = generate_restaurant_map(screen)

    # ------- масштаб и отступ (тот же, что и в generate_restaurant_map) -------
    scale = min(scr_w, scr_h) / 100.0
    offset_x = (scr_w - 100 * scale) / 2.0
    offset_y = (scr_h - 100 * scale) / 2.0

    # --------------------------------------------------------------
    # Запрос координат цели у пользователя
    # --------------------------------------------------------------
    target_px = get_target_pixel_from_user(scale, offset_x, offset_y)

    # --------------------------------------------------------------
    # Планировщик A* (с учётом запаса от стен)
    # --------------------------------------------------------------
    # Приблизительный «полупростой» размер робота – половина ширины спрайта
    robot_radius = max(args.margin, 5.0) + 20.0
    planner = AStarGrid(scr_w, scr_h, args.resolution, walls, robot_radius)

    try:
        raw_path = planner.a_star(start_pos, target_px)
    except Exception as exc:
        print(f"[error] Не удалось построить путь: {exc}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)

    # --------------------------------------------------------------
    # Сглаживание (удаляем лишние точки) и формирование команд
    # --------------------------------------------------------------
    smooth = smooth_path(raw_path, walls)
    commands = path_to_commands(smooth, start_angle=0.0)

    # --------------------------------------------------------------
    # Запись команд в файл
    # --------------------------------------------------------------
    write_commands(commands, args.outfile)
    print(f"\n[info] Путь найден! Записано {len(commands)} команд в файл «{args.outfile}».\n")

    pygame.quit()


if __name__ == "__main__":
    main()
