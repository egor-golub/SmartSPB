#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сеточная система для планировщика пути.
"""

import math
import heapq
from typing import List, Tuple, Set, Dict, Optional
import pygame


class GridMap:
    """Карта, разделенная на квадратные ячейки."""

    def __init__(self, width_px: int, height_px: int, cell_size: int):
        self.width_px = width_px
        self.height_px = height_px
        self.cell_size = cell_size

        self.cols = math.ceil(width_px / cell_size)
        self.rows = math.ceil(height_px / cell_size)

        # Сетка: True - свободно, False - занято (препятствие)
        self.grid = [[True for _ in range(self.cols)] for _ in range(self.rows)]

    def world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Преобразует мировые координаты в координаты сетки."""
        x, y = world_pos
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)

        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        return (col, row)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Преобразует координаты сетки в мировые координаты (центр ячейки)."""
        col, row = grid_pos
        x = col * self.cell_size + self.cell_size / 2
        y = row * self.cell_size + self.cell_size / 2
        return (x, y)

    def mark_occupied(self, rect: pygame.Rect) -> None:
        """Помечает ячейки, пересекающиеся с прямоугольником, как занятые."""
        left_col = max(0, int(rect.left // self.cell_size))
        right_col = min(self.cols - 1, int(rect.right // self.cell_size))
        top_row = max(0, int(rect.top // self.cell_size))
        bottom_row = min(self.rows - 1, int(rect.bottom // self.cell_size))

        for row in range(top_row, bottom_row + 1):
            for col in range(left_col, right_col + 1):
                self.grid[row][col] = False

    def is_free(self, grid_pos: Tuple[int, int]) -> bool:
        """Проверяет, свободна ли ячейка."""
        col, row = grid_pos
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return self.grid[row][col]
        return False

    def get_neighbors(self, grid_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Возвращает соседние свободные ячейки (8-связность)."""
        col, row = grid_pos
        neighbors = []

        for dcol in [-1, 0, 1]:
            for drow in [-1, 0, 1]:
                if dcol == 0 and drow == 0:
                    continue

                new_col = col + dcol
                new_row = row + drow

                if (0 <= new_col < self.cols and
                        0 <= new_row < self.rows and
                        self.is_free((new_col, new_row))):
                    neighbors.append((new_col, new_row))

        return neighbors

    def find_path(self, start_world: Tuple[float, float],
                  goal_world: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Ищет путь от старта к цели используя алгоритм A*.
        """
        start_grid = self.world_to_grid(start_world)
        goal_grid = self.world_to_grid(goal_world)

        if not self.is_free(start_grid):
            start_grid = self.find_nearest_free(start_grid)
            if start_grid is None:
                return None

        if not self.is_free(goal_grid):
            goal_grid = self.find_nearest_free(goal_grid)
            if goal_grid is None:
                return None

        open_set = []
        heapq.heappush(open_set, (0, 0, start_grid))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_grid: None}
        g_score: Dict[Tuple[int, int], float] = {start_grid: 0}
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal_grid:
                path_grid = []
                while current is not None:
                    path_grid.append(current)
                    current = came_from[current]
                path_grid.reverse()

                path_world = [self.grid_to_world(pos) for pos in path_grid]
                return path_world

            for neighbor in self.get_neighbors(current):
                # Стоимость перехода
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                cost = 1 if abs(dx) + abs(dy) == 1 else 1.414  # 1 для соседей по стороне, √2 для диагональных

                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # Манхэттенское расстояние в качестве эвристики
                    h = abs(neighbor[0] - goal_grid[0]) + abs(neighbor[1] - goal_grid[1])
                    f = tentative_g + h

                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None

    def find_nearest_free(self, grid_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Ищет ближайшую свободную ячейку от заданной позиции."""
        col, row = grid_pos

        visited = set()
        queue = [(col, row)]
        visited.add((col, row))

        while queue:
            current_col, current_row = queue.pop(0)

            if self.is_free((current_col, current_row)):
                return (current_col, current_row)

            for dcol, drow in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_col = current_col + dcol
                new_row = current_row + drow

                if (0 <= new_col < self.cols and
                        0 <= new_row < self.rows and
                        (new_col, new_row) not in visited):
                    visited.add((new_col, new_row))
                    queue.append((new_col, new_row))

        return None

    def get_safe_retreat_position(self, current_pos: Tuple[float, float],
                                  walls: List[pygame.Rect]) -> Optional[Tuple[float, float]]:
        """
        Находит безопасную позицию для отъезда от препятствия.
        Ищет ближайшую свободную позицию в обратном направлении от ближайшей стены.
        """
        from utils import robot_collides

        current_grid = self.world_to_grid(current_pos)

        # Проверяем все направления отъезда
        directions = [
            (0, -1),  # вверх
            (0, 1),  # вниз
            (-1, 0),  # влево
            (1, 0),  # вправо
            (-1, -1),  # вверх-влево
            (1, -1),  # вверх-вправо
            (-1, 1),  # вниз-влево
            (1, 1)  # вниз-вправо
        ]

        # Ищем безопасное направление для отъезда
        for dx, dy in directions:
            for distance in [1, 2, 3]:  # пробуем разные расстояния
                new_col = current_grid[0] + dx * distance
                new_row = current_grid[1] + dy * distance

                if 0 <= new_col < self.cols and 0 <= new_row < self.rows:
                    new_world_pos = self.grid_to_world((new_col, new_row))

                    # Проверяем, нет ли столкновения на новой позиции
                    if not robot_collides(new_world_pos, walls, 20.0):
                        # Проверяем, можно ли проехать к этой позиции
                        if self.is_free((new_col, new_row)):
                            return new_world_pos

        # Если не нашли безопасную позицию, возвращаем текущую
        return current_pos