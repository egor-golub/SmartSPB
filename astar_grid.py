#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Класс для поиска пути методом A*.
"""

import math
import heapq
from collections import deque
from typing import List, Tuple, Optional
import pygame


class AStarGrid:
    """Дискретная сетка для поиска пути методом A*."""

    def __init__(
        self,
        width_px: int,
        height_px: int,
        res: int,
        walls: List[pygame.Rect],
        robot_radius: float = 10.0,
    ):
        self.res = float(res)
        self.cols = int(math.ceil(width_px / self.res))
        self.rows = int(math.ceil(height_px / self.res))
        self.walls = walls
        self.robot_radius = robot_radius

        self.grid = [[True for _ in range(self.cols)] for _ in range(self.rows)]
        self._build_grid()

    def _cell_center(self, gx: int, gy: int) -> Tuple[float, float]:
        """Центр ячейки в пикселях."""
        return (gx * self.res + self.res / 2.0, gy * self.res + self.res / 2.0)

    def _build_grid(self) -> None:
        """Помечаем клетки, пересекающиеся со стенами (с учётом запаса)."""
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

    def neighbors(
        self,
        cell: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int], float]]:
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

    @staticmethod
    def heuristic(
        a: Tuple[int, int],
        b: Tuple[int, int],
        scale: float,
    ) -> float:
        """Эвристика – евклидово расстояние (в пикселях)."""
        ax, ay = a
        bx, by = b
        return math.hypot((ax - bx) * scale, (ay - by) * scale)

    def _nearest_free(
        self,
        start: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Ищет ближайшую свободную клетку, если старт/цель заняты."""
        visited = set()
        q = deque([start])
        visited.add(start)

        while q:
            cx, cy = q.popleft()
            if self.grid[cy][cx]:
                return cx, cy
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return None

    def a_star(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """Возвращает путь (список точек в пикселях)."""
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)

        if not self.grid[start[1]][start[0]]:
            alt = self._nearest_free(start)
            if alt is None:
                raise ValueError(
                    "Стартовая позиция внутри препятствия и свободных соседних ячеек нет"
                )
            start = alt
        if not self.grid[goal[1]][goal[0]]:
            alt = self._nearest_free(goal)
            if alt is None:
                raise ValueError(
                    "Целевая позиция внутри препятствия и свободных соседних ячеек нет"
                )
            goal = alt

        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0, start))
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        f_score: dict[Tuple[int, int], float] = {
            start: self.heuristic(start, goal, self.res)
        }
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # восстановление пути
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