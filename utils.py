#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Вспомогательные функции для планировщика пути.
"""

import math
from typing import List, Tuple, Optional
import pygame


def path_to_commands(
        path: List[Tuple[float, float]],
        start_angle: float = 0.0,
) -> List[Tuple[str, float]]:
    """Преобразует путь в команды ROTATE/FORWARD."""
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

        target_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
        delta = ((target_angle - cur_angle + 180) % 360) - 180

        if abs(delta) > 1e-4:
            cmds.append(("ROTATE", round(delta, 2)))
        cmds.append(("FORWARD", round(dist, 2)))

        cur_angle = target_angle

    return cmds


def logical_to_world(logical: Tuple[float, float], scale: float) -> Tuple[float, float]:
    """Переводит логические координаты (0-100) в мир-пиксели."""
    lx, ly = logical
    return (float(lx) * scale, float(ly) * scale)


def print_commands_summary(commands: List[Tuple[str, float]]) -> None:
    """Печатает сводку о командах."""
    total_distance = sum(val for cmd, val in commands if cmd == "FORWARD")
    total_rotation = sum(abs(val) for cmd, val in commands if cmd == "ROTATE")

    print(f"  Количество команд: {len(commands)}")
    print(f"  Общее расстояние: {total_distance:.1f} пикселей")
    print(f"  Общий поворот: {total_rotation:.1f} градусов")

    print("  Первые 5 команд:")
    for i, (cmd, val) in enumerate(commands[:5], 1):
        print(f"    {i}. {cmd} {val}")
    if len(commands) > 5:
        print(f"    ... и ещё {len(commands) - 5} команд")


def robot_collides(
        pos: Tuple[float, float],
        walls: List[pygame.Rect],
        radius: float,
) -> bool:
    """True, если круг радиуса radius с центром pos попадает в стену."""
    robot_rect = pygame.Rect(
        pos[0] - radius,
        pos[1] - radius,
        radius * 2,
        radius * 2,
    )
    return any(wall.colliderect(robot_rect) for wall in walls)


def find_closest_wall(pos: Tuple[float, float], walls: List[pygame.Rect]) -> Tuple[Optional[pygame.Rect], float]:
    """Находит ближайшую стену и расстояние до нее."""
    if not walls:
        return None, float('inf')

    closest_wall = None
    min_distance = float('inf')

    for wall in walls:
        # Расстояние от точки до прямоугольника
        dx = max(wall.left - pos[0], 0, pos[0] - wall.right)
        dy = max(wall.top - pos[1], 0, pos[1] - wall.bottom)
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < min_distance:
            min_distance = distance
            closest_wall = wall

    return closest_wall, min_distance


def get_retreat_direction(pos: Tuple[float, float], wall: pygame.Rect) -> Tuple[float, float]:
    """
    Определяет направление для отъезда от стены.
    Возвращает вектор направления (dx, dy).
    """
    # Находим ближайшую сторону стены
    distances = [
        (abs(pos[0] - wall.left), (-1, 0)),  # слева
        (abs(pos[0] - wall.right), (1, 0)),  # справа
        (abs(pos[1] - wall.top), (0, -1)),  # сверху
        (abs(pos[1] - wall.bottom), (0, 1))  # снизу
    ]

    # Выбираем направление, противоположное ближайшей стороне
    distances.sort(key=lambda x: x[0])
    direction = distances[0][1]

    # Инвертируем направление (отъезжаем от стены)
    return (-direction[0], -direction[1])