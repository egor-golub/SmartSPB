#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Небольшой «фабричный» модуль, который собирает всё нужное для работы
планировщика: карта, сервер, планировщик‑обходчик.
Его удобно импортировать из консольного скрипта и из Telegram‑бота.
"""

import time
import logging
import pygame

from constants import MAP_SCALE, MAP_WIDTH, MAP_HEIGHT
from utils import logical_to_world
from grid_map import GridMap
from robot import generate_restaurant_map
from main import (   # импортируем только нужные классы/функции, чтобы не дублировать код
    RobotServer,
    ObstacleAvoidancePlanner,
    create_grid_map,
)

def init_system(
    *,
    cell_size: int = 50,
    host: str = "127.0.0.1",
    port: int = 5555,
    max_attempts: int = 5,
) -> tuple[ObstacleAvoidancePlanner, RobotServer]:
    """
    Создаёт карту, запускает TCP‑сервер и возвращает готовый
    объект `ObstacleAvoidancePlanner` и сервер.

    Возврат:
        planner – объект, умеющий `plan_and_execute(...)`
        server  – объект, откуда можно получать текущее состояние робота.
    """
    # ------------------------------------------------------------------
    # 1. Генерация карты ресторана (мировые координаты)
    # ------------------------------------------------------------------
    pygame.init()
    walls, draw_items, start_pos = generate_restaurant_map()

    # ------------------------------------------------------------------
    # 2. Сетка (GridMap)
    # ------------------------------------------------------------------
    grid_map = create_grid_map(walls, cell_size)

    # ------------------------------------------------------------------
    # 3. TCP‑сервер (RobotServer из main.py)
    # ------------------------------------------------------------------
    server = RobotServer(host=host, port=port)
    server.start()

    # Ждём подключения робота (до 30 сек)
    for _ in range(30):
        if server.connected_evt.wait(timeout=1.0):
            logging.info("[planner_core] Робот подключён")
            break
        time.sleep(0.2)
    else:
        raise RuntimeError("Не удалось подключиться к роботу в течение 30 сек.")

    # ------------------------------------------------------------------
    # 4. Планировщик‑обходчик
    # ------------------------------------------------------------------
    planner = ObstacleAvoidancePlanner(grid_map, walls, server)
    planner.max_attempts = max_attempts

    return planner, server
