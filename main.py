#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Планировщик пути с обходом препятствий.
Если указать параметр ``--route <файл)``, в качестве маршрута берётся список
логических точек из JSON‑файла и они сразу передаются планировщику.
"""

import argparse
import json
import socket
import threading
import time
import sys
import math
from typing import List, Tuple, Optional

import pygame

# ----------------------------------------------------------------------
#  Импорт из наших модулей
# ----------------------------------------------------------------------
from constants import MAP_SCALE, MAP_WIDTH, MAP_HEIGHT
from utils import (
    path_to_commands,
    logical_to_world,
    robot_collides,
)
from grid_map import GridMap
from route_parser import parse_route_from_text   # локальный парсер (на случай fallback)

# ----------------------------------------------------------------------
#  Импорт функции генерации карты ресторана
# ----------------------------------------------------------------------
# generate_restaurant_map находится в robot.py.  Если её не импортировать,
# код падает с «Unresolved reference».  Оборачиваем в try/except – как было в
# оригинальном файле, чтобы дать понятное сообщение в случае потери файла.
try:
    from robot import generate_restaurant_map
except ImportError:
    print("[error] Не удалось импортировать generate_restaurant_map из robot.py.")
    sys.exit(1)

# ----------------------------------------------------------------------
#  TCP‑сервер (тот же, что был)
# ----------------------------------------------------------------------
class RobotServer(threading.Thread):
    """TCP‑сервер для общения с роботом."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.robot_state: Optional[dict] = None
        self.state_lock = threading.Lock()
        self.conn: Optional[socket.socket] = None
        self.running = True
        self.connected_evt = threading.Event()

    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            print(f"\n[сервер] Ожидание подключения робота на {self.host}:{self.port} …")
            self.conn, addr = srv.accept()
            print(f"[сервер] Робот подключён: {addr}")
            self.conn.settimeout(0.5)
            self.connected_evt.set()
            buffer = b""

            while self.running:
                try:
                    data = self.conn.recv(1024)
                    if data:
                        buffer += data
                        while b'\n' in buffer:
                            line, buffer = buffer.split(b'\n', 1)
                            try:
                                state = json.loads(line.decode())
                                with self.state_lock:
                                    self.robot_state = state
                            except Exception as e:
                                print("[сервер] Ошибка разбора сообщения:", e)
                    else:
                        print("[сервер] Робот отключился.")
                        self.running = False
                except socket.timeout:
                    pass
                except Exception as e:
                    print("[сервер] Ошибка сокета:", e)
                    self.running = False

    def get_state(self) -> Optional[dict]:
        with self.state_lock:
            return self.robot_state

    def send_commands(self, commands: List[Tuple[str, float]]) -> None:
        if not self.conn:
            print("[сервер] Нет подключения к роботу")
            return
        try:
            for cmd, val in commands:
                line = f"{cmd} {val}\n"
                self.conn.sendall(line.encode())
        except Exception as e:
            print("[сервер] Ошибка отправки команд:", e)
            self.running = False

    def stop(self) -> None:
        self.running = False
        if self.conn:
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
                self.conn.close()
            except Exception:
                pass
        self.connected_evt.clear()


# ----------------------------------------------------------------------
#  Сеточная карта (без изменений)
# ----------------------------------------------------------------------
def create_grid_map(walls: List[pygame.Rect], cell_size: int = 50) -> GridMap:
    print(f"\n[сетка] Создание сетки {MAP_WIDTH}x{MAP_HEIGHT} с ячейкой {cell_size}px")
    grid_map = GridMap(MAP_WIDTH, MAP_HEIGHT, cell_size)

    print(f"[сетка] Обработка {len(walls)} препятствий…")
    for wall in walls:
        grid_map.mark_occupied(wall)

    free_cells = sum(1 for row in grid_map.grid for cell in row if cell)
    occupied_cells = sum(1 for row in grid_map.grid for cell in row if not cell)

    print(f"[сетка] Сетка: {grid_map.cols}x{grid_map.rows} ячеек")
    print(f"[сетка] Свободных: {free_cells}, занятых: {occupied_cells}")
    return grid_map


# ----------------------------------------------------------------------
#  Упрощение пути (по‑прежнему используется)
# ----------------------------------------------------------------------
def simplify_path(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    for i in range(3, len(path), 3):
        simplified.append(path[i])
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    return simplified


# ----------------------------------------------------------------------
#  Планировщик с обходом препятствий (исправлен метод plan_and_execute)
# ----------------------------------------------------------------------
class ObstacleAvoidancePlanner:
    """Планировщик с обходом препятствий."""

    def __init__(self, grid_map: GridMap, walls: List[pygame.Rect], server: RobotServer):
        self.grid_map = grid_map
        self.walls = walls
        self.server = server
        self.max_attempts = 5
        self.retreat_distance = 100.0

    # --------------------------------------------------------------
    def wait_for_robot_completion(self, timeout: float = 30.0) -> Tuple[bool, Tuple[float, float]]:
        start = time.time()
        last_pos = None
        stationary = 0.0

        while time.time() - start < timeout:
            state = self.server.get_state()
            if state:
                cur = (float(state["pos"][0]), float(state["pos"][1]))
                if last_pos:
                    moved = math.hypot(cur[0] - last_pos[0], cur[1] - last_pos[1])
                    if moved < 5.0:
                        stationary += 0.5
                    else:
                        stationary = 0.0
                    if stationary > 2.0:
                        return True, cur
                last_pos = cur
            time.sleep(0.5)

        return (False, last_pos if last_pos else (0, 0))

    # --------------------------------------------------------------
    def retreat_from_obstacle(self, cur_pos: Tuple[float, float],
                              cur_angle: float) -> Optional[Tuple[float, float]]:
        print("[обход] Отъезд от препятствия...")
        safe = self.grid_map.get_safe_retreat_position(cur_pos, self.walls)

        if safe and safe != cur_pos:
            cmds = path_to_commands([cur_pos, safe], start_angle=cur_angle)
            if cmds:
                self.server.send_commands(cmds)
                ok, new = self.wait_for_robot_completion(10.0)
                if ok:
                    print("[обход] Успешный отъезд")
                    return new

        # запасной вариант – назад
        print("[обход] Прямой отъезд назад")
        self.server.send_commands([("BACKWARD", self.retreat_distance)])
        ok, new = self.wait_for_robot_completion(10.0)
        return new if ok else None

    # --------------------------------------------------------------
    def plan_and_execute(self,
                         start_pos: Tuple[float, float],
                         start_angle: float,
                         target_pos: Tuple[float, float],
                         target_name: str) -> bool:
        """
        Планирует путь к *одной* цели и исполняет его.
        Теперь считается, что цель достигнута, как только робот
        завершил движение (без проверки расстояния), но при этом
        проверяется наличие столкновения – в этом случае выполняется
        отъезд и повторная попытка.
        """
        # ------------------------------------------------------------------
        #  Проверка валидности целевой точки
        # ------------------------------------------------------------------
        if target_pos is None:
            print(f"[план] Ошибка: целевая точка не задана для «{target_name}»")
            return False

        # Если уже на месте (с небольшим допуском) – считаем шаг выполненным
        if math.isclose(start_pos[0], target_pos[0], abs_tol=1e-2) and \
           math.isclose(start_pos[1], target_pos[1], abs_tol=1e-2):
            print(f"[план] Целевая точка уже достигнута ( {target_name} ). "
                  "Продолжаем со следующей.")
            return True

        # ------------------------------------------------------------------
        cur_pos = start_pos
        cur_angle = start_angle

        print(f"\n[план] Движение к цели «{target_name}»")
        print(f"[план] Старт {cur_pos} → цель {target_pos}")

        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            print(f"\n[попытка {attempt}/{self.max_attempts}] Поиск пути…")
            path = self.grid_map.find_path(cur_pos, target_pos)

            if not path:
                print("[ошибка] Путь не найден")
                return False

            print(f"[план] Путь найден ({len(path)} точек)")
            simple = simplify_path(path)
            cmds = path_to_commands(simple, start_angle=cur_angle)
            print(f"[план] Сгенерировано {len(cmds)} команд")
            self.server.send_commands(cmds)

            print("[план] Ожидание завершения движения…")
            success, new_pos = self.wait_for_robot_completion()

            if not success:
                print("[план] Таймаут ожидания робота")
                # Попробуем ещё раз (переходим к следующей попытке)
                continue

            # ------------------------------------------------------
            #  Проверяем, не случилось ли столкновение
            # ------------------------------------------------------
            if robot_collides(new_pos, self.walls, 20.0):
                print("[обнаружено] Столкновение!")
                state = self.server.get_state()
                if state:
                    cur_angle = float(state["angle"])
                retreat_pos = self.retreat_from_obstacle(new_pos, cur_angle)
                if retreat_pos:
                    cur_pos = retreat_pos
                    print(f"[обход] Продолжаем с {cur_pos}")
                    # После отъезда пробуем снова (новая итерация цикла)
                    continue
                else:
                    print("[ошибка] Не удалось отъехать")
                    return False

            # ------------------------------------------------------
            #  Если столкновения НЕ было – считаем шаг выполненным
            # ------------------------------------------------------
            print(f"[успех] Шаг «{target_name}» завершён (проверка прибытия отключена).")
            return True

        print("[неудача] Не удалось достичь цели за максимум попыток")
        return False


# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Планировщик пути с обходом препятствий + поддержка готового маршрута."
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=50,
        help="Размер ячейки сетки в пикселях (по умолчанию 50).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="IP‑адрес для TCP‑сервера.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Порт для TCP‑сервера.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Максимальное количество попыток достижения цели.",
    )
    parser.add_argument(
        "--route",
        type=str,
        help="Путь к JSON‑файлу с готовым маршрутом (логические координаты).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Отключить генерацию маршрута через ИИ (использовать лишь локальный парсер).",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Инициализация Pygame
    # --------------------------------------------------------------
    pygame.init()

    # --------------------------------------------------------------
    # Генерация карты ресторана
    # --------------------------------------------------------------
    print("\n[инициализация] Генерация карты ресторана…")
    walls, draw_items, start_pos = generate_restaurant_map()
    print(f"[инициализация] Карта сгенерирована, препятствий: {len(walls)}")

    # --------------------------------------------------------------
    # Сетка
    # --------------------------------------------------------------
    grid_map = create_grid_map(walls, args.cell_size)

    # --------------------------------------------------------------
    # Запуск TCP‑сервера и ожидание робота
    # --------------------------------------------------------------
    print("\n[сервер] Запуск TCP‑сервера…")
    server = RobotServer(host=args.host, port=args.port)
    server.start()

    print("[сервер] Ожидание подключения робота…")
    for attempt in range(30):
        if server.connected_evt.wait(timeout=1.0):
            print("[сервер] ✓ Робот подключён!")
            break
        if attempt % 5 == 0:
            print(f"[сервер] Ожидание… ({attempt + 1}/30)")
    else:
        print("[ошибка] Робот не подключился за 30 сек.")
        server.stop()
        pygame.quit()
        sys.exit(1)

    # --------------------------------------------------------------
    # Планировщик
    # --------------------------------------------------------------
    planner = ObstacleAvoidancePlanner(grid_map, walls, server)
    planner.max_attempts = args.max_attempts

    # ------------------------------------------------------------------
    #  Если передан файл с готовым маршрутом – запускаем его сразу
    # ------------------------------------------------------------------
    if args.route:
        try:
            with open(args.route, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ошибка] Не удалось открыть файл маршрута: {e}")
            server.stop()
            pygame.quit()
            sys.exit(1)

        # Поддерживаем два варианта структуры: {"route": [...]} или {"points": [...]}
        logical_points = data.get("route") or data.get("points") or data
        if not isinstance(logical_points, list):
            print("[ошибка] Неверный формат файла маршрута (ожидается список точек).")
            server.stop()
            pygame.quit()
            sys.exit(1)

        print("[info] Маршрут из файла:", logical_points)

        # ------------------------------------------------------------------
        # Получаем текущую позицию робота
        # ------------------------------------------------------------------
        robot_state = None
        while robot_state is None:
            robot_state = server.get_state()
            time.sleep(0.1)

        robot_pos = (float(robot_state["pos"][0]), float(robot_state["pos"][1]))
        robot_angle = float(robot_state["angle"])

        # ------------------------------------------------------------------
        # Последовательно проходим все точки маршрута
        # ------------------------------------------------------------------
        all_ok = True
        for step_idx, logical_pt in enumerate(logical_points, 1):
            target_px = logical_to_world(logical_pt, MAP_SCALE)
            target_name = f"step_{step_idx}"
            print(f"\n[маршрут] Шаг {step_idx}/{len(logical_points)} → {target_name}")

            success = planner.plan_and_execute(
                start_pos=robot_pos,
                start_angle=robot_angle,
                target_pos=target_px,
                target_name=target_name,
            )

            if not success:
                print(f"[ошибка] Не удалось выполнить шаг {step_idx}")
                all_ok = False
                break

            # Обновляем позицию и угол робота перед следующей точкой
            state = server.get_state()
            if state:
                robot_pos = (float(state["pos"][0]), float(state["pos"][1]))
                robot_angle = float(state["angle"])

        if all_ok:
            print("\n✅ Маршрут полностью выполнен!")
        else:
            print("\n⚠️ Маршрут прерван из‑за ошибки.")

        # Всё – завершаем программу
        server.stop()
        pygame.quit()
        sys.exit(0)

    # ------------------------------------------------------------------
    #  Если файл маршрута НЕ указан – работаем в интерактивном режиме
    # ------------------------------------------------------------------
    try:
        while True:
            print("\n" + "=" * 60)

            # ---------- 1. Запрос у пользователя ----------
            description = input(
                "\nВведите описание маршрута (пример: «от кухни к бару, потом к столу 1 и 4»)\n>> "
            ).strip()
            if not description:
                continue

            # ---------- 2. Попытка получить маршрут от ИИ ----------
            logical_route: List[Tuple[float, float]] = []

            if not args.no_ai:
                # Попытаемся запросить ИИ (это отдельный скрипт‑модуль, но он может
                # вернуть пустой список, тогда будем использовать локальный парсер).
                try:
                    from route_generator import RouteGenerator
                    generator = RouteGenerator()
                    logical_route = generator.generate_route(description)
                    if logical_route:
                        print("[ИИ] Маршрут получен от модели.")
                except Exception as e:
                    print(f"[ИИ] Ошибка обращения к модели: {e}")

            # ---------- 3. Если ИИ не дал результата – fallback ----------
            if not logical_route:
                logical_route = parse_route_from_text(description)
                if logical_route:
                    print("[fallback] Маршрут получен локальным парсером.")
                else:
                    print("[ошибка] Не удалось извлечь точки из запроса.")
                    continue

            print("[info] Точки (логические координаты):")
            for i, pt in enumerate(logical_route, 1):
                print(f"  {i}. {pt}")

            # ---------- 4. Текущее состояние робота ----------
            robot_state = None
            while robot_state is None:
                robot_state = server.get_state()
                time.sleep(0.1)

            robot_pos = (float(robot_state["pos"][0]), float(robot_state["pos"][1]))
            robot_angle = float(robot_state["angle"])

            # ---------- 5. Последовательно выполняем точки ----------
            all_ok = True
            for step_idx, logical_pt in enumerate(logical_route, 1):
                target_px = logical_to_world(logical_pt, MAP_SCALE)
                target_name = f"step_{step_idx}"
                print(f"\n[маршрут] Шаг {step_idx}/{len(logical_route)} → {target_name}")

                success = planner.plan_and_execute(
                    start_pos=robot_pos,
                    start_angle=robot_angle,
                    target_pos=target_px,
                    target_name=target_name,
                )

                if not success:
                    print(f"[ошибка] Не удалось выполнить шаг {step_idx}")
                    all_ok = False
                    break

                # Обновляем позицию/угол перед следующей точкой
                state = server.get_state()
                if state:
                    robot_pos = (float(state["pos"][0]), float(state["pos"][1]))
                    robot_angle = float(state["angle"])

            if all_ok:
                print("\n✅ Маршрут полностью выполнен!")
            else:
                print("\n⚠️ Маршрут прерван.")

            # --------------------------------------------------------------
            # Спрашиваем, хотите ли построить новый маршрут?
            # --------------------------------------------------------------
            again = input("\nСформировать новый маршрут? (да/нет) ").strip().lower()
            if again not in ("да", "д", "yes", "y"):
                print("\n[завершение] Выход из планировщика.")
                break

    except KeyboardInterrupt:
        print("\n\n[info] Работа прервана пользователем")
    finally:
        print("\n[завершение] Остановка сервера…")
        server.stop()
        pygame.quit()
        print("[завершение] Программа завершена")


if __name__ == "__main__":
    main()
