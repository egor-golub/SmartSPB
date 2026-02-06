#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telegram‑бот, генерирующий маршрут (список точек) для робота и
выполняющий его «по‑одному»: после каждой успешно пройденной
точки отправляется следующая.  Ядро‑планировщик в виде отдельного
файла‑маршрута больше не используется.
"""

import os
import sys
import json
import time
import asyncio
import logging
import queue                       # <-- добавлен импорт
from datetime import datetime
from typing import List, Tuple   # <-- Deque удалён, он не нужен

from dotenv import load_dotenv

from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    filters,
)

# ----------------------------------------------------------------------
#  Модули проекта
# ----------------------------------------------------------------------
from constants import MAP_SCALE                     # масштаб карты (пиксели = логика * MAP_SCALE)
from utils import logical_to_world                 # логические → пиксели
from route_parser import parse_route_from_text    # fallback‑парсер
from route_generator import RouteGenerator         # генератор маршрута через LLM
from planner_core import init_system                # создаёт GridMap, сервер и планировщик

# ----------------------------------------------------------------------
#  Конфигурация
# ----------------------------------------------------------------------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")   # «disabled» → без LLM

if not TELEGRAM_BOT_TOKEN:
    sys.exit("❌ Установите TELEGRAM_BOT_TOKEN в .env файле")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#  Вспомогательные функции (необязательно сохраняют маршрут в файл)
# ----------------------------------------------------------------------
def save_route_to_file(route_points: List[Tuple[float, float]], file_path: str = "route.json") -> None:
    """Сохраняет полученный маршрут в JSON‑файл (оставлено для совместимости)."""
    data = {"route": route_points, "timestamp": datetime.now().isoformat()}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Маршрут сохранён в {file_path} ({len(route_points)} точек)")

def parse_ai_response_to_points(response_text: str) -> List[Tuple[float, float]]:
    """
    Выделяет из ответа LLM массив точек.
    Поддерживает JSON вида {"route": [[x,y], …]} и простой набор координат.
    """
    import re

    pts: List[Tuple[float, float]] = []

    # 1️⃣ JSON‑блок
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            if isinstance(obj, dict) and "route" in obj:
                for p in obj["route"]:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        pts.append((float(p[0]), float(p[1])))
                return pts
        except Exception:
            pass

    # 2️⃣ Любой набор «x y», «x, y», «[x, y]» …
    coord_pat = re.compile(r"(\d+(?:\.\d+)?)\s*[ ,;]\s*(\d+(?:\.\d+)?)")
    for x, y in coord_pat.findall(response_text):
        pts.append((float(x), float(y)))
    return pts

# ----------------------------------------------------------------------
#  Класс, который хранит очередь точек и посылает их планировщику
# ----------------------------------------------------------------------
class SequentialRouteExecutor:
    """
    Очередь логических точек → планировщик.
    После каждой успешно пройденной точки ждёт подтверждения
    (внутри `ObstacleAvoidancePlanner.plan_and_execute` уже реализовано
    ожидание завершения движения и проверка столкновений).
    """

    def __init__(self, planner, server):
        """
        **Важно**: ранее использовалась ``asyncio.Queue``.  Она работает
        только внутри одного event‑loop, а здесь мы вызываем `run()` в отдельном
        потоке, что приводило к «застыванию» после первой точки.
        Теперь используется ``queue.Queue`` – потокобезопасный объект из
        стандартной библиотеки, который корректно работает в любом потоке.
        """
        self.planner = planner          # ObstacleAvoidancePlanner
        self.server = server            # RobotServer
        self._queue = queue.Queue()     # потокобезопасная очередь (логика → мир)

    # --------------------------------------------------------------
    def add_points(self, logical_pts: List[Tuple[float, float]]) -> None:
        """Заполняет очередь точек (логические координаты → мир‑пиксели делаем позже)."""
        start_idx = self._queue.qsize() + 1
        for i, pt in enumerate(logical_pts, start=start_idx):
            # Планировщик ждёт имя, чтобы вывести в логах
            self._queue.put_nowait((pt, f"step_{i}"))  # очередь из (точка, имя)

    # --------------------------------------------------------------
    def _wait_for_state(self) -> Tuple[Tuple[float, float], float]:
        """Блокирующий опрос сервера до получения текущего состояния робота."""
        while True:
            state = self.server.get_state()
            if state:
                pos = (float(state["pos"][0]), float(state["pos"][1]))
                ang = float(state["angle"])
                return pos, ang
            time.sleep(0.1)

    # --------------------------------------------------------------
    def run(self) -> bool:
        """
        Последовательно берёт точку из очереди, переводит её в пиксели,
        запускает планировщик и ждёт завершения.
        Возвращает ``True``, если весь маршрут пройден, иначе ``False``.
        """
        # Текущее положение робота – получаем один раз в начале,
        # а потом обновляем после каждой итерации.
        cur_pos, cur_ang = self._wait_for_state()

        while not self._queue.empty():
            logical_pt, name = self._queue.get_nowait()

            # ---- проверка наличия следующей точки ---------------------------------
            if logical_pt is None:
                # Пустая точка может появиться, если пользователь ввёл
                # некорректный запрос.  Пропускаем её и продолжаем.
                continue
            # ---------------------------------------------------------------------

            target_px = logical_to_world(logical_pt, MAP_SCALE)

            success = self.planner.plan_and_execute(
                start_pos=cur_pos,
                start_angle=cur_ang,
                target_pos=target_px,
                target_name=name,
            )
            if not success:
                return False

            # Обновляем положение/угол перед следующей точкой
            cur_pos, cur_ang = self._wait_for_state()

        return True

# ----------------------------------------------------------------------
#  Основной класс бота
# ----------------------------------------------------------------------
class RouteGeneratorBot:
    """Telegram‑бот с генерацией и последовательным выполнением маршрутов."""

    def __init__(self):
        self.logger = logger
        self.ollama_client = None          # lazy‑init
        self._run_lock = asyncio.Lock()    # один активный маршрут одновременно
        self.bot = None                    # будет установлен в generate_route

    # ------------------------------------------------------------------
    async def _ensure_ollama_client(self):
        """Отложенно создаёт клиент Ollama (если нужен)."""
        if self.ollama_client is None and OLLAMA_MODEL != "disabled":
            from ollama_client import OllamaClient

            self.ollama_client = OllamaClient(model=OLLAMA_MODEL)
        return self.ollama_client

    # ------------------------------------------------------------------
    async def start(self, update: Update, context: CallbackContext) -> None:
        """Ответ на /start."""
        user = update.effective_user
        await update.message.reply_html(
            fr"🗺️ **Привет, {user.mention_html()}!**\n\n"
            "Я генерирую маршрут (список точек) для вашего робота.\n"
            "🔹 **Как пользоваться** – отправьте описание маршрута.\n"
            "Примеры:\n"
            "• `От кухни к бару, потом к столу 1 и столу 3`\n"
            "• `Координаты: 12,12 потом 50,6 потом 35,27`\n"
            "После получения маршрута я сразу запущу планировщик."
        )

    # ------------------------------------------------------------------
    async def help_command(self, update: Update, context: CallbackContext) -> None:
        """Ответ на /help."""
        help_text = (
            "ℹ️ **Помощь**\n\n"
            "Список известных точек (логические координаты 0‑100):\n"
            "• КУХНЯ (12,12)\n"
            "• БАР (50,6)\n"
            "• СТОЛ 1 (35,27)\n"
            "• СТОЛ 2 (55,27)\n"
            "• СТОЛ 3 (35,54)\n"
            "• СТОЛ 4 (55,54)\n"
            "• ПИЛЛАР (68,70)\n\n"
            "Можно указывать названия по‑русски, по‑английски, через цифры и/или запятые."
        )
        await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)

    # ------------------------------------------------------------------
    async def status_command(self, update: Update, context: CallbackContext) -> None:
        """Состояние бота и Ollama."""
        ollama_ok = "✅" if (await self._ensure_ollama_client()).check_connection() else "❌"
        await update.message.reply_text(
            f"🔧 **Статус**\n"
            f"🤖 Модель ИИ: {OLLAMA_MODEL}\n"
            f"📡 Ollama: {ollama_ok}\n"
            f"✅ Bot активен",
            parse_mode=constants.ParseMode.MARKDOWN,
        )

    # ------------------------------------------------------------------
    async def _send(self, chat_id: int, text: str,
                    parse_mode=constants.ParseMode.MARKDOWN) -> None:
        """Универсальная отправка сообщений через `self.bot`."""
        try:
            await self.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as exc:   # pragma: no cover
            self.logger.exception(f"Не удалось отправить сообщение {chat_id}: {exc}")

    # ------------------------------------------------------------------
    async def generate_route(self, update: Update, context: CallbackContext) -> None:
        """
        1️⃣ Приём текста от пользователя.
        2️⃣ Генерация списка логических точек (LLM → fallback‑парсер).
        3️⃣ Инициализация планировщика и последовательное выполнение точек.
        """
        description = update.message.text.strip()
        if not description:
            await update.message.reply_text("❌ Пустой запрос. Попробуйте ещё раз.")
            return

        chat_id = update.effective_chat.id
        self.bot = context.bot               # нужен для `_send`
        await update.message.reply_text(
            f"🔄 Обрабатываю запрос…\n`{description}`",
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ------------------------------------------------------ 2️⃣ Получаем маршрут
        route_points: List[Tuple[float, float]] = []

        # 2.1 – Попытка через LLM (если модель не отключена)
        if OLLAMA_MODEL != "disabled":
            try:
                generator = RouteGenerator(ollama_model=OLLAMA_MODEL)
                route_points = generator.generate_route(description)
            except Exception as e:   # pragma: no cover
                self.logger.warning(f"Ошибка LLM‑генератора: {e}")

        # 2.2 – fallback‑парсер, если LLM ничего не вернул
        if not route_points:
            route_points = parse_route_from_text(description)
            if not route_points:
                await update.message.reply_text(
                    "❌ Не удалось извлечь точки ни из LLM, ни локальным парсером.\n"
                    "Укажите координаты явно, например `12 12 50 6`."
                )
                return
            else:
                await update.message.reply_text("⚙️ LLM не сработал – использован локальный парсер.")

        # Информируем пользователя о полученном маршруте
        await update.message.reply_text(
            f"✅ Получено {len(route_points)} точек.\n"
            "🚀 Запускаю планировщик и начинаю последовательную отправку…"
        )

        # ------------------------------------------------------ 3️⃣ Инициализируем планировщик
        try:
            planner, server = init_system(
                cell_size=50,
                host="127.0.0.1",
                port=5555,
                max_attempts=5,
            )
        except Exception as e:   # pragma: no cover
            await self._send(chat_id, f"❌ Не удалось инициализировать планировщик: {e}")
            return

        # ------------------------------------------------------ 4️⃣ Последовательное исполнение
        async with self._run_lock:                     # гарантируем один активный маршрут
            executor = SequentialRouteExecutor(planner, server)
            executor.add_points(route_points)

            # Запуск в отдельном потоке, чтобы не блокировать event‑loop
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(None, executor.run)

            # Останавливаем сервер независимо от результата
            server.stop()

            if success:
                await self._send(chat_id, "✅ Маршрут полностью выполнен!")
            else:
                await self._send(chat_id, "❌ Ошибка во время выполнения маршрута – остановлен.")

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Запуск бота (polling)."""
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CommandHandler("route", self.generate_route))
        # Любой обычный текст трактуем как запрос маршрута
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.generate_route))

        self.logger.info("✅ Bot запущен и ждёт сообщений.")
        app.run_polling()


# ----------------------------------------------------------------------
#  Точка входа
# ----------------------------------------------------------------------
if __name__ == "__main__":
    RouteGeneratorBot().run()
