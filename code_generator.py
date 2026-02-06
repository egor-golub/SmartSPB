#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Утилита для вытягивания списка команд робота из текста ответа
LLM‑модели. Всё, что связано с отправкой кода пользователю,
полностью удалено – в проекте теперь не используется.
"""

import re
import json
from typing import List, Union


class CodeGenerator:
    """
    Парсит ответ модели и оставляет только команды, которые понимает
    наш Pygame‑робот:

        FORWARD  <distance>
        BACKWARD <distance>
        ROTATE  <angle>
        WAIT    <seconds>

    Поддерживаются три формата:
        1) markdown‑блок (``` … ```);
        2) JSON‑объект {"commands":[...]};
        3) обычный текст (по строкам).
    """

    def __init__(self, ollama_model: str = "llama2"):
        self.ollama_model = ollama_model

    # ------------------------------------------------------------------
    # Главный метод – извлекает «чистый» список команд robota
    # ------------------------------------------------------------------
    def extract_commands_from_response(self, response: str) -> str:
        """
        Возвращает многострочную строку, где каждая строка – валидная команда.
        Если ничего не найдено – возвращает пустую строку.
        """
        # 1️⃣ markdown‑блок
        block_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```',
                               response, re.DOTALL)
        if block_match:
            cleaned = self._clean_commands(block_match.group(1))
            if cleaned:
                return cleaned

        # 2️⃣ JSON‑объект
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "commands" in data:
                    lines = []
                    for entry in data["commands"]:
                        cmd = str(entry.get("command", "")).upper()
                        val = entry.get("value")
                        if cmd and val is not None:
                            lines.append(f"{cmd} {val}")
                    return "\n".join(lines)
            except Exception:
                pass  # если JSON не распарсился – переходим к обычному тексту

        # 3️⃣ Обычный текст
        return self._clean_commands(response)

    # ------------------------------------------------------------------
    # Внутренний «чистильщик» – оставляем только нужные строки
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_commands(text: str) -> str:
        """
        Оставляет только строки вида:
            FORWARD 120
            BACKWARD 30
            ROTATE -90
            WAIT 0.5
        """
        allowed = ("FORWARD", "BACKWARD", "ROTATE", "WAIT")
        lines: List[str] = []

        for raw in text.splitlines():
            line = raw.strip().strip('`').strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            cmd, val = parts[0].upper(), parts[1].strip()
            if cmd not in allowed:
                continue
            # Оставляем только числовые значения (целые или float, могут быть со знаком)
            if re.fullmatch(r'-?\d+(\.\d+)?', val):
                lines.append(f"{cmd} {val}")

        return "\n".join(lines)
