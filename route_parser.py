#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Парсер, превращающий произвольный русский/английский текст
в упорядоченный список логических точек (координат 0‑100)
для планировщика пути.
"""

import re
from typing import List, Tuple

from constants import LOGICAL_GOALS

# Псевдонимы, которыми пользователь может назвать точки.
USER_ALIAS = {
    "бар":          "BAR",
    "bar":          "BAR",
    "кухня":        "KITCHEN",
    "kitchen":      "KITCHEN",
    "стол 1":       "TABLE_1",
    "стол1":        "TABLE_1",
    "стол_1":       "TABLE_1",
    "table 1":      "TABLE_1",
    "table1":       "TABLE_1",
    "стол 2":       "TABLE_2",
    "стол2":        "TABLE_2",
    "стол_2":       "TABLE_2",
    "table 2":      "TABLE_2",
    "table2":       "TABLE_2",
    "стол 3":       "TABLE_3",
    "стол3":        "TABLE_3",
    "стол_3":       "TABLE_3",
    "table 3":      "TABLE_3",
    "table3":       "TABLE_3",
    "стол 4":       "TABLE_4",
    "стол4":        "TABLE_4",
    "стол_4":       "TABLE_4",
    "table 4":      "TABLE_4",
    "table4":       "TABLE_4",
    "пиллар":       "PILLAR",
    "pillar":       "PILLAR",
    "колонна":      "PILLAR",
    "опора":        "PILLAR",
}


def parse_route_from_text(text: str) -> List[Tuple[float, float]]:
    """
    Преобразует свободный текст запроса в список логических координат.
    Поддерживается:
      • названия точек из USER_ALIAS (в любом регистре);
      • координаты вида «12 12», «12,12», «12;12», «[12, 12]».
    Возвращается список уникальных точек **в порядке появления**.
    """
    points: List[Tuple[float, float]] = []

    # 1) Именованные точки
    if USER_ALIAS:
        alias_patterns = sorted(USER_ALIAS.keys(), key=lambda x: -len(x))
        combined = r'|'.join(r'\b' + re.escape(a) + r'\b' for a in alias_patterns)
        name_regex = re.compile(combined, flags=re.IGNORECASE)

        for m in name_regex.finditer(text):
            alias = m.group(0).lower()
            goal_key = USER_ALIAS.get(alias)
            if goal_key and goal_key in LOGICAL_GOALS:
                points.append(LOGICAL_GOALS[goal_key])

    # 2) Явные координаты
    coord_regex = re.compile(r'(\d+(?:\.\d+)?)\s*[,;]?\s*(\d+(?:\.\d+)?)')
    for cx, cy in coord_regex.findall(text):
        x, y = float(cx), float(cy)
        if 0 <= x <= 100 and 0 <= y <= 100:
            points.append((x, y))

    # 3) Удаляем дубли, сохраняем порядок
    seen = set()
    unique: List[Tuple[float, float]] = []
    for pt in points:
        if pt not in seen:
            seen.add(pt)
            unique.append(pt)

    return unique
