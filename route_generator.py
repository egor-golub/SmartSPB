# route_generator.py
"""
Утилита для преобразования текстовых запросов в маршруты.
Может использоваться как самостоятельный скрипт.
"""

import json
import re
import sys
from typing import List, Tuple
from ollama_client import OllamaClient

class RouteGenerator:
    """Преобразует текстовые описания в маршруты (списки точек)"""
    
    # Словарь известных точек
    KNOWN_POINTS = {
        "кухня": (12, 12),
        "бар": (50, 6),
        "стойка": (50, 6),
        "стол 1": (35, 27),
        "стол1": (35, 27),
        "стол_1": (35, 27),
        "стол 2": (55, 27),
        "стол2": (55, 27),
        "стол_2": (55, 27),
        "стол 3": (35, 54),
        "стол3": (35, 54),
        "стол_3": (35, 54),
        "стол 4": (55, 54),
        "стол4": (55, 54),
        "стол_4": (55, 54),
        "колонна": (70, 70),
        "столб": (70, 70),
        "опора": (70, 70),
        "хранилище": (80, 20),
        "склад": (80, 20),
        "сад": (10, 70),
        "огород": (10, 70),
        "начало": (50, 85),
        "старт": (50, 85),
        "центр": (50, 50),
    }
    
    def __init__(self, ollama_model: str = "llama2"):
        self.ollama = OllamaClient(model=ollama_model)
        self.use_ai = ollama_model and ollama_model != "disabled"
    
    def generate_route(self, description: str) -> List[Tuple[float, float]]:
        """
        Генерирует маршрут из текстового описания.
        Сначала пробует извлечь известные точки, затем использует AI.
        """
        points = []
        
        # Шаг 1: Попробуем извлечь известные точки из текста
        description_lower = description.lower()
        for name, coords in self.KNOWN_POINTS.items():
            if name in description_lower:
                points.append(coords)
        
        # Шаг 2: Попробуем извлечь координаты вида "x y" или "x, y"
        coord_patterns = [
            r'(\d+(?:\.\d+)?)\s*[,;]\s*(\d+(?:\.\d+)?)',  # x, y
            r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',         # x y
            r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]',  # [x, y]
            r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)',  # (x, y)
        ]
        
        for pattern in coord_patterns:
            matches = re.findall(pattern, description)
            for match in matches:
                try:
                    x = float(match[0])
                    y = float(match[1])
                    if 0 <= x <= 100 and 0 <= y <= 100:
                        points.append((x, y))
                except (ValueError, IndexError):
                    continue
        
        # Удаляем дубликаты, сохраняя порядок
        seen = set()
        unique_points = []
        for point in points:
            if point not in seen:
                seen.add(point)
                unique_points.append(point)
        
        # Если нашли точки, возвращаем их
        if unique_points:
            return unique_points
        
        # Шаг 3: Используем AI, если доступно
        if self.use_ai and self.ollama.check_connection():
            return self._generate_with_ai(description)
        
        # Шаг 4: Альтернативный метод - простой парсинг
        return self._simple_parse(description)
    
    def _generate_with_ai(self, description: str) -> List[Tuple[float, float]]:
        """Использует AI для генерации маршрута"""
        system_prompt = """Ты - ассистент для генерации маршрутов робота.
Ресторан: карта 100x100 условных единиц.
Известные точки: КУХНЯ(12,12), БАР(50,6), СТОЛ_1(35,27), СТОЛ_2(55,27), 
СТОЛ_3(35,54), СТОЛ_4(55,54), КОЛОННА(70,70).

Верни только JSON в формате: {"route": [[x1,y1], [x2,y2], ...]}
Только JSON, без пояснений. Координаты от 0 до 100."""
        
        response = self.ollama.generate(
            prompt=f"Маршрут: {description}",
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=500
        )
        
        # Парсим JSON из ответа
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "route" in data:
                    points = [(float(p[0]), float(p[1])) for p in data["route"]]
                    return points
        except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
            print(f"Ошибка парсинга AI ответа: {e}")
        
        return []
    
    def _simple_parse(self, description: str) -> List[Tuple[float, float]]:
        """Простой парсинг без AI"""
        # Определяем ключевые слова и их порядок
        keywords = {
            "кухня": (12, 12),
            "бар": (50, 6),
            "стол1": (35, 27),
            "стол2": (55, 27),
            "стол3": (35, 54),
            "стол4": (55, 54),
            "стол 1": (35, 27),
            "стол 2": (55, 27),
            "стол 3": (35, 54),
            "стол 4": (55, 54),
        }
        
        points = []
        desc_lower = description.lower()
        
        # Ищем последовательности типа "от X к Y"
        patterns = [
            r'от\s+(\w+)\s+к\s+(\w+)',
            r'из\s+(\w+)\s+в\s+(\w+)',
            r'сначала\s+(\w+)\s+потом\s+(\w+)',
            r'(\w+)\s+затем\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, desc_lower)
            for match in matches:
                for word in match:
                    if word in keywords:
                        points.append(keywords[word])
        
        return points
    
    def save_route(self, points: List[Tuple[float, float]], filename: str = "route.json"):
        """Сохраняет маршрут в JSON файл"""
        route_data = {
            "route": points,
            "description": "Сгенерированный маршрут",
            "points_count": len(points)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(route_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Маршрут сохранён в {filename}")
        print(f"📍 Точек: {len(points)}")
        for i, (x, y) in enumerate(points, 1):
            print(f"  {i}. ({x}, {y})")

def main():
    """CLI интерфейс для генерации маршрутов"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Генератор маршрутов для робота")
    parser.add_argument("description", nargs="?", help="Описание маршрута")
    parser.add_argument("--file", "-f", help="Файл с описанием маршрута")
    parser.add_argument("--output", "-o", default="route.json", help="Выходной файл")
    parser.add_argument("--no-ai", action="store_true", help="Не использовать AI")
    
    args = parser.parse_args()
    
    # Получаем описание
    description = ""
    if args.description:
        description = args.description
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            description = f.read().strip()
    else:
        description = input("Введите описание маршрута: ").strip()
    
    if not description:
        print("❌ Не указано описание маршрута")
        sys.exit(1)
    
    # Генерируем маршрут
    ollama_model = "disabled" if args.no_ai else "llama2"
    generator = RouteGenerator(ollama_model=ollama_model)
    
    print(f"🔄 Генерация маршрута: {description[:50]}...")
    route = generator.generate_route(description)
    
    if route:
        generator.save_route(route, args.output)
    else:
        print("❌ Не удалось сгенерировать маршрут")

if __name__ == "__main__":
    main()