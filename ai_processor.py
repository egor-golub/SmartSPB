# ai_processor.py
"""
Обёртка над Ollama‑моделью. Сейчас реализовано только получение списка
команд для робота в Pygame.
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class SimpleAIProcessor:
    """
    Минимальный процессор: формирует запрос к модели, получает ответ,
    который затем будет обработан CodeGenerator'ом.
    """

    def __init__(self, ollama_model: str = "llama2"):
        self.ollama_model = ollama_model
        self.llm = ChatOllama(model=ollama_model, temperature=0.1)
        logger.info(f"✅ AI‑процессор инициализирован (модель: {ollama_model})")

    # ------------------------------------------------------------------
    # Генерация списка команд робота
    # ------------------------------------------------------------------
    def get_robot_commands(self, user_request: str) -> str:
        """
        Запрашивает у модели перечень команд для робота.

        Поддерживаются четыре команды:
            FORWARD <dist>
            BACKWARD <dist>
            ROTATE <angle>
            WAIT <seconds>

        Возвращается **только** список команд (можно в markdown‑блоке,
        в JSON‑формате или простой текст – без пояснений.
        """
        system_prompt = """Ты — эксперт по управлению роботом в Pygame.
Робот понимает четыре текстовые команды (по одной на строку):

FORWARD <distance>   # движение вперёд, пиксели
BACKWARD <distance>  # движение назад
ROTATE <angle>       # вращение (положительно – по часовой)
WAIT <seconds>       # пауза

Пользователь описывает задачу на русском языке.
Твоя задача — вернуть **только** список команд, без всякого объяснения
и без дополнительного текста. Если удобно, можешь поместить их в
markdown‑блок (```). Округляй числа до целых."""
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Сгенерируй команды для робота: {user_request}")
            ])
            answer = getattr(response, "content", str(response))
            logger.info("✅ Получен ответ от модели (команды робота)")
            return answer
        except Exception as e:
            logger.error(f"❌ Ошибка при запросе к модели: {e}")
            return f"# Ошибка генерации команд: {e}"
