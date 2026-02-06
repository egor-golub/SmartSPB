# ollama_client.py
"""
Клиент для работы с Ollama API.
"""
import requests
import json
import logging
from typing import Dict, List, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)

        self.timeout = 60          # секунд
        self.max_retries = 3

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Ошибка подключения к Ollama: {e}")
            return False

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            self.logger.error(f"Ошибка получения списка моделей: {e}")
            return []

    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 1000) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "options": {"temperature": temperature,
                        "num_predict": max_tokens},
            "stream": False
        }
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "")
                else:
                    self.logger.error(
                        f"Ollama error (attempt {attempt+1}): {resp.status_code}")
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout (attempt {attempt+1})")
                if attempt == self.max_retries - 1:
                    return "Ошибка: таймаут при обращении к ИИ"
            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Network error (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    return f"Ошибка сети: {str(e)}"
        return "Ошибка: не удалось получить ответ от Ollama"

    def generate_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict]:
        try:
            json_prompt = f"{prompt}\n\nВерни ответ ТОЛЬКО в формате JSON."
            response_text = self.generate(
                json_prompt, system_prompt, temperature=0.1)
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response_text)
        except Exception as e:
            self.logger.error(f"JSON‑парсинг не удался: {e}")
            return None

    def test_model(self) -> bool:
        try:
            test_prompt = "Ответь словом 'работает'"
            resp = self.generate(test_prompt, max_tokens=10)
            return "работает" in resp.lower()
        except Exception:
            return False
