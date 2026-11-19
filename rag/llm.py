import os
import threading
import time

import requests
from dotenv import load_dotenv

from rag.limits import RateLimiter, retry_pause

load_dotenv()

URL = "https://claude.gigaparsers.ru/v1/messages"
MODEL = "deepseek-flash"
ATTEMPTS = 12

# лимит ключа 10 запросов в минуту, судья в eval ходит через этот же лимитер
limiter = RateLimiter(per_minute=float(os.environ.get("LLM_RPM", 9.5)), burst=2)
_local = threading.local()


def _session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def post(payload):
    # отправить запрос с учётом лимита и повторами, вернуть json ответа
    headers = {"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"}
    error = None
    for attempt in range(ATTEMPTS):
        limiter.wait()
        try:
            response = _session().post(URL, headers=headers, json=payload, timeout=(10, 120))
        except requests.RequestException as exc:
            error = exc
            time.sleep(retry_pause(attempt))
            continue
        if response.status_code == 200:
            return response.json()
        error = RuntimeError(f"{response.status_code}: {response.text[:200]}")
        if response.status_code == 429:
            limiter.back_off(retry_pause(attempt + 2, response.headers.get("Retry-After")))
        elif response.status_code >= 500:
            time.sleep(retry_pause(attempt))
        else:
            raise error
    raise RuntimeError(f"модель не ответила за {ATTEMPTS} попыток: {error}")


def chat(system, messages, max_tokens=1200):
    data = post({
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "system": system,
        "messages": messages,
    })
    text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
    return text.strip(), data.get("usage", {})
