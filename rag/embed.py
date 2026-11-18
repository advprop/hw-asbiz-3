import os
import threading
import time

import requests
from dotenv import load_dotenv

from rag.limits import RateLimiter, retry_pause

load_dotenv()

URL = "https://embeddings.parsers360.ru:10443/embed"
MAX_CHARS = 19000  # сервис не берёт текст длиннее 20000 символов
ATTEMPTS = 8

# у сервиса лимит 60 запросов в минуту, берём чуть меньше
limiter = RateLimiter(per_minute=float(os.environ.get("EMB_RPM", 57)), burst=5)
_local = threading.local()


def _session():
    # своя сессия на поток, requests.Session между потоками лучше не делить
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def embed(text):
    headers = {
        "Authorization": "Bearer " + os.environ["EMB_KEY"],
        "Content-Type": "text/plain; charset=utf-8",
    }
    body = text[:MAX_CHARS].encode("utf-8")
    error = None
    for attempt in range(ATTEMPTS):
        limiter.wait()
        try:
            response = _session().post(URL, headers=headers, data=body, timeout=(10, 120))
        except requests.RequestException as exc:
            error = exc
            time.sleep(retry_pause(attempt))
            continue
        if response.status_code == 200:
            vector = response.json().get("embedding")
            if isinstance(vector, list) and vector:
                return vector
            error = RuntimeError("в ответе нет эмбеддинга")
        elif response.status_code == 429:
            error = RuntimeError("429 от сервиса эмбеддингов")
            limiter.back_off(retry_pause(attempt, response.headers.get("Retry-After")))
            continue
        elif response.status_code >= 500:
            error = RuntimeError(f"{response.status_code} от сервиса эмбеддингов")
        else:
            # 4xx повторять бесполезно
            response.raise_for_status()
        time.sleep(retry_pause(attempt))
    raise RuntimeError(f"сервис эмбеддингов не ответил за {ATTEMPTS} попыток: {error}")
