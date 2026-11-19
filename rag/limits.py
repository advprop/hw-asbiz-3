import random
import threading
import time


class RateLimiter:
    # общий на все потоки: не больше per_minute запросов в минуту

    def __init__(self, per_minute, burst=1):
        self.interval = 60.0 / per_minute
        self.burst = burst
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self.blocked_until = 0.0
        self.lock = threading.Lock()

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.updated) / self.interval)
                self.updated = now
                if now >= self.blocked_until and self.tokens >= 1:
                    self.tokens -= 1
                    return
                pause = max(self.blocked_until - now, (1 - self.tokens) * self.interval)
            time.sleep(pause)

    def back_off(self, seconds):
        # сервер ответил 429: все потоки ждут, а не только тот, что получил отказ
        with self.lock:
            self.tokens = 0.0
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


def retry_pause(attempt, retry_after=None):
    if retry_after:
        try:
            return float(retry_after) + random.uniform(0, 0.5)
        except ValueError:
            pass
    return min(60.0, 2.0**attempt) + random.uniform(0, 1)
