import time
from datetime import datetime


# -------------------------
# SIMPLE LOGGER
# -------------------------
class Logger:

    def __init__(self, name="SIM"):
        self.name = name
        self.start_time = time.time()

    def _now(self):
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self):
        return time.time() - self.start_time

    def info(self, msg):
        print(f"[{self._now()}][{self.name}] {msg}")

    def step(self, msg):
        print(f"[{self._now()}][{self.name}][+{self._elapsed():.1f}s] {msg}")

    def warn(self, msg):
        print(f"[{self._now()}][WARNING] {msg}")

    def error(self, msg):
        print(f"[{self._now()}][ERROR] {msg}")


# -------------------------
# TIMING DECORATOR
# -------------------------
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"[TIMER] {func.__name__} took {end - start:.2f}s")

        return result

    return wrapper


# -------------------------
# PROGRESS PRINTER
# -------------------------
def progress(current, total, step=1000):
    """
    Lightweight progress tracking (no tqdm overhead)
    """

    if current % step == 0 or current == total - 1:
        pct = (current / total) * 100
        print(f"Progress: {current}/{total} ({pct:.1f}%)")