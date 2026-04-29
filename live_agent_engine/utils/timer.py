import time


# -------------------------
# SIMPLE TIMER
# -------------------------
class Timer:
    def __init__(self, name="TIMER"):
        self.name = name
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer was not started")

        elapsed = time.time() - self.start_time
        print(f"[{self.name}] Elapsed: {elapsed:.2f}s")
        self.start_time = None
        return elapsed


# -------------------------
# CONTEXT MANAGER TIMER
# -------------------------
class TimerBlock:
    def __init__(self, name="BLOCK"):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"[{self.name}] Took {elapsed:.2f}s")


# -------------------------
# DECORATOR TIMER
# -------------------------
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        print(f"[{func.__name__}] {elapsed:.2f}s")

        return result

    return wrapper


# -------------------------
# STAGE TIMER (FOR YOUR PIPELINE)
# -------------------------
class StageTimer:
    def __init__(self):
        self.times = {}
        self.current = None
        self.start_time = None

    def start(self, name):
        self.current = name
        self.start_time = time.time()

    def stop(self):
        if self.current is None:
            return

        elapsed = time.time() - self.start_time

        if self.current not in self.times:
            self.times[self.current] = 0

        self.times[self.current] += elapsed

        print(f"[{self.current}] {elapsed:.2f}s")

        self.current = None

    def summary(self):
        print("\n=== TIMER SUMMARY ===")
        for k, v in self.times.items():
            print(f"{k}: {v:.2f}s")