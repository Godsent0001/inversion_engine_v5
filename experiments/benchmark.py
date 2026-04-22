import time
import numpy as np

from core.population import create_population
from simulation.engine import run_simulation


def benchmark(features, high, low, close, atr, n_agents=100_000):
    print("\n=== BENCHMARK START ===")

    start_total = time.time()

    # -------------------------
    # CREATE POPULATION
    # -------------------------
    t0 = time.time()
    pop = create_population(
        n_agents=n_agents,
        input_size=features.shape[1]
    )
    t1 = time.time()

    print(f"Population creation: {t1 - t0:.2f}s")

    # -------------------------
    # SIMULATION
    # -------------------------
    t2 = time.time()

    stats = run_simulation(pop, features, high, low, close, atr)

    t3 = time.time()

    sim_time = t3 - t2

    print(f"Simulation time: {sim_time:.2f}s")

    # -------------------------
    # THROUGHPUT
    # -------------------------
    n_candles = len(close)
    total_ops = n_agents * n_candles

    print(f"Agents: {n_agents}")
    print(f"Candles: {n_candles}")
    print(f"Total evaluations: {total_ops:,}")

    print(f"Speed: {total_ops / sim_time:,.0f} ops/sec")

    total_time = time.time() - start_total

    print(f"Total time: {total_time:.2f}s")
    print("=== BENCHMARK END ===\n")

    return stats