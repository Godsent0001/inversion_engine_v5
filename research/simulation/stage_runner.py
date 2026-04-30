import numpy as np

from research.simulation.engine import run_simulation


# -------------------------
# STAGE FILTER
# -------------------------
def stage_filter(stats):
    """
    Survival criteria per stage
    """

    equity = stats["equity"]
    trades = stats["trades"]
    winrate = stats["winrate"]

    cond1 = equity > 1.05     # must grow capital
    cond2 = trades > 20       # must actually trade
    cond3 = winrate > 0.3     # avoid randomness

    return cond1 & cond2 & cond3


# -------------------------
# SPLIT INTO STAGES
# -------------------------
def split_stages(features, high, low, close, atr, n_stages=4):

    n = len(close)
    step = n // n_stages

    stages = []

    for i in range(n_stages):
        start = i * step
        end = (i + 1) * step if i < n_stages - 1 else n

        stages.append((
            features[start:end],
            high[start:end],
            low[start:end],
            close[start:end],
            atr[start:end]
        ))

    return stages


# -------------------------
# RUN MULTI-STAGE SURVIVAL
# -------------------------
def run_stages(pop, features, high, low, close, atr):

    stages = split_stages(features, high, low, close, atr)

    survivors = np.arange(len(pop["rrr"]))

    history = []

    # -------------------------
    # STAGE LOOP
    # -------------------------
    for i, stage in enumerate(stages):

        print(f"\n=== STAGE {i+1} ===")

        f, h, l, c, a = stage

        # Subset population
        pop_stage = {
            k: v[survivors]
            for k, v in pop.items()
            if isinstance(v, np.ndarray)
        }

        stats = run_simulation(pop_stage, f, h, l, c, a)

        mask = stage_filter(stats)

        survivors = survivors[mask]

        print(f"Survivors: {len(survivors)}")

        history.append({
            "stage": i + 1,
            "survivors": len(survivors)
        })

        # Early stop
        if len(survivors) < 20:
            print("Too few survivors, stopping early.")
            break

    return survivors, history