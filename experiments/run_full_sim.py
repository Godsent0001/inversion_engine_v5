import numpy as np
import pandas as pd

from core.population import create_population
from simulation.engine import run_simulation
from portfolio.exporter import export_agents
from core.metrics import compute_metrics

from indicators.pipeline import build_features


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path="data/raw/xauusd_30m.csv"):

    df = pd.read_csv(path)

    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)

    return high, low, close


# -------------------------
# SPLIT INTO 4 STAGES
# -------------------------
def split_stages(features, high, low, close, atr):

    n = len(close)
    step = n // 4

    stages = []

    for i in range(4):
        start = i * step
        end = (i + 1) * step if i < 3 else n

        stages.append((
            features[start:end],
            high[start:end],
            low[start:end],
            close[start:end],
            atr[start:end]
        ))

    return stages


# -------------------------
# STAGE FILTER (UPGRADED)
# -------------------------
def stage_filter(stats):
    """
    Strong survival filter
    """

    equity = stats["equity"]
    trades = stats["trades"]
    winrate = stats.get("winrate", np.zeros_like(equity))
    max_dd = stats.get("max_drawdown", np.ones_like(equity))

    cond1 = equity > 1.05         # profitable
    cond2 = trades > 20           # enough trades
    cond3 = winrate > 0.3         # not random
    cond4 = max_dd <= 0.20        # <= 20% drawdown

    return cond1 & cond2 & cond3 & cond4


# -------------------------
# SAFE SUBSET (IMPORTANT)
# -------------------------
def subset_population(pop, indices):
    """
    Keeps both numpy arrays and non-array fields like 'family'
    """

    new_pop = {}

    for k, v in pop.items():

        if isinstance(v, np.ndarray):
            new_pop[k] = v[indices]

        elif isinstance(v, list):
            new_pop[k] = [v[i] for i in indices]

        else:
            # keep scalar/global config
            new_pop[k] = v

    return new_pop


# -------------------------
# MAIN
# -------------------------
def main():

    print("Loading data...")
    high, low, close = load_data()

    # -------------------------
    # FEATURES
    # -------------------------
    print("Building features (pipeline)...")
    features, atr = build_features(high, low, close)

    # -------------------------
    # WARMUP TRIM
    # -------------------------
    warmup = 50

    features = features[warmup:]
    atr = atr[warmup:]
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]

    print(f"Data after warmup: {len(close)} candles")

    # -------------------------
    # POPULATION
    # -------------------------
    print("Creating population...")
    pop = create_population(
        n_agents=100_000,
        input_size=features.shape[1]
    )

    # -------------------------
    # STAGES
    # -------------------------
    print("Splitting into 4 stages...")
    stages = split_stages(features, high, low, close, atr)

    survivors = np.arange(len(pop["rrr"]))

    # -------------------------
    # STAGE LOOP
    # -------------------------
    for i, stage in enumerate(stages):

        print(f"\n=== STAGE {i+1} ===")

        f, h, l, c, a = stage

        pop_stage = subset_population(pop, survivors)

        stats = run_simulation(pop_stage, f, h, l, c, a)

        mask = stage_filter(stats)
        survivors = survivors[mask]

        print(f"Survivors: {len(survivors)}")

        # Diagnostics
        if len(survivors) > 0:
            print(f"Avg equity: {stats['equity'][mask].mean():.3f}")
            print(f"Avg trades: {stats['trades'][mask].mean():.1f}")

        # Safety stop
        if len(survivors) < 20:
            print("Too few survivors, stopping early.")
            break

    # -------------------------
    # FINAL RUN
    # -------------------------
    print("\nRunning final evaluation on full dataset...")

    final_pop = subset_population(pop, survivors)

    final_stats = run_simulation(final_pop, features, high, low, close, atr)

    # -------------------------
    # METRICS
    # -------------------------
    print("Computing metrics...")

    equity_curve = np.tile(final_stats["equity"][:, None], (1, 50))
    trade_pnl = np.zeros((len(survivors), 50))

    metrics = compute_metrics(equity_curve, trade_pnl)

    # -------------------------
    # SAVE
    # -------------------------
    print("Saving results...")

    np.save("outputs/survivors.npy", {
        "metrics": metrics,
        "population": final_pop,
        "stats": final_stats
    })

    # -------------------------
    # EXPORT
    # -------------------------
    export_path = export_agents(pop, survivors)

    print(f"Exported models → {export_path}")

    print("\nDONE ✅")


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    main()