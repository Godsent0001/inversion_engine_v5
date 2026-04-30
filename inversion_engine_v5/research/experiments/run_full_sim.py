import numpy as np
import pandas as pd

from research.core.population import create_population
from research.simulation.engine import run_simulation
from research.portfolio.exporter import export_agents
from research.core.metrics import compute_metrics

from shared.indicators.pipeline import build_features


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path="data/raw/xauusd_30m.csv"):

    df = pd.read_csv(path)

    open_ = df["open"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)

    return open_, high, low, close


# -------------------------
# SPLIT INTO 4 STAGES
# -------------------------
def split_stages(features, open_, high, low, close, atr):

    n = len(close)
    step = n // 4

    stages = []

    for i in range(4):
        start = i * step
        end = (i + 1) * step if i < 3 else n

        stages.append((
            features[start:end],
            open_[start:end],
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
    # max_dd = stats.get("max_drawdown", np.ones_like(equity))

    cond1 = equity > 1.02         # profitable (lowered for stages)
    cond2 = trades > 5            # enough trades
    cond3 = winrate > 0.25        # not random

    return cond1 & cond2 & cond3


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
    open_, high, low, close = load_data()

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
    open_ = open_[warmup:]
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
    stages = split_stages(features, open_, high, low, close, atr)

    survivors = np.arange(len(pop["rrr"]))

    # -------------------------
    # STAGE LOOP
    # -------------------------
    for i, stage in enumerate(stages):

        print(f"\n=== STAGE {i+1} ===", flush=True)

        f, o, h, l, c, a = stage

        pop_stage = subset_population(pop, survivors)

        stats = run_simulation(pop_stage, f, o, h, l, c, a)

        mask = stage_filter(stats)
        survivors = survivors[mask]

        print(f"Survivors: {len(survivors)}", flush=True)

        # Diagnostics
        if len(survivors) > 0:
            print(f"Avg equity: {stats['equity'][mask].mean():.3f}", flush=True)
            print(f"Avg trades: {stats['trades'][mask].mean():.1f}", flush=True)

        # Safety stop
        if len(survivors) < 20:
            print("Too few survivors, stopping early.")
            break

    # -------------------------
    # FINAL RUN
    # -------------------------
    print("\nRunning final evaluation on full dataset...")

    final_pop = subset_population(pop, survivors)

    final_stats = run_simulation(final_pop, features, open_, high, low, close, atr)

    # -------------------------
    # METRICS
    # -------------------------
    print("Computing metrics...")

    # We use a simplified metrics computation for now as the JIT returns final stats
    metrics = {
        "final_equity": final_stats["equity"],
        "winrate": final_stats["winrate"],
        "trades": final_stats["trades"],
        "max_drawdown": final_stats["max_drawdown"]
    }

    # -------------------------
    # SAVE
    # -------------------------
    print("Saving results...")

    # 1. Full binary data
    np.save("outputs/survivors.npy", {
        "metrics": metrics,
        "population": final_pop,
        "stats": final_stats
    })

    # 2. Human-readable metrics (JSON)
    import json
    def convert_to_list(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return obj

    readable_metrics = {k: convert_to_list(v) for k, v in metrics.items()}
    with open("outputs/survivors_metrics.json", "w") as f:
        json.dump(readable_metrics, f, indent=2)

    # 3. Top agents ranking (CSV)
    df_metrics = pd.DataFrame({
        "agent_idx": np.arange(len(metrics["final_equity"])),
        "final_equity": metrics["final_equity"],
        "winrate": metrics["winrate"],
        "trades": metrics["trades"],
        "max_drawdown": metrics["max_drawdown"]
    })
    df_metrics = df_metrics.sort_values("final_equity", ascending=False)
    df_metrics.to_csv("outputs/top_agents.csv", index=False)

    print("Saved: outputs/survivors.npy, outputs/survivors_metrics.json, outputs/top_agents.csv")

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