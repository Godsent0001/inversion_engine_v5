import numpy as np
import pandas as pd
import os
import json

from research_stop_orders.core.population import create_population
from research_stop_orders.simulation.engine import run_stop_simulation
from research_stop_orders.portfolio.exporter import export_agents
from research_stop_orders.core.metrics import compute_metrics

from shared.indicators.pipeline import build_features


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path="inversion_engine_v5/research/data/raw/xauusd_30m.csv"):
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found at {path}")

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
# STAGE FILTER
# -------------------------
def stage_filter(stats):
    equity = stats["equity"]
    trades = stats["trades"]
    winrate = stats.get("winrate", np.zeros_like(equity))

    cond1 = equity > 1.01         # slightly lower for stop orders?
    cond2 = trades > 3            # at least some trades per stage
    cond3 = winrate > 0.20

    return cond1 & cond2 & cond3


# -------------------------
# SAFE SUBSET
# -------------------------
def subset_population(pop, indices):
    new_pop = {}
    for k, v in pop.items():
        if isinstance(v, np.ndarray):
            new_pop[k] = v[indices]
        elif isinstance(v, list):
            new_pop[k] = [v[i] for i in indices]
        else:
            new_pop[k] = v
    return new_pop


# -------------------------
# MAIN
# -------------------------
def main():
    # Ensure output directories exist
    os.makedirs("inversion_engine_v5/outputs_stop_orders", exist_ok=True)
    os.makedirs("inversion_engine_v5/models_stop_orders", exist_ok=True)

    print("Loading data...")
    open_, high, low, close = load_data()

    # -------------------------
    # FEATURES
    # -------------------------
    print("Building features (pipeline)...")
    features, atr = build_features(high, low, close)

    # -------------------------
    # WARMUP TRIM (Requested start index >= 100)
    # -------------------------
    warmup = 100

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

        stats = run_stop_simulation(pop_stage, f, o, h, l, c, a)

        mask = stage_filter(stats)
        survivors = survivors[mask]

        print(f"Survivors: {len(survivors)}", flush=True)

        if len(survivors) > 0:
            print(f"Avg equity: {stats['equity'][mask].mean():.3f}", flush=True)
            print(f"Avg trades: {stats['trades'][mask].mean():.1f}", flush=True)
            print(f"Avg orders placed: {stats['orders_placed'][mask].mean():.1f}", flush=True)

        if len(survivors) < 10:
            print("Too few survivors, stopping early.")
            break

    # -------------------------
    # FINAL RUN
    # -------------------------
    if len(survivors) == 0:
        print("No survivors after all stages.")
        return

    print("\nRunning final evaluation on full dataset...")

    final_pop = subset_population(pop, survivors)
    final_stats = run_stop_simulation(final_pop, features, open_, high, low, close, atr)

    # -------------------------
    # METRICS
    # -------------------------
    print("Computing metrics...")
    metrics = {
        "final_equity": final_stats["equity"],
        "winrate": final_stats["winrate"],
        "trades": final_stats["trades"],
        "max_drawdown": final_stats["max_drawdown"],
        "sharpe": final_stats["sharpe"],
        "orders_placed": final_stats["orders_placed"],
        "orders_cancelled": final_stats["orders_cancelled"]
    }

    # -------------------------
    # SAVE
    # -------------------------
    print("Saving results...")

    np.save("inversion_engine_v5/outputs_stop_orders/survivors.npy", {
        "metrics": metrics,
        "population": final_pop,
        "stats": final_stats
    })

    def convert_to_list(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return obj

    readable_metrics = {k: convert_to_list(v) for k, v in metrics.items()}
    with open("inversion_engine_v5/outputs_stop_orders/survivors_metrics.json", "w") as f:
        json.dump(readable_metrics, f, indent=2)

    df_metrics = pd.DataFrame({
        "agent_idx": np.arange(len(metrics["final_equity"])),
        "final_equity": metrics["final_equity"],
        "winrate": metrics["winrate"],
        "trades": metrics["trades"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "orders_placed": metrics["orders_placed"],
        "orders_cancelled": metrics["orders_cancelled"]
    })
    df_metrics = df_metrics.sort_values("final_equity", ascending=False)
    df_metrics.to_csv("inversion_engine_v5/outputs_stop_orders/top_agents.csv", index=False)

    print("Saved: outputs_stop_orders/survivors.npy, survivors_metrics.json, top_agents.csv")

    # -------------------------
    # EXPORT
    # -------------------------
    if len(survivors) > 0:
        # Export ALL survivors
        export_path = export_agents(final_pop, np.arange(len(survivors)), folder="inversion_engine_v5/models_stop_orders")
        print(f"Exported all models → {export_path}")

        # Export TOP 100 survivors
        top_100_indices = df_metrics.head(100)["agent_idx"].values
        top_100_folder = "inversion_engine_v5/models_stop_orders/top_100"
        os.makedirs(top_100_folder, exist_ok=True)
        export_path_top = export_agents(final_pop, top_100_indices, folder=top_100_folder)
        print(f"Exported top 100 models → {export_path_top}")
    else:
        print("No survivors to export.")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
