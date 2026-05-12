import numpy as np
import pandas as pd
import os
import json

from inversion_engine_v5.research.core.population import create_population
from inversion_engine_v5.research.simulation.engine import run_simulation
from inversion_engine_v5.research.portfolio.exporter import export_agents
from inversion_engine_v5.shared.indicators.pipeline import build_features

# -------------------------
# LOAD DATA
# -------------------------
def load_data(path="inversion_engine_v5/research/data/raw/xauusd_5m.csv"):
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])

    # Extract features for simulation
    open_ = df["open"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)

    # For GMT Friday close
    dayofweek = df['time'].dt.dayofweek.values.astype(np.int8)
    hour = df['time'].dt.hour.values.astype(np.int8)

    # For monthly splitting
    df['year_month'] = df['time'].dt.to_period('M')
    groups = df.groupby('year_month').groups

    # Sort groups by date
    sorted_months = sorted(groups.keys())

    month_indices = [groups[m] for m in sorted_months]

    return open_, high, low, close, dayofweek, hour, month_indices, sorted_months

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
    print("Loading data...")
    open_, high, low, close, dow, hour, month_indices, month_labels = load_data()

    print("Building features...")
    features, atr = build_features(high, low, close)

    # WARMUP
    warmup = 50
    # Trim everything
    features = features[warmup:]
    atr = atr[warmup:]
    open_ = open_[warmup:]
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]
    dow = dow[warmup:]
    hour = hour[warmup:]

    # Adjust month indices
    new_month_indices = []
    for idx_list in month_indices:
        # Subtract warmup from indices and filter out those < 0
        adjusted = [i - warmup for i in idx_list if i >= warmup]
        if adjusted:
            new_month_indices.append(adjusted)

    print(f"Data after warmup: {len(close)} candles, {len(new_month_indices)} months")

    # POPULATION
    print("Creating population (100,000 agents)...")
    pop = create_population(
        n_agents=100_000,
        input_size=features.shape[1]
    )

    survivors = np.arange(len(pop["rrr"]))
    monthly_profits = [] # To store profits per month for survivors

    # STAGE LOOP (24 rounds/months)
    n_rounds = min(24, len(new_month_indices))

    for i in range(n_rounds):
        idx = new_month_indices[i]
        label = month_labels[i]

        print(f"\n=== ROUND {i+1} ({label}) ===", flush=True)

        f = features[idx]
        o = open_[idx]
        h = high[idx]
        l = low[idx]
        c = close[idx]
        a = atr[idx]
        d = dow[idx]
        hr = hour[idx]

        pop_stage = subset_population(pop, survivors)

        stats = run_simulation(pop_stage, f, o, h, l, c, a, d, hr)

        # Survivors are those profitable in this round (equity > 1.0)
        # Note: stats['equity'] in my engine starts at 1.0 and adds PnL (non-compounding)
        mask = stats["equity"] > 1.0

        # Log some stats for current survivors
        if len(survivors) > 0:
            print(f"Survivors before: {len(survivors)}", flush=True)
            survivors = survivors[mask]
            print(f"Survivors after: {len(survivors)}", flush=True)

        if len(survivors) == 0:
            print("No survivors left!")
            break

        if len(survivors) < 10:
            print("Very few survivors left, continuing anyway...")

    # FINAL RUN on full history for remaining survivors
    if len(survivors) > 0:
        print(f"\nRunning final evaluation on full dataset for {len(survivors)} survivors...")
        final_pop = subset_population(pop, survivors)
        final_stats = run_simulation(final_pop, features, open_, high, low, close, atr, dow, hour)

        # SAVE RESULTS
        os.makedirs("outputs", exist_ok=True)

        metrics = {
            "final_equity": final_stats["equity"].tolist(),
            "winrate": final_stats["winrate"].tolist(),
            "trades": final_stats["trades"].tolist(),
            "max_drawdown": final_stats["max_drawdown"].tolist(),
            "sharpe": final_stats["sharpe"].tolist(),
            "max_losing_streak": final_stats["max_losing_streak"].tolist()
        }

        with open("outputs/survivors_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        df_metrics = pd.DataFrame({
            "agent_idx": survivors,
            "final_equity": final_stats["equity"],
            "winrate": final_stats["winrate"],
            "trades": final_stats["trades"],
            "max_drawdown": final_stats["max_drawdown"],
            "sharpe": final_stats["sharpe"],
            "max_losing_streak": final_stats["max_losing_streak"]
        })
        df_metrics = df_metrics.sort_values("final_equity", ascending=False)
        df_metrics.to_csv("outputs/top_agents.csv", index=False)

        print(f"Saved results to outputs/survivors_metrics.json and outputs/top_agents.csv")

        # EXPORT MODELS
        export_path = export_agents(pop, survivors)
        print(f"Exported models to {export_path}")
    else:
        print("No survivors to export.")

    print("\nDONE ✅")

if __name__ == "__main__":
    main()
