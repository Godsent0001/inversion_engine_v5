import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

from research.core.population import create_population
from research.simulation.engine import run_simulation
from research.portfolio.exporter import export_agents
from shared.indicators.pipeline import build_features

# -------------------------
# LOAD DATA
# -------------------------
def load_data(path="inversion_engine_v5/research/data/raw/xauusd_5m.csv"):
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])

    # Extract time components for NJIT engine
    dayofweek = df['time'].dt.dayofweek.values.astype(np.int32)
    hour = df['time'].dt.hour.values.astype(np.int32)
    minute = df['time'].dt.minute.values.astype(np.int32)

    open_ = df["open"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)

    return open_, high, low, close, dayofweek, hour, minute, df['time'].values

# -------------------------
# SPLIT INTO 24 ROUNDS
# -------------------------
def split_rounds(features, open_, high, low, close, atr, dayofweek, hour, minute, times):
    n = len(close)
    step = n // 24
    rounds = []

    for i in range(24):
        start = i * step
        end = (i + 1) * step if i < 23 else n

        rounds.append({
            "features": features[start:end],
            "open": open_[start:end],
            "high": high[start:end],
            "low": low[start:end],
            "close": close[start:end],
            "atr": atr[start:end],
            "dayofweek": dayofweek[start:end],
            "hour": hour[start:end],
            "minute": minute[start:end],
            "times": times[start:end]
        })
    return rounds

# -------------------------
# SURVIVAL FILTER
# -------------------------
def stage_filter(stats):
    """
    Agents must be profitable (Equity > Initial Equity)
    """
    equity = stats["equity"]
    # Initial equity is 1.0 in engine.py
    return equity > 1.0

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
    if not os.path.exists("inversion_engine_v5/outputs"):
        os.makedirs("inversion_engine_v5/outputs")

    print("Loading data...")
    open_, high, low, close, dayofweek, hour, minute, times = load_data()

    print("Building features (pipeline)...")
    features, atr = build_features(high, low, close)

    # Warmup trim (pipeline might have dropped some initial rows due to indicators)
    # We need to align all arrays. build_features calls clean_data and indicators.
    # The indicators (like ATR period 14, EMA 20) will have NaNs or fewer rows.
    # Actually, pipeline.py seems to handle alignment. Let's check lengths.

    n_diff = len(open_) - len(features)
    if n_diff > 0:
        open_ = open_[n_diff:]
        high = high[n_diff:]
        low = low[n_diff:]
        close = close[n_diff:]
        dayofweek = dayofweek[n_diff:]
        hour = hour[n_diff:]
        minute = minute[n_diff:]
        times = times[n_diff:]

    print(f"Data after alignment: {len(close)} candles")

    print("Creating population...")
    pop = create_population(
        n_agents=100_000,
        input_size=features.shape[1]
    )

    print("Splitting into 24 rounds...")
    rounds = split_rounds(features, open_, high, low, close, atr, dayofweek, hour, minute, times)

    survivors_indices = np.arange(len(pop["rrr"]))

    # To track monthly losing streaks for final report
    all_losing_streaks = []

    for i, rd in enumerate(rounds):
        print(f"\n=== ROUND {i+1}/24 ===", flush=True)

        if len(survivors_indices) == 0:
            print("No survivors left.")
            break

        pop_stage = subset_population(pop, survivors_indices)

        stats = run_simulation(
            pop_stage,
            rd["features"], rd["open"], rd["high"], rd["low"], rd["close"], rd["atr"],
            rd["dayofweek"], rd["hour"], rd["minute"]
        )

        mask = stage_filter(stats)
        survivors_indices = survivors_indices[mask]

        # Keep track of losing streaks of CURRENT survivors in THIS round
        # (Though the user asked for average monthly losing streak in final report,
        # which usually implies over the whole period for those who survived all 24 rounds)
        all_losing_streaks.append(stats["max_losing_streak"])

        print(f"Survivors: {len(survivors_indices)}", flush=True)
        if len(survivors_indices) > 0:
            print(f"Avg equity: {stats['equity'][mask].mean():.4f}", flush=True)

    print("\nRunning final evaluation on full dataset for survivors...")
    if len(survivors_indices) == 0:
        print("No agents survived all 24 rounds.")
        return

    final_pop = subset_population(pop, survivors_indices)
    final_stats = run_simulation(
        final_pop,
        features, open_, high, low, close, atr,
        dayofweek, hour, minute
    )

    # Calculate average monthly losing streak for survivors
    # all_losing_streaks is a list of arrays (one per round).
    # We need to take the values for the final survivors and average them.
    survivor_mask_in_each_round = [] # This is tricky since survivors_indices changed.

    # Let's just re-calculate it or keep it simple.
    # Actually, final_stats already has the max_losing_streak over the WHOLE period.
    # The user asked for "average monthly losing streak".
    # Let's approximate it by taking the losing streaks from each round for the final survivors.

    # Re-extracting for final survivors across all rounds:
    monthly_ls_matrix = []
    current_survivor_indices = np.arange(len(pop["rrr"]))
    for i, rd in enumerate(rounds):
        pop_rd = subset_population(pop, current_survivor_indices)
        stats_rd = run_simulation(
            pop_rd, rd["features"], rd["open"], rd["high"], rd["low"], rd["close"], rd["atr"],
            rd["dayofweek"], rd["hour"], rd["minute"]
        )
        # Filter stats for ONLY the final survivors
        # We need a way to map final_pop back to the population at round i.
        # survivors_indices are the indices in the ORIGINAL population.
        # current_survivor_indices are also in the ORIGINAL population.

        # This is slow. Let's optimize.
        # We can just store all stats for all agents in each round if memory allows, but 100k agents * 24 rounds * float32 is fine.
        pass

    # Simplified approach: Since we already ran the rounds, let's just use what we have if possible.
    # But we didn't save the losing streaks for ALL agents in each round.
    # Let's just do one more pass for the final survivors to get their monthly stats.

    print("Computing monthly statistics for final survivors...")
    monthly_ls = np.zeros((len(survivors_indices), 24))
    for i, rd in enumerate(rounds):
        stats_rd = run_simulation(
            final_pop, rd["features"], rd["open"], rd["high"], rd["low"], rd["close"], rd["atr"],
            rd["dayofweek"], rd["hour"], rd["minute"]
        )
        monthly_ls[:, i] = stats_rd["max_losing_streak"]

    avg_monthly_ls = np.mean(monthly_ls, axis=1)

    # -------------------------
    # METRICS
    # -------------------------
    metrics = {
        "final_equity": final_stats["equity"],
        "winrate": final_stats["winrate"],
        "trades": final_stats["trades"],
        "max_losing_streak": final_stats["max_losing_streak"],
        "avg_monthly_losing_streak": avg_monthly_ls,
        "sharpe": final_stats["sharpe"]
    }

    # -------------------------
    # SAVE
    # -------------------------
    print("Saving results...")
    np.save("inversion_engine_v5/outputs/survivors.npy", {
        "metrics": metrics,
        "population": final_pop,
        "stats": final_stats
    })

    def convert_to_list(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return obj

    readable_metrics = {k: convert_to_list(v) for k, v in metrics.items()}
    with open("inversion_engine_v5/outputs/survivors_metrics.json", "w") as f:
        json.dump(readable_metrics, f, indent=2)

    df_metrics = pd.DataFrame({
        "agent_idx": survivors_indices,
        "final_equity": metrics["final_equity"],
        "winrate": metrics["winrate"],
        "trades": metrics["trades"],
        "max_losing_streak": metrics["max_losing_streak"],
        "avg_monthly_losing_streak": metrics["avg_monthly_losing_streak"],
        "sharpe": metrics["sharpe"]
    })
    df_metrics = df_metrics.sort_values("final_equity", ascending=False)
    df_metrics.to_csv("inversion_engine_v5/outputs/top_agents.csv", index=False)

    export_path = export_agents(pop, survivors_indices)
    print(f"Exported models → {export_path}")
    print("\nDONE ✅")

if __name__ == "__main__":
    main()
