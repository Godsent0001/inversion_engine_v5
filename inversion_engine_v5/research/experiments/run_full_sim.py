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
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])

    # Pre-calculate Friday Evening Mask (>= 19:00 Friday)
    # Friday is weekday 4.
    df["is_friday_evening"] = (df["time"].dt.weekday == 4) & (df["time"].dt.hour >= 19)

    open_ = df["open"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    is_friday_evening = df["is_friday_evening"].values.astype(bool)

    return open_, high, low, close, is_friday_evening, df["time"]

# -------------------------
# SPLIT INTO 24 ROUNDS (CALENDAR MONTHS)
# -------------------------
def split_into_months(features, open_, high, low, close, atr, is_friday_evening, timestamps):
    df = pd.DataFrame({"time": timestamps})
    df["month_idx"] = df["time"].dt.year * 12 + df["time"].dt.month
    unique_months = sorted(df["month_idx"].unique())

    rounds = []
    for m_idx in unique_months[:24]:
        mask = (df["month_idx"] == m_idx).values
        rounds.append((
            features[mask],
            open_[mask],
            high[mask],
            low[mask],
            close[mask],
            atr[mask],
            is_friday_evening[mask],
            m_idx
        ))

    return rounds

# -------------------------
# STAGE FILTER
# -------------------------
def stage_filter(stats):
    """
    Survival filter: Profitable (Equity > 1.0)
    """
    equity = stats["equity"]
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
    output_dir = "inversion_engine_v5/outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading 5m data...")
    open_, high, low, close, is_friday_evening, timestamps = load_data()

    print("Building features...")
    features, atr = build_features(high, low, close)

    # Warmup trim
    warmup = 50
    features = features[warmup:]
    atr = atr[warmup:]
    open_ = open_[warmup:]
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]
    is_friday_evening = is_friday_evening[warmup:]
    timestamps = timestamps[warmup:]

    print(f"Data after warmup: {len(close)} candles")

    print("Creating population (100,000 agents)...")
    pop = create_population(
        n_agents=100_000,
        input_size=features.shape[1]
    )

    print("Splitting into 24 monthly rounds...")
    rounds = split_into_months(features, open_, high, low, close, atr, is_friday_evening, timestamps)

    survivors_indices = np.arange(len(pop["rrr"]))

    # -------------------------
    # TOURNAMENT LOOP
    # -------------------------
    for i, round_data in enumerate(rounds):
        f, o, h, l, c, a, ife, m_idx = round_data
        print(f"\n=== ROUND {i+1} (Month Index: {m_idx}) ===", flush=True)

        pop_round = subset_population(pop, survivors_indices)
        stats = run_simulation(pop_round, f, o, h, l, c, a, ife)

        mask = stage_filter(stats)
        survivors_indices = survivors_indices[mask]

        print(f"Survivors: {len(survivors_indices)}", flush=True)

        if len(survivors_indices) == 0:
            print("No survivors left.")
            break

    # -------------------------
    # FINAL EVALUATION (Full Dataset)
    # -------------------------
    if len(survivors_indices) > 0:
        print("\nRunning final evaluation on full dataset for survivors...")
        final_pop = subset_population(pop, survivors_indices)
        final_stats = run_simulation(final_pop, features, open_, high, low, close, atr, is_friday_evening)

        print("Calculating Average Monthly Losing Streak...")
        final_agent_count = len(survivors_indices)
        monthly_streaks_matrix = np.zeros((final_agent_count, len(rounds)), dtype=np.int32)

        for i, round_data in enumerate(rounds):
            f, o, h, l, c, a, ife, _ = round_data
            round_stats = run_simulation(final_pop, f, o, h, l, c, a, ife)
            monthly_streaks_matrix[:, i] = round_stats["max_losing_streak"]

        avg_monthly_losing_streak = np.mean(monthly_streaks_matrix, axis=1)

        # -------------------------
        # METRICS
        # -------------------------
        metrics = {
            "final_equity": final_stats["equity"],
            "winrate": final_stats["winrate"],
            "trades": final_stats["trades"],
            "max_losing_streak": final_stats["max_losing_streak"],
            "avg_monthly_losing_streak": avg_monthly_losing_streak,
            "sharpe": final_stats["sharpe"]
        }

        # -------------------------
        # SAVE
        # -------------------------
        print("Saving results...")

        np.save(os.path.join(output_dir, "survivors.npy"), {
            "metrics": metrics,
            "population": final_pop,
            "stats": final_stats
        })

        def convert_to_list(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.float16)): return float(obj)
            return obj

        readable_metrics = {k: convert_to_list(v) for k, v in metrics.items()}
        with open(os.path.join(output_dir, "survivors_metrics.json"), "w") as f:
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
        df_metrics.to_csv(os.path.join(output_dir, "top_agents.csv"), index=False)

        export_path = export_agents(pop, survivors_indices, folder=output_dir)
        print(f"Exported models -> {export_path}")

    else:
        print("No agents survived the tournament.")

    print("\nDONE ✅")

if __name__ == "__main__":
    main()
