import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import sys

# Add the inversion_engine_v5 directory to the path to import research and shared
sys.path.append(os.path.join(os.getcwd(), "inversion_engine_v5"))

from research.simulation.engine import run_simulation
from shared.indicators.pipeline import build_features

def load_test_data(path="inversion_engine_v5/research/data/raw/mt5_test_sample_5m.csv"):
    if not os.path.exists(path):
        # Try local path if absolute fails or environment differs
        path = "inversion_engine_v5/research/data/raw/mt5_test_sample_5m.csv"

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])

    # Friday Evening Mask (>= 19:00 Friday)
    df["is_friday_evening"] = (df["time"].dt.weekday == 4) & (df["time"].dt.hour >= 19)

    open_ = df["open"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    is_friday_evening = df["is_friday_evening"].values.astype(bool)

    return open_, high, low, close, is_friday_evening, df["time"]

def main():
    print("Loading test data...")
    open_, high, low, close, is_friday_evening, timestamps = load_test_data()

    print("Building features...")
    features, atr = build_features(high, low, close)

    # Warmup trim (50 candles as per research script)
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

    # Load Agent 91036 from bundle
    bundle_path = "inversion_engine_v5/outputs/export_bundle_20260513_050051.pkl"
    print(f"Loading agent 91036 from {bundle_path}...")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    # We found earlier that 91036 is at bundle index 193
    agent_idx_in_bundle = 193
    pop = {
        "w1": np.expand_dims(bundle["w1"][agent_idx_in_bundle], axis=0),
        "w2": np.expand_dims(bundle["w2"][agent_idx_in_bundle], axis=0),
        "b1": np.expand_dims(bundle["b1"][agent_idx_in_bundle], axis=0),
        "b2": np.expand_dims(bundle["b2"][agent_idx_in_bundle], axis=0),
        "rrr": np.array([bundle["rrr"][agent_idx_in_bundle]], dtype=np.float32),
        "atr": np.array([bundle["atr"][agent_idx_in_bundle]], dtype=np.float32),
        "threshold": np.array([bundle["threshold"][agent_idx_in_bundle]], dtype=np.float32),
        "cooldown": np.array([bundle["cooldown"][agent_idx_in_bundle]], dtype=np.int32),
        "aggression": np.array([bundle["aggression"][agent_idx_in_bundle]], dtype=np.float32)
    }

    print("Running full simulation...")
    stats = run_simulation(pop, features, open_, high, low, close, atr, is_friday_evening)

    # Summary Metrics
    summary_metrics = [
        ["--- SUMMARY METRICS ---", ""],
        ["Total_Trades", stats["trades"][0]],
        ["Win_Rate", stats["winrate"][0]],
        ["Final_Equity", stats["equity"][0]],
        ["Max_Losing_Streak", stats["max_losing_streak"][0]],
        ["Sharpe_Ratio", stats["sharpe"][0]],
        ["", ""]
    ]

    # Monthly Breakdown
    df = pd.DataFrame({"time": timestamps})
    df["month_idx"] = df["time"].dt.year.astype(str) + "-" + df["time"].dt.month.astype(str).str.zfill(2)
    unique_months = sorted(df["month_idx"].unique())

    monthly_breakdown_header = [
        ["--- MONTHLY BREAKDOWN ---", "", ""],
        ["Month", "Monthly_Profit", "Cumulative_Equity"]
    ]

    current_equity = 1.0
    monthly_rows = []
    print("Calculating monthly metrics...")
    for month in unique_months:
        mask = (df["month_idx"] == month).values
        m_stats = run_simulation(pop, features[mask], open_[mask], high[mask], low[mask], close[mask], atr[mask], is_friday_evening[mask])

        # Monthly Profit (relative to 1.0 starting balance each month)
        month_profit = m_stats["equity"][0] - 1.0
        current_equity += month_profit
        monthly_rows.append([month, month_profit, current_equity])

    # Combine all for CSV
    all_rows = summary_metrics + monthly_breakdown_header + monthly_rows

    results_df = pd.DataFrame(all_rows)
    output_file = "agent_91036_test_metrics.csv"
    results_df.to_csv(output_file, index=False, header=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
