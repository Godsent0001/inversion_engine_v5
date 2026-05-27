import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

from research.core.population import create_population
from research.simulation.engine import run_simulation
from shared.indicators.pipeline import build_features
from filter.population import create_filter_population
from filter.engine import run_agent_and_filter_sim

def load_data(path="inversion_engine_v5/research/data/raw/xauusd_5m.csv"):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df["is_friday_evening"] = (df["time"].dt.weekday == 4) & (df["time"].dt.hour >= 19)
    return df

def split_into_months(features, open_, high, low, close, atr, is_friday_evening, timestamps):
    df = pd.DataFrame({"time": timestamps})
    df["month_idx"] = df["time"].dt.year * 12 + df["time"].dt.month
    unique_months = sorted(df["month_idx"].unique())
    rounds = []
    for m_idx in unique_months[:24]:
        mask = (df["month_idx"] == m_idx).values
        rounds.append((
            features[mask], open_[mask], high[mask], low[mask], close[mask],
            atr[mask], is_friday_evening[mask], m_idx
        ))
    return rounds

def subset_filter_population(pop, indices):
    new_pop = {}
    for k, v in pop.items():
        if isinstance(v, np.ndarray):
            new_pop[k] = v[indices]
        else:
            new_pop[k] = v
    return new_pop

def main():
    output_dir = "inversion_engine_v5/filter_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading Agent 91036 params...")
    agent_params = np.load("agent_91036_params.npy", allow_pickle=True).item()

    print("Loading 5m data...")
    df = load_data()
    open_ = df["open"].values.astype(np.float32)
    high = df["high"].values.astype(np.float32)
    low = df["low"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    is_friday_evening = df["is_friday_evening"].values.astype(bool)
    timestamps = df["time"]

    print("Building features...")
    features, atr = build_features(high, low, close)

    warmup = 50
    features = features[warmup:]
    atr = atr[warmup:]
    open_ = open_[warmup:]
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]
    is_friday_evening = is_friday_evening[warmup:]
    timestamps = timestamps[warmup:]

    filter_input_size = features.shape[1] + 2
    print(f"Creating filter population (100,000 filters), input size: {filter_input_size}...")
    filter_pop = create_filter_population(
        n_filters=100_000,
        input_size=filter_input_size
    )

    print("Splitting into 24 monthly rounds...")
    rounds = split_into_months(features, open_, high, low, close, atr, is_friday_evening, timestamps)

    survivors_indices = np.arange(100_000)

    for i, round_data in enumerate(rounds):
        f, o, h, l, c, a, ife, m_idx = round_data
        print(f"\n=== ROUND {i+1} (Month Index: {m_idx}) ===", flush=True)

        pop_round = subset_filter_population(filter_pop, survivors_indices)

        equity, trades, wins, max_losing_streak, sharpe = run_agent_and_filter_sim(
            f, o, h, l, c, a, ife,
            agent_params["w1"], agent_params["b1"], agent_params["w2"], agent_params["b2"],
            agent_params["rrr"], agent_params["atr"], agent_params["threshold"],
            agent_params["cooldown"], agent_params["aggression"],
            pop_round["w1"], pop_round["b1"], pop_round["w2"], pop_round["b2"],
            pop_round["threshold"]
        )

        mask = equity > 1.0
        survivors_indices = survivors_indices[mask]

        print(f"Survivors: {len(survivors_indices)}", flush=True)
        if len(survivors_indices) == 0:
            print("No filters survived.")
            break

    if len(survivors_indices) > 0:
        print("\nRunning final evaluation on full dataset for survivors...")
        final_pop = subset_filter_population(filter_pop, survivors_indices)

        eq, tr, wi, mls, sha = run_agent_and_filter_sim(
            features, open_, high, low, close, atr, is_friday_evening,
            agent_params["w1"], agent_params["b1"], agent_params["w2"], agent_params["b2"],
            agent_params["rrr"], agent_params["atr"], agent_params["threshold"],
            agent_params["cooldown"], agent_params["aggression"],
            final_pop["w1"], final_pop["b1"], final_pop["w2"], final_pop["b2"],
            final_pop["threshold"]
        )

        print("Calculating Average Monthly Losing Streak...")
        monthly_streaks = np.zeros((len(survivors_indices), len(rounds)), dtype=np.int32)
        for i, round_data in enumerate(rounds):
            f, o, h, l, c, a, ife, _ = round_data
            pop_subset = subset_filter_population(final_pop, np.arange(len(survivors_indices)))
            _, _, _, round_mls, _ = run_agent_and_filter_sim(
                f, o, h, l, c, a, ife,
                agent_params["w1"], agent_params["b1"], agent_params["w2"], agent_params["b2"],
                agent_params["rrr"], agent_params["atr"], agent_params["threshold"],
                agent_params["cooldown"], agent_params["aggression"],
                pop_subset["w1"], pop_subset["b1"], pop_subset["w2"], pop_subset["b2"],
                pop_subset["threshold"]
            )
            monthly_streaks[:, i] = round_mls

        avg_monthly_losing_streak = np.mean(monthly_streaks, axis=1)

        metrics = {
            "final_equity": eq,
            "trades": tr,
            "max_losing_streak": mls,
            "avg_monthly_losing_streak": avg_monthly_losing_streak,
            "sharpe": sha
        }

        print("Saving results...")
        np.save(os.path.join(output_dir, "survivors.npy"), {
            "metrics": metrics,
            "population": final_pop
        })

        df_metrics = pd.DataFrame({
            "filter_idx": survivors_indices,
            "final_equity": eq,
            "trades": tr,
            "max_losing_streak": mls,
            "avg_monthly_losing_streak": avg_monthly_losing_streak,
            "sharpe": sha
        })
        base_equity = 35.489674
        base_trades = 15861

        filtered_df = df_metrics[(df_metrics["final_equity"] > base_equity) & (df_metrics["trades"] < base_trades)]
        if len(filtered_df) == 0:
            print("No filters beat the user's base metrics. Saving best filters relative to actual base.")
            filtered_df = df_metrics.sort_values("final_equity", ascending=False).head(100)

        filtered_df.to_csv(os.path.join(output_dir, "top_filters_metrics.csv"), index=False)
        print(f"Saved top filters to {os.path.join(output_dir, 'top_filters_metrics.csv')}")

    else:
        print("No filters survived.")

if __name__ == "__main__":
    main()
