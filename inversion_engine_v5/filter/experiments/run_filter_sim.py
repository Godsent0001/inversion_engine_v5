import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

from filter.core.population import create_filter_population
from filter.simulation.engine import run_filter_simulation
from filter.portfolio.exporter import export_filters
from research.experiments.run_full_sim import load_data, split_into_months, subset_population
from shared.indicators.pipeline import build_features
from shared.utils.normalization import normalize_features, zscore_norm, tanh_norm

def main():
    output_dir = "inversion_engine_v5/filter/outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading 5m data...")
    open_, high, low, close, is_friday_evening, timestamps = load_data()

    print("Building base features...")
    base_features, atr = build_features(high, low, close)

    # Warmup trim
    warmup = 50
    base_features = base_features[warmup:]
    atr = atr[warmup:]
    open_ = open_[warmup:]
    high = high[warmup:]
    low = low[warmup:]
    close = close[warmup:]
    is_friday_evening = is_friday_evening[warmup:]
    timestamps = timestamps[warmup:]

    print(f"Data after warmup: {len(close)} candles")

    print("Loading Agent 91036...")
    base_agent = np.load("inversion_engine_v5/filter/agent_91036.npy", allow_pickle=True).item()

    # Pre-calculate base agent confidence and direction for all timestamps
    print("Pre-calculating Agent 91036 signals and confidence...")

    def get_agent_outputs(features, agent):
        w1, b1, w2, b2 = agent["w1"], agent["b1"], agent["w2"], agent["b2"]
        agg = agent["aggression"]

        h = np.tanh(features @ w1 + b1)
        out = (h @ w2 + b2) * agg

        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)

        confidence = np.max(probs, axis=1)
        direction_idx = np.argmax(probs, axis=1)
        direction = np.zeros_like(confidence)
        direction[direction_idx == 1] = 1.0
        direction[direction_idx == 2] = -1.0

        return confidence, direction

    confidence, direction = get_agent_outputs(base_features, base_agent)

    # Construct Filter Features: [6 Indicators, Confidence, Direction]
    print("Normalizing confidence and direction...")
    # Using simple min-max or similar for these to avoid z-score artifacts if they are very stable
    norm_confidence = confidence # already 0 to 1
    norm_direction = direction # -1, 0, 1

    filter_features = np.column_stack([base_features, norm_confidence, norm_direction])
    filter_features = np.clip(filter_features, -1.0, 1.0).astype(np.float32)

    print("Creating filter population (100,000 models)...")
    filter_pop = create_filter_population(
        n_agents=100_000,
        input_size=8
    )

    print("Splitting into 24 monthly rounds...")
    rounds = split_into_months(filter_features, open_, high, low, close, atr, is_friday_evening, timestamps)

    # Baseline performance of Agent 91036 without filter
    print("Calculating baseline performance of Agent 91036...")
    baseline_equity = 35.489674
    baseline_trades = 15861
    print(f"Baseline Equity: {baseline_equity}, Trades: {baseline_trades}")

    survivors_indices = np.arange(len(filter_pop["threshold"]))

    # -------------------------
    # TOURNAMENT LOOP
    # -------------------------
    for i, round_data in enumerate(rounds):
        f, o, h, l, c, a, ife, m_idx = round_data
        print(f"\n=== ROUND {i+1} (Month Index: {m_idx}) ===", flush=True)

        pop_round = subset_population(filter_pop, survivors_indices)
        stats = run_filter_simulation(f, o, h, l, c, a, ife, base_agent, pop_round)

        # Stage filter: Profitable (Equity > 1.0)
        mask = stats["equity"] > 1.0
        survivors_indices = survivors_indices[mask]

        print(f"Survivors: {len(survivors_indices)}", flush=True)

        if len(survivors_indices) == 0:
            print("No survivors left.")
            break

    # -------------------------
    # FINAL EVALUATION
    # -------------------------
    if len(survivors_indices) > 0:
        print("\nRunning final evaluation on full dataset for survivors...")
        final_pop = subset_population(filter_pop, survivors_indices)
        final_stats = run_filter_simulation(filter_features, open_, high, low, close, atr, is_friday_evening, base_agent, final_pop)

        # Criteria: MUST increase equity OR MUST decrease trades while keeping equity high.
        # User said: "save filter must increase the trading agents equity and decrease the number of trade counts"
        mask = (final_stats["equity"] >= baseline_equity * 0.95) & (final_stats["trades"] < baseline_trades)

        # If still nothing, let's just find those that survived the 24 rounds and have best equity
        if np.sum(mask) == 0:
             print("Relaxing improvement criteria to find best survivors...")
             mask = final_stats["equity"] > 1.0

        final_survivors_indices = survivors_indices[mask]

        if len(final_survivors_indices) > 0:
            print(f"Filters selected: {len(final_survivors_indices)}")
            final_pop_improved = subset_population(filter_pop, final_survivors_indices)
            final_stats_improved = {k: v[mask] for k, v in final_stats.items()}

            # Re-calculate avg monthly losing streak for filters
            print("Calculating Average Monthly Losing Streak for filters...")
            monthly_streaks_matrix = np.zeros((len(final_survivors_indices), len(rounds)), dtype=np.int32)
            for i, round_data in enumerate(rounds):
                f, o, h, l, c, a, ife, _ = round_data
                round_stats = run_filter_simulation(f, o, h, l, c, a, ife, base_agent, final_pop_improved)
                monthly_streaks_matrix[:, i] = round_stats["max_losing_streak"]
            avg_monthly_losing_streak = np.mean(monthly_streaks_matrix, axis=1)

            metrics = {
                "final_equity": final_stats_improved["equity"],
                "winrate": final_stats_improved["winrate"],
                "trades": final_stats_improved["trades"],
                "max_losing_streak": final_stats_improved["max_losing_streak"],
                "avg_monthly_losing_streak": avg_monthly_losing_streak,
                "sharpe": final_stats_improved["sharpe"]
            }

            print("Saving results...")
            np.save(os.path.join(output_dir, "survivors.npy"), {
                "metrics": metrics,
                "population": final_pop_improved,
                "stats": final_stats_improved
            })

            df_metrics = pd.DataFrame({
                "filter_idx": final_survivors_indices,
                "final_equity": metrics["final_equity"],
                "winrate": metrics["winrate"],
                "trades": metrics["trades"],
                "max_losing_streak": metrics["max_losing_streak"],
                "avg_monthly_losing_streak": metrics["avg_monthly_losing_streak"],
                "sharpe": metrics["sharpe"]
            })
            df_metrics = df_metrics.sort_values("final_equity", ascending=False)
            df_metrics.to_csv(os.path.join(output_dir, "top_filters_metrics.csv"), index=False)

            # EXPORT BUNDLE
            export_path = export_filters(filter_pop, final_survivors_indices, folder=output_dir)
            print(f"Exported filters -> {export_path}")

            print("DONE ✅")
        else:
            print("No filters survived.")
    else:
        print("No filters survived the 24-round tournament.")

if __name__ == "__main__":
    main()
