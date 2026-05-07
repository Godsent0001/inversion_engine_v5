import numpy as np
import pandas as pd
import os
import json

def load_stats(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=True).item()
        return data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    market_path = "inversion_engine_v5/outputs/survivors.npy"
    stop_path = "inversion_engine_v5/outputs_stop_orders/survivors.npy"

    market_data = load_stats(market_path)
    stop_data = load_stats(stop_path)

    if market_data is None:
        print("Market execution data not found. Run existing engine first if you want comparison.")

    if stop_data is None:
        print("Stop order data not found. Run stop engine first.")
        return

    results = []

    if market_data:
        m_stats = market_data["stats"]
        results.append({
            "Engine": "Market Execution",
            "Survivors": len(market_data["metrics"]["final_equity"]),
            "Avg Equity": np.mean(m_stats["equity"]),
            "Max Equity": np.max(m_stats["equity"]),
            "Avg Winrate": np.mean(m_stats.get("winrate", 0)),
            "Avg Trades": np.mean(m_stats["trades"]),
            "Avg MaxDD": np.mean(m_stats["max_drawdown"]),
            "Avg Sharpe": np.mean(m_stats.get("sharpe", 0))
        })

    s_stats = stop_data["stats"]
    results.append({
        "Engine": "Stop Orders",
        "Survivors": len(stop_data["metrics"]["final_equity"]),
        "Avg Equity": np.mean(s_stats["equity"]),
        "Max Equity": np.max(s_stats["equity"]),
        "Avg Winrate": np.mean(s_stats.get("winrate", 0)),
        "Avg Trades": np.mean(s_stats["trades"]),
        "Avg MaxDD": np.mean(s_stats["max_drawdown"]),
        "Avg Sharpe": np.mean(s_stats.get("sharpe", 0)),
        "Avg Placed": np.mean(s_stats["orders_placed"]),
        "Avg Cancelled": np.mean(s_stats["orders_cancelled"]),
        "Trigger %": (np.sum(s_stats["trades"]) / np.sum(s_stats["orders_placed"])) * 100 if np.sum(s_stats["orders_placed"]) > 0 else 0
    })

    df = pd.DataFrame(results).T
    print("\n=== ENGINE COMPARISON ===")
    print(df)

    df.to_csv("inversion_engine_v5/outputs_stop_orders/comparison.csv")
    print("\nComparison saved to outputs_stop_orders/comparison.csv")

if __name__ == "__main__":
    main()
