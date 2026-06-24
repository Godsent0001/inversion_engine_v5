import numpy as np
import pandas as pd
import os
from inversion_engine_v5.research_stop_orders.portfolio.exporter import export_agents

def main():
    src_path = "inversion_engine_v5/outputs_stop_orders/survivors.npy"
    if not os.path.exists(src_path):
        print(f"Error: {src_path} not found.")
        return

    data = np.load(src_path, allow_pickle=True).item()
    final_pop = data["population"]
    metrics = data["metrics"]

    # Rank agents
    df_metrics = pd.DataFrame({
        "agent_idx": np.arange(len(metrics["final_equity"])),
        "final_equity": metrics["final_equity"]
    })
    df_metrics = df_metrics.sort_values("final_equity", ascending=False)

    # Export top 100
    top_100_indices = df_metrics.head(100)["agent_idx"].values
    top_100_folder = "inversion_engine_v5/models_stop_orders/top_100"
    os.makedirs(top_100_folder, exist_ok=True)

    export_path_top = export_agents(final_pop, top_100_indices, folder=top_100_folder)
    print(f"Exported top 100 models → {export_path_top}")

if __name__ == "__main__":
    main()
