import numpy as np
import pandas as pd
import os
import pickle

def export_filter_models(pop, survivors_indices, folder="inversion_engine_v5/filter_outputs"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    survivor_pop = {}
    for k, v in pop.items():
        if isinstance(v, np.ndarray) and len(v) == 100000:
            survivor_pop[k] = v[survivors_indices]
        else:
            survivor_pop[k] = v

    bundle_path = os.path.join(folder, "filter_models_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(survivor_pop, f)

    return bundle_path

def main():
    data = np.load("inversion_engine_v5/filter_outputs/survivors.npy", allow_pickle=True).item()
    pop = data["population"]
    df = pd.read_csv("inversion_engine_v5/filter_outputs/top_filters_metrics.csv")

    survivors_indices = df["filter_idx"].values

    print(f"Exporting {len(survivors_indices)} filter models...")
    path = export_filter_models(pop, survivors_indices, folder="inversion_engine_v5/filter_outputs")
    print(f"Filter models bundle saved to {path}")

if __name__ == "__main__":
    main()
