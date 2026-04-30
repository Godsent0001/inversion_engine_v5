import numpy as np
import json
import os

filepath = "outputs/survivors.npy"

if not os.path.exists(filepath):
    print(f"Error: {filepath} not found.")
else:
    data = np.load(filepath, allow_pickle=True).item()

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    # Sample top 100 agents for JSON visibility
    n_sample = min(100, len(data["metrics"]["final_equity"]))

    clean = {
        "metrics": {k: convert(v) for k, v in data["metrics"].items()},
        "population_sample": {
            "rrr": convert(data["population"]["rrr"][:n_sample]),
            "atr": convert(data["population"]["atr"][:n_sample]),
            "threshold": convert(data["population"]["threshold"][:n_sample]),
            "family": convert(data["population"].get("family", [])[:n_sample])
        }
    }

    with open("outputs/survivors_sample.json", "w") as f:
        json.dump(clean, f, indent=2)

    print("Saved → outputs/survivors_sample.json")
