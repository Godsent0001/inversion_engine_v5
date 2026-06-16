import numpy as np
import pickle
import os
from datetime import datetime

def build_filter_export_bundle(pop, survivors):
    export = {}
    export["w1"] = pop["w1"][survivors]
    export["w2"] = pop["w2"][survivors]
    export["b1"] = pop["b1"][survivors]
    export["b2"] = pop["b2"][survivors]
    export["threshold"] = pop["threshold"][survivors]

    export["n_agents"] = len(survivors)
    export["input_size"] = pop["w1"].shape[1]
    export["hidden_size"] = pop["w1"].shape[2]
    return export

def export_filters(pop, survivors, folder="inversion_engine_v5/filter/outputs"):
    if len(survivors) == 0:
        raise ValueError("No survivors to export")
    os.makedirs(folder, exist_ok=True)
    bundle = build_filter_export_bundle(pop, survivors)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"filter_bundle_{timestamp}.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nExported {bundle['n_agents']} filters")
    return path
