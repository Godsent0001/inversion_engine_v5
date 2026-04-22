import numpy as np
import pickle
import os
from datetime import datetime


# -------------------------
# CLEAN EXPORT DATA
# -------------------------
def build_export_bundle(pop, survivors):
    """
    Extract only necessary fields for live trading
    """

    export = {}

    # Core model params
    export["w1"] = pop["w1"][survivors]
    export["w2"] = pop["w2"][survivors]
    export["b1"] = pop["b1"][survivors]
    export["b2"] = pop["b2"][survivors]

    # Strategy parameters
    export["rrr"] = pop["rrr"][survivors]
    export["atr"] = pop["atr"][survivors]
    export["threshold"] = pop["threshold"][survivors]
    export["cooldown"] = pop["cooldown"][survivors]
    export["aggression"] = pop["aggression"][survivors]

    # Optional (if exists)
    if "family" in pop:
        export["family"] = pop["family"][survivors]

    # Metadata
    export["n_agents"] = len(survivors)
    export["input_size"] = pop["w1"].shape[1]
    export["hidden_size"] = pop["w1"].shape[2]

    return export


# -------------------------
# EXPORT FUNCTION
# -------------------------
def export_agents(pop, survivors, folder="outputs"):

    if len(survivors) == 0:
        raise ValueError("No survivors to export")

    os.makedirs(folder, exist_ok=True)

    bundle = build_export_bundle(pop, survivors)

    # Timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"export_bundle_{timestamp}.pkl")

    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nExported {bundle['n_agents']} agents")

    return path