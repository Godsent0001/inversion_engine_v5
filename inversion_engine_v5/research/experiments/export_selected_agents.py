import numpy as np
import pickle
import os

def export_specific_agents(agents_indices, src_path="outputs/survivors.npy", dest_folder="live_agent_engine/models"):
    if not os.path.exists(src_path):
        print(f"Error: {src_path} not found.")
        return

    data = np.load(src_path, allow_pickle=True).item()
    pop = data["population"]

    os.makedirs(dest_folder, exist_ok=True)

    for idx in agents_indices:
        agent_data = {}
        # Find the actual index in the survivor population
        # Note: The user refers to agent IDs like 36, 57...
        # In run_full_sim, survivors is an array of indices into the ORIGINAL population.
        # However, final_pop in survivors.npy ALREADY contains only the survivors.
        # So we need to be careful if the user means indices in the survivor pool or the original 100k.
        # Usually, when someone says "agent 36", they might mean the 36th agent in the exported list.

        # We'll extract based on the provided index in the CURRENT survivors list.
        if idx >= len(pop["rrr"]):
            print(f"Warning: Index {idx} out of range for survivors (total {len(pop['rrr'])}).")
            continue

        for k, v in pop.items():
            if isinstance(v, np.ndarray):
                agent_data[k] = v[idx]
            elif isinstance(v, list):
                agent_data[k] = v[idx]
            else:
                agent_data[k] = v

        # Network parameters
        agent_data["w1"] = pop["w1"][idx]
        agent_data["w2"] = pop["w2"][idx]
        agent_data["b1"] = pop["b1"][idx]
        agent_data["b2"] = pop["b2"][idx]

        out_path = os.path.join(dest_folder, f"agent_{idx}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(agent_data, f)
        print(f"Exported agent {idx} to {out_path}")

if __name__ == "__main__":
    target_agents = [36, 57, 58, 4408, 2669]
    export_specific_agents(target_agents)
