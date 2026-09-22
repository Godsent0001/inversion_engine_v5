import os
import pickle
import json
import torch
import sys

# Ensure inversion_engine_v5 root is accessible
base_engine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if base_engine_dir not in sys.path:
    sys.path.append(base_engine_dir)

from research.core.gru_sim_engine import GRUClassifier


class AgentLoader:
    def __init__(self):
        """
        Dynamically resolve paths so this works
        no matter where the script is executed from.
        """

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.config_path = os.path.join(base_dir, "config", "models_config.json")
        self.models_dir = os.path.join(base_dir, "models")

    # =========================
    # VALIDATION LAYER
    # =========================
    def _validate_agent(self, agent):
        required = ["rrr", "atr"]

        for key in required:
            if key not in agent:
                raise ValueError(
                    f"Agent {agent.get('id')} is missing required field: {key}"
                )

        return agent

    # =========================
    # LOAD AGENTS
    # =========================
    def load_agents(self):
        """
        Loads all agents defined in models_config.json
        and attaches their model parameters or PyTorch neural networks.
        """

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        agents = []

        for agent_cfg in config.get("agents", []):
            agent_id = agent_cfg.get("id")
            model_file = agent_cfg.get("model")

            if model_file is None:
                print(f"[WARNING] Agent {agent_id} has no model file. Skipping.")
                continue

            model_path = os.path.join(self.models_dir, model_file)

            if not os.path.exists(model_path):
                print(f"[WARNING] Model file not found: {model_path}. Skipping agent {agent_id}.")
                continue

            try:
                if model_file.endswith(".pt") or agent_cfg.get("model_type") == "gru":
                    checkpoint = torch.load(model_path, map_location="cpu")
                    input_dim = checkpoint.get("input_dim", 328)
                    hidden_dim = checkpoint.get("hidden_dim", 64)
                    num_classes = checkpoint.get("num_classes", 3)
                    seq_len = checkpoint.get("seq_len", 10)

                    net = GRUClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
                    net.load_state_dict(checkpoint["state_dict"])
                    net.eval()

                    agent_data = {
                        "id": agent_id,
                        "model_type": "gru",
                        "net": net,
                        "seq_len": seq_len,
                        "input_dim": input_dim,
                        "rrr": agent_cfg.get("rrr", 2.0),
                        "atr": agent_cfg.get("atr", 3.6),
                        "checkpoint_info": checkpoint
                    }
                else:
                    with open(model_path, "rb") as f:
                        agent_data = pickle.load(f)

                    agent_data["id"] = agent_id
                    agent_data["model_type"] = "mlp"
                    if "rrr" not in agent_data and "rrr" in agent_cfg:
                        agent_data["rrr"] = agent_cfg["rrr"]
                    if "atr" not in agent_data and "atr" in agent_cfg:
                        agent_data["atr"] = agent_cfg["atr"]

                # ✅ VALIDATE BEFORE ADDING
                agent_data = self._validate_agent(agent_data)

                agents.append(agent_data)

            except Exception as e:
                print(f"[ERROR] Failed to load agent {agent_id}: {e}")

        if len(agents) == 0:
            raise RuntimeError("No agents were successfully loaded.")

        print(f"[INFO] Loaded {len(agents)} agents successfully.")

        return agents