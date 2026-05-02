import os
import pickle
import json


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
        required = ["rrr", "atr", "threshold", "aggression", "cooldown"]

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
        and attaches their model parameters.
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
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)

                # Attach agent ID
                model_data["id"] = agent_id

                # ✅ VALIDATE BEFORE ADDING
                model_data = self._validate_agent(model_data)

                agents.append(model_data)

            except Exception as e:
                print(f"[ERROR] Failed to load agent {agent_id}: {e}")

        if len(agents) == 0:
            raise RuntimeError("No agents were successfully loaded.")

        print(f"[INFO] Loaded {len(agents)} agents successfully.")

        return agents