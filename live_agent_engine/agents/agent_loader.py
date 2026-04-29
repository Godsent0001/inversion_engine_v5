import os
import pickle
import json

class AgentLoader:
    def __init__(self, config_path="config/models_config.json", models_dir="models"):
        self.config_path = config_path
        self.models_dir = models_dir

    def load_agents(self):
        with open(self.config_path, "r") as f:
            config = json.load(f)

        agents = []
        for agent_cfg in config["agents"]:
            path = os.path.join(self.models_dir, agent_cfg["model"])
            with open(path, "rb") as f:
                model_data = pickle.load(f)
                model_data["id"] = agent_cfg["id"]
                agents.append(model_data)
        return agents
