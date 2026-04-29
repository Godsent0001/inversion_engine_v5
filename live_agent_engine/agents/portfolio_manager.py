import os
import json

class PortfolioManager:
    def __init__(self, agent_ids, starting_balance=10000.0, storage_path="storage/performance.json"):
        self.storage_path = storage_path
        self.starting_balance = starting_balance
        self.portfolios = self._load_or_init(agent_ids)

    def _load_or_init(self, agent_ids):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                # Ensure all agents are present
                for aid in agent_ids:
                    if str(aid) not in data:
                        data[str(aid)] = {"equity": self.starting_balance, "cooldown": 0}
                return data
        else:
            return {str(aid): {"equity": self.starting_balance, "cooldown": 0} for aid in agent_ids}

    def update_equity(self, agent_id, pnl_percent):
        self.portfolios[str(agent_id)]["equity"] *= (1.0 + pnl_percent)
        self._save()

    def set_cooldown(self, agent_id, value):
        self.portfolios[str(agent_id)]["cooldown"] = value
        self._save()

    def decrement_cooldowns(self):
        for aid in self.portfolios:
            if self.portfolios[aid]["cooldown"] > 0:
                self.portfolios[aid]["cooldown"] -= 1
        self._save()

    def get_equity(self, agent_id):
        return self.portfolios[str(agent_id)]["equity"]

    def _save(self):
        with open(self.storage_path, "w") as f:
            json.dump(self.portfolios, f, indent=2)
