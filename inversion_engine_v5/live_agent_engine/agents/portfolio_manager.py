import os
import json
import numpy as np


class PortfolioManager:
    def __init__(self, agent_ids, starting_balance=10000.0):
        # Absolute path (robust)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.storage_path = os.path.join(base_dir, "storage", "performance.json")

        self.starting_balance = float(starting_balance)
        self.portfolios = self._load_or_init(agent_ids)

    # -------------------------
    # LOAD OR INIT
    # -------------------------
    def _load_or_init(self, agent_ids):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}

        for aid in agent_ids:
            if str(aid) not in data:
                data[str(aid)] = {
                    "equity": float(self.starting_balance),
                    "cooldown": int(0)
                }

        self._save(data)
        return data

    # -------------------------
    # UPDATE EQUITY
    # -------------------------
    def update_equity(self, agent_id, pnl_percent):
        aid = str(agent_id)

        if aid not in self.portfolios:
            return

        current_equity = float(self.portfolios[aid]["equity"])
        pnl_percent = float(pnl_percent)

        self.portfolios[aid]["equity"] = current_equity * (1.0 + pnl_percent)

        self._save()

    # -------------------------
    # SET COOLDOWN
    # -------------------------
    def set_cooldown(self, agent_id, value):
        aid = str(agent_id)

        if aid not in self.portfolios:
            return

        self.portfolios[aid]["cooldown"] = int(value)

        self._save()

    # -------------------------
    # DECREMENT COOLDOWNS
    # -------------------------
    def decrement_cooldowns(self):
        for aid in self.portfolios:
            cooldown = self.portfolios[aid]["cooldown"]

            if cooldown > 0:
                self.portfolios[aid]["cooldown"] = int(cooldown) - 1

        self._save()

    # -------------------------
    # GET EQUITY
    # -------------------------
    def get_equity(self, agent_id):
        return float(self.portfolios[str(agent_id)]["equity"])

    # -------------------------
    # SAFE SAVE (FIXED)
    # -------------------------
    def _save(self, data=None):
        if data is None:
            data = self.portfolios

        try:
            safe_data = self._convert_to_native(data)

            with open(self.storage_path, "w") as f:
                json.dump(safe_data, f, indent=2)

        except Exception as e:
            print(f"[Portfolio Save Error] {e}")

    # -------------------------
    # 🔥 CONVERT NUMPY TYPES
    # -------------------------
    def _convert_to_native(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._convert_to_native(v) for v in obj]

        elif isinstance(obj, (np.integer,)):
            return int(obj)

        elif isinstance(obj, (np.floating,)):
            return float(obj)

        return obj