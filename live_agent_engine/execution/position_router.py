import MetaTrader5 as mt5

class PositionRouter:
    def __init__(self):
        pass

    def get_agent_positions(self, agent_id):
        positions = mt5.positions_get(magic=agent_id)
        return positions

    def has_open_position(self, agent_id):
        positions = self.get_agent_positions(agent_id)
        return len(positions) > 0 if positions is not None else False
