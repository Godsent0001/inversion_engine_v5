import MetaTrader5 as mt5

class PositionRouter:
    def __init__(self):
        pass

    def get_agent_positions(self, agent_id, symbol="XAUUSD"):
        # Filter by both magic number AND symbol to ensure total isolation
        positions = mt5.positions_get(magic=agent_id, symbol=symbol)
        return positions

    def has_open_position(self, agent_id, symbol="XAUUSD"):
        positions = self.get_agent_positions(agent_id, symbol)
        return len(positions) > 0 if positions is not None else False
