import MetaTrader5 as mt5

class RiskEngine:
    def __init__(self, risk_per_trade=0.05):
        self.risk_per_trade = risk_per_trade

    def calculate_lot_size(self, symbol, equity, entry_price, sl_price):
        if sl_price == 0 or entry_price == sl_price:
            return 0.01

        risk_amount = equity * self.risk_per_trade
        sl_distance = abs(entry_price - sl_price)

        # Simple lot calculation for XAUUSD (Gold)
        # 1 lot = 100 ounces. 1 USD move on 1 lot = 100 USD.
        # So pip_value for 1 lot per 1 unit price move is 100.

        if sl_distance == 0: return 0.01

        # units_per_lot depends on the broker. Usually 100 for XAUUSD.
        units_per_lot = 100.0

        lots = risk_amount / (sl_distance * units_per_lot)

        # Round to 2 decimals and ensure within limits
        lots = round(lots, 2)
        if lots < 0.01: lots = 0.01
        if lots > 10.0: lots = 10.0 # safety cap

        return lots
