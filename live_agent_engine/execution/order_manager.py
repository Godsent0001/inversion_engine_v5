import MetaTrader5 as mt5

class OrderManager:
    def __init__(self, connector, slippage=3):
        self.connector = connector
        self.slippage = slippage

    def build_request(self, agent_id, symbol, action, volume, price, sl, tp, comment=""):
        # action: 1 (BUY), -1 (SELL)
        order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.slippage,
            "magic": agent_id,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        return request

    def execute(self, request):
        return self.connector.send_order(request)
