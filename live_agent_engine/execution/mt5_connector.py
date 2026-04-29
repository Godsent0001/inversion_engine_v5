import MetaTrader5 as mt5
import time

class MT5Connector:
    def __init__(self):
        pass

    def connect(self):
        if not mt5.initialize():
            print("MT5 initialize failed")
            return False
        return True

    def get_latest_data(self, symbol, timeframe_str, n_bars=100):
        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M30)

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
        if rates is None:
            return None

        import pandas as pd
        df = pd.DataFrame(rates)
        return df

    def send_order(self, request):
        result = mt5.order_send(request)
        return result

    def close(self):
        mt5.shutdown()
