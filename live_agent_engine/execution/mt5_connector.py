import MetaTrader5 as mt5
import time
from config import settings
from monitoring.logger import execution_logger, error_logger

class MT5Connector:
    def __init__(self):
        pass

    def connect(self):
        if not mt5.initialize(
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER
        ):
            error_logger.error(f"MT5 initialize/login failed, error code: {mt5.last_error()}")
            return False

        # Check connection status
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            error_logger.error("Failed to get terminal info")
            mt5.shutdown()
            return False

        if not terminal_info.connected:
            error_logger.error("MT5 terminal is not connected to the server")
            # We don't return False here as it might connect later, but we log it.

        execution_logger.info("MT5 Connector Initialized")
        return True

    def ensure_connection(self):
        """Verify and attempt reconnection if lost."""
        if mt5.terminal_info() is None or not mt5.terminal_info().connected:
            execution_logger.warning("Connection lost. Attempting to reconnect...")
            mt5.shutdown()
            time.sleep(5)
            return self.connect()
        return True

    def get_spread(self, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        return symbol_info.spread

    def get_latest_data(self, symbol, timeframe_str, n_bars=100):
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M30)

        # Try to get data with retries
        for _ in range(3):
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
            if rates is not None:
                import pandas as pd
                df = pd.DataFrame(rates)
                return df
            time.sleep(1)

        error_logger.error(f"Failed to get rates for {symbol} after retries")
        return None

    def send_order(self, request):
        # Log before sending
        execution_logger.info(f"Sending Order: {request['type']} {request['volume']} {request['symbol']} @ {request['price']}")

        result = mt5.order_send(request)

        if result is None:
            error_logger.error(f"Order send failed, no result. Error: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_logger.error(f"Order failed. Retcode: {result.retcode}, Comment: {result.comment}")
        else:
            execution_logger.info(f"Order Completed: Ticket {result.order}, Fill Price: {result.price}")

        return result

    def close(self):
        mt5.shutdown()
        execution_logger.info("MT5 Connector Closed")
