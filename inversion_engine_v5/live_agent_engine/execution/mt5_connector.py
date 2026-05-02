import MetaTrader5 as mt5
import time
import pandas as pd

from config import settings
from monitoring.logger import execution_logger, error_logger


class MT5Connector:
    def __init__(self):

        self.connected = False
        self.last_reconnect_time = 0
        self.reconnect_cooldown = 5  # seconds

    # =========================
    # CONNECT TO MT5
    # =========================
    def connect(self):

        try:
            if not mt5.initialize(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            ):
                error_logger.error(
                    f"MT5 initialize/login failed: {mt5.last_error()}"
                )
                return False

            terminal_info = mt5.terminal_info()

            if terminal_info is None:
                error_logger.error("MT5 terminal_info is None")
                mt5.shutdown()
                return False

            if not terminal_info.connected:
                error_logger.error("MT5 terminal NOT connected")
                mt5.shutdown()
                return False

            self.connected = True
            execution_logger.info("MT5 CONNECTED SUCCESSFULLY")

            return True

        except Exception as e:
            error_logger.error(f"MT5 connection exception: {e}")
            return False

    # =========================
    # SAFE CONNECTION CHECK
    # =========================
    def ensure_connection(self):

        try:
            info = mt5.terminal_info()

            if info is None or not info.connected:
                now = time.time()

                # prevent reconnect spam (VERY IMPORTANT for stability)
                if now - self.last_reconnect_time < self.reconnect_cooldown:
                    return False

                self.last_reconnect_time = now

                execution_logger.warning("MT5 disconnected → reconnecting...")

                mt5.shutdown()
                time.sleep(1)

                return self.connect()

            return True

        except Exception as e:
            error_logger.error(f"Connection check failed: {e}")
            return False

    # =========================
    # GET SPREAD
    # =========================
    def get_spread(self, symbol):

        try:
            info = mt5.symbol_info(symbol)

            if info is None:
                return None

            return info.spread

        except Exception as e:
            error_logger.error(f"Spread error: {e}")
            return None

    # =========================
    # GET MARKET DATA
    # =========================
    def get_latest_data(self, symbol, timeframe_str, n_bars=100):

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

        try:
            for attempt in range(3):

                rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)

                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    return df

                time.sleep(0.5)

            error_logger.error(
                f"Failed to fetch rates for {symbol} after retries"
            )

            return None

        except Exception as e:
            error_logger.error(f"Market data error: {e}")
            return None

    # =========================
    # SEND ORDER
    # =========================
    def send_order(self, request):

        try:
            execution_logger.info(
                f"ORDER SENT | {request['symbol']} | vol={request['volume']} | magic={request['magic']}"
            )

            result = mt5.order_send(request)

            if result is None:
                error_logger.error(f"Order failed: {mt5.last_error()}")
                return None

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_logger.error(
                    f"ORDER FAILED | retcode={result.retcode} | {result.comment}"
                )
            else:
                execution_logger.info(
                    f"ORDER FILLED | ticket={result.order} | price={result.price}"
                )

            return result

        except Exception as e:
            error_logger.error(f"Order exception: {e}")
            return None

    # =========================
    # SHUTDOWN MT5
    # =========================
    def close(self):

        try:
            mt5.shutdown()
            self.connected = False
            execution_logger.info("MT5 DISCONNECTED CLEANLY")

        except Exception as e:
            error_logger.error(f"Shutdown error: {e}")