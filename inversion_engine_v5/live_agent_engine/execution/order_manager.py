import time
import MetaTrader5 as mt5
from monitoring.logger import execution_logger, error_logger


class OrderManager:
    def __init__(self, connector, slippage=3, max_retries=3):
        self.connector = connector
        self.slippage = slippage
        self.max_retries = max_retries

    # =========================
    # BUILD ORDER REQUEST
    # =========================
    def build_request(
        self,
        agent_id,
        symbol,
        action,
        volume,
        price,
        sl,
        tp,
        comment=""
    ):
        """
        Builds a request for a PENDING STOP order.
        action=1 -> BUY_STOP
        action=-1 -> SELL_STOP
        """
        if action == 1:
            order_type = mt5.ORDER_TYPE_BUY_STOP
        elif action == -1:
            order_type = mt5.ORDER_TYPE_SELL_STOP
        else:
            raise ValueError(f"Invalid action for stop order: {action}")

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": self.slippage,
            "magic": int(agent_id),
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        return request

    def build_cancel_request(self, ticket):
        """
        Builds a request to cancel a pending order.
        """
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(ticket),
        }
        return request

    # =========================
    # EXECUTE ORDER (SAFE + RETRY)
    # =========================
    def execute(self, request):

        last_error = None

        for attempt in range(1, self.max_retries + 1):

            try:
                # Ensure MT5 is still alive
                if not self.connector.ensure_connection():
                    error_logger.error("MT5 connection lost during execution")
                    time.sleep(1)
                    continue

                execution_logger.info(
                    f"Sending order attempt {attempt}: "
                    f"{request['symbol']} | vol={request['volume']} | magic={request['magic']}"
                )

                result = mt5.order_send(request)

                # =========================
                # NULL RESULT HANDLING
                # =========================
                if result is None:
                    last_error = mt5.last_error()
                    error_logger.error(
                        f"Order NULL response (attempt {attempt}) | error={last_error}"
                    )
                    time.sleep(0.5)
                    continue

                # =========================
                # SUCCESS
                # =========================
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    execution_logger.info(
                        f"ORDER FILLED | ticket={result.order} | price={result.price}"
                    )
                    return result

                # =========================
                # FAILED ORDER
                # =========================
                last_error = f"{result.retcode} | {result.comment}"
                error_logger.error(
                    f"Order failed (attempt {attempt}) | {last_error}"
                )

                # retry only for retryable errors
                if result.retcode in [
                    10004,  # trade server busy
                    10006,  # no connection
                    10030,  # price changed
                    10016   # trade timeout
                ]:
                    time.sleep(0.5)
                    continue
                else:
                    # non-retryable error → stop early
                    break

            except Exception as e:
                last_error = str(e)
                error_logger.error(
                    f"Exception during order execution (attempt {attempt}): {e}"
                )
                time.sleep(0.5)

        error_logger.error(f"FINAL ORDER FAILURE | {last_error}")
        return None