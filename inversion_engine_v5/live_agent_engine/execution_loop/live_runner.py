import time
import numpy as np
from datetime import datetime

from config import settings
from shared.indicators.pipeline import build_features
from agents.agent_loader import AgentLoader
from agents.decision_engine import DecisionEngine
from agents.portfolio_manager import PortfolioManager
from execution.mt5_connector import MT5Connector
from execution.order_manager import OrderManager
from execution.position_router import PositionRouter
from risk.risk_engine import RiskEngine
from monitoring.logger import execution_logger, error_logger, trade_logger
from monitoring.agent_tracker import AgentTracker


class LiveRunner:
    def __init__(self):

        self.connector = MT5Connector()
        self.loader = AgentLoader()
        self.decision_engine = DecisionEngine()
        self.router = PositionRouter()

        self.risk_engine = RiskEngine(risk_per_trade=settings.RISK_PER_TRADE)

        self.agents = self.loader.load_agents()
        self.agent_ids = [a["id"] for a in self.agents]

        self.portfolio = PortfolioManager(self.agent_ids)
        self.tracker = AgentTracker(self.portfolio, self.agents)

        self.order_manager = OrderManager(self.connector, slippage=settings.SLIPPAGE)

        self.last_candle_time = None
        self.last_sync_time = 0

        self.execution_lock = set()

    def run_once(self):

        try:
            if not self.connector.ensure_connection():
                return

            if time.time() - self.last_sync_time > 5:
                self.tracker.sync_with_mt5(self.agent_ids)

                for agent_id in list(self.execution_lock):
                    if not self.router.has_open_position(agent_id):
                        self.execution_lock.discard(agent_id)

                self.last_sync_time = time.time()

            execution_logger.info("Checking for new candle...")

            df = self.connector.get_latest_data(
                settings.SYMBOL,
                settings.TIMEFRAME,
                n_bars=200
            )

            if df is None or len(df) < 60:
                return

            current_candle_time = df.iloc[-1]["time"]

            if self.last_candle_time == current_candle_time:
                return

            self.last_candle_time = current_candle_time

            execution_logger.info(
                f"New candle detected: {datetime.fromtimestamp(current_candle_time)}"
            )

            df_closed = df.iloc[:-1]

            high = df_closed["high"].values.astype(np.float32)
            low = df_closed["low"].values.astype(np.float32)
            close = df_closed["close"].values.astype(np.float32)

            features_full, atr_full = build_features(high, low, close)

            latest_features = features_full[-1]
            latest_atr = atr_full[-1]

            current_price = df.iloc[-1]["open"]

            spread = self.connector.get_spread(settings.SYMBOL)

            execution_logger.info(
                f"Market | Price: {current_price:.2f} | ATR: {latest_atr:.4f} | Spread: {spread}"
            )

            # =========================
            # AGENT LOOP
            # =========================
            for agent in self.agents:

                agent_id = agent["id"]

                if agent_id in self.execution_lock:
                    continue

                # 1. CHECK IF IN POSITION
                if self.router.has_open_position(agent_id):
                    continue

                # 2. CHECK COOLDOWN
                if self.portfolio.portfolios[str(agent_id)]["cooldown"] > 0:
                    continue

                # 3. GET AGENT DECISION (INCLUDING STOP PRICE)
                action, confidence, entry_p = self.decision_engine.decide(
                    agent,
                    latest_features,
                    high,
                    low,
                    close
                )

                # 4. HANDLE PENDING ORDERS
                # Fetch currently active pending orders from MT5 for this agent
                active_pending = self.router.get_agent_pending_orders(agent_id, settings.SYMBOL)

                # We also check our stored ticket to be safe
                stored_ticket = self.portfolio.get_pending_ticket(agent_id)

                if action == 0:
                    # Neutral signal does NOT cancel pending orders (as per research logic)
                    # But we should verify if the stored ticket still exists on server
                    if stored_ticket:
                        exists = any(o.ticket == stored_ticket for o in active_pending)
                        if not exists:
                            # It was either triggered or manually cancelled
                            self.portfolio.set_pending_ticket(agent_id, None)
                    continue

                # If we have a directional signal (1 or -1), we must ensure we have the NEWEST one
                # First, cancel ALL existing pending orders for this agent on MT5
                for po in active_pending:
                    cancel_req = self.order_manager.build_cancel_request(po.ticket)
                    self.order_manager.execute(cancel_req)
                    execution_logger.info(f"Cancelled old pending order {po.ticket} for agent {agent_id}")

                self.portfolio.set_pending_ticket(agent_id, None)

                # 5. EXECUTE NEW PENDING STOP ORDER
                equity = self.portfolio.get_equity(agent_id)

                rrr_used = float(agent["rrr"])
                atr_mult_used = float(agent["atr"])

                dist = latest_atr * atr_mult_used

                if action == 1: # BUY STOP
                    sl = entry_p - dist
                    tp = entry_p + dist * rrr_used
                else: # SELL STOP
                    sl = entry_p + dist
                    tp = entry_p - dist * rrr_used

                lots = self.risk_engine.calculate_lot_size(
                    settings.SYMBOL,
                    equity,
                    entry_p,
                    sl
                )

                request = self.order_manager.build_request(
                    agent_id,
                    settings.SYMBOL,
                    action,
                    lots,
                    entry_p,
                    sl,
                    tp,
                    comment=f"Agent {agent_id}"
                )

                result = self.order_manager.execute(request)

                if result and result.retcode == 10009:
                    # Store the new ticket ID for persistence
                    self.portfolio.set_pending_ticket(agent_id, result.order)

                    trade_logger.info(
                        f"""
                        ================================
                        STOP ORDER PLACED
                        Agent     : {agent_id}
                        Action    : {action} (Stop)
                        Confidence: {confidence:.2f}
                        Price     : {entry_p}
                        SL        : {sl}
                        TP        : {tp}
                        Lot       : {lots}
                        ================================
                        """
                    )
                    time.sleep(0.5)

            self.portfolio.decrement_cooldowns()

        except Exception as e:
            import traceback
            error_logger.error(f"{str(e)}\n{traceback.format_exc()}")

    def start(self):

        if not self.connector.connect():
            print("CRITICAL: MT5 failed")
            return

        execution_logger.info("Live Engine Started")

        while True:
            self.run_once()
            time.sleep(1)


if __name__ == "__main__":
    LiveRunner().start()