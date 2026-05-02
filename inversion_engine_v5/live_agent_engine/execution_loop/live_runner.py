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

                if self.router.has_open_position(agent_id):
                    continue

                if self.portfolio.portfolios[str(agent_id)]["cooldown"] > 0:
                    continue

                action, confidence = self.decision_engine.decide(
                    agent,
                    latest_features
                )

                if action == 0:
                    continue

                equity = self.portfolio.get_equity(agent_id)

                # =========================
                # TRADE PARAMETERS (IMPORTANT PART)
                # =========================
                rrr_used = float(agent["rrr"])
                atr_mult_used = float(agent["atr"])

                dist = latest_atr * atr_mult_used

                if action == 1:
                    sl = current_price - dist
                    tp = current_price + dist * rrr_used
                else:
                    sl = current_price + dist
                    tp = current_price - dist * rrr_used

                lots = self.risk_engine.calculate_lot_size(
                    settings.SYMBOL,
                    equity,
                    current_price,
                    sl
                )

                request = self.order_manager.build_request(
                    agent_id,
                    settings.SYMBOL,
                    action,
                    lots,
                    current_price,
                    sl,
                    tp,
                    comment=f"Agent {agent_id}"
                )

                result = self.order_manager.execute(request)

                if result and result.retcode == 10009:

                    # 🔥 ENHANCED TRADE LOG
                    trade_logger.info(
                        f"""
                        ================================
                        TRADE EXECUTED
                        Agent     : {agent_id}
                        Action    : {action}
                        Confidence: {confidence:.2f}
                        RRR       : {rrr_used}
                        ATR Mult  : {atr_mult_used}
                        Entry     : {current_price}
                        SL        : {sl}
                        TP        : {tp}
                        Lot       : {lots}
                        ================================
                        """
                    )

                    self.execution_lock.add(agent_id)
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