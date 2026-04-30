import time
import pandas as pd
import numpy as np
from datetime import datetime

from config import settings
from research.data.feed import MT5DataFeed
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

    def run_once(self):
        try:
            # 0. ENSURE CONNECTION
            if not self.connector.ensure_connection():
                execution_logger.error("Failed to maintain MT5 connection.")
                return

            # 1. SYNC TRADES & UPDATE EQUITY
            self.tracker.sync_with_mt5(self.agent_ids)

            # 2. FETCH DATA
            execution_logger.info("Checking for new candle...")
            df = self.connector.get_latest_data(settings.SYMBOL, settings.TIMEFRAME, n_bars=200)

            if df is None or len(df) < 50:
                return

            current_candle_time = df.iloc[-1]['time']

            # Check for new candle (M30)
            if self.last_candle_time is not None and current_candle_time == self.last_candle_time:
                return

            execution_logger.info(f"New candle detected: {datetime.fromtimestamp(current_candle_time)}")
            self.last_candle_time = current_candle_time

            # Safeguard: Remove the current (open) candle from feature calculation
            # to ensure we only use completed bars.
            df_closed = df.iloc[:-1]

            high = df_closed['high'].values.astype(np.float32)
            low = df_closed['low'].values.astype(np.float32)
            close = df_closed['close'].values.astype(np.float32)

            # Features built from CLOSED bars
            features_full, atr_full = build_features(high, low, close)

            latest_features = features_full[-1]
            latest_atr = atr_full[-1]

            # Execution price is the OPEN of the current candle
            current_open_price = df.iloc[-1]['open']

            # Log spread
            spread = self.connector.get_spread(settings.SYMBOL)
            execution_logger.info(f"Market Info | Price: {current_open_price:.2f}, ATR: {latest_atr:.4f}, Spread: {spread}")

            # 3. PROCESS AGENTS
            for agent in self.agents:
                agent_id = agent["id"]

                # Check for existing position
                if self.router.has_open_position(agent_id):
                    continue

                # Check cooldown
                if self.portfolio.portfolios[str(agent_id)]["cooldown"] > 0:
                    continue

                # Decision
                action, confidence = self.decision_engine.decide(agent, latest_features)

                if action != 0:
                    execution_logger.info(f"Agent {agent_id} signaled {action} (conf: {confidence:.2f})")

                    equity = self.portfolio.get_equity(agent_id)

                    # SL/TP
                    dist = latest_atr * agent["atr"]
                    if action == 1:
                        sl = current_open_price - dist
                        tp = current_open_price + dist * agent["rrr"]
                    else:
                        sl = current_open_price + dist
                        tp = current_open_price - dist * agent["rrr"]

                    lots = self.risk_engine.calculate_lot_size(settings.SYMBOL, equity, current_open_price, sl)

                    # Execute
                    request = self.order_manager.build_request(
                        agent_id, settings.SYMBOL, action, lots, current_open_price, sl, tp,
                        comment=f"Agent {agent_id}"
                    )

                    result = self.order_manager.execute(request)
                    if result and result.retcode == 10009:
                        # Cooldown is set on trade CLOSE in AgentTracker to match backtest logic
                        trade_logger.info(f"Agent {agent_id} BUY/SELL executed. Waiting for trade closure to set cooldown.")

            # Decrement cooldowns
            self.portfolio.decrement_cooldowns()

        except Exception as e:
            import traceback
            error_msg = f"Exception in main loop: {str(e)}\n{traceback.format_exc()}"
            error_logger.error(error_msg)

    def start(self):
        if not self.connector.connect():
            print("CRITICAL: Failed to initialize MT5 Connector. Check logs.")
            return

        execution_logger.info("Live Runner Started. Monitoring XAUUSD M30...")
        try:
            while True:
                self.run_once()
                time.sleep(60) # Wait 1 minute between checks
        except KeyboardInterrupt:
            execution_logger.info("Manual shutdown triggered.")
            self.connector.close()

if __name__ == "__main__":
    runner = LiveRunner()
    runner.start()
