import time
import pandas as pd
import numpy as np
from datetime import datetime

from config import settings
from data.feed import MT5DataFeed
from indicators.pipeline import build_features
from agents.agent_loader import AgentLoader
from agents.decision_engine import DecisionEngine
from agents.portfolio_manager import PortfolioManager
from execution.mt5_connector import MT5Connector
from execution.order_manager import OrderManager
from execution.position_router import PositionRouter
from risk.risk_engine import RiskEngine

class LiveRunner:
    def __init__(self):
        self.connector = MT5Connector()
        self.loader = AgentLoader()
        self.decision_engine = DecisionEngine()
        self.router = PositionRouter()
        self.risk_engine = RiskEngine(risk_per_trade=settings.RISK_PER_TRADE)

        self.agents = self.loader.load_agents()
        self.portfolio = PortfolioManager([a["id"] for a in self.agents])
        self.order_manager = OrderManager(self.connector, slippage=settings.SLIPPAGE)

        self.last_candle_time = None

    def run_once(self):
        print(f"[{datetime.now()}] Checking for new candle...")

        df = self.connector.get_latest_data(settings.SYMBOL, settings.TIMEFRAME, n_bars=200)
        if df is None or len(df) < 50:
            print("Failed to get data")
            return

        current_candle_time = df.iloc[-1]['time']

        # We trade at the OPEN of a new candle
        if self.last_candle_time is not None and current_candle_time == self.last_candle_time:
            return

        print(f"New candle detected: {datetime.fromtimestamp(current_candle_time)}")
        self.last_candle_time = current_candle_time

        # Prepare data for indicators
        # build_features expects high, low, close as numpy arrays
        high = df['high'].values.astype(np.float32)
        low = df['low'].values.astype(np.float32)
        close = df['close'].values.astype(np.float32)

        features_full, atr_full = build_features(high, low, close)

        # Latest features (from the candle that just closed)
        latest_features = features_full[-1]
        latest_atr = atr_full[-1]
        current_price = df.iloc[-1]['open'] # We execute at the open of the current (new) candle

        # 1. MONITOR CLOSED POSITIONS (Update Equity & Cooldown)
        # We check MT5 history or current positions to see what changed
        # For simplicity in this implementation, we compare current MT5 positions
        # with our internal "tracked" positions.

        # NOTE: In a production environment, you would use mt5.history_deals_get
        # to get actual fill prices and PnL.

        # 2. PROCESS EACH AGENT
        for agent in self.agents:
            agent_id = agent["id"]
            has_pos = self.router.has_open_position(agent_id)

            # Check if a position was just closed (primitive tracking)
            # You might want a dedicated 'monitoring/agent_tracker.py' for this

            # 3. Check if agent is already in a position
            if has_pos:
                continue

            # 4. Check cooldown
            if self.portfolio.portfolios[str(agent_id)]["cooldown"] > 0:
                continue

            # 3. Get decision
            action, confidence = self.decision_engine.decide(agent, latest_features)

            if action != 0:
                print(f"Agent {agent_id} signaled {action} (conf: {confidence:.2f})")

                # 4. Risk & Order Params
                equity = self.portfolio.get_equity(agent_id)

                # SL/TP calculation (same as simulation)
                dist = latest_atr * agent["atr"]
                if action == 1:
                    sl = current_price - dist
                    tp = current_price + dist * agent["rrr"]
                else:
                    sl = current_price + dist
                    tp = current_price - dist * agent["rrr"]

                lots = self.risk_engine.calculate_lot_size(settings.SYMBOL, equity, current_price, sl)

                # 5. Execute
                request = self.order_manager.build_request(
                    agent_id, settings.SYMBOL, action, lots, current_price, sl, tp,
                    comment=f"Agent {agent_id}"
                )

                result = self.order_manager.execute(request)
                if result.retcode == 10009: # Request completed
                    print(f"Agent {agent_id} trade executed successfully")
                else:
                    print(f"Agent {agent_id} trade failed: {result.comment}")

        # Decrement cooldowns at each bar
        self.portfolio.decrement_cooldowns()

    def start(self):
        if not self.connector.connect():
            return

        print("Live Runner Started. Waiting for M30 candles...")
        try:
            while True:
                self.run_once()
                # Check every minute
                time.sleep(60)
        except KeyboardInterrupt:
            print("Shutting down...")
            self.connector.close()

if __name__ == "__main__":
    runner = LiveRunner()
    runner.start()
