import time
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import settings

# Ensure inversion_engine_v5 root is accessible
base_engine_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if base_engine_dir not in sys.path:
    sys.path.append(base_engine_dir)

from research.core.feature_engine_fast import build_all_features_fast
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

        # Track 2-hour window locking per agent: agent_id -> last_traded_window_id
        self.last_traded_window = {}

    def _close_all_positions_for_weekend(self):
        """Closes all active open positions before Friday market close."""
        for agent_id in self.agent_ids:
            if self.router.has_open_position(agent_id):
                execution_logger.info(f"Friday exit rule: Closing position for Agent {agent_id}")
                self.order_manager.close_position(agent_id, settings.SYMBOL)

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

            fetch_n = getattr(settings, "FETCH_BARS", 350)
            df = self.connector.get_latest_data(
                settings.SYMBOL,
                settings.TIMEFRAME,
                n_bars=fetch_n
            )

            if df is None or len(df) < 250:
                return

            current_candle_time = df.iloc[-1]["time"]

            if self.last_candle_time == current_candle_time:
                return

            self.last_candle_time = current_candle_time

            dt_utc = datetime.fromtimestamp(current_candle_time, tz=timezone.utc)
            execution_logger.info(f"New candle detected: {dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # =========================
            # FRIDAY WEEKEND EXIT RULE
            # =========================
            # Friday = day 4 (Monday=0). Close open positions & suppress new entries starting at 20:00 UTC (1hr before close)
            friday_close_hour = getattr(settings, "FRIDAY_CLOSE_HOUR_GMT", 20)
            if dt_utc.weekday() == 4 and dt_utc.hour >= friday_close_hour:
                execution_logger.info(f"Friday restriction active ({dt_utc.strftime('%H:%M UTC')}). Closing positions and skipping entries.")
                self._close_all_positions_for_weekend()
                return

            # Closed bars evaluation: use df.iloc[:-1] (completed candles up to candle t)
            df_closed = df.iloc[:-1].copy()

            if not pd.api.types.is_datetime64_any_dtype(df_closed['time']):
                df_closed['time'] = pd.to_datetime(df_closed['time'], unit='s')

            # Calculate ATR14 from closed bars
            highs = df_closed["high"].values.astype(np.float64)
            lows = df_closed["low"].values.astype(np.float64)
            atrs_rolling = pd.Series(highs - lows).rolling(getattr(settings, "ATR_PERIOD", 14)).mean().ffill().bfill().values
            latest_atr = float(atrs_rolling[-1])

            # Build full 328 feature matrix using exact research feature engine
            X_np, cleaned_df = build_all_features_fast(df_closed)

            # Determine 2-hour window ID of current completed bar
            current_bar_time = cleaned_df.iloc[-1]['time']
            window_id = int((current_bar_time.floor('2h') - pd.Timestamp('1970-01-01')).total_seconds() // 7200)

            # Entry price is Open of the new current bar (t+1)
            current_price = df.iloc[-1]["open"]
            spread = self.connector.get_spread(settings.SYMBOL)

            execution_logger.info(
                f"Market | Entry Price: {current_price:.2f} | ATR: {latest_atr:.4f} | Spread: {spread}"
            )

            # =========================
            # AGENT LOOP
            # =========================
            for agent in self.agents:

                agent_id = agent["id"]

                # Enforce 2-hour window locking (max 1 trade per 2-hour window)
                if self.last_traded_window.get(agent_id) == window_id:
                    execution_logger.info(f"Agent {agent_id} already traded in 2H window {window_id}. Skipping.")
                    continue

                # Format input features for model: sequence length 10 -> shape (10, 328)
                seq_len = agent.get("seq_len", 10)
                if len(X_np) < seq_len:
                    continue

                latest_seq = X_np[-seq_len:]

                action, confidence = self.decision_engine.decide(
                    agent,
                    latest_seq
                )

                if action == 0:
                    continue

                equity = self.portfolio.get_equity(agent_id)

                # =========================
                # TRADE PARAMETERS PARITY
                # =========================
                rrr_used = float(agent["rrr"])        # 2.0
                atr_mult_used = float(agent["atr"])   # 3.6

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

                    self.last_traded_window[agent_id] = window_id

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