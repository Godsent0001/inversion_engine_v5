"""Standalone M5 ZigZag portfolio runner.

Order submission is controlled by LIVE_TRADING_ENABLED in zigzag_portfolio/settings.py.
The connector uses MT5 credentials loaded from live_agent_engine/.env.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

from zigzag_portfolio.mt5_connection import ZigZagMT5Connection
from zigzag_portfolio import settings
from zigzag_portfolio.risk_state import PersistentRiskReference
from zigzag_portfolio.strategy import (
    detect_confirmed_swings,
    in_dynamic_gold_fill_window,
    in_dynamic_weekly_exit_window,
    signal_from_swing,
)

log = logging.getLogger("zigzag_portfolio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ZigZagPortfolioRunner:
    def __init__(self) -> None:
        self.connector = ZigZagMT5Connection()
        self.state_path = Path(__file__).resolve().parents[1] / settings.STATE_FILE
        self.risk_path = Path(__file__).resolve().parents[1] / settings.RISK_STATE_FILE
        self.risk_reference = PersistentRiskReference(self.risk_path)
        self.state = self._load_state()
        self.symbols: dict[str, str] = {}

    def _load_state(self) -> dict:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"last_swing": {}, "symbol_map": {}}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.state_path)

    def _resolve_symbol(self, candidates: tuple[str, ...]) -> str | None:
        for symbol in candidates:
            info = mt5.symbol_info(symbol)
            if info is not None:
                if not info.visible and not mt5.symbol_select(symbol, True):
                    continue
                return symbol
        return None

    def _resolve_symbols(self) -> bool:
        for asset, candidates in settings.SYMBOL_CANDIDATES.items():
            symbol = self._resolve_symbol(candidates)
            if symbol is None:
                log.error("No available MT5 symbol among configured candidates for %s", asset)
                return False
            self.symbols[asset] = symbol
        self.state["symbol_map"] = self.symbols
        self._save_state()
        return True

    @staticmethod
    def _volume_for_risk(symbol: str, side: str, entry: float, stop: float, risk_cash: float) -> float:
        info = mt5.symbol_info(symbol)
        if info is None or risk_cash <= 0:
            return 0.0
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        profit = mt5.order_calc_profit(order_type, symbol, 1.0, entry, stop)
        if profit is None or not math.isfinite(profit) or abs(profit) <= 0:
            log.error("Cannot calculate broker-native stop loss value for %s", symbol)
            return 0.0
        raw_volume = risk_cash / abs(profit)
        step = float(info.volume_step)
        minimum = float(info.volume_min)
        maximum = float(info.volume_max)
        if step <= 0 or minimum <= 0 or maximum < minimum:
            return 0.0
        volume = math.floor(raw_volume / step) * step
        volume = round(volume, 8)
        # Never round up to the broker minimum: that would exceed the risk budget.
        if volume < minimum:
            return 0.0
        volume = min(volume, maximum)
        return volume

    def _pending_orders_for_side(self, symbol: str, side: str):
        """Return this strategy's pending orders for one side; never inspect open positions."""
        orders = mt5.orders_get(symbol=symbol)
        if orders is None:
            log.error("Could not inspect pending orders for %s: %s", symbol, mt5.last_error())
            return None
        pending_type = mt5.ORDER_TYPE_BUY_STOP if side == "buy" else mt5.ORDER_TYPE_SELL_STOP
        return [
            order for order in orders
            if order.magic == settings.MAGIC and order.type == pending_type
        ]

    def _remove_pending_orders(self, symbol: str, side: str | None = None) -> bool:
        """Remove this strategy's pending orders; optionally limit removal to one side."""
        orders = mt5.orders_get(symbol=symbol)
        if orders is None:
            log.error("Could not inspect pending orders for %s: %s", symbol, mt5.last_error())
            return False
        pending_types = {
            mt5.ORDER_TYPE_BUY_STOP: "buy",
            mt5.ORDER_TYPE_SELL_STOP: "sell",
        }
        success = True
        for order in orders:
            if order.magic != settings.MAGIC or order.type not in pending_types:
                continue
            if side is not None and pending_types[order.type] != side:
                continue
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "symbol": symbol,
                "magic": settings.MAGIC,
                "comment": "ZZ replace",
            })
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(
                    "Could not remove %s pending order %s: %s",
                    pending_types[order.type], order.ticket,
                    mt5.last_error() if result is None else result.comment,
                )
                success = False
            else:
                log.info("Removed %s pending order ticket=%s", pending_types[order.type], order.ticket)
        return success

    def _place_signal(self, asset: str, symbol: str, signal, balance: float, reference: float) -> None:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None or info.point <= 0:
            log.error("Missing symbol/tick metadata for %s", symbol)
            return
        entry = round(signal.entry / info.point) * info.point
        stop = round(signal.stop_loss / info.point) * info.point
        target = round(signal.take_profit / info.point) * info.point
        if signal.side == "buy" and entry <= tick.ask:
            log.info("Buy-stop level is not above current ask for %s; skipping stale swing", symbol)
            return
        if signal.side == "sell" and entry >= tick.bid:
            log.info("Sell-stop level is not below current bid for %s; skipping stale swing", symbol)
            return
        risk_pct = settings.GOLD_RISK_PERCENT if asset == "gold" else settings.OIL_RISK_PERCENT
        risk_cash = reference * risk_pct
        volume = self._volume_for_risk(symbol, signal.side, entry, stop, risk_cash)
        if volume <= 0:
            log.warning("Volume is below broker minimum or risk calculation failed; skipping %s", symbol)
            return
        order_type = mt5.ORDER_TYPE_BUY_STOP if signal.side == "buy" else mt5.ORDER_TYPE_SELL_STOP

        # A new qualifying swing supersedes the old pending order on its
        # side. Remove the old order first, so a broker rejection of the new
        # request cannot leave the superseded order active. The opposite side
        # is untouched.
        if not self._remove_pending_orders(symbol, signal.side):
            log.error("Could not clear old %s pending order(s); refusing replacement.", signal.side)
            return

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": entry,
            "sl": stop,
            "tp": target,
            "deviation": 20,
            "magic": settings.MAGIC,
            "comment": "ZigZag M5 portfolio",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        margin = mt5.order_calc_margin(order_type, symbol, volume, entry)
        account = mt5.account_info()
        if margin is None or account is None or margin > float(account.margin_free):
            log.warning("Insufficient/unknown free margin for %s; skipping order", symbol)
            return
        log.info(
            "%s %s | balance=%.2f active_reference=%.2f risk_cash=%.2f volume=%.4f entry=%.*f SL=%.*f TP=%.*f",
            asset, signal.side, balance, reference, risk_cash, volume,
            info.digits, entry, info.digits, stop, info.digits, target,
        )
        if not settings.LIVE_TRADING_ENABLED:
            log.info("[DRY RUN] Pending order not sent.")
            return
        result = mt5.order_send(request)
        if result is None or result.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            log.error("Pending order rejected: %s", mt5.last_error() if result is None else result.comment)
            return
        log.info("Pending %s accepted: ticket=%s", signal.side, result.order)
        if not result.order:
            log.error("Accepted pending order returned no ticket; no older same-side order remains.")

    def _close_gold_positions_opened_outside_window(self, times: list[int], now_timestamp: int) -> None:
        """Close a Gold position immediately if a pending order raced past session cleanup."""
        positions = mt5.positions_get()
        if positions is None:
            log.error("Could not inspect positions for off-session Gold fills.")
            return
        for position in positions:
            if position.magic != settings.MAGIC or position.symbol != self.symbols.get("gold"):
                continue
            if in_dynamic_gold_fill_window(
                times,
                int(position.time),
                wait_minutes=60,
                session_end_hour_utc=settings.GOLD_SESSION_END_UTC,
            ):
                continue
            info = mt5.symbol_info(position.symbol)
            tick = mt5.symbol_info_tick(position.symbol)
            if info is None or tick is None:
                log.error("Cannot close off-session Gold position %s: missing symbol/tick data.", position.ticket)
                continue
            is_buy = position.type == mt5.POSITION_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "position": position.ticket,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if is_buy else tick.ask,
                "deviation": 20,
                "magic": settings.MAGIC,
                "comment": "ZZ off-session cleanup",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            if not settings.LIVE_TRADING_ENABLED:
                log.warning("[DRY RUN] Would close off-session Gold position ticket=%s", position.ticket)
                continue
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error(
                    "Failed to close off-session Gold position %s: %s",
                    position.ticket, mt5.last_error() if result is None else result.comment,
                )
            else:
                log.warning("Closed Gold position %s opened outside the allowed fill window.", position.ticket)

    def _process_symbol(self, asset: str, symbol: str, balance: float, reference: float) -> None:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, settings.FETCH_BARS)
        if rates is None or len(rates) < settings.DEPTH + settings.BACKSTEP + 5:
            log.warning("Insufficient M5 bars for %s", symbol)
            return
        # Drop the still-forming candle; the strategy acts only on completed bars.
        closed = rates[:-1]
        highs = [float(row["high"]) for row in closed]
        lows = [float(row["low"]) for row in closed]
        times = [int(row["time"]) for row in closed]
        info = mt5.symbol_info(symbol)
        if info is None:
            return
        swings = detect_confirmed_swings(
            highs, lows, times, settings.DEPTH, settings.DEVIATION_POINTS,
            float(info.point), settings.BACKSTEP, repainting=True,
        )
        gold_fill_allowed = True
        if asset == "gold":
            # Use wall-clock UTC, not the last completed bar's open time, so
            # the 08:00 UTC cutoff is enforced without a five-minute delay.
            now_timestamp = int(datetime.now(timezone.utc).timestamp())
            gold_fill_allowed = in_dynamic_gold_fill_window(
                times,
                now_timestamp,
                wait_minutes=60,
                session_end_hour_utc=settings.GOLD_SESSION_END_UTC,
            )
            if not gold_fill_allowed:
                # GTC orders must not survive outside Gold's permitted fill window.
                if not self._remove_pending_orders(symbol):
                    log.error("One or more Gold pending orders could not be cancelled outside session.")
                self._close_gold_positions_opened_outside_window(times, now_timestamp)
        if not swings:
            return
        # Process unseen swing events in event-time order, as the strategy
        # runs bar by bar. Persistent timestamps prevent replaying events
        # already handled before a restart.
        last_swing = self.state.setdefault("last_swing", {})
        for swing in sorted(
            swings,
            key=lambda item: int(item.event_timestamp or item.timestamp),
        ):
            event_timestamp = int(swing.event_timestamp or swing.timestamp)
            key = f"{asset}:{swing.kind}"
            if int(last_swing.get(key, 0)) >= event_timestamp:
                continue
            if asset == "gold" and not gold_fill_allowed:
                log.info(
                    "Gold is outside its permitted fill window; keeping %s swing unprocessed "
                    "for evaluation when the window opens.",
                    swing.kind,
                )
                continue
            signal = signal_from_swing(swing, float(info.point), settings.REWARD_RISK)
            self._place_signal(asset, symbol, signal, balance, reference)
            # Record this event after the placement attempt so a rejected or
            # stale signal is not retried every five-second polling cycle.
            last_swing[key] = event_timestamp
            self._save_state()

    def _weekly_exit(self) -> bool:
        # Infer the next close from the most recent weekend-sized gap in MT5 bars.
        # Fetch enough history to retain the previous weekend gap during Friday.
        times: list[int] = []
        for symbol in self.symbols.values():
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, settings.FETCH_BARS)
            if rates is not None and len(rates) >= 2:
                times = [int(row["time"]) for row in rates[:-1]]
                if times:
                    break
        if not times:
            log.error("Cannot infer weekly close: no usable M5 history.")
            return False
        now_timestamp = int(datetime.now(timezone.utc).timestamp())
        if not in_dynamic_weekly_exit_window(
            times,
            now_timestamp,
            exit_minutes_before_close=settings.EXIT_MINUTES_BEFORE_WEEKLY_CLOSE,
            timeframe_minutes=5,
            gap_hours=settings.WEEKEND_GAP_HOURS,
        ):
            return False
        log.warning("Dynamic weekly exit window detected from observed weekend market gap.")
        positions = mt5.positions_get()
        if positions is None:
            log.error("Could not inspect positions during weekly exit window.")
            return True
        for position in positions:
            if position.magic != settings.MAGIC:
                continue
            info = mt5.symbol_info(position.symbol)
            tick = mt5.symbol_info_tick(position.symbol)
            if info is None or tick is None:
                continue
            is_buy = position.type == mt5.POSITION_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "position": position.ticket,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if is_buy else tick.ask,
                "deviation": 20,
                "magic": settings.MAGIC,
                "comment": "ZigZag weekly exit",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            if settings.LIVE_TRADING_ENABLED:
                result = mt5.order_send(request)
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    log.error("Weekly close failed for ticket %s", position.ticket)
            else:
                log.info("[DRY RUN] Would close position ticket=%s", position.ticket)
        return True

    def run_once(self) -> None:
        if not self.connector.ensure_connection():
            return
        account = mt5.account_info()
        if account is None or float(account.balance) <= 0:
            log.error("Live MT5 balance unavailable; refusing all entry processing.")
            return
        if not self.symbols and not self._resolve_symbols():
            return
        balance = float(account.balance)
        try:
            risk_state = self.risk_reference.update(balance)
        except Exception:
            log.exception("Risk reference state failed; refusing new entries.")
            return
        if self._weekly_exit():
            return
        for asset, symbol in self.symbols.items():
            try:
                self._process_symbol(asset, symbol, balance, risk_state.active_reference)
            except Exception:
                log.exception("Failed processing %s", symbol)

    def start(self) -> None:
        if not self.connector.connect():
            raise RuntimeError("MT5 connection failed.")
        if not self._resolve_symbols():
            raise RuntimeError("Could not resolve configured MT5 symbols.")
        mode = "LIVE" if settings.LIVE_TRADING_ENABLED else "DRY RUN"
        log.warning("ZigZag M5 portfolio runner started in %s mode.", mode)
        while True:
            self.run_once()
            time.sleep(settings.POLL_SECONDS)


if __name__ == "__main__":
    ZigZagPortfolioRunner().start()
