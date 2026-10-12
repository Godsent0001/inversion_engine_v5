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
    detect_gold_reopen_timestamp,
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
        self.state.setdefault("shadow_book", {})
        self.symbols: dict[str, str] = {}

    def _load_state(self) -> dict:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"last_swing": {}, "symbol_map": {}, "shadow_book": {}}

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

        # Round DOWN to the nearest broker-valid volume to avoid exceeding
        # the configured cash-risk budget. If that volume is below the broker
        # minimum, skip the trade rather than forcing a larger risky volume.
        volume = math.floor(raw_volume / step + 1e-12) * step
        volume = round(volume, 8)
        if volume < minimum:
            log.warning(
                "Risk-derived volume %.8f for %s is below broker minimum %.8f; "
                "skipping trade to preserve risk limits.",
                raw_volume, symbol, minimum,
            )
            return 0.0
        return min(volume, maximum)

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

    def _place_signal(self, asset: str, symbol: str, signal, balance: float, reference: float) -> str:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None or info.point <= 0:
            log.error("Missing symbol/tick metadata for %s", symbol)
            return "retry"
        entry = round(signal.entry / info.point) * info.point
        stop = round(signal.stop_loss / info.point) * info.point
        target = round(signal.take_profit / info.point) * info.point
        if signal.side == "buy" and entry <= tick.ask:
            log.info("Buy-stop level is not above current ask for %s; skipping stale swing", symbol)
            return "stale"
        if signal.side == "sell" and entry >= tick.bid:
            log.info("Sell-stop level is not below current bid for %s; skipping stale swing", symbol)
            return "stale"
        risk_pct = settings.GOLD_RISK_PERCENT if asset == "gold" else settings.OIL_RISK_PERCENT
        risk_cash = reference * risk_pct
        volume = self._volume_for_risk(symbol, signal.side, entry, stop, risk_cash)
        if volume <= 0:
            log.warning("Volume is below broker minimum or risk calculation failed; skipping %s", symbol)
            return "retry"
        order_type = mt5.ORDER_TYPE_BUY_STOP if signal.side == "buy" else mt5.ORDER_TYPE_SELL_STOP

        # Snapshot old same-side orders. The replacement is sent first; only
        # after MT5 accepts it do we cancel the superseded order(s). The
        # opposite side is never affected.
        old_orders = self._pending_orders_for_side(symbol, signal.side)
        if old_orders is None:
            log.error("Could not inspect existing %s orders; refusing replacement.", signal.side)
            return "retry"

        # Preserve the best existing entry. A BUY STOP may only be replaced by
        # a strictly lower entry; a SELL STOP may only be replaced by a
        # strictly higher entry. Open positions and the opposite side are not
        # considered here, so they never block an independent pending setup.
        if old_orders:
            old_prices = [float(order.price_open) for order in old_orders]
            best_existing = min(old_prices) if signal.side == "buy" else max(old_prices)
            improves_existing = entry < best_existing if signal.side == "buy" else entry > best_existing
            if not improves_existing:
                log.info(
                    "Keeping better existing %s pending order for %s: existing_entry=%.*f, "
                    "new_entry=%.*f; new setup not submitted.",
                    signal.side, symbol, info.digits, best_existing, info.digits, entry,
                )
                return "superseded"

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
            return "retry"
        log.info(
            "%s %s | balance=%.2f active_reference=%.2f risk_cash=%.2f volume=%.4f entry=%.*f SL=%.*f TP=%.*f",
            asset, signal.side, balance, reference, risk_cash, volume,
            info.digits, entry, info.digits, stop, info.digits, target,
        )
        if not settings.LIVE_TRADING_ENABLED:
            log.info("[DRY RUN] Pending order not sent.")
            return "dry_run"
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            result = mt5.order_send(request)
            if result is not None and result.retcode in (
                mt5.TRADE_RETCODE_DONE,
                mt5.TRADE_RETCODE_PLACED,
            ):
                log.info(
                    "Pending %s accepted on attempt %s/%s: ticket=%s",
                    signal.side, attempt, max_attempts, result.order,
                )
                new_ticket = int(result.order or 0)
                if new_ticket:
                    # Persist the broker ticket so a restart can distinguish a
                    # manually cancelled order from an order that actually filled.
                    # Older state files may not have this field; those are handled
                    # conservatively by checking current positions and entry validity.
                    self._shadow_for(asset)[signal.side]["order_ticket"] = new_ticket
                for old_order in old_orders:
                    if new_ticket and int(old_order.ticket) == new_ticket:
                        continue
                    removed = mt5.order_send({
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": old_order.ticket,
                        "symbol": symbol,
                        "magic": settings.MAGIC,
                        "comment": "ZZ replaced after acceptance",
                    })
                    if removed is None or removed.retcode != mt5.TRADE_RETCODE_DONE:
                        log.error(
                            "New %s order accepted, but old order %s could not be cancelled: %s",
                            signal.side, old_order.ticket,
                            mt5.last_error() if removed is None else removed.comment,
                        )
                if not new_ticket:
                    log.error("Accepted pending order returned no ticket; verify broker order list.")
                return "accepted"

            reason = mt5.last_error() if result is None else (
                f"retcode={result.retcode}, comment={result.comment}"
            )
            if attempt < max_attempts:
                log.warning(
                    "Pending %s rejected on attempt %s/%s for %s: %s; retrying in 1 second.",
                    signal.side, attempt, max_attempts, symbol, reason,
                )
                time.sleep(1)
            else:
                log.error(
                    "Pending %s rejected after %s attempts for %s: %s. "
                    "The existing same-side order was retained.",
                    signal.side, max_attempts, symbol, reason,
                )
        return "retry"

    def _close_gold_positions_opened_outside_window(self, times: list[int], now_timestamp: int) -> None:
        """Close a Gold position immediately if a pending order raced past session cleanup."""
        positions = mt5.positions_get()
        if positions is None:
            log.error("Could not inspect positions for off-session Gold fills.")
            return
        reopen_timestamp = detect_gold_reopen_timestamp(times)
        if reopen_timestamp is None:
            return
        for position in positions:
            if position.magic != settings.MAGIC or position.symbol != self.symbols.get("gold"):
                continue
            # Do not close a legitimate position carried over from a previous
            # permitted session. Only inspect fills after this session's reopen.
            if int(position.time) < reopen_timestamp:
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

    @staticmethod
    def _shadow_status(order) -> str:
        if not order:
            return "none"
        return (
            f"side={order.get('side')} entry={order.get('entry')} "
            f"submitted={order.get('submitted', False)} consumed={order.get('consumed', False)}"
        )

    def _shadow_for(self, asset: str) -> dict:
        book = self.state.setdefault("shadow_book", {})
        return book.setdefault(asset, {"buy": None, "sell": None})

    def _remove_side_if_present(self, symbol: str, side: str) -> None:
        """Best-effort remove broker pending orders for a consumed shadow signal."""
        orders = self._pending_orders_for_side(symbol, side)
        if orders is None:
            return
        for order in orders:
            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "symbol": symbol,
                "magic": settings.MAGIC,
                "comment": "ZZ shadow consumed",
            })
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                log.error("Could not remove consumed %s order %s", side, order.ticket)

    def _store_shadow_signal(self, asset: str, symbol: str, signal, gold_fill_allowed: bool) -> None:
        """Replace only the same-side shadow order, preserving the opposite side."""
        book = self._shadow_for(asset)
        previous = book.get(signal.side)
        # Compare unsubmitted shadows only. Submitted shadows are checked against
        # actual broker orders in _place_signal; consumed shadows are fresh history.
        if previous and not previous.get("submitted") and not previous.get("consumed"):
            previous_entry = float(previous.get("entry", signal.entry))
            improves_shadow = (
                float(signal.entry) < previous_entry
                if signal.side == "buy"
                else float(signal.entry) > previous_entry
            )
            if not improves_shadow:
                log.info(
                    "Keeping better shadow %s setup for %s: existing_entry=%.8f, new_entry=%.8f.",
                    signal.side, symbol, previous_entry, float(signal.entry),
                )
                return
        if previous and previous.get("submitted") and asset == "gold" and not gold_fill_allowed:
            # Outside the permitted Gold session, the superseded broker order
            # must not remain executable while its replacement waits in shadow.
            self._remove_side_if_present(symbol, signal.side)
        book[signal.side] = {
            "side": signal.side,
            "entry": float(signal.entry),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit),
            "swing_timestamp": int(signal.swing_timestamp),
            "last_checked_timestamp": int(signal.swing_timestamp),
            "submitted": False,
            "consumed": False,
        }
        self._save_state()

    def _advance_shadow_orders(
        self, asset: str, symbol: str, closed, times: list[int],
        gold_fill_allowed: bool, balance: float, reference: float,
    ) -> None:
        """Advance shadow orders across every completed bar, regardless of session.

        A shadow order touched while not submitted is consumed, not resurrected
        at the next session. Untouched orders are sent to MT5 when entry is
        allowed. Once submitted, MT5 owns the actual fill lifecycle.
        """
        book = self._shadow_for(asset)
        changed = False
        for side in ("buy", "sell"):
            order = book.get(side)
            if not order or order.get("consumed"):
                continue

            # Reconcile persisted submissions against MT5. A missing pending order
            # may have been manually cancelled, so do not permanently consume the
            # setup just because it is absent from orders_get(). Use its saved ticket
            # to detect a completed fill; otherwise re-arm it, subject to current
            # position, session, risk, and entry-price checks below.
            if order.get("submitted"):
                broker_orders = self._pending_orders_for_side(symbol, side)
                if broker_orders is None:
                    continue
                if broker_orders:
                    log.info(
                        "%s %s shadow remains submitted: broker pending orders=%s",
                        asset, side,
                        [(int(item.ticket), float(item.price_open)) for item in broker_orders],
                    )
                    continue

                order_ticket = int(order.get("order_ticket") or 0)
                if order_ticket:
                    try:
                        history_orders = mt5.history_orders_get(ticket=order_ticket)
                    except (AttributeError, TypeError):
                        history_orders = None
                    if history_orders and any(
                        int(getattr(history_order, "ticket", 0)) == order_ticket
                        and getattr(history_order, "state", None) == mt5.ORDER_STATE_FILLED
                        for history_order in history_orders
                    ):
                        order["consumed"] = True
                        changed = True
                        log.info(
                            "%s %s setup ticket=%s was filled; not re-arming it.",
                            asset, side, order_ticket,
                        )
                        continue

                positions = mt5.positions_get(symbol=symbol)
                if positions is None:
                    log.error("Could not inspect open positions for %s; not re-arming %s.", symbol, side)
                    continue
                position_type = mt5.POSITION_TYPE_BUY if side == "buy" else mt5.POSITION_TYPE_SELL
                same_side_positions = [
                    position for position in positions
                    if position.magic == settings.MAGIC and position.type == position_type
                ]
                if same_side_positions:
                    order["consumed"] = True
                    changed = True
                    log.info(
                        "%s %s setup has an existing strategy position; not re-arming.",
                        asset, side,
                    )
                    continue

                order["submitted"] = False
                order.pop("order_ticket", None)
                # The setup is being reconciled now. Do not replay historical bars
                # from before this restart/cancellation; _place_signal() still checks
                # the live tick and refuses an already-crossed pending entry.
                if closed:
                    order["last_checked_timestamp"] = int(closed[-1]["time"])
                changed = True
                log.warning(
                    "%s %s pending order is missing from MT5; re-arming saved setup "
                    "if its entry is still valid.",
                    asset, side,
                )

            entry = float(order["entry"])
            created = int(order["swing_timestamp"])
            checked = int(order.get("last_checked_timestamp", created))
            touched_at = None
            for row in closed:
                bar_time = int(row["time"])
                if bar_time <= max(created, checked):
                    continue
                touched = float(row["high"]) >= entry if side == "buy" else float(row["low"]) <= entry
                if touched:
                    touched_at = bar_time
                    break
                checked = bar_time

            if touched_at is not None:
                order["last_checked_timestamp"] = touched_at
                order["consumed"] = True
                changed = True
                # This signal superseded any older same-side order. If the new
                # shadow signal was touched before submission, do not leave the
                # old broker order active as an accidental substitute.
                if settings.LIVE_TRADING_ENABLED:
                    self._remove_side_if_present(symbol, side)
                log.info(
                    "%s shadow %s at %.8f was touched on %s without a live order; consumed.",
                    asset, side, entry, datetime.fromtimestamp(touched_at, timezone.utc).isoformat(),
                )
                continue

            if checked != int(order.get("last_checked_timestamp", created)):
                order["last_checked_timestamp"] = checked
                changed = True

            if asset == "gold" and not gold_fill_allowed:
                log.info(
                    "Gold %s setup deferred by session window: entry=%.8f "
                    "submitted=%s consumed=%s",
                    side, entry, order.get("submitted", False), order.get("consumed", False),
                )
                continue

            from zigzag_portfolio.strategy import PendingSignal
            signal = PendingSignal(
                side=side,
                entry=entry,
                stop_loss=float(order["stop_loss"]),
                take_profit=float(order["take_profit"]),
                swing_timestamp=created,
            )
            log.info(
                "Attempting %s %s shadow placement: entry=%.8f SL=%.8f TP=%.8f",
                asset, side, entry, float(order["stop_loss"]), float(order["take_profit"]),
            )
            placement_status = self._place_signal(asset, symbol, signal, balance, reference)
            log.info("%s %s shadow placement result: %s", asset, side, placement_status)
            # Mark submitted only when MT5 accepted this specific submission;
            # an older same-side pending order is not proof of acceptance.
            if placement_status == "accepted":
                order["submitted"] = True
                changed = True
            elif placement_status in ("superseded", "stale"):
                order["consumed"] = True
                changed = True
            # In dry-run mode, retain the shadow order for inspection and replay.
        if changed:
            self._save_state()

    def _process_symbol(self, asset: str, symbol: str, balance: float, reference: float) -> None:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, settings.FETCH_BARS)
        if rates is None:
            log.error("M5 history request failed for %s: %s", symbol, mt5.last_error())
            return
        if len(rates) < settings.DEPTH + settings.BACKSTEP + 5:
            log.warning(
                "Insufficient M5 bars for %s: received=%s required_at_least=%s last_error=%s",
                symbol, len(rates), settings.DEPTH + settings.BACKSTEP + 5, mt5.last_error(),
            )
            return
        log.info("DEBUG cycle: %s/%s fetched %s M5 bars.", asset, symbol, len(rates))
        # Drop the forming candle; process every completed bar in order.
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
        last_swing_state = self.state.setdefault("last_swing", {})
        unseen_swings = [
            swing for swing in swings
            if int(last_swing_state.get(f"{asset}:{swing.kind}", 0))
            < int(swing.event_timestamp or swing.timestamp)
        ]
        shadow_book = self._shadow_for(asset)
        log.info(
            "DEBUG cycle: %s/%s closed_bars=%s detected_swings=%s unseen_swings=%s "
            "shadow_buy=%s shadow_sell=%s",
            asset, symbol, len(closed), len(swings), len(unseen_swings),
            self._shadow_status(shadow_book.get("buy")),
            self._shadow_status(shadow_book.get("sell")),
        )
        now_timestamp = int(datetime.now(timezone.utc).timestamp())
        gold_fill_allowed = True
        if asset == "gold":
            gold_fill_allowed = in_dynamic_gold_fill_window(
                times, now_timestamp, wait_minutes=60,
                session_end_hour_utc=settings.GOLD_SESSION_END_UTC,
            )
            log.info(
                "Gold session check: fill_allowed=%s current_utc=%s session_end_hour_utc=%s",
                gold_fill_allowed, datetime.now(timezone.utc).isoformat(),
                settings.GOLD_SESSION_END_UTC,
            )
        # Every new signal updates only its own side's shadow order. Signals are
        # recorded even outside the Gold window, because the backtest order book
        # exists outside the permitted fill session too.
        last_swing = self.state.setdefault("last_swing", {})
        for swing in sorted(swings, key=lambda item: int(item.event_timestamp or item.timestamp)):
            event_timestamp = int(swing.event_timestamp or swing.timestamp)
            key = f"{asset}:{swing.kind}"
            if int(last_swing.get(key, 0)) >= event_timestamp:
                continue
            signal = signal_from_swing(swing, float(info.point), settings.REWARD_RISK)
            log.info(
                "Confirmed %s swing for %s: side=%s entry=%.8f event_age=%.1fs",
                swing.kind, symbol, signal.side, float(signal.entry),
                max(0.0, datetime.now(timezone.utc).timestamp() - event_timestamp),
            )
            self._store_shadow_signal(asset, symbol, signal, gold_fill_allowed)
            last_swing[key] = event_timestamp
            self._save_state()

        self._advance_shadow_orders(
            asset, symbol, closed, times, gold_fill_allowed, balance, reference
        )

    def _weekly_exit(self) -> bool:
        """Close strategy positions in each symbol's own observed weekly-close window.

        The Python MT5 API does not expose broker session calendars, so each
        symbol's schedule is inferred independently from its own M5 history.
        This is safer than using Gold's gap for UK Oil, but holiday/DST changes
        still require broker-session verification.
        """
        now_timestamp = int(datetime.now(timezone.utc).timestamp())
        exiting_symbols: set[str] = set()
        for symbol in set(self.symbols.values()):
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, settings.FETCH_BARS)
            if rates is None or len(rates) < 2:
                log.error("Cannot infer weekly close for %s: no usable M5 history.", symbol)
                continue
            times = [int(row["time"]) for row in rates[:-1]]
            if in_dynamic_weekly_exit_window(
                times, now_timestamp,
                exit_minutes_before_close=settings.EXIT_MINUTES_BEFORE_WEEKLY_CLOSE,
                timeframe_minutes=5, gap_hours=settings.WEEKEND_GAP_HOURS,
            ):
                exiting_symbols.add(symbol)
        if not exiting_symbols:
            return False

        log.warning("Weekly exit window detected for symbols: %s", ", ".join(sorted(exiting_symbols)))
        positions = mt5.positions_get()
        if positions is None:
            log.error("Could not inspect positions during weekly exit window.")
            return True
        for position in positions:
            if position.magic != settings.MAGIC or position.symbol not in exiting_symbols:
                continue
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                log.error("Cannot close weekly position %s: missing tick.", position.ticket)
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
        log.info("DEBUG cycle: polling MT5 for new ZigZag setups.")
        if not self.connector.ensure_connection():
            log.error("DEBUG cycle: MT5 connection is unavailable; skipping this cycle.")
            return
        account = mt5.account_info()
        if account is None or float(account.balance) <= 0:
            log.error("Live MT5 balance unavailable; refusing all entry processing.")
            return
        if not self.symbols and not self._resolve_symbols():
            log.error("DEBUG cycle: symbol resolution failed; skipping this cycle.")
            return
        balance = float(account.balance)
        log.info(
            "DEBUG cycle: balance=%.2f active_symbols=%s",
            balance, self.symbols,
        )
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
