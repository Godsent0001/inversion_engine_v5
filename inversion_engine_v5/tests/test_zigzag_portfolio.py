import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from zigzag_portfolio import runner as runner_module
from zigzag_portfolio.risk_state import PersistentRiskReference
from zigzag_portfolio.strategy import (
    Swing,
    detect_confirmed_swings,
    detect_gold_reopen_timestamp,
    detect_weekend_close_bar_timestamp,
    detect_weekend_reopen_timestamp,
    detect_daily_close_bar_timestamp,
    in_dynamic_daily_exit_window,
    in_dynamic_gold_fill_window,
    gold_fill_window_timestamps,
    in_dynamic_weekly_exit_window,
    in_gold_session,
    signal_from_swing,
)


class RiskReferenceTests(unittest.TestCase):
    def test_high_water_reference_does_not_fall_with_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PersistentRiskReference(Path(tmp) / "risk.json")
            self.assertEqual(manager.update(1000).active_reference, 1000)
            self.assertEqual(manager.update(850).active_reference, 1000)
            self.assertEqual(manager.update(999).active_reference, 1000)
            self.assertEqual(manager.update(1200).active_reference, 1200)

    def test_recursive_50_percent_resets_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk.json"
            manager = PersistentRiskReference(path)
            manager.update(1000)
            state = manager.update(500)
            self.assertEqual(state.active_reference, 500)
            self.assertEqual(state.reset_count, 1)
            state = manager.update(250)
            self.assertEqual(state.active_reference, 250)
            self.assertEqual(state.reset_count, 2)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["active_reference"], 250)
            self.assertEqual(raw["reset_count"], 2)

    def test_invalid_balance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = PersistentRiskReference(Path(tmp) / "risk.json")
            with self.assertRaises(ValueError):
                manager.update(0)


class StrategyTests(unittest.TestCase):
    def test_gold_risk_is_exactly_twice_oil_risk(self):
        from zigzag_portfolio import settings
        self.assertAlmostEqual(settings.GOLD_RISK_PERCENT, 2 * settings.OIL_RISK_PERCENT)


    def test_gold_session_crosses_midnight_utc(self):
        import datetime

        def stamp(hour, minute=0):
            return int(datetime.datetime(2026, 1, 5, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        self.assertTrue(in_gold_session(stamp(23)))
        self.assertTrue(in_gold_session(stamp(7, 59)))
        self.assertFalse(in_gold_session(stamp(8)))


    def test_gold_reopen_is_detected_from_gap_and_fill_window_is_dynamic(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        # 21:55 -> 23:00 is a 65-minute gap; 23:00 is the reopen bar.
        bars = [stamp(5, 21, 50), stamp(5, 21, 55), stamp(5, 23), stamp(5, 23, 5), stamp(6, 0)]
        self.assertEqual(detect_gold_reopen_timestamp(bars), stamp(5, 23))
        self.assertFalse(in_dynamic_gold_fill_window(bars, stamp(5, 23, 55)))
        self.assertTrue(in_dynamic_gold_fill_window(bars, stamp(6, 0)))
        self.assertTrue(in_dynamic_gold_fill_window(bars, stamp(6, 7, 55)))
        self.assertFalse(in_dynamic_gold_fill_window(bars, stamp(6, 8)))

    def test_gold_22_utc_reopen_window_ends_at_next_08_utc_not_23_utc(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        # A 22:00 reopen means the 60-minute wait ends at 23:00; the
        # permitted window continues until 08:00 the following day.
        bars = [stamp(5, 21, 10), stamp(5, 21, 15), stamp(5, 22), stamp(5, 22, 5)]
        self.assertEqual(
            gold_fill_window_timestamps(bars, stamp(5, 23)),
            (stamp(5, 23), stamp(6, 8)),
        )
        self.assertTrue(in_dynamic_gold_fill_window(bars, stamp(6, 7, 55)))
        self.assertFalse(in_dynamic_gold_fill_window(bars, stamp(6, 8)))

    def test_gold_reopen_requires_gap_and_22_or_23_utc_hour(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        no_gap = [stamp(5, 22), stamp(5, 22, 5), stamp(5, 22, 10)]
        wrong_hour = [stamp(5, 20, 55), stamp(5, 21, 50)]
        self.assertIsNone(detect_gold_reopen_timestamp(no_gap))
        self.assertIsNone(detect_gold_reopen_timestamp(wrong_hour))

    def test_weekend_open_close_and_dynamic_exit_window(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        # Friday 20:55 final bar, then Monday 00:00 first bar: 51h05m gap.
        bars = [stamp(9, 20, 45), stamp(9, 20, 50), stamp(9, 20, 55), stamp(12, 0), stamp(12, 0, 5)]
        self.assertEqual(detect_weekend_close_bar_timestamp(bars), stamp(9, 20, 55))
        self.assertEqual(detect_weekend_reopen_timestamp(bars), stamp(12, 0))
        # The next inferred close is Friday 21:00 UTC; exit window starts 20:30 UTC.
        self.assertFalse(in_dynamic_weekly_exit_window(bars, stamp(16, 20, 25)))
        self.assertTrue(in_dynamic_weekly_exit_window(bars, stamp(16, 20, 30)))
        self.assertTrue(in_dynamic_weekly_exit_window(bars, stamp(16, 20, 59)))
        self.assertFalse(in_dynamic_weekly_exit_window(bars, stamp(16, 21, 0)))

    def test_weekend_detector_requires_large_gap(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        bars = [stamp(9, 20), stamp(9, 20, 5), stamp(9, 20, 10)]
        self.assertIsNone(detect_weekend_close_bar_timestamp(bars))
        self.assertIsNone(detect_weekend_reopen_timestamp(bars))

    def test_swing_signal_uses_half_leg_stop_and_three_r_target(self):
        high = Swing(index=10, timestamp=100, price=110.0, kind="high", leg_start_price=100.0)
        signal = signal_from_swing(high, point=0.001, reward_risk=3.0)
        self.assertEqual(signal.side, "buy")
        self.assertAlmostEqual(signal.stop_loss, signal.entry - 5.0)
        self.assertAlmostEqual(signal.take_profit, signal.entry + 15.0)

        low = Swing(index=11, timestamp=200, price=90.0, kind="low", leg_start_price=110.0)
        sell = signal_from_swing(low, point=0.001, reward_risk=3.0)
        self.assertEqual(sell.side, "sell")
        self.assertAlmostEqual(sell.stop_loss, sell.entry + 10.0)
        self.assertAlmostEqual(sell.take_profit, sell.entry - 30.0)

    def test_detector_rejects_unequal_arrays(self):
        with self.assertRaises(ValueError):
            detect_confirmed_swings([1, 2], [1], [10, 20])

    def test_detector_is_repeatable_and_preserves_event_timing(self):
        highs = [10, 11, 12, 11, 10, 9, 10, 11, 13, 12, 10, 8, 9, 11, 12, 10, 8, 7, 9, 12, 13, 11, 9, 8, 10, 12, 11, 9, 7, 8]
        lows = [8, 9, 10, 9, 8, 7, 8, 9, 11, 10, 8, 6, 7, 9, 10, 8, 6, 5, 7, 10, 11, 9, 7, 6, 8, 10, 9, 7, 5, 6]
        times = list(range(1000, 1000 + len(highs)))
        kwargs = dict(depth=5, deviation_points=3, point=0.001, backstep=2)
        first = detect_confirmed_swings(highs, lows, times, **kwargs)
        second = detect_confirmed_swings(highs, lows, times, **kwargs)
        self.assertEqual(first, second)
        self.assertTrue(first)
        for swing in first:
            self.assertIsNotNone(swing.event_index)
            self.assertIsNotNone(swing.event_timestamp)
            self.assertEqual(swing.event_timestamp, times[swing.event_index])
            self.assertEqual(swing.flip_timestamp, times[swing.flip_index])
            self.assertEqual(swing.event_index - swing.flip_index, 2)
            self.assertGreaterEqual(swing.event_index, swing.index)

        live = detect_confirmed_swings(
            highs, lows, times, depth=5, deviation_points=3, point=0.001,
            backstep=2, repainting=True,
        )
        self.assertTrue(live)
        for swing in live:
            self.assertEqual(swing.event_index, swing.flip_index)
            self.assertEqual(swing.event_timestamp, swing.flip_timestamp)


    def test_daily_oil_exit_window_is_inferred_from_daily_gap(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        bars = [stamp(5, 20, 40), stamp(5, 20, 45), stamp(5, 20, 50), stamp(6, 0), stamp(6, 0, 5)]
        self.assertEqual(detect_daily_close_bar_timestamp(bars), stamp(5, 20, 50))
        self.assertTrue(in_dynamic_daily_exit_window(bars, stamp(6, 20, 25)))
        self.assertTrue(in_dynamic_daily_exit_window(bars, stamp(6, 20, 54)))
        self.assertFalse(in_dynamic_daily_exit_window(bars, stamp(6, 20, 55)))

    def test_daily_exit_detector_ignores_weekend_sized_gaps(self):
        import datetime

        def stamp(day, hour, minute=0):
            return int(datetime.datetime(2026, 1, day, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        bars = [stamp(9, 20, 45), stamp(9, 20, 50), stamp(12, 0), stamp(12, 0, 5)]
        self.assertIsNone(detect_daily_close_bar_timestamp(bars))
        self.assertFalse(in_dynamic_daily_exit_window(bars, stamp(12, 20, 30)))



class RunnerReconciliationTests(unittest.TestCase):
    def test_missing_saved_pending_order_is_rearmed_if_no_fill_or_position(self):
        runner = runner_module.ZigZagPortfolioRunner.__new__(
            runner_module.ZigZagPortfolioRunner
        )
        shadow = {
            "side": "buy",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 115.0,
            "swing_timestamp": 100,
            "last_checked_timestamp": 100,
            "submitted": True,
            "consumed": False,
        }
        runner.state = {"shadow_book": {"oil": {"buy": shadow, "sell": None}}}
        runner._pending_orders_for_side = lambda symbol, side: []
        runner._save_state = Mock()
        runner._place_signal = Mock(return_value="accepted")

        closed = [{"time": 200, "high": 99.0, "low": 98.0}]
        with patch.object(runner_module.mt5, "positions_get", return_value=[]):
            runner._advance_shadow_orders(
                "oil", "UKOILm", closed, [200], True, 1000.0, 1000.0
            )

        runner._place_signal.assert_called_once()
        self.assertTrue(shadow["submitted"])
        self.assertFalse(shadow["consumed"])
        self.assertEqual(shadow["last_checked_timestamp"], 200)


if __name__ == "__main__":
    unittest.main()
