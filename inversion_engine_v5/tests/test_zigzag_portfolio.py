import json
import tempfile
import unittest
from pathlib import Path

from zigzag_portfolio.risk_state import PersistentRiskReference
from zigzag_portfolio.strategy import Swing, detect_confirmed_swings, in_gold_session, signal_from_swing


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
    def test_gold_session_crosses_midnight_utc(self):
        import datetime

        def stamp(hour, minute=0):
            return int(datetime.datetime(2026, 1, 5, hour, minute, tzinfo=datetime.timezone.utc).timestamp())

        self.assertTrue(in_gold_session(stamp(23)))
        self.assertTrue(in_gold_session(stamp(7, 59)))
        self.assertFalse(in_gold_session(stamp(8)))

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
        lows = [value - 2 for value in highs]
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


if __name__ == "__main__":
    unittest.main()
