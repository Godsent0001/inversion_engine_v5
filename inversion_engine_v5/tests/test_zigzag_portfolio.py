import json
import tempfile
import unittest
from pathlib import Path

from zigzag_portfolio.risk_state import PersistentRiskReference
from zigzag_portfolio.strategy import Swing, in_gold_session, signal_from_swing


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
        # UTC timestamps: 23:00 and 07:59 are in session; 08:00 is not.
        import datetime
        def stamp(hour, minute=0):
            return int(datetime.datetime(2026, 1, 5, hour, minute, tzinfo=datetime.timezone.utc).timestamp())
        self.assertTrue(in_gold_session(stamp(23)))
        self.assertTrue(in_gold_session(stamp(7, 59)))
        self.assertFalse(in_gold_session(stamp(8)))

    def test_swing_signal_uses_half_leg_stop_and_three_r_target(self):
        high = Swing(index=10, timestamp=100, price=110.0, kind="high", leg_start_price=100.0)
        signal = signal_from_swing(high, point=0.01, reward_risk=3.0)
        self.assertEqual(signal.side, "buy")
        self.assertAlmostEqual(signal.stop_loss, signal.entry - 5.0)
        self.assertAlmostEqual(signal.take_profit, signal.entry + 15.0)
        low = Swing(index=11, timestamp=200, price=90.0, kind="low", leg_start_price=110.0)
        sell = signal_from_swing(low, point=0.01, reward_risk=3.0)
        self.assertEqual(sell.side, "sell")
        self.assertAlmostEqual(sell.stop_loss, sell.entry + 10.0)
        self.assertAlmostEqual(sell.take_profit, sell.entry - 30.0)


if __name__ == "__main__":
    unittest.main()
