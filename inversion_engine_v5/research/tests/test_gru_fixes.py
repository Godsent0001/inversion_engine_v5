import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from inversion_engine_v5.research.core.feature_engine_fast import build_all_features_fast
from inversion_engine_v5.research.core.gru_sim_engine import simulate_trading_numba

def test_expanding_standardization():
    print("Test 1: Verifying Expanding Cumulative Standardization (Zero Look-Ahead Bias)...")
    # Generate synthetic OHLC data for 100 bars
    np.random.seed(42)
    prices = 2000.0 + np.cumsum(np.random.randn(200) * 0.5)
    df = pd.DataFrame({
        'time': pd.date_range('2024-10-01', periods=200, freq='1min'),
        'open': prices,
        'high': prices + np.abs(np.random.randn(200)),
        'low': prices - np.abs(np.random.randn(200)),
        'close': prices + np.random.randn(200) * 0.1,
        'tick_volume': np.random.randint(10, 100, size=200),
        'spread': np.ones(200) * 20.0
    })

    # Full run
    X_full, _ = build_all_features_fast(df)

    # Sliced run up to bar 100
    df_half = df.iloc[:100].copy()
    X_half, _ = build_all_features_fast(df_half)

    # Compare features at bar 99 (100th bar) in both runs
    row_full_99 = X_full[99]
    row_half_99 = X_half[99]

    max_diff = np.max(np.abs(row_full_99 - row_half_99))
    print(f"Max feature difference at bar 99 between full data (200 bars) and truncated data (100 bars): {max_diff:.8f}")
    assert max_diff < 1e-4, f"Lookahead bias detected! Max diff = {max_diff}"
    print("SUCCESS: Expanding cumulative standardization passed zero look-ahead bias test!")

def test_daily_sharpe_calculation():
    print("\nTest 2: Verifying Daily Annualized Sharpe Calculation in simulate_trading_numba...")
    N = 1000
    preds = np.zeros(N, dtype=np.int32)
    # Trigger BUY every 50 bars
    preds[::50] = 0

    win_ids = (np.arange(N) // 120).astype(np.int64)
    month_ids = np.ones(N, dtype=np.int32)
    day_ids = (np.arange(N) // 200).astype(np.int32) # 5 days (0..4)
    num_days = 5

    opens = np.full(N, 2000.0)
    highs = np.full(N, 2010.0) # High reaches 2010 -> triggers TP (2000 + 3.6*0.6*2 = 2004.32)
    lows = np.full(N, 1998.0)
    closes = np.full(N, 2000.0)
    atrs = np.full(N, 0.6)
    spreads = np.full(N, 10.0)

    m_disq, m_surv, m_pnls, m_sharpes, m_cnts = simulate_trading_numba(
        preds, win_ids, month_ids, day_ids, opens, highs, lows, closes, atrs, spreads, num_days=num_days
    )

    print(f"Disqualified: {m_disq}, Survived: {m_surv}")
    print(f"Month 1 PnL: {m_pnls[1]:.6f}, Month 1 Sharpe: {m_sharpes[1]:.6f}, Trade Count: {m_cnts[1]}")
    assert not m_disq, "Model should not be disqualified"
    assert m_sharpes[1] > 0, "Sharpe ratio should be positive for profitable model"
    print("SUCCESS: Daily Sharpe calculation test passed!")

if __name__ == '__main__':
    test_expanding_standardization()
    test_daily_sharpe_calculation()
