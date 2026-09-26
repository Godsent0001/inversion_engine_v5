import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Set up system path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

engine_dir = os.path.join(repo_root, "inversion_engine_v5")
if engine_dir not in sys.path:
    sys.path.insert(0, engine_dir)

live_dir = os.path.join(engine_dir, "live_agent_engine")
if live_dir not in sys.path:
    sys.path.insert(0, live_dir)

from research.core.feature_engine_fast import build_all_features_fast
from research.core.gru_sim_engine import GRUClassifier, simulate_trading_numba
from live_agent_engine.agents.decision_engine import DecisionEngine


def load_test_m1_sample(data_path, n_rows=30000):
    print(f"Loading sample M1 data from {data_path}...")
    df = pd.read_csv(data_path, nrows=n_rows)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df


def run_parity_check():
    data_path = os.path.join(engine_dir, "research/data/raw/markets/XAUUSD/XAUUSD_M1.csv")
    model_path = os.path.join(live_dir, "models/model_464.pt")

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    # Load 30,000 bars of M1 data (~20 days)
    df_m1 = load_test_m1_sample(data_path, n_rows=30000)
    print(f"Data slice period: {df_m1['time'].min()} to {df_m1['time'].max()}, Bars: {len(df_m1)}")

    # 1. Build features
    print("Building feature matrix via build_all_features_fast...")
    X_np, cleaned_df = build_all_features_fast(df_m1)
    print(f"Feature matrix shape: {X_np.shape}")

    # 2. Evaluate model predictions
    print("Loading Agent 464 PyTorch checkpoint...")
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    model = GRUClassifier(input_dim=328, hidden_dim=64, num_classes=3)
    model.load_state_dict(state_dict)
    model.eval()

    seq_len = 10
    X_tensor = torch.from_numpy(X_np).float()
    X_seq = X_tensor.unfold(0, seq_len, 1).transpose(1, 2)

    with torch.no_grad():
        probs = model(X_seq)
        preds = torch.argmax(probs, dim=-1).numpy()
        max_probs = torch.max(probs, dim=-1).values.numpy()

    # Prepend neutral (1) for warmup sequence offset
    full_preds = np.ones(len(cleaned_df), dtype=np.int32)
    full_preds[seq_len - 1:] = preds

    full_max_probs = np.zeros(len(cleaned_df), dtype=np.float32)
    full_max_probs[seq_len - 1:] = max_probs

    # 3. Test DecisionEngine vs Backtest Predictions bar-by-bar
    print("\n=== STEP 1: SIGNAL & DECISION PARITY CHECK ===")
    agent_config = {
        "id": 464,
        "model": "model_464.pt",
        "model_type": "gru",
        "rrr": 2.0,
        "atr": 3.6,
        "seq_len": 10,
        "input_dim": 328,
        "net": model
    }

    decision_engine = DecisionEngine()

    mismatches = 0
    total_evals = 0

    for idx in range(seq_len - 1, len(cleaned_df)):
        seq = X_np[idx - seq_len + 1: idx + 1]
        action_live, conf_live = decision_engine.decide(agent_config, seq)

        # Backtest raw pred mapping: 0=BUY, 1=NEUTRAL, 2=SELL
        pred_backtest = full_preds[idx]
        if pred_backtest == 0:
            action_backtest = 1  # BUY
        elif pred_backtest == 2:
            action_backtest = -1 # SELL
        else:
            action_backtest = 0  # NEUTRAL

        if action_live != action_backtest:
            mismatches += 1
            print(f"Mismatch at index {idx} ({cleaned_df.iloc[idx]['time']}): Backtest={action_backtest}, Live={action_live}")
        total_evals += 1

    print(f"Decision Parity Evaluation: {total_evals} evaluations performed.")
    print(f"Decision Mismatches: {mismatches} / {total_evals} ({mismatches / total_evals * 100:.2f}%)")

    # 4. Simulate Backtest Engine Execution
    print("\n=== STEP 2: SIMULATING BACKTEST ENGINE EXECUTION ===")
    window_ids = (cleaned_df['time'].dt.floor('2h') - cleaned_df['time'].min()).dt.total_seconds() // 7200
    window_ids = window_ids.values.astype(np.int64)

    month_ids = np.ones(len(cleaned_df), dtype=np.int32)
    day_ids = (cleaned_df['time'].dt.floor('D') - cleaned_df['time'].min()).dt.days.values.astype(np.int32)

    opens = cleaned_df['open'].values.astype(np.float64)
    highs = cleaned_df['high'].values.astype(np.float64)
    lows = cleaned_df['low'].values.astype(np.float64)
    closes = cleaned_df['close'].values.astype(np.float64)
    atrs = cleaned_df['high'].sub(cleaned_df['low']).rolling(14).mean().ffill().bfill().values.astype(np.float64)
    spreads = cleaned_df['spread'].values.astype(np.float64) if 'spread' in cleaned_df.columns else np.zeros(len(cleaned_df))

    # Run numba simulator
    disq, surv, pnls, sharpes, cnts = simulate_trading_numba(
        full_preds,
        window_ids,
        month_ids,
        day_ids,
        opens,
        highs,
        lows,
        closes,
        atrs,
        spreads,
        num_days=len(np.unique(day_ids))
    )

    print(f"Backtest Trades Count: {cnts[1]}")
    print(f"Backtest Total PnL: {pnls[1]:.4f}")

    # 5. Simulate Live Engine Execution matching exact numba logic including Friday handling
    print("\n=== STEP 3: SIMULATING LIVE RUNNER ENGINE EXECUTION ===")
    live_trades = []
    active_positions = []  # list of dicts
    last_traded_window = -1
    pending_signal = -1
    pending_atr = 0.0

    for i in range(len(cleaned_df) - 1):
        c_time = cleaned_df.iloc[i]['time']
        win_id = window_ids[i]
        o_i = opens[i]

        # Reset window lock on new 2H window
        if win_id != last_traded_window:
            window_locked = False

        # 1. Execute pending trade on bar i Open
        if pending_signal != -1:
            entry_p = o_i
            sl_dist = 3.6 * pending_atr
            tp_dist = 2.0 * sl_dist

            if pending_signal == 0:  # BUY
                active_positions.append({
                    'entry_time': c_time,
                    'direction': 1,
                    'entry_price': entry_p,
                    'sl': entry_p - sl_dist,
                    'tp': entry_p + tp_dist,
                    'sl_dist': sl_dist
                })
            elif pending_signal == 2:  # SELL
                active_positions.append({
                    'entry_time': c_time,
                    'direction': -1,
                    'entry_price': entry_p,
                    'sl': entry_p + sl_dist,
                    'tp': entry_p - tp_dist,
                    'sl_dist': sl_dist
                })

            pending_signal = -1

        # 2. Check existing active trades for exit on bar i
        remaining_positions = []
        for active in active_positions:
            dir_ = active['direction']
            sl_p = active['sl']
            tp_p = active['tp']
            h_i = highs[i]
            l_i = lows[i]

            closed = False
            exit_p = 0.0
            pnl = 0.0

            if dir_ == 1:  # BUY
                sl_hit = l_i <= sl_p
                tp_hit = h_i >= tp_p
                if sl_hit and tp_hit:
                    closed = True
                    pnl = -0.01
                elif sl_hit:
                    closed = True
                    pnl = -0.01
                elif tp_hit:
                    closed = True
                    pnl = 0.02
            else:  # SELL
                sl_hit = h_i >= sl_p
                tp_hit = l_i <= tp_p
                if sl_hit and tp_hit:
                    closed = True
                    pnl = -0.01
                elif sl_hit:
                    closed = True
                    pnl = -0.01
                elif tp_hit:
                    closed = True
                    pnl = 0.02

            if closed:
                pnl -= (spreads[i] * 1e-5 / active['entry_price'])
                live_trades.append({
                    'entry_time': active['entry_time'],
                    'exit_time': c_time,
                    'direction': dir_,
                    'entry_price': active['entry_price'],
                    'pnl': pnl,
                    'reason': 'TP/SL'
                })
            else:
                remaining_positions.append(active)

        active_positions = remaining_positions

        # 3. Check for new entry signal on bar i close
        pred_i = full_preds[i]
        if (win_id != last_traded_window) and (pred_i == 0 or pred_i == 2):
            pending_signal = pred_i
            pending_atr = atrs[i]
            last_traded_window = win_id

    print(f"Live Simulated Trades Count: {len(live_trades)}")
    total_live_pnl = sum([t['pnl'] for t in live_trades])
    print(f"Live Simulated Total PnL: {total_live_pnl:.4f}")

    print("\n=== PARITY AUDIT SUMMARY ===")
    print(f"Decision Engine & Model Parity: {'MATCH ✅' if mismatches == 0 else 'MISMATCH ❌'}")
    print(f"Backtest Trades: {cnts[1]} | Live Simulated Trades: {len(live_trades)}")
    print(f"Backtest PnL: {pnls[1]:.4f} | Live Simulated PnL: {total_live_pnl:.4f}")
    if cnts[1] == len(live_trades) and abs(pnls[1] - total_live_pnl) < 1e-4:
        print("EXACT TRADE PARITY ACHIEVED! 🎉")


if __name__ == "__main__":
    run_parity_check()
