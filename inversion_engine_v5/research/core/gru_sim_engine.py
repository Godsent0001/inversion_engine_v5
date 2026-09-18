import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from numba import njit

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Take hidden state of last time step
        last_out = out[:, -1, :]
        h1 = self.relu(self.fc1(last_out))
        logits = self.fc2(h1)
        probs = self.softmax(logits)
        return probs

@njit
def simulate_trading_numba(
    predictions,      # int array (N,) -> 0: BUY, 1: NEUTRAL, 2: SELL
    window_ids,       # int array (N,) -> 2H window index
    month_ids,        # int array (N,) -> 1..12 month index
    day_ids,          # int array (N,) -> 0..num_days-1 day index
    opens,            # float array (N,)
    highs,            # float array (N,)
    lows,             # float array (N,)
    closes,           # float array (N,)
    atrs,             # float array (N,)
    spreads,          # float array (N,)
    num_days=22,      # int -> total number of trading days in the slice
    atr_mult=3.6,
    rrr=2.0
):
    """
    Lightning-fast Numba simulator enforcing 2-hour window locking,
    1:2 RRR, entry on next bar open, and monthly early termination if monthly return < 0.
    Calculates standard daily annualized Sharpe ratio: (mean_daily / std_daily) * sqrt(252).
    """
    N = len(predictions)

    # Track monthly returns and trade counts
    monthly_pnls = np.zeros(13, dtype=np.float64)       # index 1..12
    monthly_sharpes = np.zeros(13, dtype=np.float64)    # index 1..12
    monthly_trade_counts = np.zeros(13, dtype=np.int32) # index 1..12

    # Daily PnL tracking for Sharpe calculation
    daily_pnls = np.zeros(35, dtype=np.float64)

    # Active trade storage
    MAX_TRADES = 1000
    active_dir = np.zeros(MAX_TRADES, dtype=np.int32)
    active_entry = np.zeros(MAX_TRADES, dtype=np.float64)
    active_sl = np.zeros(MAX_TRADES, dtype=np.float64)
    active_tp = np.zeros(MAX_TRADES, dtype=np.float64)
    active_month = np.zeros(MAX_TRADES, dtype=np.int32)
    num_active = 0

    window_locked = False
    last_win_id = -1
    pending_signal = -1  # 0: BUY, 2: SELL
    pending_atr = 0.0

    current_month = month_ids[0]
    month_trade_idx = 0

    disqualified = False
    months_survived = 0

    for i in range(N - 1):
        m = month_ids[i]

        # Check if month changed
        if m != current_month:
            # Evaluate ended month using daily returns
            sum_pnl = 0.0
            for d in range(num_days):
                sum_pnl += daily_pnls[d]

            monthly_pnls[current_month] = sum_pnl
            monthly_trade_counts[current_month] = month_trade_idx

            if num_days > 1:
                mean_d = sum_pnl / num_days
                var_d = 0.0
                for d in range(num_days):
                    diff = daily_pnls[d] - mean_d
                    var_d += diff * diff
                std_d = np.sqrt(var_d / (num_days - 1)) + 1e-8
                if std_d > 1e-7:
                    monthly_sharpes[current_month] = (mean_d / std_d) * np.sqrt(252.0)
                else:
                    monthly_sharpes[current_month] = 0.0
            else:
                monthly_sharpes[current_month] = 0.0

            # EARLY TERMINATION RULE: negative month -> stop testing
            if sum_pnl < 0.0:
                disqualified = True
                break

            months_survived += 1
            current_month = m
            month_trade_idx = 0
            daily_pnls.fill(0.0)

        # Reset window lock on new 2H window
        win_id = window_ids[i]
        if win_id != last_win_id:
            window_locked = False
            last_win_id = win_id

        # 1. Execute pending trade on bar i Open if present
        if pending_signal != -1:
            entry_p = opens[i]
            atr_val = pending_atr
            sl_dist = atr_mult * atr_val
            tp_dist = rrr * sl_dist

            if num_active < MAX_TRADES:
                if pending_signal == 0: # BUY
                    active_dir[num_active] = 1
                    active_entry[num_active] = entry_p
                    active_sl[num_active] = entry_p - sl_dist
                    active_tp[num_active] = entry_p + tp_dist
                    active_month[num_active] = m
                    num_active += 1
                elif pending_signal == 2: # SELL
                    active_dir[num_active] = -1
                    active_entry[num_active] = entry_p
                    active_sl[num_active] = entry_p + sl_dist
                    active_tp[num_active] = entry_p - tp_dist
                    active_month[num_active] = m
                    num_active += 1

            pending_signal = -1

        # 2. Check active trades for TP/SL hit on current bar i
        c_high = highs[i]
        c_low = lows[i]
        c_spread = spreads[i]

        k = 0
        while k < num_active:
            direction = active_dir[k]
            sl_p = active_sl[k]
            tp_p = active_tp[k]
            entry_p = active_entry[k]
            tr_month = active_month[k]

            closed = False
            pnl = 0.0

            if direction == 1: # BUY
                sl_hit = c_low <= sl_p
                tp_hit = c_high >= tp_p
                if sl_hit and tp_hit:
                    # Conservative: SL hit first
                    pnl = -0.01
                    closed = True
                elif sl_hit:
                    pnl = -0.01
                    closed = True
                elif tp_hit:
                    pnl = 0.02
                    closed = True
            elif direction == -1: # SELL
                sl_hit = c_high >= sl_p
                tp_hit = c_low <= tp_p
                if sl_hit and tp_hit:
                    # Conservative: SL hit first
                    pnl = -0.01
                    closed = True
                elif sl_hit:
                    pnl = -0.01
                    closed = True
                elif tp_hit:
                    pnl = 0.02
                    closed = True

            if closed:
                # Deduct spread cost (proportional to price)
                spread_cost = (c_spread * 1e-5) / entry_p
                net_pnl = pnl - spread_cost

                # Accumulate into daily PnL
                d_idx = day_ids[i]
                if d_idx >= 0 and d_idx < 35:
                    daily_pnls[d_idx] += net_pnl
                month_trade_idx += 1

                # Remove trade by swapping with last active
                num_active -= 1
                if k < num_active:
                    active_dir[k] = active_dir[num_active]
                    active_entry[k] = active_entry[num_active]
                    active_sl[k] = active_sl[num_active]
                    active_tp[k] = active_tp[num_active]
                    active_month[k] = active_month[num_active]
            else:
                k += 1

        # 3. Check for new prediction signal on current bar i close
        pred = predictions[i]
        if not window_locked and (pred == 0 or pred == 2):
            pending_signal = pred
            pending_atr = atrs[i]
            window_locked = True

    # Evaluate final month if not disqualified
    if not disqualified:
        sum_pnl = 0.0
        for d in range(num_days):
            sum_pnl += daily_pnls[d]
        monthly_pnls[current_month] = sum_pnl
        monthly_trade_counts[current_month] = month_trade_idx

        if num_days > 1:
            mean_d = sum_pnl / num_days
            var_d = 0.0
            for d in range(num_days):
                diff = daily_pnls[d] - mean_d
                var_d += diff * diff
            std_d = np.sqrt(var_d / (num_days - 1)) + 1e-8
            if std_d > 1e-7:
                monthly_sharpes[current_month] = (mean_d / std_d) * np.sqrt(252.0)
            else:
                monthly_sharpes[current_month] = 0.0
        else:
            monthly_sharpes[current_month] = 0.0

        if sum_pnl < 0.0:
            disqualified = True
        else:
            months_survived += 1

    return disqualified, months_survived, monthly_pnls, monthly_sharpes, monthly_trade_counts
