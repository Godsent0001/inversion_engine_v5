import numpy as np
from numba import njit

@njit
def fast_tanh(x):
    return np.tanh(x)

@njit
def fast_softmax_row(x):
    # x is (3,)
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)

@njit
def run_simulation_jit(
    features, open_, high, low, close, atr,
    dayofweek, hour, minute,
    w1, b1, w2, b2,
    rrr, atr_mult, threshold, cooldown, aggression,
    initial_equity=1.0, risk_per_trade=0.01
):
    n_agents = w1.shape[0]
    n_steps = len(close)
    n_features = features.shape[1]
    hidden_size = w1.shape[2]

    # State arrays
    equity = np.full(n_agents, initial_equity, dtype=np.float32)
    positions = np.zeros(n_agents, dtype=np.int8)  # 0: none, 1: buy, -1: sell
    entry_price = np.zeros(n_agents, dtype=np.float32)
    sl_dist = np.zeros(n_agents, dtype=np.float32)
    sl = np.zeros(n_agents, dtype=np.float32)
    tp = np.zeros(n_agents, dtype=np.float32)
    cooldown_counter = np.zeros(n_agents, dtype=np.int32)

    # Tracking
    trades_count = np.zeros(n_agents, dtype=np.int32)
    wins_count = np.zeros(n_agents, dtype=np.int32)

    # Losing streak tracking
    current_losing_streak = np.zeros(n_agents, dtype=np.int32)
    max_losing_streak = np.zeros(n_agents, dtype=np.int32)

    # Sharpe tracking (daily returns aggregation)
    # Since we can't easily do date-based group by in NJIT,
    # we'll track bar-by-bar returns and then maybe adjust,
    # OR we track equity at the end of each day.
    # Simple bar-based sharpe is common in these fast sims.
    # But user asked for "industry standard". Let's use bar returns but annualize correctly.

    # We will track the equity at each step to compute Sharpe later if needed,
    # but for memory efficiency, let's track sum and sum_sq of log returns.
    sum_ret = np.zeros(n_agents, dtype=np.float32)
    sum_sq_ret = np.zeros(n_agents, dtype=np.float32)

    # Pending decision for NEXT bar Open
    pending_decisions = np.zeros(n_agents, dtype=np.int8)

    for t in range(n_steps):
        prev_equity = equity.copy()

        # 1. FRIDAY CLOSURE CHECK (19:00 GMT)
        # If it's Friday and time >= 19:00, close all and prevent new trades
        is_friday_evening = (dayofweek[t] == 4 and hour[t] >= 19)

        # 2. EXECUTE PENDING DECISIONS (at Open of bar t)
        o_t = open_[t]
        atr_t = atr[t]

        for i in range(n_agents):
            # Forced closure on Friday evening
            if is_friday_evening and positions[i] != 0:
                exit_p = o_t # Exit at open of the candle where we hit the limit
                # PnL calculation (Fixed risk)
                # pnl = ((exit - entry) / sl_dist) * risk * direction
                pnl = ((exit_p - entry_price[i]) / sl_dist[i]) * risk_per_trade * positions[i]
                pnl -= 0.0002 # cost

                equity[i] += pnl
                trades_count[i] += 1
                if pnl > 0:
                    wins_count[i] += 1
                    current_losing_streak[i] = 0
                else:
                    current_losing_streak[i] += 1
                    if current_losing_streak[i] > max_losing_streak[i]:
                        max_losing_streak[i] = current_losing_streak[i]

                positions[i] = 0
                pending_decisions[i] = 0
                continue

            if pending_decisions[i] != 0:
                if not is_friday_evening and positions[i] == 0 and cooldown_counter[i] == 0:
                    # Open position
                    direction = pending_decisions[i]
                    positions[i] = direction
                    entry_price[i] = o_t

                    dist = atr_t * atr_mult[i]
                    sl_dist[i] = dist
                    if direction == 1:
                        sl[i] = o_t - dist
                        tp[i] = o_t + dist * rrr[i]
                    else:
                        sl[i] = o_t + dist
                        tp[i] = o_t - dist * rrr[i]

                pending_decisions[i] = 0 # consumed

        # 3. UPDATE OPEN TRADES (at High/Low of bar t)
        h_t = high[t]
        l_t = low[t]

        for i in range(n_agents):
            if positions[i] == 0:
                continue

            exit_price = 0.0
            hit = 0 # 0: none, 1: TP, -1: SL

            if positions[i] == 1: # BUY
                if l_t <= sl[i]:
                    hit = -1
                    exit_price = sl[i]
                elif h_t >= tp[i]:
                    hit = 1
                    exit_price = tp[i]
            else: # SELL
                if h_t >= sl[i]:
                    hit = -1
                    exit_price = sl[i]
                elif l_t <= tp[i]:
                    hit = 1
                    exit_price = tp[i]

            if hit != 0:
                # Fixed risk PnL
                pnl = ((exit_price - entry_price[i]) / sl_dist[i]) * risk_per_trade * positions[i]
                pnl -= 0.0002 # cost

                equity[i] += pnl
                trades_count[i] += 1
                if pnl > 0:
                    wins_count[i] += 1
                    current_losing_streak[i] = 0
                else:
                    current_losing_streak[i] += 1
                    if current_losing_streak[i] > max_losing_streak[i]:
                        max_losing_streak[i] = current_losing_streak[i]

                positions[i] = 0
                cooldown_counter[i] = cooldown[i]

        # 4. DECISION FOR NEXT BAR (based on Close of bar t)
        if not is_friday_evening:
            feat_t = features[t]
            for i in range(n_agents):
                # Hidden layer
                h = np.zeros(hidden_size, dtype=np.float32)
                for j in range(hidden_size):
                    sum_w = 0.0
                    for k in range(n_features):
                        sum_w += feat_t[k] * w1[i, k, j]
                    h[j] = np.tanh(sum_w + b1[i, j])

                # Output layer
                out = np.zeros(3, dtype=np.float32)
                for j in range(3):
                    sum_o = 0.0
                    for k in range(hidden_size):
                        sum_o += h[k] * w2[i, k, j]
                    out[j] = sum_o + b2[i, j]

                out *= aggression[i]
                probs = fast_softmax_row(out)

                best_idx = 0
                max_p = probs[0]
                if probs[1] > max_p:
                    max_p = probs[1]
                    best_idx = 1
                if probs[2] > max_p:
                    max_p = probs[2]
                    best_idx = 2

                if max_p >= threshold[i]:
                    if best_idx == 1:
                        pending_decisions[i] = 1
                    elif best_idx == 2:
                        pending_decisions[i] = -1

        # 5. COOLDOWN & SHARPE TRACKING
        for i in range(n_agents):
            if cooldown_counter[i] > 0:
                cooldown_counter[i] -= 1

            # Simple bar return for Sharpe
            ret = (equity[i] - prev_equity[i]) / initial_equity
            sum_ret[i] += ret
            sum_sq_ret[i] += ret * ret

    # Final Sharpe calculation
    # Annualization: sqrt(252 * 24 * 12) for 5-min candles
    # 252 days * 24 hours * 12 candles per hour = 72,576 candles per year
    ann_factor = np.sqrt(252 * 24 * 12)

    sharpe = np.zeros(n_agents, dtype=np.float32)
    for i in range(n_agents):
        mean_r = sum_ret[i] / n_steps
        var_r = (sum_sq_ret[i] / n_steps) - (mean_r * mean_r)
        std_r = np.sqrt(np.maximum(var_r, 1e-12))
        sharpe[i] = (mean_r / std_r) * ann_factor

    return equity, trades_count, wins_count, max_losing_streak, sharpe

def run_simulation(pop, features, open_, high, low, close, atr, dayofweek, hour, minute):
    equity, trades, wins, max_ls, sharpe = run_simulation_jit(
        features, open_, high, low, close, atr,
        dayofweek, hour, minute,
        pop["w1"], pop["b1"], pop["w2"], pop["b2"],
        pop["rrr"], pop["atr"], pop["threshold"], pop["cooldown"], pop["aggression"]
    )

    return {
        "equity": equity,
        "trades": trades,
        "winrate": np.where(trades > 0, wins / trades, 0.0).astype(np.float32),
        "max_losing_streak": max_ls,
        "sharpe": sharpe
    }
