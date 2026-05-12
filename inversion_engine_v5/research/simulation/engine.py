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
    dayofweek, hour,
    w1, b1, w2, b2,
    rrr, atr_mult, threshold, aggression
):
    n_agents = w1.shape[0]
    n_steps = len(close)
    n_features = features.shape[1]
    hidden_size = w1.shape[2]

    # State arrays
    equity = np.ones(n_agents, dtype=np.float32)
    positions = np.zeros(n_agents, dtype=np.int8)  # 0: none, 1: buy, -1: sell
    entry_price = np.zeros(n_agents, dtype=np.float32)
    sl = np.zeros(n_agents, dtype=np.float32)
    tp = np.zeros(n_agents, dtype=np.float32)

    # Tracking
    trades_count = np.zeros(n_agents, dtype=np.int32)
    wins_count = np.zeros(n_agents, dtype=np.int32)
    peak_equity = np.ones(n_agents, dtype=np.float32)
    max_drawdown = np.zeros(n_agents, dtype=np.float32)
    max_losing_streak = np.zeros(n_agents, dtype=np.int32)
    current_losing_streak = np.zeros(n_agents, dtype=np.int32)

    # For Sharpe (bar-by-bar return tracking)
    sum_ret = np.zeros(n_agents, dtype=np.float32)
    sum_sq_ret = np.zeros(n_agents, dtype=np.float32)

    # Pending decision for NEXT bar Open
    pending_decisions = np.zeros(n_agents, dtype=np.int8)

    for t in range(n_steps):
        o_t = open_[t]
        h_t = high[t]
        l_t = low[t]
        a_t = atr[t]
        dow_t = dayofweek[t]
        hour_t = hour[t]

        # 1. FRIDAY 19:00 GMT CLOSURE
        is_friday_end = (dow_t == 4 and hour_t >= 19)

        if is_friday_end:
            for i in range(n_agents):
                if positions[i] != 0:
                    # Close at Open price of 19:00 bar
                    pnl = ((o_t - entry_price[i]) / entry_price[i]) * positions[i] - 0.0002
                    equity[i] += pnl
                    sum_ret[i] += pnl
                    sum_sq_ret[i] += pnl**2

                    trades_count[i] += 1
                    if pnl > 0:
                        wins_count[i] += 1
                        current_losing_streak[i] = 0
                    else:
                        current_losing_streak[i] += 1
                        if current_losing_streak[i] > max_losing_streak[i]:
                            max_losing_streak[i] = current_losing_streak[i]
                    positions[i] = 0
                pending_decisions[i] = 0 # No new trades
        else:
            # 2. EXECUTE PENDING DECISIONS (at Open of bar t)
            for i in range(n_agents):
                if pending_decisions[i] != 0:
                    if positions[i] == 0:
                        # Open position
                        direction = pending_decisions[i]
                        positions[i] = direction
                        entry_price[i] = o_t

                        dist = a_t * atr_mult[i]
                        if direction == 1:
                            sl[i] = o_t - dist
                            tp[i] = o_t + dist * rrr[i]
                        else:
                            sl[i] = o_t + dist
                            tp[i] = o_t - dist * rrr[i]
                    pending_decisions[i] = 0

            # 3. UPDATE OPEN TRADES (at High/Low of bar t)
            for i in range(n_agents):
                if positions[i] == 0:
                    continue

                exit_p = 0.0
                hit = 0 # 0: none, 1: TP, -1: SL

                if positions[i] == 1: # BUY
                    # Worst-case: SL hit first if both touched
                    if l_t <= sl[i]:
                        hit = -1
                        exit_p = sl[i]
                    elif h_t >= tp[i]:
                        hit = 1
                        exit_p = tp[i]
                else: # SELL
                    if h_t >= sl[i]:
                        hit = -1
                        exit_p = sl[i]
                    elif l_t <= tp[i]:
                        hit = 1
                        exit_p = tp[i]

                if hit != 0:
                    pnl = ((exit_p - entry_price[i]) / entry_price[i]) * positions[i] - 0.0002
                    equity[i] += pnl
                    sum_ret[i] += pnl
                    sum_sq_ret[i] += pnl**2

                    trades_count[i] += 1
                    if pnl > 0:
                        wins_count[i] += 1
                        current_losing_streak[i] = 0
                    else:
                        current_losing_streak[i] += 1
                        if current_losing_streak[i] > max_losing_streak[i]:
                            max_losing_streak[i] = current_losing_streak[i]

                    positions[i] = 0

        # 4. DECISION FOR NEXT BAR (based on Close of bar t)
        feat_t = features[t]
        for i in range(n_agents):
            h = np.zeros(hidden_size, dtype=np.float32)
            for j in range(hidden_size):
                sum_w = 0.0
                for k in range(n_features):
                    sum_w += feat_t[k] * w1[i, k, j]
                h[j] = np.tanh(sum_w + b1[i, j])

            out = np.zeros(3, dtype=np.float32)
            for j in range(3):
                sum_o = 0.0
                for k in range(hidden_size):
                    sum_o += h[k] * w2[i, k, j]
                out[j] = sum_o + b2[i, j]

            out *= aggression[i]
            # Softmax
            mx = np.max(out)
            ex = np.exp(out - mx)
            probs = ex / np.sum(ex)

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

        # 5. EQUITY TRACKING
        for i in range(n_agents):
            if equity[i] > peak_equity[i]:
                peak_equity[i] = equity[i]

            dd = (peak_equity[i] - equity[i]) / peak_equity[i]
            if dd > max_drawdown[i]:
                max_drawdown[i] = dd

    # Final Sharpe Ratio (Annualized for 30m bars)
    sharpe = np.zeros(n_agents, dtype=np.float32)
    for i in range(n_agents):
        mean_ret = sum_ret[i] / n_steps
        var_ret = (sum_sq_ret[i] / n_steps) - (mean_ret ** 2)
        std_ret = np.sqrt(np.maximum(var_ret, 1e-12))
        if std_ret > 1e-9:
            sharpe[i] = (mean_ret / std_ret) * np.sqrt(252 * 48)

    return equity, trades_count, wins_count, max_drawdown, sharpe, max_losing_streak


def run_simulation(pop, features, open_, high, low, close, atr, dayofweek, hour):
    equity, trades, wins, max_dd, sharpe, max_losing_streak = run_simulation_jit(
        features, open_, high, low, close, atr,
        dayofweek, hour,
        pop["w1"], pop["b1"], pop["w2"], pop["b2"],
        pop["rrr"], pop["atr"], pop["threshold"], pop["aggression"]
    )

    return {
        "equity": equity,
        "trades": trades,
        "winrate": np.where(trades > 0, wins / trades, 0.0).astype(np.float32),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "max_losing_streak": max_losing_streak
    }
