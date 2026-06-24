import numpy as np
from numba import njit

@njit
def fast_tanh(x):
    return np.tanh(x)

@njit
def fast_softmax_row(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)

@njit
def run_stop_simulation_jit(
    features, open_, high, low, close, atr,
    w1, b1, w2, b2,
    rrr, atr_mult, threshold, cooldown, aggression
):
    n_agents = w1.shape[0]
    n_steps = len(close)
    n_features = features.shape[1]
    hidden_size = w1.shape[2]

    # State arrays
    equity = np.ones(n_agents, dtype=np.float32)

    # Position state
    positions = np.zeros(n_agents, dtype=np.int8)  # 0: none, 1: buy, -1: sell
    pos_entry_price = np.zeros(n_agents, dtype=np.float32)
    pos_sl = np.zeros(n_agents, dtype=np.float32)
    pos_tp = np.zeros(n_agents, dtype=np.float32)

    # Pending order state
    pending_type = np.zeros(n_agents, dtype=np.int8) # 0: none, 1: buy stop, -1: sell stop
    pending_entry_price = np.zeros(n_agents, dtype=np.float32)
    pending_sl = np.zeros(n_agents, dtype=np.float32)
    pending_tp = np.zeros(n_agents, dtype=np.float32)

    cooldown_counter = np.zeros(n_agents, dtype=np.int32)

    # Tracking
    trades_count = np.zeros(n_agents, dtype=np.int32)
    wins_count = np.zeros(n_agents, dtype=np.int32)
    peak_equity = np.ones(n_agents, dtype=np.float32)
    max_drawdown = np.zeros(n_agents, dtype=np.float32)

    # Extra metrics for stop orders
    orders_placed_count = np.zeros(n_agents, dtype=np.int32)
    orders_cancelled_count = np.zeros(n_agents, dtype=np.int32)

    # For Sharpe
    sum_ret = np.zeros(n_agents, dtype=np.float32)
    sum_sq_ret = np.zeros(n_agents, dtype=np.float32)

    for t in range(n_steps):
        prev_equity = equity.copy()

        o_t = open_[t]
        h_t = high[t]
        l_t = low[t]
        c_t = close[t]
        atr_t = atr[t]

        for i in range(n_agents):
            # 1. UPDATE OPEN TRADES
            if positions[i] != 0:
                exit_price = 0.0
                hit = 0 # 0: none, 1: TP, -1: SL

                if positions[i] == 1: # BUY
                    # Conservative: Check SL first if both might be hit
                    if l_t <= pos_sl[i] and h_t >= pos_tp[i]:
                        hit = -1
                        exit_price = pos_sl[i]
                    elif l_t <= pos_sl[i]:
                        hit = -1
                        exit_price = pos_sl[i]
                    elif h_t >= pos_tp[i]:
                        hit = 1
                        exit_price = pos_tp[i]
                else: # SELL
                    if h_t >= pos_sl[i] and l_t <= pos_tp[i]:
                        hit = -1
                        exit_price = pos_sl[i]
                    elif h_t >= pos_sl[i]:
                        hit = -1
                        exit_price = pos_sl[i]
                    elif l_t <= pos_tp[i]:
                        hit = 1
                        exit_price = pos_tp[i]

                if hit != 0:
                    pnl = ((exit_price - pos_entry_price[i]) / pos_entry_price[i]) * positions[i]
                    pnl -= 0.0002 # cost
                    equity[i] *= (1.0 + pnl)
                    trades_count[i] += 1
                    if pnl > 0:
                        wins_count[i] += 1
                    positions[i] = 0
                    cooldown_counter[i] = cooldown[i]

                # If we were in a position, we skip checking pending triggers or new signals for this agent
                # Skip to COOLDOWN & EQUITY TRACKING
            else:
                # 2. CHECK PENDING STOP ORDERS TRIGGER (only if no open position)
                if pending_type[i] != 0:
                    triggered = False
                    if pending_type[i] == 1: # Buy Stop
                        if h_t >= pending_entry_price[i]:
                            triggered = True
                            # Entry at the stop price
                            actual_entry = pending_entry_price[i]
                            # Check if SL/TP hit in the same bar
                            hit = 0
                            exit_price = 0.0
                            if l_t <= pending_sl[i] and h_t >= pending_tp[i]:
                                hit = -1; exit_price = pending_sl[i]
                            elif l_t <= pending_sl[i]:
                                hit = -1; exit_price = pending_sl[i]
                            elif h_t >= pending_tp[i]:
                                hit = 1; exit_price = pending_tp[i]

                            if hit != 0:
                                pnl = ((exit_price - actual_entry) / actual_entry) * 1
                                pnl -= 0.0002
                                equity[i] *= (1.0 + pnl)
                                trades_count[i] += 1
                                if pnl > 0: wins_count[i] += 1
                                cooldown_counter[i] = cooldown[i]
                            else:
                                positions[i] = 1
                                pos_entry_price[i] = actual_entry
                                pos_sl[i] = pending_sl[i]
                                pos_tp[i] = pending_tp[i]

                            pending_type[i] = 0 # consumed

                    elif pending_type[i] == -1: # Sell Stop
                        if l_t <= pending_entry_price[i]:
                            triggered = True
                            actual_entry = pending_entry_price[i]
                            hit = 0
                            exit_price = 0.0
                            if h_t >= pending_sl[i] and l_t <= pending_tp[i]:
                                hit = -1; exit_price = pending_sl[i]
                            elif h_t >= pending_sl[i]:
                                hit = -1; exit_price = pending_sl[i]
                            elif l_t <= pending_tp[i]:
                                hit = 1; exit_price = pending_tp[i]

                            if hit != 0:
                                pnl = ((exit_price - actual_entry) / actual_entry) * -1
                                pnl -= 0.0002
                                equity[i] *= (1.0 + pnl)
                                trades_count[i] += 1
                                if pnl > 0: wins_count[i] += 1
                                cooldown_counter[i] = cooldown[i]
                            else:
                                positions[i] = -1
                                pos_entry_price[i] = actual_entry
                                pos_sl[i] = pending_sl[i]
                                pos_tp[i] = pending_tp[i]

                            pending_type[i] = 0 # consumed

                # 3. DECISION FOR NEXT BAR (only if no open position and cooldown finished)
                if positions[i] == 0 and cooldown_counter[i] == 0:
                    # Hidden layer
                    h_layer = np.zeros(hidden_size, dtype=np.float32)
                    feat_t = features[t]
                    for j in range(hidden_size):
                        sum_w = 0.0
                        for k in range(n_features):
                            sum_w += feat_t[k] * w1[i, k, j]
                        h_layer[j] = np.tanh(sum_w + b1[i, j])

                    # Output layer
                    out = np.zeros(3, dtype=np.float32)
                    for j in range(3):
                        sum_o = 0.0
                        for k in range(hidden_size):
                            sum_o += h_layer[k] * w2[i, k, j]
                        out[j] = sum_o + b2[i, j]

                    out *= aggression[i]
                    probs = fast_softmax_row(out)

                    best_idx = 0
                    max_p = probs[0]
                    if probs[1] > max_p:
                        max_p = probs[1]; best_idx = 1
                    if probs[2] > max_p:
                        max_p = probs[2]; best_idx = 2

                    if max_p >= threshold[i]:
                        if best_idx == 1: # Buy Stop signal
                            found = False
                            entry_p = 0.0
                            for k in range(t, -1, -1):
                                if c_t < high[k]:
                                    entry_p = high[k]
                                    found = True
                                    break

                            if found:
                                if pending_type[i] != 0:
                                    orders_cancelled_count[i] += 1

                                pending_type[i] = 1
                                pending_entry_price[i] = entry_p
                                dist = atr_t * atr_mult[i]
                                pending_sl[i] = entry_p - dist
                                pending_tp[i] = entry_p + dist * rrr[i]
                                orders_placed_count[i] += 1

                        elif best_idx == 2: # Sell Stop signal
                            found = False
                            entry_p = 0.0
                            for k in range(t, -1, -1):
                                if c_t > low[k]:
                                    entry_p = low[k]
                                    found = True
                                    break

                            if found:
                                if pending_type[i] != 0:
                                    orders_cancelled_count[i] += 1

                                pending_type[i] = -1
                                pending_entry_price[i] = entry_p
                                dist = atr_t * atr_mult[i]
                                pending_sl[i] = entry_p + dist
                                pending_tp[i] = entry_p - dist * rrr[i]
                                orders_placed_count[i] += 1

            # 4. COOLDOWN & EQUITY TRACKING
            if cooldown_counter[i] > 0:
                cooldown_counter[i] -= 1

            if equity[i] > peak_equity[i]:
                peak_equity[i] = equity[i]

            dd = (peak_equity[i] - equity[i]) / peak_equity[i]
            if dd > max_drawdown[i]:
                max_drawdown[i] = dd

            # Sharpe components
            bar_return = (equity[i] - prev_equity[i]) / prev_equity[i]
            sum_ret[i] += bar_return
            sum_sq_ret[i] += bar_return ** 2

    # Final Sharpe
    mean_ret = sum_ret / n_steps
    var_ret = (sum_sq_ret / n_steps) - (mean_ret ** 2)
    std_ret = np.sqrt(np.maximum(var_ret, 1e-12))
    sharpe = (mean_ret / std_ret) * np.sqrt(252 * 48)

    return (equity, trades_count, wins_count, max_drawdown, sharpe,
            orders_placed_count, orders_cancelled_count)

def run_stop_simulation(pop, features, open_, high, low, close, atr):
    (equity, trades, wins, max_dd, sharpe,
     placed, cancelled) = run_stop_simulation_jit(
        features, open_, high, low, close, atr,
        pop["w1"], pop["b1"], pop["w2"], pop["b2"],
        pop["rrr"], pop["atr"], pop["threshold"], pop["cooldown"], pop["aggression"]
    )

    return {
        "equity": equity,
        "trades": trades,
        "winrate": np.where(trades > 0, wins / trades, 0.0).astype(np.float32),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "orders_placed": placed,
        "orders_cancelled": cancelled
    }
