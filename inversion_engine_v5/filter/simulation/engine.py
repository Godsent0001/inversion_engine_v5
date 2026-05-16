import numpy as np
from numba import njit

@njit
def fast_tanh(x):
    return np.tanh(x)

@njit
def fast_softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)

@njit
def run_filter_simulation_jit(
    features, # Normalized (n_steps, 8)
    open_, high, low, close, atr,
    is_friday_evening,
    # Base Agent Params
    base_w1, base_b1, base_w2, base_b2,
    base_rrr, base_atr_mult, base_threshold, base_cooldown, base_aggression,
    # Filter Population Params
    filter_w1, filter_b1, filter_w2, filter_b2,
    filter_threshold
):
    n_filters = filter_w1.shape[0]
    n_steps = len(close)
    n_features_base = base_w1.shape[0]
    hidden_size_base = base_w1.shape[1]
    hidden_size_filter = filter_w1.shape[2]

    # State arrays
    equity = np.ones(n_filters, dtype=np.float32)
    positions = np.zeros(n_filters, dtype=np.int8)
    entry_price = np.zeros(n_filters, dtype=np.float32)
    sl = np.zeros(n_filters, dtype=np.float32)
    tp = np.zeros(n_filters, dtype=np.float32)
    cooldown_counter = np.zeros(n_filters, dtype=np.int32)

    # Tracking
    trades_count = np.zeros(n_filters, dtype=np.int32)
    wins_count = np.zeros(n_filters, dtype=np.int32)
    max_losing_streak = np.zeros(n_filters, dtype=np.int32)
    current_losing_streak = np.zeros(n_filters, dtype=np.int32)

    # Sharpe tracking
    sum_ret = np.zeros(n_filters, dtype=np.float32)
    sum_sq_ret = np.zeros(n_filters, dtype=np.float32)

    pending_decisions = np.zeros(n_filters, dtype=np.int8)

    for t in range(n_steps):
        o_t = open_[t]
        atr_t = atr[t]
        h_t = high[t]
        l_t = low[t]
        friday_evening = is_friday_evening[t]

        start_equity = equity.copy()

        # 1. EXECUTE PENDING DECISIONS
        for i in range(n_filters):
            if pending_decisions[i] != 0:
                if positions[i] == 0 and cooldown_counter[i] == 0 and not friday_evening:
                    positions[i] = pending_decisions[i]
                    entry_price[i] = o_t
                    dist = atr_t * base_atr_mult
                    if positions[i] == 1:
                        sl[i] = o_t - dist
                        tp[i] = o_t + dist * base_rrr
                    else:
                        sl[i] = o_t + dist
                        tp[i] = o_t - dist * base_rrr
                pending_decisions[i] = 0

        # 2. UPDATE OPEN TRADES
        for i in range(n_filters):
            if positions[i] == 0:
                continue

            exit_price = 0.0
            hit = 0

            if friday_evening:
                hit = 2
                exit_price = o_t
            else:
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
                pnl = 0.0
                if hit == -1:
                    pnl = -0.01
                elif hit == 1:
                    pnl = 0.01 * base_rrr
                elif hit == 2:
                    dist = abs(entry_price[i] - sl[i])
                    if dist > 1e-8:
                        pnl = ((exit_price - entry_price[i]) * positions[i] / dist) * 0.01

                pnl -= 0.0002
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
                cooldown_counter[i] = base_cooldown

        # 3. BASE AGENT DECISION
        feat_base_t = features[t, :6]

        h_base = np.zeros(hidden_size_base, dtype=np.float32)
        for j in range(hidden_size_base):
            sum_w = 0.0
            for k in range(6):
                sum_w += feat_base_t[k] * base_w1[k, j]
            h_base[j] = np.tanh(sum_w + base_b1[j])

        out_base = np.zeros(3, dtype=np.float32)
        for j in range(3):
            sum_o = 0.0
            for k in range(hidden_size_base):
                sum_o += h_base[k] * base_w2[k, j]
            out_base[j] = sum_o + base_b2[j]

        out_base *= base_aggression
        probs_base = fast_softmax(out_base)

        best_idx_base = 0
        max_p_base = probs_base[0]
        if probs_base[1] > max_p_base:
            max_p_base = probs_base[1]
            best_idx_base = 1
        if probs_base[2] > max_p_base:
            max_p_base = probs_base[2]
            best_idx_base = 2

        # 4. FILTER DECISION
        if max_p_base >= base_threshold and best_idx_base != 0 and not friday_evening:
            base_direction = 1.0 if best_idx_base == 1 else -1.0
            feat_filter_t = features[t]

            for i in range(n_filters):
                if positions[i] == 0 and cooldown_counter[i] == 0:
                    h_filter = np.zeros(hidden_size_filter, dtype=np.float32)
                    for j in range(hidden_size_filter):
                        sum_wf = 0.0
                        for k in range(8):
                            sum_wf += feat_filter_t[k] * filter_w1[i, k, j]
                        h_filter[j] = np.tanh(sum_wf + filter_b1[i, j])

                    out_filter = np.zeros(2, dtype=np.float32)
                    for j in range(2):
                        sum_of = 0.0
                        for k in range(hidden_size_filter):
                            sum_of += h_filter[k] * filter_w2[i, k, j]
                        out_filter[j] = sum_of + filter_b2[i, j]

                    probs_filter = fast_softmax(out_filter)
                    if probs_filter[1] > probs_filter[0] and probs_filter[1] >= filter_threshold[i]:
                        pending_decisions[i] = int(base_direction)

        # Update cooldown
        for i in range(n_filters):
            if cooldown_counter[i] > 0:
                cooldown_counter[i] -= 1

        # Sharpe tracking
        for i in range(n_filters):
            ret = equity[i] - start_equity[i]
            sum_ret[i] += ret
            sum_sq_ret[i] += ret ** 2

    sharpe = np.zeros(n_filters, dtype=np.float32)
    for i in range(n_filters):
        mean_r = sum_ret[i] / n_steps
        var_r = (sum_sq_ret[i] / n_steps) - (mean_r ** 2)
        std_r = np.sqrt(max(var_r, 1e-12))
        sharpe[i] = (mean_r / std_r) * np.sqrt(72576.0)

    return equity, trades_count, wins_count, max_losing_streak, sharpe

def run_filter_simulation(
    features, open_, high, low, close, atr, is_friday_evening,
    base_agent, filter_pop
):
    equity, trades, wins, max_losing_streak, sharpe = run_filter_simulation_jit(
        features, open_, high, low, close, atr, is_friday_evening,
        base_agent["w1"], base_agent["b1"], base_agent["w2"], base_agent["b2"],
        base_agent["rrr"], base_agent["atr"], base_agent["threshold"],
        base_agent["cooldown"], base_agent["aggression"],
        filter_pop["w1"], filter_pop["b1"], filter_pop["w2"], filter_pop["b2"],
        filter_pop["threshold"]
    )

    winrate = np.zeros_like(equity)
    mask = trades > 0
    winrate[mask] = wins[mask] / trades[mask]

    return {
        "equity": equity,
        "trades": trades,
        "winrate": winrate.astype(np.float32),
        "max_losing_streak": max_losing_streak,
        "sharpe": sharpe
    }
