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
def run_simulation_jit_v2(
    features, open_, high, low, close, atr,
    is_friday_evening,
    w1, b1, w2, b2,
    rrr, atr_mult, threshold, cooldown, aggression
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
    cooldown_counter = np.zeros(n_agents, dtype=np.int32)

    # Tracking
    trades_count = np.zeros(n_agents, dtype=np.int32)
    wins_count = np.zeros(n_agents, dtype=np.int32)
    current_losing_streak = np.zeros(n_agents, dtype=np.int32)
    max_losing_streak = np.zeros(n_agents, dtype=np.int32)

    # Sharpe tracking
    sum_ret = np.zeros(n_agents, dtype=np.float32)
    sum_sq_ret = np.zeros(n_agents, dtype=np.float32)

    # Pending decision for NEXT bar Open
    pending_decisions = np.zeros(n_agents, dtype=np.int8)

    for t in range(n_steps):
        o_t = open_[t]
        atr_t = atr[t]
        h_t = high[t]
        l_t = low[t]
        friday_evening = is_friday_evening[t]

        start_equity = equity.copy()

        # 1. EXECUTE PENDING DECISIONS (at Open of bar t)
        for i in range(n_agents):
            if pending_decisions[i] != 0:
                if positions[i] == 0 and cooldown_counter[i] == 0 and not friday_evening:
                    positions[i] = pending_decisions[i]
                    entry_price[i] = o_t
                    dist = atr_t * atr_mult[i]
                    if positions[i] == 1:
                        sl[i] = o_t - dist
                        tp[i] = o_t + dist * rrr[i]
                    else:
                        sl[i] = o_t + dist
                        tp[i] = o_t - dist * rrr[i]
                pending_decisions[i] = 0

        # 2. UPDATE OPEN TRADES (at High/Low of bar t)
        for i in range(n_agents):
            if positions[i] == 0:
                continue

            exit_price = 0.0
            hit = 0 # 0: none, 1: TP, -1: SL, 2: Friday close

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
                    pnl = 0.01 * rrr[i]
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
                cooldown_counter[i] = cooldown[i]

        # 3. DECISION FOR NEXT BAR
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
            probs = fast_softmax_row(out)

            best_idx = 0
            max_p = probs[0]
            if probs[1] > max_p:
                max_p = probs[1]
                best_idx = 1
            if probs[2] > max_p:
                max_p = probs[2]
                best_idx = 2

            if max_p >= threshold[i] and not friday_evening:
                if best_idx == 1:
                    pending_decisions[i] = 1
                elif best_idx == 2:
                    pending_decisions[i] = -1

            if cooldown_counter[i] > 0:
                cooldown_counter[i] -= 1

            # Sharpe tracking
            ret = equity[i] - start_equity[i]
            sum_ret[i] += ret
            sum_sq_ret[i] += ret ** 2

    sharpe = np.zeros(n_agents, dtype=np.float32)
    for i in range(n_agents):
        mean_r = sum_ret[i] / n_steps
        var_r = (sum_sq_ret[i] / n_steps) - (mean_r ** 2)
        std_r = np.sqrt(max(var_r, 1e-12))
        sharpe[i] = (mean_r / std_r) * np.sqrt(72576.0)

    return equity, trades_count, wins_count, max_losing_streak, sharpe

def run_simulation(pop, features, open_, high, low, close, atr, is_friday_evening):
    equity, trades, wins, max_losing_streak, sharpe = run_simulation_jit_v2(
        features, open_, high, low, close, atr, is_friday_evening,
        pop["w1"], pop["b1"], pop["w2"], pop["b2"],
        pop["rrr"], pop["atr"], pop["threshold"], pop["cooldown"], pop["aggression"]
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
