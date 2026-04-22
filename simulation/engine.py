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
    peak_equity = np.ones(n_agents, dtype=np.float32)
    max_drawdown = np.zeros(n_agents, dtype=np.float32)

    # Pending decision for NEXT bar Open
    pending_decisions = np.zeros(n_agents, dtype=np.int8)

    for t in range(n_steps):
        # 1. EXECUTE PENDING DECISIONS (at Open of bar t)
        o_t = open_[t]
        atr_t = atr[t]

        for i in range(n_agents):
            if pending_decisions[i] != 0:
                if positions[i] == 0 and cooldown_counter[i] == 0:
                    # Open position
                    direction = pending_decisions[i]
                    positions[i] = direction
                    entry_price[i] = o_t

                    dist = atr_t * atr_mult[i]
                    if direction == 1:
                        sl[i] = o_t - dist
                        tp[i] = o_t + dist * rrr[i]
                    else:
                        sl[i] = o_t + dist
                        tp[i] = o_t - dist * rrr[i]

                pending_decisions[i] = 0 # consumed

        # 2. UPDATE OPEN TRADES (at High/Low of bar t)
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
                # Correct fractional PnL calculation
                pnl = ((exit_price - entry_price[i]) / entry_price[i]) * positions[i]
                pnl -= 0.0002 # cost (0.02% spread/slippage)

                equity[i] *= (1.0 + pnl)
                trades_count[i] += 1
                if pnl > 0:
                    wins_count[i] += 1

                positions[i] = 0
                cooldown_counter[i] = cooldown[i]

        # 3. DECISION FOR NEXT BAR (based on Close of bar t)
        # Vectorized NN pass for this timestep across all agents
        # h = tanh(x @ w1 + b1) -> x is (F,), w1 is (N, F, H), b1 is (N, H)
        # out = (h @ w2 + b2) -> h is (N, H), w2 is (N, H, 3), b2 is (N, 3)

        feat_t = features[t]

        for i in range(n_agents):
            # Hidden layer
            # Manual dot product for JIT efficiency if needed, but np.dot works
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

            # Aggression scaling
            out *= aggression[i]

            # Softmax
            probs = fast_softmax_row(out)

            # Argmax
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

            # 4. COOLDOWN & EQUITY TRACKING
            if cooldown_counter[i] > 0:
                cooldown_counter[i] -= 1

            if equity[i] > peak_equity[i]:
                peak_equity[i] = equity[i]

            dd = (peak_equity[i] - equity[i]) / peak_equity[i]
            if dd > max_drawdown[i]:
                max_drawdown[i] = dd

    return equity, trades_count, wins_count, max_drawdown

def run_simulation(pop, features, open_, high, low, close, atr):
    equity, trades, wins, max_dd = run_simulation_jit(
        features, open_, high, low, close, atr,
        pop["w1"], pop["b1"], pop["w2"], pop["b2"],
        pop["rrr"], pop["atr"], pop["threshold"], pop["cooldown"], pop["aggression"]
    )

    return {
        "equity": equity,
        "trades": trades,
        "winrate": np.where(trades > 0, wins / trades, 0.0).astype(np.float32),
        "max_drawdown": max_dd
    }
