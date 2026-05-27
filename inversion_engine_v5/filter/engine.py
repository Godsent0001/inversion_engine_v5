import numpy as np
from numba import njit

@njit
def fast_tanh(x):
    return np.tanh(x)

@njit
def fast_softmax_2(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)

@njit
def fast_softmax_3(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x)

@njit
def run_agent_and_filter_sim(
    features, open_, high, low, close, atr,
    is_friday_evening,
    agent_w1, agent_b1, agent_w2, agent_b2,
    agent_rrr, agent_atr_mult, agent_threshold, agent_cooldown, agent_aggression,
    filter_w1, filter_b1, filter_w2, filter_b2, filter_threshold
):
    n_filters = filter_w1.shape[0]
    n_steps = len(close)

    n_feat_agent = agent_w1.shape[0]
    n_hidden_agent = agent_w1.shape[1]

    n_feat_filter = filter_w1.shape[1]
    n_hidden_filter = filter_w1.shape[2]

    # State
    equity = np.ones(n_filters, dtype=np.float32)
    positions = np.zeros(n_filters, dtype=np.int8)
    entry_price = np.zeros(n_filters, dtype=np.float32)
    sl = np.zeros(n_filters, dtype=np.float32)
    tp = np.zeros(n_filters, dtype=np.float32)

    # Agent virtual state
    agent_position = 0
    agent_sl = 0.0
    agent_tp = 0.0
    agent_cooldown_counter = 0

    # Tracking
    trades_count = np.zeros(n_filters, dtype=np.int32)
    wins_count = np.zeros(n_filters, dtype=np.int32)
    current_losing_streak = np.zeros(n_filters, dtype=np.int32)
    max_losing_streak = np.zeros(n_filters, dtype=np.int32)
    sum_ret = np.zeros(n_filters, dtype=np.float32)
    sum_sq_ret = np.zeros(n_filters, dtype=np.float32)

    pending_signal_direction = 0
    pending_signal_confidence = 0.0
    pending_signal_features = np.zeros(n_feat_agent, dtype=np.float32)

    for t in range(n_steps):
        o_t = open_[t]
        atr_t = atr[t]
        h_t = high[t]
        l_t = low[t]
        ife_t = is_friday_evening[t]

        start_equity = equity.copy()

        # 1. EXECUTE PENDING TRADES (at Open of bar t)
        if pending_signal_direction != 0:
            if not ife_t:
                filter_input = np.zeros(n_feat_filter, dtype=np.float32)
                for k in range(n_feat_agent):
                    filter_input[k] = pending_signal_features[k]
                filter_input[n_feat_agent] = pending_signal_confidence
                filter_input[n_feat_agent + 1] = float(pending_signal_direction)

                # Agent virtual execution
                agent_position = pending_signal_direction
                dist_ag = atr_t * agent_atr_mult
                if agent_position == 1:
                    agent_sl = o_t - dist_ag
                    agent_tp = o_t + dist_ag * agent_rrr
                else:
                    agent_sl = o_t + dist_ag
                    agent_tp = o_t - dist_ag * agent_rrr

                # Filters execution
                for i in range(n_filters):
                    if positions[i] == 0:
                        h_filt = np.zeros(n_hidden_filter, dtype=np.float32)
                        for j in range(n_hidden_filter):
                            val = 0.0
                            for k in range(n_feat_filter):
                                val += filter_input[k] * filter_w1[i, k, j]
                            h_filt[j] = np.tanh(val + filter_b1[i, j])

                        out_filt = np.zeros(2, dtype=np.float32)
                        for j in range(2):
                            val = 0.0
                            for k in range(n_hidden_filter):
                                val += h_filt[k] * filter_w2[i, k, j]
                            out_filt[j] = val + filter_b2[i, j]

                        probs_filt = fast_softmax_2(out_filt)
                        if probs_filt[0] >= filter_threshold[i]:
                            positions[i] = pending_signal_direction
                            entry_price[i] = o_t
                            dist = atr_t * agent_atr_mult
                            if positions[i] == 1:
                                sl[i] = o_t - dist
                                tp[i] = o_t + dist * agent_rrr
                            else:
                                sl[i] = o_t + dist
                                tp[i] = o_t - dist * agent_rrr

            pending_signal_direction = 0

        # 2. UPDATE OPEN TRADES (at High/Low of bar t)
        # Agent virtual update
        if agent_position != 0:
            hit_ag = 0
            if ife_t: hit_ag = 2
            else:
                if agent_position == 1:
                    if l_t <= agent_sl: hit_ag = -1
                    elif h_t >= agent_tp: hit_ag = 1
                else:
                    if h_t >= agent_sl: hit_ag = -1
                    elif l_t <= agent_tp: hit_ag = 1

            if hit_ag != 0:
                agent_position = 0
                agent_cooldown_counter = agent_cooldown

        # Filters update
        for i in range(n_filters):
            if positions[i] == 0:
                continue

            hit = 0
            exit_price = 0.0
            if ife_t:
                hit = 2
                exit_price = o_t
            else:
                if positions[i] == 1:
                    if l_t <= sl[i]: hit = -1; exit_price = sl[i]
                    elif h_t >= tp[i]: hit = 1; exit_price = tp[i]
                else:
                    if h_t >= sl[i]: hit = -1; exit_price = sl[i]
                    elif l_t <= tp[i]: hit = 1; exit_price = tp[i]

            if hit != 0:
                pnl = 0.0
                if hit == -1: pnl = -0.01
                elif hit == 1: pnl = 0.01 * agent_rrr
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

        # 3. AGENT DECISION (for Next bar)
        if agent_position == 0 and agent_cooldown_counter == 0 and not ife_t:
            feat_t = features[t]
            h_ag = np.zeros(n_hidden_agent, dtype=np.float32)
            for j in range(n_hidden_agent):
                val = 0.0
                for k in range(n_feat_agent):
                    val += feat_t[k] * agent_w1[k, j]
                h_ag[j] = np.tanh(val + agent_b1[j])

            out_ag = np.zeros(3, dtype=np.float32)
            for j in range(3):
                val = 0.0
                for k in range(n_hidden_agent):
                    val += h_ag[k] * agent_w2[k, j]
                out_ag[j] = (val + agent_b2[j])

            out_ag *= agent_aggression
            probs_ag = fast_softmax_3(out_ag)

            best_idx = np.argmax(probs_ag)
            max_p = probs_ag[best_idx]

            if max_p >= agent_threshold:
                if best_idx == 1:
                    pending_signal_direction = 1
                    pending_signal_confidence = max_p
                    pending_signal_features = feat_t
                elif best_idx == 2:
                    pending_signal_direction = -1
                    pending_signal_confidence = max_p
                    pending_signal_features = feat_t

        if agent_cooldown_counter > 0:
            agent_cooldown_counter -= 1

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
