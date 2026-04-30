import numpy as np


# -------------------------
# OPEN TRADE
# -------------------------
def open_trade(state, i, price, direction, atr_value):
    """
    Opens a trade for agent i
    """

    state["positions"][i] = direction
    state["entry_price"][i] = price

    atr_mult = state["atr"][i]
    rrr = state["rrr"][i]

    stop_distance = atr_value * atr_mult

    if direction == 1:  # BUY
        state["sl"][i] = price - stop_distance
        state["tp"][i] = price + stop_distance * rrr

    elif direction == -1:  # SELL
        state["sl"][i] = price + stop_distance
        state["tp"][i] = price - stop_distance * rrr


# -------------------------
# CLOSE TRADE
# -------------------------
def close_trade(state, i, price):

    entry = state["entry_price"][i]
    direction = state["positions"][i]

    pnl = (price - entry) * direction

    # Apply cost (spread + slippage)
    cost = state.get("cost", 0.0002)
    pnl -= cost

    state["equity"][i] *= (1 + pnl)

    state["positions"][i] = 0
    state["entry_price"][i] = 0.0
    state["cooldown_counter"][i] = state["cooldown"][i]

    state["trades"][i] += 1


# -------------------------
# UPDATE OPEN TRADES
# -------------------------
def update_trades(state, high, low):
    """
    Check SL/TP hits
    """

    for i in range(len(state["positions"])):

        if state["positions"][i] == 0:
            continue

        direction = state["positions"][i]

        if direction == 1:  # BUY

            if low <= state["sl"][i]:
                close_trade(state, i, state["sl"][i])

            elif high >= state["tp"][i]:
                close_trade(state, i, state["tp"][i])

        elif direction == -1:  # SELL

            if high >= state["sl"][i]:
                close_trade(state, i, state["sl"][i])

            elif low <= state["tp"][i]:
                close_trade(state, i, state["tp"][i])


# -------------------------
# APPLY DECISIONS
# -------------------------
def apply_decisions(state, decisions, price, atr):

    for i in range(len(decisions)):

        # Cooldown logic
        if state["cooldown_counter"][i] > 0:
            state["cooldown_counter"][i] -= 1
            continue

        if state["positions"][i] != 0:
            continue

        if decisions[i] == 1:
            open_trade(state, i, price, 1, atr)

        elif decisions[i] == -1:
            open_trade(state, i, price, -1, atr)


# -------------------------
# INIT TRADE STATE
# -------------------------
def initialize_trade_state(pop):

    n = len(pop["rrr"])

    state = {}

    for k, v in pop.items():
        state[k] = v.copy() if isinstance(v, np.ndarray) else v

    # Runtime fields
    state["positions"] = np.zeros(n, dtype=np.int8)
    state["entry_price"] = np.zeros(n, dtype=np.float32)
    state["sl"] = np.zeros(n, dtype=np.float32)
    state["tp"] = np.zeros(n, dtype=np.float32)

    state["equity"] = np.ones(n, dtype=np.float32)
    state["trades"] = np.zeros(n, dtype=np.int32)

    return state