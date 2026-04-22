import numpy as np


def execute_step(
    positions,
    entry_price,
    equity,
    actions,
    price,
    high,
    low,
    atr,
    pop
):
    """
    One candle step execution (vectorized)

    positions:
        0 = flat
        1 = buy
        -1 = sell
    """

    n = len(positions)

    # -------------------------
    # ENTRY
    # -------------------------
    flat_mask = positions == 0

    buy_mask = (actions == 0) & flat_mask
    sell_mask = (actions == 1) & flat_mask

    positions[buy_mask] = 1
    positions[sell_mask] = -1

    entry_price[buy_mask | sell_mask] = price


    # -------------------------
    # SL / TP distances
    # -------------------------
    sl_dist = atr * pop["atr"]
    tp_dist = sl_dist * pop["rrr"]


    # -------------------------
    # LONG CONDITIONS
    # -------------------------
    long_mask = positions == 1

    long_tp = long_mask & (high >= entry_price + tp_dist)
    long_sl = long_mask & (low <= entry_price - sl_dist)


    # -------------------------
    # SHORT CONDITIONS
    # -------------------------
    short_mask = positions == -1

    short_tp = short_mask & (low <= entry_price - tp_dist)
    short_sl = short_mask & (high >= entry_price + sl_dist)


    # -------------------------
    # EXIT MASK
    # -------------------------
    exit_mask = long_tp | long_sl | short_tp | short_sl


    # -------------------------
    # PNL CALCULATION (1% risk)
    # -------------------------
    risk_amount = equity * 0.01

    # stop distance per agent
    stop_distance = sl_dist

    position_size = np.where(
        stop_distance > 0,
        risk_amount / (stop_distance + 1e-8),
        0
    )

    pnl = np.zeros(n, dtype=np.float32)

    # LONG PNL
    pnl[long_tp] = position_size[long_tp] * tp_dist[long_tp]
    pnl[long_sl] = -position_size[long_sl] * sl_dist[long_sl]

    # SHORT PNL
    pnl[short_tp] = position_size[short_tp] * tp_dist[short_tp]
    pnl[short_sl] = -position_size[short_sl] * sl_dist[short_sl]


    # -------------------------
    # UPDATE EQUITY
    # -------------------------
    equity += pnl


    # -------------------------
    # RESET EXITED POSITIONS
    # -------------------------
    positions[exit_mask] = 0
    entry_price[exit_mask] = 0


    return positions, entry_price, equity, pnl