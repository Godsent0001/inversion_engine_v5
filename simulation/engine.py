import numpy as np

from core.decision import decide_batch
from simulation.trade_sim import (
    initialize_trade_state,
    update_trades,
    apply_decisions
)


# -------------------------
# MAIN SIMULATION ENGINE
# -------------------------
def run_simulation(pop, features, high, low, close, atr):
    """
    Runs full simulation for all agents

    Returns:
        stats dict
    """

    n_agents = len(pop["rrr"])
    n_steps = len(close)

    # -------------------------
    # INIT STATE
    # -------------------------
    state = initialize_trade_state(pop)

    # -------------------------
    # TRACKING
    # -------------------------
    equity_curve = np.ones((n_agents, n_steps), dtype=np.float32)

    peak_equity = np.ones(n_agents, dtype=np.float32)
    max_drawdown = np.zeros(n_agents, dtype=np.float32)

    prev_decisions = np.zeros(n_agents, dtype=np.int8)

    # -------------------------
    # MAIN LOOP
    # -------------------------
    for t in range(n_steps):

        price = close[t]

        # -------------------------
        # 1. EXECUTE (t-1 → t)
        # -------------------------
        apply_decisions(
            state,
            prev_decisions,
            price,
            atr[t]
        )

        # -------------------------
        # 2. UPDATE OPEN TRADES
        # -------------------------
        update_trades(
            state,
            high[t],
            low[t]
        )

        # -------------------------
        # 3. COOLDOWN UPDATE
        # -------------------------
        if "cooldown_counter" in state:
            state["cooldown_counter"] = np.maximum(
                state["cooldown_counter"] - 1,
                0
            )

        # -------------------------
        # 4. DECISION (FOR NEXT BAR)
        # -------------------------
        features_t = features[t]

        decisions, confidence = decide_batch(
            pop,
            features_t
        )

        # -------------------------
        # 5. BLOCK INVALID ACTIONS
        # -------------------------
        in_position = state["positions"] != 0

        # prevent re-entry while already in trade
        decisions[in_position] = 0

        # apply cooldown restriction
        if "cooldown_counter" in state:
            cooldown_mask = state["cooldown_counter"] > 0
            decisions[cooldown_mask] = 0

        prev_decisions = decisions

        # -------------------------
        # 6. TRACK EQUITY
        # -------------------------
        eq = state["equity"]

        equity_curve[:, t] = eq

        # -------------------------
        # 7. DRAWDOWN TRACKING
        # -------------------------
        peak_equity = np.maximum(peak_equity, eq)

        dd = (peak_equity - eq) / (peak_equity + 1e-8)
        max_drawdown = np.maximum(max_drawdown, dd)

    # -------------------------
    # FINAL STATS
    # -------------------------
    final_equity = state["equity"]
    trades = state["trades"]

    # Better winrate proxy (still approximate)
    profitable = final_equity > 1.0
    winrate = profitable.astype(np.float32)

    stats = {
        "equity": final_equity,
        "equity_curve": equity_curve,
        "trades": trades,
        "winrate": winrate,
        "max_drawdown": max_drawdown
    }

    return stats