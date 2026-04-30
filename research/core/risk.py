import numpy as np


# -------------------------
# POSITION SIZING
# -------------------------
def compute_position_size(equity, stop_distance, risk_pct=0.01):
    """
    equity: (n_agents,)
    stop_distance: (n_agents,)
    """

    risk_amount = equity * risk_pct

    size = np.where(
        stop_distance > 0,
        risk_amount / (stop_distance + 1e-8),
        0.0
    )

    return size.astype(np.float32)


# -------------------------
# COST MODEL
# -------------------------
def apply_costs(pnl, spread=0.0, slippage=0.0, position_size=None):
    """
    Subtract trading costs

    spread: price cost per trade
    slippage: additional execution loss
    """

    if position_size is None:
        return pnl

    cost = (spread + slippage) * position_size

    return pnl - cost


# -------------------------
# DRAWDOWN FILTER
# -------------------------
def max_drawdown_filter(equity_curve, max_dd_limit=-0.20):
    """
    Returns mask of agents that pass drawdown constraint
    """

    peak = np.maximum.accumulate(equity_curve, axis=1)
    dd = (equity_curve - peak) / peak

    max_dd = np.min(dd, axis=1)

    return max_dd >= max_dd_limit


# -------------------------
# OVERTRADING CONTROL (OPTIONAL)
# -------------------------
def trade_frequency_filter(trade_counts, max_trades_per_month=200, months=6):
    """
    Prevent unrealistic scalping behavior
    """

    max_allowed = max_trades_per_month * months

    return trade_counts <= max_allowed