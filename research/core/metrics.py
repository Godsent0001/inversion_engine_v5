import numpy as np


def compute_metrics(equity_curve, trade_pnl):
    """
    equity_curve: (n_agents, n_steps)
    trade_pnl: (n_agents, n_trades)
    """

    final_equity = equity_curve[:, -1]

    # -------------------------
    # RETURNS
    # -------------------------
    returns = final_equity - 1.0


    # -------------------------
    # DRAWDOWN
    # -------------------------
    peak = np.maximum.accumulate(equity_curve, axis=1)
    drawdown = (equity_curve - peak) / peak

    max_dd = np.min(drawdown, axis=1)


    # -------------------------
    # TRADE STATS
    # -------------------------
    total_trades = np.sum(trade_pnl != 0, axis=1)

    wins = np.sum(trade_pnl > 0, axis=1)
    losses = np.sum(trade_pnl < 0, axis=1)

    win_rate = np.where(
        total_trades > 0,
        wins / total_trades,
        0
    )


    # -------------------------
    # SHARPE (simple version)
    # -------------------------
    returns_series = np.diff(equity_curve, axis=1)

    mean_ret = np.mean(returns_series, axis=1)
    std_ret = np.std(returns_series, axis=1)

    sharpe = np.where(
        std_ret > 0,
        mean_ret / std_ret,
        0
    )


    return {
        "final_equity": final_equity,
        "returns": returns,
        "max_drawdown": max_dd,
        "trades": total_trades,
        "win_rate": win_rate,
        "sharpe": sharpe
    }