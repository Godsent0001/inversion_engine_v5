import numpy as np

def compute_metrics(equity_curve, trade_pnl):
    """
    equity_curve: (n_agents, n_steps)
    trade_pnl: (n_agents, n_trades)
    """

    final_equity = equity_curve[:, -1]
    returns = final_equity - 1.0

    total_trades = np.sum(trade_pnl != 0, axis=1)
    wins = np.sum(trade_pnl > 0, axis=1)

    win_rate = np.where(
        total_trades > 0,
        wins / total_trades,
        0
    )

    # Sharpe (annualized for 5-min candles)
    returns_series = np.diff(equity_curve, axis=1)
    mean_ret = np.mean(returns_series, axis=1)
    std_ret = np.std(returns_series, axis=1)

    ann_factor = np.sqrt(252 * 24 * 12)
    sharpe = np.where(
        std_ret > 0,
        (mean_ret / std_ret) * ann_factor,
        0
    )

    # Losing Streak
    def get_max_losing_streak(pnl_arr):
        max_streak = 0
        current_streak = 0
        for pnl in pnl_arr:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            elif pnl > 0:
                current_streak = 0
        return max_streak

    max_losing_streaks = np.array([get_max_losing_streak(row[row != 0]) for row in trade_pnl])

    return {
        "final_equity": final_equity,
        "returns": returns,
        "trades": total_trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_losing_streak": max_losing_streaks
    }
