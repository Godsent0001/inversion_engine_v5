# ZigZag M5 portfolio strategy (separate runner)

This feature is isolated from the existing GRU/M1 runner. It does not modify the
existing runner or replace the GRU strategy.

## Current status

- The new runner defaults to dry-run mode. No orders are sent unless
  `LIVE_TRADING_ENABLED = True` is explicitly set in
  `live_agent_engine/zigzag_portfolio/settings.py`.
- It uses live MT5 account balance (not equity) and refuses entry processing if
  account balance is unavailable.
- The active risk reference is persisted. It rises with new balance highs and
  recursively resets to the current balance at each 50% drawdown threshold.
- Gold base risk is 2% of the active reference; UK Oil base risk is 1%.
- Lot sizing uses MT5's `order_calc_profit`, symbol volume limits/step, and a
  free-margin check. It does not force a trade when minimum volume would exceed
  the calculated risk budget.
- Default symbol preference is XAUUSDc then XAUUSDm; UKOILm then UKOILc.
- Weekly close clock settings are UTC and must be verified against the broker's
  actual weekly session close before any live activation.

## Important strategy-parity limitation

The current swing detector is a causal Python approximation using a trailing
window and backstep delay. It is **not yet proven bit-for-bit equivalent** to
the native MT5 ZigZag indicator or the user's backtest's `zigzag_events`
implementation. The original backtest's pivot-confirmation/order-activation
timing must be reconciled with the user's actual live method before live mode
is enabled. Do not treat dry-run output as backtest parity.

The runner drops the currently forming M5 candle and only evaluates completed
bars. This is deliberately conservative and may differ from a backtest that
retroactively makes an order active before a pivot is confirmed.

## Run dry-run

From the `inversion_engine_v5/live_agent_engine` directory:

```bash
python -m zigzag_portfolio.runner
```

Unit tests:

```bash
python -m unittest discover -s ../tests -p "test_zigzag_portfolio.py"
```

## Before live use

1. Compare swing timestamps, entries, stops, targets, pending-order replacement,
   gold session boundaries, and weekly exits against the original backtest.
2. Verify all times against the broker's server/session schedule.
3. Run on an MT5 demo account and confirm symbol contract details, minimum stop
   distances, filling rules, pending-order behavior, and margin.
4. Review the state files in `live_agent_engine/storage/`; do not delete the
   risk-reference state unless intentionally resetting risk history.
5. Keep the existing GRU runner and this runner from trading the same account
   simultaneously unless that interaction has been explicitly managed.
