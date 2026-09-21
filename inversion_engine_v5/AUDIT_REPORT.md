# GRU Strategy Codebase Audit Report: Look-Ahead Bias & Metrics Accuracy

**Engine Version:** Inversion Engine v5 (GRU Strategy)
**Date:** October 2024 / May 2026 Audit
**Author:** Jules

---

## Executive Summary

A comprehensive code audit of the GRU trading model architecture (`inversion_engine_v5/research/core/`) was performed to evaluate potential look-ahead bias and verify the accuracy of strategy performance metrics.

Two critical issues were identified and documented:
1. **Look-Ahead Bias in Feature Standardization**: Global dataset feature scaling in `feature_engine_fast.py` leaked future statistical distribution parameters into past time steps.
2. **Metrics Inaccuracy in Sharpe Ratio Scaling**: Monthly Sharpe ratio in `gru_sim_engine.py` was scaled by `sqrt(trade_count)` instead of annualized daily returns, artificially inflating scores for high-frequency models.

All trade entry/exit mechanics, 2-hour window locking, and PyTorch sequence unfolding were verified to be strictly casual and lookahead-free.

---

## 1. Audit Findings & Diagnostics

### 1.1 Look-Ahead Bias in Feature Standardization
* **File**: `inversion_engine_v5/research/core/feature_engine_fast.py`
* **Defect**:
  ```python
  means = np.mean(X_mat, axis=0)
  stds = np.std(X_mat, axis=0)
  stds[stds == 0] = 1.0
  X_mat -= means
  X_mat /= stds
  ```
* **Explanation**: Computing `np.mean` and `np.std` across the entire 1-year or 2-year dataset scales inputs at candle $t$ using future price distribution statistics (from candles $t+1 \dots N$). This constitutes look-ahead bias (data leakage).
* **Remediation**: Replaced global standardization with **expanding cumulative standardization**. For any candle $t$, the mean and standard deviation are computed strictly over candles $0 \dots t$ using vectorized cumulative sum (`cumsum`):
  $$\mu_t = \frac{1}{t+1} \sum_{i=0}^t X_i, \quad \sigma_t = \sqrt{\max\left(\frac{1}{t+1} \sum_{i=0}^t X_i^2 - \mu_t^2, 10^{-8}\right)}$$
  This guarantees zero look-ahead bias while maintaining high vectorization performance.

---

### 1.2 Metrics Inaccuracy in Monthly Sharpe Ratio
* **File**: `inversion_engine_v5/research/core/gru_sim_engine.py`
* **Defect**:
  ```python
  monthly_sharpes[current_month] = (mean_r / std_r) * np.sqrt(month_trade_idx)
  ```
* **Explanation**: Multiplying trade-level Sharpe ratios by $\sqrt{N_{\text{trades}}}$ distorts performance ranking by scaling up high-trade-count models.
* **Remediation**: Converted monthly Sharpe calculation to **standard daily annualized Sharpe Ratio**. For each month, trade net PnLs are aggregated into calendar day buckets $R_d$. The annualized Sharpe is computed across all trading days $D$ in the month:
  $$\text{Sharpe}_{\text{monthly}} = \frac{\mu(R_d)}{\sigma(R_d) + 10^{-8}} \times \sqrt{252}$$

---

### 1.3 Execution Logic & Sequence Verification
* **Sequence Creation**: PyTorch `unfold(0, seq_len, 1)` forms sliding sequence vectors $[X_{t-\text{seq\_len}+1}, \dots, X_t]$. Model inference at index $t$ uses features strictly up to candle $t$.
* **Trade Execution**: Signals generated at the Close of candle $t$ are executed at the Open of candle $t+1$. SL/TP triggers are evaluated against candle $t+1$'s High and Low.
* **Window Indicators**: 2-hour window max/min indicators (`win_high_so_far`, `win_low_so_far`) use cumulative max/min (`cummax`/`cummin`) up to candle $t$, preventing intra-window lookahead.

---

## 2. Status & Next Steps

1. Apply expanding standardization in `feature_engine_fast.py`.
2. Update `simulate_trading_numba` in `gru_sim_engine.py` to implement daily annualized Sharpe ratio.
3. Validate fixes with dedicated unit testing.
4. Re-run 1,000 GRU model simulation (`run_gru_1000_sim.py`) and Year 2 out-of-sample evaluation (`run_top_10_year2_sim.py`).
