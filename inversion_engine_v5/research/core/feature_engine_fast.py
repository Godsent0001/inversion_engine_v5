import numpy as np
import pandas as pd
import os

def build_all_features_fast(df_m1, markets_dir=None):
    """
    Ultra low memory feature builder assigning directly to pre-allocated float32 array.
    Total RAM footprint < 300MB.
    """
    df = df_m1.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    close = df['close'].values.astype(np.float32)
    open_p = df['open'].values.astype(np.float32)
    high = df['high'].values.astype(np.float32)
    low = df['low'].values.astype(np.float32)
    volume = (df['tick_volume'] if 'tick_volume' in df.columns else df['volume']).values.astype(np.float32)
    spread = (df['spread'] if 'spread' in df.columns else np.zeros(len(df))).values.astype(np.float32)

    N = len(close)
    num_cols = 328
    X_mat = np.empty((N, num_cols), dtype=np.float32)
    col_idx = 0

    def add_col(arr):
        nonlocal col_idx
        if col_idx < num_cols:
            X_mat[:, col_idx] = arr.astype(np.float32)
            col_idx += 1

    # 1. Returns
    for k in [1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 240]:
        shift_c = np.roll(close, k)
        shift_c[:k] = close[0]
        ret = (close - shift_c) / (shift_c + 1e-8)
        add_col(ret)
        add_col(np.log(close / (shift_c + 1e-8)))
        add_col(np.abs(ret))

    shift1 = np.roll(close, 1)
    shift1[0] = close[0]
    add_col(close - shift1)

    # 2. Geometry
    rng = high - low
    add_col(rng)
    add_col(rng / (close + 1e-8))
    body = close - open_p
    add_col(body)
    abs_body = np.abs(body)
    add_col(abs_body)
    add_col(abs_body / (close + 1e-8))
    add_col(abs_body / (rng + 1e-8))

    uw = high - np.maximum(open_p, close)
    lw = np.minimum(open_p, close) - low
    add_col(uw)
    add_col(lw)
    add_col(uw / (rng + 1e-8))
    add_col(lw / (rng + 1e-8))
    add_col((close - low) / (rng + 1e-8))

    # 3. Volatility & ATR
    tr1 = high - low
    tr2 = np.abs(high - shift1)
    tr3 = np.abs(low - shift1)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr_dict = {}
    for p in [5, 7, 10, 14, 20, 21, 30, 50, 100, 200]:
        atr_p = pd.Series(tr).rolling(p, min_periods=1).mean().values.astype(np.float32)
        atr_dict[p] = atr_p
        if p in [5, 10, 14, 20, 30, 50, 100, 200]:
            add_col(atr_p)
            add_col(atr_p / (close + 1e-8))
            shift3_atr = np.roll(atr_p, 3)
            shift3_atr[:3] = atr_p[0]
            slope = (atr_p - shift3_atr) / 3.0
            add_col(slope)
            shift1_atr = np.roll(atr_p, 1)
            shift1_atr[0] = atr_p[0]
            atr_chg = atr_p - shift1_atr
            add_col(atr_chg)
            shift1_chg = np.roll(atr_chg, 1)
            shift1_chg[0] = atr_chg[0]
            add_col(atr_chg - shift1_chg)

    add_col(atr_dict[5] / (atr_dict[20] + 1e-8))
    add_col(atr_dict[10] / (atr_dict[50] + 1e-8))
    add_col(atr_dict[20] / (atr_dict[100] + 1e-8))

    atr14 = atr_dict[14]

    # 4. Realized Volatility
    ret1 = X_mat[:, 0]
    s_ret1 = pd.Series(ret1)
    for p in [5, 10, 20, 30, 50, 100, 200]:
        rvol = s_ret1.rolling(p, min_periods=1).std().fillna(0.0).values.astype(np.float32)
        add_col(rvol)
        shift1_rv = np.roll(rvol, 1)
        shift1_rv[0] = rvol[0]
        rv_chg = rvol - shift1_rv
        add_col(rv_chg)
        shift1_rvc = np.roll(rv_chg, 1)
        shift1_rvc[0] = rv_chg[0]
        add_col(rv_chg - shift1_rvc)

    # 5. Bollinger Bands
    s_close = pd.Series(close)
    for p in [20, 50, 100]:
        mid = s_close.rolling(p, min_periods=1).mean().values.astype(np.float32)
        std = s_close.rolling(p, min_periods=1).std().fillna(0.0).values.astype(np.float32)
        upper = mid + 2.0 * std
        lower = mid - 2.0 * std
        bw = (upper - lower) / (mid + 1e-8)
        add_col(mid)
        add_col(bw)
        add_col(bw / (atr14 + 1e-8))
        add_col((close - lower) / (upper - lower + 1e-8))
        add_col((upper - close) / (atr14 + 1e-8))
        add_col((close - lower) / (atr14 + 1e-8))
        shift1_bw = np.roll(bw, 1)
        shift1_bw[0] = bw[0]
        add_col(bw - shift1_bw)

    # 6. Trend EMAs
    ema_dict = {}
    for p in [5, 10, 12, 20, 26, 50, 100, 200]:
        ema_p = s_close.ewm(span=p, adjust=False).mean().values.astype(np.float32)
        ema_dict[p] = ema_p
        if p in [5, 10, 20, 50, 100, 200]:
            add_col(ema_p)
            add_col((close - ema_p) / (atr14 + 1e-8))
            shift3_e = np.roll(ema_p, 3)
            shift3_e[:3] = ema_p[0]
            add_col((ema_p - shift3_e) / 3.0)
            shift1_e = np.roll(ema_p, 1)
            shift1_e[0] = ema_p[0]
            echg = ema_p - shift1_e
            add_col(echg)
            shift1_ec = np.roll(echg, 1)
            shift1_ec[0] = echg[0]
            add_col(echg - shift1_ec)

    add_col((ema_dict[5] - ema_dict[10]) / (atr14 + 1e-8))
    add_col((ema_dict[5] - ema_dict[20]) / (atr14 + 1e-8))
    add_col((ema_dict[10] - ema_dict[20]) / (atr14 + 1e-8))
    add_col((ema_dict[20] - ema_dict[50]) / (atr14 + 1e-8))
    add_col((ema_dict[50] - ema_dict[100]) / (atr14 + 1e-8))
    add_col((ema_dict[100] - ema_dict[200]) / (atr14 + 1e-8))

    # 7. ADX & Momentum
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    up_m = s_high.diff().fillna(0).values
    dn_m = (-s_low.diff().fillna(0)).values
    pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    mdm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)

    for p in [7, 14, 21]:
        atr_p = atr_dict[p]
        pdi = 100.0 * pd.Series(pdm).ewm(alpha=1.0/p, adjust=False).mean().values / (atr_p + 1e-8)
        mdi = 100.0 * pd.Series(mdm).ewm(alpha=1.0/p, adjust=False).mean().values / (atr_p + 1e-8)
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-8)
        adx = pd.Series(dx).ewm(alpha=1.0/p, adjust=False).mean().values.astype(np.float32)
        add_col(adx)
        add_col(pdi)
        add_col(mdi)
        add_col(pdi - mdi)

    # 8. RSI & ROC
    for p in [5, 7, 14, 21, 50]:
        delta = s_close.diff().fillna(0)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ag = gain.ewm(alpha=1.0/p, adjust=False).mean()
        al = loss.ewm(alpha=1.0/p, adjust=False).mean()
        rs = ag / (al + 1e-8)
        rsi = (100.0 - (100.0 / (1.0 + rs))).values.astype(np.float32)
        add_col(rsi)
        shift1_r = np.roll(rsi, 1)
        shift1_r[0] = rsi[0]
        add_col(rsi - shift1_r)

    for p in [1, 3, 5, 10, 20, 30]:
        shift_p = np.roll(close, p)
        shift_p[:p] = close[0]
        add_col((close - shift_p) / (shift_p + 1e-8))

    for p in [3, 5, 10, 20]:
        shift_p = np.roll(close, p)
        shift_p[:p] = close[0]
        add_col(close - shift_p)

    # 9. MACD
    ema12 = ema_dict[12]
    ema26 = ema_dict[26]
    macd = ema12 - ema26
    macd_sig = pd.Series(macd).ewm(span=9, adjust=False).mean().values.astype(np.float32)
    macd_hist = macd - macd_sig
    add_col(macd)
    add_col(macd_sig)
    add_col(macd_hist)
    add_col(macd / (atr14 + 1e-8))

    # 10. Market structure
    for p in [5, 10, 20, 30, 50, 100, 200]:
        rh = s_high.rolling(p, min_periods=1).max().values.astype(np.float32)
        rl = s_low.rolling(p, min_periods=1).min().values.astype(np.float32)
        rr = rh - rl
        add_col(rh)
        add_col(rl)
        add_col(rr / (atr14 + 1e-8))
        add_col((rh - close) / (atr14 + 1e-8))
        add_col((close - rl) / (atr14 + 1e-8))
        add_col((close - rl) / (rr + 1e-8))

    # 11 & 12. Breakouts
    for p in [5, 10, 20, 30, 50, 100, 200]:
        prev_h = s_high.shift(1).rolling(p-1, min_periods=1).max().fillna(high[0]).values.astype(np.float32)
        prev_l = s_low.shift(1).rolling(p-1, min_periods=1).min().fillna(low[0]).values.astype(np.float32)
        add_col((close > prev_h).astype(np.float32))
        add_col((close < prev_l).astype(np.float32))
        add_col((close - prev_h) / (atr14 + 1e-8))
        add_col((prev_l - close) / (atr14 + 1e-8))

    # 13 & 14. Candle behavior & Streaks
    is_bull = (close > open_p).astype(np.float32)
    is_bear = (close < open_p).astype(np.float32)
    for p in [5, 10, 20, 50, 100]:
        add_col(pd.Series(is_bull).rolling(p, min_periods=1).mean().values.astype(np.float32))
        add_col(pd.Series(is_bear).rolling(p, min_periods=1).mean().values.astype(np.float32))
        add_col(pd.Series(abs_body).rolling(p, min_periods=1).mean().values.astype(np.float32))
        add_col(pd.Series(rng).rolling(p, min_periods=1).mean().values.astype(np.float32))

    bull_streak = np.zeros(N, dtype=np.float32)
    bear_streak = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        if is_bull[i] == 1.0:
            bull_streak[i] = bull_streak[i-1] + 1.0
        if is_bear[i] == 1.0:
            bear_streak[i] = bear_streak[i-1] + 1.0
    add_col(bull_streak)
    add_col(bear_streak)

    # 15. Volume & Spread
    s_vol = pd.Series(volume)
    for p in [5, 10, 20, 50, 100]:
        add_col(s_vol.rolling(p, min_periods=1).mean().values.astype(np.float32))
    vol_sma20 = s_vol.rolling(20, min_periods=1).mean().values.astype(np.float32)
    add_col(volume / (vol_sma20 + 1e-8))

    s_spr = pd.Series(spread)
    add_col(spread)
    add_col(spread / (atr14 + 1e-8))
    for p in [5, 20, 50]:
        add_col(s_spr.rolling(p, min_periods=1).mean().values.astype(np.float32))

    # 17. Cyclical time
    hours = df['time'].dt.hour.values
    minutes = df['time'].dt.minute.values
    dows = df['time'].dt.dayofweek.values

    add_col(np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32))
    add_col(np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32))
    add_col(np.sin(2.0 * np.pi * minutes / 60.0).astype(np.float32))
    add_col(np.cos(2.0 * np.pi * minutes / 60.0).astype(np.float32))
    add_col(np.sin(2.0 * np.pi * dows / 7.0).astype(np.float32))
    add_col(np.cos(2.0 * np.pi * dows / 7.0).astype(np.float32))

    # 18. Sessions
    add_col(((hours >= 0) & (hours < 8)).astype(np.float32))
    add_col(((hours >= 8) & (hours < 16)).astype(np.float32))
    add_col(((hours >= 13) & (hours < 21)).astype(np.float32))

    # 26. 2-HOUR WINDOW FEATURES
    window_id = df['time'].dt.floor('2h')
    minutes_in_window = ((df['time'] - window_id).dt.total_seconds() / 60.0).values.astype(np.float32)
    add_col(minutes_in_window)
    add_col(minutes_in_window / 120.0)
    add_col(120.0 - minutes_in_window)

    df_win = pd.DataFrame({'window_id': window_id, 'open': open_p, 'high': high, 'low': low, 'close': close})
    win_group = df_win.groupby('window_id')
    win_open = win_group['open'].transform('first').values.astype(np.float32)
    win_high_so_far = win_group['high'].cummax().values.astype(np.float32)
    win_low_so_far = win_group['low'].cummin().values.astype(np.float32)

    add_col(win_open)
    add_col(win_high_so_far)
    add_col(win_low_so_far)
    win_range = win_high_so_far - win_low_so_far
    add_col(win_range)
    add_col((close - win_open) / (atr14 + 1e-8))
    add_col((win_high_so_far - close) / (atr14 + 1e-8))
    add_col((close - win_low_so_far) / (atr14 + 1e-8))
    add_col((close - win_low_so_far) / (win_range + 1e-8))
    add_col((close - win_open) / (win_open + 1e-8))
    add_col((win_high_so_far - win_open) / (atr14 + 1e-8))
    add_col((win_open - win_low_so_far) / (atr14 + 1e-8))

    # Trim X_mat to actual assigned columns
    X_mat = X_mat[:, :col_idx]
    X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize columns in place
    means = np.mean(X_mat, axis=0)
    stds = np.std(X_mat, axis=0)
    stds[stds == 0] = 1.0
    X_mat -= means
    X_mat /= stds

    return X_mat.astype(np.float32), df
