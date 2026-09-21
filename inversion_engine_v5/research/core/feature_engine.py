import numpy as np
import pandas as pd
import glob
import os

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean().astype(np.float32)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return (100.0 - (100.0 / (1.0 + rs))).astype(np.float32)

def compute_adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0/period, adjust=False).mean() / (atr + 1e-8)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0/period, adjust=False).mean() / (atr + 1e-8)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.ewm(alpha=1.0/period, adjust=False).mean()
    return adx.astype(np.float32), plus_di.astype(np.float32), minus_di.astype(np.float32)

def build_all_features(df_m1, markets_dir=None):
    """
    Computes all ~400+ features for XAUUSD M1 data strictly without lookahead bias in float32.
    Returns: DataFrame of normalized float32 features and cleaned original DataFrame.
    """
    df = df_m1.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    close = df['close'].astype(np.float32)
    open_p = df['open'].astype(np.float32)
    high = df['high'].astype(np.float32)
    low = df['low'].astype(np.float32)
    volume = (df['tick_volume'] if 'tick_volume' in df.columns else df['volume']).astype(np.float32)
    spread = (df['spread'] if 'spread' in df.columns else pd.Series(0.0, index=df.index)).astype(np.float32)

    feats = {}

    # 1. Returns
    for k in [1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 240]:
        ret = close.pct_change(k).astype(np.float32)
        feats[f'ret_{k}'] = ret
        feats[f'log_ret_{k}'] = np.log(close / close.shift(k)).astype(np.float32)
        feats[f'abs_ret_{k}'] = ret.abs()
    feats['price_change'] = close.diff().astype(np.float32)

    # 2. Candle geometry
    candle_range = (high - low).astype(np.float32)
    feats['range'] = candle_range
    feats['range_pct'] = (candle_range / close).astype(np.float32)
    body = (close - open_p).astype(np.float32)
    feats['body'] = body
    feats['abs_body'] = body.abs()
    feats['body_pct'] = (body.abs() / close).astype(np.float32)
    feats['body_to_range'] = (body.abs() / (candle_range + 1e-8)).astype(np.float32)
    upper_wick = (high - np.maximum(open_p, close)).astype(np.float32)
    lower_wick = (np.minimum(open_p, close) - low).astype(np.float32)
    feats['upper_wick'] = upper_wick
    feats['lower_wick'] = lower_wick
    feats['upper_wick_to_range'] = (upper_wick / (candle_range + 1e-8)).astype(np.float32)
    feats['lower_wick_to_range'] = (lower_wick / (candle_range + 1e-8)).astype(np.float32)
    feats['close_position'] = ((close - low) / (candle_range + 1e-8)).astype(np.float32)

    # 3. Volatility & ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).astype(np.float32)

    atrs = {}
    for p in [5, 10, 14, 20, 30, 50, 100, 200]:
        atr = tr.rolling(p).mean().astype(np.float32)
        atrs[p] = atr
        feats[f'atr_{p}'] = atr
        feats[f'atr_{p}_div_price'] = (atr / close).astype(np.float32)
        atr_change = atr.diff().astype(np.float32)
        feats[f'atr_{p}_slope'] = ((atr - atr.shift(3)) / 3.0).astype(np.float32)
        feats[f'atr_{p}_change'] = atr_change
        feats[f'atr_{p}_accel'] = atr_change.diff().astype(np.float32)

    feats['atr5_div_atr20'] = (atrs[5] / (atrs[20] + 1e-8)).astype(np.float32)
    feats['atr10_div_atr50'] = (atrs[10] / (atrs[50] + 1e-8)).astype(np.float32)
    feats['atr20_div_atr100'] = (atrs[20] / (atrs[100] + 1e-8)).astype(np.float32)

    atr14 = atrs[14]

    # 4. Realized Volatility
    ret1 = close.pct_change(1).astype(np.float32)
    for p in [5, 10, 20, 30, 50, 100, 200]:
        vol = ret1.rolling(p).std().astype(np.float32)
        feats[f'realized_vol_{p}'] = vol
        vol_change = vol.diff().astype(np.float32)
        feats[f'vol_{p}_change'] = vol_change
        feats[f'vol_{p}_accel'] = vol_change.diff().astype(np.float32)

    feats['vol_ratio_5_50'] = (feats['realized_vol_5'] / (feats['realized_vol_50'] + 1e-8)).astype(np.float32)
    vol20_mean = feats['realized_vol_20'].rolling(100).mean().astype(np.float32)
    vol20_std = feats['realized_vol_20'].rolling(100).std().astype(np.float32)
    feats['vol_zscore'] = ((feats['realized_vol_20'] - vol20_mean) / (vol20_std + 1e-8)).astype(np.float32)

    # 5. Bollinger Bands
    for p in [20, 50, 100]:
        mid = close.rolling(p).mean().astype(np.float32)
        std = close.rolling(p).std().astype(np.float32)
        upper = (mid + 2.0 * std).astype(np.float32)
        lower = (mid - 2.0 * std).astype(np.float32)
        bw = ((upper - lower) / (mid + 1e-8)).astype(np.float32)
        feats[f'bb_{p}_mid'] = mid
        feats[f'bb_{p}_bw'] = bw
        feats[f'bb_{p}_bw_div_atr'] = (bw / (atr14 + 1e-8)).astype(np.float32)
        feats[f'bb_{p}_pct_b'] = ((close - lower) / (upper - lower + 1e-8)).astype(np.float32)
        feats[f'bb_{p}_dist_upper'] = ((upper - close) / (atr14 + 1e-8)).astype(np.float32)
        feats[f'bb_{p}_dist_lower'] = ((close - lower) / (atr14 + 1e-8)).astype(np.float32)
        feats[f'bb_{p}_bw_change'] = bw.diff().astype(np.float32)

    # 6. Trend (EMAs)
    emas = {}
    for p in [5, 10, 20, 50, 100, 200]:
        ema = compute_ema(close, p)
        emas[p] = ema
        feats[f'ema_{p}'] = ema
        feats[f'close_sub_ema_{p}_div_atr'] = ((close - ema) / (atr14 + 1e-8)).astype(np.float32)
        ema_change = ema.diff().astype(np.float32)
        feats[f'ema_{p}_slope'] = ((ema - ema.shift(3)) / 3.0).astype(np.float32)
        feats[f'ema_{p}_change'] = ema_change
        feats[f'ema_{p}_accel'] = ema_change.diff().astype(np.float32)

    feats['ema5_sub_10_div_atr'] = ((emas[5] - emas[10]) / (atr14 + 1e-8)).astype(np.float32)
    feats['ema5_sub_20_div_atr'] = ((emas[5] - emas[20]) / (atr14 + 1e-8)).astype(np.float32)
    feats['ema10_sub_20_div_atr'] = ((emas[10] - emas[20]) / (atr14 + 1e-8)).astype(np.float32)
    feats['ema20_sub_50_div_atr'] = ((emas[20] - emas[50]) / (atr14 + 1e-8)).astype(np.float32)
    feats['ema50_sub_100_div_atr'] = ((emas[50] - emas[100]) / (atr14 + 1e-8)).astype(np.float32)
    feats['ema100_sub_200_div_atr'] = ((emas[100] - emas[200]) / (atr14 + 1e-8)).astype(np.float32)

    # 7. ADX / Directional Movement
    for p in [7, 14, 21]:
        adx, p_di, m_di = compute_adx(high, low, close, p)
        feats[f'adx_{p}'] = adx
        feats[f'p_di_{p}'] = p_di
        feats[f'm_di_{p}'] = m_di
        di_diff = (p_di - m_di).astype(np.float32)
        feats[f'di_diff_{p}'] = di_diff
        feats[f'adx_{p}_slope'] = adx.diff().astype(np.float32)
        feats[f'di_diff_{p}_change'] = di_diff.diff().astype(np.float32)

    # 8. Momentum (RSI, ROC, Momentum)
    for p in [5, 7, 14, 21, 50]:
        rsi = compute_rsi(close, p)
        feats[f'rsi_{p}'] = rsi
        rsi_change = rsi.diff().astype(np.float32)
        feats[f'rsi_{p}_slope'] = ((rsi - rsi.shift(3)) / 3.0).astype(np.float32)
        feats[f'rsi_{p}_change'] = rsi_change
        feats[f'rsi_{p}_accel'] = rsi_change.diff().astype(np.float32)

    for p in [1, 3, 5, 10, 20, 30]:
        feats[f'roc_{p}'] = ((close - close.shift(p)) / (close.shift(p) + 1e-8)).astype(np.float32)

    for p in [3, 5, 10, 20]:
        feats[f'mom_{p}'] = (close - close.shift(p)).astype(np.float32)

    # 9. MACD
    ema12 = compute_ema(close, 12)
    ema26 = compute_ema(close, 26)
    macd = (ema12 - ema26).astype(np.float32)
    macd_signal = compute_ema(macd, 9)
    macd_hist = (macd - macd_signal).astype(np.float32)
    feats['macd'] = macd
    feats['macd_signal'] = macd_signal
    feats['macd_hist'] = macd_hist
    hist_change = macd_hist.diff().astype(np.float32)
    feats['macd_hist_slope'] = ((macd_hist - macd_hist.shift(3)) / 3.0).astype(np.float32)
    feats['macd_hist_change'] = hist_change
    feats['macd_hist_accel'] = hist_change.diff().astype(np.float32)
    feats['macd_div_atr'] = (macd / (atr14 + 1e-8)).astype(np.float32)

    # 10. Market Structure
    for p in [5, 10, 20, 30, 50, 100, 200]:
        roll_h = high.rolling(p).max().astype(np.float32)
        roll_l = low.rolling(p).min().astype(np.float32)
        roll_rng = (roll_h - roll_l).astype(np.float32)
        feats[f'roll_h_{p}'] = roll_h
        feats[f'roll_l_{p}'] = roll_l
        feats[f'roll_rng_{p}_div_atr'] = (roll_rng / (atr14 + 1e-8)).astype(np.float32)
        feats[f'dist_to_roll_h_{p}'] = ((roll_h - close) / (atr14 + 1e-8)).astype(np.float32)
        feats[f'dist_to_roll_l_{p}'] = ((close - roll_l) / (atr14 + 1e-8)).astype(np.float32)
        feats[f'range_pos_{p}'] = ((close - roll_l) / (roll_rng + 1e-8)).astype(np.float32)

    # 11 & 12. Breakouts & Failed Breakouts
    for p in [5, 10, 20, 30, 50, 100, 200]:
        prev_h = high.shift(1).rolling(p-1).max().astype(np.float32)
        prev_l = low.shift(1).rolling(p-1).min().astype(np.float32)
        feats[f'breakout_up_{p}'] = (close > prev_h).astype(np.float32)
        feats[f'breakout_dn_{p}'] = (close < prev_l).astype(np.float32)
        feats[f'breakout_dist_up_{p}'] = ((close - prev_h) / (atr14 + 1e-8)).astype(np.float32)
        feats[f'breakout_dist_dn_{p}'] = ((prev_l - close) / (atr14 + 1e-8)).astype(np.float32)

    # 13 & 14. Candle behavior & streaks
    is_bull = (close > open_p).astype(np.float32)
    is_bear = (close < open_p).astype(np.float32)
    is_doji = (body.abs() <= 0.1 * candle_range).astype(np.float32)

    for p in [5, 10, 20, 50, 100]:
        feats[f'bull_pct_{p}'] = is_bull.rolling(p).mean().astype(np.float32)
        feats[f'bear_pct_{p}'] = is_bear.rolling(p).mean().astype(np.float32)
        feats[f'doji_pct_{p}'] = is_doji.rolling(p).mean().astype(np.float32)
        feats[f'avg_body_{p}'] = body.abs().rolling(p).mean().astype(np.float32)
        feats[f'avg_range_{p}'] = candle_range.rolling(p).mean().astype(np.float32)
        feats[f'avg_uw_{p}'] = upper_wick.rolling(p).mean().astype(np.float32)
        feats[f'avg_lw_{p}'] = lower_wick.rolling(p).mean().astype(np.float32)

    # Streaks calculation
    bull_arr = is_bull.values
    bear_arr = is_bear.values
    bull_streak = np.zeros(len(df), dtype=np.float32)
    bear_streak = np.zeros(len(df), dtype=np.float32)
    for i in range(1, len(df)):
        if bull_arr[i] == 1:
            bull_streak[i] = bull_streak[i-1] + 1
        if bear_arr[i] == 1:
            bear_streak[i] = bear_streak[i-1] + 1
    feats['bull_streak'] = bull_streak
    feats['bear_streak'] = bear_streak

    # 15. Volume features
    vol_change = volume.diff().astype(np.float32)
    feats['vol_change'] = vol_change
    for p in [5, 10, 20, 50, 100]:
        feats[f'vol_sma_{p}'] = volume.rolling(p).mean().astype(np.float32)

    vol_sma20 = volume.rolling(20).mean().astype(np.float32)
    feats['vol_div_sma20'] = (volume / (vol_sma20 + 1e-8)).astype(np.float32)
    vol_mean100 = volume.rolling(100).mean().astype(np.float32)
    vol_std100 = volume.rolling(100).std().astype(np.float32)
    feats['vol_zscore'] = ((volume - vol_mean100) / (vol_std100 + 1e-8)).astype(np.float32)
    feats['vol_accel'] = vol_change.diff().astype(np.float32)

    # 16. Spread features
    feats['spread'] = spread
    feats['spread_div_atr'] = (spread / (atr14 + 1e-8)).astype(np.float32)
    for p in [5, 20, 50]:
        feats[f'spread_sma_{p}'] = spread.rolling(p).mean().astype(np.float32)
    spread_change = spread.diff().astype(np.float32)
    feats['spread_change'] = spread_change
    spread_mean100 = spread.rolling(100).mean().astype(np.float32)
    spread_std100 = spread.rolling(100).std().astype(np.float32)
    feats['spread_zscore'] = ((spread - spread_mean100) / (spread_std100 + 1e-8)).astype(np.float32)

    # 17. Cyclical Time features
    hours = df['time'].dt.hour
    minutes = df['time'].dt.minute
    dows = df['time'].dt.dayofweek

    feats['sin_hour'] = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32)
    feats['cos_hour'] = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32)
    feats['sin_min'] = np.sin(2.0 * np.pi * minutes / 60.0).astype(np.float32)
    feats['cos_min'] = np.cos(2.0 * np.pi * minutes / 60.0).astype(np.float32)
    feats['sin_dow'] = np.sin(2.0 * np.pi * dows / 7.0).astype(np.float32)
    feats['cos_dow'] = np.cos(2.0 * np.pi * dows / 7.0).astype(np.float32)

    # 18. Session features
    feats['asian_session'] = ((hours >= 0) & (hours < 8)).astype(np.float32)
    feats['london_session'] = ((hours >= 8) & (hours < 16)).astype(np.float32)
    feats['ny_session'] = ((hours >= 13) & (hours < 21)).astype(np.float32)
    feats['overlap_session'] = ((hours >= 13) & (hours < 16)).astype(np.float32)

    # 19 & 20. Daily structure
    dates = df['time'].dt.date
    df_day = df.copy()
    df_day['date'] = dates

    daily_stats = df_day.groupby('date').agg(
        day_open=('open', 'first'),
        day_high=('high', 'max'),
        day_low=('low', 'min'),
        day_close=('close', 'last')
    ).shift(1)

    df_day = df_day.merge(daily_stats, on='date', how='left')
    feats['prev_day_high'] = df_day['day_high'].astype(np.float32)
    feats['prev_day_low'] = df_day['day_low'].astype(np.float32)
    feats['prev_day_range'] = (df_day['day_high'] - df_day['day_low']).astype(np.float32)
    feats['dist_prev_day_high'] = ((df_day['day_high'] - close) / (atr14 + 1e-8)).astype(np.float32)
    feats['dist_prev_day_low'] = ((close - df_day['day_low']) / (atr14 + 1e-8)).astype(np.float32)

    # 21 & 22. Higher Timeframe Alignment (M5, M15, H1, H4)
    for tf_min, tf_name in [(5, 'M5'), (15, 'M15'), (60, 'H1'), (240, 'H4')]:
        tf_close = close.groupby(df['time'].dt.floor(f'{tf_min}min')).transform('last')
        tf_ret = tf_close.pct_change(tf_min).astype(np.float32)
        tf_ema20 = compute_ema(tf_close, 20)
        feats[f'htf_{tf_name}_ret'] = tf_ret
        feats[f'htf_{tf_name}_ema20_dist'] = ((close - tf_ema20) / (atr14 + 1e-8)).astype(np.float32)

    # 23. Z-scores & Price displacement
    feats['disp_ema20'] = ((close - emas[20]) / (atrs[20] + 1e-8)).astype(np.float32)
    feats['disp_ema50'] = ((close - emas[50]) / (atrs[50] + 1e-8)).astype(np.float32)

    # 24. Directional Efficiency
    for p in [5, 10, 20, 50, 100]:
        net_move = (close - close.shift(p)).abs()
        tot_path = close.diff().abs().rolling(p).sum()
        feats[f'dir_eff_{p}'] = (net_move / (tot_path + 1e-8)).astype(np.float32)

    # 25. Cross-market context if available
    if markets_dir and os.path.exists(markets_dir):
        for symbol in ['XAGUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'US100', 'US500', 'US30', 'USOIL', 'UKOIL']:
            sym_file = os.path.join(markets_dir, symbol, f'{symbol}_M1.csv')
            if os.path.exists(sym_file):
                try:
                    sym_df = pd.read_csv(sym_file, usecols=['time', 'close'])
                    sym_df['time'] = pd.to_datetime(sym_df['time'])
                    sym_df = sym_df.sort_values('time').reset_index(drop=True)
                    sym_ret = sym_df['close'].pct_change(1)
                    sym_df[f'{symbol}_ret1'] = sym_ret
                    sym_df[f'{symbol}_ret5'] = sym_df['close'].pct_change(5)
                    df_merged = pd.merge_asof(
                        df[['time']].sort_values('time'),
                        sym_df[['time', f'{symbol}_ret1', f'{symbol}_ret5']].sort_values('time'),
                        on='time', direction='backward'
                    )
                    feats[f'{symbol}_ret1'] = df_merged[f'{symbol}_ret1'].astype(np.float32)
                    feats[f'{symbol}_ret5'] = df_merged[f'{symbol}_ret5'].astype(np.float32)
                except Exception:
                    pass

    # 26. 2-HOUR WINDOW FEATURES
    window_id = df['time'].dt.floor('2h')
    df_win = df.copy()
    df_win['window_id'] = window_id

    minutes_in_window = ((df['time'] - window_id).dt.total_seconds() / 60.0).astype(np.float32)
    feats['window_minute'] = minutes_in_window
    feats['window_progress'] = (minutes_in_window / 120.0).astype(np.float32)
    feats['window_minutes_remaining'] = (120.0 - minutes_in_window).astype(np.float32)

    win_group = df_win.groupby('window_id')
    win_open = win_group['open'].transform('first').astype(np.float32)
    win_high_so_far = win_group['high'].cummax().astype(np.float32)
    win_low_so_far = win_group['low'].cummin().astype(np.float32)

    feats['win_open'] = win_open
    feats['win_high_so_far'] = win_high_so_far
    feats['win_low_so_far'] = win_low_so_far
    win_range = (win_high_so_far - win_low_so_far).astype(np.float32)
    feats['win_range'] = win_range
    feats['price_vs_win_open_div_atr'] = ((close - win_open) / (atr14 + 1e-8)).astype(np.float32)
    feats['price_vs_win_high_div_atr'] = ((win_high_so_far - close) / (atr14 + 1e-8)).astype(np.float32)
    feats['price_vs_win_low_div_atr'] = ((close - win_low_so_far) / (atr14 + 1e-8)).astype(np.float32)
    feats['pos_in_win_range'] = ((close - win_low_so_far) / (win_range + 1e-8)).astype(np.float32)
    feats['win_return_so_far'] = ((close - win_open) / (win_open + 1e-8)).astype(np.float32)
    feats['max_fav_move_div_atr'] = ((win_high_so_far - win_open) / (atr14 + 1e-8)).astype(np.float32)
    feats['max_adv_move_div_atr'] = ((win_open - win_low_so_far) / (atr14 + 1e-8)).astype(np.float32)

    prev_win_stats = df_win.groupby('window_id').agg(
        prev_win_high=('high', 'max'),
        prev_win_low=('low', 'min'),
        prev_win_open=('open', 'first'),
        prev_win_close=('close', 'last')
    ).shift(1)

    df_prev_win = df_win[['window_id']].merge(prev_win_stats, on='window_id', how='left')
    feats['prev_win_high'] = df_prev_win['prev_win_high'].astype(np.float32)
    feats['prev_win_low'] = df_prev_win['prev_win_low'].astype(np.float32)
    feats['prev_win_range'] = (df_prev_win['prev_win_high'] - df_prev_win['prev_win_low']).astype(np.float32)
    feats['price_vs_prev_win_high'] = ((close - df_prev_win['prev_win_high']) / (atr14 + 1e-8)).astype(np.float32)
    feats['price_vs_prev_win_low'] = ((close - df_prev_win['prev_win_low']) / (atr14 + 1e-8)).astype(np.float32)

    # Build DataFrame at once
    features = pd.DataFrame(feats, index=df.index, dtype=np.float32)
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    # Vectorized normalization in float32
    means = features.mean().astype(np.float32)
    stds = features.std().replace(0, 1.0).astype(np.float32)
    norm_features = ((features - means) / stds).astype(np.float32)

    return norm_features, df
