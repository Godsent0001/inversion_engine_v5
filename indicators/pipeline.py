import numpy as np
import pandas as pd

from indicators.dmao import compute_dmao
from indicators.dllco import compute_dllco
from indicators.mrpo import compute_mrpo
from indicators.energy import compute_energy

from utils.normalization import normalize_features


# =========================================================
# DATA CLEANING (IMPORTANT FOR REALISTIC BACKTESTS)
# =========================================================
def clean_data(high, low, close):
    """
    Removes bad candles and ensures data integrity.
    """

    mask = (
        ~np.isnan(high) &
        ~np.isnan(low) &
        ~np.isnan(close) &
        (high > low)  # invalid candles removed
    )

    high = high[mask]
    low = low[mask]
    close = close[mask]

    return high.astype(np.float32), low.astype(np.float32), close.astype(np.float32)


# =========================================================
# EMA (shared)
# =========================================================
def ema(arr, period):
    # Using pandas for fast vectorized EMA
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values.astype(np.float32)


# =========================================================
# ATR (shared)
# =========================================================
def compute_atr(high, low, close, period=14):

    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate(([tr[0]], tr))

    atr = ema(tr, period)

    return atr.astype(np.float32)


# =========================================================
# FEATURE CLIPPING (stability layer)
# =========================================================
def clip_extremes(x, limit=5.0):
    """
    Prevents extreme indicator spikes from destabilizing NN.
    """
    return np.clip(x, -limit, limit)


# =========================================================
# MAIN PIPELINE
# =========================================================
def build_features(high, low, close):
    """
    Converts raw OHLC into stable ML-ready features.
    """

    # -------------------------
    # 1. CLEAN INPUT DATA
    # -------------------------
    high, low, close = clean_data(high, low, close)

    # -------------------------
    # 2. SHARED CALCULATIONS
    # -------------------------
    atr = compute_atr(high, low, close)
    ema_close = ema(close, 20)

    # -------------------------
    # 3. INDICATORS (your alpha signals)
    # -------------------------
    cil, hal = compute_dmao(close, atr)
    crl, hcl = compute_dllco(high, low, close, atr)
    mrpo = compute_mrpo(close, ema_close, atr)
    energy = compute_energy(close, atr)

    # -------------------------
    # 4. RAW FEATURE STACK
    # -------------------------
    raw_features = np.stack([
        cil,
        hal,
        crl,
        hcl,
        mrpo,
        energy
    ], axis=1).astype(np.float32)

    # -------------------------
    # 5. STABILITY CLIP (protect NN)
    # -------------------------
    raw_features = clip_extremes(raw_features, limit=10.0)

    # -------------------------
    # 6. NORMALIZATION (critical)
    # -------------------------
    features = normalize_features(raw_features)

    # -------------------------
    # 7. FINAL SAFETY CLIP
    # -------------------------
    features = np.clip(features, -1.0, 1.0)

    return features.astype(np.float32), atr