import numpy as np


def ema(arr, period):
    alpha = 2 / (period + 1)
    out = np.zeros_like(arr, dtype=np.float32)
    out[0] = arr[0]

    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]

    return out


def compute_mrpo(close, ema_close, atr, smooth=5):
    """
    Mean Reversion Pressure Oscillator (MRPO)

    Returns:
        mrpo (scaled -100 to 100)
    """

    # -------------------------
    # 1. DEVIATION
    # -------------------------
    deviation = np.where(
        atr > 0,
        (close - ema_close) / (atr + 1e-8),
        0
    )

    # -------------------------
    # 2. REVERSION (velocity)
    # -------------------------
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    reversion = np.where(
        atr > 0,
        (prev_close - close) / (atr + 1e-8),
        0
    )

    # -------------------------
    # 3. RAW MRPO
    # -------------------------
    raw = deviation * reversion

    # -------------------------
    # 4. SMOOTH
    # -------------------------
    smooth_mrpo = ema(raw, smooth)

    # -------------------------
    # 5. SCALE & CLAMP
    # -------------------------
    scaled = smooth_mrpo * 50.0
    scaled = np.clip(scaled, -100, 100)

    return scaled.astype(np.float32)