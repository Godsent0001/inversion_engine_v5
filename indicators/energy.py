import numpy as np


def ema(arr, period):
    alpha = 2 / (period + 1)
    out = np.zeros_like(arr, dtype=np.float32)
    out[0] = arr[0]

    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]

    return out


def compute_energy(close, atr, period=14, smooth=3):
    """
    Market Energy Oscillator (0–100)

    Returns:
        energy (0 to 100)
    """

    n = len(close)

    # -------------------------
    # 1. EFFICIENCY RATIO (ER)
    # -------------------------
    direction = np.abs(close - np.roll(close, period))
    direction[:period] = 0

    noise = np.abs(np.diff(close, prepend=close[0]))
    noise = np.convolve(noise, np.ones(period), mode="same")

    er = np.where(
        noise > 0,
        direction / (noise + 1e-8),
        0
    )

    # -------------------------
    # 2. STD DEV (rolling)
    # -------------------------
    cumsum = np.cumsum(np.insert(close, 0, 0))
    cumsum2 = np.cumsum(np.insert(close**2, 0, 0))

    mean = (cumsum[period:] - cumsum[:-period]) / period
    var = (cumsum2[period:] - cumsum2[:-period]) / period - mean**2

    std = np.sqrt(np.maximum(var, 0))
    std = np.pad(std, (period, 0), mode='edge')

    # -------------------------
    # 3. ENERGY BASE
    # -------------------------
    energy = er * 100.0

    # -------------------------
    # 4. VOLATILITY BOOST
    # -------------------------
    boost = np.where(
        (atr > 0) & (std > atr),
        std / (atr + 1e-8),
        1.0
    )

    energy = energy * boost

    # -------------------------
    # 5. CLAMP
    # -------------------------
    energy = np.clip(energy, 0, 100)

    # -------------------------
    # 6. SMOOTH
    # -------------------------
    energy = ema(energy, smooth)

    return energy.astype(np.float32)