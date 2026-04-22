import numpy as np


def ema(arr, period):
    alpha = 2 / (period + 1)
    out = np.zeros_like(arr, dtype=np.float32)
    out[0] = arr[0]

    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]

    return out


def compute_dmao(close, atr,
                 smoothing=3,
                 memory=10,
                 cons_window=10):
    """
    Returns:
        cil, hal
    """

    n = len(close)

    # -------------------------
    # 1. RAW CIL (momentum / ATR)
    # -------------------------
    move = np.diff(close, prepend=close[0])

    raw_cil = np.where(
        atr > 0,
        (move / (atr + 1e-8)) * 10.0,
        0
    ).astype(np.float32)

    # -------------------------
    # 2. SMOOTHED CIL
    # -------------------------
    cil = ema(raw_cil, smoothing)

    # -------------------------
    # 3. MEMORY (EMA of CIL)
    # -------------------------
    mem = ema(cil, memory)

    # -------------------------
    # 4. CONSISTENCY
    # -------------------------
    abs_raw = np.abs(raw_cil)

    total = np.convolve(abs_raw, np.ones(cons_window), mode="same")

    # counter-force: opposite direction to current CIL
    sign_cil = np.sign(cil)
    sign_raw = np.sign(raw_cil)

    counter = np.where(sign_raw != sign_cil, abs_raw, 0)
    counter_sum = np.convolve(counter, np.ones(cons_window), mode="same")

    consistency = np.where(
        total > 0,
        1.0 - (counter_sum / (total + 1e-8)),
        0
    ).astype(np.float32)

    # -------------------------
    # 5. HAL
    # -------------------------
    hal = mem * consistency

    return cil.astype(np.float32), hal.astype(np.float32)