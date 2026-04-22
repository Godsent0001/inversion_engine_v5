import numpy as np


def ema(arr, period):
    alpha = 2 / (period + 1)
    out = np.zeros_like(arr, dtype=np.float32)
    out[0] = arr[0]

    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]

    return out


def compute_dllco(high, low, close, atr,
                  memory_period=12,
                  cons_window=12,
                  smoothing=3,
                  min_range_mult=0.1):
    """
    Returns:
        crl, hcl
    """

    n = len(close)

    # -------------------------
    # 1. RAW CRL (rejection)
    # -------------------------
    range_ = high - low
    min_thresh = atr * min_range_mult

    valid = (range_ > 0) & (range_ >= min_thresh)

    lower_reject = np.where(valid, (close - low) / (range_ + 1e-8), 0)
    upper_reject = np.where(valid, (high - close) / (range_ + 1e-8), 0)

    raw_crl = (lower_reject - upper_reject) * 100.0
    raw_crl = raw_crl.astype(np.float32)

    # -------------------------
    # 2. SMOOTHED CRL
    # -------------------------
    crl = ema(raw_crl, smoothing)

    # -------------------------
    # 3. MEMORY EMA
    # -------------------------
    mem = ema(raw_crl, memory_period)

    # -------------------------
    # 4. CONSISTENCY (SIGN STABILITY)
    # -------------------------
    sign = np.zeros_like(raw_crl)

    sign[raw_crl > 0.5] = 1
    sign[raw_crl < -0.5] = -1

    sign_sum = np.convolve(sign, np.ones(cons_window), mode="same")

    consistency = np.abs(sign_sum) / cons_window

    # -------------------------
    # 5. HCL
    # -------------------------
    hcl = mem * consistency

    return crl.astype(np.float32), hcl.astype(np.float32)