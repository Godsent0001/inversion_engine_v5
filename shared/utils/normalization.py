import numpy as np


# -------------------------
# SAFE TANH NORMALIZATION
# -------------------------
def tanh_norm(x, scale=1.0):
    """
    Smooth normalization to [-1, 1]
    """

    return np.tanh(x / scale)


# -------------------------
# MIN-MAX NORMALIZATION
# -------------------------
def minmax_norm(x, min_val=None, max_val=None):
    """
    Normalize to [0, 1]
    """

    if min_val is None:
        min_val = np.min(x)

    if max_val is None:
        max_val = np.max(x)

    denom = max_val - min_val + 1e-8

    return (x - min_val) / denom


# -------------------------
# Z-SCORE NORMALIZATION
# -------------------------
def zscore_norm(x):
    """
    Standardization (mean=0, std=1)
    """

    mean = np.mean(x)
    std = np.std(x) + 1e-8

    return (x - mean) / std


# -------------------------
# CLIPPING
# -------------------------
def clip(x, min_val=-1.0, max_val=1.0):
    return np.clip(x, min_val, max_val)


# -------------------------
# FEATURE NORMALIZATION PIPELINE
# -------------------------
def normalize_features(features):
    """
    Normalize each feature column independently
    """

    norm = np.zeros_like(features, dtype=np.float32)

    for i in range(features.shape[1]):
        col = features[:, i]

        # Strategy:
        # - z-score → remove scale
        # - tanh → squash outliers

        col = zscore_norm(col)
        col = tanh_norm(col, scale=2.0)

        norm[:, i] = col

    return norm.astype(np.float32)


# -------------------------
# ONLINE NORMALIZATION (LIVE USE)
# -------------------------
class RunningNormalizer:
    """
    For live trading (streaming data)
    """

    def __init__(self, size):
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        """
        x: shape (features,)
        """

        self.count += 1

        delta = x - self.mean
        self.mean += delta / self.count

        delta2 = x - self.mean
        self.var += delta * delta2

    def normalize(self, x):
        std = np.sqrt(self.var / self.count) + 1e-8
        return (x - self.mean) / std