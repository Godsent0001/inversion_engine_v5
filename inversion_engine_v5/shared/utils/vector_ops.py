import numpy as np


# -------------------------
# SAFE TANH
# -------------------------
def fast_tanh(x):
    """
    Fast tanh approximation (optional)
    """
    return np.tanh(x)


# -------------------------
# BATCH MATRIX MULTIPLY
# -------------------------
def batch_matmul(a, b):
    """
    Efficient batch matrix multiply
    a: (N, i, j)
    b: (N, j, k)
    → (N, i, k)
    """
    return np.einsum("nij,njk->nik", a, b)


# -------------------------
# BATCH VECTOR DOT
# -------------------------
def batch_dot(x, w):
    """
    x: (features,)
    w: (N, features, hidden)
    → (N, hidden)
    """
    return np.einsum("f,nfh->nh", x, w)


# -------------------------
# FAST FORWARD PASS
# -------------------------
def forward_nn(x, w1, b1, w2, b2):
    """
    Handles both batch (N, F) and single agent (F,) cases.
    x: (F,) or (N, F)
    w1: (F, H) or (N, F, H)
    w2: (H, 3) or (N, H, 3)
    """

    if w1.ndim == 3:
        # Batch mode
        h = np.tanh(np.einsum("f,nfh->nh", x, w1) + b1)
        out = np.einsum("nh,nhk->nk", h, w2) + b2
    else:
        # Single agent mode
        h = np.tanh(np.dot(x, w1) + b1)
        out = np.dot(h, w2) + b2

    return out


# -------------------------
# STABLE SOFTMAX
# -------------------------
def softmax(x):
    """
    Handles both (N, K) and (K,) cases.
    """
    if x.ndim == 2:
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x_shift)
        return exp / np.sum(exp, axis=1, keepdims=True)
    else:
        x_shift = x - np.max(x)
        exp = np.exp(x_shift)
        return exp / np.sum(exp)


# -------------------------
# ARGMAX WITH CONFIDENCE
# -------------------------
def argmax_with_confidence(probs):
    """
    Returns:
        action: (N,)
        confidence: (N,)
    """

    action = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    return action, confidence


# -------------------------
# MASK APPLY
# -------------------------
def apply_mask(values, mask, fill_value=0):
    """
    Replace values where mask is False
    """

    out = values.copy()
    out[~mask] = fill_value

    return out


# -------------------------
# VECTOR CLIP
# -------------------------
def clip(x, min_val=-1.0, max_val=1.0):
    return np.clip(x, min_val, max_val)


# -------------------------
# NORMALIZE ROWS
# -------------------------
def normalize_rows(x):
    """
    Normalize rows to sum to 1
    """

    s = np.sum(x, axis=1, keepdims=True) + 1e-8
    return x / s


# -------------------------
# SAFE DIVISION
# -------------------------
def safe_div(a, b):
    return a / (b + 1e-8)