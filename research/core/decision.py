import numpy as np

from shared.utils.vector_ops import forward_nn, softmax


# -------------------------
# ACTION MAPPING
# -------------------------
# NN output index → trading action
# 0 = HOLD, 1 = BUY, 2 = SELL
ACTION_MAP = np.array([0, 1, -1], dtype=np.int8)


# -------------------------
# DECIDE FROM LOGITS
# -------------------------
def decide_from_logits(logits, threshold):
    """
    logits: (N, 3)
    threshold: (N,)
    """

    probs = softmax(logits)

    action_idx = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    # Apply confidence threshold
    mask = confidence >= threshold

    # Map to trading actions
    actions = ACTION_MAP[action_idx]

    # Apply neutral filter
    actions[~mask] = 0

    return actions.astype(np.int8), confidence


# -------------------------
# FULL BATCH DECISION
# -------------------------
def decide_batch(pop, features_t):
    """
    pop: population dict
    features_t: (features,) single timestep

    returns:
        actions: (N,) → -1, 0, +1
        confidence: (N,)
    """

    w1 = pop["w1"]
    w2 = pop["w2"]
    b1 = pop["b1"]
    b2 = pop["b2"]

    threshold = pop["threshold"]
    aggression = pop.get("aggression", None)

    # -------------------------
    # FORWARD PASS
    # -------------------------
    logits = forward_nn(features_t, w1, b1, w2, b2)

    # -------------------------
    # OPTIONAL: AGGRESSION SCALING
    # -------------------------
    if aggression is not None:
        logits = logits * aggression[:, None]

    # -------------------------
    # DECISION
    # -------------------------
    actions, confidence = decide_from_logits(logits, threshold)

    return actions, confidence


# -------------------------
# DECISION WITH COOLDOWN FILTER
# -------------------------
def apply_cooldown(actions, cooldown_counter):
    """
    Prevent trading during cooldown
    """

    mask = cooldown_counter > 0
    actions[mask] = 0

    return actions


# -------------------------
# FULL DECISION PIPELINE (OPTIONAL)
# -------------------------
def decide_with_state(pop, features_t):
    """
    Includes cooldown handling (if used directly)
    """

    actions, confidence = decide_batch(pop, features_t)

    if "cooldown_counter" in pop:
        actions = apply_cooldown(actions, pop["cooldown_counter"])

    return actions, confidence