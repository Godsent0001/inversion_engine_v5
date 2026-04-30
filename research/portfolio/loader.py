import pickle
import numpy as np


# -------------------------
# LOAD EXPORT BUNDLE
# -------------------------
def load_agents(path):
    """
    Load exported agents for live trading
    """

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    print(f"Loaded {bundle['n_agents']} agents")

    return bundle


# -------------------------
# REBUILD RUNTIME STATE
# -------------------------
def initialize_runtime(bundle):
    """
    Add runtime state needed for live execution
    """

    n = bundle["n_agents"]

    state = {}

    # Copy model + strategy
    for k, v in bundle.items():
        state[k] = v

    # Add runtime fields
    state["positions"] = np.zeros(n, dtype=np.int8)
    state["entry_price"] = np.zeros(n, dtype=np.float32)
    state["cooldown_counter"] = np.zeros(n, dtype=np.int32)

    return state


# -------------------------
# SINGLE STEP INFERENCE
# -------------------------
def forward_pass(state, features):
    """
    Run NN for all agents on one feature vector
    """

    w1 = state["w1"]
    w2 = state["w2"]
    b1 = state["b1"]
    b2 = state["b2"]

    # Layer 1
    h = np.tanh(np.einsum("ai,aij->aj", features, w1) + b1)

    # Output layer
    out = np.einsum("aj,ajk->ak", h, w2) + b2

    return out


# -------------------------
# DECISION FUNCTION
# -------------------------
def decide(state, features):
    """
    Convert NN output → trading decisions
    """

    logits = forward_pass(state, features)

    # Softmax
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    action = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    # Apply threshold
    mask = confidence > state["threshold"]

    # 0 = hold, 1 = buy, 2 = sell
    decisions = np.zeros_like(action)

    decisions[(action == 1) & mask] = 1
    decisions[(action == 2) & mask] = -1

    return decisions, confidence