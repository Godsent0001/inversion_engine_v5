import numpy as np


def forward(pop, x):
    """
    Forward pass for ALL agents (vectorized)

    x: (n_features,)
    returns: (n_agents, 3)
    """

    # Expand input
    x = x.astype(np.float32)[None, :]  # (1, features)

    # -------------------------
    # LAYER 1
    # -------------------------
    # (1, f) x (n, f, h) -> (n, h)
    h = np.einsum('ij,ijk->ik', x, pop["w1"]) + pop["b1"]

    # Activation
    h = np.tanh(h)

    # -------------------------
    # LAYER 2
    # -------------------------
    # (n, h) x (n, h, 3) -> (n, 3)
    out = np.einsum('ij,ijk->ik', h, pop["w2"]) + pop["b2"]

    return out.astype(np.float32)