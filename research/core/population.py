import numpy as np


RRR_VALUES = np.array([2,3,4,5,6,7,8,9,10], dtype=np.float32)
ATR_VALUES = np.array([0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0], dtype=np.float32)


def create_population(
    n_agents: int,
    input_size: int,
    hidden_size: int = 8,
    seed: int = 42
):
    """
    Creates a heterogeneous population of neural trading agents.
    """

    np.random.seed(seed)

    pop = {}

    # =========================================================
    # 1. STRATEGY FAMILY (HIDDEN STRUCTURE)
    # =========================================================
    families = np.random.choice(
        ["scalper", "intraday", "swing", "sniper"],
        n_agents,
        p=[0.25, 0.35, 0.25, 0.15]
    )

    pop["family"] = families

    # =========================================================
    # 2. WEIGHT INITIALIZATION (CONSTRAINED RANDOMNESS)
    # =========================================================
    scale = 0.5

    w1 = np.random.randn(n_agents, input_size, hidden_size).astype(np.float32) * scale
    w2 = np.random.randn(n_agents, hidden_size, 3).astype(np.float32) * scale

    # Sparsity (important for stability + diversity)
    w1_mask = np.random.rand(n_agents, input_size, hidden_size) > 0.3
    w2_mask = np.random.rand(n_agents, hidden_size, 3) > 0.3

    w1 *= w1_mask
    w2 *= w2_mask

    # Feature biasing (some features matter more per agent)
    feature_bias = np.random.uniform(
        0.5, 1.5, (n_agents, input_size, 1)
    ).astype(np.float32)

    w1 *= feature_bias

    # =========================================================
    # 3. BIASES
    # =========================================================
    b1 = np.random.randn(n_agents, hidden_size).astype(np.float32) * 0.2
    b2 = np.random.randn(n_agents, 3).astype(np.float32) * 0.2

    # =========================================================
    # 4. RRR (Risk-Reward Ratio) — FAMILY AWARE
    # =========================================================
    rrr = np.zeros(n_agents, dtype=np.float32)

    rrr[families == "scalper"]  = np.random.choice([2,3,4], np.sum(families == "scalper"))
    rrr[families == "intraday"] = np.random.choice([3,4,5,6], np.sum(families == "intraday"))
    rrr[families == "swing"]    = np.random.choice([5,6,7,8], np.sum(families == "swing"))
    rrr[families == "sniper"]   = np.random.choice([6,7,8,9,10], np.sum(families == "sniper"))

    pop["rrr"] = rrr

    # =========================================================
    # 5. ATR MULTIPLIER — FAMILY AWARE
    # =========================================================
    atr = np.zeros(n_agents, dtype=np.float32)

    atr[families == "scalper"]  = np.random.choice([0.6,0.8,1.0], np.sum(families == "scalper"))
    atr[families == "intraday"] = np.random.choice([0.8,1.0,1.2,1.4], np.sum(families == "intraday"))
    atr[families == "swing"]    = np.random.choice([1.2,1.4,1.6,1.8], np.sum(families == "swing"))
    atr[families == "sniper"]   = np.random.choice([1.4,1.6,1.8,2.0], np.sum(families == "sniper"))

    pop["atr"] = atr

    # =========================================================
    # 6. THRESHOLD (CRITICAL UPGRADE)
    # =========================================================
    threshold = np.zeros(n_agents, dtype=np.float32)

    threshold[families == "scalper"]  = np.random.uniform(0.30, 0.55, np.sum(families == "scalper"))
    threshold[families == "intraday"] = np.random.uniform(0.45, 0.70, np.sum(families == "intraday"))
    threshold[families == "swing"]    = np.random.uniform(0.55, 0.80, np.sum(families == "swing"))
    threshold[families == "sniper"]   = np.random.uniform(0.70, 0.90, np.sum(families == "sniper"))

    pop["threshold"] = threshold.astype(np.float32)

    # =========================================================
    # 7. COOLDOWN (ANTI-OVERTRADING)
    # =========================================================
    cooldown = np.zeros(n_agents, dtype=np.int32)

    cooldown[families == "scalper"]  = np.random.randint(1, 5, np.sum(families == "scalper"))
    cooldown[families == "intraday"] = np.random.randint(3, 10, np.sum(families == "intraday"))
    cooldown[families == "swing"]    = np.random.randint(5, 20, np.sum(families == "swing"))
    cooldown[families == "sniper"]   = np.random.randint(10, 30, np.sum(families == "sniper"))

    pop["cooldown"] = cooldown

    # =========================================================
    # 8. AGGRESSION (POSITION INTENSITY)
    # =========================================================
    pop["aggression"] = np.random.uniform(0.5, 1.5, n_agents).astype(np.float32)

    # =========================================================
    # 9. NETWORK PARAMETERS
    # =========================================================
    pop["w1"] = w1
    pop["w2"] = w2
    pop["b1"] = b1
    pop["b2"] = b2

    # =========================================================
    # 10. RUNTIME STATE
    # =========================================================
    pop["positions"] = np.zeros(n_agents, dtype=np.int8)
    pop["entry_price"] = np.zeros(n_agents, dtype=np.float32)
    pop["cooldown_counter"] = np.zeros(n_agents, dtype=np.int32)

    return pop