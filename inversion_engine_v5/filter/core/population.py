import numpy as np

def create_filter_population(
    n_agents: int,
    input_size: int,
    hidden_size: int = 8,
    seed: int = 42
):
    """
    Creates a population of filter models.
    Inputs: [Indicators(6), Confidence, Direction]
    Outputs: [Disallow, Allow]
    """

    np.random.seed(seed)

    pop = {}

    scale = 0.5

    # Weight initialization
    w1 = np.random.randn(n_agents, input_size, hidden_size).astype(np.float32) * scale
    w2 = np.random.randn(n_agents, hidden_size, 2).astype(np.float32) * scale

    # Sparsity
    w1_mask = np.random.rand(n_agents, input_size, hidden_size) > 0.3
    w2_mask = np.random.rand(n_agents, hidden_size, 2) > 0.3

    w1 *= w1_mask
    w2 *= w2_mask

    # Biases
    b1 = np.random.randn(n_agents, hidden_size).astype(np.float32) * 0.2
    b2 = np.random.randn(n_agents, 2).astype(np.float32) * 0.2

    pop["threshold"] = np.random.uniform(0.5, 0.9, n_agents).astype(np.float32)

    pop["w1"] = w1
    pop["w2"] = w2
    pop["b1"] = b1
    pop["b2"] = b2

    return pop
