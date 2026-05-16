import numpy as np

def create_filter_population(
    n_filters: int,
    input_size: int,
    hidden_size: int = 8,
    seed: int = 42
):
    """
    Creates a population of neural filter models.
    Filters have 2 outputs (Allow, Disallow).
    """
    np.random.seed(seed)
    pop = {}

    scale = 0.5
    output_size = 2

    w1 = np.random.randn(n_filters, input_size, hidden_size).astype(np.float32) * scale
    w2 = np.random.randn(n_filters, hidden_size, output_size).astype(np.float32) * scale

    w1_mask = np.random.rand(n_filters, input_size, hidden_size) > 0.3
    w2_mask = np.random.rand(n_filters, hidden_size, output_size) > 0.3

    w1 *= w1_mask
    w2 *= w2_mask

    feature_bias = np.random.uniform(0.5, 1.5, (n_filters, input_size, 1)).astype(np.float32)
    w1 *= feature_bias

    b1 = np.random.randn(n_filters, hidden_size).astype(np.float32) * 0.2
    b2 = np.random.randn(n_filters, output_size).astype(np.float32) * 0.2

    pop["threshold"] = np.random.uniform(0.4, 0.8, n_filters).astype(np.float32)

    pop["w1"] = w1
    pop["w2"] = w2
    pop["b1"] = b1
    pop["b2"] = b2

    return pop
