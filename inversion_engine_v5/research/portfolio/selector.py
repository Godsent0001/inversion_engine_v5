import numpy as np


# -------------------------
# SCORE FUNCTION
# -------------------------
def compute_score(metrics):
    returns = metrics["returns"]
    dd = metrics["max_drawdown"]
    sharpe = metrics["sharpe"]

    # Balanced score
    score = (returns * 0.6) + (sharpe * 0.4) - (dd * 0.8)

    return score


# -------------------------
# SELECT TOP AGENTS
# -------------------------
def select_top_agents(metrics, top_k=20):

    score = compute_score(metrics)

    idx = np.argsort(score)[-top_k:][::-1]

    return idx


# -------------------------
# DIVERSIFIED SELECTION
# -------------------------
def diversified_selection(metrics, pop, top_k=20):
    """
    Ensures we don't pick only one strategy type
    """

    if "family" not in pop:
        return select_top_agents(metrics, top_k)

    score = compute_score(metrics)

    families = pop["family"]
    unique_families = np.unique(families)

    selected = []

    # Allocate slots per family
    per_family = max(1, top_k // len(unique_families))

    for f in unique_families:

        mask = families == f
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        family_scores = score[idx]
        top_idx = idx[np.argsort(family_scores)[-per_family:]]

        selected.extend(top_idx.tolist())

    # Fill remaining slots globally
    if len(selected) < top_k:
        remaining = list(set(range(len(score))) - set(selected))
        extra = sorted(remaining, key=lambda i: score[i], reverse=True)
        selected.extend(extra[:top_k - len(selected)])

    return np.array(selected[:top_k])


# -------------------------
# BUILD PORTFOLIO WEIGHTS
# -------------------------
def build_weights(metrics, selected_idx):
    """
    Weight agents based on score
    """

    score = compute_score(metrics)
    s = score[selected_idx]

    # Normalize weights
    weights = s - s.min() + 1e-6
    weights = weights / weights.sum()

    return weights


# -------------------------
# MAIN SELECTOR PIPELINE
# -------------------------
def build_portfolio(metrics, pop, top_k=20):

    selected_idx = diversified_selection(metrics, pop, top_k)
    weights = build_weights(metrics, selected_idx)

    print("\n=== PORTFOLIO ===")
    print(f"Selected {len(selected_idx)} agents")

    return selected_idx, weights