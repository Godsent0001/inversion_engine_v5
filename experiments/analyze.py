import numpy as np
import pickle


# -------------------------
# LOAD RESULTS
# -------------------------
def load_results():
    data = np.load("outputs/survivors.npy", allow_pickle=True).item()
    metrics = data["metrics"]

    # Add calculated metrics if missing
    if "returns" not in metrics:
        metrics["returns"] = metrics["final_equity"] - 1.0
    if "sharpe" not in metrics:
        metrics["sharpe"] = np.zeros_like(metrics["final_equity"])
    if "win_rate" not in metrics and "winrate" in metrics:
        metrics["win_rate"] = metrics["winrate"]

    return metrics, data["population"], data.get("stats", None)


# -------------------------
# SUMMARY
# -------------------------
def summary(metrics):
    returns = metrics["returns"]
    dd = metrics["max_drawdown"]
    sharpe = metrics["sharpe"]

    print("\n=== SUMMARY ===")
    print(f"Agents: {len(returns)}")
    print(f"Avg Return: {np.mean(returns):.4f}")
    print(f"Median Return: {np.median(returns):.4f}")
    print(f"Max Return: {np.max(returns):.4f}")
    print(f"Min Return: {np.min(returns):.4f}")
    print(f"Avg Drawdown: {np.mean(dd):.4f}")


# -------------------------
# DISTRIBUTION
# -------------------------
def distribution(metrics):
    returns = metrics["returns"]

    print("\n=== DISTRIBUTION ===")
    print(f"% Profitable: {(returns > 0).mean() * 100:.2f}%")
    print(f"% > 50% Return: {(returns > 0.5).mean() * 100:.2f}%")
    print(f"% > 100% Return: {(returns > 1.0).mean() * 100:.2f}%")
    print(f"% Blow-ups (< -50%): {(returns < -0.5).mean() * 100:.2f}%")


# -------------------------
# RISK-ADJUSTED SCORING
# -------------------------
def compute_score(metrics):
    returns = metrics["returns"]
    dd = metrics["max_drawdown"]
    # score = (returns * 0.6) - (dd * 0.8)
    # Simple score for now
    score = returns / (dd + 0.1)

    return score


# -------------------------
# TOP AGENTS
# -------------------------
def top_agents(metrics, pop, top_n=10):

    score = compute_score(metrics)
    idx = np.argsort(score)[-top_n:][::-1]

    print("\n=== TOP AGENTS (EQUITY/DD RATIO) ===")

    for i in idx:
        fam = pop.get("family", ["N/A"] * len(score))[i]

        print(
            f"Agent {i} | "
            f"Family: {fam} | "
            f"Return: {metrics['returns'][i]:.3f} | "
            f"DD: {metrics['max_drawdown'][i]:.3f} | "
            f"Trades: {metrics['trades'][i]}"
        )

    return idx


# -------------------------
# FAMILY ANALYSIS
# -------------------------
def family_analysis(metrics, pop):

    if "family" not in pop:
        print("\n(No family data found)")
        return

    print("\n=== STRATEGY FAMILY ANALYSIS ===")

    families = np.unique(pop["family"])
    returns = metrics["returns"]

    for f in families:
        mask = pop["family"] == f

        if np.sum(mask) == 0:
            continue

        print(
            f"{f.upper():>10} | "
            f"Count: {np.sum(mask):4d} | "
            f"Avg Return: {returns[mask].mean():.3f} | "
            f"Win %: {(returns[mask] > 0).mean()*100:.1f}%"
        )


# -------------------------
# MAIN PIPELINE
# -------------------------
def run_analysis(metrics, pop):

    summary(metrics)
    distribution(metrics)
    family_analysis(metrics, pop)
    top_agents(metrics, pop, top_n=10)


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":

    print("Loading results...")

    try:
        metrics, pop, stats = load_results()
        run_analysis(metrics, pop)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error loading data:", e)
