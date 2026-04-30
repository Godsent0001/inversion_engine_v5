import numpy as np


# -------------------------
# LOAD RESULTS (ROBUST)
# -------------------------
def load_results():
    data = np.load("outputs/survivors.npy", allow_pickle=True).item()
    metrics = data["metrics"]

    # Ensure required metrics exist
    if "returns" not in metrics:
        metrics["returns"] = metrics["final_equity"] - 1.0

    if "sharpe" not in metrics:
        metrics["sharpe"] = np.zeros_like(metrics["returns"])

    if "win_rate" not in metrics:
        if "winrate" in metrics:
            metrics["win_rate"] = metrics["winrate"]
        else:
            metrics["win_rate"] = np.zeros_like(metrics["returns"])

    return metrics, data["population"], data.get("stats", None)


# -------------------------
# SUMMARY
# -------------------------
def summary(metrics):
    returns = metrics["returns"]
    dd = metrics["max_drawdown"]

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
# RISK-ADJUSTED SCORING (IMPROVED)
# -------------------------
def compute_score(metrics):
    returns = metrics["returns"]
    dd = metrics["max_drawdown"]

    # Better universal scoring (works for all strategy types)
    score = returns / (dd + 0.05)

    return score


# -------------------------
# TOP AGENTS
# -------------------------
def top_agents(metrics, pop, top_n=10):

    score = compute_score(metrics)
    idx = np.argsort(score)[-top_n:][::-1]

    print("\n=== TOP AGENTS (RETURN / DD) ===")

    for i in idx:
        fam = pop.get("family", ["N/A"] * len(score))[i]

        print(
            f"Agent {i} | "
            f"Family: {fam} | "
            f"Return: {metrics['returns'][i]:.3f} | "
            f"DD: {metrics['max_drawdown'][i]:.3f} | "
            f"WinRate: {metrics['win_rate'][i]:.3f} | "
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
# PARAMETER ANALYSIS
# -------------------------
def parameter_analysis(metrics, pop):

    print("\n=== PARAMETER INSIGHT ===")

    returns = metrics["returns"]

    # RRR
    print("\n-- RRR --")
    for r in sorted(np.unique(pop["rrr"])):
        mask = pop["rrr"] == r
        if np.sum(mask) > 0:
            print(f"RRR {r}: Avg Return = {returns[mask].mean():.3f}")

    # ATR MULTIPLIER
    print("\n-- ATR MULTIPLIER --")
    for a in sorted(np.unique(pop["atr"])):
        mask = pop["atr"] == a
        if np.sum(mask) > 0:
            print(f"ATR {a}: Avg Return = {returns[mask].mean():.3f}")

    # THRESHOLD
    print("\n-- THRESHOLD BUCKETS --")
    bins = [0.3, 0.5, 0.7, 0.9]

    for i in range(len(bins) - 1):
        mask = (pop["threshold"] >= bins[i]) & (pop["threshold"] < bins[i+1])

        if np.sum(mask) > 0:
            print(
                f"{bins[i]:.1f}-{bins[i+1]:.1f}: "
                f"Avg Return = {returns[mask].mean():.3f}"
            )


# -------------------------
# PORTFOLIO HINT (VERY IMPORTANT)
# -------------------------
def portfolio_hint(metrics, pop):

    print("\n=== PORTFOLIO HINT ===")

    score = compute_score(metrics)
    top_mask = score > np.percentile(score, 90)

    families = pop.get("family", None)

    if families is None:
        print("No family info available.")
        return

    unique = np.unique(families)

    for f in unique:
        mask = (families == f) & top_mask
        print(f"{f}: {np.sum(mask)} agents in top 10%")


# -------------------------
# MAIN PIPELINE
# -------------------------
def run_analysis(metrics, pop):

    summary(metrics)
    distribution(metrics)
    family_analysis(metrics, pop)
    parameter_analysis(metrics, pop)
    top_agents(metrics, pop, top_n=10)
    portfolio_hint(metrics, pop)


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
        print("Make sure simulation has been run first.")
