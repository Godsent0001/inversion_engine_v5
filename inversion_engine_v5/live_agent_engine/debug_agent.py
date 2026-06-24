import pickle
import os

# =========================
# BASE PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =========================
# LIST YOUR AGENTS HERE
# =========================
AGENT_FILES = [
    "agent_2865.pkl",
    "agent_3709.pkl",
    "agent_3229.pkl",
    "agent_546.pkl",
    "agent_404.pkl",
]

# =========================
# SAFE LOAD FUNCTION
# =========================
def load_agent(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None

# =========================
# CLEAN PRINT FUNCTION
# =========================
def print_agent(agent_id, agent):
    print("\n" + "=" * 50)
    print(f"        AGENT {agent_id}")
    print("=" * 50)

    if agent is None:
        print("❌ Failed to load agent.")
        return

    # 🔑 Print important parameters first
    print("\n--- CORE PARAMETERS ---")
    print("rrr          :", agent.get("rrr"))
    print("atr          :", agent.get("atr"))
    print("threshold    :", agent.get("threshold"))
    print("aggression   :", agent.get("aggression"))
    print("cooldown     :", agent.get("cooldown"))

    # 🧠 Optional: show shapes instead of dumping huge arrays
    print("\n--- MODEL SHAPES ---")
    if "w1" in agent:
        print("w1 shape     :", getattr(agent["w1"], "shape", "N/A"))
    if "w2" in agent:
        print("w2 shape     :", getattr(agent["w2"], "shape", "N/A"))


# =========================
# MAIN LOOP
# =========================
def main():
    print("\n🔍 LOADING ALL AGENTS...\n")

    for file_name in AGENT_FILES:
        model_path = os.path.join(MODELS_DIR, file_name)

        if not os.path.exists(model_path):
            print(f"[WARNING] File not found: {model_path}")
            continue

        agent = load_agent(model_path)

        # Extract ID from filename
        agent_id = file_name.replace("agent_", "").replace(".pkl", "")

        print_agent(agent_id, agent)


if __name__ == "__main__":
    main()
