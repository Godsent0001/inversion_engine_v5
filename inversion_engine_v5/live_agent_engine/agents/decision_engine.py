import numpy as np
from shared.utils.vector_ops import forward_nn, softmax

class DecisionEngine:
    def __init__(self):
        pass

    def decide(self, agent_model, features, high, low, close):
        """
        Generates trading decision and stop level based on research logic.
        """
        w1, b1 = agent_model["w1"], agent_model["b1"]
        w2, b2 = agent_model["w2"], agent_model["b2"]

        logits = forward_nn(features, w1, b1, w2, b2)

        # Aggression scaling
        if "aggression" in agent_model:
            logits *= agent_model["aggression"]

        probs = softmax(logits)

        action_idx = np.argmax(probs)
        confidence = probs[action_idx]

        # 0: HOLD, 1: BUY_STOP, -1: SELL_STOP
        action = 0
        entry_price = 0.0

        current_close = close[-1]

        if confidence >= agent_model["threshold"]:
            if action_idx == 1: # Buy Stop
                # Search backward for first High > Close
                found = False
                for k in range(len(high) - 1, -1, -1):
                    if high[k] > current_close:
                        entry_price = float(high[k])
                        found = True
                        break
                if found:
                    action = 1

            elif action_idx == 2: # Sell Stop
                # Search backward for first Low < Close
                found = False
                for k in range(len(low) - 1, -1, -1):
                    if low[k] < current_close:
                        entry_price = float(low[k])
                        found = True
                        break
                if found:
                    action = -1

        return action, confidence, entry_price
