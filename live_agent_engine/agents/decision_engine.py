import numpy as np
from utils.vector_ops import forward_nn, softmax

class DecisionEngine:
    def __init__(self):
        pass

    def decide(self, agent_model, features):
        w1, b1 = agent_model["w1"], agent_model["b1"]
        w2, b2 = agent_model["w2"], agent_model["b2"]

        logits = forward_nn(features, w1, b1, w2, b2)

        # Aggression scaling
        if "aggression" in agent_model:
            logits *= agent_model["aggression"]

        probs = softmax(logits)

        action_idx = np.argmax(probs)
        confidence = probs[action_idx]

        # 0: HOLD, 1: BUY, 2: SELL
        action = 0
        if confidence >= agent_model["threshold"]:
            if action_idx == 1:
                action = 1
            elif action_idx == 2:
                action = -1

        return action, confidence
