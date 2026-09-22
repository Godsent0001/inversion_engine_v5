import numpy as np
import torch
from shared.utils.vector_ops import forward_nn, softmax

class DecisionEngine:
    def __init__(self):
        pass

    def decide(self, agent_model, features):
        """
        Decide trading action for an agent model.

        Parameters:
        - agent_model: dict containing model parameters or 'net' (GRU model)
        - features: numpy array
            For GRU: shape (seq_len, input_dim) e.g., (10, 328)
            For legacy MLP: 1D vector of features

        Returns:
        - action: int -> 1 (BUY), -1 (SELL), 0 (HOLD/NEUTRAL)
        - confidence: float -> model probability for the chosen class
        """
        if agent_model.get("model_type") == "gru":
            net = agent_model["net"]

            if not isinstance(features, torch.Tensor):
                features_tensor = torch.from_numpy(features).float()
            else:
                features_tensor = features.float()

            if features_tensor.dim() == 2:
                # Add batch dimension: (1, seq_len, input_dim)
                features_tensor = features_tensor.unsqueeze(0)

            with torch.no_grad():
                probs = net(features_tensor).squeeze(0).numpy()

            action_idx = int(np.argmax(probs))
            confidence = float(probs[action_idx])

            # Backtest class mapping:
            # 0: BUY, 1: NEUTRAL, 2: SELL
            if action_idx == 0:
                action = 1
            elif action_idx == 2:
                action = -1
            else:
                action = 0

            return action, confidence

        else:
            w1, b1 = agent_model["w1"], agent_model["b1"]
            w2, b2 = agent_model["w2"], agent_model["b2"]

            logits = forward_nn(features, w1, b1, w2, b2)

            if "aggression" in agent_model:
                logits *= agent_model["aggression"]

            probs = softmax(logits)

            action_idx = np.argmax(probs)
            confidence = float(probs[action_idx])

            threshold = agent_model.get("threshold", 0.0)

            action = 0
            if confidence >= threshold:
                if action_idx == 1:
                    action = 1
                elif action_idx == 2:
                    action = -1

            return action, confidence
