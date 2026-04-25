import torch
import torch.nn as nn


class PPOMLP(nn.Module):
    """
    Simple shared trunk with:
      - policy head: logits for 3 Bernoulli actions (keep/switch per intersection)
      - value head: scalar V(s)
    """

    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(max(1, depth)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.Tanh())
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(d, action_dim)
        self.value = nn.Linear(d, 1)

    def forward(self, x):
        z = self.trunk(x)
        logits = self.policy(z)
        v = self.value(z).squeeze(-1)
        return logits, v

