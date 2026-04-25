import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers = []
        in_d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

