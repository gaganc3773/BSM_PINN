import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_sizes=(50, 50), activation: str = "tanh"):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_layer: nn.Module
        if activation == "relu":
            act_ctor = nn.ReLU
        elif activation == "gelu":
            act_ctor = nn.GELU
        else:
            act_ctor = nn.Tanh
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_ctor())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
