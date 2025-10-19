from __future__ import annotations
import torch
import torch.optim as optim
from typing import Tuple
from .losses import black_scholes_pde_loss


def make_collocation_points(n_f: int = 10000, t_max: float = 3.0, s_max: float = 500.0, device: str | torch.device = "cpu") -> torch.Tensor:
    t_samples = torch.rand(n_f, 1, device=device) * t_max
    s_samples = torch.rand(n_f, 1, device=device) * s_max
    return torch.cat((t_samples, s_samples), dim=1)


def train_pinn(
    model: torch.nn.Module,
    n_epochs: int = 200,
    n_f: int = 10000,
    lr: float = 5e-3,
    t_max: float = 3.0,
    s_max: float = 500.0,
    sigma: float = 0.4,
    r: float = 0.03,
    device: str | torch.device = "cpu",
) -> Tuple[list[float], torch.nn.Module]:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        X_f = make_collocation_points(n_f=n_f, t_max=t_max, s_max=s_max, device=device)
        loss = black_scholes_pde_loss(model, X_f, sigma=sigma, r=r)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses, model
