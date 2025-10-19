import torch


def black_scholes_pde_loss(model, X: torch.Tensor, sigma: float = 0.4, r: float = 0.03) -> torch.Tensor:
    # X = [t, S]
    X = X.clone().detach().requires_grad_(True)
    u_pred = model(X)

    grads = torch.autograd.grad(u_pred.sum(), X, create_graph=True)[0]
    u_t, u_s = grads[:, 0:1], grads[:, 1:2]

    grads2 = torch.autograd.grad(u_s.sum(), X, create_graph=True)[0]
    u_ss = grads2[:, 1:2]

    S = X[:, 1:2]
    f_pred = u_t + 0.5 * sigma**2 * S**2 * u_ss + (r * S * u_s) - r * u_pred
    return torch.mean(f_pred**2)
