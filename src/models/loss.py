import torch

def transient_heat_residual(x, t, model, rho, cp, k):
    xt = torch.cat([x, t], dim=1)
    xt.requires_grad = True

    T = model(xt)

    T_t = torch.autograd.grad(
        T, t, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    T_x = torch.autograd.grad(
        T, x, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    T_xx = torch.autograd.grad(
        T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True
    )[0]

    residual = rho * cp * T_t - k * T_xx
    return residual

