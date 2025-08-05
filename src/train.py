import torch
from torch import optim
from src.models.pinn import PINN
from src.models.loss import transient_heat_residual
from src.models.boundary import apply_dirichlet_boundary

# Hyperparameters
layers = [2, 32, 32, 32, 1]
lr = 1e-3
epochs = 5000

# Physical constants
rho = 2700.0     # kg/m^3
cp = 896.0       # J/kg.K
k = 237.0        # W/m.K

# Model and optimizer
model = PINN(layers)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Domain samples
N_f = 1000  # Collocation points
x_f = torch.rand(N_f, 1)
t_f = torch.rand(N_f, 1)

# Boundary conditions at x=0
N_b = 100
x_b = torch.zeros(N_b, 1)
t_b = torch.rand(N_b, 1)
T_b_val = 1000.0
xt_b, T_b = apply_dirichlet_boundary(x_b, t_b, T_b_val)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # PDE residual
    res = transient_heat_residual(x_f, t_f, model, rho, cp, k)
    loss_f = torch.mean(res**2)

    # Boundary loss
    pred_b = model(xt_b)
    loss_b = torch.mean((pred_b - T_b)**2)

    loss = loss_f + loss_b
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Total Loss: {loss.item():.4e}")

# Save model
torch.save(model.state_dict(), "pinn_thermal.pth")
