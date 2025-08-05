import torch
import matplotlib.pyplot as plt
from src.models.pinn import PINN

# Rebuild model
model = PINN([2, 32, 32, 32, 1])
model.load_state_dict(torch.load("pinn_thermal.pth"))
model.eval()

# Evaluation grid
x = torch.linspace(0, 1, 100).view(-1, 1)
t_fixed = torch.full_like(x, 0.5)  # Midpoint in time

xt = torch.cat([x, t_fixed], dim=1)
with torch.no_grad():
    T_pred = model(xt)

# Plot
plt.plot(x.numpy(), T_pred.numpy())
plt.xlabel("Position (x)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Profile at t=0.5")
plt.grid(True)
plt.show()
