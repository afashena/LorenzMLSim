from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from model import PINN
from physics import burgers_residual

print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Domain
N_f = 20000
N_bc = 2000
N_ic = 2000

nu = 0.01

# Collocation points

# make x and y dimensions be random samples from [-1, 1)
x = torch.rand(N_f, 1) * 2 - 1
y = torch.rand(N_f, 1) * 2 - 1
t = torch.rand(N_f, 1)
X_f = torch.cat([x, y, t], dim=1).to(DEVICE)

# Initial condition: u(x,y,0) = -sin(pi x) sin(pi y)
x_ic = torch.rand(N_ic, 1) * 2 - 1
y_ic = torch.rand(N_ic, 1) * 2 - 1
t_ic = torch.zeros_like(x_ic)
X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1).to(DEVICE)
u_ic = -torch.sin(np.pi * x_ic) * torch.sin(np.pi * y_ic)

# Boundary conditions: u = 0
xb = torch.rand(N_bc, 1) * 2 - 1
tb = torch.rand(N_bc, 1)

X_bc = torch.cat([
    torch.cat([-torch.ones_like(xb), xb, tb], dim=1),
    torch.cat([torch.ones_like(xb), xb, tb], dim=1),
    torch.cat([xb, -torch.ones_like(xb), tb], dim=1),
    torch.cat([xb, torch.ones_like(xb), tb], dim=1)
], dim=0).to(DEVICE)

model = PINN([3, 128, 128, 128, 1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100

for epoch in tqdm(range(num_epochs), desc="Training"):
    optimizer.zero_grad()

    # PDE loss
    res = burgers_residual(model, X_f, nu)
    loss_pde = torch.mean(res ** 2)

    # IC loss
    u_pred_ic = model(X_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic.to(DEVICE)) ** 2)

    # BC loss
    u_pred_bc = model(X_bc)
    loss_bc = torch.mean(u_pred_bc ** 2)

    loss = loss_pde + loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: total={loss.item():.3e}, pde={loss_pde.item():.3e}")

        # Save model
        save_path = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\PINN\toy_example\src")
        torch.save(model.state_dict(), save_path / f"pinn_burgers_model_{epoch}_loss_{loss.item():.3e}.pt")