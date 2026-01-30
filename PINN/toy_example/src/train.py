from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from model import PINN
from physics import burgers_residual
from config import Config

print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

config = Config.load(Path(__file__).parent / "config.json")

# Domain
N_f = config.data.interiror_points
N_bc = config.data.boundary_points
N_ic = config.data.initial_condition_points

nu = 0.01

def random_sine_initial_condition(X, Y, K=6, amp=1.0, decay=2.0, seed=None):
    """
    X, Y: meshgrid on [-1,1]x[-1,1]
    Returns u0(X,Y) with u=0 on boundary automatically.
    """
    rng = np.random.default_rng(seed)

    # Map [-1,1] -> [0,1]
    Xp = (X + 1.0) / 2.0
    Yp = (Y + 1.0) / 2.0

    u0 = np.zeros_like(X)

    for k in range(1, K + 1):
        for l in range(1, K + 1):
            a = rng.normal() / ((k**2 + l**2) ** (decay / 2))
            u0 += a * np.sin(k * np.pi * Xp) * np.sin(l * np.pi * Yp)

    # normalize and scale amplitude
    u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
    u0 = amp * u0
    return u0


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
optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

num_epochs = config.train.epochs

pde_weight = config.train.pde_weight
ic_weight = config.train.ic_weight
bc_weight = config.train.bc_weight

save_path = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\PINN\toy_example\src")
model_save_dir = save_path / "checkpoints"
model_save_dir.mkdir(parents=True, exist_ok=True)

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

    loss = pde_weight * loss_pde + ic_weight * loss_ic + bc_weight * loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: total={loss.item():.3e}, pde={loss_pde.item():.3e}, ic={loss_ic.item():.3e}, bc={loss_bc.item():.3e}")

        # Save model
        torch.save(model.state_dict(), model_save_dir / f"pinn_burgers_model_{epoch}_loss_{loss.item():.3e}.pt")