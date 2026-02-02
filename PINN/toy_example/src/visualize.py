import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from model import PINN

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\PINN\toy_example\src\checkpoints\pinn_burgers_model_9950_loss_7.335e-03.pt")
OUTPUT_VIDEO = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\PINN\toy_example\src") / "burgers2d_evolution_weights_big.gif"

NX, NY = 100, 100          # spatial resolution
NT = 200                   # number of frames
T_MAX = 2.0                # final time
FPS = 20                   # ~10 seconds total
NU = 0.01                 # viscosity matching training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model
# -----------------------------
model = PINN([3, 128, 128, 128, 1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =============================
# Spatial grid
# =============================
x = np.linspace(-1, 1, NX)
y = np.linspace(-1, 1, NY)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# =============================
# Initial condition
# =============================
def initial_condition(x, y):
    return - np.sin(np.pi * x) * np.sin(np.pi * y)

u_truth = initial_condition(X, Y)

# Enforce BC at t=0 for Dirichlet condition
u_truth[0, :] = 0
u_truth[-1, :] = 0
u_truth[:, 0] = 0
u_truth[:, -1] = 0

# =============================
# Ground truth solver (FD)
# =============================
def burgers_step(u, dt):
    ux = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    uy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)

    uxx = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
    uyy = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dy**2

    return u + dt * (-u * ux - u * uy + NU * (uxx + uyy))

# =============================
# Ground truth solver (FD) with u=0 boundary (physically ridiculous Dirichlet BC)
# =============================
def burgers_step_dirichlet(u, dt):
    """
    One time step of 2D viscous Burgers equation with Dirichlet BC:
        u = 0 on boundary

    PDE:
        u_t + u*u_x + u*u_y = NU*(u_xx + u_yy)
    """
    u_new = u.copy()

    # Interior slice
    ui = u[1:-1, 1:-1]

    # First derivatives (central differences)
    ux = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
    uy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)

    # Second derivatives (Laplacian terms)
    uxx = (u[1:-1, 2:] - 2 * ui + u[1:-1, :-2]) / (dx**2)
    uyy = (u[2:, 1:-1] - 2 * ui + u[:-2, 1:-1]) / (dy**2)

    # Time update on interior
    u_new[1:-1, 1:-1] = ui + dt * (-ui * ux - ui * uy + NU * (uxx + uyy))

    # Enforce Dirichlet boundary condition u=0
    u_new[0, :] = 0
    u_new[-1, :] = 0
    u_new[:, 0] = 0
    u_new[:, -1] = 0

    return u_new

# Precompute truth trajectory
dt = T_MAX / NT
truth_history = []
u = u_truth.copy()
for _ in range(NT):
    truth_history.append(u.copy())
    #u = burgers_step(u, dt)
    u = burgers_step_dirichlet(u, dt)

truth_history = np.array(truth_history)


# =============================
# Prepare PINN input
# =============================
x_t = torch.tensor(X.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
y_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

# =============================
# Figure
# =============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im_pinn = axes[0].imshow(
    np.zeros((NY, NX)),
    extent=[-1, 1, -1, 1],
    origin="lower",
    cmap="viridis",
    vmin=-1,
    vmax=1,
)
axes[0].set_title("PINN Prediction")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

im_truth = axes[1].imshow(
    np.zeros((NY, NX)),
    extent=[-1, 1, -1, 1],
    origin="lower",
    cmap="viridis",
    vmin=-1,
    vmax=1,
)
axes[1].set_title("Numerical Ground Truth")
axes[1].set_xlabel("x")

fig.colorbar(im_truth, ax=axes.ravel().tolist(), shrink=0.8)

text = fig.text(
    0.5, 0.01,
    "2D Viscous Burgers' Equation: "
    "advection transports velocity, viscosity smooths gradients.\n"
    "Left: PINN solution learned from physics constraints only. "
    "Right: finite-difference numerical solution.",
    ha="center",
    fontsize=10
)

# =============================
# Animation update
# =============================
@torch.no_grad()
def update(frame):
    t_val = frame / (NT - 1) * T_MAX
    t_t = torch.full_like(x_t, t_val)

    X = torch.cat([x_t, y_t, t_t], dim=1).to(DEVICE)

    u_pinn = model(X)
    u_pinn = u_pinn.cpu().numpy().reshape(NY, NX)

    im_pinn.set_array(u_pinn)
    im_truth.set_array(truth_history[frame])

    fig.suptitle(f"Time t = {t_val:.3f}", fontsize=14)
    return [im_pinn, im_truth]

# =============================
# Create animation
# =============================
anim = FuncAnimation(
    fig,
    update,
    frames=NT,
    interval=1000 / FPS,
)

# =============================
# Save GIF
# =============================
print("Saving GIF...")
anim.save(OUTPUT_VIDEO, writer="pillow", fps=FPS)
print(f"Saved: {OUTPUT_VIDEO}")

plt.close()
