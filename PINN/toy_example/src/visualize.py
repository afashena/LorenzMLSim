import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from model import PINN

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\PINN\toy_example\src\pinn_burgers_model_90_loss_1.444e-01.pt")
OUTPUT_VIDEO = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\PINN\toy_example\src") / "burgers2d_evolution.gif"

NX, NY = 100, 100          # spatial resolution
NT = 200                   # number of frames
T_MAX = 2.0                # final time
FPS = 20                   # ~10 seconds total

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model
# -----------------------------
model = PINN([3, 128, 128, 128, 1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Spatial grid
# -----------------------------
x = np.linspace(-1, 1, NX)
y = np.linspace(-1, 1, NY)
X, Y = np.meshgrid(x, y)

x_t = torch.tensor(X.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
y_t = torch.tensor(Y.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    np.zeros((NY, NX)),
    extent=[-1, 1, -1, 1],
    origin="lower",
    cmap="viridis",
    vmin=-1,
    vmax=1,
)
ax.set_title("2D Burgers' Equation (PINN)")
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("u(x, y, t)")


# -----------------------------
# Animation update
# -----------------------------
@torch.no_grad()
def update(frame):
    t_val = frame / (NT - 1) * T_MAX
    t_t = torch.full_like(x_t, t_val)

    X = torch.cat([x_t, y_t, t_t], dim=1).to(DEVICE)

    u = model(X)
    u = u.cpu().numpy().reshape(NY, NX)

    im.set_array(u)
    ax.set_title(f"2D Burgers' Equation (t = {t_val:.3f})")
    return [im]


# -----------------------------
# Create animation
# -----------------------------
anim = FuncAnimation(
    fig,
    update,
    frames=NT,
    interval=1000 / FPS,
)

# -----------------------------
# Save movie
# -----------------------------
print("Saving video...")
anim.save(OUTPUT_VIDEO, fps=FPS, dpi=150)
print(f"Saved: {OUTPUT_VIDEO}")

plt.close()
