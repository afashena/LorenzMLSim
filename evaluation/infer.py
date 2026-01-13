from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import diffrax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
from data.generate_data import generate_trajectory
from train.train import solve_ode
from utils.util import load_params

def main():
    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    checkpoint_path = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\checkpoints\chckpt_hrzn_(10, 300)_ep_280")  # change if needed
    params = load_params(checkpoint_path)

    # -----------------------------
    # SIMULATION SETTINGS
    # -----------------------------
    dt = 0.01
    steps = 30
    ts = jnp.linspace(0.0, dt * steps, steps)

    x0 = np.array([1.0, 1.0, 1.0])

    # -----------------------------
    # GROUND TRUTH
    # -----------------------------
    gt_traj = generate_trajectory(x0, dt, steps)

    # -----------------------------
    # NEURAL ODE PREDICTION
    # -----------------------------
    pred_traj = solve_ode(params, x0, ts)

    # -----------------------------
    # ERROR METRIC
    # -----------------------------
    error = np.linalg.norm(gt_traj - pred_traj, axis=1)

    # -----------------------------
    # PLOTTING
    # -----------------------------
    fig = plt.figure(figsize=(15, 5))

    # Trajectory comparison (x–z plane)
    ax1 = fig.add_subplot(131)
    ax1.plot(gt_traj[:, 0], gt_traj[:, 2], label="Ground Truth")
    ax1.plot(pred_traj[:, 0], pred_traj[:, 2], label="Neural ODE", linestyle="--")
    ax1.set_title("Trajectory (x-z plane)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.legend()

    # Time series (x component)
    ax2 = fig.add_subplot(132)
    ax2.plot(gt_traj[:, 0], label="Ground Truth x(t)")
    ax2.plot(pred_traj[:, 0], label="Predicted x(t)", linestyle="--")
    ax2.set_title("x(t) over time")
    ax2.set_xlabel("Time step")
    ax2.legend()

    # Error growth
    ax3 = fig.add_subplot(133)
    ax3.plot(error)
    ax3.set_yscale("log")
    ax3.set_title("Prediction Error ||x̂(t) - x(t)||")
    ax3.set_xlabel("Time step")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
