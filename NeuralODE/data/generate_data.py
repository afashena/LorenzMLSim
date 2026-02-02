# data.py
import numpy as np


def lorenz(x, sigma=10.0, rho=28.0, beta=8 / 3):
    return np.array([
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ])


def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_trajectory(x0, dt, steps):
    traj = np.zeros((steps, 3))
    x = x0
    for i in range(steps):
        traj[i] = x
        x = rk4_step(lorenz, x, dt)
    return traj


def generate_dataset(
    num_trajectories=2000,
    steps=200,
    dt=0.01,
):
    data = []
    for _ in range(num_trajectories):
        x0 = np.random.uniform(-15, 15, size=3)
        traj = generate_trajectory(x0, dt, steps)
        data.append(traj)
    return np.array(data)  # shape: (N, T, 3)
