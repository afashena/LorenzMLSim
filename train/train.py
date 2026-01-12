# train_neural_ode.py
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import diffrax
import optax

from data.generate_data import generate_dataset
from utils.util import save_params


# -----------------------
# MLP VECTOR FIELD
# -----------------------

def init_mlp(key, layers):
    params = []
    for m, n in zip(layers[:-1], layers[1:]):
        key, subkey = random.split(key)
        w = random.normal(subkey, (m, n)) * jnp.sqrt(2 / m)
        b = jnp.zeros((n,))
        params.append((w, b))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


# -----------------------
# NEURAL ODE DEFINITION
# -----------------------

def neural_ode(t, x, params):
    return mlp(params, x)


def solve_ode(params, x0, ts):
    term = diffrax.ODETerm(neural_ode)
    solver = diffrax.Tsit5()  # adaptive Runge–Kutta

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=0.01,
        y0=x0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
    )
    return sol.ys


# -----------------------
# LOSS FUNCTION
# -----------------------

def trajectory_loss(params, traj, ts):
    pred = solve_ode(params, traj[0], ts)
    return jnp.mean((pred - traj) ** 2)


batched_loss = jax.vmap(
    trajectory_loss,
    in_axes=(None, 0, None),
)


# -----------------------
# TRAINING STEP
# -----------------------

optimizer = optax.adam(1e-3)


@jax.jit
def train_step(params, opt_state, batch, ts):
    loss, grads = jax.value_and_grad(
        lambda p: jnp.mean(batched_loss(p, batch, ts))
    )(params)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# -----------------------
# MAIN TRAINING LOOP
# -----------------------

def main():

    chckpt_path = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\checkpoints")

    key = random.PRNGKey(0)
    params = init_mlp(key, [3, 128, 128, 3])
    opt_state = optimizer.init(params)

    dt = 0.01
    data = generate_dataset()
    print("Done generating dataset.")

    # Curriculum: increasing rollout horizon
    horizons = [10, 25, 50, 100]

    for horizon in horizons:
        ts = jnp.linspace(0.0, horizon * dt, horizon)
        print(f"\nTraining with horizon = {horizon}")

        for epoch in range(200):
            params, opt_state, loss = train_step(
                params,
                opt_state,
                data[:, :horizon],
                ts,
            )

            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss {loss:.6f}")
                save_params(params, chckpt_path / f"chckpt_hrzn_{horizon}_ep_{epoch}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
