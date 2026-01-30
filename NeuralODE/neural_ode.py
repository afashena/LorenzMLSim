from functools import partial
import diffrax
import jax.numpy as jnp
import jax
import optax
import jax.random as random

from functools import partial
import jax
import jax.numpy as jnp
from jax import random
import optax
import diffrax


class NeuralODE:
    def __init__(self, solver=diffrax.Tsit5(), dt=0.01, params=None):
        self.solver = solver
        self.dt = dt
        self.optimizer = optax.adam(1e-3)
        self.params = params

        key = random.PRNGKey(0)
        if self.params is None:
            self.params = self.init_mlp(key, [3, 128, 128, 3])
        self.opt_state = self.optimizer.init(self.params)

    # -------------------------
    # Model definition
    # -------------------------
    def init_mlp(self, key, layers):
        params = []
        for m, n in zip(layers[:-1], layers[1:]):
            key, subkey = random.split(key)
            w = random.normal(subkey, (m, n)) * jnp.sqrt(2 / m)
            b = jnp.zeros((n,))
            params.append((w, b))
        return params

    def mlp(self, params, x):
        for w, b in params[:-1]:
            x = jnp.tanh(x @ w + b)
        w, b = params[-1]
        return x @ w + b

    # -------------------------
    # ODE definition
    # -------------------------
    def vector_field(self, t, x, params):
        return self.mlp(params, x)

    def solve_ode(self, params, x0, ts):
        term = diffrax.ODETerm(self.vector_field)

        sol = diffrax.diffeqsolve(
            term,
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt,
            y0=x0,
            args=params,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    # -------------------------
    # Losses
    # -------------------------
    def trajectory_loss(self, params, traj, ts):
        x0 = traj[0]
        pred = self.solve_ode(params, x0, ts)
        return jnp.mean((pred - traj) ** 2)

    def batched_loss(self, params, batch, ts):
        losses = jax.vmap(
            self.trajectory_loss,
            in_axes=(None, 0, None),
        )(params, batch, ts)
        return jnp.mean(losses)

    # -------------------------
    # Training step (PURE)
    # -------------------------
    @partial(jax.jit, static_argnums=0)
    def train_step(self, params, opt_state, batch, ts):
        loss, grads = jax.value_and_grad(self.batched_loss)(
            params, batch, ts
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
