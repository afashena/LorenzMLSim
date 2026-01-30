# train_neural_ode.py
from pathlib import Path
from pyexpat import model
import jax.numpy as jnp

from data.generate_data import generate_dataset
from neural_ode import NeuralODE
from utils.util import save_params


# -----------------------
# MAIN TRAINING LOOP
# -----------------------

def main():

    chckpt_path = Path(r"C:\Users\BabyBunny\Documents\Repos\LorenzMLSim\checkpoints")

    neural_ode = NeuralODE()

    data = generate_dataset()
    print("Done generating dataset.")

    # Curriculum: increasing rollout horizon
    horizons = [(10, 1000), (25, 800), (50, 500), (100, 200)]

    for horizon in horizons:
        ts = jnp.linspace(0.0, horizon[0] * neural_ode.dt, horizon[0])
        print(f"\nTraining with horizon = {horizon}")

        for epoch in range(horizon[1]):
            neural_ode.params, neural_ode.opt_state, loss = neural_ode.train_step(
                neural_ode.params,
                neural_ode.opt_state,
                data[:, :horizon[0]],
                ts,
            )

            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Loss {loss:.6f}")
                save_params(neural_ode.params, chckpt_path / f"chckpt_hrzn_{horizon}_ep_{epoch}_loss_{loss:.6f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
