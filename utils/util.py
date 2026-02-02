from pathlib import Path
import orbax.checkpoint as ocp
import os

def save_params(params, path: Path):
    #os.makedirs(path, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params)

def load_params(checkpoint_path):
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(checkpoint_path)

