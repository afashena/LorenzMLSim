from pathlib import Path
import orbax.checkpoint as ocp
import os

def save_params(params, path: Path):
    #os.makedirs(path, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params)
