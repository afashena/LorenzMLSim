from __future__ import annotations
from pydantic import BaseModel
import json

class TrainConfig(BaseModel):
    epochs: int = 10000
    lr: float = 1e-4
    batch_size: int | None = None  # If None, use full-batch training
    pde_weight: float = 0.5
    ic_weight: float = 0.25
    bc_weight: float = 0.25


class DataConfig(BaseModel):
    interiror_points: int = 20000
    boundary_points: int = 2000
    initial_condition_points: int = 2000
    nu: float | None = 0.01
    nu_bounds: tuple[float, float] | None = [0.001, 0.1]


class Config(BaseModel):
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return Config.model_validate(raw)  # pydantic v2
