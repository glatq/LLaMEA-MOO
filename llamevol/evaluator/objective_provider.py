# objective_provider.py
from typing import Protocol, Optional
import numpy as np


class ObjectiveLike(Protocol):
    name: str
    bounds: np.ndarray
    optimum_x: Optional[np.ndarray]
    optimum_y: Optional[float]
    evaluations: int

    def __call__(self, x: np.ndarray) -> float:
        ...


class ObjectiveProvider(Protocol):
    def get(self, problem_id: int, instance_id: int, dim: int) -> ObjectiveLike:
        ...
