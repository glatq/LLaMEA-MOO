from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from pymoo.problems import get_problem

from .re_problems import get_re_problem


@dataclass
class ProblemSpec:
    """Static metadata for a multi-objective problem."""

    name: str
    dim: int
    n_obj: int
    lb: np.ndarray
    ub: np.ndarray
    ref_point: Optional[np.ndarray] = None


class _PymooProblemWrapper:
    """
    Lightweight wrapper that presents a uniform interface to the evaluator.

    Exposes:
        - name: str
        - bounds: np.ndarray of shape (2, dim)
        - n_obj: int
        - ref_point: Optional[np.ndarray]
        - __call__(x) -> np.ndarray of shape (n_obj,)
    """

    def __init__(self, prob, spec: ProblemSpec):
        self._prob = prob
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def bounds(self) -> np.ndarray:
        """Shape (2, dim): [lb; ub]."""
        return np.vstack([self._spec.lb, self._spec.ub])

    @property
    def n_obj(self) -> int:
        return self._spec.n_obj

    @property
    def ref_point(self) -> Optional[np.ndarray]:
        return self._spec.ref_point

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        F = self._prob.evaluate(x, return_values_of=["F"])[0]
        return np.asarray(F, dtype=float)


class PymooMOProvider:
    """
    Provider of multi-objective test problems via pymoo.

    get(problem_id, dim, ref_point=None, seed=None, **kwargs)
    -> wrapper with __call__, name, bounds, n_obj, ref_point.
    """

    def get(
        self,
        problem_id: str,
        dim: int,
        ref_point: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> _PymooProblemWrapper:
        # Real-world RE problems are provided locally (not available in pymoo);
        # everything else (ZDT/DTLZ/WFG/...) goes through pymoo as before.
        prob = get_re_problem(problem_id)
        if prob is None:
            base_kwargs = dict(n_var=dim, **kwargs)

            # Some problems (DTLZ) take n_obj; others (ZDT, BNH) don't.
            try:
                prob = get_problem(problem_id, **base_kwargs)
            except TypeError:
                # First, try again without n_obj (for problems with fixed #objectives)
                if "n_obj" in base_kwargs:
                    base_kwargs = {k: v for k, v in base_kwargs.items() if k != "n_obj"}
                try:
                    prob = get_problem(problem_id, **base_kwargs)
                except TypeError:
                    # Final fallback: also drop n_var (for fixed-dim problems like BNH)
                    base_kwargs = {k: v for k, v in base_kwargs.items() if k != "n_var"}
                    if base_kwargs:
                        prob = get_problem(problem_id, **base_kwargs)
                    else:
                        prob = get_problem(problem_id)

        if seed is not None and hasattr(prob, "rng"):
            try:
                prob.rng = np.random.default_rng(seed)
            except Exception:
                pass

        def _as_1d(a, fallback):
            if a is None:
                return np.full(dim, fallback, dtype=float)
            arr = np.asarray(a, dtype=float).ravel()
            return np.resize(arr, dim)

        xl = getattr(prob, "xl", None)
        xu = getattr(prob, "xu", None)
        lb = _as_1d(xl, -5.0)
        ub = _as_1d(xu, 5.0)

        n_obj = int(getattr(prob, "n_obj", 2))

        spec = ProblemSpec(
            name=str(problem_id),
            dim=int(dim),
            n_obj=n_obj,
            lb=lb,
            ub=ub,
            ref_point=(
                None if ref_point is None else np.asarray(ref_point, float).ravel()
            ),
        )
        return _PymooProblemWrapper(prob, spec)
