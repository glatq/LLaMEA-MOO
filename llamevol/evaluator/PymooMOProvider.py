# PymooMOProvider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Any

import numpy as np
from pymoo.problems import get_problem


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
    Thin wrapper around a pymoo problem instance.

    This wrapper exposes:
      - spec: ProblemSpec with metadata
      - lb, ub: lower/upper bounds as 1D numpy arrays
      - n_obj, dim: basic scalar properties
      - __call__(x): evaluate a single point x (1D array) and
        return an M-dimensional numpy array of objective values.
    """

    def __init__(self, prob: Any, spec: ProblemSpec):
        self.prob = prob
        self.spec = spec

    @property
    def lb(self) -> np.ndarray:
        return self.spec.lb

    @property
    def ub(self) -> np.ndarray:
        return self.spec.ub

    @property
    def dim(self) -> int:
        return self.spec.dim

    @property
    def n_obj(self) -> int:
        return self.spec.n_obj

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a single point x.

        Parameters
        ----------
        x:
            1D numpy array of shape (dim,). It is assumed that the caller
            has already enforced bounds and shape.

        Returns
        -------
        np.ndarray
            1D numpy array of shape (n_obj,) with objective values
            (minimization).
        """
        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError(
                f"Expected x to be 1D with shape (dim,), got shape {x.shape}."
            )
        if x.size != self.spec.dim:
            raise ValueError(
                f"Expected x.size == dim == {self.spec.dim}, got {x.size}."
            )

        # pymoo evaluate expects 2D input; we give a batch of size 1
        X = x.reshape(1, -1)

        # Different pymoo versions return different things:
        #   - dict with key "F"
        #   - plain numpy array
        # Handle both.
        try:
            res = self.prob.evaluate(X, return_values_of=["F"])
        except TypeError:
            # Older versions may not support return_values_of as a list
            res = self.prob.evaluate(X)

        if isinstance(res, dict):
            F = res["F"]
        else:
            F = res

        F = np.asarray(F, dtype=float)
        if F.ndim == 2 and F.shape[0] == 1:
            F = F[0]
        if F.ndim != 1:
            raise RuntimeError(
                f"Unexpected objective shape from pymoo problem: {F.shape}"
            )
        if F.size != self.spec.n_obj:
            raise RuntimeError(f"Expected {self.spec.n_obj} objectives, got {F.size}.")
        return F


class PymooMOProvider:
    """
    Provider for multi-objective problems based on pymoo's problem factory.

    Usage:

        provider = PymooMOProvider(default_dim=5, problem_ids=["dtlz2"])
        problem = provider.make()  # returns _PymooProblemWrapper
        x = np.zeros(problem.dim)
        f = problem(x)  # f.shape == (n_obj,)

    The evaluator will typically hold one instance of this provider and call
    `make(...)` for each requested problem.
    """

    def __init__(
        self,
        default_dim: int = 5,
        problem_ids: Optional[Iterable[str]] = None,
    ):
        self.default_dim = int(default_dim)
        self.problem_ids = list(problem_ids) if problem_ids is not None else ["dtlz2"]

    def _resolve_problem_id(self, problem_id: Optional[str]) -> str:
        if problem_id is not None:
            return str(problem_id)
        # fallback: first configured problem
        if not self.problem_ids:
            raise ValueError("No problem_ids configured for PymooMOProvider.")
        return str(self.problem_ids[0])

    def make(
        self,
        problem_id: Optional[str] = None,
        dim: Optional[int] = None,
        ref_point: Optional[np.ndarray] = None,
    ) -> _PymooProblemWrapper:
        """
        Construct a pymoo problem and wrap it.

        Parameters
        ----------
        problem_id:
            Identifier understood by pymoo's get_problem (e.g. "dtlz2").
            If None, the first ID from self.problem_ids is used.
        dim:
            Dimensionality (number of decision variables). If None,
            self.default_dim is used. For many pymoo problems, this
            is passed as n_var to get_problem.
        ref_point:
            Optional reference point for hypervolume. If None, left unset
            in the ProblemSpec and can be filled later.

        Returns
        -------
        _PymooProblemWrapper
        """
        name = self._resolve_problem_id(problem_id)
        dim = int(dim) if dim is not None else self.default_dim

        # Try to construct with n_var first; fall back to simpler call if needed.
        try:
            prob = get_problem(name, n_var=dim)
        except TypeError:
            prob = get_problem(name)

        # Bounds from pymoo problem; fall back to [-5, 5] if missing
        lb = getattr(prob, "xl", None)
        ub = getattr(prob, "xu", None)

        if lb is None or ub is None:
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)
        else:
            lb = np.asarray(lb, dtype=float).ravel()
            ub = np.asarray(ub, dtype=float).ravel()
            if lb.size != dim or ub.size != dim:
                # If problem uses a different internal dimension, rescale or broadcast
                # to match the requested dim as a simple fallback.
                lb = np.resize(lb, dim)
                ub = np.resize(ub, dim)

        n_obj = int(getattr(prob, "n_obj", 2))

        spec = ProblemSpec(
            name=name,
            dim=dim,
            n_obj=n_obj,
            lb=lb,
            ub=ub,
            ref_point=None
            if ref_point is None
            else np.asarray(ref_point, dtype=float).ravel(),
        )

        return _PymooProblemWrapper(prob, spec)
