"""ParEGO baseline: DDMTOLab's implementation, run unchanged.

``DDMTOLabParEGOWrapper`` runs ``ddmtolab.Algorithms.STMO.ParEGO`` (DDMTOLab 1.0.10,
MIT License; Knowles, IEEE TEVC 10(1), 2006) through the interface the
``MultiObjEvaluator`` expects (``__init__(budget, dim, bounds)`` and
``__call__(func) -> (F, X)``). Its settings are DDMTOLab's defaults: an initial
Latin hypercube design of 11d-1 points, then one point per iteration: a weight
drawn at random from 100 (2 objectives) or 91 (3 objectives) uniform vectors, the
augmented Tchebycheff value (rho 0.05) of the objectives normalised by their
observed range, a BoTorch GP on it and LogEI maximised by BoTorch's optimiser.

DDMTOLab's ParEGO builds its weight vectors before it evaluates anything, so it
needs the number of objectives up front: pass ``n_obj``. ``_SingleTaskProblem`` is
the same minimal one-task adapter as in ``ddmtolab_moead_ego.py``. DDMTOLab works
in the unit cube and maps points to the problem bounds itself. Its random draws
come from numpy's global generator and torch, which ``seed`` sets.
"""

import random

import numpy as np


class _SingleTaskProblem:
    """The parts of DDMTOLab's ``MTOP`` that ParEGO uses, for one task."""

    n_tasks = 1

    def __init__(self, func, dim, lb, ub, n_obj):
        self._func = func
        self.dims = [int(dim)]
        self.n_objs = [int(n_obj)]
        self.bounds = [(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))]
        self.X, self.F = [], []

    def evaluate_task(self, index, decs_real, eval_objectives=True, eval_constraints=True):
        F = np.array([np.asarray(self._func(x), dtype=float).ravel() for x in decs_real])
        self.X.append(np.asarray(decs_real, dtype=float))
        self.F.append(F)
        return F, np.zeros((F.shape[0], 0))

    def get_n_constraints(self, index):
        return 0


class DDMTOLabParEGOWrapper:
    """DDMTOLab ParEGO with its default settings."""

    def __init__(self, budget, dim, bounds=None, seed=None, n_obj=None):
        if n_obj is None:
            raise ValueError("DDMTOLab's ParEGO needs the number of objectives (n_obj) up front")
        self.budget = int(budget)
        self.dim = int(dim)
        if bounds is None:
            bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        b = np.asarray(bounds, dtype=float)
        self.lb, self.ub = b[0], b[1]
        self.seed = seed
        self.n_obj = int(n_obj)

    def __call__(self, func):
        from ddmtolab.Algorithms.STMO.ParEGO import ParEGO

        if self.seed is not None:  # as ddmtolab's set_seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            import torch

            torch.manual_seed(self.seed)
        problem = _SingleTaskProblem(func, self.dim, self.lb, self.ub, self.n_obj)
        ParEGO(problem, max_nfes=self.budget, save_data=False, disable_tqdm=True).optimize()
        return np.vstack(problem.F), np.vstack(problem.X)
