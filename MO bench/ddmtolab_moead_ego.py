"""MOEA/D-EGO baseline: DDMTOLab's implementation, run unchanged.

``DDMTOLabMOEADEGOWrapper`` runs ``ddmtolab.Algorithms.STMO.MOEA_D_EGO`` (DDMTOLab
1.0.10, MIT License; Zhang, Liu, Tsang & Virginas, IEEE TEVC 14(3), 2010) through
the interface the ``MultiObjEvaluator`` expects (``__init__(budget, dim, bounds)``
and ``__call__(func) -> (F, X)``). Its settings are DDMTOLab's defaults: an
initial Latin hypercube design of 11d-1 points, batches of 5, 200 (2 objectives)
or 210 (3 objectives) subproblems, k-means local GP models (BoTorch) above 80
points and expected Tchebycheff improvement maximised by MOEA/D-DE.

DDMTOLab runs on problem objects (``MTOP``); ``_SingleTaskProblem`` below is the
minimal one-task adapter. DDMTOLab works in the unit cube and maps points to the
problem bounds itself (``decs * (ub - lb) + lb``). Its random draws come from
numpy's global generator and torch, which ``seed`` sets.
"""

import random

import numpy as np


class _SingleTaskProblem:
    """The parts of DDMTOLab's ``MTOP`` that MOEA_D_EGO uses, for one task."""

    n_tasks = 1

    def __init__(self, func, dim, lb, ub):
        self._func = func
        self.dims = [int(dim)]
        # Read by MOEA_D_EGO only after the initial design has been evaluated,
        # so the objective count is filled in on the first evaluation.
        self.n_objs = [None]
        self.bounds = [(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))]
        self.X, self.F = [], []

    def evaluate_task(self, index, decs_real, eval_objectives=True, eval_constraints=True):
        F = np.array([np.asarray(self._func(x), dtype=float).ravel() for x in decs_real])
        self.n_objs[0] = F.shape[1]
        self.X.append(np.asarray(decs_real, dtype=float))
        self.F.append(F)
        return F, np.zeros((F.shape[0], 0))

    def get_n_constraints(self, index):
        return 0


class DDMTOLabMOEADEGOWrapper:
    """DDMTOLab MOEA/D-EGO with its default settings."""

    def __init__(self, budget, dim, bounds=None, seed=None, batch_size=5, n_initial=None):
        self.budget = int(budget)
        self.dim = int(dim)
        if bounds is None:
            bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        b = np.asarray(bounds, dtype=float)
        self.lb, self.ub = b[0], b[1]
        self.seed = seed
        self.batch_size = int(batch_size)
        self.n_initial = n_initial

    def __call__(self, func):
        from ddmtolab.Algorithms.STMO.MOEA_D_EGO import MOEA_D_EGO

        if self.seed is not None:  # as ddmtolab's set_seed
            random.seed(self.seed)
            np.random.seed(self.seed)
            import torch

            torch.manual_seed(self.seed)
        problem = _SingleTaskProblem(func, self.dim, self.lb, self.ub)
        MOEA_D_EGO(
            problem,
            n_initial=self.n_initial,
            max_nfes=self.budget,
            batch_size=self.batch_size,
            save_data=False,
            disable_tqdm=True,
        ).optimize()
        return np.vstack(problem.F), np.vstack(problem.X)
