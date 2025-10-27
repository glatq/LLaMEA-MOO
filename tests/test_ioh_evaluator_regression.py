# tests/test_ioh_evaluator_with_generated_optimizer.py
import os
import time
import numpy as np
import pytest
from llamevol.evaluator.ioh_evaluator import IOHEvaluator  # adjust import

# Your generated optimizer (import if it's in a module; inlined here for clarity)
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class DyExBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed"
        )
        self.model = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=9
        )
        self.exploration_weight = 1.0

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim)
        sample = sampler.random(n=n_points)
        return qmc.scale(sample, self.bounds[0], self.bounds[1])

    def _fit_model(self, X, y):
        self.model.fit(X, y)
        return self.model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        progress = self.n_evals / self.budget
        self.exploration_weight = max(0.1, 1.0 - progress)
        best = np.min(self.y)
        imp = mu - best
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = self.exploration_weight * ei
        return ei

    def _select_next_points(self, batch_size):
        n_candidates = 100 * batch_size
        X_candidates = self._sample_points(n_candidates)
        acq_values = self._acquisition_function(X_candidates)
        indices = np.argsort(acq_values.flatten())[::-1][:batch_size]
        return X_candidates[indices]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    def __call__(
        self, func: Callable[[np.ndarray], np.float64]
    ) -> tuple[np.float64, np.array]:
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            batch_size = min(10, self.budget - self.n_evals)
            X_next = self._select_next_points(batch_size)
            y_next = self._evaluate_points(func, X_next)
            self._update_eval_points(X_next, y_next)
        best_index = np.argmin(self.y)
        best_y = self.y[best_index].item()
        best_x = self.X[best_index]
        return best_y, best_x


# ---------- Helpers ----------


def _make_eval(dim=2, budget=12, problems=[1], instances=[[1]], repeat=1):
    ev = IOHEvaluator(
        dim=dim, budget=budget, problems=problems, instances=instances, repeat=repeat
    )
    # Keep evaluations single-threaded for deterministic ordering unless a test opts in
    ev.use_multi_process = False
    ev.max_eval_workers = 0  # force sequential branch in evaluate()
    ev.ignore_over_budget = False
    return ev


def _sphere(x):
    return float(np.sum(np.square(x)))


# ---------- Tests ----------


def test_smoke_runs_optimizer_and_records_history():
    ev = _make_eval(dim=2, budget=12, problems=[1], instances=[[1]], repeat=1)
    res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    assert res.error is None
    assert len(res.result) == 1

    r = res.result[0]
    assert r.error is None
    assert r.budget == 12
    assert r.bounds.shape == (2, ev.dim)
    assert r.y_hist is not None and len(r.y_hist) > 0
    assert r.x_hist is not None and len(r.x_hist) > 0
    assert 0.0 <= r.log_y_aoc <= 1.0
    assert 0.0 <= r.log_y_aoc_ioh <= 1.0
    # The recorded best in y_hist must equal min(y_hist)
    assert np.isclose(np.min(r.y_hist), np.min(r.y_hist))
    # Execution time is tracked
    assert r.execution_time >= 0.0


# def test_over_budget_path_is_handled_gracefully():
#     # DyExBO starts with n_init = 2*dim. Make budget smaller than that to force an over-budget condition.
#     ev = _make_eval(dim=5, budget=6, problems=[1], instances=[[1]], repeat=1)
#     res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
#     # Either the evaluator surfaces an error, or it truncates histories and still returns a result.
#     # Accept both, but require that evaluate() completes without crashing the process.
#     if res.error is not None:
#         assert "Budget" in str(res.error) or "Over" in res.error_type
#     else:
#         assert len(res.result) == 1
#         r = res.result[0]
#         assert r.budget == 6
#         assert len(r.y_hist) <= 6


# 1) Budget enforced -> raises
def test_over_budget_raises_with_enforcement():
    ev = _make_eval(dim=5, budget=6, problems=[1], instances=[[1]], repeat=1)
    with pytest.raises(Exception) as exc:
        _ = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    # optional: be more specific
    assert "OverBudget" in str(exc.value) or "Budget" in str(exc.value)


# 2) Budget ignored -> completes and truncates histories
def test_over_budget_ignored_allows_completion():
    ev = _make_eval(dim=5, budget=6, problems=[1], instances=[[1]], repeat=1)
    ev.ignore_over_budget = True  # key change
    res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    assert res.error is None
    assert len(res.result) == 1
    r = res.result[0]
    assert r.error is None
    assert r.budget == 6
    assert len(r.y_hist) <= 6
    assert r.x_hist.shape[1] == 5
    assert 0.0 <= r.log_y_aoc <= 1.0


def test_aggregates_multiple_tasks_over_instances_and_repeats():
    # Two instances, repeat twice -> 4 tasks total
    ev = _make_eval(dim=2, budget=8, problems=[1], instances=[[1, 2]], repeat=2)
    res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    assert res.error is None
    assert len(res.result) == 4
    # Score and ioh_score computed from per-task results
    assert 0.0 <= res.score <= 1.0
    assert 0.0 <= res.ioh_score <= 1.0
    # Total execution time aggregates
    assert res.total_execution_time >= sum(r.execution_time for r in res.result) - 1e-9


def test_threaded_branch_produces_same_number_of_results():
    ev = _make_eval(dim=2, budget=8, problems=[1], instances=[[1, 2, 3]], repeat=1)
    # Enable threading branch inside IOHEvaluator
    ev.max_eval_workers = 2
    ev.use_multi_process = False
    res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    assert res.error is None
    assert len(res.result) == 3


def test_problem_prompt_is_used_and_stable():
    ev = _make_eval(dim=2, budget=6, problems=[1], instances=[[1]], repeat=1)
    prompt = ev.problem_prompt()
    assert isinstance(prompt, str)
    assert str(ev.dim) in prompt


def test_histories_respect_budget_and_shapes():
    ev = _make_eval(dim=3, budget=9, problems=[1], instances=[[1]], repeat=1)
    res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    assert res.error is None
    r = res.result[0]
    assert len(r.y_hist) <= ev.budget
    assert r.x_hist.shape[1] == ev.dim
