# tests/test_evaluator_with_fake_provider.py
import types
import numpy as np
import pytest

from llamevol.evaluator.ioh_evaluator import IOHEvaluator


# --- Minimal fake provider (sphere) ---
class FakeProvider:
    def get(self, problem_id: int, instance_id: int, dim: int):
        class _W:
            def __init__(self, d: int):
                self._dim = d
                self._evals = 0
                self.name = f"fake-sphere-{d}"
                self.bounds = np.array([np.full(d, -5.0), np.full(d, 5.0)])
                self.optimum_x = np.zeros(d)
                self.optimum_y = 0.0

            # match attributes the evaluator/tests read
            @property
            def evaluations(self) -> int:
                return self._evals

            @property
            def state(self):
                # simple live view for tests that access .state.evaluations
                return types.SimpleNamespace(evaluations=self._evals)

            def __call__(self, x: np.ndarray) -> float:
                x = np.asarray(x)
                self._evals += 1
                return float(np.sum(x * x))

        return _W(dim)


# --- Your generated optimizer (copied for test isolation) ---
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class DyExBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * self.dim
        self.kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            1.0, length_scale_bounds="fixed"
        )
        self.model = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=0
        )
        self.exploration_weight = 1.0

    def _sample_points(self, n):
        sampler = qmc.LatinHypercube(d=self.dim)
        return qmc.scale(sampler.random(n=n), self.bounds[0], self.bounds[1])

    def _fit_model(self, X, Y):
        self.model.fit(X, Y)
        return self.model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        progress = self.n_evals / self.budget
        self.exploration_weight = max(0.1, 1.0 - progress)
        best = np.min(self.y)
        Z = (mu - best) / sigma
        ei = (mu - best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return self.exploration_weight * ei

    def _select_next_points(self, batch_size):
        n_candidates = 100 * batch_size
        Xc = self._sample_points(n_candidates)
        acq = self._acquisition_function(Xc).flatten()
        return Xc[np.argsort(acq)[::-1][:batch_size]]

    def _evaluate_points(self, func, X):
        y = np.array([func(x) for x in X])
        self.n_evals += len(X)
        return y.reshape(-1, 1)

    def _update_eval_points(self, X, y):
        if self.X is None:
            self.X, self.y = X, y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.y, y))

    def __call__(self, func):
        X0 = self._sample_points(self.n_init)
        y0 = self._evaluate_points(func, X0)
        self._update_eval_points(X0, y0)
        while self.n_evals < self.budget:
            self._fit_model(self.X, self.y)
            b = min(10, self.budget - self.n_evals)
            Xn = self._select_next_points(b)
            yn = self._evaluate_points(func, Xn)
            self._update_eval_points(Xn, yn)
        i = np.argmin(self.y)
        return self.y[i].item(), self.X[i]


# --- The actual test ---
def _make_eval(dim=2, budget=12, problems=[1], instances=[[1]], repeat=1):
    ev = IOHEvaluator(
        dim=dim, budget=budget, problems=problems, instances=instances, repeat=repeat
    )
    ev.use_multi_process = False
    ev.max_eval_workers = 0  # sequential branch for determinism
    ev.provider = FakeProvider()  # key substitution: no IOH
    # keep runtime short
    ev.obj_fn_params = ev.obj_fn_params[:1]
    return ev


def test_evaluate_with_fake_provider_smoke():
    ev = _make_eval(dim=2, budget=12, problems=[1], instances=[[1]], repeat=1)
    res = ev.evaluate(code=None, cls_name="DyExBO", cls=DyExBO)
    assert res.error is None
    assert len(res.result) == 1

    r = res.result[0]
    assert r.error is None
    assert r.bounds.shape == (2, 2)
    assert r.x_hist is not None and r.x_hist.shape[1] == 2
    assert r.y_hist is not None and len(r.y_hist) > 0
    assert 0.0 <= r.log_y_aoc <= 1.0
    assert r.execution_time >= 0.0
