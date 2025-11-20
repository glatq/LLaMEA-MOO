# tests/test_aoc_paths.py
import numpy as np
import pytest
from llamevol.evaluator.ioh_evaluator import compute_log_aoc, IOHEvaluator
import llamevol.evaluator.ioh_evaluator as mod


# 1) Unit test: compute_log_aoc behavior on simple sequences
def test_compute_log_aoc_unit(monkeypatch):
    # strictly improving best-so-far should reduce area (=> higher score after 1 - normalized_aoc)
    y_hist = np.array([10.0, 9.0, 8.0, 7.0])
    score1 = compute_log_aoc(
        y_hist=y_hist, budget=4, optimum_value=0.0, lower=1e-8, upper=1e4
    )

    # flat sequence should accumulate more area (=> lower score)
    y_flat = np.array([10.0, 10.0, 10.0, 10.0])
    score2 = compute_log_aoc(
        y_hist=y_flat, budget=4, optimum_value=0.0, lower=1e-8, upper=1e4
    )

    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0
    assert score1 > score2


# 2a) Integration: evaluator uses fallback compute_log_aoc with FakeProvider
def test_fallback_aoc_used_with_fake_provider(monkeypatch):
    # Fake provider like in your previous test
    class FakeProvider:
        def get(self, problem_id, instance_id, dim):
            class W:
                def __init__(self, d):
                    self._evals = 0
                    self.name = f"fake-sphere-{d}"
                    self.bounds = np.array([np.full(d, -5.0), np.full(d, 5.0)])
                    self.optimum_x = np.zeros(d)
                    self.optimum_y = 0.0
                    self._d = d

                @property
                def evaluations(self):
                    return self._evals

                @property
                def state(self):
                    return type("S", (), {"evaluations": self._evals})()

                def __call__(self, x):
                    self._evals += 1
                    return float(np.sum(np.asarray(x) ** 2))

            return W(dim)

    # Monkeypatch the fallback to a sentinel value so we can assert it was used
    sentinel = 0.123456

    def fake_compute_log_aoc(y_hist, budget, optimum_value=0.0, lower=1e-8, upper=1e4):
        return sentinel

    monkeypatch.setattr(mod, "compute_log_aoc", fake_compute_log_aoc, raising=True)

    # Also make sure the IOH path isn't called (if present)
    if hasattr(mod, "correct_aoc"):

        def boom(*a, **k):
            raise AssertionError("correct_aoc should not be called for FakeProvider")

        monkeypatch.setattr(mod, "correct_aoc", boom, raising=True)

    # Minimal optimizer
    class TinyOpt:
        def __init__(self, budget, dim):
            self.budget, self.dim = budget, dim

        def __call__(self, func):
            for _ in range(self.budget):
                _ = func(np.zeros(self.dim))
            return 0.0, np.zeros(self.dim)

    ev = IOHEvaluator(dim=2, budget=6, problems=[1], instances=[[1]], repeat=1)
    ev.use_multi_process = False
    ev.max_eval_workers = 0
    ev.provider = FakeProvider()
    ev.obj_fn_params = ev.obj_fn_params[:1]

    res = ev.evaluate(code=None, cls_name="TinyOpt", cls=TinyOpt)
    assert res.error is None
    r = res.result[0]
    assert r.log_y_aoc_ioh == pytest.approx(sentinel)


# 2b) Integration: evaluator uses IOH correct_aoc for IOHProvider (and not the fallback)
@pytest.mark.skipif(
    pytest.importorskip("ioh", reason="IOH not installed") is None, reason="IOH missing"
)
def test_ioh_aoc_used_with_ioh_provider(monkeypatch):
    # Make correct_aoc return a sentinel, and the fallback raise if called
    sentinel = 0.789

    def fake_correct_aoc(*a, **k):
        return sentinel

    monkeypatch.setattr(mod, "correct_aoc", fake_correct_aoc, raising=True)

    def boom(*a, **k):
        raise AssertionError("compute_log_aoc should not be called for IOHProvider")

    if hasattr(mod, "compute_log_aoc"):
        monkeypatch.setattr(mod, "compute_log_aoc", boom, raising=True)

    class TinyOpt:
        def __init__(self, budget, dim):
            self.budget, self.dim = budget, dim

        def __call__(self, func):
            for _ in range(self.budget):
                _ = func(np.zeros(self.dim))
            return 0.0, np.zeros(self.dim)

    ev = IOHEvaluator(dim=2, budget=6, problems=[1], instances=[[1]], repeat=1)
    ev.use_multi_process = False
    ev.max_eval_workers = 0
    # provider left as default IOHProvider
    ev.obj_fn_params = ev.obj_fn_params[:1]

    res = ev.evaluate(code=None, cls_name="TinyOpt", cls=TinyOpt)
    assert res.error is None
    r = res.result[0]
    assert r.log_y_aoc_ioh == pytest.approx(sentinel)
