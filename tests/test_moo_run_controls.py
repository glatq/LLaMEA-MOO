"""Run controls of the MOO evaluator and the BOFire baseline diagnostics.

* every run is capped at ``threads_per_run`` threads, torch included (torch keeps
  its own intra-op pool that threadpoolctl does not reach);
* a run that reaches ``run_timeout`` stops, keeps its partial history and is
  flagged instead of being dropped;
* an algorithm's optional ``diagnostics`` dict reaches the run result, and the
  BOFire wrappers count their random fallbacks there.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from llamevol.evaluator.multiobj_evaluator import MOOProblemSpec, MultiObjEvaluator

ZDT1 = [MOOProblemSpec(name="zdt1", dim=3, n_obj=2, ref_point=[1.1, 7.2])]

THREAD_PROBE_CODE = """
import numpy as np
import torch
from threadpoolctl import threadpool_info

class ThreadProbe:
    def __init__(self, budget, dim, bounds=None):
        self.budget, self.dim = int(budget), int(dim)
        self.diagnostics = {}

    def __call__(self, func):
        self.diagnostics["torch_threads"] = torch.get_num_threads()
        self.diagnostics["pool_threads"] = sorted(
            {p["num_threads"] for p in threadpool_info()}
        )
        rng = np.random.default_rng(0)
        for _ in range(self.budget):
            func(rng.random(self.dim))
        return None
"""

SLOW_SEARCH_CODE = """
import time
import numpy as np

class SlowSearch:
    def __init__(self, budget, dim, bounds=None):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        rng = np.random.default_rng(0)
        for _ in range(self.budget):
            func(rng.random(self.dim))
            time.sleep(0.02)
        return None
"""


def test_worker_runs_are_single_threaded():
    ev = MultiObjEvaluator(
        budget=6,
        problems=ZDT1,
        repeat=2,
        use_multiprocessing=True,
        max_workers=2,
        threads_per_run=1,
    )
    res = ev.evaluate(code=THREAD_PROBE_CODE, cls_name="ThreadProbe")

    assert res.error is None
    assert res.metadata["run_controls"]["max_workers"] == 2
    assert len(res.result) == 2
    for run in res.result:
        assert run.diagnostics["torch_threads"] == 1
        assert run.diagnostics["pool_threads"] in ([], [1])
        assert run.torch_threads == 1
        assert run.n_evals == 6
        assert run.timed_out is False


def test_sequential_mode_restores_torch_threads():
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    ev = MultiObjEvaluator(
        budget=4, problems=ZDT1, repeat=1, use_multiprocessing=False
    )
    res = ev.evaluate(code=THREAD_PROBE_CODE, cls_name="ThreadProbe")
    assert res.result[0].diagnostics["torch_threads"] == 1
    assert torch.get_num_threads() == before


def test_default_pool_is_one_worker_per_physical_core():
    from llamevol.evaluator.multiobj_evaluator import physical_core_count

    ev = MultiObjEvaluator(budget=4, problems=ZDT1)
    assert ev.max_workers == physical_core_count()
    assert ev.threads_per_run == 1
    assert ev.run_timeout is None


def test_run_timeout_keeps_partial_history():
    budget = 500
    ev = MultiObjEvaluator(
        budget=budget,
        problems=ZDT1,
        repeat=1,
        use_multiprocessing=False,
        calculate_hv_history=True,
        run_timeout=0.3,
    )
    res = ev.evaluate(code=SLOW_SEARCH_CODE, cls_name="SlowSearch")

    run = res.result[0]
    assert run.timed_out is True
    assert run.error_type == "RunTimeLimit"
    assert 0 < run.n_evals < budget
    # The partial run is still scored and keeps its HV curve.
    assert run.best_y is not None and np.isfinite(run.best_y)
    assert len(run.hv_hist) == run.n_evals == len(run.raw_y_hist)
    assert np.isfinite(res.score)


def test_runs_without_limit_are_not_flagged():
    ev = MultiObjEvaluator(
        budget=5, problems=ZDT1, repeat=1, use_multiprocessing=False, run_timeout=60
    )
    run = ev.evaluate(code=SLOW_SEARCH_CODE, cls_name="SlowSearch").result[0]
    assert run.timed_out is False
    assert run.error is None
    assert run.n_evals == 5


# ------------------------------------------------------------ BOFire wrappers
_MOD_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "MO bench" / "bofire_baseline.py"
)


def _load_bofire_module():
    spec = importlib.util.spec_from_file_location("bofire_baseline", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOUNDS = np.array([[0.0, 0.0], [1.0, 1.0]])


def _two_objectives(x):
    x = np.asarray(x, dtype=float).ravel()
    return np.array([x[0], (x[0] - 0.5) ** 2 + x[1]])


class _FailingAskStrategy:
    def tell(self, experiments):
        pass

    def ask(self, q):
        raise RuntimeError("acquisition optimisation failed")


def test_bofire_ask_failures_are_counted(monkeypatch):
    mod = _load_bofire_module()
    monkeypatch.setattr(mod.fstrat, "map", lambda data_model: _FailingAskStrategy())

    budget, n_init = 9, 4
    wrapper = mod.BoFireQparEGOWrapper(
        budget=budget, dim=2, bounds=BOUNDS, n_init=n_init, seed=0
    )
    F, X = wrapper(_two_objectives)

    assert F.shape == (budget, 2) and X.shape == (budget, 2)
    assert wrapper.diagnostics["ask_fallbacks"] == budget - n_init
    assert wrapper.diagnostics["tell_failures"] == 0
    assert "RuntimeError" in wrapper.diagnostics["first_error"]


def test_bofire_diagnostics_reach_the_run_result(monkeypatch):
    """The evaluator reads the wrapper's counters through the instance recorder."""
    code = _MOD_PATH.read_text()
    ev = MultiObjEvaluator(budget=6, problems=ZDT1, repeat=1, use_multiprocessing=False)

    import bofire.strategies.api as fstrat

    monkeypatch.setattr(fstrat, "map", lambda data_model: _FailingAskStrategy())
    run = ev.evaluate(
        code=code,
        cls_name="BoFireQparEGOWrapper",
        cls_init_kwargs={"n_init": 3},
    ).result[0]

    assert run.diagnostics["ask_fallbacks"] == 3
    assert run.n_evals == 6


@pytest.mark.slow
def test_qehvi_wrapper_runs_with_bofire_defaults():
    mod = _load_bofire_module()
    budget = 8
    wrapper = mod.BoFireQEHVIWrapper(budget=budget, dim=2, bounds=BOUNDS, n_init=4, seed=0)
    F, X = wrapper(_two_objectives)

    assert F.shape == (budget, 2) and X.shape == (budget, 2)
    assert np.all((X >= 0.0) & (X <= 1.0))
    assert wrapper.diagnostics["ask_fallbacks"] == 0
    assert wrapper.diagnostics["tell_failures"] == 0
