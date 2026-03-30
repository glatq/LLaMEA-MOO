import numpy as np
import pytest
import time
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec


class DummyRandomSearchMO:
    """Simple MO random search that just calls func within the budget."""

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim], dtype=float)

    def __call__(self, func):
        rng = np.random.default_rng(0)
        lb, ub = self.bounds
        for _ in range(self.budget):
            x = rng.uniform(lb, ub)
            func(x)  # ignore return, evaluator tracks history internally
        # return value is ignored by evaluator
        return None


DUMMY_CODE = """
import numpy as np

class DummyRandomSearchMO:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)

    def __call__(self, func):
        rng = np.random.default_rng(0)
        lb, ub = self.bounds
        for _ in range(self.budget):
            x = rng.uniform(lb, ub)
            func(x)
        return None
"""


def test_multiobj_evaluator_with_real_pymoo():
    problem = [MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0])]
    evaluator = MultiObjEvaluator(
        budget=10, problems=problem, repeat=1, use_multiprocessing=False
    )

    res = evaluator.evaluate(
        code=DUMMY_CODE,
        cls_name="DummyRandomSearchMO",
        cls=None,
        cls_init_kwargs=None,
        cls_call_kwargs=None,
        injector=None,
    )

    # Basic sanity checks
    assert res.error is None
    assert res.result is not None
    assert len(res.result) == 1

    basic = res.result[0]
    # We stored the archive X in x_hist; it should not exceed the budget
    assert basic.x_hist is not None
    assert basic.x_hist.shape[0] <= evaluator.budget

    # Score should be finite and equal to -HV for this run
    assert np.isfinite(res.score)
    assert np.isfinite(basic.best_y)

    # Because repeat == 1 and score is defined as -mean(HV), we expect:
    # res.score == -basic.best_y
    assert pytest.approx(res.score, rel=1e-8) == basic.best_y


# 1. Define a "bad" optimizer that intentionally exceeds a short timeout
SLEEPY_CODE = """
import numpy as np
import time

class SleepyOptimizer:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None):
        self.budget = budget

    def __call__(self, func):
        # Simulate a process that takes longer than the timeout
        time.sleep(3) 
        for _ in range(self.budget):
            func(np.zeros(1))
        return None
"""


def test_multiobj_evaluator_timeout_enforcement():
    """Tests if the evaluator correctly stops an execution that exceeds the timeout."""
    timeout_limit = 1
    problem = [MOOProblemSpec(name="zdt1", dim=2, n_obj=2)]

    evaluator = MultiObjEvaluator(
        budget=5, problems=problem, repeat=1, timeout=timeout_limit
    )

    start_time = time.time()

    res = evaluator.evaluate(code=SLEEPY_CODE, cls_name="SleepyOptimizer")

    total_duration = time.time() - start_time

    # 1. Verify the process was actually killed near the 1s mark (and didn't run for 3s)
    assert (
        total_duration < 2.5
    ), f"Timeout failed! Code ran for {total_duration:.2f}s despite {timeout_limit}s limit."

    # 2. Verify that the error captured is indeed a Timeout
    assert (
        res.error_type == "TimeoutError"
    ), f"Expected TimeoutError, but got {res.error_type}"
    assert "Timeout" in str(res.error)

    # 3. Because it timed out, result should be empty (as confirmed by your failed test)
    assert len(res.result) == 0
    print(f"\nSuccess! Timeout enforced in {total_duration:.2f}s")
