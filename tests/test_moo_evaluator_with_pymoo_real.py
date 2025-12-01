import numpy as np
import pytest

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
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim], dtype=float)

    def __call__(self, func):
        rng = np.random.default_rng(0)
        lb, ub = self.bounds
        for _ in range(self.budget):
            x = rng.uniform(lb, ub)
            func(x)
        return None
"""


def test_multiobj_evaluator_with_real_pymoo():
    problem = MOOProblemSpec(name="dtlz2", dim=5, n_obj=2)
    evaluator = MultiObjEvaluator(budget=20, problem=problem, repeat=1)

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
    assert pytest.approx(res.score, rel=1e-8) == -basic.best_y
