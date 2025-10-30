# tests/test_ioh_objective_fn_real.py
import numpy as np
import pytest
from llamevol.evaluator.ioh_evaluator import IOHObjectiveFn
from llamevol.utils import BOOverBudgetException

# Use parameters consistent with get_IOHEvaluator()
PROBLEM_ID = 2  # from your problems list
INSTANCE_ID = 1
DIM = 5
DEFAULT_BUDGET = 100


def make_obj(budget=DEFAULT_BUDGET, show_pb=False, problem_id=PROBLEM_ID):
    return IOHObjectiveFn(
        problem_id=problem_id,
        instance_id=INSTANCE_ID,
        exec_id=0,
        dim=DIM,
        budget=budget,
        show_progress_bar=show_pb,
    )


def test_init_populates_metadata_real_ioh():
    obj = make_obj()
    assert obj.name.startswith(f"F{PROBLEM_ID}-")
    assert obj.bounds.shape == (2, DIM)
    assert obj.optimal_x.shape == (DIM,)
    assert isinstance(obj.optimal_value, float)


def test_call_records_histories_and_returns_scalar():
    obj = make_obj()
    y1 = obj(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    y2 = obj(np.array([0.0, 2.0, 0.0, 0.0, 0.0]))
    assert np.isscalar(y1) and np.isscalar(y2)
    assert obj.x_hist.shape == (2, DIM)
    assert obj.y_hist.shape == (2,)
    assert obj.obj_fn.state.evaluations == 2


def test_budget_guard_raises_after_exceeding():
    # keep DIM=5; shrink budget to exercise guard
    obj = make_obj(budget=3)
    for _ in range(4):  # allow a few calls first
        _ = obj(np.zeros(DIM))
    with pytest.raises(BOOverBudgetException):
        _ = obj(np.zeros(DIM))


def test_nan_input_raises():
    obj = make_obj()
    bad = np.zeros(DIM)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        obj(bad)


def test_maximize_negates_value():
    obj = make_obj()
    x = np.zeros(DIM)
    x[0] = 1.0

    # get raw value without negation
    raw = obj.stateless_call(x)  # maximize is False by default

    obj.maximize = True
    y = obj(x)  # now negated

    assert y == pytest.approx(-raw)


def test_progress_bar_updates_and_resets():
    obj = make_obj(show_pb=True)
    assert obj.progress_bar is not None and obj.progress_bar.n == 0
    _ = obj(np.zeros(DIM))
    assert obj.progress_bar.n == 1
    obj.reset()
    assert obj.obj_fn is None and obj.progress_bar is None


def test_stateless_call_uses_fresh_problem():
    obj = make_obj()
    start = obj.obj_fn.state.evaluations
    x = np.arange(DIM, dtype=float)
    y_stateless = obj.stateless_call(x)
    assert obj.obj_fn.state.evaluations == start
    y_stateful = obj(x)
    assert obj.obj_fn.state.evaluations == start + 1
    assert y_stateless == pytest.approx(y_stateful)
