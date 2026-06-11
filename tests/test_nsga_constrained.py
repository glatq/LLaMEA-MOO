"""Constraint-readiness of the NSGA-II / NSGA-III baselines.

Both wrappers must (a) run on constrained problems by declaring n_ieq_constr to
pymoo and feeding G (so CDP applies), producing feasibility metadata, and
(b) stay unchanged on unconstrained problems. Run through the real evaluator on
BNH (constrained) and ZDT1 (unconstrained); no LLM involved.
"""

import pathlib

import numpy as np
import pytest

from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec

_ROOT = pathlib.Path(__file__).resolve().parents[1]
NSGA2_CODE = (_ROOT / "MO bench" / "NSGA-II.py").read_text()
NSGA3_CODE = (_ROOT / "MO bench" / "NSGA-III.py").read_text()
WRAPPERS = [(NSGA2_CODE, "NSGA2Wrapper"), (NSGA3_CODE, "NSGA3Wrapper")]


def _run(code, cls, spec, budget=16):
    ev = MultiObjEvaluator(
        budget=budget, problems=[spec], repeat=1, use_multiprocessing=False
    )
    return ev.evaluate(code=code, cls_name=cls)


@pytest.mark.parametrize("code,cls", WRAPPERS)
def test_nsga_runs_on_constrained_problem(code, cls):
    res = _run(code, cls, MOOProblemSpec("bnh", 2, 2, [140.0, 50.0]))
    assert res.error is None, res.error
    b = res.result[0]
    assert b.error is None, b.error  # would crash on the (F, G) tuple pre-fix
    # Constraint metadata populated (the data layer saw G).
    assert b.feasibility_rate is not None and 0.0 <= b.feasibility_rate <= 1.0
    assert b.cv_history is not None and len(b.cv_history) == b.raw_y_hist.shape[0]
    assert b.best_y is not None and b.best_y <= 0.0


@pytest.mark.parametrize("code,cls", WRAPPERS)
def test_nsga_unconstrained_unchanged(code, cls):
    res = _run(code, cls, MOOProblemSpec("zdt1", 6, 2, [1.1, 6.0]))
    assert res.error is None, res.error
    b = res.result[0]
    assert b.error is None, b.error
    # Unconstrained path: no constraint metadata attached.
    assert b.feasibility_rate is None
    assert b.cv_history is None
