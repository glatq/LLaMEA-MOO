"""PR1 data-layer tests: constraint-aware provider, evaluator and metrics.

Covers two guarantees:
  1. Backward compatibility -- the unconstrained path (ZDT1, RE problems) is
     numerically and structurally unchanged.
  2. The constrained path (BNH and friends) auto-detects n_constr, returns the
     (F, G) tuple, and produces feasible-HV scoring + feasibility metadata.
"""

import numpy as np
import pytest

from llamevol.evaluator.PymooMOProvider import PymooMOProvider
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
from llamevol.evaluator import moo_metrics
from llamevol.evaluator.moo_metrics import (
    constraint_violation,
    normalized_hv,
    score_moo_run,
)


# --------------------------------------------------------------------------- #
# Provider: constraint auto-detection + __call__ contract                     #
# --------------------------------------------------------------------------- #
def test_provider_autodetects_n_constr():
    provider = PymooMOProvider()
    assert provider.get("zdt1", dim=30, n_obj=2).n_constr == 0
    assert provider.get("bnh", dim=2, n_obj=2).n_constr == 2
    assert provider.get("mw1", dim=15, n_obj=2).n_constr == 1
    assert provider.get("ctp1", dim=10, n_obj=2).n_constr == 2


def test_provider_re_problem_is_unconstrained():
    # RE problems bypass pymoo and expose no n_ieq_constr -> must default to 0.
    assert PymooMOProvider().get("re21", dim=4).n_constr == 0


def test_unconstrained_call_returns_F_only():
    wrapper = PymooMOProvider().get("zdt1", dim=10, n_obj=2)
    out = wrapper(np.full(10, 0.5))
    assert not isinstance(out, tuple)
    F = np.asarray(out)
    assert F.reshape(-1, 2).shape[1] == 2


def test_constrained_call_returns_F_and_G():
    wrapper = PymooMOProvider().get("bnh", dim=2, n_obj=2)
    out = wrapper((wrapper.bounds[0] + wrapper.bounds[1]) / 2.0)
    assert isinstance(out, tuple) and len(out) == 2
    F, G = out
    assert np.asarray(F).reshape(-1, 2).shape[1] == 2
    assert np.asarray(G).reshape(-1, wrapper.n_constr).shape[1] == 2


# --------------------------------------------------------------------------- #
# moo_metrics: pure scoring helpers                                           #
# --------------------------------------------------------------------------- #
def test_constraint_violation_sum_of_positive_parts():
    G = np.array([[-1.0, -2.0], [0.0, 3.0], [1.0, 4.0]])
    np.testing.assert_allclose(constraint_violation(G), [0.0, 3.0, 5.0])


def test_constraint_violation_empty_columns():
    # zdt1-style: G has zero columns -> no violation for any row.
    np.testing.assert_array_equal(constraint_violation(np.zeros((4, 0))), np.zeros(4))


def test_normalized_hv_matches_manual_reference():
    # Two non-dominated points; ref [2, 2]; volume = 4.
    # HV of {(0,1),(1,0)} w.r.t (2,2) = 2*2 - 1*1 = 3 -> normalized 3/4.
    Y = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert normalized_hv(Y, [2.0, 2.0]) == pytest.approx(0.75)


def test_score_unconstrained_equals_normalized_hv():
    Y = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    ref = [2.0, 2.0]
    s = score_moo_run(Y, ref, G=None)
    assert s.score == pytest.approx(normalized_hv(Y, ref))
    assert s.feasibility_rate == 1.0
    assert s.n_feasible == len(Y)


def test_score_constrained_uses_feasible_subset_only():
    Y = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.1]])
    # Third point infeasible; only the first two count.
    G = np.array([[-1.0], [-1.0], [5.0]])
    ref = [2.0, 2.0]
    s = score_moo_run(Y, ref, G=G)
    assert s.n_feasible == 2
    assert s.feasibility_rate == pytest.approx(2 / 3)
    assert s.score == pytest.approx(normalized_hv(Y[:2], ref))


def test_score_all_infeasible_soft_fallback_is_tiny_and_monotone():
    Y = np.array([[0.0, 0.0], [0.0, 0.0]])
    ref = [2.0, 2.0]
    eps = moo_metrics.DEFAULT_INFEASIBLE_EPSILON

    high_cv = score_moo_run(Y, ref, G=np.array([[10.0], [10.0]]))
    low_cv = score_moo_run(Y, ref, G=np.array([[10.0], [0.001]]))

    assert high_cv.feasibility_rate == 0.0 and low_cv.feasibility_rate == 0.0
    # Both are tiny (dominated by any feasible run) ...
    assert 0.0 <= high_cv.score <= eps and 0.0 <= low_cv.score <= eps
    # ... but the run that is "closer" to feasibility scores higher.
    assert low_cv.score > high_cv.score


# --------------------------------------------------------------------------- #
# Evaluator end-to-end                                                        #
# --------------------------------------------------------------------------- #
UNCONSTRAINED_CODE = """
import numpy as np

class DummyUnconstrainedMOBO:
    def __init__(self, budget, dim, bounds=None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = (np.asarray(bounds, dtype=float)
                       if bounds is not None
                       else np.array([[0.0]*dim, [1.0]*dim], dtype=float))

    def __call__(self, func):
        rng = np.random.default_rng(0)
        lb, ub = self.bounds[0], self.bounds[1]
        for _ in range(self.budget):
            f = func(rng.uniform(lb, ub))   # unconstrained: just F
            assert not isinstance(f, tuple)
        return None
"""

CONSTRAINED_CODE = """
import numpy as np

class DummyConstrainedMOBO:
    def __init__(self, budget, dim, bounds=None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = (np.asarray(bounds, dtype=float)
                       if bounds is not None
                       else np.array([[0.0]*dim, [1.0]*dim], dtype=float))

    def __call__(self, func):
        rng = np.random.default_rng(0)
        lb, ub = self.bounds[0], self.bounds[1]
        F, X, G = [], [], []
        for _ in range(self.budget):
            x = rng.uniform(lb, ub)
            f, g = func(x)                  # constrained contract: (F, G) tuple
            F.append(np.asarray(f).ravel())
            G.append(np.asarray(g).ravel())
            X.append(x)
        return np.vstack(F), np.vstack(X), np.vstack(G)
"""


def test_evaluator_unconstrained_backward_compat():
    spec = [MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0])]
    ev = MultiObjEvaluator(
        budget=20, problems=spec, repeat=1, use_multiprocessing=False
    )
    res = ev.evaluate(code=UNCONSTRAINED_CODE, cls_name="DummyUnconstrainedMOBO")

    assert res.error is None
    basic = res.result[0]
    # Unconstrained: no constraint metadata is attached.
    assert basic.feasibility_rate is None
    assert basic.cv_history is None
    assert basic.raw_g_hist is None
    # Score is exactly -normalized_hv over the captured archive (locks the
    # legacy formula).
    expected = -normalized_hv(basic.raw_y_hist, [1.1, 6.0])
    assert basic.best_y == pytest.approx(expected, rel=1e-12)
    assert res.score == pytest.approx(basic.best_y, rel=1e-12)


def test_evaluator_constrained_bnh_produces_feasibility_metadata():
    spec = [MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0])]
    ev = MultiObjEvaluator(
        budget=30, problems=spec, repeat=1, use_multiprocessing=False
    )
    res = ev.evaluate(code=CONSTRAINED_CODE, cls_name="DummyConstrainedMOBO")

    assert res.error is None, res.error
    basic = res.result[0]
    n = basic.raw_y_hist.shape[0]
    assert n > 0

    # Constraint metadata populated and consistent.
    assert basic.feasibility_rate is not None
    assert 0.0 <= basic.feasibility_rate <= 1.0
    assert basic.cv_history is not None and len(basic.cv_history) == n
    assert basic.raw_g_hist is not None and basic.raw_g_hist.shape == (n, 2)
    # cv == 0 exactly where every constraint is satisfied.
    expected_feasible = float(np.mean((np.maximum(0.0, basic.raw_g_hist).sum(1)) == 0))
    assert basic.feasibility_rate == pytest.approx(expected_feasible)

    # score = -feasible_HV is finite and non-positive.
    assert np.isfinite(basic.best_y)
    assert basic.best_y <= 0.0


def test_evaluator_constrained_hv_history_tracks_feasible_front():
    spec = [MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0])]
    ev = MultiObjEvaluator(
        budget=20,
        problems=spec,
        repeat=1,
        use_multiprocessing=False,
        calculate_hv_history=True,
    )
    res = ev.evaluate(code=CONSTRAINED_CODE, cls_name="DummyConstrainedMOBO")
    basic = res.result[0]
    assert basic.hv_hist is not None
    assert len(basic.hv_hist) == basic.raw_y_hist.shape[0]
    # Feasible HV is monotonically non-decreasing as more points accumulate.
    assert np.all(np.diff(basic.hv_hist) >= -1e-9)
