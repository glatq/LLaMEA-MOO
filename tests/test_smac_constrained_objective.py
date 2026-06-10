"""PR7: constraint-aware SMAC objective.

Covers the pure cost function ``_moo_objective`` (unconstrained parity +
constrained feasible-HV with infeasibility penalty) and that
``validate_with_random_config`` tolerates the constrained (F, G) contract.
"""

import numpy as np
import pytest

from llamevol.evaluator.moo_metrics import normalized_hv, score_moo_run
from llamevol.evaluator.multiobj_evaluator import MOOProblemSpec, MultiObjEvaluator
from llamevol.evaluator.smac_hpo_wrapper import (
    SMAC_AVAILABLE,
    _moo_objective,
    validate_with_random_config,
)
from llamevol.configspace_ext.configspace_utils import extract_configspace_from_response

REF = [2.0, 2.0]


# --------------------------------------------------------------------------- #
# _moo_objective                                                              #
# --------------------------------------------------------------------------- #
def test_objective_unconstrained_matches_legacy():
    Y = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    # Legacy objective was 1 - normalized_HV (HV > 0 here).
    assert _moo_objective(Y, REF) == pytest.approx(1.0 - normalized_hv(Y, REF))


def test_objective_constrained_feasible_subset():
    Y = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.1]])
    G = np.array([[-1.0], [-1.0], [5.0]])  # third point infeasible
    lam = 0.1
    res = score_moo_run(Y, REF, G=G)
    expected = 1.0 - res.score + lam * (1.0 - res.feasibility_rate)
    assert _moo_objective(Y, REF, G=G, infeasibility_penalty=lam) == pytest.approx(
        expected
    )


def test_feasible_run_beats_fully_infeasible_run():
    lam = 0.1
    Y = np.array([[0.2, 0.2], [0.3, 0.3]])
    feasible_cost = _moo_objective(
        Y, REF, G=np.array([[-1.0], [-1.0]]), infeasibility_penalty=lam
    )
    infeasible_cost = _moo_objective(
        Y, REF, G=np.array([[3.0], [4.0]]), infeasibility_penalty=lam
    )
    # Lower cost is better: a feasible run must outrank a fully-infeasible one,
    # and the infeasible run pays the full penalty (cost > 1).
    assert feasible_cost < infeasible_cost
    assert infeasible_cost > 1.0


def test_infeasibility_penalty_increases_cost():
    Y = np.array([[0.2, 0.2], [0.3, 0.3]])
    G = np.array([[3.0], [4.0]])  # fully infeasible -> infeasibility_rate == 1
    base = _moo_objective(Y, REF, G=G, infeasibility_penalty=0.0)
    penalized = _moo_objective(Y, REF, G=G, infeasibility_penalty=0.5)
    assert penalized == pytest.approx(base + 0.5)


# --------------------------------------------------------------------------- #
# lambda threads from the evaluator into SMACHPOConfig                         #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SMAC_AVAILABLE, reason="SMAC/ConfigSpace not installed")
def test_evaluator_threads_infeasibility_penalty():
    spec = [MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0])]
    ev = MultiObjEvaluator(
        budget=10, problems=spec, use_hpo=True, hpo_infeasibility_penalty=0.3
    )
    assert ev.hpo_config.infeasibility_penalty == pytest.approx(0.3)
    # Default when not supplied.
    ev_default = MultiObjEvaluator(budget=10, problems=spec, use_hpo=True)
    assert ev_default.hpo_config.infeasibility_penalty == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# validate_with_random_config on a constrained problem                        #
# --------------------------------------------------------------------------- #
CONSTRAINED_ALGO = """
import numpy as np

class MOBODummyConstrained:
    def __init__(self, budget, dim, bounds=None, num_samples=10):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = (np.asarray(bounds, dtype=float)
                       if bounds is not None
                       else np.array([[0.0]*dim, [1.0]*dim], dtype=float))
        self.num_samples = int(num_samples)

    def __call__(self, func):
        rng = np.random.default_rng(0)
        lb, ub = self.bounds[0], self.bounds[1]
        F, X, G = [], [], []
        # Deliberately over-call past the budget so the validator's budget guard
        # fires and MUST still return the (F, G) tuple (the PR7 fix).
        for _ in range(self.budget + 5):
            x = rng.uniform(lb, ub)
            f, g = func(x)              # constrained contract: (F, G) tuple
            F.append(np.asarray(f).ravel())
            G.append(np.asarray(g).ravel())
            X.append(x)
        return np.vstack(F), np.vstack(X), np.vstack(G)
"""

SPACE_RESPONSE = """
# Space
```python
{
    "num_samples": (5, 20)
}
```
"""


@pytest.mark.skipif(not SMAC_AVAILABLE, reason="SMAC/ConfigSpace not installed")
def test_validate_tolerates_constrained_tuple():
    configspace = extract_configspace_from_response(SPACE_RESPONSE)
    assert configspace is not None
    spec = MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0])

    ok, err = validate_with_random_config(
        code=CONSTRAINED_ALGO,
        cls_name="MOBODummyConstrained",
        configspace=configspace,
        problem_spec=spec,
        budget=12,  # smaller than the algorithm's loop so the budget guard fires
        n_retries=1,
    )
    assert ok, f"constrained validation should pass, got error: {err}"
