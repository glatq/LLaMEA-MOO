"""Tests for the BOFire constrained qLogNEHVI baseline.

``BoFireConstrainedQLogNEHVIWrapper`` lives in MO bench/bofire_baseline.py
(not an importable package), so it is loaded by file path. It depends only on
numpy/pandas/bofire -- not on llamevol -- and is exercised with a mock
evaluation function following the constrained data-layer contract
func(x) -> (F, G) (feasible when G <= 0).
"""

import importlib.util
import pathlib

import numpy as np
import pytest

_MOD_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "MO bench" / "bofire_baseline.py"
)
_spec = importlib.util.spec_from_file_location("bofire_baseline", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BoFireConstrainedQLogNEHVIWrapper = _mod.BoFireConstrainedQLogNEHVIWrapper

BOUNDS = np.array([[0.0, 0.0], [1.0, 1.0]])


def _constrained_func(x):
    """2 objectives (minimize), 1 constraint: feasible when x0 + x1 <= 1.2."""
    x = np.asarray(x, dtype=float).ravel()
    F = np.array([x[0], (x[0] - 0.5) ** 2 + x[1]])
    G = np.array([x[0] + x[1] - 1.2])  # G <= 0  <=>  feasible
    return F, G


def _unconstrained_func(x):
    x = np.asarray(x, dtype=float).ravel()
    return np.array([x[0], (x[0] - 0.5) ** 2 + x[1]])


def _counting(fn):
    calls = {"x": [], "g": []}

    def wrapped(x):
        out = fn(x)
        calls["x"].append(np.asarray(x, dtype=float).ravel())
        calls["g"].append(out[1] if isinstance(out, tuple) else None)
        return out

    return wrapped, calls


# ------------------------------------------------------------------ fast tests
def test_split_detects_constraints():
    f, g = BoFireConstrainedQLogNEHVIWrapper._split(
        (np.array([1.0, 2.0]), np.array([0.3]))
    )
    assert g is not None and g.shape == (1,)
    np.testing.assert_array_equal(f, [1.0, 2.0])

    f2, g2 = BoFireConstrainedQLogNEHVIWrapper._split(np.array([1.0, 2.0]))
    assert g2 is None
    np.testing.assert_array_equal(f2, [1.0, 2.0])


# ------------------------------------------------------------------ slow tests
@pytest.mark.slow
def test_constrained_bo_runs_and_uses_full_budget():
    func, calls = _counting(_constrained_func)
    algo = BoFireConstrainedQLogNEHVIWrapper(
        budget=6, dim=2, bounds=BOUNDS, n_init=4, seed=0
    )
    Y, X = algo(func)
    # Full budget consumed; GP loop actually ran (budget > n_init).
    assert len(calls["x"]) == 6
    assert Y.shape == (6, 2) and X.shape == (6, 2)
    # Candidates stay inside the box.
    assert np.all(X >= BOUNDS[0] - 1e-9) and np.all(X <= BOUNDS[1] + 1e-9)
    # The constrained contract was exercised (G captured at every eval).
    assert all(g is not None and g.shape == (1,) for g in calls["g"])
    # At least one evaluated point is feasible (G <= 0).
    g_all = np.vstack(calls["g"])
    assert np.any((g_all <= 0).all(axis=1))


@pytest.mark.slow
def test_unconstrained_fallback_runs():
    func, calls = _counting(_unconstrained_func)
    algo = BoFireConstrainedQLogNEHVIWrapper(
        budget=6, dim=2, bounds=BOUNDS, n_init=4, seed=0
    )
    Y, X = algo(func)
    assert all(g is None for g in calls["g"])  # constraint-free contract detected
    assert Y.shape == (6, 2) and X.shape == (6, 2)
