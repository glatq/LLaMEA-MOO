"""Tests for the RE (real-world engineering) benchmark problems.

The objective formulas are transcribed from Tanabe & Ishibuchi (2020); these
tests pin (a) the problem structure and (b) that our adapted, batched
implementation matches the original single-point formulas exactly, so an
accidental coefficient change is caught.
"""

import numpy as np
import pytest

from llamevol.evaluator.re_problems import RE21, RE34, RE37, get_re_problem
from llamevol.evaluator.PymooMOProvider import PymooMOProvider


# --- Verbatim Tanabe single-point reference implementations (ground truth) ---
def _re21_ref(x):
    x1, x2, x3, x4 = x
    F, E, L = 10.0, 2.0e5, 200.0
    return np.array(
        [
            L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4),
            ((F * L) / E)
            * (
                (2.0 / x1)
                + (2.0 * np.sqrt(2.0) / x2)
                - (2.0 * np.sqrt(2.0) / x3)
                + (2.0 / x4)
            ),
        ]
    )


def _re34_ref(x):
    x1, x2, x3, x4, x5 = x
    return np.array(
        [
            1640.2823
            + 2.3573285 * x1
            + 2.3220035 * x2
            + 4.5688768 * x3
            + 7.7213633 * x4
            + 4.4559504 * x5,
            6.5856
            + 1.15 * x1
            - 1.0427 * x2
            + 0.9738 * x3
            + 0.8364 * x4
            - 0.3695 * x1 * x4
            + 0.0861 * x1 * x5
            + 0.3628 * x2 * x4
            - 0.1106 * x1 * x1
            - 0.3437 * x3 * x3
            + 0.1764 * x4 * x4,
            -0.0551
            + 0.0181 * x1
            + 0.1024 * x2
            + 0.0421 * x3
            - 0.0073 * x1 * x2
            + 0.024 * x2 * x3
            - 0.0118 * x2 * x4
            - 0.0204 * x3 * x4
            - 0.008 * x3 * x5
            - 0.0241 * x2 * x2
            + 0.0109 * x4 * x4,
        ]
    )


def _re37_ref(x):
    a, h, o, t = x  # xAlpha, xHA, xOA, xOPTT
    return np.array(
        [
            0.692
            + 0.477 * a
            - 0.687 * h
            - 0.080 * o
            - 0.0650 * t
            - 0.167 * a * a
            - 0.0129 * h * a
            + 0.0796 * h * h
            - 0.0634 * o * a
            - 0.0257 * o * h
            + 0.0877 * o * o
            - 0.0521 * t * a
            + 0.00156 * t * h
            + 0.00198 * t * o
            + 0.0184 * t * t,
            0.153
            - 0.322 * a
            + 0.396 * h
            + 0.424 * o
            + 0.0226 * t
            + 0.175 * a * a
            + 0.0185 * h * a
            - 0.0701 * h * h
            - 0.251 * o * a
            + 0.179 * o * h
            + 0.0150 * o * o
            + 0.0134 * t * a
            + 0.0296 * t * h
            + 0.0752 * t * o
            + 0.0192 * t * t,
            0.370
            - 0.205 * a
            + 0.0307 * h
            + 0.108 * o
            + 1.019 * t
            - 0.135 * a * a
            + 0.0141 * h * a
            + 0.0998 * h * h
            + 0.208 * o * a
            - 0.0301 * o * h
            - 0.226 * o * o
            + 0.353 * t * a
            - 0.0497 * t * o
            - 0.423 * t * t
            + 0.202 * h * a * a
            - 0.281 * o * a * a
            - 0.342 * h * h * a
            - 0.245 * h * h * o
            + 0.281 * o * o * h
            - 0.184 * t * t * a
            - 0.281 * h * a * o,
        ]
    )


CASES = {
    "re21": (RE21, _re21_ref, 4, 2),
    "re34": (RE34, _re34_ref, 5, 3),
    "re37": (RE37, _re37_ref, 4, 3),
}


@pytest.mark.parametrize("pid", list(CASES))
def test_structure(pid):
    cls, _ref, dim, n_obj = CASES[pid]
    p = cls()
    assert p.n_obj == n_obj
    assert len(p.xl) == dim and len(p.xu) == dim
    assert np.all(p.xl <= p.xu)


@pytest.mark.parametrize("pid", list(CASES))
def test_matches_tanabe_reference(pid):
    cls, ref, dim, n_obj = CASES[pid]
    p = cls()
    rng = np.random.default_rng(0)
    pts = np.vstack(
        [p.xl, p.xu, (p.xl + p.xu) / 2, p.xl + (p.xu - p.xl) * rng.random((500, dim))]
    )
    ours = p.evaluate(pts)[0]
    refs = np.array([ref(x) for x in pts])
    assert ours.shape == (len(pts), n_obj)
    assert np.allclose(ours, refs, rtol=1e-12, atol=1e-12)


def test_registry_dispatch():
    assert get_re_problem("re21") is not None
    assert get_re_problem("RE37") is not None  # case-insensitive
    assert get_re_problem("zdt1") is None  # non-RE ids fall through to pymoo


@pytest.mark.parametrize(
    "pid,dim,n_obj,ref_point",
    [
        ("re21", 4, 2, [2851.9, 0.037]),
        ("re34", 5, 3, [1862.3, 12.17, 0.196]),
        ("re37", 4, 3, [0.887, 0.913, 1.012]),
    ],
)
def test_provider_serves_re(pid, dim, n_obj, ref_point):
    w = PymooMOProvider().get(problem_id=pid, dim=dim, ref_point=ref_point, n_obj=n_obj)
    assert w.n_obj == n_obj
    assert np.asarray(w.bounds).shape == (2, dim)
    mid = (w.bounds[0] + w.bounds[1]) / 2
    y = np.asarray(w(mid)).ravel()
    assert y.shape == (n_obj,) and np.all(np.isfinite(y))
