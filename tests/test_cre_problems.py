"""Tests for the constrained RE (CRE) benchmark problems.

Objectives and constraints are transcribed from Tanabe & Ishibuchi (2020),
``reproblem_python_ver/reproblem.py``. These tests pin our implementation
against verbatim reference formulas:

  * objectives F must match exactly (rtol 1e-12);
  * the per-point violation ``max(0, G)`` must equal Tanabe's clipped violation.

Tanabe uses the convention ``g >= 0`` feasible and returns ``max(0, -g)``; we
expose the *signed* constraint in the ``G <= 0`` feasible convention. The
references below reproduce Tanabe's original (clipped) form, so matching them
catches any coefficient or sign error in our restructured implementation.
"""

import numpy as np
import pytest

from llamevol.evaluator import re_problems as RP
from llamevol.evaluator.re_problems import compute_ref_point, get_re_problem
from llamevol.evaluator.PymooMOProvider import PymooMOProvider
from llamevol.evaluator.moo_metrics import constraint_violation


# --------------------------------------------------------------------------- #
# Verbatim Tanabe references -> (f, g_violation) with g_violation = max(0, -g) #
# --------------------------------------------------------------------------- #
def _clip(g):
    g = np.asarray(g, dtype=float)
    return np.where(g < 0, -g, 0.0)


def _cre21(x):
    x1, x2, x3 = x
    f0 = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
    f1 = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
    g = [
        0.1 - f0,
        100000.0 - f1,
        100000.0 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2)),
    ]
    return np.array([f0, f1]), _clip(g)


def _cre22(x):
    x1, x2, x3, x4 = x
    P, L, E, Gm, tauMax, sigmaMax = 6000.0, 14.0, 30 * 1e6, 12 * 1e6, 13600.0, 30000.0
    f0 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
    f1 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)
    M = P * (L + (x2 / 2))
    R = np.sqrt(((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2))
    J = 2 * np.sqrt(2) * x1 * x2 * (((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2))
    tauDashDash = (M * R) / J
    tauDash = P / (np.sqrt(2) * x1 * x2)
    tau = np.sqrt(
        tauDash * tauDash
        + ((2 * tauDash * tauDashDash * x2) / (2 * R))
        + (tauDashDash * tauDashDash)
    )
    sigma = (6 * P * L) / (x4 * x3 * x3)
    tmp = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
    PC = tmp * (1 - (x3 / (2 * L)) * np.sqrt(E / (4 * Gm)))
    g = [tauMax - tau, sigmaMax - sigma, x4 - x1, PC - P]
    return np.array([f0, f1]), _clip(g)


def _cre23(x):
    x1, x2, x3, x4 = x
    f0 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
    f1 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (
        x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)
    )
    g = [
        (x2 - x1) - 20.0,
        0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1))),
        1.0
        - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1))
        / np.power((x2 * x2 - x1 * x1), 2),
        (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1)
        - 900.0,
    ]
    return np.array([f0, f1]), _clip(g)


def _cre24(x):
    x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], np.round(x[2]), x[3], x[4], x[5], x[6]
    f0 = (
        0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934)
        - 1.508 * x1 * (x6 * x6 + x7 * x7)
        + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
        + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    )
    f1 = np.sqrt(np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7) / (
        0.1 * x6 * x6 * x6
    )
    g = np.zeros(11)
    g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
    g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
    g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
    g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
    g[4] = -(x2 * x3) + 40.0
    g[5] = -(x1 / x2) + 12.0
    g[6] = -5.0 + (x1 / x2)
    g[7] = -1.9 + x4 - 1.5 * x6
    g[8] = -1.9 + x5 - 1.1 * x7
    g[9] = -f1 + 1300.0
    g[10] = (
        -np.sqrt(np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8)
        / (0.1 * x7 * x7 * x7)
        + 1100.0
    )
    return np.array([f0, f1]), _clip(g)


def _cre25(x):
    x1, x2, x3, x4 = np.round(x[0]), np.round(x[1]), np.round(x[2]), np.round(x[3])
    f0 = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
    f1 = max([x1, x2, x3, x4])
    return np.array([f0, f1]), _clip([0.5 - (f0 / 6.931)])


def _cre31(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    f0 = (
        1.98
        + 4.9 * x1
        + 6.67 * x2
        + 6.98 * x3
        + 4.01 * x4
        + 1.78 * x5
        + 0.00001 * x6
        + 2.73 * x7
    )
    f1 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
    Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
    Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
    f2 = 0.5 * (Vmbp + Vfd)
    g = np.zeros(10)
    g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
    g[1] = 0.32 - (
        0.261
        - 0.0159 * x1 * x2
        - 0.06486 * x1
        - 0.019 * x2 * x7
        + 0.0144 * x3 * x5
        + 0.0154464 * x6
    )
    g[2] = 0.32 - (
        0.214
        + 0.00817 * x5
        - 0.045195 * x1
        - 0.0135168 * x1
        + 0.03099 * x2 * x6
        - 0.018 * x2 * x7
        + 0.007176 * x3
        + 0.023232 * x3
        - 0.00364 * x5 * x6
        - 0.018 * x2 * x2
    )
    g[3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
    g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
    g[5] = 32 - (
        33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
    )
    g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
    g[7] = 4 - f1
    g[8] = 9.9 - Vmbp
    g[9] = 15.7 - Vfd
    return np.array([f0, f1, f2]), _clip(g)


def _cre32(x):
    x_L, x_B, x_D, x_T, x_Vk, x_CB = x
    displacement = 1.025 * x_L * x_B * x_T * x_CB
    V = 0.5144 * x_Vk
    Fn = V / np.power(9.8065 * x_L, 0.5)
    a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
    b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32
    power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
    outfit_weight = (
        np.power(x_L, 0.8)
        * np.power(x_B, 0.6)
        * np.power(x_D, 0.3)
        * np.power(x_CB, 0.1)
    )
    steel_weight = (
        0.034
        * np.power(x_L, 1.7)
        * np.power(x_B, 0.7)
        * np.power(x_D, 0.4)
        * np.power(x_CB, 0.5)
    )
    machinery_weight = 0.17 * np.power(power, 0.9)
    light_ship_weight = steel_weight + outfit_weight + machinery_weight
    ship_cost = 1.3 * (
        (2000.0 * np.power(steel_weight, 0.85))
        + (3500.0 * outfit_weight)
        + (2400.0 * np.power(power, 0.8))
    )
    capital_costs = 0.2 * ship_cost
    DWT = displacement - light_ship_weight
    running_costs = 40000.0 * np.power(DWT, 0.3)
    sea_days = (5000.0 / 24.0) * x_Vk
    daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
    fuel_cost = 1.05 * daily_consumption * sea_days * 100.0
    port_cost = 6.3 * np.power(DWT, 0.8)
    fuel_carried = daily_consumption * (sea_days + 5.0)
    miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)
    cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
    port_days = 2.0 * ((cargo_DWT / 8000.0) + 0.5)
    RTPA = 350.0 / (sea_days + port_days)
    voyage_costs = (fuel_cost + port_cost) * RTPA
    annual_costs = capital_costs + running_costs + voyage_costs
    annual_cargo = cargo_DWT * RTPA
    f = np.array([annual_costs / annual_cargo, light_ship_weight, -annual_cargo])
    c = np.zeros(9)
    c[0] = (x_L / x_B) - 6.0
    c[1] = -(x_L / x_D) + 15.0
    c[2] = -(x_L / x_T) + 19.0
    c[3] = 0.45 * np.power(DWT, 0.31) - x_T
    c[4] = 0.7 * x_D + 0.7 - x_T
    c[5] = 500000.0 - DWT
    c[6] = DWT - 3000.0
    c[7] = 0.32 - Fn
    KB = 0.53 * x_T
    BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
    KG = 1.0 + 0.52 * x_D
    c[8] = (KB + BMT - KG) - (0.07 * x_B)
    return f, _clip(c)


def _cre51(x):
    x0, x1, x2 = x
    f0 = 106780.37 * (x1 + x2) + 61704.67
    f1 = 3000 * x0
    f2 = 305700 * 2289 * x1 / np.power(0.06 * 2289, 0.65)
    f3 = 250 * 2289 * np.exp(-39.75 * x1 + 9.9 * x2 + 2.74)
    f4 = 25 * (1.39 / (x0 * x1) + 4940 * x2 - 80)
    g = np.zeros(7)
    g[0] = 1 - (0.00139 / (x0 * x1) + 4.94 * x2 - 0.08)
    g[1] = 1 - (0.000306 / (x0 * x1) + 1.082 * x2 - 0.0986)
    g[2] = 50000 - (12.307 / (x0 * x1) + 49408.24 * x2 + 4051.02)
    g[3] = 16000 - (2.098 / (x0 * x1) + 8046.33 * x2 - 696.71)
    g[4] = 10000 - (2.138 / (x0 * x1) + 7883.39 * x2 - 705.04)
    g[5] = 2000 - (0.417 * x0 * x1 + 1721.26 * x2 - 136.54)
    g[6] = 550 - (0.164 / (x0 * x1) + 631.13 * x2 - 54.48)
    return np.array([f0, f1, f2, f3, f4]), _clip(g)


# id -> (class, ref_fn, n_obj, dim, n_constr)
CASES = {
    "cre21": (RP.CRE21, _cre21, 2, 3, 3),
    "cre22": (RP.CRE22, _cre22, 2, 4, 4),
    "cre23": (RP.CRE23, _cre23, 2, 4, 4),
    "cre24": (RP.CRE24, _cre24, 2, 7, 11),
    "cre25": (RP.CRE25, _cre25, 2, 4, 1),
    "cre31": (RP.CRE31, _cre31, 3, 7, 10),
    "cre32": (RP.CRE32, _cre32, 3, 6, 9),
    "cre51": (RP.CRE51, _cre51, 5, 3, 7),
}


def _sample(p, n=200, seed=123):
    xl, xu = np.asarray(p.xl), np.asarray(p.xu)
    rng = np.random.default_rng(seed)
    return np.vstack([xl, xu, (xl + xu) / 2, xl + (xu - xl) * rng.random((n, len(xl)))])


@pytest.mark.parametrize("pid", list(CASES))
def test_structure(pid):
    cls, _ref, n_obj, dim, n_c = CASES[pid]
    p = cls()
    assert p.n_obj == n_obj
    assert p.n_ieq_constr == n_c
    assert len(p.xl) == dim and len(p.xu) == dim
    assert np.all(p.xl <= p.xu)


@pytest.mark.parametrize("pid", list(CASES))
def test_matches_tanabe_reference(pid):
    cls, ref, n_obj, dim, n_c = CASES[pid]
    p = cls()
    pts = _sample(p)
    F, G = p.evaluate(pts, return_values_of=["F", "G"])
    assert F.shape == (len(pts), n_obj)
    assert G.shape == (len(pts), n_c)

    f_ref = np.array([ref(x)[0] for x in pts])
    gviol_ref = np.array([ref(x)[1] for x in pts])
    # Objectives match verbatim.
    assert np.allclose(F, f_ref, rtol=1e-12, atol=1e-12)
    # Our signed G, clipped to its violation, equals Tanabe's violation. This
    # pins each constraint and the G <= 0 feasible convention.
    assert np.allclose(np.maximum(0.0, G), gviol_ref, rtol=1e-12, atol=1e-12)
    # cv equals total Tanabe violation.
    assert np.allclose(
        constraint_violation(G), gviol_ref.sum(axis=1), rtol=1e-12, atol=1e-12
    )


def test_registry_dispatch():
    assert get_re_problem("cre21") is not None
    assert get_re_problem("CRE32") is not None  # case-insensitive
    assert get_re_problem("re34") is not None  # unconstrained RE still resolves
    assert get_re_problem("zdt1") is None  # pymoo ids fall through


@pytest.mark.parametrize("pid", list(CASES))
def test_provider_serves_cre_as_constrained(pid):
    _cls, _ref, n_obj, dim, n_c = CASES[pid]
    w = PymooMOProvider().get(problem_id=pid, dim=dim, n_obj=n_obj)
    assert w.n_constr == n_c
    assert w.n_obj == n_obj
    assert np.asarray(w.bounds).shape == (2, dim)

    out = w((w.bounds[0] + w.bounds[1]) / 2.0)
    assert isinstance(out, tuple) and len(out) == 2
    F, G = out
    assert np.asarray(F).reshape(-1, n_obj).shape[1] == n_obj
    assert np.asarray(G).reshape(-1, n_c).shape[1] == n_c
    assert np.all(np.isfinite(np.asarray(F)))


def test_unconstrained_re_unchanged():
    # Backward compat: RE34 stays unconstrained and returns F only (no tuple).
    w = PymooMOProvider().get(problem_id="re34", dim=5, n_obj=3)
    assert w.n_constr == 0
    out = w((w.bounds[0] + w.bounds[1]) / 2.0)
    assert not isinstance(out, tuple)


def test_cv_zero_iff_all_constraints_satisfied():
    # On a loose problem (CRE51) there are both feasible and infeasible samples,
    # and cv == 0 exactly matches "all G <= 0".
    p = RP.CRE51()
    pts = _sample(p, n=300)
    _F, G = p.evaluate(pts, return_values_of=["F", "G"])
    cv = constraint_violation(G)
    all_satisfied = np.all(G <= 0.0, axis=1)
    assert np.array_equal(cv == 0.0, all_satisfied)
    assert all_satisfied.any() and (~all_satisfied).any()


def test_compute_ref_point_deterministic_and_shaped():
    p = RP.CRE23()
    r1 = compute_ref_point(p, n_samples=2000, seed=0)
    r2 = compute_ref_point(p, n_samples=2000, seed=0)
    assert r1.shape == (p.n_obj,)
    assert np.allclose(r1, r2)  # deterministic given seed
    assert np.all(np.isfinite(r1))
    # feasible-only variant is also finite and correctly shaped
    rf = compute_ref_point(p, n_samples=2000, seed=0, feasible_only=True)
    assert rf.shape == (p.n_obj,) and np.all(np.isfinite(rf))
