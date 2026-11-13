"""The ablation framework: both ends reproduce their originals; every mix runs.

* ``TTTTTTTTTT`` evaluates exactly the points MOEAD-EI Hybrid evaluates (same seed);
* ``SSSSSSSSSS`` evaluates exactly the points DDMTOLab's MOEA/D-EGO evaluates;
* the DDMTOLab helpers copied with the prediction passed in behave as the originals;
* the augmented-Tchebycheff extension of ETI (rule a) has the right moments;
* configurations mixing the two algorithms run to the full budget.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from ablation import END, START, HYBRID_TUNED, ModularMOEAD, neighbours, parse_config
from ablation import moead_ego_parts as ego
from ablation.modular import hybrid_class, seed_everything

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _problem(n_obj, lo=0.0, hi=1.0):
    """A ZDT1-like (2 objectives) or DTLZ1-like (3 objectives) test function."""

    def f(x):
        x = (np.asarray(x, dtype=float) - lo) / (hi - lo)
        g = 1 + 9 * np.mean(x[1:])
        if n_obj == 2:
            return np.array([x[0], g * (1 - np.sqrt(x[0] / g))])
        return np.array([x[0] * x[1] * g, x[0] * (1 - x[1]) * g, (1 - x[0]) * g])

    return f


def _recorded(f):
    X, Y = [], []

    def g(x):
        X.append(np.array(x, copy=True))
        y = f(x)
        Y.append(y)
        return y

    return g, X, Y


def _same(a, b):
    return len(a) == len(b) and all(np.array_equal(p, q) for p, q in zip(a, b))


# --------------------------------------------------------------------- configs
def test_config_strings():
    assert parse_config(START)["search"] == "S"
    assert parse_config("ttttttttts")["selection"] == "S"
    with pytest.raises(ValueError):
        parse_config("STX" + "S" * 7)
    with pytest.raises(ValueError):
        parse_config("S" * 9)
    first_round = neighbours(START, END)
    assert len(first_round) == 10
    assert first_round[0] == (0, "T" + "S" * 9)
    assert neighbours(END, END) == []


# ----------------------------------------------------- end: MOEAD-EI Hybrid
@pytest.mark.parametrize(
    "n_obj, dim, budget, lo, hi",
    [(2, 4, 45, 0.0, 1.0), (3, 5, 50, 0.0, 1.0), (2, 6, 45, -2.0, 3.0)],
)
def test_end_configuration_reproduces_moead_ei_hybrid(n_obj, dim, budget, lo, hi):
    bounds = np.array([[lo] * dim, [hi] * dim])
    g1, X1, Y1 = _recorded(_problem(n_obj, lo, hi))
    np.random.seed(7)
    hybrid_class()(budget=budget, dim=dim, bounds=bounds, **HYBRID_TUNED)(g1)

    g2, X2, Y2 = _recorded(_problem(n_obj, lo, hi))
    ModularMOEAD(budget, dim, bounds, config=END, seed=7)(g2)

    assert len(X1) == budget
    assert _same(X1, X2) and _same(Y1, Y2)


# ------------------------------------------------------ start: DDMTOLab MOEA/D-EGO
def _ddmtolab_wrapper():
    pytest.importorskip("ddmtolab")
    spec = importlib.util.spec_from_file_location(
        "ddmtolab_moead_ego", ROOT / "MO bench" / "ddmtolab_moead_ego.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DDMTOLabMOEADEGOWrapper


@pytest.mark.slow
def test_start_configuration_reproduces_ddmtolab():
    wrapper = _ddmtolab_wrapper()
    dim, budget = 3, 37  # 32 initial points + one batch of 5
    bounds = np.array([[0.0] * dim, [1.0] * dim])
    g1, X1, Y1 = _recorded(_problem(2))
    wrapper(budget, dim, bounds, seed=3)(g1)

    g2, X2, Y2 = _recorded(_problem(2))
    ModularMOEAD(budget, dim, bounds, config=START, seed=3)(g2)

    assert len(X1) == budget
    assert _same(X1, X2) and _same(Y1, Y2)


def _tiny_gp_models(n_points=12, dim=2, n_obj=2):
    mod = pytest.importorskip("ddmtolab.Algorithms.STMO.MOEA_D_EGO")
    rng = np.random.default_rng(0)
    X = rng.random((n_points, dim))
    Y = np.column_stack([X[:, 0], 1 - np.sqrt(X[:, 0]) + X[:, 1]])[:, :n_obj]
    seed_everything(0)
    models, centers = mod._build_gp_models_fcm(X, Y)
    return mod, X, Y, models, centers


def test_copied_ideal_point_estimation_matches_ddmtolab():
    mod, X, Y, models, centers = _tiny_gp_models()
    from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point

    W, _ = uniform_point(12, 2, method="NBI")
    np.random.seed(5)
    z_ref = mod._get_estimation_z(2, models, centers, W, Y.min(axis=0))
    np.random.seed(5)
    z_ours = ego.estimate_ideal_point(
        2, lambda Xu: mod._gp_evaluate_mean_fcm(Xu, models, centers), W, Y.min(axis=0)
    )
    assert np.array_equal(z_ref, z_ours)


def test_copied_moead_search_matches_ddmtolab():
    mod, X, Y, models, centers = _tiny_gp_models()
    from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point

    W, _ = uniform_point(12, 2, method="NBI")
    z = Y.min(axis=0) - 0.1
    gmin = mod._get_gmin(Y, W, z)
    np.random.seed(9)
    eti_ref, x_ref, _, _ = mod._moead_eti(2, models, centers, W, gmin, z)
    np.random.seed(9)
    eti_ours, x_ours = ego.moead_search(
        2,
        lambda Xu: mod._gp_evaluate_fcm(Xu, models, centers),
        W,
        gmin,
        lambda mu, sd, Wr, g: mod._get_eti(mu, sd, Wr, g, z),
    )
    assert np.array_equal(eti_ref, eti_ours) and np.array_equal(x_ref, x_ours)


def test_cluster_assignment_matches_ddmtolab_local_models():
    mod = pytest.importorskip("ddmtolab.Algorithms.STMO.MOEA_D_EGO")
    rng = np.random.default_rng(1)
    X = rng.random((100, 2))  # > 80 points: two local models
    clusters, centers = ego.cluster_assignment(X)
    assert len(clusters) == 1 + int(np.ceil((100 - 80) / 20))
    assert all(len(c) == 80 for c in clusters)
    Y = np.column_stack([X[:, 0], X[:, 1]])
    seed_everything(0)
    _, centers_ref = mod._build_gp_models_fcm(X, Y)
    assert np.array_equal(centers, centers_ref)


# ------------------------------------------------- rule a: augmented ETI
def test_augmented_eti_without_rho_is_ddmtolab_eti():
    mod = pytest.importorskip("ddmtolab.Algorithms.STMO.MOEA_D_EGO")
    rng = np.random.default_rng(2)
    for m in (2, 3):
        u, s = rng.random((50, m)), 0.05 + 0.3 * rng.random((50, m))
        W = rng.dirichlet(np.ones(m), size=50)
        z, g = np.zeros(m), rng.random(50)
        assert np.allclose(
            ego.eti_augmented(u, s, W, g, z, 0.0), mod._get_eti(u, s, W, g, z), atol=1e-12
        )


@pytest.mark.parametrize("m, tol", [(2, 0.01), (3, 0.05)])
def test_augmented_tchebycheff_moments_match_monte_carlo(m, tol):
    rng = np.random.default_rng(3)
    u, s = rng.random((1, m)), 0.1 + 0.3 * rng.random((1, m))
    w = rng.dirichlet(np.ones(m), size=1)
    z, rho = np.zeros(m), 0.2
    mean, var = ego.augmented_tchebycheff_moments(u, s, w, z, rho)

    a = w * (u - z) + w * s * rng.standard_normal((400_000, m))
    g = a.max(axis=1) + rho * a.sum(axis=1)
    # exact for two objectives; Clark's recursion is an approximation beyond
    assert abs(mean[0] - g.mean()) < tol * (abs(g.mean()) + g.std())
    assert abs(np.sqrt(var[0]) - g.std()) < tol * g.std() + 1e-3


# ------------------------------------------------------- mixed configurations
def _run(config, n_obj=2, dim=2, budget=None):
    if "S" in config:
        pytest.importorskip("ddmtolab")
    budget = budget or (11 * dim - 1 + 6)
    g, X, Y = _recorded(_problem(n_obj))
    alg = ModularMOEAD(budget, dim, np.array([[0.0] * dim, [1.0] * dim]), config=config, seed=1)
    F, Xr = alg(g)
    assert len(X) == budget and F.shape == (budget, n_obj)
    assert np.all(np.isfinite(F)) and np.all((Xr >= 0) & (Xr <= 1))
    return alg


@pytest.mark.parametrize("block, config", neighbours(END, START))
def test_every_single_change_from_the_end_runs(block, config):
    if config[6] == "S" or config[8] == "S":
        pytest.skip("MOEA/D-DE inner loops: covered by the slow test")
    _run(config, n_obj=3, dim=3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [c for _, c in neighbours(START, END)] + [c for _, c in neighbours(END, START)]
    + ["STSTSTSTST", "TSTSTSTSTS", "TTSSTTSSTT"],
)
def test_mixed_configurations_run_to_the_budget(config):
    pytest.importorskip("ddmtolab")
    _run(config)


# ------------------------------------------- fast predictions (fast_predict=True)
@pytest.mark.parametrize("n_obj, n_points, scale", [(2, 60, 1.0), (3, 130, 1.0), (2, 130, 0.01)])
def test_fast_predictor_matches_ddmtolab_gp_evaluate(n_obj, n_points, scale):
    from ablation.fast_predict import ddmtolab_fcm_predictor

    mod = pytest.importorskip("ddmtolab.Algorithms.STMO.MOEA_D_EGO")
    rng = np.random.default_rng(4)
    X = rng.random((n_points, 4))
    Y = scale * np.array([_problem(n_obj)(x) for x in X])
    seed_everything(0)
    models, centers = mod._build_gp_models_fcm(X, Y)  # 130 points: three local models
    fast = ddmtolab_fcm_predictor(models, centers)
    T = np.vstack([rng.random((400, 4)), X[:20]])  # also at training points (variance floor)
    u0, s0 = mod._gp_evaluate_fcm(T, models, centers)
    u1, s1 = fast(T)
    tol = 1e-3 * Y.std(axis=0)  # BoTorch predicts in float32
    assert np.all(np.abs(u0 - u1) <= tol) and np.all(np.abs(s0 - s1) <= tol)
    # one point at a time, as the inner loops call it (its own code path)
    for i in list(range(5)) + [len(T) - 1]:
        u_one, s_one = fast(T[i][None, :])
        assert np.allclose(u_one, u1[i], rtol=1e-12, atol=0) and np.allclose(s_one, s1[i], rtol=1e-12, atol=0)


@pytest.mark.parametrize("fallback_kernel", [False, True])
def test_fast_predictor_matches_moead_ei_hybrid_gp(fallback_kernel):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    from ablation.fast_predict import hybrid_predictor

    rng = np.random.default_rng(5)
    dim = 5
    bounds = np.array([[-2.0] * dim, [3.0] * dim])
    X = -2 + 5 * rng.random((90, dim))
    Y = np.array([_problem(3, -2.0, 3.0)(x) for x in X])
    fitter = hybrid_class()(budget=100, dim=dim, bounds=bounds, **HYBRID_TUNED)
    fitter.n_obj = 3
    fitter._fit_model(X, Y)
    if fallback_kernel:  # the Hybrid's fallback when its first fit fails
        Xs = fitter.scaler_X.transform(X)
        fitter.models = [
            GaussianProcessRegressor(kernel=Matern(length_scale=[1.0] * dim, nu=0.5) + WhiteKernel(1e-5),
                                     normalize_y=True, random_state=42 + i).fit(Xs, Y[:, i])
            for i in range(3)
        ]
    fast = hybrid_predictor(fitter.scaler_X, fitter.models)
    T = np.vstack([-2 + 5 * rng.random((300, dim)), X[:10]])
    Ts = fitter.scaler_X.transform(T)
    mu0 = np.array([m.predict(Ts, return_std=True)[0] for m in fitter.models]).T
    sd0 = np.array([m.predict(Ts, return_std=True)[1] for m in fitter.models]).T
    mu1, sd1 = fast(T)
    assert np.allclose(mu0, mu1, rtol=1e-8, atol=1e-10 * np.abs(Y).max())
    assert np.allclose(sd0, sd1, rtol=1e-6, atol=1e-8 * np.abs(Y).max())


def test_fast_predict_leaves_the_end_configuration_exact():
    bounds = np.array([[0.0] * 4, [1.0] * 4])
    g1, X1, Y1 = _recorded(_problem(2))
    np.random.seed(7)
    hybrid_class()(budget=45, dim=4, bounds=bounds, **HYBRID_TUNED)(g1)
    g2, X2, Y2 = _recorded(_problem(2))
    alg = ModularMOEAD(45, 4, bounds, config=END, seed=7, fast_predict=True)
    alg(g2)
    assert _same(X1, X2) and _same(Y1, Y2)
    assert alg.diagnostics["fast_predict"] is False  # no S inner loop in this configuration


@pytest.mark.slow
@pytest.mark.parametrize("config", [START, "TTTTTTSTTT", "TTTTTTTTST", "STTTTTSTST", "SSTSSSSSSS"])
def test_fast_predict_configurations_run_to_the_budget(config):
    pytest.importorskip("ddmtolab")
    dim = 3
    g, X, Y = _recorded(_problem(3))
    alg = ModularMOEAD(11 * dim - 1 + 10, dim, np.array([[0.0] * dim, [1.0] * dim]),
                       config=config, seed=1, fast_predict=True)
    F, Xr = alg(g)
    assert len(X) == 11 * dim - 1 + 10 and np.all(np.isfinite(F))
    assert alg.diagnostics["fast_predict"] is True


def test_copied_eti_is_ddmtolab_eti_bit_for_bit():
    from scipy.stats import norm

    mod = pytest.importorskip("ddmtolab.Algorithms.STMO.MOEA_D_EGO")
    x = np.concatenate([np.linspace(-40, 40, 2001), [0.0, -0.0, np.inf, -np.inf, 1e-300, -1e-300]])
    assert np.array_equal(ego.norm_cdf(x), norm.cdf(x)) and np.array_equal(ego.norm_pdf(x), norm.pdf(x))
    rng = np.random.default_rng(6)
    for m in (2, 3):
        u, s = rng.normal(size=(500, m)), np.abs(rng.normal(size=(500, m))) * rng.random((500, 1)) ** 4
        s[:20] = 0.0  # zero predicted std
        W = rng.dirichlet(np.ones(m), size=500)
        z, g = rng.normal(size=m) - 1, rng.normal(size=500)
        assert np.array_equal(ego.get_eti(u, s, W, g, z), mod._get_eti(u, s, W, g, z))
