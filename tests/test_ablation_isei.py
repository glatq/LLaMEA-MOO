"""The ParEGO <-> Improved-Scalarized-EI ablation: both ends reproduce their
originals, the pieces that join them compute what they should, every mix runs.

* ``TTTTTTT`` evaluates exactly the points ISEI evaluates (same seed);
* ``SSSSSSS`` evaluates exactly the points DDMTOLab's ParEGO evaluates;
* the torch posterior of ISEI's GP and the torch Clark moments match the
  scikit-learn and numpy values; a wrapped DDMTOLab GP gives DDMTOLab's candidate;
* ISEI's genetic algorithm with fast predictions scores like the original;
* every configuration one block away from either end runs to the full budget.
"""

import pathlib

import numpy as np
import pytest
import yaml

from ablation import moead_ego_parts as ego
from ablation import parego_isei_parts as pp
from ablation.config import ISEI_BLOCKS, ISEI_END, ISEI_START, neighbours, parse_config
from ablation.parego_isei import ModularParEGOISEI

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMALL_GA = dict(ea_pop_size=12, ea_generations=4, model_window_size=15)


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
    assert len(ISEI_BLOCKS) == 7 and ISEI_START == "SSSSSSS" and ISEI_END == "TTTTTTT"
    assert parse_config("tsssssS", ISEI_BLOCKS)["init"] == "T"
    with pytest.raises(ValueError):
        parse_config("S" * 10, ISEI_BLOCKS)
    assert [c for _, c in neighbours(ISEI_START, ISEI_END)][-1] == "SSSSSST"
    with pytest.raises(ValueError):
        neighbours(ISEI_START, "T" * 10)


def test_isei_settings_are_the_papers_incumbent():
    inc = yaml.safe_load(open(ROOT / "generated_algorithms" / "incumbents.yaml"))
    assert pp.ISEI_TUNED == inc["MOBOImprovedScalarizedEI"]["init_kwargs"]


# ---------------------------------------------------------------- end: ISEI
@pytest.mark.parametrize(
    "n_obj, dim, budget, lo, hi, kwargs",
    [(2, 3, 18, 0.0, 1.0, SMALL_GA), (3, 4, 20, 0.0, 1.0, SMALL_GA), (2, 4, 30, -2.0, 3.0, SMALL_GA),
     (2, 2, 9, 0.0, 1.0, {})],  # the tuned settings themselves, a few iterations
)
def test_end_configuration_reproduces_isei(n_obj, dim, budget, lo, hi, kwargs):
    bounds = np.array([[lo] * dim, [hi] * dim])
    g1, X1, Y1 = _recorded(_problem(n_obj, lo, hi))
    pp.isei_class()(budget=budget, dim=dim, bounds=bounds, random_seed=7, **{**pp.ISEI_TUNED, **kwargs})(g1)

    g2, X2, Y2 = _recorded(_problem(n_obj, lo, hi))
    ModularParEGOISEI(budget, dim, bounds, config=ISEI_END, seed=7, **kwargs)(g2)

    assert len(X1) == budget
    assert _same(X1, X2) and _same(Y1, Y2)


# -------------------------------------------------------------- start: ParEGO
def _parego_wrapper():
    pytest.importorskip("ddmtolab")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ddm_parego", ROOT / "MO bench" / "ddmtolab_parego.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DDMTOLabParEGOWrapper


@pytest.mark.parametrize("n_obj, dim, lo, hi", [(2, 2, 0.0, 1.0), (3, 3, -1.0, 2.0)])
def test_start_configuration_reproduces_ddmtolab_parego(n_obj, dim, lo, hi):
    wrapper = _parego_wrapper()
    budget = 11 * dim - 1 + 4
    bounds = np.array([[lo] * dim, [hi] * dim])
    g1, X1, Y1 = _recorded(_problem(n_obj, lo, hi))
    wrapper(budget, dim, bounds, seed=3, n_obj=n_obj)(g1)

    g2, X2, Y2 = _recorded(_problem(n_obj, lo, hi))
    ModularParEGOISEI(budget, dim, bounds, config=ISEI_START, seed=3)(g2)

    assert len(X1) == budget
    assert _same(X1, X2) and _same(Y1, Y2)


def test_parego_wrapper_needs_the_objective_count():
    with pytest.raises(ValueError, match="n_obj"):
        _parego_wrapper()(10, 2)


# ------------------------------------------------------------ joining pieces
def _isei_gp(n=25, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 2, (n, dim))
    y = np.sin(X).sum(1) / 3 + 0.1 * X[:, 0] ** 2
    gp, _ = pp.fit_isei_gp(X, y, np.random.default_rng(1), n_restarts=1)
    return gp, X, rng


def test_torch_posterior_of_isei_gp_matches_sklearn():
    torch = pytest.importorskip("torch")
    gp, X, rng = _isei_gp()
    Xq = np.vstack([rng.uniform(-1, 2, (40, 3)), X[:3]])  # include training points
    mu, sd = gp.predict(Xq, return_std=True)
    f = pp.sklearn_gp_torch(gp)
    m, v = f(torch.tensor(Xq, dtype=torch.float64))
    assert np.allclose(m.numpy(), mu, rtol=1e-9, atol=1e-12)
    assert np.allclose(np.sqrt(v.numpy()), sd, rtol=1e-6, atol=1e-9)
    # finite gradients, also at a training point
    Xt = torch.tensor(X[:2], dtype=torch.float64, requires_grad=True)
    (f(Xt)[0].sum() + f(Xt)[1].sum()).backward()
    assert torch.isfinite(Xt.grad).all()


@pytest.mark.parametrize("m", [2, 3])
def test_torch_clark_moments_match_numpy(m):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(m)
    mu, sd = rng.uniform(0, 1, (50, m)), rng.uniform(0.01, 0.5, (50, m))
    w = rng.dirichlet(np.ones(m))
    m_np, v_np = ego.augmented_tchebycheff_moments(mu, sd, w, np.zeros(m), pp.PAREGO_RHO)
    m_t, v_t = pp.augmented_tchebycheff_moments_torch(
        torch.tensor(mu), torch.tensor(sd**2), w, pp.PAREGO_RHO)
    assert np.allclose(m_t.numpy(), m_np, rtol=1e-12) and np.allclose(v_t.numpy(), v_np, rtol=1e-10)


def test_wrapped_ddmtolab_gp_gives_ddmtolab_candidate():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ddmtolab")
    from ablation.modular import seed_everything

    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (20, 3))
    s = np.cos(3 * X).sum(1)
    gp = pp.botorch_gp(X, s)

    def f(Xq):
        post = gp.posterior(Xq.to(torch.float32))
        return -post.mean[..., 0].to(torch.float64), post.variance[..., 0].to(torch.float64)

    seed_everything(5)
    direct = pp.optimize_logei(gp, s, 3)
    seed_everything(5)
    wrapped = pp.optimize_logei(pp.ScalarisedModel(f), s, 3)
    assert np.allclose(direct, wrapped, atol=1e-5)


def test_fast_genetic_algorithm_scores_like_the_original():
    """One generation's fitness, batch vs one individual at a time."""
    m = ModularParEGOISEI(30, 3, config=ISEI_END, seed=2, **SMALL_GA)
    g, _, _ = _recorded(_problem(2))
    m(g)  # fitted state from a short run
    Yn = m._normalised()
    models = [pp.fit_isei_gp(m.X_real, Yn[:, j], np.random.default_rng(0), 0)[0] for j in range(2)]
    w = np.array([0.3, 0.7])
    P = np.random.default_rng(9).uniform(0, 1, (40, 3))
    m.fast_predict = False
    exact = m._fitness(models, w, 0.2)(P)
    m.fast_predict = True
    fast = m._fitness(models, w, 0.2)(P)
    assert np.allclose(fast, exact, rtol=1e-9, atol=1e-14)


# ------------------------------------------------------------ every mix runs
def _run(config, n_obj=2, dim=2, fast=True):
    if "S" in config:
        pytest.importorskip("ddmtolab")
    budget = (11 * dim - 1 if config[0] == "S" else 5) + 4
    g, X, _ = _recorded(_problem(n_obj))
    model = ModularParEGOISEI(budget, dim, config=config, seed=1, fast_predict=fast, **SMALL_GA)
    F, Xout = model(g)
    assert len(X) == budget and F.shape == (budget, n_obj) and np.all(np.isfinite(F))


@pytest.mark.parametrize("block, config", neighbours(ISEI_END, ISEI_START))
def test_every_single_change_from_the_end_runs(block, config):
    pytest.importorskip("ddmtolab")
    _run(config)


@pytest.mark.parametrize("block, config", neighbours(ISEI_START, ISEI_END))
def test_every_single_change_from_the_start_runs(block, config):
    pytest.importorskip("ddmtolab")
    _run(config, n_obj=3)


@pytest.mark.parametrize("config", ["STSTSTS", "TSTSTST", "SSTTSST", "TTSSTTS"])
def test_mixed_configurations_run_exact_and_fast(config):
    pytest.importorskip("ddmtolab")
    _run(config, n_obj=3, fast=False)
    _run(config, n_obj=2, fast=True)
