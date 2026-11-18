"""ParEGO and Improved-Scalarized-EI building blocks for the second ablation.

The start of the path is ParEGO as implemented in DDMTOLab 1.0.10
(``ddmtolab.Algorithms.STMO.ParEGO``; Knowles, IEEE TEVC 10(1), 2006). Its pieces
are called unchanged: the weight set (``uniform_point``), the initial design
(``initialization``), the scalarisation (``ParEGO._scalarize``), the GP
(``gp_build``) and the LogEI search (``optimize_logei``).

The end is Improved-Scalarized-EI (ISEI, ``generated_algorithms/
MOBOImprovedScalarizedEI.py``) with the SMAC-tuned settings of the paper
(``ISEI_TUNED``). Its sampling, weights, Pareto front and genetic-algorithm
operators are its own methods, called on an ISEI instance. Two of its methods use
its own models directly, so they cannot be combined with another model as they
are: the GP fit inside ``_fit_model`` and the genetic algorithm inside
``_select_next_points`` (with its fitness ``_acquisition_function_for_ea``). They
are copied below with one change each: the training targets (fit) and the
fitness function (genetic algorithm) are passed in. With ISEI's own targets and
fitness passed in, they make the same random draws and compute the same values
as the originals, which the tests check.

Mixing the two needs three more pieces, agreed for this ablation:

* ISEI's per-objective models with ParEGO's augmented Tchebycheff (blocks 4 T,
  3 S): the scalarised value is a maximum of correlated Gaussians, whose mean and
  variance come from Clark's moment matching (``moead_ego_parts.
  augmented_tchebycheff_moments``, as MOEA/D-EGO's expected Tchebycheff
  improvement), here also in torch for the gradient search;
* BoTorch's gradient search (block 7 S) on scikit-learn GPs (block 5 T): the
  fitted GP's posterior is rebuilt in torch from its fitted kernel
  (``sklearn_gp_torch``), so ``optimize_logei`` runs unchanged on it;
* any model behind ``optimize_logei``: ``ScalarisedModel`` wraps a function
  returning the mean and variance of the scalarised value as a BoTorch model.
"""

import importlib.util
import math
import pathlib

import numpy as np
from scipy.stats import norm

from . import moead_ego_parts as ego

ROOT = pathlib.Path(__file__).resolve().parents[1]
_ISEI_PATH = ROOT / "generated_algorithms" / "MOBOImprovedScalarizedEI.py"
_isei_module = None

# Improved-Scalarized-EI as benchmarked in the paper: the SMAC incumbent in
# generated_algorithms/incumbents.yaml (used by paper_reproduction/rerun_isei_seeds.py).
ISEI_TUNED = dict(
    ea_crossover_rate=0.8069829779829,
    ea_generations=39,
    ea_mutation_rate=0.4022169454306,
    ea_pop_size=85,
    gp_restarts_optimizer=1,
    model_window_size=91,
    n_uniform_weights=48,
)

PAREGO_N_WEIGHTS = 100  # DDMTOLab ParEGO default n_weights
PAREGO_RHO = 0.05       # DDMTOLab ParEGO default rho
SIGMA_FLOOR = 1e-10     # ISEI's floor on predicted standard deviations


def isei_class():
    """``MOBOImprovedScalarizedEI`` loaded from ``generated_algorithms/``."""
    global _isei_module
    if _isei_module is None:
        spec = importlib.util.spec_from_file_location("_isei", _ISEI_PATH)
        _isei_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_isei_module)
    return _isei_module.MOBOImprovedScalarizedEI


# ===================================================================== ParEGO
def parego_weights(n_obj):
    """DDMTOLab ParEGO's weight set: ``uniform_point(100, M)`` (100 for M=2, 91 for M=3)."""
    from ddmtolab.Methods.Algo_Methods.uniform_point import uniform_point

    W, _ = uniform_point(PAREGO_N_WEIGHTS, n_obj)
    return W


def parego_scalarise(Y, w):
    """DDMTOLab ``ParEGO._scalarize``: normalise by the observed range, then the
    augmented Tchebycheff value. Returns shape ``(n,)``."""
    from ddmtolab.Algorithms.STMO.ParEGO import ParEGO

    return ParEGO(problem=None, rho=PAREGO_RHO)._scalarize(Y, w).ravel()


def parego_normalise(Y):
    """The normalisation inside ``ParEGO._scalarize`` (range floored at 1e-10)."""
    obj_min, obj_max = np.min(Y, axis=0), np.max(Y, axis=0)
    return (Y - obj_min) / np.maximum(obj_max - obj_min, 1e-10)


def botorch_gp(X_unit, y):
    """DDMTOLab ``gp_build`` on a single target (float32, fits ``-y``)."""
    import torch
    from ddmtolab.Methods.Algo_Methods.bo_utils import gp_build

    return gp_build(X_unit, np.asarray(y, dtype=float).reshape(-1, 1), data_type=torch.float)


def botorch_predict(gp, X_unit):
    """DDMTOLab ``gp_predict``: mean (sign of ``y``) and std, shape ``(n,)``."""
    import torch
    from ddmtolab.Methods.Algo_Methods.bo_utils import gp_predict

    mu, sd = gp_predict(gp, X_unit, data_type=torch.float)
    return mu.reshape(-1).astype(float), sd.reshape(-1).astype(float)


def optimize_logei(model, objs, dim):
    """DDMTOLab ``optimize_logei`` (5 restarts, 20 raw samples) over the unit cube;
    ``objs`` are the observed values (minimised) that set ``best_f``."""
    import torch
    from ddmtolab.Methods.Algo_Methods.bo_utils import optimize_logei as _opt

    return _opt(model, np.asarray(objs, dtype=float).reshape(-1, 1), dim, data_type=torch.float)


# ======================================================================= ISEI
def isei_normalise(Y_all):
    """ISEI's normalisation (``_fit_model``): observed min/max over all points, + 1e-6."""
    y_min, y_max = np.min(Y_all, axis=0), np.max(Y_all, axis=0)
    return (Y_all - y_min) / (y_max - y_min + 1e-6)


def isei_weighted_sum(Yn, w):
    """ISEI's scalarisation of normalised objectives: ``sum(w * y_norm)``."""
    return np.sum(w * Yn, axis=1)


def fit_isei_gp(X_train, y_train, rng, n_restarts):
    """ISEI ``_fit_model`` for one target (copy; the target is passed in).

    The same kernel, the same ``random_state`` draw from ``rng`` and the same
    fallback (a second draw) when the fit fails. Returns ``(gp, fell_back)``.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    dim = X_train.shape[1]
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * \
        Matern(length_scale=np.ones(dim), nu=2.5, length_scale_bounds=(1e-2, 1e2)) + \
        WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=n_restarts,
                                  normalize_y=False, random_state=rng.integers(0, 2**30))
    try:
        gp.fit(X_train, y_train)
    except Exception:
        gp_fallback = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=0,
                                               normalize_y=False, random_state=rng.integers(0, 2**30))
        gp_fallback.fit(X_train, y_train)
        return gp_fallback, True
    return gp, False


def isei_ei(mu, sigma, f_best):
    """ISEI's expected improvement of a scalarised Gaussian (minimised), floored at 0.

    Arrays in, arrays out; element by element the arithmetic of
    ``_acquisition_function_for_ea`` (sigma below 1e-10 gives 0).
    """
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    ei = np.zeros_like(mu)
    ok = sigma >= 1e-10
    Z = (f_best - mu[ok]) / sigma[ok]
    ei[ok] = (f_best - mu[ok]) * norm.cdf(Z) + sigma[ok] * norm.pdf(Z)
    return np.maximum(0.0, ei)


def isei_genetic_algorithm(isei, X_pareto, fitness):
    """ISEI ``_select_next_points`` from the initial population on (copy; the
    fitness is passed in).

    ``fitness(population) -> scores`` for a ``(pop, dim)`` array in ISEI's bounds;
    larger is better. Returns the best individual seen, shape ``(1, dim)``.
    """
    self = isei
    population = self._sample_points(self.ea_pop_size)
    population = np.clip(population, self.bounds[0], self.bounds[1])
    if X_pareto is not None and X_pareto.shape[0] > 0:
        num_pareto_to_add = min(X_pareto.shape[0], self.ea_pop_size // 5)
        if num_pareto_to_add > 0:
            pareto_indices = self.rng.choice(X_pareto.shape[0], num_pareto_to_add, replace=False)
            population[:num_pareto_to_add] = X_pareto[pareto_indices]

    best_individual_so_far = None
    best_fitness_so_far = -np.inf
    for generation in range(self.ea_generations):
        fitness_scores = np.asarray(fitness(population), dtype=float)
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness_so_far:
            best_fitness_so_far = fitness_scores[current_best_idx]
            best_individual_so_far = population[current_best_idx]
        new_population = []
        if best_individual_so_far is not None:
            new_population.append(best_individual_so_far)
        while len(new_population) < self.ea_pop_size:
            p1 = self._tournament_selection(population, fitness_scores)
            p2 = self._tournament_selection(population, fitness_scores)
            child1, child2 = self._sbx_crossover(p1, p2)
            mutation_range = (self.bounds[1] - self.bounds[0]) * 0.1
            if self.rng.random() < self.ea_mutation_rate:
                child1 += self.rng.normal(0, mutation_range, self.dim)
            if self.rng.random() < self.ea_mutation_rate:
                child2 += self.rng.normal(0, mutation_range, self.dim)
            child1 = np.clip(child1, self.bounds[0], self.bounds[1])
            child2 = np.clip(child2, self.bounds[0], self.bounds[1])
            new_population.extend([child1, child2])
        population = np.array(new_population[:self.ea_pop_size])

    if best_individual_so_far is None:
        return self._sample_points(1)
    return np.array([best_individual_so_far])


# ================================================================ torch pieces
def _ndtr(x):
    import torch

    return torch.special.ndtr(x)


def _npdf(x):
    import torch

    return torch.exp(-x * x / 2.0) / math.sqrt(2 * math.pi)


def augmented_tchebycheff_moments_torch(u, var, w, rho):
    """``moead_ego_parts.augmented_tchebycheff_moments`` with z = 0, in torch.

    ``u``/``var``: ``(..., M)`` means and variances of the normalised objectives;
    returns the mean and variance of ``max_j w_j f_j + rho * sum_j w_j f_j``.
    """
    import torch

    w = torch.as_tensor(w, dtype=u.dtype)
    a_mu = w * u
    a_var = w * w * var  # (w sigma)^2, without a square root at zero variance
    m = a_mu.shape[-1]
    s_mu, s_var = a_mu.sum(-1), a_var.sum(-1)
    b_mu = a_mu + rho * s_mu[..., None]
    cov = rho * (a_var[..., :, None] + a_var[..., None, :]) + (rho ** 2) * s_var[..., None, None]
    cov = cov + torch.diag_embed(a_var)
    m_y, v_y, c_y = b_mu[..., 0], cov[..., 0, 0], cov[..., 0, :]
    for k in range(1, m):
        v_k = cov[..., k, k]
        tau = torch.sqrt(torch.clamp(v_y + v_k - 2.0 * c_y[..., k], min=1e-20))
        alpha = (m_y - b_mu[..., k]) / tau
        phi, Phi, Phi_n = _npdf(alpha), _ndtr(alpha), _ndtr(-alpha)
        new_m = m_y * Phi + b_mu[..., k] * Phi_n + tau * phi
        second = (m_y ** 2 + v_y) * Phi + (b_mu[..., k] ** 2 + v_k) * Phi_n + (m_y + b_mu[..., k]) * tau * phi
        c_y = c_y * Phi[..., None] + cov[..., k, :] * Phi_n[..., None]
        m_y = new_m
        v_y = torch.clamp(second - new_m ** 2, min=0.0)
    return m_y, v_y


def sklearn_gp_torch(gp):
    """The posterior of ISEI's fitted ``GaussianProcessRegressor`` as a torch function.

    Kernel ``C * Matern(nu=2.5, ARD) + White`` with ``normalize_y=False``, as ISEI
    fits it. Returns ``f(X) -> (mean, var)`` for a float64 tensor ``(..., d)``,
    differentiable in ``X``; the values are scikit-learn's ``predict`` to float64
    rounding (tests/test_ablation_isei.py).
    """
    import torch
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, Sum, WhiteKernel

    k = gp.kernel_
    if not (isinstance(k, Sum) and isinstance(k.k2, WhiteKernel) and isinstance(k.k1, Product)
            and isinstance(k.k1.k1, ConstantKernel) and isinstance(k.k1.k2, Matern) and k.k1.k2.nu == 2.5):
        raise ValueError(f"unexpected ISEI kernel {k}")
    amp, white = float(k.k1.k1.constant_value), float(k.k2.noise_level)
    ls = np.broadcast_to(np.asarray(k.k1.k2.length_scale, dtype=float), (gp.X_train_.shape[1],))
    t = dict(dtype=torch.float64)
    Xt = torch.tensor(gp.X_train_ / ls, **t)
    inv_ls = torch.tensor(1.0 / ls, **t)
    alpha = torch.tensor(gp.alpha_.ravel(), **t)
    Linv = torch.tensor(np.linalg.solve(gp.L_, np.eye(gp.L_.shape[0])), **t)
    y_mean = float(np.ravel(gp._y_train_mean)[0])
    y_std = float(np.ravel(gp._y_train_std)[0])

    def f(X):
        D = (X * inv_ls)[..., None, :] - Xt
        r = torch.sqrt(torch.clamp((D * D).sum(-1), min=1e-36))  # finite gradient at r = 0
        a = math.sqrt(5.0) * r
        kx = amp * (1.0 + a + a * a / 3.0) * torch.exp(-a)
        v = kx @ Linv.T
        var = torch.clamp(amp + white - (v * v).sum(-1), min=0.0)
        return y_std * (kx @ alpha) + y_mean, var * y_std ** 2

    return f


class ScalarisedModel:
    """Factory for a one-output BoTorch model from ``f(X) -> (mean, var)``.

    ``f`` gives the mean and variance of the scalarised value (minimised) for unit-
    cube points ``(..., d)``; the model's posterior is that Gaussian on the
    maximisation scale (``-mean``), as DDMTOLab's GPs model ``-y``. It stands in
    for DDMTOLab's GP in ``optimize_logei``.
    """

    def __new__(cls, f):
        from botorch.models.model import Model
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        class _Model(Model):
            _is_ensemble = False

            def __init__(self):
                super().__init__()
                self._f = f

            @property
            def num_outputs(self):
                return 1

            @property
            def batch_shape(self):
                import torch

                return torch.Size([])

            def posterior(self, X, output_indices=None, observation_noise=False,
                          posterior_transform=None, **kwargs):
                import torch

                mean, var = self._f(X.to(torch.float64))  # (..., q)
                mean, var = -mean.to(X.dtype), var.to(X.dtype)
                post = GPyTorchPosterior(MultivariateNormal(mean, torch.diag_embed(var)))
                return post if posterior_transform is None else posterior_transform(post)

        return _Model()


# ======================================================================= numpy
def clark_moments(mu, sd, w, rho=PAREGO_RHO):
    """Mean and std of the augmented Tchebycheff value (z = 0) of per-objective Gaussians."""
    m, v = ego.augmented_tchebycheff_moments(mu, sd, w, np.zeros(mu.shape[1]), rho)
    return m, np.sqrt(np.maximum(v, 0.0))
