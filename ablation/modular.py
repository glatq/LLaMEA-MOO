"""MOEA/D-EGO and MOEAD-EI Hybrid as one algorithm built from swappable blocks.

``ModularMOEAD(config=...)`` runs the configuration named by a ten-letter S/T
string (see ``ablation.config``): each block comes either from MOEA/D-EGO as
implemented in DDMTOLab (``S``) or from the tuned MOEAD-EI Hybrid (``T``).

Each block calls the original code of the algorithm it comes from: the T blocks
call the methods of ``MOBO_MOEAD_EI_Hybrid_Fixed`` and the S blocks call
DDMTOLab's functions (``ablation.moead_ego_parts``). Both pure configurations
make the same random draws in the same order as their originals, so under the
same seed ``TTTTTTTTTT`` reproduces MOEAD-EI Hybrid and ``SSSSSSSSSS``
reproduces DDMTOLab's MOEA/D-EGO evaluation for evaluation (tests/test_ablation_
framework.py; ``fast_predict=False``, the default). ``fast_predict=True`` gives
MOEA/D-EGO's two inner loops closed-form predictions from the same fitted models
(``ablation.fast_predict``), equal to float32 rounding and about ten times faster.
Mixed configurations follow the rules agreed for the ablation (rules a-i), noted
where they apply.

Coordinates: the S blocks work in the unit cube (as DDMTOLab does) and the T
blocks in the problem's own bounds (as MOEAD-EI Hybrid does). The archive keeps
every point in both, each converted from the coordinates it was generated in.
"""

import importlib.util
import pathlib
import random

import numpy as np
from scipy.spatial.distance import cdist

from . import moead_ego_parts as ego
from .config import parse_config
from .fast_predict import ddmtolab_fcm_predictor, hybrid_predictor

_HYBRID_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "MO bench" / "MOBO_MOEAD_EI_Hybrid_fixed.py"
)
_hybrid_module = None

# MOEAD-EI Hybrid as benchmarked in the paper (SMAC-tuned, conf/benchmark_phase2.yaml).
HYBRID_TUNED = dict(
    batch_size=6,
    kernel_nu=0.5,
    n_init_ratio=0.275,
    n_model_points=200,
    n_offspring_surrogate=250,
    population_size=60,
    rho=0.01,
)


def hybrid_class():
    """``MOBO_MOEAD_EI_Hybrid_Fixed`` loaded from ``MO bench/`` (path has a space)."""
    global _hybrid_module
    if _hybrid_module is None:
        spec = importlib.util.spec_from_file_location("_moead_ei_hybrid", _HYBRID_PATH)
        _hybrid_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_hybrid_module)
    return _hybrid_module.MOBO_MOEAD_EI_Hybrid_Fixed


def seed_everything(seed):
    """Seed every generator the two algorithms draw from (as DDMTOLab ``set_seed``)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class _Predictor:
    """One fitted surrogate (blocks 2 + 3): ``predict(X, native) -> (mean, std)``.

    ``fast=True`` uses the closed-form equivalent of the same fitted model
    (``ablation.fast_predict``) when one was built, else the model itself.
    """

    def __init__(self, predict_unit=None, predict_real=None, to_unit=None, to_real=None,
                 fast_unit=None, fast_real=None):
        self._unit, self._real = predict_unit, predict_real
        self._fast_unit, self._fast_real = fast_unit, fast_real
        self._to_unit, self._to_real = to_unit, to_real

    def predict(self, X, native, fast=False):
        unit, real = self._unit, self._real
        if fast and (self._fast_unit is not None or self._fast_real is not None):
            unit, real = self._fast_unit, self._fast_real
        if native == "unit":
            if unit is not None:
                return unit(X)
            return real(self._to_real(X))
        if real is not None:
            return real(X)
        return unit(self._to_unit(X))


class ModularMOEAD:
    """MOEA/D-EGO <-> MOEAD-EI Hybrid, one configuration of the ablation."""

    def __init__(self, budget, dim, bounds=None, config="TTTTTTTTTT", seed=None,
                 fast_predict=False, **hybrid_kwargs):
        self.budget = int(budget)
        self.dim = int(dim)
        if bounds is None:
            bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)
        self.lb, self.ub = self.bounds[0], self.bounds[1]
        self.config = str(config).upper()
        self.b = parse_config(self.config)
        self.seed = seed
        # fast_predict: closed-form predictions inside MOEA/D-EGO's two inner loops
        # (blocks 7 and 9 as S); see ablation.fast_predict
        self.fast_predict = bool(fast_predict) and (self.b["ideal"] == "S" or self.b["search"] == "S")
        self.hybrid_kwargs = {**HYBRID_TUNED, **hybrid_kwargs}
        # The T blocks call the methods of this MOEAD-EI Hybrid instance.
        self.hyb = hybrid_class()(budget=self.budget, dim=self.dim, bounds=self.bounds, **self.hybrid_kwargs)
        self.rho = self.hyb.rho
        self.s_batch_size = 5  # DDMTOLab default
        # "ask_fallbacks" counts batches evaluated at random because the S
        # surrogate could not be built (DDMTOLab's fallback); the benchmark
        # writes it to run_status_log.csv like the BOFire fallbacks.
        self.diagnostics = {"config": self.config, "ask_fallbacks": 0, "first_error": None,
                            "fast_predict": self.fast_predict}

    # ------------------------------------------------------------ coordinates
    def _to_unit(self, X):
        return (X - self.lb) / (self.ub - self.lb)

    def _to_real(self, X):
        return X * (self.ub - self.lb) + self.lb

    # ---------------------------------------------------------------- archive
    def _evaluate(self, func, X, native):
        """Evaluate up to the remaining budget; store the points in both coordinates."""
        n = min(X.shape[0], self.budget - self.n_evals)
        if n <= 0:
            return np.empty((0, self.dim)), np.empty((0, self.n_obj or 0))
        X = X[:n]
        if native == "unit":  # DDMTOLab: decs * (ub - lb) + lb, evaluated as is
            X_unit, X_real = X, self._to_real(X)
            to_eval = X_real
        else:  # MOEAD-EI Hybrid: evaluates the clipped point, stores the point
            X_real, X_unit = X, self._to_unit(X)
            to_eval = np.clip(X_real, self.lb, self.ub)
        Y = []
        for x in to_eval:
            y = np.asarray(func(x), dtype=float).ravel()
            if self.n_obj is None:
                self.n_obj = y.shape[0]
                self.hyb.n_obj = self.n_obj
            Y.append(y)
        Y = np.array(Y)
        self.n_evals += n
        if self.X_unit is None:
            self.X_unit, self.X_real, self.Y = X_unit, X_real, Y
        else:
            self.X_unit = np.vstack([self.X_unit, X_unit])
            self.X_real = np.vstack([self.X_real, X_real])
            self.Y = np.vstack([self.Y, Y])
        return X_real, Y

    # ------------------------------------------------- block 1: initial design
    def _initial_design(self):
        if self.b["init"] == "S":
            n = min(11 * self.dim - 1, self.budget)
            return ego.initial_design(n, self.dim), "unit"
        return self.hyb._sample_points(self.hyb.n_init, sampler_type="lhs"), "real"

    # ------------------------------------------------------ block 5: weights
    def _weights(self):
        if self.b["weights"] == "S":
            return ego.weight_vectors(self.n_obj)
        # Rule i: the population size follows the initial design actually used.
        n_sub = min(self.hybrid_kwargs["population_size"], self.n_init)
        return self.hyb._generate_weight_vectors(self.n_obj, n_sub)

    # ------------------------------------- blocks 2 + 3: surrogate and its data
    def _fit_hybrid_gp(self, X_real, Y, window):
        """MOEAD-EI Hybrid's GP fit (``_fit_model``) on its own instance."""
        fitter = hybrid_class()(
            budget=self.budget, dim=self.dim, bounds=self.bounds,
            **{**self.hybrid_kwargs, "n_model_points": window},
        )
        fitter.n_obj = self.n_obj
        fitter._fit_model(X_real, Y)
        if len(fitter.models) != self.n_obj:
            return None, None

        def predict(Xr):
            Xs = fitter.scaler_X.transform(Xr)
            mus, sigmas = [], []
            for model in fitter.models:
                mu, sigma = model.predict(Xs, return_std=True)
                mus.append(mu)
                sigmas.append(sigma)
            return np.array(mus).T, np.array(sigmas).T

        fast = hybrid_predictor(fitter.scaler_X, fitter.models) if self.fast_predict else None
        return predict, fast

    def _fit_surrogate(self):
        surrogate, training = self.b["surrogate"], self.b["training"]
        conv = dict(to_unit=self._to_unit, to_real=self._to_real)
        if surrogate == "S" and training == "S":
            mod = ego.ddmtolab()
            models, centers = mod._build_gp_models_fcm(self.X_unit, self.Y)
            return _Predictor(predict_unit=lambda X: mod._gp_evaluate_fcm(X, models, centers),
                              fast_unit=self._fast_fcm(models, centers), **conv)
        if surrogate == "T" and training == "T":
            predict, fast = self._fit_hybrid_gp(self.X_real, self.Y, self.hybrid_kwargs["n_model_points"])
            return None if predict is None else _Predictor(predict_real=predict, fast_real=fast, **conv)
        if surrogate == "S":  # DDMTOLab GP on MOEAD-EI Hybrid's sliding window
            mod = ego.ddmtolab()
            w = self.hybrid_kwargs["n_model_points"]
            Xw, Yw = (self.X_unit[-w:], self.Y[-w:]) if self.X_unit.shape[0] > w else (self.X_unit, self.Y)
            models = [[mod.gp_build(Xw, Yw[:, j:j + 1]) for j in range(self.n_obj)]]
            centers = Xw.mean(axis=0, keepdims=True)
            return _Predictor(predict_unit=lambda X: mod._gp_evaluate_fcm(X, models, centers),
                              fast_unit=self._fast_fcm(models, centers), **conv)
        # MOEAD-EI Hybrid GP as DDMTOLab's local models: one per k-means cluster
        clusters, centers = ego.cluster_assignment(self.X_unit)
        local, local_fast = [], []
        for idx in clusters:
            predict, fast = self._fit_hybrid_gp(self.X_real[idx], self.Y[idx], window=10 ** 9)
            if predict is None:
                return None
            local.append(predict)
            local_fast.append(fast)

        def by_cluster(predictors):
            def predict_unit(Xu):
                nearest = np.argmin(cdist(Xu, centers), axis=1)
                mu = np.zeros((Xu.shape[0], self.n_obj))
                sd = np.zeros((Xu.shape[0], self.n_obj))
                Xr = self._to_real(Xu)
                for ci, predict in enumerate(predictors):
                    mask = nearest == ci
                    if np.any(mask):
                        mu[mask], sd[mask] = predict(Xr[mask])
                return mu, np.maximum(sd, 0)

            return predict_unit

        fast = by_cluster(local_fast) if all(f is not None for f in local_fast) else None
        return _Predictor(predict_unit=by_cluster(local), fast_unit=fast, **conv)

    def _fast_fcm(self, models, centers):
        return ddmtolab_fcm_predictor(models, centers) if self.fast_predict else None

    # ---------------------------------------------------- block 7: ideal point
    def _ideal_point(self, predictor):
        z_obs = self.Y.min(axis=0)
        if self.b["ideal"] == "T":
            return z_obs
        return ego.estimate_ideal_point(
            self.dim, lambda X: predictor.predict(X, "unit", fast=self.fast_predict)[0], self.W, z_obs
        )

    # ---------------------------------------- blocks 4 + 6: scaling, scalarising
    def _scaler(self, z, Y_ref):
        """Return ``(scale_y, scale_s, z_scaled)`` for objective scaling (block 4)."""
        if self.b["scaling"] == "S":
            return (lambda Y: Y), (lambda s: s), z
        rng = Y_ref.max(axis=0) - z  # MOEAD-EI Hybrid: nadir - ideal, zeros -> 1e-6
        rng[rng == 0] = 1e-6
        return (lambda Y: (Y - z) / rng), (lambda s: s / rng), np.zeros(self.n_obj)

    def _scalarise(self, Y, w, z):
        """Block 6 on objectives ``Y`` (rows) for weights ``w`` (one vector or rows)."""
        if self.b["scalarisation"] == "T":
            return self.hyb._augmented_tchebycheff_scalarization(Y, w, z, self.rho)
        return np.max(w * (Y - z), axis=-1)

    def _gmin(self, scale_y, z_s):
        """Best observed scalarised value of every subproblem."""
        Ys = scale_y(self.Y)
        if self.b["scalarisation"] == "S":
            return ego.ddmtolab()._get_gmin(Ys, self.W, z_s)
        return np.array([
            np.min(self.hyb._augmented_tchebycheff_scalarization(Ys, self.W[j], z_s, self.rho))
            for j in range(self.W.shape[0])
        ])

    # ---------------------------------------------------- block 8: acquisition
    def _acquisition(self, scale_y, scale_s, z_s):
        """``acq(mean, std, W, gmin) -> values``; W and gmin per row or one each."""
        scal_t = self.b["scalarisation"] == "T"

        if self.b["acquisition"] == "S":
            get_eti = ego.get_eti  # DDMTOLab's _get_eti, same values (moead_ego_parts)

            def acq(mu, sd, W, g):
                mu, sd = scale_y(mu), scale_s(sd)
                if scal_t:  # rule a
                    return ego.eti_augmented(mu, sd, W, g, z_s, self.rho)
                return get_eti(mu, sd, W, g, z_s)

            return acq

        def acq(mu, sd, W, g):  # MOEAD-EI Hybrid's EI (rule b covers plain Tchebycheff)
            mu, sd = scale_y(mu), scale_s(sd)
            mean = self._scalarise(mu, W, z_s)
            k_max = np.argmax(W * (mu - z_s), axis=1)
            sigma = sd[np.arange(mu.shape[0]), k_max]
            u = np.zeros_like(sigma)
            ei = np.zeros_like(sigma)
            ok = sigma > 1e-10
            u[ok] = (g - mean[ok]) / sigma[ok] if np.ndim(g) == 0 else (g[ok] - mean[ok]) / sigma[ok]
            ei[ok] = sigma[ok] * (u[ok] * ego.norm_cdf(u[ok]) + ego.norm_pdf(u[ok]))
            return ei

        return acq

    # --------------------------------------------- block 9: candidate search
    def _init_population(self):
        """MOEAD-EI Hybrid's population of evaluated points (search block T)."""
        n_sub = self.W.shape[0]
        if self.X_real.shape[0] >= n_sub:  # MOEAD-EI Hybrid: the initial design itself
            self.X_pop, self.F_pop = self.X_real.copy(), self.Y.copy()
            return
        # Rule d: each subproblem starts from its best evaluated point.
        z = self.Y.min(axis=0)
        scale_y, _, z_s = self._scaler(z, self.Y)
        Ys = scale_y(self.Y)
        best = [int(np.argmin(self._scalarise(Ys, self.W[j], z_s))) for j in range(n_sub)]
        self.X_pop, self.F_pop = self.X_real[best].copy(), self.Y[best].copy()

    def _update_population(self, X_new, Y_new, z_iter):
        """MOEAD-EI Hybrid's replacement over every subproblem (search block T)."""
        z = np.minimum(z_iter, self.Y.min(axis=0))
        scale_y, _, z_s = self._scaler(z, self.Y)
        for x, f in zip(X_new, Y_new):
            f_s = scale_y(f)
            for j in range(self.W.shape[0]):
                if self._scalarise(f_s, self.W[j], z_s) < self._scalarise(scale_y(self.F_pop[j]), self.W[j], z_s):
                    self.X_pop[j] = x
                    self.F_pop[j] = f

    def _search(self, predictor, acq, gmin):
        """Return ``(candidates, native, best_per_sub, cand_acq, sub_acq)``.

        ``best_per_sub[j]`` is the candidate that is best for subproblem j and
        ``sub_acq[c]`` the best value of candidate c over the subproblems.
        """
        if self.b["search"] == "S":
            pop_acq, pop_x = ego.moead_search(
                self.dim, lambda X: predictor.predict(X, "unit", fast=self.fast_predict), self.W, gmin, acq
            )
            return pop_x, "unit", np.arange(pop_x.shape[0]), pop_acq, pop_acq

        hyb = self.hyb
        offspring = []
        for _ in range(hyb.n_offspring_surrogate // 2):
            idx1, idx2 = np.random.choice(len(self.X_pop), 2, replace=False)
            c1, c2 = hyb._sbx_crossover(self.X_pop[idx1], self.X_pop[idx2], self.lb, self.ub, hyb.eta_c)
            c1 = hyb._polynomial_mutation(c1, self.lb, self.ub, hyb.eta_m)
            c2 = hyb._polynomial_mutation(c2, self.lb, self.ub, hyb.eta_m)
            offspring.extend([c1, c2])
        X_off = np.array(offspring[: hyb.n_offspring_surrogate])
        mu, sd = predictor.predict(X_off, "real")
        A = np.zeros((X_off.shape[0], self.W.shape[0]))
        for j in range(self.W.shape[0]):
            A[:, j] = acq(mu, sd, self.W[j], gmin[j])
        return X_off, "real", np.argmax(A, axis=0), A.max(axis=1), A

    # --------------------------------------------- block 10: batch selection
    def _select(self, cand, native, best_per_sub, cand_acq, acq_info, remaining):
        if self.b["selection"] == "S":
            q = min(self.s_batch_size, remaining)
            cand_unit = cand if native == "unit" else self._to_unit(cand)  # rule f
            Q, Q_acq = ego.filter_candidates(cand_unit, cand_acq, self.X_unit, self.dim)
            new_x = ego.ddmtolab()._kmeans_batch_select(Q, q, cand_unit, Q_acq)
            new_x = ego.remove_duplicates(new_x, self.X_unit)
            if new_x.shape[0] == 0:
                new_x = np.clip(np.random.rand(1, self.dim), 0, 1)
            return new_x[:q], "unit"

        q = self.hybrid_kwargs["batch_size"]
        unique = np.unique(best_per_sub)
        if len(unique) > q:
            if native == "real":  # max EI over the subproblems each was best for
                score = np.max(acq_info[unique], axis=1)
            else:  # rule g: the MOEA/D-DE population holds one candidate per subproblem
                score = cand_acq[unique]
            chosen = unique[np.argsort(score)[::-1]][:q]
        else:
            chosen = unique
            if len(chosen) < q:
                available = np.setdiff1d(np.arange(cand.shape[0]), chosen)
                if len(available) > 0:
                    extra = np.random.choice(available, min(q - len(chosen), len(available)), replace=False)
                    chosen = np.concatenate((chosen, extra))
        if len(chosen) == 0:
            return self.hyb._sample_points(q, sampler_type="random"), "real"
        return cand[chosen], native

    # ------------------------------------------------------------------- run
    def __call__(self, func):
        if self.seed is not None:
            seed_everything(self.seed)
        self.n_evals, self.n_obj = 0, None
        self.X_unit = self.X_real = self.Y = None

        X0, native = self._initial_design()
        self._evaluate(func, X0, native)
        self.n_init = self.Y.shape[0]
        self.W = self._weights()
        if self.b["search"] == "T":
            self._init_population()

        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals
            try:
                predictor = self._fit_surrogate()
            except Exception as e:
                if self.b["surrogate"] != "S":
                    raise
                # DDMTOLab: if the GPs cannot be built, evaluate a random batch.
                self.diagnostics["ask_fallbacks"] += 1
                if self.diagnostics["first_error"] is None:
                    self.diagnostics["first_error"] = f"{type(e).__name__}: {e}"
                q = min(self.s_batch_size, remaining) if self.b["selection"] == "S" else self.hybrid_kwargs["batch_size"]
                self._evaluate(func, np.random.rand(q, self.dim), "unit")
                continue
            if predictor is None:  # MOEAD-EI Hybrid: fewer than two points to fit
                self._evaluate(func, self.hyb._sample_points(self.hybrid_kwargs["batch_size"], "random"), "real")
                continue

            z = self._ideal_point(predictor)
            scale_y, scale_s, z_s = self._scaler(z, self.Y)
            gmin = self._gmin(scale_y, z_s)
            acq = self._acquisition(scale_y, scale_s, z_s)
            cand, cnative, best_per_sub, cand_acq, acq_info = self._search(predictor, acq, gmin)
            X_new, xnative = self._select(cand, cnative, best_per_sub, cand_acq, acq_info, remaining)
            X_real_new, Y_new = self._evaluate(func, X_new, xnative)
            if self.b["search"] == "T" and Y_new.shape[0] > 0:
                self._update_population(X_real_new, Y_new, z)

        return self.Y, self.X_real
