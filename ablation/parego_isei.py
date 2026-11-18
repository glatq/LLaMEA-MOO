"""ParEGO and Improved-Scalarized-EI as one algorithm built from swappable blocks.

``ModularParEGOISEI(config=...)`` runs the configuration named by a seven-letter
S/T string (``ablation.config.ISEI_BLOCKS``): each block comes either from ParEGO
as implemented in DDMTOLab (``S``) or from the tuned Improved-Scalarized-EI
(``T``, ISEI). Both evaluate one point per iteration:

1. ``init``           Latin hypercube of 11d-1 points | Sobol, 2d+1 points
2. ``weights``        one drawn at random from 100 (M=2) / 91 (M=3) uniform vectors |
                      48 Sobol vectors taken in turn
3. ``scalarisation``  augmented Tchebycheff (rho 0.05) on objectives normalised by
                      their observed range | weighted sum, ISEI's normalisation
4. ``modelled``       one GP on the scalarised value | one GP per objective
5. ``gp``             DDMTOLab's BoTorch SingleTaskGP | ISEI's scikit-learn
                      C * Matern-5/2 + noise
6. ``training``       all evaluated points | the 91 most recent
7. ``search``         LogEI maximised by BoTorch's optimiser (5 restarts, 20 raw
                      samples) | EI maximised by ISEI's genetic algorithm (85 x 39)

Each block calls the code of the algorithm it comes from (``ablation.
parego_isei_parts``). Both pure configurations make the same random draws in the
same order as their originals, so under the same seed ``TTTTTTT`` reproduces ISEI
and ``SSSSSSS`` reproduces DDMTOLab's ParEGO evaluation for evaluation
(tests/test_ablation_isei.py; ``fast_predict=False``).

``fast_predict=True`` lets ISEI's genetic algorithm (block 7 T) predict a whole
generation in one call instead of one individual at a time: the same model and the
same formulas, which differ only in floating-point summation order.

Mixed configurations follow these rules:

* the scalarisation brings its own normalisation (they differ only in the 1e-10
  floor and the 1e-6 added to the range);
* the incumbent of the expected improvement is the best scalarised value over all
  evaluated points (as in both originals);
* a GP on the scalarised value needs the weight before it is fitted, so ISEI's
  weights (block 2 T) are then taken before the fit rather than after it;
* per-objective GPs with the augmented Tchebycheff (blocks 4 T, 3 S) give the
  scalarised value's mean and variance by Clark's moment matching;
* BoTorch's search on other models (block 7 S) runs ``optimize_logei`` unchanged
  on a model whose posterior is the scalarised Gaussian (``ScalarisedModel``);
  scikit-learn GPs enter it through their posterior rebuilt in torch.

Coordinates: the S blocks work in the unit cube (as DDMTOLab) and the T blocks in
the problem's bounds (as ISEI). The archive keeps every point in both.
"""

import numpy as np

from . import moead_ego_parts as ego
from . import parego_isei_parts as pp
from .config import ISEI_BLOCKS, parse_config
from .modular import seed_everything

SIGMA_FLOOR = pp.SIGMA_FLOOR  # ISEI's floor on predicted standard deviations


class ModularParEGOISEI:
    """ParEGO <-> Improved-Scalarized-EI, one configuration of the ablation."""

    def __init__(self, budget, dim, bounds=None, config="TTTTTTT", seed=None,
                 fast_predict=False, **isei_kwargs):
        self.budget = int(budget)
        self.dim = int(dim)
        if bounds is None:
            bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)
        self.lb, self.ub = self.bounds[0], self.bounds[1]
        self.config = str(config).strip().upper()
        self.b = parse_config(self.config, ISEI_BLOCKS)
        self.seed = seed
        # fast_predict: one prediction call per generation in ISEI's genetic algorithm
        self.fast_predict = bool(fast_predict) and self.b["search"] == "T"
        self.isei_kwargs = {**pp.ISEI_TUNED, **isei_kwargs}
        # "ask_fallbacks" counts ISEI GP fits that fell back to no optimiser restarts.
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
            return
        X = X[:n]
        if native == "unit":  # DDMTOLab: decs * (ub - lb) + lb, evaluated as is
            X_unit, X_real = X, self._to_real(X)
            to_eval = X_real
        else:  # ISEI: evaluates the clipped point, stores the point
            X_real, X_unit = X, self._to_unit(X)
            to_eval = np.clip(X_real, self.lb, self.ub)
        Y = []
        for x in to_eval:
            y = np.asarray(func(x), dtype=float).ravel()
            if self.n_obj is None:
                self.n_obj = y.shape[0]
            Y.append(y)
        Y = np.array(Y)
        self.n_evals += n
        if self.X_unit is None:
            self.X_unit, self.X_real, self.Y = X_unit, X_real, Y
        else:
            self.X_unit = np.vstack([self.X_unit, X_unit])
            self.X_real = np.vstack([self.X_real, X_real])
            self.Y = np.vstack([self.Y, Y])

    # ------------------------------------------------ block 1: initial design
    def _initial_design(self):
        if self.b["init"] == "S":
            return ego.initial_design(min(11 * self.dim - 1, self.budget), self.dim), "unit"
        return self.isei._sample_points(self.isei.n_init), "real"

    # ------------------------------------------------------ block 2: weights
    def _isei_weight(self):
        if self.U is None:
            self.U = self.isei._generate_uniform_weights(self.n_obj, self.isei.n_uniform_weights)
        return self.U[self.n_evals % self.isei.n_uniform_weights]

    # ----------------------------------------------- block 3: scalarisation
    def _normalised(self):
        return pp.parego_normalise(self.Y) if self.b["scalarisation"] == "S" else pp.isei_normalise(self.Y)

    def _scalarised(self, Yn, w):
        """Scalarised values of every evaluated point (the GP target and incumbent)."""
        if self.b["scalarisation"] == "S":
            return pp.parego_scalarise(self.Y, w)  # DDMTOLab's own, normalisation included
        return pp.isei_weighted_sum(Yn, w)

    # ---------------------------------------------- blocks 4-6: the models
    def _window(self):
        n = self.Y.shape[0]
        if self.b["training"] == "S":
            return slice(0, n)
        return slice(max(0, n - self.isei.model_window_size), n)

    def _fit(self, X_unit, X_real, y):
        """One GP on target ``y`` (block 5)."""
        if self.b["gp"] == "S":
            return pp.botorch_gp(X_unit, y)
        gp, fell_back = pp.fit_isei_gp(X_real, y, self.isei.rng, self.isei.gp_restarts_optimizer)
        self.diagnostics["ask_fallbacks"] += int(fell_back)
        return gp

    def _predict(self, gp, X_real):
        """``(mean, std)`` of one GP at ``X_real`` (numpy, ISEI's genetic algorithm)."""
        if self.b["gp"] == "S":
            return pp.botorch_predict(gp, self._to_unit(X_real))
        return gp.predict(X_real, return_std=True)

    def _torch_posterior(self, gp):
        """``f(X_unit) -> (mean, var)`` of one GP, differentiable (BoTorch's search)."""
        import torch

        if self.b["gp"] == "S":
            def f(X):
                post = gp.posterior(X.to(torch.float32))
                return -post.mean[..., 0].to(torch.float64), post.variance[..., 0].to(torch.float64)

            return f
        g = pp.sklearn_gp_torch(gp)
        lb = torch.tensor(self.lb, dtype=torch.float64)
        span = torch.tensor(self.ub - self.lb, dtype=torch.float64)
        return lambda X: g(X * span + lb)

    # -------------------------------------- the scalarised Gaussian at points
    def _combine(self, mu, sd, w):
        """Per-objective means/stds (numpy) -> mean and std of the scalarised value."""
        sd = np.maximum(SIGMA_FLOOR, sd)
        if self.b["scalarisation"] == "T":  # ISEI: independent GPs, weighted sum
            return np.sum(w * mu, axis=1), np.sqrt(np.sum(w**2 * sd**2, axis=1))
        return pp.clark_moments(mu, sd, w)

    def _fitness(self, models, w, f_best):
        """ISEI's fitness: EI of the scalarised value, for a population (real coordinates)."""

        def ei(X):
            if self.b["modelled"] == "S":
                m, s = self._predict(models[0], X)
                m, s = np.asarray(m, dtype=float), np.maximum(SIGMA_FLOOR, np.asarray(s, dtype=float))
            else:
                mu = np.empty((X.shape[0], self.n_obj))
                sd = np.empty((X.shape[0], self.n_obj))
                for i, gp in enumerate(models):
                    mu[:, i], sd[:, i] = self._predict(gp, X)
                m, s = self._combine(mu, sd, w)
            return pp.isei_ei(m, s, f_best)

        if self.fast_predict:
            return ei
        return lambda P: np.array([ei(x[None, :])[0] for x in P])  # one individual at a time

    def _torch_scalarised(self, models, w):
        """``f(X_unit) -> (mean, var)`` of the scalarised value, for BoTorch's search."""
        import torch

        posts = [self._torch_posterior(gp) for gp in models]
        wt = torch.as_tensor(w, dtype=torch.float64)

        def f(X):
            if self.b["modelled"] == "S":  # the GP models the scalarised value itself
                m, v = posts[0](X)
                return m, torch.clamp(v, min=SIGMA_FLOOR**2)
            ms, vs = zip(*(p(X) for p in posts))
            mu = torch.stack(ms, -1)
            var = torch.clamp(torch.stack(vs, -1), min=SIGMA_FLOOR**2)
            if self.b["scalarisation"] == "T":
                return (wt * mu).sum(-1), (wt * wt * var).sum(-1)
            return pp.augmented_tchebycheff_moments_torch(mu, var, wt, pp.PAREGO_RHO)

        return f

    # ------------------------------------------------------------------- run
    def __call__(self, func):
        if self.seed is not None:
            seed_everything(self.seed)
        # The T blocks call the methods of this ISEI instance (and draw from its rng).
        self.isei = pp.isei_class()(
            budget=self.budget, dim=self.dim, bounds=self.bounds,
            random_seed=42 if self.seed is None else self.seed, **self.isei_kwargs,
        )
        self.n_evals, self.n_obj = 0, None
        self.X_unit = self.X_real = self.Y = None
        self.U = None

        X0, native = self._initial_design()
        self._evaluate(func, X0, native)
        W = pp.parego_weights(self.n_obj) if self.b["weights"] == "S" else None

        while self.n_evals < self.budget:
            w = None
            if W is not None:  # ParEGO: a weight drawn at random, before the fit
                w = W[np.random.randint(0, W.shape[0])]
            elif self.b["modelled"] == "S":  # the GP's target needs it
                w = self._isei_weight()
            Yn = self._normalised()
            win = self._window()
            if self.b["modelled"] == "S":
                s_all = self._scalarised(Yn, w)
                models = [self._fit(self.X_unit[win], self.X_real[win], s_all[win])]
            else:
                models = [self._fit(self.X_unit[win], self.X_real[win], Yn[win, j]) for j in range(self.n_obj)]
                if w is None:  # ISEI: the weight comes after the fit
                    w = self._isei_weight()
                s_all = self._scalarised(Yn, w)

            if self.b["search"] == "S":
                if self.b["modelled"] == "S" and self.b["gp"] == "S":
                    model = models[0]  # DDMTOLab's GP itself
                else:
                    model = pp.ScalarisedModel(self._torch_scalarised(models, w))
                self._evaluate(func, pp.optimize_logei(model, s_all, self.dim), "unit")
            else:
                X_pareto = self.isei._get_pareto_front(self.X_real, self.Y)[0]
                fitness = self._fitness(models, w, float(np.min(s_all)))
                self._evaluate(func, pp.isei_genetic_algorithm(self.isei, X_pareto, fitness), "real")

        return self.Y, self.X_real

