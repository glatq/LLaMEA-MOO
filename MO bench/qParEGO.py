import numpy as np
import torch
import warnings
from botorch.models import SingleTaskGP

warnings.filterwarnings("ignore")
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class qParEGOWrapper:
    """BoTorch qParEGO: scalarized EI with random Tchebycheff weights."""

    def __init__(self, budget, dim, bounds, n_init_ratio=0.25, batch_size=1,
                 num_restarts=3, raw_samples=128, rho=0.05):
        self.budget = budget
        self.dim = dim
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_init = max(min(int(budget * n_init_ratio), 2 * dim + 1), 2)
        self.n_init = min(self.n_init, budget)
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.rho = rho
        self.tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    def _augmented_tchebycheff(self, Y, weights):
        """Scalarize objectives using augmented Tchebycheff (minimization)."""
        weighted = weights * Y
        return weighted.max(dim=-1).values + self.rho * weighted.sum(dim=-1)

    def __call__(self, func):
        lb = self.bounds[0]
        ub = self.bounds[1]
        bounds_torch = torch.tensor(np.vstack([lb, ub]), **self.tkwargs)
        standard_bounds = torch.zeros(2, self.dim, **self.tkwargs)
        standard_bounds[1] = 1.0

        # Initial LHS design
        from scipy.stats import qmc as scipy_qmc
        sampler = scipy_qmc.LatinHypercube(d=self.dim)
        X_init_unit = sampler.random(self.n_init)
        X_init = scipy_qmc.scale(X_init_unit, lb, ub)

        X_all = []
        Y_all = []
        for x in X_init:
            y = np.asarray(func(x), dtype=np.float64).ravel()
            if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
                X_all.append(x)
                Y_all.append(y)

        n_evals = self.n_init
        n_obj = len(Y_all[0])

        while n_evals < self.budget:
            X_tensor = torch.tensor(np.array(X_all), **self.tkwargs)
            Y_tensor = torch.tensor(np.array(Y_all), **self.tkwargs)

            # Normalize X to [0, 1]
            X_norm = normalize(X_tensor, bounds_torch)

            # Normalize objectives to [0, 1] using observed ideal/nadir
            ideal = Y_tensor.min(dim=0).values
            nadir = Y_tensor.max(dim=0).values
            obj_range = nadir - ideal
            obj_range = torch.clamp(obj_range, min=1e-8)
            Y_norm = (Y_tensor - ideal) / obj_range

            # Random Tchebycheff weight vector
            weights = torch.rand(n_obj, **self.tkwargs)
            weights = weights / weights.sum()

            # Scalarize (minimization -> negate for BoTorch maximization)
            Y_scalar = -self._augmented_tchebycheff(Y_norm, weights)
            Y_scalar = Y_scalar.unsqueeze(-1)

            # Fit GP on scalarized objective
            gp = SingleTaskGP(X_norm, Y_scalar)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass

            # qLogEI acquisition
            best_f = Y_scalar.max()
            acqf = qLogExpectedImprovement(
                model=gp,
                best_f=best_f,
            )

            actual_batch = min(self.batch_size, self.budget - n_evals)
            try:
                candidates, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=standard_bounds,
                    q=actual_batch,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
            except Exception:
                candidates = torch.rand(actual_batch, self.dim, **self.tkwargs)

            # Unnormalize and evaluate
            X_new = unnormalize(candidates.detach(), bounds_torch).cpu().numpy()
            for x in X_new:
                x = np.clip(x, lb, ub)
                y = np.asarray(func(x), dtype=np.float64).ravel()
                n_evals += 1
                if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
                    X_all.append(x)
                    Y_all.append(y)
                if n_evals >= self.budget:
                    break

        # Return Pareto front
        Y_final = np.array(Y_all)
        X_final = np.array(X_all)
        is_nondom = self._get_nondominated(Y_final)
        return Y_final[is_nondom], X_final[is_nondom]

    @staticmethod
    def _get_nondominated(Y):
        n = Y.shape[0]
        is_nondom = np.ones(n, dtype=bool)
        for i in range(n):
            if is_nondom[i]:
                dominated = np.all(Y[i] <= Y, axis=1) & np.any(Y[i] < Y, axis=1)
                dominated[i] = False
                is_nondom[dominated] = False
        return is_nondom
