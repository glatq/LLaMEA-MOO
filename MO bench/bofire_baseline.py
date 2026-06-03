"""BOFire-backed multi-objective Bayesian optimization baselines.

These wrappers expose BOFire's strategies through the same interface the
``MultiObjEvaluator`` expects from every algorithm
(``__init__(budget, dim, bounds)`` and ``__call__(func) -> (F, X)``), so they
can be benchmarked side by side with the LLaMEA-generated algorithms and the
other baselines.

We use BOFire (a maintained, external MOBO framework that wraps BoTorch) for
the state-of-the-art Bayesian baselines instead of in-house wrappers, so the
SOTA comparison rests on a trusted third-party implementation:

  * ``BoFireQparEGOWrapper``   -> BOFire ``QparegoStrategy``  (qParEGO)
  * ``BoFireQLogNEHVIWrapper`` -> BOFire ``MoboStrategy`` with qLogNEHVI

The number of objectives is inferred at runtime from the first evaluation
(consistent with the other wrappers), so the same class handles bi- and
many-objective problems without reconfiguration.
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import bofire.strategies.api as fstrat
from bofire.data_models.acquisition_functions.api import qLogNEHVI
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.strategies.api import MoboStrategy, QparegoStrategy

warnings.filterwarnings("ignore")


class _BoFireMOBOBase(ABC):
    """Shared ask/tell driver for BOFire multi-objective strategies.

    Concrete baselines subclass this and implement :meth:`_make_data_model`
    to supply their specific BOFire strategy; everything else (initial design,
    objective-count inference, the ask/tell loop) is shared here.
    """

    def __init__(self, budget, dim, bounds, n_init=None, batch_size=1, seed=None):
        self.budget = int(budget)
        self.dim = int(dim)
        b = np.asarray(bounds, dtype=float)
        self.lower = b[0]
        self.upper = b[1]
        self.batch_size = max(1, int(batch_size))
        # Initial design: 2*dim points by default, capped to half the budget,
        # mirroring the repo's baseline convention (>= 2, never above budget).
        if n_init is None:
            n_init = min(2 * self.dim, self.budget // 2)
        self.n_init = max(2, min(int(n_init), self.budget))
        self.seed = seed

    @abstractmethod
    def _make_data_model(self, domain):
        """Return the BOFire strategy data model for this baseline.

        Implemented by each concrete subclass (template-method pattern).
        """

    def __call__(self, func):
        rng = np.random.default_rng(self.seed)
        span = self.upper - self.lower

        def sample(n):
            return self.lower + span * rng.random((n, self.dim))

        # Probe once to infer the number of objectives; reuse the point as the
        # first initial-design sample so no evaluation budget is wasted.
        x0 = sample(1)[0]
        y0 = np.asarray(func(x0), dtype=float).ravel()
        n_obj = int(y0.shape[0])

        in_keys = [f"x{i}" for i in range(self.dim)]
        out_keys = [f"y{j}" for j in range(n_obj)]

        inputs = Inputs(
            features=[
                ContinuousInput(
                    key=in_keys[i],
                    bounds=(float(self.lower[i]), float(self.upper[i])),
                )
                for i in range(self.dim)
            ]
        )
        outputs = Outputs(
            features=[
                ContinuousOutput(key=k, objective=MinimizeObjective(w=1.0))
                for k in out_keys
            ]
        )
        domain = Domain(inputs=inputs, outputs=outputs)

        X = [np.asarray(x0, dtype=float)]
        Y = [y0]
        n_used = 1

        def make_df(xs, ys):
            data = {in_keys[i]: [float(x[i]) for x in xs] for i in range(self.dim)}
            for j, k in enumerate(out_keys):
                data[k] = [float(y[j]) for y in ys]
                data[f"valid_{k}"] = [1] * len(ys)
            return pd.DataFrame(data)

        # Remaining initial design (LHS-like uniform random).
        n_more = max(0, min(self.n_init, self.budget) - n_used)
        if n_more > 0:
            for xr in sample(n_more):
                if n_used >= self.budget:
                    break
                Y.append(np.asarray(func(xr), dtype=float).ravel())
                X.append(np.asarray(xr, dtype=float))
                n_used += 1

        strategy = fstrat.map(self._make_data_model(domain))
        strategy.tell(make_df(X, Y))

        # Bayesian optimization loop (tell appends only the new batch).
        while n_used < self.budget:
            q = min(self.batch_size, self.budget - n_used)
            try:
                cand = strategy.ask(q)
                new_x = [
                    cand.iloc[r][in_keys].to_numpy(dtype=float)
                    for r in range(len(cand))
                ]
            except Exception:
                # Robustness: if the surrogate/acqf step fails, fall back to
                # random candidates so the run still uses its full budget.
                new_x = list(sample(q))

            new_x_used, new_y = [], []
            for xr in new_x:
                if n_used >= self.budget:
                    break
                yr = np.asarray(func(xr), dtype=float).ravel()
                new_x_used.append(np.asarray(xr, dtype=float))
                new_y.append(yr)
                X.append(np.asarray(xr, dtype=float))
                Y.append(yr)
                n_used += 1

            if new_x_used:
                try:
                    strategy.tell(make_df(new_x_used, new_y))
                except Exception:
                    pass

        return np.vstack(Y), np.vstack(X)


class BoFireQparEGOWrapper(_BoFireMOBOBase):
    """BOFire qParEGO (random Tchebycheff scalarization + EI)."""

    def _make_data_model(self, domain):
        kwargs = {"domain": domain}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        return QparegoStrategy(**kwargs)


class BoFireQLogNEHVIWrapper(_BoFireMOBOBase):
    """BOFire MOBO with the qLogNEHVI acquisition function."""

    def _make_data_model(self, domain):
        kwargs = {"domain": domain, "acquisition_function": qLogNEHVI()}
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        return MoboStrategy(**kwargs)
