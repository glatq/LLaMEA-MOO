import numpy as np


class RandomSearchMO:
    def __init__(self, budget=10000, dim=10, bounds: np.ndarray | None = None):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.array([[0.0] * dim, [1.0] * dim]) if bounds is None else bounds

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        # Pareto dominance for minimization
        return np.all(a <= b) and np.any(a < b)

    @classmethod
    def _constrained_dominates(cls, fa, cva, fb, cvb) -> bool:
        """Constraint Dominance Principle (Deb): feasible beats infeasible;
        among feasible, Pareto dominance on objectives; among infeasible, the
        smaller total constraint violation wins. ``cv`` is sum(max(0, G))."""
        if cva == 0.0 and cvb == 0.0:
            return cls._dominates(fa, fb)
        if cva == 0.0 and cvb > 0.0:
            return True
        if cva > 0.0 and cvb == 0.0:
            return False
        return cva < cvb

    def __call__(self, func):
        X = []  # decision vectors in archive
        F = []  # objective vectors in archive
        CV = []  # total constraint violation per archive point (0 == feasible)

        low, high = self.bounds[0], self.bounds[1]

        # Probe once: constrained problems return (F, G) with G <= 0 feasible;
        # unconstrained ones return F only. (numpy>=1.24 raises on np.asarray of
        # a ragged (F, G) tuple, which is why the old objectives-only path
        # crashed whenever n_obj != n_constr.)
        def split(out):
            if isinstance(out, tuple) and len(out) == 2:
                f = np.asarray(out[0], dtype=float).ravel()
                g = np.asarray(out[1], dtype=float).ravel()
                return f, float(np.maximum(0.0, g).sum())
            return np.asarray(out, dtype=float).ravel(), 0.0

        for _ in range(self.budget):
            x = np.random.uniform(low, high)  # shape (dim,)
            f, cv = split(func(x))

            # If (constraint-)dominated by any current archive point -> skip
            if any(
                self._constrained_dominates(fi, cvi, f, cv)
                for fi, cvi in zip(F, CV)
            ):
                continue

            # Otherwise, drop archive points the newcomer (constraint-)dominates
            keep = [
                i
                for i, (fi, cvi) in enumerate(zip(F, CV))
                if not self._constrained_dominates(f, cv, fi, cvi)
            ]
            if len(keep) != len(F):
                X = [X[i] for i in keep]
                F = [F[i] for i in keep]
                CV = [CV[i] for i in keep]

            X.append(x)
            F.append(f)
            CV.append(cv)

        if len(F) == 0:
            return np.empty((0,)), np.empty((0, self.dim))

        # Report the feasible non-dominated front when any feasible point exists;
        # otherwise return the full (infeasible) archive. The evaluator computes
        # feasible-HV / feasibility from its own per-call (F, G) recording, so the
        # return value only seeds the final reported front.
        feas = [i for i, cv in enumerate(CV) if cv == 0.0]
        idx = feas if feas else list(range(len(F)))
        return np.vstack([F[i] for i in idx]), np.vstack([X[i] for i in idx])
