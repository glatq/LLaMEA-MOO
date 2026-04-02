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

    def __call__(self, func):
        X = []  # decision vectors in archive
        F = []  # objective vectors in archive

        low, high = self.bounds[0], self.bounds[1]

        for _ in range(self.budget):
            x = np.random.uniform(low, high)  # shape (dim,)
            f = np.asarray(func(x), dtype=float).ravel()  # shape (M,)

            # If dominated by any current archive point -> skip
            if any(self._dominates(fi, f) for fi in F):
                continue

            # Otherwise, remove points dominated by the newcomer
            keep = [i for i, fi in enumerate(F) if not self._dominates(f, fi)]
            if len(keep) != len(F):
                X = [X[i] for i in keep]
                F = [F[i] for i in keep]

            # Add newcomer
            X.append(x)
            F.append(f)

        # Return as numpy arrays, one row per solution
        if len(F) == 0:
            # No nondominated found (only possible if budget==0); fall back to empty arrays
            return np.empty((0,)), np.empty((0, self.dim))
        return np.vstack(F), np.vstack(X)
