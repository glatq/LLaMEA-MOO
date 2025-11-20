# providers/ioh_provider.py
import numpy as np
from ioh import get_problem


class IOHProvider:
    def get(self, problem_id: int, instance_id: int, dim: int):
        p = get_problem(problem_id, instance_id, dim)

        class _Wrapper:
            def __init__(self, p):
                self._p = p
                pname = getattr(getattr(p, "meta", None), "name", None)
                if pname is None and hasattr(p, "problems"):
                    pname = p.problems.get(problem_id, "unknown")
                self.name = f"F{problem_id}-{pname}"
                self.bounds = np.array([p.bounds.lb, p.bounds.ub])
                self.optimum_x = getattr(p.optimum, "x", None)
                self.optimum_y = getattr(p.optimum, "y", None)
                self.evaluations = 0  # local counter for budget guard

            # expose live IOH state instead of caching a snapshot
            @property
            def state(self):
                return self._p.state

            def __call__(self, x):
                y = self._p(x)  # IOH updates its own state internally
                self.evaluations += 1  # keep local evals for our guard
                return y

        return _Wrapper(p)
