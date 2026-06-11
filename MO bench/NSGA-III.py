import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


class NSGA3Wrapper:
    def __init__(self, budget, dim, bounds):
        """Initialize with the standard interface expected by MultiObjEvaluator."""
        self.budget = budget
        self.dim = dim
        self.bounds = bounds

    def __call__(self, func):
        """Run NSGA-III using the provided evaluation function.

        Args:
            func: A callable that takes x (shape: [dim]) and returns the
                objectives F (shape [n_obj]) on unconstrained problems, or the
                tuple (F, G) with inequality constraints G (shape [n_constr],
                G <= 0 feasible) on constrained ones.
        """
        # Probe once to infer #objectives and, for constrained problems, the
        # number of inequality constraints. func returns (F, G) when constrained.
        test_x = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        test_out = func(test_x)
        constrained = isinstance(test_out, tuple) and len(test_out) == 2
        if constrained:
            f0, g0 = test_out
            n_obj = len(np.asarray(f0).ravel())
            n_constr = len(np.asarray(g0).ravel())
        else:
            n_obj = len(np.asarray(test_out).ravel())
            n_constr = 0

        # Define pymoo Problem that wraps the provided func. Declaring
        # n_ieq_constr lets NSGA-III handle feasibility via the Constraint
        # Dominance Principle natively; pymoo's G <= 0 convention matches ours.
        class WrappedProblem(Problem):
            def __init__(self, n_var, n_obj, n_constr, bounds, eval_func):
                xl = np.full(n_var, bounds[0])
                xu = np.full(n_var, bounds[1])
                super().__init__(
                    n_var=n_var, n_obj=n_obj, n_ieq_constr=n_constr, xl=xl, xu=xu
                )
                self.eval_func = eval_func
                self._constrained = n_constr > 0

            def _evaluate(self, X, out, *args, **kwargs):
                # X is (pop_size, n_var)
                if self._constrained:
                    F, G = [], []
                    for x in X:
                        f, g = self.eval_func(x)
                        F.append(np.asarray(f).ravel())
                        G.append(np.asarray(g).ravel())
                    out["F"] = np.array(F)
                    out["G"] = np.array(G)
                else:
                    out["F"] = np.array([self.eval_func(x) for x in X])

        problem = WrappedProblem(
            n_var=self.dim,
            n_obj=n_obj,
            n_constr=n_constr,
            bounds=self.bounds,
            eval_func=func,
        )

        # Das-Dennis reference directions; keep the count small to match the
        # tight evaluation budget (mirrors the small NSGA-II population).
        n_partitions = 4 if n_obj == 2 else 2
        ref_dirs = get_reference_directions(
            "das-dennis", n_obj, n_partitions=n_partitions
        )

        # NSGA-III population size tracks the number of reference directions
        pop_size = min(len(ref_dirs), self.budget)

        res = minimize(
            problem,
            NSGA3(pop_size=pop_size, ref_dirs=ref_dirs),
            termination=("n_eval", self.budget),
            seed=None,
            verbose=False,
        )

        return res.X, res.F
