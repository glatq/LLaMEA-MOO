import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


class NSGA2Wrapper:
    def __init__(self, budget, dim, bounds):
        """Initialize with the standard interface expected by MultiObjEvaluator."""
        self.budget = budget
        self.dim = dim
        self.bounds = bounds

    def __call__(self, func):
        """Run NSGA-II using the provided evaluation function.

        Args:
            func: A callable that takes x (shape: [dim]) and returns objectives (shape: [n_obj])
        """
        # Infer number of objectives from first evaluation
        test_x = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        test_y = func(test_x)
        n_obj = len(test_y)

        # Define pymoo Problem that wraps the provided func
        class WrappedProblem(Problem):
            def __init__(self, n_var, n_obj, bounds, eval_func):
                xl = np.full(n_var, bounds[0])
                xu = np.full(n_var, bounds[1])
                super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
                self.eval_func = eval_func

            def _evaluate(self, X, out, *args, **kwargs):
                # X is (pop_size, n_var)
                out["F"] = np.array([self.eval_func(x) for x in X])

        problem = WrappedProblem(
            n_var=self.dim, n_obj=n_obj, bounds=self.bounds, eval_func=func
        )

        # Run NSGA-II with appropriate population size
        pop_size = min(5, self.budget)
        res = minimize(
            problem,
            NSGA2(pop_size=pop_size),
            termination=("n_eval", self.budget),
            seed=None,
            verbose=False,
        )

        return res.X, res.F
