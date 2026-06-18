import sys
import numpy as np
import os

# Add IOC-SAMO-COBRA to Python path
# Note: __file__ is not available when code is exec'd, so use os.getcwd()
# This assumes the wrapper is in LLaMEA-BO directory and IOC-SAMO-COBRA is a sibling
current_dir = os.getcwd()
ioc_path = os.path.join(os.path.dirname(current_dir), "IOC-SAMO-COBRA")

# Fallback: try common locations
if not os.path.exists(ioc_path):
    # Maybe we're already in LLaMEA-BO and repos is parent
    ioc_path = os.path.join(os.path.dirname(current_dir), "IOC-SAMO-COBRA")
    if not os.path.exists(ioc_path):
        # Try as sibling to current directory
        ioc_path = os.path.join(current_dir, "../..", "IOC-SAMO-COBRA")
        ioc_path = os.path.abspath(ioc_path)

if str(ioc_path) not in sys.path:
    sys.path.insert(0, str(ioc_path))

# No need to mock pygmo anymore - we replaced hypervolume.py with pymoo version
from cheap_SAMO_COBRA_Init import cheap_SAMO_COBRA_Init
from cheap_SAMO_COBRA_PhaseII import cheap_SAMO_COBRA_PhaseII


class IOCSAMOCOBRAWrapper:
    def __init__(self, budget, dim, bounds):
        """Initialize with the standard interface expected by MultiObjEvaluator."""
        self.budget = budget
        self.dim = dim
        self.bounds = bounds
        self.eval_count = 0  # Track evaluations for budget control only

    def __call__(self, func):
        """Run IOC-SAMO-COBRA using the provided evaluation function.

        Args:
            func: A callable that takes x (shape: [dim]) and returns objectives (shape: [n_obj])
        """
        print(
            f"\n[IOC-SAMO-COBRA] Starting optimization with budget={self.budget}, dim={self.dim}"
        )

        # Infer #objectives and, for constrained problems, #constraints from the
        # first evaluation. Constrained problems return (F, G) with G <= 0
        # feasible -- the same convention SAMO-COBRA uses internally
        # (paretofrontFeasible: feasible = sum(G <= 0) == n_constraints), so G is
        # passed through unchanged. (np.asarray of a ragged (F, G) tuple raises
        # on numpy>=1.24, which is why the old objectives-only probe crashed
        # whenever n_obj != n_constr.)
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
        print(f"[IOC-SAMO-COBRA] Detected {n_obj} objectives, {n_constr} constraints")

        # Reset tracking (used only for budget control, not for storing history)
        self.eval_count = 1  # Count the test evaluation

        # Create problem class compatible with IOC-SAMO-COBRA
        class WrappedProblem:
            def __init__(self, dim, n_obj, n_constr, bounds, eval_func, wrapper_ref):
                self.lower = np.full(dim, bounds[0])
                self.upper = np.full(dim, bounds[1])
                self.nConstraints = n_constr
                self.nObj = n_obj
                # Set a reasonable reference point (will be overridden by evaluator)
                self.ref = np.ones(n_obj) * 100.0
                self.nadir = None
                self.cheapConstr = [False] * n_constr  # constraints are expensive
                self.cheapObj = [False] * n_obj  # All objectives are expensive
                self.eval_func = eval_func
                self.wrapper = wrapper_ref
                self._constrained = n_constr > 0

            def evaluate(self, x):
                """Expensive evaluation method. Returns [objectives, constraints]
                with constraints in the G <= 0 feasible convention."""
                if self.wrapper.eval_count >= self.wrapper.budget:
                    # Budget exhausted, return dummy value (won't be used by evaluator)
                    return [np.zeros(self.nObj), np.zeros(self.nConstraints)]

                # Ensure x is a 1D numpy array
                x_eval = np.asarray(x).ravel()
                y = self.eval_func(x_eval)  # This call is tracked by evaluator
                if self._constrained:
                    f, g = y
                    f_array = np.asarray(f, dtype=float).ravel()
                    g_array = np.asarray(g, dtype=float).ravel()
                else:
                    f_array = np.asarray(y, dtype=float).ravel()
                    g_array = np.array([])

                self.wrapper.eval_count += 1
                return [f_array, g_array]

            def cheap_evaluate(self, x):
                """Cheap evaluation for candidate generation."""
                # Since all objectives/constraints are expensive, return NaNs.
                return [
                    np.full(self.nObj, np.nan),
                    np.full(self.nConstraints, np.nan),
                ]

        problem = WrappedProblem(
            dim=self.dim,
            n_obj=n_obj,
            n_constr=n_constr,
            bounds=self.bounds,
            eval_func=func,
            wrapper_ref=self,
        )

        # Configure IOC-SAMO-COBRA parameters
        batch_size = 5
        init_points = 20
        max_evals = self.budget

        try:
            # Generate a random seed for each run (not fixed!)
            import time
            import random

            random_seed = int((time.time() * 1000 + random.random() * 1000)) % 100000

            print(
                f"[IOC-SAMO-COBRA] Initializing with init_points={init_points}, max_evals={max_evals}, seed={random_seed}"
            )
            # Initialize with DoE
            cobra = cheap_SAMO_COBRA_Init(
                problem=problem,
                batch=batch_size,
                nCores=1,  # Single core to avoid multiprocessing issues
                computeStartingPoints=min(16, self.budget),
                cobraSeed=random_seed,  # Use random seed for each run
                iterPlot=False,
                feval=max_evals,
                initDesPoints=init_points,
            )

            print(f"[IOC-SAMO-COBRA] Init phase complete, starting Phase II")
            # Run optimization
            cobra = cheap_SAMO_COBRA_PhaseII(cobra)
            print(f"[IOC-SAMO-COBRA] Optimization completed successfully!")

        except Exception as e:
            import traceback

            print(f"\n{'='*60}")
            print(f"IOC-SAMO-COBRA ERROR: {e}")
            print(f"{'='*60}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            # If algorithm fails, fill remaining budget with random evaluations
            if self.eval_count < self.budget:
                print(f"Using fallback: generating random solutions to reach budget")
                while self.eval_count < self.budget:
                    x = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                    y = func(x)  # Evaluator tracks this
                    self.eval_count += 1
            else:
                print(f"Partial results available: {self.eval_count} evaluations")

        # Note: We don't need to return anything meaningful - the evaluator
        # tracks all func() calls in its own y_hist. Return value is ignored.
        return None, None
