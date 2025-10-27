import time
import numpy as np
from dataclasses import dataclass

from .evaluator import AbstractEvaluator
from .evaluator_result import EvaluatorResult, EvaluatorBasicResult
from .exec_utils import default_exec  # same helper used in IOHEvaluator

# pymoo for test problems and hypervolume metric
from pymoo.factory import get_problem
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class MOOProblemSpec:
    """Definition of a multi-objective benchmark problem."""

    name: str = "zdt1"  # e.g. "zdt1", "dtlz2"
    dim: int = 10  # number of decision variables
    n_obj: int = 2  # number of objectives
    ref_point: list = None  # reference point for HV


class MultiObjEvaluator(AbstractEvaluator):
    """
    Evaluator for multi-objective algorithms.

    LLM-generated optimizer contract:
        class Optimizer:
            def __init__(self, budget, dim): ...
            def __call__(self, func):  # func(x) -> np.ndarray shape (n_obj,)
                ...
    """

    def __init__(
        self, budget: int, problem: MOOProblemSpec, repeat: int = 1, timeout: int = 1800
    ):
        super().__init__()  # AbstractEvaluator.__init__ takes no args

        self.budget = int(budget)
        self.problem_spec = problem
        self.repeat = int(repeat)
        # The base class defines .timeout; we set it for parity with IOHEvaluator
        self.timeout = int(timeout)

        # Build pymoo problem (minimization by default)
        self.problem = get_problem(problem.name, n_var=problem.dim, n_obj=problem.n_obj)

        # Reference point for HV (choose worse-than-front point)
        self.ref_point = (
            np.ones(problem.n_obj, dtype=float) * 1.2
            if problem.ref_point is None
            else np.asarray(problem.ref_point, dtype=float)
        )

        # Used by selection elsewhere: HV is larger-is-better
        self.maximize_fitness = True

    # ------------ helpers ------------

    def _wrap_func(self):
        """
        Wraps the pymoo problem with budget enforcement and logging.
        Returns (func, x_hist, y_hist) where y_hist collects vector objectives.
        """
        x_hist, y_hist = [], []
        remaining = {"n": self.budget}

        def func(x):
            if remaining["n"] <= 0:
                raise RuntimeError("Budget exceeded")
            xx = np.asarray(x, dtype=float).ravel()
            F = self.problem.evaluate(xx[None, :])
            if isinstance(F, dict):
                F = F.get("F")
            yy = np.asarray(F, dtype=float).reshape(-1, self.problem_spec.n_obj)[0]
            x_hist.append(xx)
            y_hist.append(yy)
            remaining["n"] -= 1
            return yy

        return func, x_hist, y_hist

    @staticmethod
    def _nd_indices(Y):
        if len(Y) == 0:
            return np.empty((0,), dtype=int)
        return np.asarray(
            NonDominatedSorting().do(Y, only_non_dominated_front=True), dtype=int
        )

    def _hypervolume(self, Y):
        if len(Y) == 0:
            return 0.0
        Y = np.asarray(Y, dtype=float)
        Y_nd = Y[self._nd_indices(Y)]
        return float(HV(ref_point=self.ref_point).do(Y_nd))

    # ------------ required API ------------

    def evaluate(self, code, class_name, **kwargs):
        """
        Evaluate an LLM-generated optimizer on a multi-objective problem.

        Matches your default_exec signature:
          default_exec(code, cls_name, init_kwargs=None, call_kwargs=None, cls=None, injector=None)
          -> (res, captured_output, err, injector)
        """
        t0 = time.time()
        ev_res = EvaluatorResult()
        ev_res.name = f"MOO-{self.problem_spec.name}"
        ev_res.error = None
        ev_res.error_type = None
        ev_res.total_execution_time = 0
        if ev_res.result is None:
            ev_res.result = []

        hv_runs = []

        for _ in range(self.repeat):
            run_t0 = time.time()
            basic = EvaluatorBasicResult()
            basic.name = ev_res.name
            basic.error = None
            basic.error_type = None
            basic.execution_time = 0
            basic.x_hist = None
            basic.y_hist = None  # keep None; we store vector history internally only
            basic.best_y = None  # we'll store HV as best_y for this run

            # Prepare budgeted func and histories
            func, x_hist, y_hist = self._wrap_func()

            # Run the optimizer via default_exec:
            #   __init__(budget, dim)     -> init_kwargs
            #   __call__(func=func)       -> call_kwargs
            # Note: we ignore the returned `res` here and use our histories.
            init_kwargs = {"budget": self.budget, "dim": self.problem_spec.dim}
            call_kwargs = {"func": func}

            res, captured_output, err, _ = default_exec(
                code=code,
                cls_name=class_name,
                init_kwargs=init_kwargs,
                call_kwargs=call_kwargs,
                cls=None,
                injector=None,
            )

            # Fill basic result
            basic.execution_time = time.time() - run_t0
            if err is not None:
                basic.error = str(err)
                basic.error_type = "ExecError"
                basic.x_hist = None
                basic.y_hist = None
                basic.best_y = np.nan
            else:
                X = (
                    np.asarray(x_hist, dtype=float)
                    if x_hist
                    else np.empty((0, self.problem_spec.dim))
                )
                Y = (
                    np.asarray(y_hist, dtype=float)
                    if y_hist
                    else np.empty((0, self.problem_spec.n_obj))
                )
                hv = self._hypervolume(Y)
                hv_runs.append(hv)

                basic.x_hist = X  # repo expects raw arrays
                basic.y_hist = None  # leave None to avoid SO-only logic downstream
                basic.best_y = hv

            ev_res.result.append(basic)

        # Aggregate scalar fitness used by selection
        ev_res.score = float(np.mean(hv_runs)) if hv_runs else -np.inf
        ev_res.total_execution_time = time.time() - t0
        return ev_res

    def problem_name(self):
        return f"MOO-{self.problem_spec.name}"
