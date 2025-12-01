import time
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from pymoo.factory import get_problem
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from llamevol.utils import BOOverBudgetException

from .evaluator import AbstractEvaluator
from .evaluator_result import EvaluatorResult, EvaluatorBasicResult
from .exec_utils import default_exec


@dataclass
class MOOProblemSpec:
    """Static description of a multi-objective test problem."""

    name: str = "zdt1"  # e.g. "zdt1", "dtlz2"
    dim: int = 10  # number of decision variables
    n_obj: int = 2  # number of objectives
    ref_point: Optional[np.ndarray] = None  # reference point for HV (minimization)


class MultiObjEvaluator(AbstractEvaluator):
    """
    Evaluator for multi-objective algorithms.

    Contract for LLM-generated optimizer:

        class Optimizer:
            def __init__(self, budget: int, dim: int):
                ...

            def __call__(self, func):
                # func(x) -> np.ndarray with shape (n_obj,)
                ...
    """

    def __init__(
        self,
        budget: int,
        problem: MOOProblemSpec,
        repeat: int = 1,
        timeout: int = 1800,
    ):
        super().__init__()

        self.budget = int(budget)
        self.problem_spec = problem
        self.repeat = int(repeat)
        self.timeout = int(timeout)

        # Build pymoo problem (minimization by default)
        self.problem = get_problem(problem.name, n_var=problem.dim, n_obj=problem.n_obj)

        # Reference point for HV; if none is provided, use a simple worst-case point.
        if problem.ref_point is None:
            self.ref_point = np.ones(problem.n_obj, dtype=float) * 1.2
        else:
            self.ref_point = np.asarray(problem.ref_point, dtype=float).ravel()
            if self.ref_point.size != problem.n_obj:
                raise ValueError(
                    f"ref_point must have length {problem.n_obj}, "
                    f"got {self.ref_point.size}"
                )

    def _wrap_func(self):
        """
        Wrap the pymoo problem with budget enforcement and logging.

        Returns a tuple (func, x_hist, y_hist) where y_hist stores vector objectives.
        """
        x_hist: list[np.ndarray] = []
        y_hist: list[np.ndarray] = []
        remaining = {"n": self.budget}

        def func(x: np.ndarray) -> np.ndarray:
            if remaining["n"] <= 0:
                # Use the same exception type as IOHEvaluator so higher-level
                # logic can choose to ignore over-budget errors if desired.
                raise BOOverBudgetException("OverBudgetException", "Budget exceeded")

            xx = np.asarray(x, dtype=float).ravel()
            if xx.size != self.problem_spec.dim:
                raise ValueError(
                    f"Expected x.size == dim == {self.problem_spec.dim}, "
                    f"got {xx.size}"
                )

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
    def _nd_indices(Y: np.ndarray) -> np.ndarray:
        if len(Y) == 0:
            return np.empty((0,), dtype=int)
        nd = NonDominatedSorting().do(Y, only_non_dominated_front=True)
        return np.asarray(nd, dtype=int)

    def _hypervolume(self, Y: np.ndarray) -> float:
        if len(Y) == 0:
            return 0.0
        Y = np.asarray(Y, dtype=float)
        Y_nd = Y[self._nd_indices(Y)]
        hv = HV(ref_point=self.ref_point)
        return float(hv.do(Y_nd))

    def problem_prompt(self) -> str:
        return (
            f"Multi-objective problems from pymoo with name {self.problem_spec.name}, "
            f"dimension {self.problem_spec.dim}, {self.problem_spec.n_obj} objectives "
            f"and a budget of {self.budget} evaluations. The metric is Hypervolume."
        )

    def problem_name(self) -> str:
        return f"MOO-{self.problem_spec.name}"

    def evaluate(
        self,
        code: str,
        cls_name: str,
        cls: Any = None,
        cls_init_kwargs: Optional[dict[str, Any]] = None,
        cls_call_kwargs: Optional[dict[str, Any]] = None,
        injector: Any = None,
    ) -> EvaluatorResult:
        """
        Evaluate a candidate multi-objective algorithm.

        The algorithm is instantiated as Algo(budget=self.budget, dim=self.problem_spec.dim)
        and called with Algo(func) where func(x) returns an n_obj-dimensional vector.
        """
        if cls_init_kwargs is None:
            cls_init_kwargs = {}
        if cls_call_kwargs is None:
            cls_call_kwargs = {}

        t0 = time.time()

        ev_res = EvaluatorResult()
        ev_res.name = self.problem_name()
        ev_res.error = None
        ev_res.error_type = None
        ev_res.total_execution_time = 0.0
        if ev_res.result is None:
            ev_res.result = []

        hv_runs: list[float] = []

        for _ in range(self.repeat):
            run_t0 = time.time()
            basic = EvaluatorBasicResult()
            basic.name = ev_res.name
            basic.error = None
            basic.error_type = None
            basic.execution_time = 0.0
            basic.x_hist = None
            basic.y_hist = None  # keep None to avoid SO-specific downstream logic
            basic.best_y = None  # we will store HV here once computed

            func, x_hist, y_hist = self._wrap_func()

            init_kwargs = {"budget": self.budget, "dim": self.problem_spec.dim}
            init_kwargs.update(cls_init_kwargs)

            call_kwargs = {"func": func}
            call_kwargs.update(cls_call_kwargs)

            try:
                res, captured_output, err, _ = default_exec(
                    code=code,
                    cls_name=cls_name,
                    cls=cls,
                    init_kwargs=init_kwargs,
                    call_kwargs=call_kwargs,
                    injector=injector,
                )
            except BOOverBudgetException:
                # In case the exception bubbles up instead of being handled
                # inside default_exec, treat it as a normal termination.
                res = None
                captured_output = ""
                err = None

            basic.execution_time = time.time() - run_t0

            if err is not None:
                basic.error = str(err)
                basic.error_type = getattr(err, "error_type", "ExecError")
                basic.x_hist = None
                basic.y_hist = None
                basic.best_y = np.nan
            else:
                X = (
                    np.asarray(x_hist, dtype=float)
                    if x_hist
                    else np.empty((0, self.problem_spec.dim), dtype=float)
                )
                Y = (
                    np.asarray(y_hist, dtype=float)
                    if y_hist
                    else np.empty((0, self.problem_spec.n_obj), dtype=float)
                )

                hv = self._hypervolume(Y)
                hv_runs.append(hv)

                basic.x_hist = X
                basic.y_hist = None  # keep multi-objective trace internal for now
                basic.best_y = hv  # store HV as the run's scalar quality

            ev_res.result.append(basic)

        # Aggregate scalar fitness used by selection.
        # We convert HV (larger is better) into a minimization score
        # for compatibility with the single-objective AOC pipeline.
        if hv_runs:
            ev_res.score = float(-np.mean(hv_runs))  # minimize -HV
        else:
            ev_res.score = float(np.inf)

        ev_res.total_execution_time = time.time() - t0
        return ev_res
