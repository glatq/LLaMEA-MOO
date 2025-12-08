import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, List

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .evaluator import AbstractEvaluator
from .evaluator_result import EvaluatorResult, EvaluatorBasicResult
from .exec_utils import default_exec
from .PymooMOProvider import PymooMOProvider


@dataclass
class MOOProblemSpec:
    """Configuration for a single multi-objective benchmark problem."""

    name: str
    dim: int
    n_obj: int
    ref_point: Optional[Sequence[float]] = None


class MultiObjEvaluator(AbstractEvaluator):
    """Evaluate LLM-generated multi-objective optimizers on one or more problems.

    Optimizer contract for this evaluator:

        class Algo:
            def __init__(self, budget: int, dim: int):
                ...
            def __call__(self, func):
                # func(x) -> np.ndarray of shape (n_obj,)
                ...
    """

    def __init__(
        self,
        budget: int,
        problems: Optional[Sequence[MOOProblemSpec]] = None,
        repeat: int = 1,
        timeout: int = 1800,
    ):
        super().__init__()

        self.budget = int(budget)
        self.repeat = int(repeat)
        self.timeout = int(timeout)
        self.problem_specs: List[MOOProblemSpec] = list(problems)
        self._provider = PymooMOProvider()
        self.problem_spec: Optional[MOOProblemSpec] = None

    # ---------- helpers ----------

    def _wrap_func(self, wrapper, n_obj: int):
        """Return (func, x_hist, y_hist) with budget enforcement."""
        x_hist: List[np.ndarray] = []
        y_hist: List[np.ndarray] = []
        remaining = {"n": self.budget}

        def func(x):
            if remaining["n"] <= 0:
                raise RuntimeError("Budget exceeded")

            xx = np.asarray(x, dtype=float).ravel()
            F = wrapper(xx)  # wrapper(x) -> (n_obj,)
            yy = np.asarray(F, dtype=float).reshape(-1, n_obj)[0]

            x_hist.append(xx)
            y_hist.append(yy)
            remaining["n"] -= 1
            return yy

        return func, x_hist, y_hist

    # ---------- AbstractEvaluator API ----------

    def problem_name(self) -> str:
        if len(self.problem_specs) == 1:
            return f"MOO-{self.problem_specs[0].name}"
        names = ",".join(spec.name for spec in self.problem_specs)
        return f"MOO-suite({names})"

    def problem_prompt(self) -> str:
        if len(self.problem_specs) == 1:
            s = self.problem_specs[0]
            return (
                f"You are evaluated on the multi-objective problem '{s.name}' "
                f"of dimension {s.dim} with {s.n_obj} objectives. "
                "The goal is to minimize a scalar loss defined as minus the hypervolume "
                "of the non-dominated front obtained within a fixed evaluation budget."
            )
        else:
            names = ", ".join(sorted({spec.name for spec in self.problem_specs}))
            return (
                "You are evaluated on a suite of multi-objective benchmark problems: "
                f"{names}. Each problem has a limited evaluation budget; for each run "
                "we compute the hypervolume (HV) of your non-dominated front with "
                "respect to a fixed reference point. Your scalar fitness is the "
                "negative mean HV over all problems and repeats (lower is better)."
            )

    def evaluate(
        self,
        code,
        cls_name,
        cls: Any = None,
        cls_init_kwargs: Optional[dict[str, Any]] = None,
        cls_call_kwargs: Optional[dict[str, Any]] = None,
        injector=None,
    ) -> EvaluatorResult:
        """Evaluate an individual on all configured MOO problems."""
        eval_res = EvaluatorResult()
        eval_res.name = cls_name
        eval_res.error = None
        eval_res.error_type = None
        if eval_res.result is None:
            eval_res.result = []

        if code is None and cls is None:
            eval_res.error = "No code generated"
            eval_res.error_type = "NoCodeGenerated"
            return eval_res

        hv_losses: List[float] = []
        t0 = time.time()

        for spec in self.problem_specs:
            # Build problem via provider
            wrapper = self._provider.get(
                problem_id=spec.name,
                dim=spec.dim,
                ref_point=spec.ref_point,
                n_obj=spec.n_obj,
            )
            self.problem_spec = spec
            n_obj = wrapper.n_obj

            # Retrieve bounds from the wrapper (provided by PymooMOProvider)
            # Shape is (2, dim) -> [[lb...], [ub...]]
            bounds = wrapper.bounds

            # Reference point for HV
            if getattr(wrapper, "ref_point", None) is not None:
                ref_point = np.asarray(wrapper.ref_point, float).ravel()
            else:
                ref_point = np.ones(n_obj, dtype=float) * 1.2
                # ref_point = np.asarray([140.0,50.0]) #nice ref_point for BNH

            hv_indicator = HV(ref_point=ref_point)

            for rep in range(self.repeat):
                run_t0 = time.time()

                basic = EvaluatorBasicResult()
                basic.name = f"{spec.name}-rep{rep + 1}"
                basic.bounds = bounds
                basic.error = None
                basic.error_type = None
                basic.execution_time = 0.0
                basic.x_hist = None
                basic.y_hist = None
                basic.best_y = None
                basic.y_raw = None

                func, x_hist, y_hist = self._wrap_func(wrapper, n_obj=n_obj)

                init_kwargs = {"budget": self.budget, "dim": spec.dim, "bounds": bounds}
                if cls_init_kwargs:
                    init_kwargs.update(cls_init_kwargs)

                call_kwargs = {"func": func}
                if cls_call_kwargs:
                    call_kwargs.update(cls_call_kwargs)

                res, captured_output, err, injector = default_exec(
                    code=code,
                    cls_name=cls_name,
                    cls=cls,
                    init_kwargs=init_kwargs,
                    call_kwargs=call_kwargs,
                    injector=injector,
                )

                basic.execution_time = time.time() - run_t0

                if err is not None:
                    basic.error = str(err)
                    basic.error_type = "ExecError"
                    basic.x_hist = None
                    basic.y_hist = None
                    basic.best_y = float("nan")
                else:
                    # Convert histories to arrays
                    X = (
                        np.asarray(x_hist, dtype=float)
                        if x_hist
                        else np.empty((0, spec.dim), dtype=float)
                    )
                    Y = (
                        np.asarray(y_hist, dtype=float)
                        if y_hist
                        else np.empty((0, n_obj), dtype=float)
                    )

                    # Compute HV trace and scalar loss
                    if Y.size == 0:
                        hv_hist = []
                        hv_raw = 0.0
                    else:
                        nds = NonDominatedSorting()
                        hv_hist: List[float] = []

                        # HV after each evaluation (prefix of Y)
                        for i in range(1, Y.shape[0] + 1):
                            # Note: This loop can be slow for large budgets
                            front_idx = nds.do(Y[:i], only_non_dominated_front=True)
                            Y_nd = Y[:i][front_idx]
                            hv_hist.append(float(hv_indicator(Y_nd)))

                        hv_raw = hv_hist[-1]

                    hv_loss = -hv_raw  # scalar we MINIMIZE
                    hv_losses.append(hv_loss)

                    # best_x = x at maximum HV
                    if hv_hist:
                        best_idx = int(np.argmax(hv_hist))
                        best_x = X[best_idx]
                        best_hv = hv_hist[best_idx]
                    else:
                        best_x = None
                        best_hv = 0.0

                    basic.x_hist = X
                    basic.raw_y_hist = Y
                    basic.y_hist = np.asarray(hv_hist, dtype=float)
                    basic.best_y = hv_loss
                    basic.best_x = best_x
                    basic.optimal_value = best_hv

            eval_res.result.append(basic)

        eval_res.score = float(np.mean(hv_losses)) if hv_losses else float("nan")
        eval_res.total_execution_time = time.time() - t0
        return eval_res
