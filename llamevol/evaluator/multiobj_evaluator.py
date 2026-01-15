import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, List
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import concurrent.futures
import multiprocessing
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


def _run_single_moo_rep(
    spec,
    rep,
    budget,
    code,
    cls_name,
    cls,
    cls_init_kwargs,
    cls_call_kwargs,
    injector,
    stop_event=None,
):
    with threadpool_limits(limits=1, user_api="blas"):
        provider = PymooMOProvider()
        wrapper = provider.get(
            problem_id=spec.name,
            dim=spec.dim,
            ref_point=spec.ref_point,
            n_obj=spec.n_obj,
        )

        run_t0 = time.time()
        basic = EvaluatorBasicResult()
        basic.name = f"{spec.name}-rep{rep + 1}"

        x_hist, y_hist = [], []

        pbar = tqdm(total=budget, desc=f"Run {spec.name}", leave=False)

        def func(x):
            if stop_event and stop_event.is_set():
                raise StopIteration("Timeout requested by parent")
            if len(x_hist) >= budget:
                return np.zeros(wrapper.n_obj)  # Safety
            yy = np.asarray(
                wrapper(np.asarray(x, dtype=float).ravel()), dtype=float
            ).reshape(-1, wrapper.n_obj)[0]
            x_hist.append(np.asarray(x).ravel())
            y_hist.append(yy)

            pbar.update(1)
            return yy

        init_kwargs = {"budget": budget, "dim": spec.dim, "bounds": wrapper.bounds}
        if cls_init_kwargs:
            init_kwargs.update(cls_init_kwargs)

        res, _, err, _ = default_exec(
            code=code,
            cls_name=cls_name,
            cls=cls,
            init_kwargs=init_kwargs,
            call_kwargs={"func": func},
            injector=injector,
        )

        pbar.close()

        basic.execution_time = time.time() - run_t0
        if err:
            basic.error, basic.error_type = str(err), "ExecError"
        else:
            Y = np.asarray(y_hist)
            X = np.asarray(x_hist)
            if Y.size > 0:
                ref_point = (
                    np.asarray(wrapper.ref_point)
                    if getattr(wrapper, "ref_point", None) is not None
                    else np.ones(wrapper.n_obj) * 1.2
                )
                hv_indicator = HV(ref_point=ref_point)
                nds = NonDominatedSorting()
                # Get final HV
                front_idx = nds.do(Y, only_non_dominated_front=True)
                hv_raw = float(hv_indicator(Y[front_idx]))
                basic.best_y = -hv_raw
                basic.x_hist, basic.raw_y_hist = X, Y
    return basic


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
        cls=None,
        cls_init_kwargs=None,
        cls_call_kwargs=None,
        injector=None,
    ) -> EvaluatorResult:
        manager = multiprocessing.Manager()
        stop_event = manager.Event()
        eval_res = EvaluatorResult()
        eval_res.name = cls_name
        eval_res.result = []

        t0 = time.time()
        tasks = []
        for spec in self.problem_specs:
            for rep in range(self.repeat):
                tasks.append((spec, rep))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            future_to_task = {
                executor.submit(
                    _run_single_moo_rep,
                    spec,
                    rep,
                    self.budget,
                    code,
                    cls_name,
                    cls,
                    cls_init_kwargs,
                    cls_call_kwargs,
                    injector,
                    stop_event,
                ): (spec, rep)
                for spec, rep in tasks
            }

            try:
                for future in concurrent.futures.as_completed(
                    future_to_task, timeout=self.timeout
                ):
                    eval_res.result.append(future.result())
            except concurrent.futures.TimeoutError:
                eval_res.error, eval_res.error_type = (
                    f"Global Timeout ({self.timeout}s)",
                    "TimeoutError",
                )
                stop_event.set()
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                eval_res.error, eval_res.error_type = str(e), "ExecError"
                executor.shutdown(wait=False, cancel_futures=True)

        # Final Score: Only calculate if we got all results (Batch Success)
        valid_scores = [r.best_y for r in eval_res.result if r.best_y is not None]

        if not eval_res.error and len(valid_scores) == len(tasks):
            eval_res.score = float(np.mean(valid_scores))
        else:
            eval_res.score = float("nan")
        eval_res.total_execution_time = time.time() - t0
        return eval_res
