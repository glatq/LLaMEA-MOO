import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, List, Dict
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
from ..configspace_ext.configspace_utils import extract_configspace_from_response

try:
    from .smac_hpo_wrapper import (
        run_smac_hpo_moo,
        SMACHPOConfig,
        validate_with_random_config,
    )

    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    logging.warning(
        "SMAC HPO not available. Install with: pip install smac ConfigSpace"
    )


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
    calculate_hv_history=False,  # New Flag
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
            if len(y_hist) >= budget:
                return np.zeros(wrapper.n_obj)

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
            # FIX: Ensure history is saved
            basic.raw_y_hist = np.asarray(y_hist)
            basic.x_hist = np.asarray(x_hist)

            Y = basic.raw_y_hist
            if Y.size > 0:
                ref_point = (
                    np.asarray(wrapper.ref_point)
                    if getattr(wrapper, "ref_point", None) is not None
                    else np.ones(wrapper.n_obj) * 1.2
                )
                hv_indicator = HV(ref_point=ref_point)
                nds = NonDominatedSorting()

                # Final performance metric
                front_idx = nds.do(Y, only_non_dominated_front=True)
                final_hv = float(hv_indicator(Y[front_idx]))
                basic.best_y = -final_hv

                # New: Calculate HV progress curve only if requested
                if calculate_hv_history:
                    hv_curve = []
                    for i in range(1, len(Y) + 1):
                        Y_sub = Y[:i]
                        f_idx = nds.do(Y_sub, only_non_dominated_front=True)
                        hv_curve.append(float(hv_indicator(Y_sub[f_idx])))
                    basic.hv_hist = np.asarray(hv_curve)

    return basic


class MultiObjEvaluator(AbstractEvaluator):
    def __init__(
        self,
        budget: int,
        problems: Optional[Sequence[MOOProblemSpec]] = None,
        repeat: int = 1,
        timeout: int = 1800,
        calculate_hv_history: bool = False,
        use_hpo: bool = False,  # Enable HPO mode
        hpo_trials: int = 500,  # Number of SMAC trials
        hpo_min_budget: int = 50,  # Min budget for multi-fidelity
        hpo_max_budget: int = 200,  # Max budget for multi-fidelity
        hpo_walltime: int = 3600,  # HPO time limit (1 hour)
        hpo_validation_budget: int = 100,  # Budget for validation
        hpo_n_problems: int = None,  # Number of problems for HPO (None = all)
    ):
        super().__init__()
        self.budget = int(budget)
        self.repeat = int(repeat)
        self.timeout = int(timeout)
        self.problem_specs = list(problems) if problems else []
        self.calculate_hv_history = calculate_hv_history

        # HPO configuration
        self.use_hpo = use_hpo and HPO_AVAILABLE
        self.hpo_config = (
            SMACHPOConfig(
                n_trials=hpo_trials,
                min_budget=hpo_min_budget,
                max_budget=hpo_max_budget,
                walltime_limit=hpo_walltime,
            )
            if self.use_hpo
            else None
        )
        self.hpo_validation_budget = hpo_validation_budget
        self.hpo_n_problems = hpo_n_problems

        if use_hpo and not HPO_AVAILABLE:
            logging.warning("HPO requested but not available. Falling back to no HPO.")

    def evaluate(
        self,
        code,
        cls_name,
        cls=None,
        cls_init_kwargs=None,
        cls_call_kwargs=None,
        injector=None,
        llm_response: Optional[
            str
        ] = None,  # Full LLM response for ConfigSpace extraction
    ) -> EvaluatorResult:
        """
        Evaluate algorithm code on multi-objective problems.

        If use_hpo=True and a ConfigSpace is found:
        1. Validate code with random config
        2. Run SMAC HPO to find best hyperparameters
        3. Evaluate with incumbent configuration

        Args:
            code: Algorithm code
            cls_name: Algorithm class name
            cls: Pre-compiled class (optional)
            cls_init_kwargs: Additional init kwargs to merge with incumbent
            cls_call_kwargs: Call kwargs
            injector: Code injector
            llm_response: Full LLM response (for ConfigSpace extraction)

        Returns:
            EvaluatorResult with incumbent stored in metadata
        """
        # Initialize result
        eval_res = EvaluatorResult()
        eval_res.name = cls_name
        eval_res.result = []
        eval_res.metadata = {}

        t0 = time.time()

        # HPO Mode: Extract ConfigSpace and run SMAC
        incumbent_dict = {}
        if self.use_hpo and llm_response:
            logging.info(f"HPO mode enabled for {cls_name}")

            # Extract ConfigSpace from LLM response
            configspace = extract_configspace_from_response(llm_response)

            if configspace is None or len(configspace) == 0:
                logging.warning(
                    "No valid ConfigSpace found. Using default hyperparameters."
                )
                eval_res.metadata["hpo_error"] = "ConfigSpace not found or empty"
            else:
                logging.info(
                    f"ConfigSpace found with {len(configspace)} hyperparameters: {list(configspace.keys())}"
                )

                # Step 1: Quick validation with random config
                if self.problem_specs:
                    validation_passed, validation_error = validate_with_random_config(
                        code=code,
                        cls_name=cls_name,
                        configspace=configspace,
                        problem_spec=self.problem_specs[0],
                        budget=self.hpo_validation_budget,
                        injector=injector,
                    )

                    if not validation_passed:
                        logging.error(
                            f"Validation failed: {validation_error}. Skipping HPO."
                        )
                        eval_res.metadata[
                            "hpo_error"
                        ] = f"Validation failed: {validation_error}"
                    else:
                        # Step 2: Run SMAC HPO
                        try:
                            # Select subset of problems for HPO if configured
                            hpo_specs = self.problem_specs
                            if self.hpo_n_problems and self.hpo_n_problems < len(
                                self.problem_specs
                            ):
                                indices = [
                                    int(i)
                                    for i in np.linspace(
                                        0,
                                        len(self.problem_specs) - 1,
                                        self.hpo_n_problems,
                                    )
                                ]
                                hpo_specs = [self.problem_specs[i] for i in indices]
                                logging.info(
                                    f"HPO using {self.hpo_n_problems}/{len(self.problem_specs)} problems: {[s.name for s in hpo_specs]}"
                                )

                            logging.info("Starting SMAC HPO...")
                            incumbent_dict, incumbent_hv = run_smac_hpo_moo(
                                code=code,
                                cls_name=cls_name,
                                configspace=configspace,
                                problem_specs=hpo_specs,
                                budget=self.budget,
                                hpo_config=self.hpo_config,
                                injector=injector,
                            )

                            # Restore root logger after SMAC
                            logging.getLogger().setLevel(logging.INFO)
                            logging.info(
                                f"SMAC completed. Incumbent: {incumbent_dict}, HV: {incumbent_hv:.4f}"
                            )
                            eval_res.metadata["incumbent"] = incumbent_dict
                            eval_res.metadata["incumbent_hv"] = incumbent_hv

                        except Exception as e:
                            logging.getLogger().setLevel(logging.INFO)
                            logging.error(f"HPO failed: {e}")
                            eval_res.metadata["hpo_error"] = str(e)
                            incumbent_dict = {}

        # Merge incumbent with any additional init kwargs
        final_init_kwargs = dict(incumbent_dict) if incumbent_dict else {}
        if cls_init_kwargs:
            final_init_kwargs.update(cls_init_kwargs)

        # Step 3: Final evaluation with incumbent (or defaults)
        logging.info(f"Running final evaluation with config: {final_init_kwargs}")

        manager = multiprocessing.Manager()
        stop_event = manager.Event()

        tasks = [
            (spec, rep) for spec in self.problem_specs for rep in range(self.repeat)
        ]

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
                    final_init_kwargs,  # Use incumbent config
                    cls_call_kwargs,
                    injector,
                    stop_event,
                    self.calculate_hv_history,  # Pass flag to worker
                ): (spec, rep)
                for spec, rep in tasks
            }

            try:
                for future in concurrent.futures.as_completed(
                    future_to_task, timeout=self.timeout
                ):
                    eval_res.result.append(future.result())
            except concurrent.futures.TimeoutError:
                stop_event.set()
                eval_res.error, eval_res.error_type = (
                    f"Timeout ({self.timeout}s)",
                    "TimeoutError",
                )
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                eval_res.error, eval_res.error_type = str(e), "ExecError"
                executor.shutdown(wait=False, cancel_futures=True)

        valid_scores = [r.best_y for r in eval_res.result if r.best_y is not None]
        eval_res.score = float(np.mean(valid_scores)) if valid_scores else float("nan")
        eval_res.total_execution_time = time.time() - t0
        return eval_res

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
