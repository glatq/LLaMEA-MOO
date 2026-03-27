import random
import logging
from typing import Any, Optional
import time
import os
import concurrent.futures
from tqdm import tqdm
import numpy as np
from misc import aoc_logger, correct_aoc

from llamevol.utils import BOOverBudgetException
from .ioh_objective_provider import IOHProvider
from .evaluator import AbstractEvaluator
from .evaluator_result import EvaluatorResult, EvaluatorBasicResult
from .exec_utils import default_exec, ExecInjector
from ..configspace_ext.configspace_utils import extract_configspace_from_response

try:
    from .smac_hpo_wrapper_so import (
        run_smac_hpo_so,
        SMACHPOConfig,
        validate_with_random_config,
    )

    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    logging.warning(
        "SMAC HPO not available. Install with: pip install smac ConfigSpace"
    )

_logger = logging.getLogger(__name__)


def compute_log_aoc(y_hist, budget, optimum_value=None, lower=1e-8, upper=1e4):
    if y_hist is None or len(y_hist) == 0:
        return 0.0
    y = y_hist.reshape(-1)
    best = np.minimum.accumulate(y)
    if optimum_value is not None:
        best = best - optimum_value
    best = best[:budget] if len(best) > budget else best
    best = np.clip(best, lower, upper)
    logbest = np.log10(best)
    lo, hi = np.log10(lower), np.log10(upper)
    a = np.clip((logbest - lo) / (hi - lo), 0.0, 1.0)
    return float(1.0 - np.sum(a) / budget)


class ObjectiveFn:
    def __init__(
        self,
        *,
        provider,
        problem_id,
        instance_id,
        exec_id,
        dim,
        budget,
        show_progress_bar=False,
    ):
        self._provider = provider
        self.problem_id = problem_id
        self.instance_id = instance_id
        self.exec_id = exec_id
        self.dim = dim
        self.budget = budget
        self.maximize = False

        self.obj = provider.get(problem_id, instance_id, dim)

        # COMPAT: alias for tests expecting .obj_fn and .obj_fn.state
        self.obj_fn = self.obj

        self.name = self.obj.name
        self.bounds = self.obj.bounds
        self.optimal_x = self.obj.optimum_x
        self.optimal_value = self.obj.optimum_y

        self.x_hist = None
        self.y_hist = None
        self.ioh_aoc = None

        self._progress_bar = None
        self.show_progress_bar = show_progress_bar
        self.ignore_over_budget = False

    def reset(self):
        # COMPAT: old reset closed progress and nulled obj_fn
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        self.obj = None
        self.obj_fn = None

    def stateless_call(self, x):
        # fresh problem; honor maximize flag on the returned value like before
        fresh = self._provider.get(self.problem_id, self.instance_id, self.dim)
        y = fresh(x)
        return -y if self.maximize else y

    @property
    def show_progress_bar(self):
        return self._show_progress_bar

    @show_progress_bar.setter
    def show_progress_bar(self, value: bool):
        self._show_progress_bar = bool(value)
        if self._show_progress_bar:
            self._progress_bar = tqdm(total=self.budget, desc=f"Evaluating {self.name}")
        else:
            if self._progress_bar is not None:
                self._progress_bar.close()
                self._progress_bar = None

    # COMPAT: expose attribute named progress_bar
    @property
    def progress_bar(self):
        return self._progress_bar

    def __call__(self, x):
        if np.isnan(x).any():
            raise ValueError(f"x({x}) contains nan values")

        # COMPAT: old guard triggers when prior evaluations > budget (not >=)
        if (
            self.obj is not None
            and self.budget is not None
            and not self.ignore_over_budget
            and getattr(self.obj, "evaluations", 0) > self.budget
        ):
            raise BOOverBudgetException("OverBudgetException", "Budget exceeded")

        # record x history
        self.x_hist = x if self.x_hist is None else np.vstack((self.x_hist, x))

        y = self.obj(x)

        # record y history; keep shape behavior (eventually flattens to (n,))
        y_arr = (
            np.array(y).reshape(-1, 1)
            if isinstance(y, list)
            else np.array([y]).reshape(-1, 1)
        )
        self.y_hist = y_arr if self.y_hist is None else np.append(self.y_hist, y_arr)

        # progress updates: +rows if batched
        if self._show_progress_bar and self._progress_bar is not None:
            step = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 1 else 1
            self._progress_bar.update(step)

        # apply maximize after getting y
        if self.maximize:
            y = -y

        return np.array(y).reshape(-1, 1) if isinstance(y, list) else y


def evaluate_block(
    provider,
    problem_id,
    instance_id,
    exec_id,
    dim,
    budget,
    code,
    cls_name,
    cls=None,
    cls_init_kwargs=None,
    cls_call_kwargs=None,
    ignore_over_budget: bool = False,
    ignore_capture: bool = True,
    injector=None,
):
    obj_fn = ObjectiveFn(
        provider=provider,
        problem_id=problem_id,
        instance_id=instance_id,
        exec_id=exec_id,
        dim=dim,
        budget=budget,
        show_progress_bar=False,
    )
    obj_fn.ignore_over_budget = ignore_over_budget

    # Try to attach the IOH AOC logger if this is the IOH provider
    l2 = None
    raw_ioh_problem = None
    try:
        from llamevol.evaluator.ioh_objective_provider import IOHProvider

        is_ioh = isinstance(provider, IOHProvider)
    except Exception:
        is_ioh = False
    if is_ioh:
        try:
            from ioh import logger as ioh_logger

            # IOHProvider stores the raw IOH problem as _p
            raw_ioh_problem = getattr(obj_fn.obj, "_p", None)
            if raw_ioh_problem is not None and hasattr(
                raw_ioh_problem, "attach_logger"
            ):
                # aoc_logger is already imported in this module
                l2 = aoc_logger(budget, upper=1e4, triggers=[ioh_logger.trigger.ALWAYS])
                raw_ioh_problem.attach_logger(l2)
        except Exception:
            l2 = None
            raw_ioh_problem = None

    start_time = time.perf_counter()

    init_kwargs = {"dim": dim, "budget": budget}
    if cls_init_kwargs is not None:
        init_kwargs.update(cls_init_kwargs)

    call_kwargs = {"func": obj_fn}
    if cls_call_kwargs is not None:
        call_kwargs.update(cls_call_kwargs)

    res, captured_output, err, new_injector = default_exec(
        code=code,
        cls_name=cls_name,
        cls=cls,
        init_kwargs=init_kwargs,
        call_kwargs=call_kwargs,
        injector=injector,
    )
    exec_time = time.perf_counter() - start_time

    # Prefer IOH-native AOC if we attached the IOH logger; otherwise fall back
    aoc_value = None
    if l2 is not None and raw_ioh_problem is not None:
        try:
            # correct_aoc is imported at module level and patched by tests
            aoc_value = correct_aoc(raw_ioh_problem, l2, budget)
        except Exception:
            aoc_value = None

    if aoc_value is None:
        # compute_log_aoc is the IOH-free fallback (also patched by tests)
        aoc_value = compute_log_aoc(
            y_hist=obj_fn.y_hist,
            budget=budget,
            optimum_value=obj_fn.optimal_value,
            lower=1e-8,
            upper=1e4,
        )

    obj_fn.ioh_aoc = aoc_value
    obj_fn.reset()

    if ignore_capture:
        captured_output = None

    return res, captured_output, err, exec_time, obj_fn, new_injector


def ioh_evaluate_block(**kwargs):
    return evaluate_block(**kwargs)


class IOHEvaluator(AbstractEvaluator):
    def __str__(self):
        return f"IOHEvaluator: {self._problem_name}_dim-{self.dim}_budget-{self.budget}_instances-{self.instances[0]}_repeat-{self.reapeat}"

    def __init__(
        self,
        dim: int = 5,
        budget: int = 40,
        problems: list[int] = None,
        instances: list[list[int]] = None,
        repeat: int = 1,
        use_hpo: bool = False,  # Enable HPO mode
        hpo_trials: int = 500,  # Number of SMAC trials
        hpo_min_budget: int = 50,  # Min budget for multi-fidelity
        hpo_max_budget: int = 200,  # Max budget for multi-fidelity
        hpo_walltime: int = 3600,  # HPO time limit (1 hour)
        hpo_validation_budget: int = 100,  # Budget for validation
        hpo_n_problems: int = None,  # Number of problems for HPO (None = all)
    ):
        super().__init__()
        if (
            problems is not None
            and instances is not None
            and len(problems) != len(instances)
        ):
            raise ValueError("The length of problems and instances should be the same")

        feasible_dim = [2, 3, 5, 10, 20, 40]
        if dim not in feasible_dim:
            raise ValueError(f"dim should be in {feasible_dim}")

        self.problems = None
        feasible_problems = list(range(1, 25))
        if problems is not None:
            for problem in problems:
                if problem not in feasible_problems:
                    raise ValueError("problem should be in range(1, 25)")
            self.problems = problems
        else:
            # https://numbbo.github.io/coco/testsuites/bbob
            separable_problems = list(range(1, 6))
            low_conditioning_problems = list(range(6, 10))
            high_conditioning_problems = list(range(10, 15))
            adequate_structure_problems = list(range(15, 20))
            weak_structure_problems = list(range(20, 25))
            group_problems = [
                separable_problems,
                low_conditioning_problems,
                high_conditioning_problems,
                adequate_structure_problems,
                weak_structure_problems,
            ]

            selected_problems = [random.choice(group) for group in group_problems]
            self.problems = random.sample(selected_problems, 1)

        feasible_instances = list(range(1, 15))
        self.instances = None
        if instances is not None:
            for p_instances in instances:
                for instance in p_instances:
                    if instance not in feasible_instances:
                        raise ValueError(f"instance should be in {feasible_instances}")
            self.instances = instances
        else:
            # self.instances = [random.sample(feasible_instances, 1)] * len(self.problems)
            self.instances = [[1]] * len(self.problems)

        self.reapeat = repeat
        self.dim = dim
        self.budget = budget

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
            _logger.warning("HPO requested but not available. Falling back to no HPO.")

        obj_fn_params = []
        for problem, instances in zip(self.problems, self.instances):
            for instance in instances:
                for i in range(self.reapeat):
                    params = {
                        "problem_id": problem,
                        "instance_id": instance,
                        "exec_id": i,
                        "dim": self.dim,
                        "budget": self.budget,
                    }
                    obj_fn_params.append(params)

        self.obj_fn_params = obj_fn_params

        problem_name = "_".join([f"f{problem}" for problem in self.problems])
        self._problem_name = problem_name

        self.timeout = 60 * 60  # 60 minutes
        self.provider = IOHProvider()

        if self.use_hpo:
            _logger.info("=" * 60)
            _logger.info("SMAC HPO ENABLED")
            _logger.info(f"  Trials: {hpo_trials}")
            _logger.info(f"  Budget range: {hpo_min_budget}-{hpo_max_budget}")
            _logger.info(f"  Walltime: {hpo_walltime}s")
            _logger.info("=" * 60)

    def eval_bugdet(self) -> int:
        return self.budget

    def problem_name(self) -> str:
        return self._problem_name

    def problem_prompt(self) -> str:
        prompt = f"Problems from the BBOB test suite with dimensions {self.dim}\n"
        return prompt

    def __process_results(
        self, res, captured_output, err, exec_time, obj_fn, injector
    ) -> EvaluatorBasicResult:
        eval_basic_result = EvaluatorBasicResult()
        if self.dim > 5:
            eval_basic_result.aoc_upper_bound = 1e9
        eval_basic_result.id = (
            f"{obj_fn.problem_id}-{obj_fn.instance_id}-{obj_fn.exec_id}"
        )
        eval_basic_result.budget = obj_fn.budget
        eval_basic_result.name = obj_fn.name
        eval_basic_result.bounds = obj_fn.bounds
        eval_basic_result.execution_time = exec_time
        eval_basic_result.set_capture_output(captured_output)

        if err is not None:
            eval_basic_result.error = str(err)
            eval_basic_result.error_type = getattr(
                err, "error_type", err.__class__.__name__
            )

        if eval_basic_result.error is None and self.return_checker is not None:
            # check the return value
            return_check_str = self.return_checker(res)
            if len(return_check_str) > 0:
                eval_basic_result.error = return_check_str
                eval_basic_result.error_type = "ReturnCheckError"

        if eval_basic_result.error is None:
            # best_y, best_x = res
            y_hist = (
                obj_fn.y_hist
                if len(obj_fn.y_hist) <= self.budget
                else obj_fn.y_hist[: self.budget]
            )
            x_hist = (
                obj_fn.x_hist
                if len(obj_fn.x_hist) <= self.budget
                else obj_fn.x_hist[: self.budget]
            )

            eval_basic_result.name = obj_fn.name
            eval_basic_result.bounds = obj_fn.bounds
            eval_basic_result.optimal_value = obj_fn.optimal_value
            eval_basic_result.optimal_x = obj_fn.optimal_x
            eval_basic_result.y_hist = (
                y_hist.reshape(-1) if len(y_hist.shape) > 1 else y_hist
            )
            eval_basic_result.x_hist = x_hist

            if injector is not None:
                if injector.critic is not None:
                    critic = injector.critic
                    eval_basic_result.n_initial_points = critic.n_init
                    eval_basic_result.r2_list = critic.r_2_list
                    eval_basic_result.r2_list_on_train = critic.r_2_list_on_train
                    eval_basic_result.uncertainty_list = critic.uncertainty_list
                    eval_basic_result.uncertainty_list_on_train = (
                        critic.uncertainty_list_on_train
                    )
                    eval_basic_result.search_result = critic.search_result
                if not injector.ignore_metric:
                    eval_basic_result.update_coverage()
                eval_basic_result.fill_short_data(obj_fn.budget)

            eval_basic_result.update_stats()
            eval_basic_result.update_aoc(optimal_value=obj_fn.optimal_value, min_y=1e-8)

            eval_basic_result.log_y_aoc_ioh = obj_fn.ioh_aoc

        return eval_basic_result

    def _check_timeout(self, start_time, timeout):
        if timeout is not None:
            _current_eval_time = time.perf_counter()
            _time_diff = _current_eval_time - start_time
            if _time_diff > timeout:
                return True
        return False

    def _post_process_error_check(
        self,
        eval_result: EvaluatorResult,
        eval_basic_result: EvaluatorBasicResult,
        timeout,
        start_time,
    ):
        _err = eval_basic_result.error
        _err_type = eval_basic_result.error_type
        if _err is None and self._check_timeout(start_time, timeout):
            _err = TimeoutError("Evaluation timed out (%d)", timeout)
            _err_type = "Timeout"
        if _err is not None:
            eval_result.error = _err
            eval_result.error_type = _err_type
        else:
            eval_result.result.append(eval_basic_result)

    def _logging_eval_process(self, eval_result, interval, total_tasks):
        cls_name = eval_result.name
        if eval_result.error is not None:
            _logger.error("Evaluating %s: %s", cls_name, eval_result.error_type)
        else:
            done_tasks = len(eval_result.result)
            if done_tasks % interval == 0:
                _logger.info("Evaluating %s: %s/%s", cls_name, done_tasks, total_tasks)

    def start_as_completed(
        self,
        eval_result,
        futures,
        timeout,
        task_manager=None,
        executor=None,
        cls_name=None,
        interval=None,
        total_tasks=None,
    ):
        _should_cancel = False
        _as_completed = None
        if task_manager is not None:
            _as_completed = task_manager.as_completed(futures.keys(), timeout=timeout)
        else:
            _as_completed = concurrent.futures.as_completed(
                futures.keys(), timeout=timeout
            )
        try:
            for future in _as_completed:
                res = future.result()
                eval_basic_result = self.__process_results(*res)

                _err = eval_basic_result.error
                _err_type = eval_basic_result.error_type
                if _err is not None:
                    eval_result.error = _err
                    eval_result.error_type = _err_type
                else:
                    eval_result.result.append(eval_basic_result)

                self._logging_eval_process(eval_result, interval, total_tasks)

                if eval_result.error is not None:
                    _should_cancel = True
                    break
        except TimeoutError:
            _err = TimeoutError("Evaluation timed out (%d)", timeout)
            _err_type = "Timeout"
            eval_result.error = _err
            eval_result.error_type = _err_type
            self._logging_eval_process(eval_result, interval, total_tasks)
            _should_cancel = True

        _logger.info("Evaluating %s: Shutting down executor", cls_name)
        # better to wait for all running tasks to finish in case of resource competition
        if task_manager is not None:
            task_manager.shutdown(wait=True, cancel_futures=_should_cancel)
        else:
            executor.shutdown(wait=True, cancel_futures=_should_cancel)
        _logger.info("Evaluating %s: Executor shut down", cls_name)

        return eval_result

    def evaluate(
        self,
        code,
        cls_name,
        cls=None,
        cls_init_kwargs: dict[str, Any] = None,
        cls_call_kwargs: dict[str, Any] = None,
        injector=None,
        llm_response: Optional[
            str
        ] = None,  # Full LLM response for ConfigSpace extraction
    ) -> EvaluatorResult:
        """Evaluate an individual.

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
        eval_result = EvaluatorResult()
        eval_result.name = cls_name
        if code is None and cls is None:
            eval_result.error = "No code generated"
            eval_result.error_type = "NoCodeGenerated"
            return eval_result

        # HPO Mode: Extract ConfigSpace and run SMAC
        incumbent_dict = {}
        if self.use_hpo and llm_response:
            _logger.info(f"HPO mode enabled for {cls_name}")

            # Extract ConfigSpace from LLM response
            configspace = extract_configspace_from_response(llm_response)

            if configspace is None or len(configspace) == 0:
                _logger.warning(
                    "No valid ConfigSpace found. Using default hyperparameters."
                )
                if not hasattr(eval_result, "metadata"):
                    eval_result.metadata = {}
                eval_result.metadata["hpo_error"] = "ConfigSpace not found or empty"
            else:
                _logger.info(
                    f"ConfigSpace found with {len(configspace)} hyperparameters: {list(configspace.keys())}"
                )

                # Step 1: Quick validation with random config
                validation_passed = validate_with_random_config(
                    code=code,
                    cls_name=cls_name,
                    configspace=configspace,
                    problem_id=self.problems[0],
                    instance_id=self.instances[0][0],
                    dim=self.dim,
                    budget=self.hpo_validation_budget,
                    injector=injector,
                )

                if not validation_passed:
                    _logger.error("Validation failed. Skipping HPO.")
                    if not hasattr(eval_result, "metadata"):
                        eval_result.metadata = {}
                    eval_result.metadata["hpo_error"] = "Validation failed"
                else:
                    # Step 2: Run SMAC HPO
                    try:
                        # Select subset of problems for HPO if configured
                        hpo_problems = self.problems
                        hpo_instances = self.instances
                        if self.hpo_n_problems and self.hpo_n_problems < len(
                            self.problems
                        ):
                            indices = [
                                int(i)
                                for i in np.linspace(
                                    0, len(self.problems) - 1, self.hpo_n_problems
                                )
                            ]
                            hpo_problems = [self.problems[i] for i in indices]
                            hpo_instances = [self.instances[i] for i in indices]
                            _logger.info(
                                f"HPO using {self.hpo_n_problems}/{len(self.problems)} problems: {hpo_problems}"
                            )

                        _logger.info("Starting SMAC HPO...")
                        incumbent_dict, incumbent_aoc = run_smac_hpo_so(
                            code=code,
                            cls_name=cls_name,
                            configspace=configspace,
                            problem_ids=hpo_problems,
                            instance_ids=hpo_instances,
                            dim=self.dim,
                            budget=self.budget,
                            hpo_config=self.hpo_config,
                            injector=injector,
                        )

                        _logger.info(
                            f"SMAC completed. Incumbent: {incumbent_dict}, AOC: {incumbent_aoc:.4f}"
                        )
                        if not hasattr(eval_result, "metadata"):
                            eval_result.metadata = {}
                        eval_result.metadata["incumbent"] = incumbent_dict
                        eval_result.metadata["incumbent_aoc"] = incumbent_aoc

                    except Exception as e:
                        _logger.error(f"HPO failed: {e}")
                        if not hasattr(eval_result, "metadata"):
                            eval_result.metadata = {}
                        eval_result.metadata["hpo_error"] = str(e)
                        incumbent_dict = {}

        # Merge incumbent with any additional init kwargs
        final_init_kwargs = dict(incumbent_dict) if incumbent_dict else {}
        if cls_init_kwargs:
            final_init_kwargs.update(cls_init_kwargs)

        # Step 3: Final evaluation with incumbent (or defaults)
        _logger.info(f"Running final evaluation with config: {final_init_kwargs}")

        # Continue with regular evaluation using final_init_kwargs
        cls_init_kwargs = final_init_kwargs if final_init_kwargs else cls_init_kwargs

        if self.gpu_name is not None and code is not None:
            code = ExecInjector.inject_code_with_device(code, self.gpu_name)

        params = []
        for param in self.obj_fn_params:
            new_param = {
                "provider": getattr(self, "provider", None),
                "code": code,
                "cls_name": cls_name,
                "cls": cls,
                "ignore_over_budget": self.ignore_over_budget,
                "cls_init_kwargs": cls_init_kwargs,
                "cls_call_kwargs": cls_call_kwargs,
                "injector": injector,
            }
            new_param.update(param)
            params.append(new_param)

        total_tasks = len(params)
        interval = min(max(1, total_tasks // 6), 20)

        _all_eval_time_start = time.perf_counter()

        max_eval_workers = self.max_eval_workers
        use_multi_process = self.use_multi_process
        timeout = self.timeout

        if max_eval_workers is None or max_eval_workers > 0:
            max_workers = min(os.cpu_count() - 1, max_eval_workers)
            if use_multi_process:
                _logger.info(
                    "Evaluating %s: %s tasks, using ProcessPoolExecutor with %s max_workers",
                    cls_name,
                    total_tasks,
                    max_workers,
                )
                executor_cls = concurrent.futures.ProcessPoolExecutor
            else:
                _logger.info(
                    "Evaluating %s: %s tasks, using ThreadPoolExecutor with %s max_workers",
                    cls_name,
                    total_tasks,
                    max_workers,
                )
                executor_cls = concurrent.futures.ThreadPoolExecutor

            executor = executor_cls(max_workers=max_workers)
            futures = {
                executor.submit(ioh_evaluate_block, **param): param for param in params
            }
            self.start_as_completed(
                eval_result,
                futures,
                timeout,
                executor=executor,
                cls_name=cls_name,
                interval=interval,
                total_tasks=total_tasks,
            )
        else:
            _logger.info("Evaluating %s: %s tasks in sequence", cls_name, total_tasks)

            for param in params:
                res = ioh_evaluate_block(**param)
                eval_basic_result = self.__process_results(*res)

                _err = eval_basic_result.error
                _err_type = eval_basic_result.error_type
                if _err is None and self._check_timeout(_all_eval_time_start, timeout):
                    _err = TimeoutError("Evaluation timed out (%d)", timeout)
                    _err_type = "Timeout"
                if _err is not None:
                    eval_result.error = _err
                    eval_result.error_type = _err_type
                else:
                    eval_result.result.append(eval_basic_result)

                self._logging_eval_process(eval_result, interval, total_tasks)
                if eval_result.error is not None:
                    break

        _all_eval_time = time.perf_counter() - _all_eval_time_start
        if eval_result.error is None:
            eval_result.score = np.mean([r.log_y_aoc for r in eval_result.result])
            eval_result.ioh_score = np.mean(
                [r.log_y_aoc_ioh for r in eval_result.result]
            )
            eval_result.total_execution_time = np.sum(
                [r.execution_time for r in eval_result.result]
            )
            _logger.info(
                "Evaluated %s: %.4f executed %.2fs in %.2fs",
                cls_name,
                eval_result.score,
                eval_result.total_execution_time,
                _all_eval_time,
            )
        else:
            _logger.error("Evaluated %s: Failed in %.2fs", cls_name, _all_eval_time)
            eval_result.score = 0.0
            eval_result.total_execution_time = 0.0

        return eval_result
