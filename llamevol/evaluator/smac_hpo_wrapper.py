"""SMAC-based hyperparameter optimization wrapper for multi-objective algorithms."""

import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
_logger.addHandler(_handler)
import time
import gc
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from threadpoolctl import threadpool_limits

from .PymooMOProvider import PymooMOProvider
from .exec_utils import default_exec
from .moo_metrics import score_moo_run

try:
    from ConfigSpace import ConfigurationSpace, Configuration
    from smac import AlgorithmConfigurationFacade, Scenario

    SMAC_AVAILABLE = True
except ImportError:
    ConfigurationSpace = None
    Configuration = None
    AlgorithmConfigurationFacade = None
    Scenario = None
    SMAC_AVAILABLE = False
    _logger.warning("SMAC or ConfigSpace not installed. HPO will not be available.")


@dataclass
class SMACHPOConfig:
    """Configuration for SMAC HPO."""

    n_trials: int = 500
    min_budget: int = 50
    max_budget: int = 200
    walltime_limit: int = 3600  # 1 hour (total budget, checked BETWEEN trials)
    # Per-trial wall-clock cap (seconds). SMAC's walltime_limit never interrupts
    # a running trial, so a single pathological config can run for hours. This
    # caps EACH evaluation via pynisher: an over-running trial is killed and
    # recorded as a crash, so SMAC learns to avoid slow configs. None = no cap
    # (legacy behaviour).
    trial_walltime_limit: Optional[float] = None
    n_workers: int = 1
    deterministic: bool = False
    # Weight on the infeasibility-rate penalty added to the constrained SMAC
    # objective: 1 - feasible_HV + infeasibility_penalty * (1 - feasibility_rate).
    # Ignored for unconstrained problems.
    infeasibility_penalty: float = 0.1


def _moo_objective(Y, ref_point, G=None, infeasibility_penalty: float = 0.0) -> float:
    """SMAC cost (lower is better) for one MO run.

    Unconstrained (G is None / empty): ``1 - normalized_HV`` -- identical to the
    legacy objective. Constrained: ``1 - normalized_feasible_HV +
    infeasibility_penalty * (1 - feasibility_rate)``, so SMAC is rewarded both
    for feasible hypervolume and for raising the feasible fraction.

    Feasible-HV is computed with ``moo_metrics.score_moo_run`` so the constrained
    SMAC cost is consistent with the evaluator's feasible-HV fitness.
    """
    has_constraints = G is not None and np.asarray(G).size > 0
    result = score_moo_run(Y, ref_point, G=(G if has_constraints else None))
    if has_constraints:
        return float(
            1.0 - result.score + infeasibility_penalty * (1.0 - result.feasibility_rate)
        )
    # Unconstrained: preserve the legacy guard (degenerate HV -> worst cost 1.0).
    return float(1.0 - result.score if result.score > 0 else 1.0)


def run_smac_hpo_moo(
    code: str,
    cls_name: str,
    configspace: ConfigurationSpace,
    problem_specs: List[Any],
    budget: int,
    hpo_config: Optional[SMACHPOConfig] = None,
    injector: Any = None,
) -> Tuple[Dict[str, Any], float]:
    """Run SMAC HPO for multi-objective optimization algorithm.

    Args:
        code: Python code containing the algorithm class
        cls_name: Name of the algorithm class
        configspace: ConfigurationSpace defining hyperparameter search space
        problem_specs: List of MOOProblemSpec objects to optimize on
        budget: Evaluation budget per problem instance
        hpo_config: SMAC configuration (uses defaults if None)
        injector: Optional code injector for modifying algorithm behavior

    Returns:
        Tuple of (incumbent_dict, incumbent_hv):
            - incumbent_dict: Best hyperparameter configuration found
            - incumbent_hv: Best average hypervolume achieved

    Raises:
        RuntimeError: If SMAC is not available
    """
    if not SMAC_AVAILABLE:
        raise RuntimeError(
            "SMAC is not installed. Install with: pip install smac ConfigSpace"
        )

    if hpo_config is None:
        hpo_config = SMACHPOConfig()

    _logger.info(f"Starting SMAC HPO for {cls_name}")
    _logger.info(f"  ConfigSpace: {len(configspace)} hyperparameters")
    _logger.info(f"  Problems: {len(problem_specs)}")
    _logger.info(f"  Budget per problem: {budget}")
    _logger.info(f"  SMAC trials: {hpo_config.n_trials}")

    def objective_function(
        config: Configuration, instance: str, seed: int = 0
    ) -> float:
        """
        SMAC objective function: minimize (1 - average_hypervolume).

        Args:
            config: Hyperparameter configuration to evaluate
            instance: Problem instance identifier (index into problem_specs)
            seed: Random seed for reproducibility

        Returns:
            Score to minimize (1 - hypervolume, so lower is better)
        """
        # Parse instance string to get problem index
        try:
            prob_idx = int(instance)
        except ValueError:
            _logger.error(f"Invalid instance format: {instance}")
            return 1.0  # Worst possible score

        if prob_idx < 0 or prob_idx >= len(problem_specs):
            _logger.error(f"Instance index out of range: {prob_idx}")
            return 1.0

        problem_spec = problem_specs[prob_idx]

        # Get problem from provider
        provider = PymooMOProvider()
        wrapper = provider.get(
            problem_id=problem_spec.name,
            dim=problem_spec.dim,
            ref_point=problem_spec.ref_point,
            n_obj=problem_spec.n_obj,
        )

        n_constr = int(getattr(wrapper, "n_constr", 0))

        # Track evaluations (constraints too, when the problem is constrained)
        y_hist = []
        g_hist = []

        # Soft per-trial wall-clock cap, checked per evaluation. We enforce it
        # here rather than via SMAC's trial_walltime_limit because the latter
        # runs the target under pynisher in a spawned subprocess, which must
        # pickle this closure -- and nested functions are not picklable, so it
        # crashes every trial. On exceed, func raises and the trial scores worst.
        _trial_deadline = (
            time.time() + hpo_config.trial_walltime_limit
            if hpo_config.trial_walltime_limit
            else None
        )

        def func(x):
            """Evaluation function that tracks history (and constraints if any)."""
            if _trial_deadline is not None and time.time() > _trial_deadline:
                raise TimeoutError(
                    f"trial exceeded hpo_trial_walltime "
                    f"({hpo_config.trial_walltime_limit}s)"
                )
            if len(y_hist) >= budget:
                if n_constr > 0:
                    return np.zeros(wrapper.n_obj), np.zeros(n_constr)
                return np.zeros(wrapper.n_obj)

            out = wrapper(np.asarray(x).ravel())
            if n_constr > 0:
                yy, gg = out
                y_hist.append(np.asarray(yy, dtype=float).ravel())
                g_hist.append(np.asarray(gg, dtype=float).ravel())
                return yy, gg
            y_hist.append(out)
            return out

        # Set random seed
        np.random.seed(seed)

        # Prepare init_kwargs with fixed parameters + hyperparameters from config
        init_kwargs = {
            "budget": budget,
            "dim": problem_spec.dim,
            "bounds": wrapper.bounds,
            **dict(config),  # Unpack hyperparameters from SMAC config
        }

        # Execute algorithm. Cap BLAS to 1 thread: each SMAC worker is its own
        # process, so without this every one of the hpo_n_workers processes
        # lets numpy/sklearn BLAS grab all cores -> n_workers x n_cores threads
        # oversubscribe the machine (16x16 = 256 on a 16-core box), pegging CPU
        # at 100% while wall-clock crawls and every HPO slams into the walltime.
        # 1 thread/worker => n_workers threads == clean parallelism. Mirrors the
        # final-eval path (multiobj_evaluator._run_single_moo_rep).
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                _, _, err, _ = default_exec(
                    code=code,
                    cls_name=cls_name,
                    cls=None,
                    init_kwargs=init_kwargs,
                    call_kwargs={"func": func},
                    injector=injector,
                )

            if err:
                logging.debug(f"Algorithm failed on {problem_spec.name}: {err}")
                return 1.0  # Worst score for failed runs

        except Exception as e:
            logging.debug(f"Execution error on {problem_spec.name}: {e}")
            return 1.0

        # Calculate hypervolume
        if len(y_hist) == 0:
            _logger.warning(f"No evaluations recorded for {problem_spec.name}")
            return 1.0

        try:
            Y = np.array(y_hist)
            ref_point = (
                np.array(problem_spec.ref_point)
                if problem_spec.ref_point
                else np.ones(wrapper.n_obj) * 1.2
            )

            # Constraint-aware SMAC cost. Unconstrained problems keep the
            # 1 - normalized_HV objective; constrained problems use
            # 1 - feasible_HV + lambda * infeasibility_rate (shared feasible-HV
            # scoring with the evaluator via moo_metrics.score_moo_run).
            G = np.array(g_hist) if n_constr > 0 else None
            score = _moo_objective(
                Y,
                ref_point,
                G=G,
                infeasibility_penalty=hpo_config.infeasibility_penalty,
            )

            logging.debug(
                f"  {problem_spec.name}: cost={score:.4f}, n_constr={n_constr}, "
                f"evals={len(y_hist)}, config={dict(config)}"
            )

            # A pathological config can emit huge/inf objective values, which
            # overflow the HV and yield a non-finite cost. SMAC fits its
            # surrogate on the costs, so a single inf/nan would crash the whole
            # HPO ("Input y contains infinity"). Treat non-finite as the worst
            # finite cost (same sentinel as the failure guards above) so one bad
            # config is just ranked worst rather than killing the run.
            cost = float(score)
            if not np.isfinite(cost):
                _logger.warning(
                    f"Non-finite cost ({score}) on {problem_spec.name}; clamping to 1.0"
                )
                return 1.0
            return cost

        except Exception as e:
            _logger.error(f"Score calculation error on {problem_spec.name}: {e}")
            return 1.0

    # Create instance strings (one per problem)
    instances = [str(i) for i in range(len(problem_specs))]

    # Create instance features (problem characteristics)
    # Using problem index as a simple feature
    instance_features = {inst: [float(i)] for i, inst in enumerate(instances)}

    # Setup SMAC scenario with unique name to avoid conflicts
    import random

    scenario = Scenario(
        configspace,
        name=f"{cls_name}_{int(time.time())}_{random.randint(0, 9999)}",
        deterministic=hpo_config.deterministic,
        min_budget=hpo_config.min_budget,
        max_budget=hpo_config.max_budget,
        n_trials=hpo_config.n_trials,
        walltime_limit=hpo_config.walltime_limit,
        # NOTE: we deliberately do NOT set trial_walltime_limit here. SMAC would
        # enforce it via pynisher in a spawned subprocess, which must pickle the
        # (nested, unpicklable) target function and crashes every trial. The
        # per-trial cap is enforced in-process in objective_function instead.
        # Finite cost for any crashed trial -- SMAC's default is inf, which the
        # surrogate cannot fit. 1.0 == the worst normal cost (matches the
        # objective's failure sentinel).
        crash_cost=1.0,
        instances=instances,
        instance_features=instance_features,
        n_workers=hpo_config.n_workers,
    )

    _logger.info("Running SMAC optimization...")
    start_time = time.time()

    # Progress tracking
    eval_count = [0]
    n_instances = len(problem_specs)

    def wrapped_objective(config, instance, seed):
        eval_count[0] += 1
        trial_num = (eval_count[0] - 1) // n_instances + 1
        instance_num = (eval_count[0] - 1) % n_instances + 1
        _logger.info(
            f"  SMAC trial {trial_num}/{hpo_config.n_trials}, "
            f"instance {instance_num}/{n_instances} starting..."
        )
        trial_start = time.time()
        result = objective_function(config, instance, seed)
        trial_elapsed = time.time() - trial_start
        total_elapsed = time.time() - start_time
        _logger.info(
            f"  SMAC trial {trial_num}/{hpo_config.n_trials}, "
            f"instance {instance_num}/{n_instances} done: "
            f"cost={result:.4f} ({trial_elapsed:.1f}s, total {total_elapsed:.1f}s)"
        )
        return result

    try:
        # Save root logger state before SMAC clobbers it
        root_logger = logging.getLogger()
        _saved_handlers = root_logger.handlers[:]
        _saved_level = root_logger.level

        # Create and run SMAC
        smac = AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=wrapped_objective,
            logging_level=logging.WARNING,  # Reduce SMAC verbosity
        )
        incumbent = smac.optimize()

        # Restore root logger state
        root_logger.handlers = _saved_handlers
        root_logger.setLevel(_saved_level)

        elapsed_time = time.time() - start_time
        _logger.info(
            f"  ✅ SMAC optimization completed in {elapsed_time:.2f}s ({eval_count[0]} evaluations)"
        )

    except Exception as e:
        # Restore root logger even on failure
        root_logger.handlers = _saved_handlers
        root_logger.setLevel(_saved_level)
        _logger.error(f"  ❌ SMAC optimization failed: {e}")
        # Return empty dict as fallback
        return {}, 0.0

    # Convert incumbent to dictionary
    incumbent_dict = dict(incumbent)

    # Incumbent HV is read from SMAC's own run history -- we do NOT re-evaluate
    # the algorithm here. The old re-evaluation ran the algorithm in this
    # long-lived process once per HPO problem, and torch/numpy caches from those
    # in-process runs were never released, so main-process memory grew ~1 GB per
    # generation across a long run (eventual OOM). SMAC already recorded the
    # incumbent's cost during the search, so reuse it (this value is logging-only;
    # it is not used for selection or feedback -- the final full-problem
    # evaluation produces the score that matters).
    try:
        incumbent_cost = float(smac.runhistory.get_cost(incumbent))
    except Exception:  # SMAC API variations / no recorded cost
        try:
            incumbent_cost = float(smac.runhistory.average_cost(incumbent))
        except Exception:
            incumbent_cost = float("nan")
    incumbent_hv = 1.0 - incumbent_cost

    # Final restore of root logger before returning
    root_logger.handlers = _saved_handlers
    root_logger.setLevel(_saved_level)
    _logger.info(f"Incumbent configuration: {incumbent_dict}")
    _logger.info(f"Incumbent normalized HV (from run history): {incumbent_hv:.4f}")

    # Release SMAC state for this individual so it does not accumulate in the
    # main process across generations.
    del smac, scenario, incumbent
    gc.collect()

    return incumbent_dict, incumbent_hv


def validate_with_random_config(
    code: str,
    cls_name: str,
    configspace: ConfigurationSpace,
    problem_spec: Any,
    budget: int = 100,
    injector: Any = None,
    n_retries: int = 3,
) -> tuple[bool, str]:
    """
    Validate that the algorithm works with random configurations.

    Tries up to n_retries different random configurations before rejecting.
    This avoids discarding algorithms that only fail on specific config
    combinations.

    Args:
        code: Algorithm code
        cls_name: Algorithm class name
        configspace: Configuration space
        problem_spec: Single problem to test on
        budget: Small budget for validation
        injector: Optional code injector
        n_retries: Number of random configs to try before rejecting

    Returns:
        (True, "") if validation passes, (False, error_message) otherwise
    """
    if not SMAC_AVAILABLE:
        _logger.warning("Cannot validate: SMAC not available")
        return False, "SMAC not available"

    _logger.info(
        f"Validating {cls_name} with up to {n_retries} random configurations..."
    )

    # Get problem once (reused across retries)
    provider = PymooMOProvider()
    wrapper = provider.get(
        problem_id=problem_spec.name,
        dim=problem_spec.dim,
        ref_point=problem_spec.ref_point,
        n_obj=problem_spec.n_obj,
    )
    n_constr = int(getattr(wrapper, "n_constr", 0))

    last_error = ""
    for attempt in range(n_retries):
        try:
            config = configspace.sample_configuration()
            _logger.info(f"  Attempt {attempt + 1}/{n_retries}, config: {dict(config)}")

            eval_count = [0]

            def func(x):
                eval_count[0] += 1
                if eval_count[0] > budget:
                    # Match the (F, G) contract so constrained algorithms that
                    # unpack the tuple don't crash once the budget is exhausted.
                    if n_constr > 0:
                        return np.zeros(wrapper.n_obj), np.zeros(n_constr)
                    return np.zeros(wrapper.n_obj)
                return wrapper(np.asarray(x).ravel())

            init_kwargs = {
                "budget": budget,
                "dim": problem_spec.dim,
                "bounds": wrapper.bounds,
                **dict(config),
            }

            _, _, err, _ = default_exec(
                code=code,
                cls_name=cls_name,
                cls=None,
                init_kwargs=init_kwargs,
                call_kwargs={"func": func},
                injector=injector,
            )

            if err:
                last_error = str(err)
                _logger.warning(f"  Attempt {attempt + 1}/{n_retries} failed: {err}")
                continue

            _logger.info(
                f"  ✅ Validation passed on attempt {attempt + 1} ({eval_count[0]} evaluations)"
            )
            return True, ""

        except Exception as e:
            last_error = str(e)
            _logger.warning(f"  Attempt {attempt + 1}/{n_retries} error: {e}")
            continue

    _logger.error(
        f"  Validation failed after {n_retries} attempts. Last error: {last_error}"
    )
    return False, last_error
