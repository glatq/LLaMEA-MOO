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
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from pymoo.indicators.hv import HV

from .PymooMOProvider import PymooMOProvider
from .exec_utils import default_exec

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
    walltime_limit: int = 3600  # 1 hour
    n_workers: int = 1
    deterministic: bool = False


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

        # Track evaluations
        y_hist = []

        def func(x):
            """Evaluation function that tracks history."""
            if len(y_hist) >= budget:
                return np.zeros(wrapper.n_obj)

            y = wrapper(np.asarray(x).ravel())
            y_hist.append(y)
            return y

        # Set random seed
        np.random.seed(seed)

        # Prepare init_kwargs with fixed parameters + hyperparameters from config
        init_kwargs = {
            "budget": budget,
            "dim": problem_spec.dim,
            "bounds": wrapper.bounds,
            **dict(config),  # Unpack hyperparameters from SMAC config
        }

        # Execute algorithm
        try:
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

            hv_indicator = HV(ref_point=ref_point)
            hv = hv_indicator(Y)

            # SMAC minimizes, so return negative HV (or 1 - normalized_hv)
            # Using 1 - hv to keep scores in [0, 1] range
            score = 1.0 - hv if hv > 0 else 1.0

            logging.debug(
                f"  {problem_spec.name}: HV={hv:.4f}, score={score:.4f}, "
                f"evals={len(y_hist)}, config={dict(config)}"
            )

            return float(score)

        except Exception as e:
            _logger.error(f"HV calculation error on {problem_spec.name}: {e}")
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

    # Calculate incumbent's average HV across all problems
    _logger.info("  Evaluating incumbent on all problems...")
    hvs = []
    for i, problem_spec in enumerate(problem_specs):
        score = objective_function(incumbent, str(i), seed=0)
        hv = 1.0 - score  # Convert back to hypervolume
        hvs.append(hv)

    incumbent_hv = float(np.mean(hvs))

    # Final restore of root logger before returning
    root_logger.handlers = _saved_handlers
    root_logger.setLevel(_saved_level)
    _logger.info(f"Incumbent configuration: {incumbent_dict}")
    _logger.info(f"Incumbent average HV: {incumbent_hv:.4f}")
    _logger.info(f"  HV per problem: {[f'{hv:.4f}' for hv in hvs]}")

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

    last_error = ""
    for attempt in range(n_retries):
        try:
            config = configspace.sample_configuration()
            _logger.info(f"  Attempt {attempt + 1}/{n_retries}, config: {dict(config)}")

            eval_count = [0]

            def func(x):
                eval_count[0] += 1
                if eval_count[0] > budget:
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
