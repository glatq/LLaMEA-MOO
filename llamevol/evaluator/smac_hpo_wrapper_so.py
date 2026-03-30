"""SMAC-based hyperparameter optimization wrapper for single-objective algorithms."""

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
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .ioh_objective_provider import IOHProvider
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


def compute_aoc_score(y_hist, budget, optimal_value, lower=1e-8, upper=1e4):
    """
    Compute Area-Over-Curve (AOC) score for single-objective optimization.

    This is the same metric used in IOHEvaluator.
    Higher AOC is better (closer to 1.0 means better performance).
    """
    if y_hist is None or len(y_hist) == 0:
        return 0.0

    y = np.array(y_hist).reshape(-1)
    best = np.minimum.accumulate(y)

    if optimal_value is not None:
        best = best - optimal_value

    best = best[:budget] if len(best) > budget else best
    best = np.clip(best, lower, upper)
    logbest = np.log10(best)
    lo, hi = np.log10(lower), np.log10(upper)
    a = np.clip((logbest - lo) / (hi - lo), 0.0, 1.0)

    return float(1.0 - np.sum(a) / budget)


def run_smac_hpo_so(
    code: str,
    cls_name: str,
    configspace: ConfigurationSpace,
    problem_ids: List[int],
    instance_ids: List[List[int]],
    dim: int,
    budget: int,
    hpo_config: Optional[SMACHPOConfig] = None,
    injector: Any = None,
) -> Tuple[Dict[str, Any], float]:
    """Run SMAC HPO for single-objective optimization algorithm.

    Args:
        code: Python code containing the algorithm class
        cls_name: Name of the algorithm class
        configspace: ConfigurationSpace defining hyperparameter search space
        problem_ids: List of IOH problem IDs to optimize on
        instance_ids: List of instance IDs for each problem
        dim: Problem dimension
        budget: Evaluation budget per problem instance
        hpo_config: SMAC configuration (uses defaults if None)
        injector: Optional code injector for modifying algorithm behavior

    Returns:
        Tuple of (incumbent_dict, incumbent_aoc):
            - incumbent_dict: Best hyperparameter configuration found
            - incumbent_aoc: Best average AOC score achieved

    Raises:
        RuntimeError: If SMAC is not available
    """
    if not SMAC_AVAILABLE:
        raise RuntimeError(
            "SMAC is not installed. Install with: pip install smac ConfigSpace"
        )

    if hpo_config is None:
        hpo_config = SMACHPOConfig()

    # Create flat list of (problem_id, instance_id) pairs
    problem_instances = []
    for prob_id, inst_list in zip(problem_ids, instance_ids):
        for inst_id in inst_list:
            problem_instances.append((prob_id, inst_id))

    _logger.info(f"Starting SMAC HPO for {cls_name}")
    _logger.info(f"  ConfigSpace: {len(configspace)} hyperparameters")
    _logger.info(f"  Problem-Instance pairs: {len(problem_instances)}")
    _logger.info(f"  Budget per instance: {budget}")
    _logger.info(f"  SMAC trials: {hpo_config.n_trials}")

    provider = IOHProvider()

    def objective_function(
        config: Configuration, instance: str, seed: int = 0
    ) -> float:
        """
        SMAC objective function: minimize (1 - AOC).

        Args:
            config: Hyperparameter configuration to evaluate
            instance: Problem instance identifier (index into problem_instances)
            seed: Random seed for reproducibility

        Returns:
            Score to minimize (1 - AOC, so lower is better)
        """
        # Parse instance string to get problem-instance pair index
        try:
            pair_idx = int(instance)
        except ValueError:
            _logger.error(f"Invalid instance format: {instance}")
            return 1.0  # Worst possible score

        if pair_idx < 0 or pair_idx >= len(problem_instances):
            _logger.error(f"Instance index out of range: {pair_idx}")
            return 1.0

        problem_id, instance_id = problem_instances[pair_idx]

        # Get IOH problem
        problem = provider.get(problem_id, instance_id, dim)

        # Track evaluations
        y_hist = []

        def func(x):
            """Evaluation function that tracks history."""
            if len(y_hist) >= budget:
                # Return dummy value if budget exceeded
                return 0.0

            y = problem(x)
            y_hist.append(float(y))
            return y

        # Set random seed
        np.random.seed(seed)

        # Prepare init_kwargs with fixed parameters + hyperparameters from config
        init_kwargs = {
            "dim": dim,
            "budget": budget,
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
                logging.debug(
                    f"Algorithm failed on problem {problem_id} instance {instance_id}: {err}"
                )
                return 1.0  # Worst score for failed runs

        except Exception as e:
            logging.debug(
                f"Execution error on problem {problem_id} instance {instance_id}: {e}"
            )
            return 1.0

        # Calculate AOC score
        if len(y_hist) == 0:
            _logger.warning(
                f"No evaluations recorded for problem {problem_id} instance {instance_id}"
            )
            return 1.0

        try:
            optimal_value = problem.optimum_y
            aoc = compute_aoc_score(y_hist, budget, optimal_value)

            # SMAC minimizes, so return 1 - AOC
            score = 1.0 - aoc

            logging.debug(
                f"  Problem {problem_id} Instance {instance_id}: AOC={aoc:.4f}, score={score:.4f}, "
                f"evals={len(y_hist)}, config={dict(config)}"
            )

            return float(score)

        except Exception as e:
            _logger.error(
                f"AOC calculation error on problem {problem_id} instance {instance_id}: {e}"
            )
            return 1.0

    # Create instance strings (one per problem-instance pair)
    instances = [str(i) for i in range(len(problem_instances))]

    # Create instance features (problem characteristics)
    # Using pair index as a simple feature
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
    n_instances = len(problem_instances)

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

    # Calculate incumbent's average AOC across all problem-instance pairs
    _logger.info("  Evaluating incumbent on all problem-instance pairs...")
    aocs = []
    for i in range(len(problem_instances)):
        score = objective_function(incumbent, str(i), seed=0)
        aoc = 1.0 - score  # Convert back to AOC
        aocs.append(aoc)

    incumbent_aoc = float(np.mean(aocs))

    # Final restore of root logger before returning
    root_logger.handlers = _saved_handlers
    root_logger.setLevel(_saved_level)
    _logger.info(f"Incumbent configuration: {incumbent_dict}")
    _logger.info(f"Incumbent average AOC: {incumbent_aoc:.4f}")
    _logger.info(f"  AOC per problem-instance: {[f'{aoc:.4f}' for aoc in aocs]}")

    return incumbent_dict, incumbent_aoc


def validate_with_random_config(
    code: str,
    cls_name: str,
    configspace: ConfigurationSpace,
    problem_id: int,
    instance_id: int,
    dim: int,
    budget: int = 100,
    injector: Any = None,
) -> tuple[bool, str]:
    """
    Validate that the algorithm works with a random configuration.

    This is a quick sanity check before running full HPO.

    Args:
        code: Algorithm code
        cls_name: Algorithm class name
        configspace: Configuration space
        problem_id: IOH problem ID to test on
        instance_id: IOH instance ID to test on
        dim: Problem dimension
        budget: Small budget for validation
        injector: Optional code injector

    Returns:
        (True, "") if validation passes, (False, error_message) otherwise
    """
    if not SMAC_AVAILABLE:
        _logger.warning("Cannot validate: SMAC not available")
        return False, "SMAC not available"

    _logger.info(f"Validating {cls_name} with random configuration...")

    try:
        # Sample a random configuration
        config = configspace.sample_configuration()
        _logger.info(f"  Test config: {dict(config)}")

        # Get IOH problem
        provider = IOHProvider()
        problem = provider.get(problem_id, instance_id, dim)

        # Simple eval function
        eval_count = [0]

        def func(x):
            eval_count[0] += 1
            if eval_count[0] > budget:
                return 0.0
            return problem(x)

        # Try to execute
        init_kwargs = {
            "dim": dim,
            "budget": budget,
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
            _logger.error(f"  Validation failed: {err}")
            return False, str(err)

        _logger.info(f"  ✅ Validation passed ({eval_count[0]} evaluations)")
        return True, ""

    except Exception as e:
        _logger.error(f"  Validation error: {e}")
        return False, str(e)
