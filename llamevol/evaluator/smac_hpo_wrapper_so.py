"""SMAC-based hyperparameter optimization wrapper for single-objective algorithms."""

import logging
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
    logging.warning("SMAC or ConfigSpace not installed. HPO will not be available.")


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

    logging.info(f"Starting SMAC HPO for {cls_name}")
    logging.info(f"  ConfigSpace: {len(configspace)} hyperparameters")
    logging.info(f"  Problem-Instance pairs: {len(problem_instances)}")
    logging.info(f"  Budget per instance: {budget}")
    logging.info(f"  SMAC trials: {hpo_config.n_trials}")

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
            logging.error(f"Invalid instance format: {instance}")
            return 1.0  # Worst possible score

        if pair_idx < 0 or pair_idx >= len(problem_instances):
            logging.error(f"Instance index out of range: {pair_idx}")
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
            logging.warning(
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
            logging.error(
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

    logging.info("Running SMAC optimization...")
    start_time = time.time()

    # Progress tracking
    trial_count = [0]
    last_log_time = [time.time()]

    def wrapped_objective(config, instance, seed):
        trial_count[0] += 1
        current_time = time.time()

        # Log progress every 10 seconds or every 10 trials (whichever comes first)
        if (current_time - last_log_time[0] > 10) or (trial_count[0] % 10 == 0):
            elapsed = current_time - start_time
            rate = trial_count[0] / elapsed if elapsed > 0 else 0
            eta = (hpo_config.n_trials - trial_count[0]) / rate if rate > 0 else 0
            logging.info(
                f"  HPO progress: {trial_count[0]}/{hpo_config.n_trials} trials, {elapsed:.1f}s elapsed, ETA: {eta:.1f}s"
            )
            last_log_time[0] = current_time

        return objective_function(config, instance, seed)

    try:
        # Create and run SMAC
        smac = AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=wrapped_objective,
            logging_level=logging.WARNING,  # Reduce SMAC verbosity
        )

        incumbent = smac.optimize()

        elapsed_time = time.time() - start_time
        logging.info(
            f"✅ SMAC optimization completed in {elapsed_time:.2f}s ({trial_count[0]} trials)"
        )

    except Exception as e:
        logging.error(f"SMAC optimization failed: {e}")
        # Return empty dict as fallback
        return {}, 0.0

    # Convert incumbent to dictionary
    incumbent_dict = dict(incumbent)

    # Calculate incumbent's average AOC across all problem-instance pairs
    logging.info("Evaluating incumbent on all problem-instance pairs...")
    aocs = []
    for i in range(len(problem_instances)):
        score = objective_function(incumbent, str(i), seed=0)
        aoc = 1.0 - score  # Convert back to AOC
        aocs.append(aoc)

    incumbent_aoc = float(np.mean(aocs))

    logging.info(f"Incumbent configuration: {incumbent_dict}")
    logging.info(f"Incumbent average AOC: {incumbent_aoc:.4f}")
    logging.info(f"  AOC per problem-instance: {[f'{aoc:.4f}' for aoc in aocs]}")

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
) -> bool:
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
        True if validation passes, False otherwise
    """
    if not SMAC_AVAILABLE:
        logging.warning("Cannot validate: SMAC not available")
        return False

    logging.info(f"Validating {cls_name} with random configuration...")

    try:
        # Sample a random configuration
        config = configspace.sample_configuration()
        logging.info(f"  Test config: {dict(config)}")

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
            logging.error(f"  Validation failed: {err}")
            return False

        logging.info(f"  ✅ Validation passed ({eval_count[0]} evaluations)")
        return True

    except Exception as e:
        logging.error(f"  Validation error: {e}")
        return False
