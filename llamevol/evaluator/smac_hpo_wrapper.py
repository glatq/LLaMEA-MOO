"""SMAC-based hyperparameter optimization wrapper for multi-objective algorithms."""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

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

    logging.info(f"Starting SMAC HPO for {cls_name}")
    logging.info(f"  ConfigSpace: {len(configspace)} hyperparameters")
    logging.info(f"  Problems: {len(problem_specs)}")
    logging.info(f"  Budget per problem: {budget}")
    logging.info(f"  SMAC trials: {hpo_config.n_trials}")

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
            logging.error(f"Invalid instance format: {instance}")
            return 1.0  # Worst possible score

        if prob_idx < 0 or prob_idx >= len(problem_specs):
            logging.error(f"Instance index out of range: {prob_idx}")
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
            logging.warning(f"No evaluations recorded for {problem_spec.name}")
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
            logging.error(f"HV calculation error on {problem_spec.name}: {e}")
            return 1.0

    # Create instance strings (one per problem)
    instances = [str(i) for i in range(len(problem_specs))]

    # Create instance features (problem characteristics)
    # Using problem index as a simple feature
    instance_features = {inst: [float(i)] for i, inst in enumerate(instances)}

    # Setup SMAC scenario
    scenario = Scenario(
        configspace,
        name=f"{cls_name}_{int(time.time())}",
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

    # Calculate incumbent's average HV across all problems
    logging.info("Evaluating incumbent on all problems...")
    hvs = []
    for i, problem_spec in enumerate(problem_specs):
        score = objective_function(incumbent, str(i), seed=0)
        hv = 1.0 - score  # Convert back to hypervolume
        hvs.append(hv)

    incumbent_hv = float(np.mean(hvs))

    logging.info(f"Incumbent configuration: {incumbent_dict}")
    logging.info(f"Incumbent average HV: {incumbent_hv:.4f}")
    logging.info(f"  HV per problem: {[f'{hv:.4f}' for hv in hvs]}")

    return incumbent_dict, incumbent_hv


def validate_with_random_config(
    code: str,
    cls_name: str,
    configspace: ConfigurationSpace,
    problem_spec: Any,
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
        problem_spec: Single problem to test on
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

        # Get problem
        provider = PymooMOProvider()
        wrapper = provider.get(
            problem_id=problem_spec.name,
            dim=problem_spec.dim,
            ref_point=problem_spec.ref_point,
            n_obj=problem_spec.n_obj,
        )

        # Simple eval function
        eval_count = [0]

        def func(x):
            eval_count[0] += 1
            if eval_count[0] > budget:
                return np.zeros(wrapper.n_obj)
            return wrapper(np.asarray(x).ravel())

        # Try to execute
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
            logging.error(f"  Validation failed: {err}")
            return False

        logging.info(f"  ✅ Validation passed ({eval_count[0]} evaluations)")
        return True

    except Exception as e:
        logging.error(f"  Validation error: {e}")
        return False
