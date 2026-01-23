"""Utilities for extracting and handling SMAC ConfigurationSpace from LLM responses."""

import re
import logging
from typing import Optional, Dict, Any

try:
    from ConfigSpace import ConfigurationSpace

    CONFIGSPACE_AVAILABLE = True
except ImportError:
    ConfigurationSpace = None
    CONFIGSPACE_AVAILABLE = False
    logging.warning(
        "ConfigSpace is not installed. Install with: pip install ConfigSpace"
    )


def extract_configspace(
    config_space_str: str,
) -> Optional[ConfigurationSpace]:
    """Extract SMAC ConfigSpace from a configuration space string.

    The string should contain a Python dictionary defining the hyperparameter
    search space in the format:
    {
        "param_name": (lower_bound, upper_bound),  # For continuous/integer ranges
        "log_param": (lower, upper, "log"),  # Log-scale continuous parameter
        "categorical_param": ["option1", "option2"]  # For categorical choices
    }

    Args:
        config_space_str: String containing the configuration space dictionary

    Returns:
        ConfigurationSpace object or None if parsing fails or ConfigSpace not installed

    Examples:
        >>> config_str = '''
        ... {
        ...     "pop_size": (10, 100),
        ...     "learning_rate": (0.001, 0.1, "log"),
        ...     "selection": ["tournament", "roulette"]
        ... }
        ... '''
        >>> cs = extract_configspace(config_str)
        >>> print(cs)
    """
    if not CONFIGSPACE_AVAILABLE:
        logging.warning("ConfigSpace not available, cannot extract configuration space")
        return None

    if not config_space_str or not config_space_str.strip():
        logging.warning("Empty configuration space string provided")
        return None

    try:
        from ConfigSpace import (
            Float,
            Integer,
            Categorical,
        )

        # Clean up the string (remove potential markdown artifacts)
        cleaned_str = config_space_str.strip()

        # Try to evaluate the string as a Python dictionary
        config_dict = eval(cleaned_str)

        if not isinstance(config_dict, dict):
            logging.error(
                f"Configuration space is not a dictionary: {type(config_dict)}"
            )
            return None

        # Manually construct ConfigurationSpace
        cs = ConfigurationSpace()

        for param_name, param_spec in config_dict.items():
            try:
                if isinstance(param_spec, (list, tuple)) and len(param_spec) >= 2:
                    if isinstance(param_spec, list):
                        # Categorical parameter
                        cs.add(Categorical(param_name, param_spec))
                    elif len(param_spec) == 2:
                        # Numeric parameter (int or float) without log flag
                        lower, upper = param_spec
                        if isinstance(lower, int) and isinstance(upper, int):
                            cs.add(Integer(param_name, bounds=(lower, upper)))
                        else:
                            cs.add(Float(param_name, bounds=(lower, upper)))
                    elif len(param_spec) == 3:
                        # Numeric parameter with log flag or other modifier
                        lower, upper, modifier = param_spec
                        log_scale = modifier in ["log", "log-uniform", "log-scale"]

                        if (
                            isinstance(lower, int)
                            and isinstance(upper, int)
                            and not log_scale
                        ):
                            cs.add(Integer(param_name, bounds=(lower, upper)))
                        else:
                            cs.add(
                                Float(param_name, bounds=(lower, upper), log=log_scale)
                            )
                    else:
                        logging.warning(
                            f"Unexpected parameter spec for '{param_name}': {param_spec}"
                        )
                else:
                    logging.warning(
                        f"Invalid parameter spec for '{param_name}': {param_spec}"
                    )
            except Exception as e:
                logging.error(f"Failed to add parameter '{param_name}': {e}")
                continue

        if len(cs) == 0:
            logging.error("No valid hyperparameters found in configuration space")
            return None

        logging.info(
            f"Successfully extracted ConfigSpace with {len(cs)} hyperparameters"
        )
        return cs

    except SyntaxError as e:
        logging.error(f"Syntax error parsing configuration space: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to parse configuration space: {e}")
        logging.debug(f"Configuration space string was: {config_space_str}")
        return None


def extract_configspace_from_response(message: str) -> Optional[ConfigurationSpace]:
    """Extract configuration space from full LLM response.

    This function searches for the "Space:" section in the LLM response
    and extracts the configuration space dictionary.

    Args:
        message: The full LLM response containing the Space section

    Returns:
        ConfigurationSpace object or None if parsing fails

    Examples:
        >>> response = '''
        ... # Description
        ... My algorithm
        ... # Code
        ... ```python
        ... class MyAlgo:
        ...     pass
        ... ```
        ... # Space
        ... ```python
        ... {
        ...     "pop_size": (10, 100)
        ... }
        ... ```
        ... '''
        >>> cs = extract_configspace_from_response(response)
    """
    if not CONFIGSPACE_AVAILABLE:
        return None

    # Pattern to match Space section with Python code block
    pattern = r"#\s*[Ss]pace[\s\S]*?```(?:python)?\s*([\s\S]*?)```"

    matches = re.finditer(pattern, message, re.IGNORECASE | re.DOTALL)

    for match in matches:
        try:
            config_str = match.group(1)
            cs = extract_configspace(config_str)
            if cs is not None:
                return cs
        except Exception as e:
            logging.warning(f"Failed to extract ConfigSpace from match: {e}")
            continue

    logging.warning("No valid configuration space found in response")
    return None


def configspace_to_dict(cs: ConfigurationSpace) -> Dict[str, Any]:
    """Convert ConfigurationSpace to a serializable dictionary.

    This is useful for logging and storing configuration spaces.

    Args:
        cs: ConfigurationSpace object

    Returns:
        Dictionary representation of the configuration space
    """
    if cs is None:
        return {}

    try:
        # ConfigSpace provides a to_serialized_dict method
        return cs.to_serialized_dict()
    except Exception as e:
        logging.error(f"Failed to serialize ConfigSpace: {e}")
        return {}


def validate_configspace(cs: ConfigurationSpace) -> bool:
    """Validate that a ConfigurationSpace is properly formed.

    Args:
        cs: ConfigurationSpace object to validate

    Returns:
        True if valid, False otherwise
    """
    if cs is None:
        return False

    if not CONFIGSPACE_AVAILABLE:
        return False

    try:
        # Check that it has at least one hyperparameter
        if len(cs) == 0:
            logging.warning("ConfigurationSpace is empty (no hyperparameters)")
            return False

        # Try to sample a configuration (validates structure)
        _ = cs.sample_configuration()

        return True
    except Exception as e:
        logging.error(f"ConfigSpace validation failed: {e}")
        return False
