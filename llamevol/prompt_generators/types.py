from enum import Enum


class GenerationTask(Enum):
    """Enum class for generation tasks."""

    INITIALIZE_SOLUTION = 0
    FIX_ERRORS = 1
    FIX_ERRORS_FROM_ERROR = 2
    OPTIMIZE_PERFORMANCE = 3
