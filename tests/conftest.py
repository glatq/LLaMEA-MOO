import pytest

from llamevol.evaluator import EvaluatorResult
from llamevol.prompt_generators.vanilla_bl_prompt_generator import (
    VanillaBaselinePromptGenerator,
    VanillaBaselineResponseHandler,
)

from .utils_for_tests import normalize_prompt


@pytest.fixture
def bl_task_description(vanilla_bl):
    return {
        "initialize_solution_no_bo": normalize_prompt(
            "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
        ),
        "fix_errors_no_bo": normalize_prompt(
            "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
        ),
        "fix_errors_from_error_no_bo": normalize_prompt(
            "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
        ),
        "optimize_performance_no_bo": normalize_prompt(
            "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
        ),
        "initialize_solution_bo": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "fix_errors_bo": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "fix_errors_from_error_bo": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "optimize_performance_bo": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "initialize_solution_bo_with_gpu": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries, and using GPU for acceleration is mandatory. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "fix_errors_bo_with_gpu": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries, and using GPU for acceleration is mandatory. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "fix_errors_from_error_bo_with_gpu": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries, and using GPU for acceleration is mandatory. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
        "optimize_performance_bo_with_gpu": normalize_prompt(
            f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries, and using GPU for acceleration is mandatory. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        ),
    }


@pytest.fixture
def bl_response_format():
    return {
        "initialize_solution": normalize_prompt(
            """
    Give the response in the format:
    # Description 
    <description>
    # Justification 
    <justification for the key components of the algorithm or the changes made>
    # Code 
    <code>
    """
        ),
        "fix_errors": normalize_prompt(
            """
    Give the response in the format:
    # Description 
    <description>
    # Justification 
    <justification for the key components of the algorithm or the changes made>
    # Code 
    <code>
    """
        ),
        "fix_errors_from_error": normalize_prompt(
            """
    Give the response in the format:
    # Description 
    <description>
    # Justification 
    <justification for the key components of the algorithm or the changes made>
    # Code 
    <code>
    """
        ),
        "optimize_performance": normalize_prompt(
            """
    Give the response in the format:
    # Description 
    <description>
    # Justification 
    <justification for the key components of the algorithm or the changes made>
    # Code 
    <code>
    """
        ),
    }


@pytest.fixture
def bl_code_structure():  # normalizing these prompts won't work on tests
    return {
        "bo_and_mini_bo": """
```python
from collections.abc import Callable
import numpy as np
class <AlgorithmName>:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = <your_strategy>

        # Do not add any other arguments without a default value

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # Return a tuple (best_y, best_x)

        while self.n_evals < budget:
            # Optimization
            pass

        return best_y, best_x
```
""",
        "bo_and_no_mini_bo": """
```python
from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
class <AlgorithmName>:
    def __init__(self, budget:int, dim:int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = <your_strategy>

        # Do not add any other arguments without a default value

    def _sample_points(self, n_points):
        # sample points
        # return array of shape (n_points, n_dims)

    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return the model
        # Do not change the function signature

    def _acquisition_function(self, X):
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size):
        # Select the next points to evaluate
        # Use a selection strategy to optimize/leverage the acquisition function 
        # The selection strategy can be any heuristic/evolutionary/mathematical/hybrid methods.
        # Your decision should consider the problem characteristics, acquisition function, and the computational efficiency.
        # return array of shape (batch_size, n_dims)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)

        self.n_evals += len(X)

    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        # Do not change the function signature

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)

        self._evaluate_points()
        self._update_eval_points()
        while self.n_evals < budget:
            # Optimization

            # select points by acquisition function
            self._evaluate_points()
            self._update_eval_points()

        return best_y, best_x
```
""",
        "no_bo_no_mini_bo": """
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        # bounds has (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(self.bounds[0], self.bounds[1])

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```
""",
    }


@pytest.fixture
def bl_evaluation_feedback_prompt():
    return {
        "example": (
            "The algorithm TestAlgo got an average Area over the convergence curve (AOCC, 1.0 is the best) "
            "score of 0.5000 with standard deviation 0.4082.\n"
            "\n"
            f"took 12.35 seconds to run."
        )
    }


@pytest.fixture
def bl_get_prompt():
    return {
        "role_prompt": "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems",
        "initialize_solution": f"""\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n\n\nA code structure guide is as follows and keep the comments from the guide when generating the code.\n
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        # bounds has (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(self.bounds[0], self.bounds[1])

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```

\nGive the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
""",
        "initialize_solution_with_pre_solution_prompt": f"""\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n\n2 algorithms have been designed. The new algorithm should be as **diverse** as possible from the previous ones on every aspect.\nIf the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n## Algorithm 2\n\nWith code:\n```python\n\n```\n\n\n## Algorithm 2\n\nWith code:\n```python\n\n```\n\n\n\n\nA code structure guide is as follows and keep the comments from the guide when generating the code.\n
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        # bounds has (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(self.bounds[0], self.bounds[1])

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```

\nGive the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
""",
        "non_initialize_solution_with_crossover": normalize_prompt(
            "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n\n\n\nThe selected solutions to update are:\n## Algorithm 2\n\nWith code:\n```python\n\n```\n\n\n## Algorithm 2\n\nWith code:\n```python\n\n```\n\n\nCombine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n\n\n\n\nGive the response in the format:\n# Description \n<description>\n# Justification \n<justification for the key components of the algorithm or the changes made>\n# Code \n<code>\n\n"
        ),
        "non_initialize_solution": normalize_prompt(
            "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n\n\n\nThe selected solution to update is:\n\nWith code:\n```python\n\n```\n\n\nRefine the strategy of the selected solution to improve it.\n\n\n\nGive the response in the format:\n# Description \n<description>\n# Justification \n<justification for the key components of the algorithm or the changes made>\n# Code \n<code>\n\n"
        ),
    }


@pytest.fixture
def vanilla_bl_bo_with_gpu():
    prompt = VanillaBaselinePromptGenerator()
    prompt.use_cuda = True
    prompt.is_bo = True
    return prompt


@pytest.fixture
def vanilla_bl():
    prompt = VanillaBaselinePromptGenerator()
    return prompt


@pytest.fixture
def vanilla_blrh():
    rh = VanillaBaselineResponseHandler()
    return rh


@pytest.fixture
def eval_res():
    er = EvaluatorResult()
    return er


@pytest.fixture
def vanilla_bl_bo():
    prompt = VanillaBaselinePromptGenerator()
    prompt.is_bo = True
    return prompt
