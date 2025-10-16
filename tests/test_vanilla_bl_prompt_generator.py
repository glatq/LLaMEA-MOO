import pytest
from llamevol.prompt_generators.abstract_prompt_generator import GenerationTask
from llamevol.prompt_generators.vanilla_bl_prompt_generator import (
    VanillaBaselinePromptGenerator,
    VanillaBaselineResponseHandler,
)
from llamevol.evaluator.evaluator_result import EvaluatorResult, EvaluatorBasicResult
import torch


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


def test_task_description(vanilla_bl):
    expected = "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
    actual = vanilla_bl.task_description(GenerationTask.INITIALIZE_SOLUTION)
    assert expected == actual

    expected = "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
    actual = vanilla_bl.task_description(GenerationTask.FIX_ERRORS)
    assert expected == actual

    expected = "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
    actual = vanilla_bl.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
    assert expected == actual

    expected = "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"
    actual = vanilla_bl.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
    assert expected == actual

    vanilla_bl.is_bo = True
    lib_prompt = "As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries."
    if torch.cuda.is_available():
        lib_prompt = "As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries, and using GPU for acceleration is mandatory."

    expected = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
    actual = vanilla_bl.task_description(GenerationTask.INITIALIZE_SOLUTION)
    assert expected == actual

    expected = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
    actual = vanilla_bl.task_description(GenerationTask.FIX_ERRORS)
    assert expected == actual

    expected = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
    actual = vanilla_bl.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
    assert expected == actual

    expected = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of {vanilla_bl.problem_desc}. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
{lib_prompt} Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
    actual = vanilla_bl.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
    assert expected == actual


def test_response_format(vanilla_bl):
    expected = """
Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
"""
    actual = vanilla_bl.response_format(GenerationTask.INITIALIZE_SOLUTION)
    assert expected == actual

    actual = vanilla_bl.response_format(GenerationTask.FIX_ERRORS)
    assert expected == actual

    actual = vanilla_bl.response_format(GenerationTask.FIX_ERRORS_FROM_ERROR)
    assert expected == actual

    actual = vanilla_bl.response_format(GenerationTask.OPTIMIZE_PERFORMANCE)
    assert expected == actual


def test_code_structure(vanilla_bl):
    vanilla_bl.is_bo = True
    vanilla_bl.use_mini_bo = True
    expected = """
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
"""
    actual = vanilla_bl.code_structure()
    assert expected == actual

    vanilla_bl.is_bo = True
    vanilla_bl.use_mini_bo = False
    expected = """
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
"""
    actual = vanilla_bl.code_structure()
    assert expected == actual

    vanilla_bl.is_bo = False
    vanilla_bl.use_mini_bo = False
    expected = """
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
"""
    actual = vanilla_bl.code_structure()
    assert expected == actual


def make_basic_result(id_str: str, log_aoc: float) -> EvaluatorBasicResult:
    r = EvaluatorBasicResult()
    r.id = id_str
    r.log_y_aoc = log_aoc
    return r


def test_evaluation_feedback_prompt_exact_string(vanilla_bl, eval_res):
    # Populate the provided EvaluatorResult
    eval_res.name = "TestAlgo"
    eval_res.total_execution_time = 12.3456
    eval_res.result = [
        make_basic_result("1-1-1", 1.0),
        make_basic_result("6-1-1", 0.5),
        make_basic_result("20-1-1", 0.0),
    ]

    expected = (
        "The algorithm TestAlgo got an average Area over the convergence curve (AOCC, 1.0 is the best) "
        "score of 0.5000 with standard deviation 0.4082.\n"
        "\n"
        f"took 12.35 seconds to run."
    )
    actual = vanilla_bl.evaluation_feedback_prompt(eval_res)
    assert actual == expected


def test_evaluation_feedback_prompt_empty_when_no_results(vanilla_bl, eval_res):
    # Your fixture returns an empty EvaluatorResult => should return ""
    assert vanilla_bl.evaluation_feedback_prompt(eval_res) == ""


def test_get_prompt_with_empty_solution_and_non_INITIALIZE_SOLUTION_task(vanilla_bl):
    expected_role_prompt, expected_final_prompt = "", ""
    actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
        task=GenerationTask.OPTIMIZE_PERFORMANCE,
        problem_desc="24 noiseless functions",
        candidates=[],
    )
    assert expected_role_prompt == actual_role_prompt
    assert expected_final_prompt == actual_final_prompt


def test_get_prompt_with_INITIALIZE_SOLUTION_task(vanilla_bl, vanilla_blrh):
    expected_role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems"

    task_prompt = "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"

    pre_solution_prompt = ""  # the only one changing with number of candidates in INITIALIZE_SOLUTION. Empty for 0 candidates.
    code_structure_prompt = """A code structure guide is as follows and keep the comments from the guide when generating the code.\n
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
"""
    response_format_prompt = """
Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
"""

    expected_final_prompt = f"""{task_prompt}\n{pre_solution_prompt}\n{code_structure_prompt}\n{response_format_prompt}"""

    actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
        task=GenerationTask.INITIALIZE_SOLUTION,
        problem_desc="24 noiseless functions",
        candidates=[],
    )

    assert expected_role_prompt == actual_role_prompt
    assert expected_final_prompt == actual_final_prompt

    c1 = vanilla_blrh
    c1.code_name = "Algorithm 1"
    c2 = vanilla_blrh
    c2.code_name = "Algorithm 2"

    pre_solution_prompt = f"2 algorithms have been designed. The new algorithm should be as **diverse** as possible from the previous ones on every aspect.\nIf the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n"

    candidate_prompt = "\nWith code:\n```python\n\n```\n\n"
    pre_solution_prompt += f"## {c1.code_name}\n{candidate_prompt}\n"
    pre_solution_prompt += f"## {c2.code_name}\n{candidate_prompt}\n"
    pre_solution_prompt += "\n"

    expected_final_prompt = f"""{task_prompt}\n{pre_solution_prompt}\n{code_structure_prompt}\n{response_format_prompt}"""
    actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
        task=GenerationTask.INITIALIZE_SOLUTION,
        problem_desc="24 noiseless functions",
        candidates=[c1, c2],
    )
    assert expected_final_prompt == actual_final_prompt


def test_get_prompt_with_non_INITIALIZE_SOLUTION_task(vanilla_bl, vanilla_blrh):
    c1 = vanilla_blrh
    c1.code_name = "Algorithm 1"
    c2 = vanilla_blrh
    c2.code_name = "Algorithm 2"

    expected_role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems"

    task_prompt = "\nThe optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.\nThe func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.\nGive an excellent, novel and computationally efficient heuristic algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main idea and justify your decision about the algorithm.\n"

    population_summary = ""
    selected_prompt = "The selected solutions to update are:\n"
    candidate_prompt = "\nWith code:\n```python\n\n```\n\n"
    selected_prompt += f"## {c1.code_name}\n{candidate_prompt}\n"
    selected_prompt += f"## {c2.code_name}\n{candidate_prompt}\n"
    selected_prompt += f"Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n\n"

    response_format_prompt = """
Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
"""
    expected_final_prompt = f"""{task_prompt}
{population_summary}

{selected_prompt}

{response_format_prompt}
"""
    actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
        task=GenerationTask.FIX_ERRORS,
        problem_desc="24 noiseless functions",
        candidates=[c1, c2],
    )
    assert expected_role_prompt == actual_role_prompt
    assert expected_final_prompt == actual_final_prompt

    selected_prompt = f"""The selected solution to update is:\n{candidate_prompt}\nRefine the strategy of the selected solution to improve it.\n"""
    expected_final_prompt = f"""{task_prompt}
{population_summary}

{selected_prompt}

{response_format_prompt}
"""
    actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
        task=GenerationTask.FIX_ERRORS_FROM_ERROR,
        problem_desc="24 noiseless functions",
        candidates=[c1],
    )
    assert expected_final_prompt == actual_final_prompt


def test_get_response_handler(vanilla_bl):
    rh = vanilla_bl.get_response_handler()
    assert type(rh) is VanillaBaselineResponseHandler
