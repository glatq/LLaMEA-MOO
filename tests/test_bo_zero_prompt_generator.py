import pytest
from llamevol.prompt_generators.abstract_prompt_generator import GenerationTask
from llamevol.prompt_generators.bo_zero_prompt_generator import (
    BoZeroPromptGenerator,
    BoZeroResponseHandler,
    BOPromptGeneratorReturnChecker,
)
from llamevol.evaluator.evaluator_result import EvaluatorResult, EvaluatorBasicResult


@pytest.fixture()
def bozero_pg():
    return BoZeroPromptGenerator()


@pytest.fixture
def eval_res():
    er = EvaluatorResult()
    return er


def test_task_description(bozero_pg):
    actual = bozero_pg.task_description(GenerationTask.INITIALIZE_SOLUTION)
    expected = """## Task Description\nYou will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible Bayesian Optimization algorithm, and implement it."""
    assert expected == actual

    actual = bozero_pg.task_description(GenerationTask.FIX_ERRORS)
    expected = """## Task Description\nYou will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."""
    assert expected == actual

    actual = bozero_pg.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
    expected = """## Task Description\nYou will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."""
    assert expected == actual

    actual = bozero_pg.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
    expected = """## Task Description\nYou will be given a Bayesian Optimization solution with evaluation feedback. Your task is to optimize the performance of the solution."""
    assert expected == actual


def test_task_instruction_for_scientist(bozero_pg):
    actual = bozero_pg.task_instruction_for_scientist(
        GenerationTask.INITIALIZE_SOLUTION
    )
    expected = """\n**as a computer scientist specialized in bayesian optimization**\n1. Analyze the minimization optimization problem.
2. Design a Bayesian Optimization algorithm that addresses the challenges of the problem. Justify your choices of techniques and hyperparameters.
3. Pseudocode: Write down the key steps of your chosen Bayesian Optimization algorithm in plain pseudocode, highlighting any novel components or adaptations.
"""
    assert expected == actual

    actual = bozero_pg.task_instruction_for_scientist(GenerationTask.FIX_ERRORS)
    expected = (
        """\n**as a computer scientist specialized in bayesian optimization**\n"""
    )
    assert expected == actual

    actual = bozero_pg.task_instruction_for_scientist(
        GenerationTask.FIX_ERRORS_FROM_ERROR
    )
    expected = (
        """\n**as a computer scientist specialized in bayesian optimization**\n"""
    )
    assert expected == actual

    actual = bozero_pg.task_instruction_for_scientist(
        GenerationTask.OPTIMIZE_PERFORMANCE
    )
    expected = """\n**as a computer scientist specialized in bayesian optimization**\n1. Analyze the minimization optimization problem.
2. Analyze the solution and its evaluation feedback.
3. Optimize the solution to improve its performance.
4. Pseudocode: Write down the key changes of your chosen strategy in plain pseudocode. 
"""
    assert expected == actual


def test_task_instruction_for_programmer(bozero_pg):
    actual = bozero_pg.task_instruction_for_programmer(
        GenerationTask.INITIALIZE_SOLUTION
    )
    expected = """\n**as a programmer specialized in python.**\n- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual

    actual = bozero_pg.task_instruction_for_programmer(GenerationTask.FIX_ERRORS)
    expected = """\n**as a programmer specialized in python.**\n- Identify the cause of the previous errors.
- Review all the code for potential errors. Here, only make most confident guesses.
- Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Correct the errors based on the identified causes and proposed solutions
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozero_pg.task_instruction_for_programmer(
        GenerationTask.FIX_ERRORS_FROM_ERROR
    )
    expected = """\n**as a programmer specialized in python.**\n- Identify the cause of the previous errors.
- Review all the code for potential errors. Here, only make most confident guesses.
- Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Correct the errors based on the identified causes and proposed solutions
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozero_pg.task_instruction_for_programmer(
        GenerationTask.OPTIMIZE_PERFORMANCE
    )

    expected = """\n**as a programmer specialized in python.**\n- Implement the algorithm in Python strictly following the previous code structure. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the modification.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual


def test_task_instruction(bozero_pg):
    actual = bozero_pg.task_instruction(GenerationTask.INITIALIZE_SOLUTION)
    expected = """## Task Instruction\nYou need to act as a computer scientist and programmer independently.\n
**as a computer scientist specialized in bayesian optimization**\n1. Analyze the minimization optimization problem.
2. Design a Bayesian Optimization algorithm that addresses the challenges of the problem. Justify your choices of techniques and hyperparameters.
3. Pseudocode: Write down the key steps of your chosen Bayesian Optimization algorithm in plain pseudocode, highlighting any novel components or adaptations.
\n**as a programmer specialized in python.**\n- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual

    actual = bozero_pg.task_instruction(GenerationTask.FIX_ERRORS)
    expected = """## Task Instruction\n\n**as a programmer specialized in python.**\n- Identify the cause of the previous errors.
- Review all the code for potential errors. Here, only make most confident guesses.
- Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Correct the errors based on the identified causes and proposed solutions
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozero_pg.task_instruction(GenerationTask.FIX_ERRORS_FROM_ERROR)
    expected = """## Task Instruction\n\n**as a programmer specialized in python.**\n- Identify the cause of the previous errors.
- Review all the code for potential errors. Here, only make most confident guesses.
- Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Correct the errors based on the identified causes and proposed solutions
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozero_pg.task_instruction(GenerationTask.OPTIMIZE_PERFORMANCE)
    expected = """## Task Instruction\nYou need to act as a computer scientist, and programmer independently.\n\n**as a computer scientist specialized in bayesian optimization**\n1. Analyze the minimization optimization problem.
2. Analyze the solution and its evaluation feedback.
3. Optimize the solution to improve its performance.
4. Pseudocode: Write down the key changes of your chosen strategy in plain pseudocode. 
\n**as a programmer specialized in python.**\n- Implement the algorithm in Python strictly following the previous code structure. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the modification.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual


def test_empty_evaluation_feedback_prompt(bozero_pg, eval_res):
    actual = bozero_pg.evaluation_feedback_prompt(eval_res)
    expected = ""
    assert expected == actual


def make_basic_result(id_str: str, log_aoc: float) -> EvaluatorBasicResult:
    r = EvaluatorBasicResult()
    r.id = id_str
    r.log_y_aoc = log_aoc
    return r


@pytest.mark.xfail(reason="Seems broken")
def test_evaluation_feedback_prompt(bozero_pg, eval_res):
    res_1 = EvaluatorBasicResult()
    res_1.budget = 100
    res_1.best_y = 1
    res_1.y_aoc = 0.1
    res_2 = EvaluatorBasicResult()
    res_2.budget = 200
    res_2.best_y = 2
    res_2.y_aoc = 0.2
    eval_res.result = [res_1, res_2]
    eval_res.name = "results"

    actual = bozero_pg.evaluation_feedback_prompt(eval_res)
    expected = f"""### Feedback\n- Budget: 100\n- Optimal Value\n#### results\n- best y: 1.00\n- AOC for all y: 0.10\n- best y: 2.00\n- AOC for all y: 0.20\n#### Note:
- AOC(Area Over the Convergence Curve): a measure of the convergence speed of the algorithm, ranged between 0.0 and 1.0. A higher value is better.
- Budget: Maximum number of function evaluations allowed for the algorithm."""
    assert expected == actual
    assert False


def test_code_structure(bozero_pg):
    actual = bozero_pg.code_structure()
    expected = """## Code Structure Guide
```python
from typing import Callable
from scipy.stats import qmc # If you are using QMC sampling. Otherwise or you have a better alternative, remove this line.
import numpy as np
class <AlgorithmName>:
    # add the docstring of the class here
    def __init__(self):
        # Initialize optimizer settings
        # Configure acquisition function
        # Do not add any other arguments without a default value

    def _sample_points(self, n_points) -> np.ndarray:
        # sample points
        # return array of shape (n_points, n_dims)
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return  the model

    def _get_model_loss(self, model, X, y) -> np.float64:
        # Calculate the loss of the model
        # return the loss of the model
    
    def _acquisition_function(self, X) -> np.ndarray:
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        # return array of shape (batch_size, n_dims)
    
    def optimize(self, objective_fn:Callable[[np.ndarray], np.ndarray], bounds:np.ndarray, budget:int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        # objective_fn: Callable[[np.ndarray], np.ndarray], takes array of shape (n_points, n_dims) and returns array of shape (n_points, 1).
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        # Do not change the function signature
        # Evaluate the model using the metric you choose and record the value as model_loss after each training. the size of the model_loss should be equal to the number of iterations plus one for the fit on initial points.
        # Return a tuple (all_y, all_x, (model_losses, loss_name), n_initial_points)
        
        n_initial_points = <your_strategy>
        rest_of_budget = budget - n_initial_points
        while rest_of_budget > 0:
           # Optimization
           
           rest_of_budget -= <the number of points evaluated by objective_fn in this iteration, e.g. x.shape[0] if x is an array>

    ## You are free to add additional methods as needed and modify the existing ones except for the optimize method and __init__ method.
    ## Rename the class based on the characteristics of the algorithm as '<anyName>BO'
    
```
"""
    assert expected == actual


def test_response_format(bozero_pg):
    actual = bozero_pg.response_format(GenerationTask.INITIALIZE_SOLUTION)
    expected = """
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- problem analysis
- the design of the algorithm
### /Description

### Pseudocode
### /Pseudocode


### Code
```
<Algorithm Implementation> 
```
### /Code
"""
    assert expected == actual

    actual = bozero_pg.response_format(GenerationTask.FIX_ERRORS)
    expected = f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- Identified Errors
- Proposed Solutions
### /Description



### Code
```
<Corrected Code>
```
### /Code
"""
    assert expected == actual

    actual = bozero_pg.response_format(GenerationTask.FIX_ERRORS_FROM_ERROR)
    expected = f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- Identified Errors
- Proposed Solutions
### /Description



### Code
```
<Corrected Code>
```
### /Code
"""
    assert expected == actual

    actual = bozero_pg.response_format(GenerationTask.OPTIMIZE_PERFORMANCE)
    expected = f"""
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- problem analysis
- feedback analysis
- the design of the algorithm
### /Description

### Pseudocode
### /Pseudocode


### Code
```
<Optimized Code>
```
### /Code
"""
    assert expected == actual


def test_get_response_handler(bozero_pg):
    rh = bozero_pg.get_response_handler()
    assert type(rh) is BoZeroResponseHandler


def test_get_return_checker(bozero_pg):
    rc = bozero_pg.get_return_checker()
    assert type(rc) is BOPromptGeneratorReturnChecker


def test_get_prompt_with_INITIALIZE_SOLUTION_task(bozero_pg):
    actual_empty_string, actual_final_prompt = bozero_pg.get_prompt(
        task=GenerationTask.INITIALIZE_SOLUTION, problem_desc=""
    )

    expected_final_prompt = f"""## Task Description\nYou will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible Bayesian Optimization algorithm, and implement it.\n## Task Instruction\nYou need to act as a computer scientist and programmer independently.\n
**as a computer scientist specialized in bayesian optimization**\n1. Analyze the minimization optimization problem.
2. Design a Bayesian Optimization algorithm that addresses the challenges of the problem. Justify your choices of techniques and hyperparameters.
3. Pseudocode: Write down the key steps of your chosen Bayesian Optimization algorithm in plain pseudocode, highlighting any novel components or adaptations.
\n**as a programmer specialized in python.**\n- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples\n
### Problem Description\n\n## Code Structure Guide
```python
from typing import Callable
from scipy.stats import qmc # If you are using QMC sampling. Otherwise or you have a better alternative, remove this line.
import numpy as np
class <AlgorithmName>:
    # add the docstring of the class here
    def __init__(self):
        # Initialize optimizer settings
        # Configure acquisition function
        # Do not add any other arguments without a default value

    def _sample_points(self, n_points) -> np.ndarray:
        # sample points
        # return array of shape (n_points, n_dims)
    
    def _fit_model(self, X, y):
        # Fit and tune surrogate model 
        # return  the model

    def _get_model_loss(self, model, X, y) -> np.float64:
        # Calculate the loss of the model
        # return the loss of the model
    
    def _acquisition_function(self, X) -> np.ndarray:
        # Implement acquisition function 
        # calculate the acquisition function value for each point in X
        # return array of shape (n_points, 1)

    def _select_next_points(self, batch_size) -> np.ndarray:
        # Implement the strategy to select the next points to evaluate
        # return array of shape (batch_size, n_dims)
    
    def optimize(self, objective_fn:Callable[[np.ndarray], np.ndarray], bounds:np.ndarray, budget:int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, str], int]:
        # Main minimize optimization loop
        # objective_fn: Callable[[np.ndarray], np.ndarray], takes array of shape (n_points, n_dims) and returns array of shape (n_points, 1).
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        # Do not change the function signature
        # Evaluate the model using the metric you choose and record the value as model_loss after each training. the size of the model_loss should be equal to the number of iterations plus one for the fit on initial points.
        # Return a tuple (all_y, all_x, (model_losses, loss_name), n_initial_points)
        
        n_initial_points = <your_strategy>
        rest_of_budget = budget - n_initial_points
        while rest_of_budget > 0:
           # Optimization
           
           rest_of_budget -= <the number of points evaluated by objective_fn in this iteration, e.g. x.shape[0] if x is an array>

    ## You are free to add additional methods as needed and modify the existing ones except for the optimize method and __init__ method.
    ## Rename the class based on the characteristics of the algorithm as '<anyName>BO'
    
```\n

## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- problem analysis
- the design of the algorithm
### /Description

### Pseudocode
### /Pseudocode


### Code
```
<Algorithm Implementation> 
```
### /Code\n
"""
    assert actual_empty_string == ""
    assert expected_final_prompt == actual_final_prompt


def test_empty_get_prompt_with_non_INITIALIZE_SOLUTION_task(bozero_pg):
    actual_empty_string, actual_final_prompt = bozero_pg.get_prompt(
        task=GenerationTask.FIX_ERRORS, problem_desc=""
    )
    assert actual_empty_string == ""
    assert actual_final_prompt == ""


def test_get_prompt_with_non_INITIALIZE_SOLUTION_task(bozero_pg):
    c1 = BoZeroResponseHandler()
    c1.eval_result = EvaluatorResult()
    c1.eval_result.error = ""
    c1.code = ""
    c2 = BoZeroResponseHandler()
    c2.eval_result = EvaluatorResult()
    c2.eval_result.error = ""
    c2.code = ""
    actual_empty_string, actual_final_prompt = bozero_pg.get_prompt(
        task=GenerationTask.FIX_ERRORS, problem_desc="", candidates=[c1, c2]
    )
    expected_final_prompt = f"""## Task Description\nYou will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution.
## Task Instruction\n\n**as a programmer specialized in python.**\n- Identify the cause of the previous errors.
- Review all the code for potential errors. Here, only make most confident guesses.
- Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Correct the errors based on the identified causes and proposed solutions
- as an expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n
- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. \n
### Errors\n```bash\n\n```\n### Solution\n```python\n\n```\n
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Description
- Identified Errors
- Proposed Solutions
### /Description



### Code
```
<Corrected Code>
```
### /Code\n
"""
    assert actual_empty_string == ""
    assert actual_final_prompt == expected_final_prompt
