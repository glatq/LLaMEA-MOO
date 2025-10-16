import pytest
from llamevol.prompt_generators.abstract_prompt_generator import GenerationTask
from llamevol.prompt_generators.bo_zeroplus_prompt_generator import (
    BoZeroPlusPromptGenerator,
    BoZeroPlusResponseHandler,
    BOPromptGeneratorReturnChecker,
)
from llamevol.evaluator.evaluator_result import EvaluatorResult, EvaluatorBasicResult


@pytest.fixture
def bozeroplus_pg():
    return BoZeroPlusPromptGenerator()


@pytest.fixture
def eval_res():
    return EvaluatorResult()


def test_surrogate_models(bozeroplus_pg):
    actual = bozeroplus_pg.surrogate_models()
    expected = [
        # Gaussian Process Based Models
        "Standard Gaussian Process (GP)",
        "Variational Sparse GP",
        "Fully Independent Training Conditional (FITC) GP",
        "Deep Gaussian Processes (Deep GPs)",
        "Heteroscedastic Gaussian Processes",
        # Tree-Based Models
        "Random Forest (RF)",
        "Tree Parzen Estimator (TPE)",
        # Neural Network Based Models
        "Bayesian Neural Networks (BNNs)",
        "Neural Networks with Attention Mechanisms",
        # Regression-Based Models
        "Support Vector Regression (SVR)",
        "Kernelized Ridge Regression",
        "Polynomial Regression",
        "Local Polynomial Regression",
        "Radial Basis Function (RBF) Networks",
        # Other Models
        "Polynomial Chaos Expansions (PCE)",
        "Gaussian Mixture Models (GMMs)",
        "B-splines",
        # Ensemble and Hybrid Approaches
        "Ensemble of Models",
        "Hybrid Models Combining Different Surrogate Models",
    ]
    assert expected == actual


def test_acquisition_functions(bozeroplus_pg):
    actual = bozeroplus_pg.acquisition_functions()
    expected = [
        "Expected Improvement (EI)",
        "Probability of Improvement (PI)",
        "Upper Confidence Bound (UCB)",
        "Thompson Sampling",
        "Entropy Search",
        "Predictive Entropy Search",
        "Information Gain",
        "Integrated Variance Reduction",
        "Max-value Entropy Search",
        "Knowledge Gradient",
        "Probability of Improvement with Gaussian Process Upper Confidence Bound (GP-UCB)",
        "Expected Improvement with Gaussian Process Upper Confidence Bound (GP-EI)",
        "Upper Confidence Bound with Gaussian Process Expected Improvement (GP-UCB-EI)",
        "Thompson Sampling with Gaussian Process Expected Improvement (GP-TS-EI)",
        "Entropy Search with Gaussian Process Expected Improvement (GP-ES-EI)",
        "Predictive Entropy Search with Gaussian Process Expected Improvement (GP-PES-EI)",
        "Information Gain with Gaussian Process Expected Improvement (GP-IG-EI)",
        "Integrated Variance Reduction with Gaussian Process Expected Improvement (GP-IVR-EI)",
        "Max-value Entropy Search with Gaussian Process Expected Improvement (GP-MES-EI)",
        "Knowledge Gradient with Gaussian Process Expected Improvement (GP-KG-EI)",
        "Trust-region-based acquisition function",
    ]
    assert expected == actual


def test_initialization_strategies(bozeroplus_pg):
    actual = bozeroplus_pg.initialization_strategies()
    expected = [
        "Uniform Sampling",
        "Latin Hypercube Sampling (LHS)",
        "Sobol Sequence",
        "Halton Sequence",
        "Faure Sequences",
        "Low-discrepancy Sampling",
        "Orthogonal Array Sampling",
        "Maximin Distance Sampling",
        "Orthogonal Latin Hypercube Sampling",
        "Quasi-Monte Carlo Sampling",
        "Domain expertise-based initialization",
    ]
    assert expected == actual


def test_other_techniques(bozeroplus_pg):
    actual = bozeroplus_pg.other_techniques()
    expected = []
    assert expected == actual


def test_task_description(bozeroplus_pg):
    actual = bozeroplus_pg.task_description(GenerationTask.INITIALIZE_SOLUTION)
    expected = """## Task Description\nYou will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible Bayesian Optimization algorithm, and implement it."""
    assert expected == actual

    actual = bozeroplus_pg.task_description(GenerationTask.FIX_ERRORS)
    expected = """## Task Description\nYou will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."""
    assert expected == actual

    actual = bozeroplus_pg.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
    expected = """## Task Description\nYou will be given a Bayesian Optimization solution with errors. Your task is to identify and correct the errors in the provided solution."""
    assert expected == actual

    actual = bozeroplus_pg.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
    expected = """## Task Description\nYou will be given a Bayesian Optimization solution with evaluation feedback, problem analysis, and other information. Your task is to optimize the performance of the solution."""
    assert expected == actual


def test_task_instruction_for_mathematician(bozeroplus_pg):
    actual = bozeroplus_pg.task_instruction_for_mathematician(
        GenerationTask.INITIALIZE_SOLUTION
    )
    expected = """\n**as a mathematician specialized in optimization**
- Identify the key characteristics of the problems relevant to optimization, not limited to its multi-modality, separability, and the location of its global minimum.
- Analyze the problem, focusing on the challenges posed by the problems for optimization algorithms. Consider aspects should be included but not limited to local optima, ruggedness, and the search space dimensionality.
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_mathematician(GenerationTask.FIX_ERRORS)
    expected = """\n**as a mathematician specialized in optimization**
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_mathematician(
        GenerationTask.FIX_ERRORS_FROM_ERROR
    )
    expected = """\n**as a mathematician specialized in optimization**
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_mathematician(
        GenerationTask.OPTIMIZE_PERFORMANCE
    )
    expected = """\n**as a mathematician specialized in optimization**
- Review the provided problem analysis on correctness and comprehensiveness.
- Propose your problem analysis. Keep it consice, clear and to the point.
"""
    assert expected == actual


def test_task_instruction_for_scientist(bozeroplus_pg):
    actual = bozeroplus_pg.task_instruction_for_scientist(
        task=GenerationTask.INITIALIZE_SOLUTION, aggressiveness=0.3
    )

    expected = """\n**as a computer scientist specialized in bayesian optimization**\n1. Based on the problem analysis, take a brainstorming session to identify the potential techniques in Bayesian Optimization that could address the challenges of the problem. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. The techniques should include but not limited to:
- Sampling Strategies
- Surrogate Models and their corresponding metrics: the options beyond Gaussian Process are encouraged.
- Acquisition Functions
- Initailization Strategies: Choose a strategy to balance the number of initial points and the number of optimization iterations based on the provided budget.
- Other Possible Techniques: Embrace the creativity and imagination.
2. Consider the options from step 1 and propose at least **three** algorithms. Here, you should just focus on the **diversity** and **performance** of the algorithms.
3. Review your options from step 2 and design a specific Bayesian Optimization algorithm based on AGGRESSIVENESS (0.0-1.0):0.30. Justify your choices in detail. 
- You can combine from less complex and more widely applicable techniques(low aggressiveness), or more advanced and specialized techniques(high aggressiveness) tailored to the specific challenges of the problem. 
- Be aware: AGGRESSIVENESS only affects the choice of techniques, not the implementation as a parameter.
4. Pseudocode: Write down the key steps of your chosen algorithm in plain and consise pseudocode, highlighting any novel components or adaptations.
"""
    assert actual == expected

    actual = bozeroplus_pg.task_instruction_for_scientist(GenerationTask.FIX_ERRORS)

    expected = """\n**as a computer scientist specialized in bayesian optimization**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to algorithm design. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention. 
4. Decide the errors which need to be fixed. justisfy your choice.
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_scientist(
        GenerationTask.FIX_ERRORS_FROM_ERROR
    )

    expected = """\n**as a computer scientist specialized in bayesian optimization**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to algorithm design. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention. 
4. Decide the errors which need to be fixed. justisfy your choice.
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_scientist(
        GenerationTask.OPTIMIZE_PERFORMANCE
    )

    expected = """\n**as a computer scientist specialized in bayesian optimization**\n1. Analyze the feedback.
- What does the feedback tell you about the algorithm's performance? Compare with the baseline.
- What are the key areas for improvement?
2. Review the previous proposed techniques, take a brainstorming session about the correctness and comprehensiveness. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. 
- Correct them if you find any errors,
- Propose new ones if you find any missing. 
- Update the proposed strategies. 
3. Based on problem analysis, feedback analysis, potential techniques and the provided solution, identify the potential improvements and propose at least **three** algorithms. Here, you focus on the **diversity** and **performance** of the algorithms.
- Instead of choosing different techniques, you could modify the existing techniques by adjusting hyperparameters
4. Considering the potential improvements and the corresponding workload required to implement them, decide the final algorithm design and provide a explanation. 
6. Pseudocode: Write down the key changes of your chosen strategy in plain and concise pseudocode. 
"""
    assert expected == actual


def test_task_instruction_for_programmer(bozeroplus_pg):
    actual = bozeroplus_pg.task_instruction_for_programmer(
        GenerationTask.INITIALIZE_SOLUTION
    )

    expected = """\n**as a programmer specialized in python.**\n- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_programmer(GenerationTask.FIX_ERRORS)
    expected = """\n**as a programmer specialized in python.**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to the implementation. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
4. Decide the errors which need to be fixed. justisfy your choice.
- The provided errors should be on the top of the list.
5. Correct the errors. 
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_programmer(
        GenerationTask.FIX_ERRORS_FROM_ERROR
    )
    expected = """\n**as a programmer specialized in python.**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to the implementation. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
4. Decide the errors which need to be fixed. justisfy your choice.
- The provided errors should be on the top of the list.
5. Correct the errors. 
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction_for_programmer(
        GenerationTask.OPTIMIZE_PERFORMANCE
    )
    expected = """\n**as a programmer specialized in python.**\n- Implement the algorithm in Python strictly following the previous code structure. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the modification.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual


def test_task_instruction(bozeroplus_pg):
    bozeroplus_pg.aggressiveness = 0.3
    actual = bozeroplus_pg.task_instruction(GenerationTask.INITIALIZE_SOLUTION)
    expected = """## Task Instruction\nYou need to act as a mathematician, computer scientist, and programmer independently.\n\n**as a mathematician specialized in optimization**
- Identify the key characteristics of the problems relevant to optimization, not limited to its multi-modality, separability, and the location of its global minimum.
- Analyze the problem, focusing on the challenges posed by the problems for optimization algorithms. Consider aspects should be included but not limited to local optima, ruggedness, and the search space dimensionality.\n
**as a computer scientist specialized in bayesian optimization**\n1. Based on the problem analysis, take a brainstorming session to identify the potential techniques in Bayesian Optimization that could address the challenges of the problem. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. The techniques should include but not limited to:
- Sampling Strategies
- Surrogate Models and their corresponding metrics: the options beyond Gaussian Process are encouraged.
- Acquisition Functions
- Initailization Strategies: Choose a strategy to balance the number of initial points and the number of optimization iterations based on the provided budget.
- Other Possible Techniques: Embrace the creativity and imagination.
2. Consider the options from step 1 and propose at least **three** algorithms. Here, you should just focus on the **diversity** and **performance** of the algorithms.
3. Review your options from step 2 and design a specific Bayesian Optimization algorithm based on AGGRESSIVENESS (0.0-1.0):0.30. Justify your choices in detail. 
- You can combine from less complex and more widely applicable techniques(low aggressiveness), or more advanced and specialized techniques(high aggressiveness) tailored to the specific challenges of the problem. 
- Be aware: AGGRESSIVENESS only affects the choice of techniques, not the implementation as a parameter.
4. Pseudocode: Write down the key steps of your chosen algorithm in plain and consise pseudocode, highlighting any novel components or adaptations.
\n**as a programmer specialized in python.**\n- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction(GenerationTask.FIX_ERRORS)
    expected = """## Task Instruction\nYou need to act as computer scientist and programmer independently.\n\n**as a computer scientist specialized in bayesian optimization**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to algorithm design. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention. 
4. Decide the errors which need to be fixed. justisfy your choice.
\n**as a programmer specialized in python.**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to the implementation. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
4. Decide the errors which need to be fixed. justisfy your choice.
- The provided errors should be on the top of the list.
5. Correct the errors. 
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction(GenerationTask.FIX_ERRORS_FROM_ERROR)
    expected = """## Task Instruction\nYou need to act as computer scientist and programmer independently.\n\n**as a computer scientist specialized in bayesian optimization**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to algorithm design. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention. 
4. Decide the errors which need to be fixed. justisfy your choice.
\n**as a programmer specialized in python.**\n1. Identify the cause of the provided errors.
2. Review the code for potential errors related to the implementation. Here, only make most confident guesses.
3. Propose solutions for the identified errors, ensuring that the proposed modifications align with the original algorithm's design and intention.
4. Decide the errors which need to be fixed. justisfy your choice.
- The provided errors should be on the top of the list.
5. Correct the errors. 
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
- Keep the algorithm class structure intact and only modify the necessary parts to fix the errors.
- Do not change the name. 
"""
    assert expected == actual

    actual = bozeroplus_pg.task_instruction(GenerationTask.OPTIMIZE_PERFORMANCE)
    expected = """## Task Instruction\nYou need to act as a mathematician, computer scientist, and programmer independently.\n\n**as a mathematician specialized in optimization**
- Review the provided problem analysis on correctness and comprehensiveness.
- Propose your problem analysis. Keep it consice, clear and to the point.
\n**as a computer scientist specialized in bayesian optimization**\n1. Analyze the feedback.
- What does the feedback tell you about the algorithm's performance? Compare with the baseline.
- What are the key areas for improvement?
2. Review the previous proposed techniques, take a brainstorming session about the correctness and comprehensiveness. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. 
- Correct them if you find any errors,
- Propose new ones if you find any missing. 
- Update the proposed strategies. 
3. Based on problem analysis, feedback analysis, potential techniques and the provided solution, identify the potential improvements and propose at least **three** algorithms. Here, you focus on the **diversity** and **performance** of the algorithms.
- Instead of choosing different techniques, you could modify the existing techniques by adjusting hyperparameters
4. Considering the potential improvements and the corresponding workload required to implement them, decide the final algorithm design and provide a explanation. 
6. Pseudocode: Write down the key changes of your chosen strategy in plain and concise pseudocode. 
\n**as a programmer specialized in python.**\n- Implement the algorithm in Python strictly following the previous code structure. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the modification.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples
"""
    assert expected == actual


def test_code_structure(bozeroplus_pg):
    actual = bozeroplus_pg.code_structure()
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


def test_response_format(bozeroplus_pg):
    actual = bozeroplus_pg.response_format(GenerationTask.INITIALIZE_SOLUTION)
    expected = """
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Problem Analysis
<Mathematical Analysis>
### /Problem Analysis

### Potential Techniques
### /Potential Techniques

### Proposed Strategies
<Proposed Strategies>
### /Proposed Strategies

### Final Algorithm Design
<Algorithm Design>
### /Final Algorithm Design

### Pseudocode
### /Pseudocode


### Code
```
<Algorithm Implementation> 
```
### /Code
"""
    assert expected == actual

    actual = bozeroplus_pg.response_format(GenerationTask.FIX_ERRORS)
    expected = """
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)
### Identified Errors
#### Algorithm design errors
    - <error>: cause, impact, original intention, solution, confidence level of the correct identification(0-10), should be fixed or not, reason of the choice
#### Implementation errors
    - <error>: cause, impact, original intention, solution, confidence level of the correct identification(0-10), should be fixed or not, reason of the choice
### /Identified Errors



### Code
```
<Corrected Code>
```
### /Code
"""
    assert expected == actual

    actual = bozeroplus_pg.response_format(GenerationTask.FIX_ERRORS_FROM_ERROR)
    expected = """
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)
### Identified Errors
#### Algorithm design errors
    - <error>: cause, impact, original intention, solution, confidence level of the correct identification(0-10), should be fixed or not, reason of the choice
#### Implementation errors
    - <error>: cause, impact, original intention, solution, confidence level of the correct identification(0-10), should be fixed or not, reason of the choice
### /Identified Errors



### Code
```
<Corrected Code>
```
### /Code
"""
    assert expected == actual

    actual = bozeroplus_pg.response_format(GenerationTask.OPTIMIZE_PERFORMANCE)
    expected = """
## Response Format('### <section_name>' and '### /<section_name>' are used to mark the start and end of each section. Do not remove them.)

### Problem Analysis
- only new problem analysis. No comment about the previous one.
### /Problem Analysis

### Feedback Analysis
### /Feedback Analysis

### Potential Techniques
### /Potential Techniques

### Improvements
### /Improvements

### Proposed Strategies
### /Proposed Strategies

### Final Algorithm Design
### /Final Algorithm Design

### Pseudocode
### /Pseudocode


### Code
```
<Optimized Code>
```
### /Code
"""
    assert expected == actual


def test_prompt_extract_keywords_from_code(bozeroplus_pg):
    actual = bozeroplus_pg.prompt_extract_keywords_from_code(code="import numpy as np")
    expected = """Extract and list up to 6 key technical components from the provided Python code implementing a Bayesian optimization algorithm.
- Focus on the core techniques and mathematical concepts used. 
- Exclude the general terms like 'BayesianOptimization', 'AcquisitionFunction', 'Minimization', etc.
- Return keywords only, separated by commas.

Code:
```python
import numpy as np
```
"""
    assert expected == actual


def test_get_response_handler(bozeroplus_pg):
    rh = bozeroplus_pg.get_response_handler()
    assert type(rh) is BoZeroPlusResponseHandler


def test_get_return_checker(bozeroplus_pg):
    rc = bozeroplus_pg.get_return_checker()
    assert type(rc) is BOPromptGeneratorReturnChecker


def test_empty_evaluation_feedback_prompt(bozeroplus_pg, eval_res):
    actual = bozeroplus_pg.evaluation_feedback_prompt(eval_res)
    expected = ""
    assert expected == actual


@pytest.mark.xfail(reason="This is a known issue.")
def test_evaluation_feedback_prompt(bozeroplus_pg, eval_res):
    eval_res.name = "test results"
    result_1 = EvaluatorBasicResult()
    result_1.name = "first result"
    result_1.optimal_value = 0.1
    result_1.budget = 1000
    result_1.best_y = 1
    result_1.y_aoc = 10
    result_2 = EvaluatorBasicResult()
    result_2.name = "second result"
    result_2.optimal_value = 0.2
    result_2.budget = 2000
    result_2.best_y = 2
    result_2.y_aoc = 20

    eval_res.result = [result_1, result_2]

    actual = bozeroplus_pg.evaluation_feedback_prompt(eval_res)

    expected = f"""### Feedback\n- Budget: 1000\n- Optimal Value\n- first result: 0.1\n- second result: 0.2\n#### test results\n##### first result\n- best y: 1.00\n- AOC for all y: 10.00\n##### second result\n- best y: 2.00\n- AOC for all y: 20.00\n#### Note:
- AOC(Area Over the Convergence Curve): a measure of the convergence speed of the algorithm, ranged between 0.0 and 1.0. A higher value is better.
- non-initial x: the x that are sampled during the optimization process, excluding the initial points.
- Budget: The maximum number(during the whole process) of the sample points which evaluated by objective_fn.
- mean and std of x: indicate exploration and exploitation in search space .
- mean and std of y: indicate the search efficiency. """
    assert expected == actual
    assert False


def test_get_prompt_in_INITIALIZE_SOLUTION_task(bozeroplus_pg):
    bozeroplus_pg.aggressiveness = 0.3
    actual_empty, actual_final_prompt = bozeroplus_pg.get_prompt(
        task=GenerationTask.INITIALIZE_SOLUTION,
        problem_desc="24 noiseless functions",
    )

    expected_empty = ""
    expected_final_prompt = """## Task Description\nYou will be given minimization optimization problems. Your tasks are to analyze the problem, design a feasible Bayesian Optimization algorithm, and implement it.\n## Task Instruction\nYou need to act as a mathematician, computer scientist, and programmer independently.\n\n**as a mathematician specialized in optimization**
- Identify the key characteristics of the problems relevant to optimization, not limited to its multi-modality, separability, and the location of its global minimum.
- Analyze the problem, focusing on the challenges posed by the problems for optimization algorithms. Consider aspects should be included but not limited to local optima, ruggedness, and the search space dimensionality.\n
**as a computer scientist specialized in bayesian optimization**\n1. Based on the problem analysis, take a brainstorming session to identify the potential techniques in Bayesian Optimization that could address the challenges of the problem. The techniques could be popularly used, state-of-the-art, or innovative but less promising. Make all techniques as diverse as possible. The techniques should include but not limited to:
- Sampling Strategies
- Surrogate Models and their corresponding metrics: the options beyond Gaussian Process are encouraged.
- Acquisition Functions
- Initailization Strategies: Choose a strategy to balance the number of initial points and the number of optimization iterations based on the provided budget.
- Other Possible Techniques: Embrace the creativity and imagination.
2. Consider the options from step 1 and propose at least **three** algorithms. Here, you should just focus on the **diversity** and **performance** of the algorithms.
3. Review your options from step 2 and design a specific Bayesian Optimization algorithm based on AGGRESSIVENESS (0.0-1.0):0.30. Justify your choices in detail. 
- You can combine from less complex and more widely applicable techniques(low aggressiveness), or more advanced and specialized techniques(high aggressiveness) tailored to the specific challenges of the problem. 
- Be aware: AGGRESSIVENESS only affects the choice of techniques, not the implementation as a parameter.
4. Pseudocode: Write down the key steps of your chosen algorithm in plain and consise pseudocode, highlighting any novel components or adaptations.
\n**as a programmer specialized in python.**\n- Name the algorithm using a descriptive name that reflects the chosen components, potentially highlighting the novel aspect of the algorithm.
- Add docstrings only to the class, not not the function. The docstring of the class should only include all the necessary techniques used in the algorithm and their corresponding parameters.
- Implement the algorithm in Python strictly following the provided code structure guide. Ensure that the implementation aligns with the pseudocode developed in the previous step, paying particular attention to the implementation of any novel methods.
- as a expert of numpy, scipy, scikit-learn, torch, GPytorch, you are allowed to use these libraries.\n- Do not use any other libraries unless they are necessary and cannot be replaced by the above libraries.\n- Code Implementation only contain the algorithm class. No usage examples\n
### Problem Description\n24 noiseless functions\n## Code Structure Guide
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

### Problem Analysis
<Mathematical Analysis>
### /Problem Analysis

### Potential Techniques
### /Potential Techniques

### Proposed Strategies
<Proposed Strategies>
### /Proposed Strategies

### Final Algorithm Design
<Algorithm Design>
### /Final Algorithm Design

### Pseudocode
### /Pseudocode


### Code
```
<Algorithm Implementation> 
```
### /Code\n
"""
    assert expected_empty == actual_empty
    assert expected_final_prompt == actual_final_prompt
