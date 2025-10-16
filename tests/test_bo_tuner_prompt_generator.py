import pytest
from llamevol.prompt_generators.abstract_prompt_generator import GenerationTask
from llamevol.prompt_generators.bo_tuner_prompt_generator import (
    TunerPromptGenerator,
    TunerResponseHandler,
)
from llamevol.evaluator.evaluator_result import EvaluatorResult, EvaluatorBasicResult


@pytest.fixture
def tuner_pg():
    return TunerPromptGenerator()


@pytest.fixture
def tuner_rh():
    return TunerResponseHandler()


@pytest.fixture
def eval_res():
    return EvaluatorResult()


def test_response_format(tuner_pg):
    # The tuner uses a different response format than the vanilla baseline
    expected = """
Give the response in the format:
## Justifications 
<Analysis of the algorithm and the feedback>
<Additional information you would like to have>
<Justifications for the changes made>
<Description of the algorithm>
## Code
<code>
"""
    actual = tuner_pg.response_format(GenerationTask.INITIALIZE_SOLUTION)
    assert expected == actual


def test_task_description(tuner_pg):
    expected = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions in a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Your task is to Improve the given algorithm's performance on BBOB test suite and maintain or lower its computational cost
1 Analyze the provided Bayesian Optimization algorithm and its feedback.
2 What additional information would you like to have to improve the algorithm? 
- Only propose the information that can be easily expressed in text as the prompt.
3 Identify the potential improvements 
- modify the existing components
- or apply new components 
- The structure of the code should be kept as much as possible. Be cautious about the big changes.
4 Justify your changes.
5 Describe the algorithm on one line.

The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.The func() can only be called as many times as the budget allows, not more. 

As an expert of numpy, scipy, scikit-learn, torch, gpytorch and botorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries. Name the class based on the characteristics of the algorithm with a template '<characteristics>BOv<version>'.
"""
    actual = tuner_pg.task_description(GenerationTask.FIX_ERRORS)
    assert expected == actual


def make_basic_result(
    id_str: str, log_aoc: float, exploitation: float = 0.0
) -> EvaluatorBasicResult:
    r = EvaluatorBasicResult()
    r.id = id_str  # expected to be "fid-iid-dim" or similar
    r.log_y_aoc = log_aoc
    return r


def test_evaluation_feedback_prompt_empty_when_no_results(tuner_pg, eval_res):
    assert tuner_pg.evaluation_feedback_prompt(eval_res) == ""


def test_evaluation_feedback_prompt_includes_stats_when_present(tuner_pg, eval_res):
    eval_res.name = "AlgoX"
    eval_res.total_execution_time = 1.23
    eval_res.result = [
        make_basic_result("1-1-1", 0.0),  # separable
        make_basic_result("6-1-1", 0.5),  # low/mod cond
        make_basic_result("20-1-1", 1.0),  # high cond & unimodal
    ]

    actual = tuner_pg.evaluation_feedback_prompt(eval_res)

    detailed_feedback = f"""
on Separable functions 0.0000 of AOC, nan of exploitation
on functions with low or moderate conditioning 0.5000 of AOC, nan of exploitation
on functions with high conditioning and unimodal 0.0000 of AOC, 0.0000 of exploitation
on Multi-modal functions with adequate global structure 0.0000 of AOC, 0.0000 of exploitation
on Multi-modal functions with weak global structure 1.0000 of AOC, nan of exploitation
"""

    expected = f"""The algorithm AlgoX got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.5000 with standard deviation 0.4082\nThe average exploitation score (1.0 mean most exploitative, 0.0 mean most explorative) is nan with standard deviation nan.\n{detailed_feedback}
"""
    assert expected == actual


def test_get_prompt_raises_without_candidates(tuner_pg):
    with pytest.raises(ValueError):
        tuner_pg.get_prompt(
            task=GenerationTask.FIX_ERRORS,
            problem_desc="24 noiseless functions",
            candidates=[],
        )


@pytest.mark.xfail(reason="Test is WIP")
def test_get_prompt_with_candidates_includes_inputs(tuner_pg):
    # Build two candidates
    c1 = TunerResponseHandler()
    c1.code_name = "Algorithm A"

    c2 = TunerResponseHandler()
    c2.code_name = "Algorithm B"

    expected_role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

    task_prompt = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions in a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Your task is to Improve the given algorithm's performance on BBOB test suite and maintain or lower its computational cost
1 Analyze the provided Bayesian Optimization algorithm and its feedback.
2 What additional information would you like to have to improve the algorithm? 
- Only propose the information that can be easily expressed in text as the prompt.
3 Identify the potential improvements 
- modify the existing components
- or apply new components 
- The structure of the code should be kept as much as possible. Be cautious about the big changes.
4 Justify your changes.
5 Describe the algorithm on one line.

The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.The func() can only be called as many times as the budget allows, not more. 

As an expert of numpy, scipy, scikit-learn, torch, gpytorch and botorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries. Name the class based on the characteristics of the algorithm with a template '<characteristics>BOv<version>'.
"""

    pre_solution_prompt = "The provided Bayesian Optimization algorithm is as follows:\n## Algorithm A\n\nWith code:\n```python\n\n```\n\n\n## Algorithm B\n\nWith code:\n```python\n\n```\n\n\n\n"

    other_solutions_prompt = ""

    response_format_prompt = """
Give the response in the format:
## Justifications 
<Analysis of the algorithm and the feedback>
<Additional information you would like to have>
<Justifications for the changes made>
<Description of the algorithm>
## Code
<code>
"""

    expected_final_prompt = f"""{task_prompt}

{pre_solution_prompt}

{other_solutions_prompt}

{response_format_prompt}
"""

    actual_role_prompt, actual_final_prompt = tuner_pg.get_prompt(
        task=GenerationTask.FIX_ERRORS,
        problem_desc="24 noiseless functions",
        candidates=[c1, c2],
        population=None,
    )

    assert expected_role_prompt == actual_role_prompt
    assert expected_final_prompt == actual_final_prompt
    assert False


def test_get_response_handler_type(tuner_pg):
    rh = tuner_pg.get_response_handler()
    assert type(rh) is TunerResponseHandler
