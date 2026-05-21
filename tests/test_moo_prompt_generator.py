import pytest
import yaml
import numpy as np
from llamevol import directories
from llamevol.prompt_generators.moo_response_handler import MooResponseHandler
from llamevol.prompt_generators.types import GenerationTask
from llamevol.evaluator.evaluator_result import EvaluatorBasicResult
from tests.conftest import mopg
from tests.utils_for_tests import normalize_prompt


@pytest.fixture
def expected_prompts():
    with open(directories.test_data(filename="expected_prompts.yaml"), "r") as file:
        return yaml.safe_load(file)


def test_task_description(mopg, expected_prompts):
    expected = expected_prompts["task_prompt"]
    actual = mopg.task_description(task=GenerationTask.INITIALIZE_SOLUTION)
    assert actual == expected


def test_response_format(mopg, expected_prompts):
    expected = expected_prompts["response_format"]
    actual = mopg.response_format(task=GenerationTask.INITIALIZE_SOLUTION)
    assert actual == expected


def test_code_structure(mopg, expected_prompts):
    expected = expected_prompts["code_structure"]
    actual = mopg.code_structure()
    assert expected == actual


def test_bo_code_structure(mopg_bo_cpu, expected_prompts):
    expected = expected_prompts["bo_code_structure"]
    actual = mopg_bo_cpu.code_structure()
    assert expected == actual


def test_mini_bo_code_structure(mopg_mini_bo_cpu, expected_prompts):
    expected = expected_prompts["mini_bo_code_structure"]
    actual = mopg_mini_bo_cpu.code_structure()
    assert expected == actual


def test_response_handler(mopg, expected_prompts):
    rh = mopg.get_response_handler()
    assert type(rh) is MooResponseHandler


def make_basic_result(id_str: str, hv: float) -> EvaluatorBasicResult:
    r = EvaluatorBasicResult()
    r.id = id_str
    r.best_y = hv
    return r


def test_evaluation_feedback_prompt(mopg, eval_res, expected_prompts):
    eval_res.name = "Test"
    eval_res.total_execution_time = 12.3456
    eval_res.result = [
        make_basic_result(
            "zdt1-1-10",
            1.0,
        ),
        make_basic_result("zdt2-2-10", 2.0),
        make_basic_result("bnh-3-10", 3.0),
    ]
    actual = mopg.evaluation_feedback_prompt(eval_res)
    expected = f"""The algorithm Test got an average Hypervolume (HV, the larger the better) score of 2.0000 with standard deviation {np.std([-1.0,-2.0,-3.0]):0.4f}.\nTook 12.35 seconds to run."""

    assert actual == expected


def test_empty_results_feedback_prompt(mopg, eval_res, expected_prompts):
    actual = mopg.evaluation_feedback_prompt(eval_res)
    expected = ""
    assert actual == expected


def test_get_prompt(mopg, expected_prompts):
    actual_role_prompt, actual_final_prompt = mopg.get_prompt(
        task=GenerationTask.INITIALIZE_SOLUTION,
        problem_desc="pymoo library multi objective problems",
        candidates=[],
    )
    expected_role_prompt = expected_prompts["role_prompt"]

    expected_final_prompt_base = expected_prompts["final_prompt_base"]

    expected_code_structure = expected_prompts[
        "code_structure_with_initial_design_prompt"
    ]

    expected_response_format = expected_prompts["response_format"]
    # COMBINE the three components as the generator does
    expected_final_prompt = (
        expected_final_prompt_base.strip()
        + "\n\n\n"
        + expected_code_structure.strip()
        + "\n\n\n"
        + expected_response_format.strip()
    )

    assert expected_role_prompt == actual_role_prompt
    assert normalize_prompt(expected_final_prompt) == normalize_prompt(
        actual_final_prompt
    )
