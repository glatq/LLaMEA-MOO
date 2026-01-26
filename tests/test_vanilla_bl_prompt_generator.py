from llamevol.prompt_generators.abstract_prompt_generator import GenerationTask
from llamevol.prompt_generators.vanilla_bl_prompt_generator import (
    VanillaBaselineResponseHandler,
)
from .conftest import bl_get_prompt

from .utils_for_tests import normalize_prompt, make_basic_result


class TestTaskDescription:
    def test_task_description(self, vanilla_bl, bl_task_description):
        expected = bl_task_description["initialize_solution_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

        expected = bl_task_description["fix_errors_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.FIX_ERRORS)
        )
        assert expected == actual

        expected = bl_task_description["fix_errors_from_error_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
        )
        assert expected == actual

        expected = bl_task_description["optimize_performance_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual

        vanilla_bl.is_bo = True

        expected = bl_task_description["initialize_solution_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

        expected = bl_task_description["fix_errors_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.FIX_ERRORS)
        )
        assert expected == actual

        expected = bl_task_description["fix_errors_from_error_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
        )
        assert expected == actual

        expected = bl_task_description["optimize_performance_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual


class TestResponseFormat:
    def test_response_format(self, vanilla_bl, bl_response_format):
        expected = bl_response_format["initialize_solution"]
        actual = normalize_prompt(
            vanilla_bl.response_format(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

        expected = bl_response_format["fix_errors"]
        actual = normalize_prompt(vanilla_bl.response_format(GenerationTask.FIX_ERRORS))
        assert expected == actual

        expected = bl_response_format["fix_errors_from_error"]
        actual = normalize_prompt(
            vanilla_bl.response_format(GenerationTask.FIX_ERRORS_FROM_ERROR)
        )
        assert expected == actual

        expected = bl_response_format["optimize_performance"]
        actual = normalize_prompt(
            vanilla_bl.response_format(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual


class TestCodeStructure:
    def test_code_structure(self, vanilla_bl, bl_code_structure):
        vanilla_bl.is_bo = True
        vanilla_bl.use_mini_bo = True
        expected = bl_code_structure["bo_and_mini_bo"]
        actual = vanilla_bl.code_structure()
        assert expected == actual

        vanilla_bl.is_bo = True
        vanilla_bl.use_mini_bo = False
        expected = bl_code_structure["bo_and_no_mini_bo"]
        actual = vanilla_bl.code_structure()
        assert expected == actual

        vanilla_bl.is_bo = False
        vanilla_bl.use_mini_bo = False
        expected = bl_code_structure["no_bo_no_mini_bo"]
        actual = vanilla_bl.code_structure()
        assert expected == actual


def test_evaluation_feedback_prompt_exact_string(
    vanilla_bl, eval_res, bl_evaluation_feedback_prompt
):
    # Populate the provided EvaluatorResult
    eval_res.name = "TestAlgo"
    eval_res.total_execution_time = 12.3456
    eval_res.result = [
        make_basic_result("1-1-1", 1.0),
        make_basic_result("6-1-1", 0.5),
        make_basic_result("20-1-1", 0.0),
    ]

    expected = bl_evaluation_feedback_prompt["example"]
    actual = vanilla_bl.evaluation_feedback_prompt(eval_res)
    assert actual == expected


def test_evaluation_feedback_prompt_empty_when_no_results(vanilla_bl, eval_res):
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


class TestGetPrompt:
    def test_get_prompt_with_initialize_solution_no_candidates(
        self, vanilla_bl, vanilla_blrh, bl_get_prompt
    ):
        expected_role_prompt = bl_get_prompt["role_prompt"]

        expected_final_prompt = bl_get_prompt["initialize_solution"]
        actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
            task=GenerationTask.INITIALIZE_SOLUTION,
            problem_desc="24 noiseless functions",
            candidates=[],
        )

        assert expected_role_prompt == actual_role_prompt
        assert expected_final_prompt == actual_final_prompt

    def test_get_prompt_with_initialize_solution_and_pre_solution_candidates(
        self, vanilla_bl, vanilla_blrh, bl_get_prompt
    ):
        c1 = vanilla_blrh
        c2 = vanilla_blrh

        c1.code_name = "Algorithm 1"
        c2.code_name = "Algorithm 2"

        expected_final_prompt = bl_get_prompt[
            "initialize_solution_with_pre_solution_prompt"
        ]
        actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
            task=GenerationTask.INITIALIZE_SOLUTION,
            problem_desc="24 noiseless functions",
            candidates=[c1, c2],
        )
        assert expected_final_prompt == actual_final_prompt

    def test_get_prompt_with_non_INITIALIZE_SOLUTION_task(
        self, vanilla_bl, vanilla_blrh, bl_get_prompt
    ):
        c1 = vanilla_blrh
        c1.code_name = "Algorithm 1"
        c2 = vanilla_blrh
        c2.code_name = "Algorithm 2"

        expected_role_prompt = bl_get_prompt["role_prompt"]
        expected_final_prompt = bl_get_prompt["non_initialize_solution_with_crossover"]
        actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
            task=GenerationTask.FIX_ERRORS,
            problem_desc="24 noiseless functions",
            candidates=[c1, c2],
        )
        assert expected_role_prompt == actual_role_prompt
        assert expected_final_prompt == normalize_prompt(actual_final_prompt)

        expected_final_prompt = bl_get_prompt["non_initialize_solution"]
        actual_role_prompt, actual_final_prompt = vanilla_bl.get_prompt(
            task=GenerationTask.FIX_ERRORS_FROM_ERROR,
            problem_desc="24 noiseless functions",
            candidates=[c1],
        )
        assert expected_final_prompt == normalize_prompt(actual_final_prompt)


def test_get_response_handler(vanilla_bl):
    rh = vanilla_bl.get_response_handler()
    assert type(rh) is VanillaBaselineResponseHandler
