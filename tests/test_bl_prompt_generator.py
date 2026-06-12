from llamevol.prompt_generators.response_handler import ResponseHandler
from llamevol.prompt_generators.types import GenerationTask
from .conftest import bl_get_prompt

from .utils_for_tests import normalize_prompt, make_basic_result


class TestTaskDescription:
    def test_initialize_solution_no_bo(self, vanilla_bl, bl_task_description):
        expected = bl_task_description["initialize_solution_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

    def test_fix_errors_no_bo(self, vanilla_bl, bl_task_description):
        expected = bl_task_description["fix_errors_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.FIX_ERRORS)
        )
        assert expected == actual

    def test_fix_errors_from_error_no_bo(self, vanilla_bl, bl_task_description):
        expected = bl_task_description["fix_errors_from_error_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
        )
        assert expected == actual

    def test_optimize_performance_no_bo(self, vanilla_bl, bl_task_description):
        expected = bl_task_description["optimize_performance_no_bo"]
        actual = normalize_prompt(
            vanilla_bl.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual

    def test_initialize_solution_bo(self, vanilla_bl_bo_cpu, bl_task_description):
        expected = bl_task_description["initialize_solution_bo"]
        actual = normalize_prompt(
            vanilla_bl_bo_cpu.task_description(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

    def test_fix_errors_bo(self, vanilla_bl_bo_cpu, bl_task_description):
        expected = bl_task_description["fix_errors_bo"]
        actual = normalize_prompt(
            vanilla_bl_bo_cpu.task_description(GenerationTask.FIX_ERRORS)
        )
        assert expected == actual

    def test_fix_errors_from_error_bo(self, vanilla_bl_bo_cpu, bl_task_description):
        expected = bl_task_description["fix_errors_from_error_bo"]
        actual = normalize_prompt(
            vanilla_bl_bo_cpu.task_description(GenerationTask.FIX_ERRORS_FROM_ERROR)
        )
        assert expected == actual

    def test_optimize_performance_bo(self, vanilla_bl_bo_cpu, bl_task_description):
        expected = bl_task_description["optimize_performance_bo"]
        actual = normalize_prompt(
            vanilla_bl_bo_cpu.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual


class TestTaskDescriptionGPU:
    def test_initialize_solution_bo(self, vanilla_bl_bo_with_gpu, bl_task_description):
        expected = bl_task_description["initialize_solution_bo_with_gpu"]
        actual = normalize_prompt(
            vanilla_bl_bo_with_gpu.task_description(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

    def test_fix_errors_bo(self, vanilla_bl_bo_with_gpu, bl_task_description):
        expected = bl_task_description["fix_errors_bo_with_gpu"]
        actual = normalize_prompt(
            vanilla_bl_bo_with_gpu.task_description(GenerationTask.FIX_ERRORS)
        )
        assert expected == actual

    def test_fix_errors_from_error_bo(
        self, vanilla_bl_bo_with_gpu, bl_task_description
    ):
        expected = bl_task_description["fix_errors_from_error_bo_with_gpu"]
        actual = normalize_prompt(
            vanilla_bl_bo_with_gpu.task_description(
                GenerationTask.FIX_ERRORS_FROM_ERROR
            )
        )
        assert expected == actual

    def test_optimize_performance_bo(self, vanilla_bl_bo_with_gpu, bl_task_description):
        expected = bl_task_description["optimize_performance_bo_with_gpu"]
        actual = normalize_prompt(
            vanilla_bl_bo_with_gpu.task_description(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual


class TestResponseFormat:
    def test_initialize_solution(self, vanilla_bl, bl_response_format):
        expected = bl_response_format["initialize_solution"]
        actual = normalize_prompt(
            vanilla_bl.response_format(GenerationTask.INITIALIZE_SOLUTION)
        )
        assert expected == actual

    def test_fix_errors(self, vanilla_bl, bl_response_format):
        expected = bl_response_format["fix_errors"]
        actual = normalize_prompt(vanilla_bl.response_format(GenerationTask.FIX_ERRORS))
        assert expected == actual

    def test_fix_errors_from_error(self, vanilla_bl, bl_response_format):
        expected = bl_response_format["fix_errors_from_error"]
        actual = normalize_prompt(
            vanilla_bl.response_format(GenerationTask.FIX_ERRORS_FROM_ERROR)
        )
        assert expected == actual

    def test_optimize_performance(self, vanilla_bl, bl_response_format):
        expected = bl_response_format["optimize_performance"]
        actual = normalize_prompt(
            vanilla_bl.response_format(GenerationTask.OPTIMIZE_PERFORMANCE)
        )
        assert expected == actual


class TestCodeStructure:
    def test_bo_and_mini_bo(self, vanilla_bl_mini_bo_cpu, bl_code_structure):
        expected = bl_code_structure["bo_and_mini_bo"]
        actual = vanilla_bl_mini_bo_cpu.code_structure()
        assert expected == actual

    def test_bo_and_no_mini_bo(self, vanilla_bl_bo_cpu, bl_code_structure):
        expected = bl_code_structure["bo_and_no_mini_bo"]
        actual = vanilla_bl_bo_cpu.code_structure()
        assert expected == actual

    def test_no_bo_no_mini_bo(self, vanilla_bl, bl_code_structure):
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


class TestGetPrompt:
    def test_non_initialize_empty_solution(self, vanilla_bl):
        messages = vanilla_bl.get_prompt(
            task=GenerationTask.OPTIMIZE_PERFORMANCE,
            problem_desc="24 noiseless functions",
            candidates=[],
        )
        assert messages == []

    def test_initialize_solution_no_candidates(
        self, vanilla_bl, vanilla_blrh, bl_get_prompt
    ):
        expected_role_prompt = bl_get_prompt["role_prompt"]

        expected_final_prompt = bl_get_prompt["initialize_solution"]
        messages = vanilla_bl.get_prompt(
            task=GenerationTask.INITIALIZE_SOLUTION,
            problem_desc="24 noiseless functions",
            candidates=[],
        )

        assert expected_role_prompt == messages[0].content
        assert expected_final_prompt == messages[1].content

    def test_initialize_solution_with_candidates(
        self, vanilla_bl, vanilla_blrh, bl_get_prompt
    ):
        c1 = vanilla_blrh
        c2 = vanilla_blrh

        c1.code_name = "Algorithm 1"
        c2.code_name = "Algorithm 2"

        expected_final_prompt = bl_get_prompt[
            "initialize_solution_with_pre_solution_prompt"
        ]
        messages = vanilla_bl.get_prompt(
            task=GenerationTask.INITIALIZE_SOLUTION,
            problem_desc="24 noiseless functions",
            candidates=[c1, c2],
        )
        assert expected_final_prompt == messages[1].content

    def test_non_initialize_solution(self, vanilla_bl, vanilla_blrh, bl_get_prompt):
        c1 = vanilla_blrh
        c1.code_name = "Algorithm 1"

        expected_role_prompt = bl_get_prompt["role_prompt"]
        expected_final_prompt = bl_get_prompt["non_initialize_solution"]
        messages = vanilla_bl.get_prompt(
            task=GenerationTask.FIX_ERRORS_FROM_ERROR,
            problem_desc="24 noiseless functions",
            candidates=[c1],
        )

        assert expected_role_prompt == messages[0].content
        assert expected_final_prompt == normalize_prompt(messages[1].content)

    def test_non_initialize_solution_with_crossover(
        self, vanilla_bl, vanilla_blrh, bl_get_prompt
    ):
        c1 = vanilla_blrh
        c1.code_name = "Algorithm 1"
        c2 = vanilla_blrh
        c2.code_name = "Algorithm 2"

        expected_role_prompt = bl_get_prompt["role_prompt"]
        expected_final_prompt = bl_get_prompt["non_initialize_solution_with_crossover"]
        messages = vanilla_bl.get_prompt(
            task=GenerationTask.FIX_ERRORS,
            problem_desc="24 noiseless functions",
            candidates=[c1, c2],
        )
        assert expected_role_prompt == messages[0].content
        assert expected_final_prompt == normalize_prompt(messages[1].content)


def test_get_response_handler(vanilla_bl):
    rh = vanilla_bl.get_response_handler()
    assert type(rh) is ResponseHandler
