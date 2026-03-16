import pytest
import yaml
from llamevol.evaluator import EvaluatorResult
from llamevol.prompt_generators import MultiObjectivePromptGenerator
from llamevol.prompt_generators.bl_prompt_generator import (
    BaselinePromptGenerator,
    BaselineResponseHandler,
)
from llamevol import directories


@pytest.fixture
def bl_task_description(vanilla_bl):
    fn = directories.test_data(filename="bl_task_description.yaml")

    with open(fn, "r") as f:
        d = yaml.safe_load(f)
    return d


@pytest.fixture
def bl_response_format():
    fn = directories.test_data(filename="bl_response_format.yaml")

    with open(fn, "r") as f:
        d = yaml.safe_load(f)
    return d


@pytest.fixture
def bl_code_structure():  # normalizing these prompts won't work on tests
    fn = directories.test_data(filename="bl_code_structure.yaml")

    with open(fn, "r") as f:
        d = yaml.safe_load(f)
    return d


@pytest.fixture
def bl_evaluation_feedback_prompt():
    fn = directories.test_data(filename="bl_evaluation_feedback_prompt.yaml")

    with open(fn, "r") as f:
        d = yaml.safe_load(f)
    return d


@pytest.fixture
def bl_get_prompt():
    fn = directories.test_data(filename="bl_prompt_prompt.yaml")

    with open(fn, "r") as f:
        d = yaml.safe_load(f)
    return d


@pytest.fixture
def vanilla_bl_config_dict():
    with open(
        directories.test_data(filename="bl_prompt_generator_for_tests.yaml"), "r"
    ) as file:
        return yaml.safe_load(file)


@pytest.fixture
def vanilla_bl_bo_with_gpu(vanilla_bl_config_dict):
    vanilla_bl_config_dict["use_cuda"] = True
    vanilla_bl_config_dict["is_bo"] = True
    prompt = BaselinePromptGenerator(vanilla_bl_config_dict)
    return prompt


@pytest.fixture
def vanilla_bl(vanilla_bl_config_dict):
    prompt = BaselinePromptGenerator(vanilla_bl_config_dict)
    return prompt


@pytest.fixture
def vanilla_bl_bo(vanilla_bl_config_dict):
    vanilla_bl_config_dict["is_bo"] = True
    prompt = BaselinePromptGenerator(vanilla_bl_config_dict)
    return prompt


@pytest.fixture
def vanilla_blrh():
    rh = BaselineResponseHandler()
    return rh


@pytest.fixture
def eval_res():
    er = EvaluatorResult()
    return er


@pytest.fixture
def expected_response_handler_prompts():
    with open(
        directories.test_data(filename="expected_response_handler_prompts.yaml"), "r"
    ) as file:
        return yaml.safe_load(file)


@pytest.fixture
def moo_config_dict():
    with open(
        directories.test_data(filename="moo_prompt_generator_for_tests.yaml"), "r"
    ) as file:
        return yaml.safe_load(file)


@pytest.fixture
def mopg(moo_config_dict):
    pg = MultiObjectivePromptGenerator(moo_config_dict)
    return pg
