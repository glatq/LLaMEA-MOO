import pytest
import yaml
from llamevol.directories import test_data
from llamevol.prompt_generators.moo_prompt_generator import BaselineResponseHandler
from llamevol.prompt_generators.abstract_prompt_generator import GenerationTask


@pytest.fixture
def config():
    with open(
        test_data(filename="expected_response_handler_prompts.yaml"), "r"
    ) as file:
        return yaml.safe_load(file)


@pytest.fixture
def baseline_rh():
    rh = BaselineResponseHandler()
    return rh


@pytest.fixture
def response(config):
    response = config["response"]
    return response


def test_extract_code_from_response(baseline_rh, response, config):
    actual_res, actual_error_str = baseline_rh.extract_from_response(response, "Code")
    expected_res = config["res"]
    expected_error_str = config["error_str"]
    assert actual_res == expected_res
    assert actual_error_str == expected_error_str


def test_extract_class_name_from_response(baseline_rh, response, config):
    actual, _ = baseline_rh.extract_from_response(response, "class_name")
    expected = config["algorithm_name"]
    assert actual == expected


def test_extract_description_from_response(baseline_rh, response, config):
    actual, actual_error = baseline_rh.extract_from_response(response, "Description")
    expected, expected_error = config["description"], config["description_error"]
    assert actual_error == expected_error
    assert actual == expected


def test_extract_justification_from_response(baseline_rh, response, config):
    actual, actual_error = baseline_rh.extract_from_response(response, "Justification")
    expected, expected_error = config["justification"], config["justification_error"]
    assert expected_error == actual_error
    assert expected == actual


def test_extract_response(baseline_rh, response, config):
    expected_code = config["code"]
    expected_justification = config["justification"]
    expected_description = config["description"]

    baseline_rh.extract_response(response, GenerationTask.INITIALIZE_SOLUTION)
    actual_code = baseline_rh.code
    actual_justification = baseline_rh.reason
    actual_description = baseline_rh.desc
    assert actual_code == expected_code
    assert actual_description == expected_description
    assert actual_justification == expected_justification
