import pytest
from llamevol.prompt_generators.moo_response_handler import MooResponseHandler
from llamevol.prompt_generators.types import GenerationTask


@pytest.fixture
def baseline_rh():
    rh = MooResponseHandler()
    return rh


@pytest.fixture
def response(expected_response_handler_prompts):
    response = expected_response_handler_prompts["response"]
    return response


def test_extract_code_from_response(
    baseline_rh, response, expected_response_handler_prompts
):
    actual_res, actual_error_str = baseline_rh.extract_from_response(response, "Code")
    expected_res = expected_response_handler_prompts["res"]
    expected_error_str = expected_response_handler_prompts["error_str"]
    assert actual_res == expected_res
    assert actual_error_str == expected_error_str


def test_extract_class_name_from_response(
    baseline_rh, response, expected_response_handler_prompts
):
    actual, _ = baseline_rh.extract_from_response(response, "class_name")
    expected = expected_response_handler_prompts["algorithm_name"]
    assert actual == expected


def test_extract_description_from_response(
    baseline_rh, response, expected_response_handler_prompts
):
    actual, actual_error = baseline_rh.extract_from_response(response, "Description")
    expected, expected_error = (
        expected_response_handler_prompts["description"],
        expected_response_handler_prompts["description_error"],
    )
    assert actual_error == expected_error
    assert actual == expected


def test_extract_justification_from_response(
    baseline_rh, response, expected_response_handler_prompts
):
    actual, actual_error = baseline_rh.extract_from_response(response, "Justification")
    expected, expected_error = (
        expected_response_handler_prompts["justification"],
        expected_response_handler_prompts["justification_error"],
    )
    assert expected_error == actual_error
    assert expected == actual


def test_extract_response(baseline_rh, response, expected_response_handler_prompts):
    expected_code = expected_response_handler_prompts["code"]
    expected_justification = expected_response_handler_prompts["justification"]
    expected_description = expected_response_handler_prompts["description"]

    baseline_rh.extract_response(response, GenerationTask.INITIALIZE_SOLUTION)
    actual_code = baseline_rh.code
    actual_justification = baseline_rh.reason
    actual_description = baseline_rh.desc
    assert actual_code == expected_code
    assert actual_description == expected_description
    assert actual_justification == expected_justification
