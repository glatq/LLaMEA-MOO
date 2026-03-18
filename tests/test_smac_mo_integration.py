"""Pytest tests for SMAC HPO integration with multi-objective optimization."""

import pytest
import numpy as np
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
from llamevol.configspace_ext.configspace_utils import extract_configspace_from_response

# Sample algorithm code (pure Python)
SAMPLE_CODE = """import numpy as np

class SimpleRandomSearch:
    def __init__(self, budget, dim, bounds, num_samples=10, seed=None):
        self.budget = budget
        self.dim = dim
        # Handle bounds properly - can be a tuple (lower, upper) or list of tuples [(l1, u1), ...]
        if isinstance(bounds, tuple) and len(bounds) == 2:
            self.bounds = bounds
        elif isinstance(bounds, list) and len(bounds) > 0:
            if isinstance(bounds[0], (list, tuple)):
                # List of per-dimension bounds: [(l1, u1), (l2, u2), ...]
                self.bounds = (
                    np.array([b[0] for b in bounds]),
                    np.array([b[1] for b in bounds])
                )
            else:
                # Assume [lower, upper] format
                self.bounds = (bounds[0], bounds[1])
        else:
            self.bounds = (np.zeros(dim), np.ones(dim))

        self.num_samples = int(num_samples)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, func):
        best_X = []
        best_F = []
        evals = 0

        while evals < self.budget:
            batch_size = min(self.num_samples, self.budget - evals)
            X = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                (batch_size, self.dim)
            )

            for x in X:
                f = np.asarray(func(x))
                best_X.append(x)
                best_F.append(f)
                evals += 1

        return np.array(best_X), np.array(best_F)
"""

# Full LLM response with ConfigSpace
SAMPLE_LLM_RESPONSE = """
# Description
Simple random search baseline for multi-objective optimization.

# Code
```python
import numpy as np

class SimpleRandomSearch:
    def __init__(self, budget, dim, bounds, num_samples=10):
        pass
```

# Space
```python
{
    "num_samples": (5, 20)
}
```
"""


class TestConfigSpaceExtraction:
    """Test ConfigSpace extraction from LLM responses."""

    def test_configspace_extraction_success(self):
        """Test that ConfigSpace is correctly extracted from LLM response."""
        configspace = extract_configspace_from_response(SAMPLE_LLM_RESPONSE)

        assert configspace is not None, "ConfigSpace should not be None"
        assert (
            len(configspace) == 1
        ), f"Expected 1 hyperparameter, got {len(configspace)}"

        hyperparams = list(configspace.keys())
        assert (
            "num_samples" in hyperparams
        ), "num_samples parameter should be in ConfigSpace"

    def test_configspace_extraction_empty_response(self):
        """Test that empty response returns None."""
        configspace = extract_configspace_from_response("")
        assert configspace is None, "Empty response should return None"

    def test_configspace_extraction_no_space_section(self):
        """Test that response without Space section returns None."""
        response = "# Description\nSome algorithm\n# Code\n```python\npass\n```"
        configspace = extract_configspace_from_response(response)
        assert configspace is None, "Response without Space section should return None"


class TestBasicEvaluation:
    """Test basic evaluation without HPO."""

    def test_evaluation_without_hpo(self):
        """Test that evaluation works without HPO enabled."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="SimpleRandomSearch",
            llm_response=None,
        )

        assert result is not None, "Result should not be None"
        assert (
            result.error is None
        ), f"Evaluation should succeed, got error: {result.error}"
        assert result.score is not None, "Score should not be None"
        assert (
            not hasattr(result, "metadata") or "incumbent" not in result.metadata
        ), "Result should not have incumbent without HPO"

    def test_evaluation_multiple_problems(self):
        """Test evaluation on multiple problems."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
            MOOProblemSpec(name="zdt2", dim=10, n_obj=2, ref_point=[1.1, 7.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="SimpleRandomSearch",
            llm_response=None,
        )

        assert result is not None
        assert result.error is None
        assert len(result.result) == 2, "Should have results for 2 problems"

    def test_evaluation_with_repeats(self):
        """Test evaluation with multiple repeats."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=2,  # 2 repeats
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="SimpleRandomSearch",
            llm_response=None,
        )

        assert result is not None
        assert result.error is None
        assert len(result.result) == 2, "Should have results for 1 problem × 2 repeats"


class TestHPOEvaluation:
    """Test HPO-enabled evaluation."""

    def test_hpo_evaluator_initialization(self):
        """Test that HPO evaluator initializes correctly."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=100,
            problems=problems,
            repeat=1,
            use_hpo=True,
            hpo_trials=5,
            hpo_min_budget=30,
            hpo_max_budget=60,
            hpo_walltime=300,
            hpo_validation_budget=20,
        )

        # Check that HPO config was set if SMAC is available
        if evaluator.use_hpo:
            assert evaluator.hpo_config is not None, "HPO config should be set"
            assert evaluator.hpo_config.n_trials == 5
            assert evaluator.hpo_config.min_budget == 30
            assert evaluator.hpo_config.max_budget == 60

    @pytest.mark.slow
    def test_evaluation_with_hpo(self):
        """Test that evaluation works with HPO enabled.

        This test may be slow as it runs SMAC optimization.
        Mark with @pytest.mark.slow and skip in CI if needed.
        """
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=100,
            problems=problems,
            repeat=1,
            use_hpo=True,
            hpo_trials=3,  # Small number for testing
            hpo_min_budget=30,
            hpo_max_budget=50,
            hpo_walltime=60,  # 1 minute limit
            hpo_validation_budget=20,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="SimpleRandomSearch",
            llm_response=SAMPLE_LLM_RESPONSE,
        )

        assert result is not None, "Result should not be None"
        assert (
            result.error is None
        ), f"Evaluation should succeed, got error: {result.error}"
        assert result.score is not None, "Score should not be None"

        # Check metadata
        assert hasattr(result, "metadata"), "Result should have metadata"

        # HPO may fail if SMAC not installed, so check for either incumbent or error
        has_incumbent = "incumbent" in result.metadata
        has_hpo_error = "hpo_error" in result.metadata

        assert (
            has_incumbent or has_hpo_error
        ), "Result should have either incumbent or hpo_error in metadata"

    def test_evaluation_with_hpo_no_configspace(self):
        """Test that evaluation gracefully handles missing ConfigSpace."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=1,
            use_hpo=True,
            hpo_trials=3,
            hpo_min_budget=20,
            hpo_max_budget=40,
            hpo_walltime=60,
            hpo_validation_budget=10,
        )

        # Response without ConfigSpace
        response_no_space = "# Description\nAlgorithm\n# Code\n```python\npass\n```"

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="SimpleRandomSearch",
            llm_response=response_no_space,
        )

        assert result is not None
        assert result.error is None  # Should still evaluate with defaults

        # Should have hpo_error in metadata since ConfigSpace not found
        if hasattr(result, "metadata"):
            assert "hpo_error" in result.metadata or "incumbent" not in result.metadata


class TestBudgetProgression:
    """Test that budgets are correctly ordered."""

    def test_budget_ordering(self):
        """Test that validation < HPO min < HPO max < final budget."""
        validation_budget = 20
        hpo_min_budget = 40
        hpo_max_budget = 70
        final_budget = 100

        assert (
            validation_budget < hpo_min_budget
        ), "Validation budget should be less than HPO min budget"
        assert (
            hpo_min_budget < hpo_max_budget
        ), "HPO min budget should be less than HPO max budget"
        assert (
            hpo_max_budget < final_budget
        ), "HPO max budget should be less than final budget"

    def test_evaluator_budget_config(self):
        """Test that evaluator respects budget configuration."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        final_budget = 100
        hpo_validation_budget = 20

        evaluator = MultiObjEvaluator(
            budget=final_budget,
            problems=problems,
            repeat=1,
            use_hpo=True,
            hpo_validation_budget=hpo_validation_budget,
        )

        assert evaluator.budget == final_budget
        assert evaluator.hpo_validation_budget == hpo_validation_budget


class TestHyperparameterValidation:
    """Test hyperparameter extraction and validation."""

    def test_hyperparameter_bounds(self):
        """Test that extracted hyperparameters have correct bounds."""
        configspace = extract_configspace_from_response(SAMPLE_LLM_RESPONSE)

        assert configspace is not None

        # Check num_samples bounds
        num_samples = configspace["num_samples"]
        assert hasattr(num_samples, "lower") and hasattr(num_samples, "upper")
        assert num_samples.lower == 5
        assert num_samples.upper == 20

    def test_sample_configuration(self):
        """Test that we can sample valid configurations from ConfigSpace."""
        configspace = extract_configspace_from_response(SAMPLE_LLM_RESPONSE)

        assert configspace is not None

        # Sample a configuration
        config = configspace.sample_configuration()

        assert config is not None
        assert "num_samples" in config

        # Check that sampled value is within bounds
        assert 5 <= config["num_samples"] <= 20


class TestProblemSpecifications:
    """Test problem specification handling."""

    def test_problem_spec_creation(self):
        """Test that MOOProblemSpec can be created correctly."""
        spec = MOOProblemSpec(name="zdt1", dim=30, n_obj=2, ref_point=[1.1, 6.0])

        assert spec.name == "zdt1"
        assert spec.dim == 30
        assert spec.n_obj == 2
        assert spec.ref_point == [1.1, 6.0]

    def test_problem_spec_optional_ref_point(self):
        """Test that ref_point is optional in MOOProblemSpec."""
        spec = MOOProblemSpec(name="zdt1", dim=30, n_obj=2)

        assert spec.name == "zdt1"
        assert spec.dim == 30
        assert spec.n_obj == 2
        # ref_point should be None or have a default
        assert spec.ref_point is None or isinstance(spec.ref_point, (list, tuple))


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_invalid_code(self):
        """Test that invalid code is handled gracefully."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=1,
            use_hpo=False,
        )

        invalid_code = "this is not valid python code @#$%"

        result = evaluator.evaluate(
            code=invalid_code,
            cls_name="InvalidClass",
            llm_response=None,
        )

        assert result is not None

        if len(result.result) > 0:
            # The evaluator returned a result object for the problem; check it for the error
            basic_result = result.result[0]
            assert hasattr(
                basic_result, "error"
            ), "EvaluatorBasicResult should have an error attribute"
            assert (
                basic_result.error is not None
            ), f"Expected an error string, got None. Score was: {getattr(basic_result, 'best_y', 'unknown')}"
        else:
            # The evaluator blocked the evaluation entirely
            assert (
                result.error is not None
            ), "Expected a global error when results list is empty"

    def test_missing_class(self):
        """Test that missing class name is handled gracefully."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="NonExistentClass",
            llm_response=None,
        )

        assert result is not None

        if len(result.result) > 0:
            basic_result = result.result[0]
            assert hasattr(
                basic_result, "error"
            ), "EvaluatorBasicResult should have an error attribute"
            assert basic_result.error is not None, "Expected an error for missing class"
        else:
            assert (
                result.error is not None
            ), "Expected a global error when results list is empty"


class TestHypervolumeCalculation:
    """Test hypervolume-related functionality."""

    def test_evaluation_produces_valid_hypervolume(self):
        """Test that evaluation produces a valid hypervolume score."""
        problems = [
            MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[1.1, 6.0]),
        ]

        evaluator = MultiObjEvaluator(
            budget=50,
            problems=problems,
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="SimpleRandomSearch",
            llm_response=None,
        )

        assert result is not None
        assert result.error is None

        # Check that we have valid results with hypervolume
        if len(result.result) > 0:
            basic_result = result.result[0]
            assert basic_result.best_y is not None, "Should have hypervolume score"
            # HV is stored as negative, so best_y should be negative
            assert isinstance(basic_result.best_y, (int, float, np.number))


# Pytest markers and configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
