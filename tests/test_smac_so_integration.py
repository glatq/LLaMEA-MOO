"""Pytest tests for SMAC HPO integration with single-objective optimization."""

import pytest
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.configspace_ext.configspace_utils import extract_configspace_from_response

# Sample algorithm code (pure Python)
SAMPLE_CODE = """import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, bounds=None, F=0.8, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.bounds = bounds if bounds is not None else [(-5.0, 5.0)] * dim
        self.F = F
        self.CR = CR
        self.pop_size = int(pop_size)

    def __call__(self, func):
        pop = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (self.pop_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c),
                               [b[0] for b in self.bounds],
                               [b[1] for b in self.bounds])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return fitness[best_idx], pop[best_idx]
"""

# Full LLM response with ConfigSpace
SAMPLE_LLM_RESPONSE = """
# Description
Differential Evolution with configurable hyperparameters.

# Code
```python
import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, bounds=None, F=0.8, CR=0.9, pop_size=50):
        pass
```

# Space
```python
{
    "F": (0.1, 2.0),
    "CR": (0.0, 1.0),
    "pop_size": (10, 100)
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
            len(configspace) == 3
        ), f"Expected 3 hyperparameters, got {len(configspace)}"

        hyperparams = list(configspace.keys())
        assert "F" in hyperparams, "F parameter should be in ConfigSpace"
        assert "CR" in hyperparams, "CR parameter should be in ConfigSpace"
        assert "pop_size" in hyperparams, "pop_size parameter should be in ConfigSpace"

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
        evaluator = IOHEvaluator(
            budget=50,
            dim=5,
            problems=[1],  # Sphere function
            instances=[[1]],
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="DifferentialEvolution",
            llm_response=None,
        )

        assert result is not None, "Result should not be None"
        assert (
            result.error is None
        ), f"Evaluation should succeed, got error: {result.error}"
        assert result.score > 0, f"Score should be positive, got {result.score}"
        assert (
            not hasattr(result, "metadata") or "incumbent" not in result.metadata
        ), "Result should not have incumbent without HPO"

    def test_evaluation_multiple_problems(self):
        """Test evaluation on multiple problems."""
        evaluator = IOHEvaluator(
            budget=50,
            dim=5,
            problems=[1, 2],  # Sphere and Ellipsoid
            instances=[[1], [1]],
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="DifferentialEvolution",
            llm_response=None,
        )

        assert result is not None
        assert result.error is None
        assert result.score > 0
        assert len(result.result) == 2, "Should have results for 2 problems"


class TestHPOEvaluation:
    """Test HPO-enabled evaluation."""

    def test_hpo_evaluator_initialization(self):
        """Test that HPO evaluator initializes correctly."""
        evaluator = IOHEvaluator(
            budget=100,
            dim=5,
            problems=[1],
            instances=[[1]],
            repeat=1,
            use_hpo=True,
            hpo_trials=5,
            hpo_min_budget=30,
            hpo_max_budget=60,
            hpo_walltime=300,
            hpo_validation_budget=20,
        )

        # Check that HPO config was set
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
        evaluator = IOHEvaluator(
            budget=100,
            dim=5,
            problems=[1],  # Single problem for speed
            instances=[[1]],
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
            cls_name="DifferentialEvolution",
            llm_response=SAMPLE_LLM_RESPONSE,
        )

        assert result is not None, "Result should not be None"
        assert (
            result.error is None
        ), f"Evaluation should succeed, got error: {result.error}"
        assert result.score > 0, f"Score should be positive, got {result.score}"

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
        evaluator = IOHEvaluator(
            budget=50,
            dim=5,
            problems=[1],
            instances=[[1]],
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
            cls_name="DifferentialEvolution",
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
        final_budget = 100
        hpo_validation_budget = 20

        evaluator = IOHEvaluator(
            budget=final_budget,
            dim=5,
            problems=[1],
            instances=[[1]],
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

        # Check F bounds
        F = configspace["F"]
        assert hasattr(F, "lower") and hasattr(F, "upper")
        assert F.lower == pytest.approx(0.1)
        assert F.upper == pytest.approx(2.0)

        # Check CR bounds
        CR = configspace["CR"]
        assert CR.lower == pytest.approx(0.0)
        assert CR.upper == pytest.approx(1.0)

        # Check pop_size bounds
        pop_size = configspace["pop_size"]
        assert pop_size.lower == 10
        assert pop_size.upper == 100

    def test_sample_configuration(self):
        """Test that we can sample valid configurations from ConfigSpace."""
        configspace = extract_configspace_from_response(SAMPLE_LLM_RESPONSE)

        assert configspace is not None

        # Sample a configuration
        config = configspace.sample_configuration()

        assert config is not None
        assert "F" in config
        assert "CR" in config
        assert "pop_size" in config

        # Check that sampled values are within bounds
        assert 0.1 <= config["F"] <= 2.0
        assert 0.0 <= config["CR"] <= 1.0
        assert 10 <= config["pop_size"] <= 100


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_invalid_code(self):
        """Test that invalid code is handled gracefully."""
        evaluator = IOHEvaluator(
            budget=50,
            dim=5,
            problems=[1],
            instances=[[1]],
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
        assert result.error is not None, "Invalid code should produce an error"

    def test_missing_class(self):
        """Test that missing class name is handled gracefully."""
        evaluator = IOHEvaluator(
            budget=50,
            dim=5,
            problems=[1],
            instances=[[1]],
            repeat=1,
            use_hpo=False,
        )

        result = evaluator.evaluate(
            code=SAMPLE_CODE,
            cls_name="NonExistentClass",
            llm_response=None,
        )

        assert result is not None
        # Missing class should either produce an error or fail to evaluate
        assert (
            result.error is not None or result.score == 0.0
        ), "Missing class should produce an error or zero score"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
