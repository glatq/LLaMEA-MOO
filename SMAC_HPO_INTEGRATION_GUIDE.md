# SMAC HPO Integration for LLaMEA-BO

## Overview

This integration adds **SMAC-based hyperparameter optimization (HPO)** to LLaMEA-BO, allowing the LLM to focus on algorithm structure while SMAC automatically tunes numerical hyperparameters.

## Key Idea

**Without HPO:**
- LLM must guess good hyperparameter values
- Multiple LLM iterations needed to refine parameters
- Algorithms compared with arbitrary/default parameters
- High LLM query cost

**With HPO:**
- LLM designs algorithm structure only
- LLM provides ConfigSpace defining parameter search ranges
- SMAC automatically finds optimal hyperparameters
- All algorithms compared with optimized parameters
- Lower LLM cost, better performance

## Installation

Ensure SMAC and ConfigSpace are installed:

```bash
pip install smac ConfigSpace pymoo
```

## How It Works

### 1. LLM Response Format

The LLM now generates:
- **Description**: Algorithm overview
- **Justification**: Design decisions
- **Code**: Algorithm implementation with hyperparameters
- **Space**: ConfigSpace definition (NEW!)

**Example LLM Response:**

```
# Description
Adaptive multi-objective Bayesian optimization with ParEGO-style scalarization

# Justification
Uses lightweight GP surrogate with random weight vectors for efficient
multi-objective optimization. Dynamic weighting ensures good Pareto coverage.

# Code
```python
import numpy as np

class MOBOAdaptiveParEGO:
    def __init__(self, budget, dim, bounds, n_init=20, acquisition_type="ei",
                 n_candidates=100, weight_rotation=True):
        # Fixed parameters
        self.budget = budget
        self.dim = dim
        self.bounds = bounds

        # Hyperparameters (tuned by SMAC)
        self.n_init = n_init
        self.acquisition_type = acquisition_type
        self.n_candidates = n_candidates
        self.weight_rotation = weight_rotation

    def __call__(self, func):
        # Algorithm implementation
        ...
```

# Space
```python
{
    "n_init": (5, 50),
    "acquisition_type": ["ei", "ucb", "poi"],
    "n_candidates": (50, 500),
    "weight_rotation": [True, False]
}
```
```

### 2. ConfigSpace Format

Configuration space is defined as a Python dictionary:

```python
{
    "param_name": (lower, upper),  # Continuous/integer range
    "categorical": ["opt1", "opt2", "opt3"]  # Categorical choices
}
```

**Important:** Do NOT include `budget`, `dim`, or `bounds` in the ConfigSpace - these are problem-specific and fixed.

### 3. HPO Workflow

```
┌──────────────────────────────────┐
│ LLM generates Code + ConfigSpace │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Extract ConfigSpace from response│
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Validate with random config      │
│ (quick sanity check)             │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ SMAC HPO (500 trials)            │
│ - Multi-fidelity (50-200 budget) │
│ - Instance-based training        │
│ - Find incumbent config          │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Final evaluation with incumbent  │
│ - Full budget (100-500)          │
│ - All problems & repeats         │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│ Feedback to LLM                  │
│ "HV: 0.85, params: {n_init: 23}│
└──────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec

# Define problems
problems = [
    MOOProblemSpec(name="zdt1", dim=10, n_obj=2, ref_point=[11.0, 11.0]),
    MOOProblemSpec(name="zdt2", dim=10, n_obj=2, ref_point=[11.0, 11.0]),
    MOOProblemSpec(name="dtlz2", dim=10, n_obj=3, ref_point=[2.5, 2.5, 2.5]),
]

# Create evaluator with HPO enabled
evaluator = MultiObjEvaluator(
    budget=100,
    problems=problems,
    repeat=5,
    use_hpo=True,  # Enable HPO!
    hpo_trials=500,
    hpo_min_budget=50,
    hpo_max_budget=200,
    hpo_walltime=3600,  # 1 hour
)

# Evaluate algorithm
# llm_response must contain the full LLM response including # Space section
result = evaluator.evaluate(
    code=algorithm_code,
    cls_name="MOBOAdaptiveParEGO",
    llm_response=llm_full_response,  # Contains ConfigSpace
)

# Access results
print(f"Score: {result.score}")
print(f"Incumbent: {result.metadata['incumbent']}")
print(f"Incumbent HV: {result.metadata['incumbent_hv']}")
```

### Advanced Configuration

```python
from llamevol.evaluator.smac_hpo_wrapper import SMACHPOConfig

# Custom SMAC configuration
hpo_config = SMACHPOConfig(
    n_trials=1000,           # More trials
    min_budget=30,           # Lower min budget
    max_budget=300,          # Higher max budget
    walltime_limit=7200,     # 2 hours
    n_workers=4,             # Parallel workers
    deterministic=False,
)

evaluator = MultiObjEvaluator(
    budget=500,
    problems=problems,
    repeat=10,
    use_hpo=True,
    hpo_trials=hpo_config.n_trials,
    hpo_min_budget=hpo_config.min_budget,
    hpo_max_budget=hpo_config.max_budget,
    hpo_walltime=hpo_config.walltime_limit,
)
```

## Feedback Format

With HPO enabled, the LLM receives feedback including optimized hyperparameters:

**Example:**
```
The algorithm MOBOAdaptiveParEGO got an average Hypervolume (HV, the larger
the better) score of 0.8532 with standard deviation 0.0234.
took 127.45 seconds to run.
Optimized hyperparameters: {'n_init': 23, 'acquisition_type': 'ei',
'n_candidates': 287, 'weight_rotation': True}
```

This tells the LLM:
- The performance achieved
- Which hyperparameter values worked best
- To focus on **structural improvements** in next iteration

## Error Handling

The system gracefully handles errors:

1. **ConfigSpace not found**: Falls back to default hyperparameters
   ```
   Note: ConfigSpace not found or empty
   ```

2. **Validation fails**: Skips HPO, uses defaults
   ```
   Note: Validation failed
   ```

3. **HPO fails**: Uses defaults, reports error
   ```
   Note: HPO failed: <error message>
   ```

## Files Modified

### Core Files
1. `llamevol/utils/configspace_utils.py` - ConfigSpace extraction utilities
2. `llamevol/evaluator/smac_hpo_wrapper.py` - SMAC HPO wrapper
3. `llamevol/evaluator/multiobj_evaluator.py` - HPO integration
4. `llamevol/prompt_generators/moo_prompt_generator.py` - Feedback with incumbent

### Prompt Files
1. `conf/prompts/base_moo_prompts.yaml` - Updated prompts (for future use)
2. `llamevol/prompt_generators/moo_prompt_generator.py` - Active prompts with HPO instructions

## Performance Considerations

### SMAC Overhead

**HPO adds computational cost:**
- Validation: ~100 evaluations
- SMAC HPO: ~500 trials × (50-200 budget) = ~75,000 evaluations
- Final evaluation: budget × problems × repeats

**When to use HPO:**
- ✅ Iterative algorithm improvement (evolutionary loop)
- ✅ Final benchmarking (fair comparison)
- ✅ When hyperparameters significantly affect performance
- ❌ Single quick test
- ❌ Very small budgets (<50)

### Multi-Fidelity Benefits

SMAC uses **successive halving**:
- Bad configs: tested with budget=50 only (cheap)
- Good configs: tested with budget=200 (expensive)
- Result: Much faster than testing all configs with full budget

### Instance-Based Training

SMAC learns which configs work for which problems:
- Each problem is a "instance"
- Instance features help generalization
- Result: Better configs across diverse problems

## Comparison: With vs Without HPO

| Aspect | Without HPO | With HPO |
|--------|-------------|----------|
| **LLM Focus** | Structure + parameters | Structure only |
| **Parameter Tuning** | Manual (LLM guesses) | Automatic (SMAC) |
| **LLM Queries** | Many (parameter refinement) | Fewer (structural changes) |
| **Evaluation Fairness** | Arbitrary parameters | Optimized parameters |
| **Performance** | Suboptimal | Better |
| **Cost** | High LLM cost | Lower LLM cost, some compute cost |
| **Feedback** | "Score: 0.75" | "Score: 0.85, params: {...}" |

## Example: Random Search Evolution

**Iteration 1 (without HPO):**
```python
class RandomSearchMO:
    def __init__(self, budget, dim, bounds):
        self.budget = budget
        # No hyperparameters - rigid algorithm
```
Feedback: "Score: 0.65"

**Iteration 2 (with HPO):**
```python
class ImprovedRandomSearchMO:
    def __init__(self, budget, dim, bounds, n_init=10, explore_ratio=0.3):
        self.n_init = n_init
        self.explore_ratio = explore_ratio
```
ConfigSpace: `{"n_init": (5, 50), "explore_ratio": (0.1, 0.9)}`

SMAC finds: `n_init=23, explore_ratio=0.42`

Feedback: "Score: 0.78, params: {n_init: 23, explore_ratio: 0.42}"

**Iteration 3:**
LLM sees that `explore_ratio=0.42` worked well, designs new exploration strategy...

## Troubleshooting

### ConfigSpace Not Extracted

**Problem:** `metadata['hpo_error'] = "ConfigSpace not found"`

**Solutions:**
- Check LLM response has `# Space` section
- Verify ConfigSpace is valid Python dict
- Check for syntax errors in Space section

### Validation Fails

**Problem:** `metadata['hpo_error'] = "Validation failed"`

**Solutions:**
- Check algorithm code has no syntax errors
- Verify `__init__` accepts hyperparameters as kwargs
- Test manually with random config

### SMAC Timeout

**Problem:** HPO takes too long

**Solutions:**
- Reduce `hpo_trials` (default: 500)
- Reduce `hpo_walltime` (default: 3600s)
- Increase `hpo_min_budget` (faster initial trials)

## Best Practices

1. **ConfigSpace Design:**
   - Include all meaningful hyperparameters
   - Use reasonable ranges (not too wide)
   - Consider categorical choices for discrete options

2. **Default Values:**
   - Provide sensible defaults in `__init__`
   - SMAC uses defaults as starting point

3. **Budget Allocation:**
   - HPO budget: 30-50% of final budget
   - Validation budget: ~100 evaluations
   - Final budget: Full budget for benchmarking

4. **Feedback Interpretation:**
   - LLM should focus on structural changes
   - Let SMAC handle parameter fine-tuning
   - Use incumbent values as hints for design

## References

- SMAC3: https://automl.github.io/SMAC3/
- ConfigSpace: https://automl.github.io/ConfigSpace/
- LLaMEA Paper: "In-the-loop Hyper-Parameter Optimization for LLM-Based Automated Design of Heuristics"
