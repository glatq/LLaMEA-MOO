# Quick Start: SMAC HPO Integration

## TL;DR

To enable SMAC hyperparameter optimization in your runs:

1. **Edit `conf/config.yaml`:**
   ```yaml
   mo_search:
     use_hpo: true  # Change from false to true
   ```

2. **Run the script:**
   ```bash
   python run_mo_es_search.py
   ```

That's it! The LLM will now generate algorithms with ConfigSpaces, and SMAC will automatically optimize hyperparameters.

---

## What Happens When HPO is Enabled

### Without HPO (default: `use_hpo: false`)
```
LLM generates code → Evaluate with default params → Get feedback
```

### With HPO (`use_hpo: true`)
```
LLM generates code + ConfigSpace
    ↓
Extract ConfigSpace
    ↓
Validate with random config
    ↓
SMAC optimizes hyperparameters (500 trials)
    ↓
Evaluate with best config found
    ↓
Feedback includes: "Optimized hyperparameters: {n_init: 23, ...}"
```

---

## Configuration Options

### Basic (in `conf/config.yaml`)

```yaml
mo_search:
  budget: 100  # Budget for final evaluation
  repeat: 5    # Number of repeats per problem

  # HPO Settings
  use_hpo: true           # Enable HPO
  hpo_trials: 500         # Number of configs SMAC will try
  hpo_min_budget: 50      # Cheap initial trials
  hpo_max_budget: 200     # Full trials for good configs
  hpo_walltime: 3600      # Max 1 hour for HPO
```

### Quick Test (Faster HPO)

For faster testing, reduce HPO effort:

```yaml
mo_search:
  use_hpo: true
  hpo_trials: 100        # Fewer trials (faster)
  hpo_min_budget: 30
  hpo_max_budget: 100
  hpo_walltime: 600      # 10 minutes max
```

### Full Benchmarking (More Thorough)

For thorough hyperparameter search:

```yaml
mo_search:
  use_hpo: true
  hpo_trials: 1000       # More trials (slower but better)
  hpo_min_budget: 50
  hpo_max_budget: 300
  hpo_walltime: 7200     # 2 hours max
```

---

## Expected Output

When HPO is enabled, you'll see:

```
INFO: ============================================================
INFO: SMAC HPO ENABLED
INFO:   Trials: 500
INFO:   Budget range: 50-200
INFO:   Walltime: 3600s
INFO: ============================================================

INFO: HPO mode enabled for MOBOAdaptiveParEGO
INFO: ConfigSpace found with 3 hyperparameters: ['n_init', 'acquisition_type', 'n_candidates']
INFO:   Validation passed (89 evaluations)
INFO: Starting SMAC HPO...
INFO: Running SMAC optimization...

... (SMAC progress) ...

INFO: SMAC optimization completed in 245.67s
INFO: Incumbent configuration: {'n_init': 23, 'acquisition_type': 'ei', 'n_candidates': 287}
INFO: Incumbent average HV: 0.8532
INFO: Running final evaluation with config: {'n_init': 23, 'acquisition_type': 'ei', 'n_candidates': 287}
```

---

## Feedback to LLM

The LLM will now receive richer feedback:

**Before (without HPO):**
```
The algorithm MOBOAdaptive got an average Hypervolume (HV, the larger the better)
score of 0.7543 with standard deviation 0.0234.
took 45.23 seconds to run.
```

**After (with HPO):**
```
The algorithm MOBOAdaptive got an average Hypervolume (HV, the larger the better)
score of 0.8532 with standard deviation 0.0234.
took 127.45 seconds to run.
Optimized hyperparameters: {'n_init': 23, 'acquisition_type': 'ei', 'n_candidates': 287}
```

This helps the LLM understand which hyperparameter values work best and focus on improving algorithm structure.

---

## Performance Impact

**Computational Cost:**
- Validation: ~100 evaluations
- SMAC HPO: ~500 trials × avg 125 budget = ~62,500 evaluations
- Final evaluation: budget × problems × repeats (e.g., 100 × 10 × 5 = 5,000)

**Total overhead:** HPO adds ~60-120 seconds per algorithm (depending on config)

**Benefits:**
- Better algorithm performance (optimized hyperparameters)
- Fairer comparison between algorithms
- Fewer LLM iterations needed (LLM focuses on structure)

---

## Troubleshooting

### HPO Not Running?

Check the log output:
- ✅ Should see: `SMAC HPO ENABLED`
- ❌ If not, check: `use_hpo: true` in config.yaml

### ConfigSpace Not Found?

LLM didn't include ConfigSpace in response:
```
WARNING: No valid ConfigSpace found. Using default hyperparameters.
Note: ConfigSpace not found or empty
```

**Solution:** The prompts instruct the LLM to include ConfigSpace, but it may take a few iterations for the LLM to learn this format.

### SMAC Timeout?

```
WARNING: HPO failed: Timeout
```

**Solution:** Increase `hpo_walltime` or reduce `hpo_trials`

---

## Disabling HPO

To go back to the default behavior:

```yaml
mo_search:
  use_hpo: false  # Disable HPO
```

Or simply omit the HPO parameters (defaults to `false`).

---

## When to Use HPO

**Use HPO when:**
- ✅ Running multiple iterations/generations (evolutionary search)
- ✅ Benchmarking different algorithms fairly
- ✅ Hyperparameters significantly affect performance
- ✅ You have computational budget for HPO

**Skip HPO when:**
- ❌ Quick single test
- ❌ Very tight time budget
- ❌ Debugging/testing prompt changes
- ❌ Algorithms have no/few hyperparameters

---

## Example Run

```bash
# 1. Enable HPO in config
vim conf/config.yaml  # Set use_hpo: true

# 2. Run the search
python run_mo_es_search.py

# 3. Check logs for HPO output
# Look for: "SMAC HPO ENABLED" and "Optimized hyperparameters"
```

---

## More Information

See `SMAC_HPO_INTEGRATION_GUIDE.md` for:
- Detailed architecture explanation
- Advanced configuration options
- Best practices
- Full troubleshooting guide
