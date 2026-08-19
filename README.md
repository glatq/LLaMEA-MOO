# LLaMEA-MOBO

LLM-driven evolutionary generation of **multi-objective Bayesian optimization**
algorithms. Large language models act as the mutation and crossover operators of an
evolution strategy that writes complete MOBO algorithm implementations; each
candidate is hyperparameter-tuned with SMAC inside the evolutionary loop and scored
by normalized hypervolume on a suite of benchmark problems.

This branch, `publication_llamea_moo_unconstrained`, is the snapshot accompanying the
paper *"Large Language Model-Driven Evolutionary Generation of Multi-Objective
Bayesian Optimization Algorithms"*. It contains the nine generated algorithms, the
exact configurations used, the benchmark result logs, and the scripts that turn those
logs into every figure and table in the paper and its supplementary material.

> **Scope.** The branch also carries in-progress infrastructure for *constrained*
> multi-objective optimization — a constrained problem suite (CTP/BNH/C3-DTLZ4 and the
> real-world CRE problems), a feasibility-aware prompt generator, feasible-hypervolume
> metrics, and constraint-aware baselines. None of it is used by, or evaluated in, the
> paper above. Everything described here is unconstrained.

## Contents

**Essential**
1. [Installation](#installation)
2. [Quick start](#quick-start) — use a generated algorithm, or run the search
3. [Reproducing the paper](#reproducing-the-paper) — figures and tables, benchmarks, search
4. [The generated algorithms](#the-generated-algorithms)
5. [Repository layout](#repository-layout)

**Reference** — framework internals, extension points, and customization, in
[Part II](#part-ii-reference) below. Not needed to use the algorithms or reproduce the paper.

---

## Installation

```bash
conda env create -f environment.yml
conda activate llamevol
poetry install
```

Copy `.env.template` to `.env` and fill in the API keys you need. Only the
*generation* phase talks to an LLM; using a generated algorithm and reproducing the
benchmarks and figures do not.

Several generated algorithms use `n_jobs=-1` internally. Pin the thread pools before
running anything heavy, or nested parallelism will oversubscribe the machine:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
```

## Quick start

### Use a generated algorithm on your own problem

Every generated algorithm is a self-contained class: construct it with a budget,
dimension and bounds, then call it with your objective function. The function takes
one point of shape `(dim,)` and returns one objective vector of shape `(M,)`; the
number of objectives is inferred at runtime, so the same algorithm handles bi- and
tri-objective problems unchanged. It returns the final non-dominated set.

```python
import numpy as np, yaml

src = open("generated_algorithms/MOBOImprovedScalarizedEI.py").read()
ns = {}; exec(src, ns)
Algo = ns["MOBOImprovedScalarizedEI"]

# the SMAC-tuned hyperparameters used in the paper (omit for the LLM's defaults)
hp = yaml.safe_load(open("generated_algorithms/incumbents.yaml")) \
         ["MOBOImprovedScalarizedEI"]["init_kwargs"]

def my_objectives(x):                        # both objectives are minimized
    return np.array([np.sum(x ** 2), np.sum((x - 1.0) ** 2)])

bounds = np.array([[-5.0] * 3, [5.0] * 3])   # shape (2, dim): [lower; upper]
algo = Algo(budget=60, dim=3, bounds=bounds, **hp)
F, X = algo(my_objectives)                   # F: (K, M) objectives, X: (K, dim) points

print("Pareto front:", F.shape, "| decision vectors:", X.shape)
```

Swap the filename for any of the nine algorithms in `generated_algorithms/`, or for
`MO bench/MOBO_MOEAD_EI_Hybrid_fixed.py`. Two of the nine need an older scikit-learn —
see [Environment note](#environment-note).

### Run the evolutionary search

This generates new algorithms and requires LLM credentials.
`conf/config_paper_search.yaml` reproduces the generation phase used in the paper: a
nine-problem training suite at a 200-evaluation budget with 3 repetitions, SMAC tuning
on an evenly-spaced three-problem subset, and Gemini-2.5-Flash at temperature 0.7.

```bash
python run_es_search.py --config-name config_paper_search
```

Switch evolution strategy on the command line:

```bash
# (1+1)-ES
python run_es_search.py --config-name config_paper_search \
    mo_search.n_parent=1 mo_search.n_offspring=1
# (8,16)-ES
python run_es_search.py --config-name config_paper_search \
    mo_search.n_parent=8 mo_search.n_offspring=16 \
    mo_search.is_elitist=false mo_search.n_population=104
```

The training suite is **not** the benchmark suite: it uses different problems,
dimensions and reference points, so search fitness is not comparable to benchmark
hypervolume. Hyperparameters were tuned at 200 evaluations during the search and then
deployed at 400 in the benchmark.

## Reproducing the paper

### 1. Figures and tables from the committed logs (no re-running, seconds)

`paper_data/` holds the benchmark logs behind every reported number, so all figures and
tables regenerate directly. Run everything from the repository root:

```bash
python paper_reproduction/emit_paper_numbers.py      # Table II + supplementary tables + in-text checklist
python paper_reproduction/significance_tests.py      # Friedman, mean ranks, Holm-corrected Wilcoxon
python paper_reproduction/cd_diagram.py              # critical-difference diagram
python paper_reproduction/plot_search_progress.py    # search progress across the nine runs
python paper_reproduction/plot_grid.py               # per-phase convergence grids
python paper_reproduction/plot_time_accuracy.py      # time-accuracy trade-off
python paper_reproduction/plot_pareto_grid.py        # non-dominated fronts
python paper_reproduction/analyze_hpo_ablation.py    # SMAC-vs-defaults, three algorithms
python paper_reproduction/analyze_moead_hpo.py       # SMAC-vs-defaults, MOEAD-EI Hybrid
python paper_reproduction/trace_lineage.py           # recorded ancestry of the selected designs
```

Figures are written to `paper_figures/` (gitignored). `plot_from_csv.py` is a
general-purpose plotter for arbitrary benchmark CSVs.

### 2. Re-running the benchmarks

Three commands per suite, in order. Problems, dimensions and reference points are exactly
those reported in the paper, at 400 evaluations and 5 independent repeats.

```bash
# a. the eight stochastic algorithms of each phase
python benchmark_best_codes.py config=conf/benchmark_phase1.yaml        # 12 synthetic, generated algorithms only
python benchmark_best_codes.py config=conf/benchmark_phase2.yaml        # 12 synthetic, generated + baselines
python benchmark_best_codes.py config=conf/benchmark_phase3.yaml        # 3 real-world RE problems

# b. Improved-Scalarized-EI, which needs five explicit seeds (see below).
#    Appends its rows into the phase output dirs, so no manual merging.
python paper_reproduction/rerun_isei_seeds.py

# c. the Table III "LLM defaults" half of the HPO ablation
python benchmark_best_codes.py config=conf/benchmark_hpo_ablation.yaml
```

**Why Improved-Scalarized-EI is separate.** It is the only one of the nine generated
algorithms that takes a `random_seed` parameter (default 42) and drives its own
`np.random.default_rng`, so it is deterministic: every repeat returns an identical
hypervolume and reports zero variance. The paper's Improved-Scalarized-EI numbers — in
Phase 1, 2 and 3 alike — come from five explicit seeds, which is what step (b) does and
what Section VI-B of the paper describes. The phase configs therefore exclude it, both to
avoid ~1.5 h of wasted compute and to avoid reporting a zero-variance result. The other
eight algorithms have no seed parameter and vary normally across repeats.

**Analysing your own re-run.** The scripts in step 1 default to the reference logs in
`paper_data/`. To point them at your own output, pass the paths:

```bash
python paper_reproduction/emit_paper_numbers.py \
    stage1=benchmark_results/phase1 stage2=benchmark_results/phase2 stage3=benchmark_results/phase3
python paper_reproduction/significance_tests.py csv=benchmark_results/phase2/hv_benchmark_log.csv
python paper_reproduction/cd_diagram.py         csv=benchmark_results/phase2/hv_benchmark_log.csv
```

The plotting scripts (`plot_grid.py`, `plot_time_accuracy.py`, `plot_pareto_grid.py`) read
`paper_data/` paths written near the top of each file; edit those two or three lines to
compare your own curves.

**Appending.** `benchmark_best_codes.py` appends to the CSVs in `output_dir`,
de-duplicating on (algorithm, problem, repeat, epoch), and `rerun_isei_seeds.py` does the
same. A phase therefore does not have to be produced in one pass — the reported logs were
built across several invocations. Output goes to `benchmark_results/` (gitignored), so
re-runs never overwrite the reference logs in `paper_data/`.

Any field can be overridden on the command line, including whole lists, e.g. `repeat=1
budget=100` or `"problems=[{name: zdt1, dim: 30, n_obj: 2, ref_point: [1.1, 7.2]}]"`. The
reported runs were launched with equivalent overrides rather than these files; the configs
encode the same settings, and the problem sets and resulting numbers have been checked to
match.

**Runtime.** The `timeout` field caps wall-clock for *one algorithm across all problem ×
repeat tasks*, which run in a process pool sized to `os.cpu_count()`. Raise it on a small
host: qParEGO alone averages roughly three hours per run on the synthetic suite, so Phase 2
is the better part of a day on ten cores.

### 3. Re-running the LLaMEA search

The generation phase is the expensive part and needs LLM credentials; the commands are
in [Quick start](#run-the-evolutionary-search). The paper's nine runs are three
repetitions of each of the three evolution strategies listed there, all under
`conf/config_paper_search.yaml`. Note that the training suite is **not** the benchmark
suite — different problems, dimensions and reference points — so search fitness is not
comparable to benchmark hypervolume, and hyperparameters tuned at the 200-evaluation
search budget are deployed at 400 in the benchmark.

### Environment note

The experiments spanned a scikit-learn upgrade, and this matters for exactly two
algorithms:

* **Phases 2 and 3, the HPO ablation, and every figure and table** reproduce with the
  committed `poetry.lock` (scikit-learn 1.7.2).
* **Phase 1** additionally includes `MOBORobustLCBFPS` and `MOBOEnsembleRidge_MPFDUWS`,
  which were generated against scikit-learn 1.3.x and call
  `BaggingRegressor(base_estimator=...)`. That argument was removed in scikit-learn 1.4,
  so those two crash on the committed lock file. To re-run them, use a separate
  environment with `pip install "scikit-learn<1.4"` (use 1.3.2 or later for Python 3.12
  wheels). The other seven Phase-1 algorithms run on any supported version.

The generated algorithms are published **verbatim**, exactly as benchmarked, so the
`base_estimator` call is left in place rather than modernized.

## What is not reproducible from this repository

* **The raw generation runs.** The nine run trees total roughly 24 GB (every
  candidate's code, prompt, response and evaluation record, plus population
  checkpoints of about 76 MB each) and are not committed. `paper_data/search_history.csv`
  and `paper_data/lineage.csv` are compact exports of the parts the paper uses.
  `trace_lineage.py` accepts `runs=<dir>` if you obtain the full trees.
* **Per-evaluation objective vectors** (`objectives_log.csv`, roughly 37 MB across the
  three phases). No reported figure or table reads them; the Pareto-front figures use
  the much smaller `pareto_front_log.csv`.
* **MOEAD-EI Hybrid's lineage.** It came from a separate development run rather than
  the nine reported runs, and its population checkpoint can no longer be loaded by the
  current code. Its benchmark results reproduce normally.

## The generated algorithms

`generated_algorithms/` holds the best algorithm from each of the nine evolutionary
runs, as generated, together with `incumbents.yaml` — the SMAC incumbent
hyperparameters recorded during the run, plus the run, generation, offspring index and
search fitness that identify each individual. The benchmark configs read those
hyperparameters directly, so no pickle files are needed.

`MO bench/` holds the baselines (NSGA-II, NSGA-III, IOC-SAMO-COBRA, multi-objective
Random Search, and the BoFire qParEGO / qLogNEHVI Bayesian baselines) alongside
`MOBO_MOEAD_EI_Hybrid_fixed.py`, the development-found generated algorithm. MOEAD-EI
Hybrid is not one of the nine, so its SMAC incumbent is not in `incumbents.yaml`; it is
written inline in `conf/benchmark_phase2.yaml` and `conf/benchmark_phase3.yaml`, which is
what makes those configs reproduce the tuned Table II values (0.971 synthetic, 0.926
real-world) rather than the Table III defaults (0.956, 0.859).

Two of the benchmarked algorithms carry a one-line mechanical fix applied before
benchmarking — an index/population-size bound that otherwise crashes on the final
batch. Both are marked in the source. No algorithmic component was altered.

## Repository layout

| Path | Contents |
| --- | --- |
| `generated_algorithms/` | the nine generated algorithms + `incumbents.yaml` |
| `MO bench/` | baselines and the development-found MOEAD-EI Hybrid |
| `conf/benchmark_phase{1,2,3}.yaml` | Phase 1–3 benchmark configurations |
| `conf/benchmark_hpo_ablation.yaml` | LLM-default-hyperparameter benchmark |
| `conf/config_paper_search.yaml` | generation-phase (LLaMEA search) configuration |
| `paper_data/benchmark_stage{1,2,3}/` | Phase 1–3 hypervolume, runtime and Pareto logs |
| `paper_data/benchmark_hpo_default/` | HPO ablation, LLM-default hyperparameters |
| `paper_data/benchmark_moead_{tuned,untuned}/` | MOEAD-EI Hybrid, SMAC-tuned vs defaults |
| `paper_data/search_history.csv` | per-run search trace for all nine runs |
| `paper_data/lineage.csv` | recorded ancestry of the selected designs |
| `paper_reproduction/` | analysis and plotting scripts |
| `benchmark_results/`, `paper_figures/` | outputs of re-runs and plots (gitignored) |

---

# Part II — Reference

Everything below documents the framework internals and extension points. **None of it
is needed to use the generated algorithms or to reproduce the paper** — those are
covered in Part I above.

## Project structure

The project follows a modular structure primarily located within the `llamevol/` directory.

- **`llamevol/`**: Contains the core implementation of the LLaMEvol algorithm.
    - **`llamevol.py`**: The main class orchestrating the LLaMEvol process.
    - **`individual.py`**: Defines the `Individual` class representing a single generated algorithm/solution.
    - **`llm.py`**: Handles interactions with the Language Model (LLM).
    - **`prompt_generators/`**: Contains classes responsible for generating prompts for the LLM.
    - **`evaluator/`**: Includes code for executing and evaluating the performance of generated algorithms, often using benchmark suites like BBOB (via IOHprofiler). It handles code execution, error capture, and metric calculation.
    - **`population/`**: Manages the collection (population) of `Individual` algorithms, implementing selection strategies and diversity maintenance.
    - **`utils.py`**: Provides utility functions, including logging, serialization and plotting.
- **`paper_reproduction/`**: Analysis and plotting scripts that regenerate the paper's
  figures and tables from the logs in `paper_data/`.
- **`generated_algorithms/`**: The nine LLaMEA-generated algorithms and their SMAC
  incumbent hyperparameters.
- **`MO bench/`**: Baseline algorithms and the development-found MOEAD-EI Hybrid.
- **`conf/`**: Hydra/OmegaConf configurations for the search and the benchmark phases.

## Running the search programmatically

The example below drives the loop directly instead of through `run_es_search.py`. It is
written for the **single-objective** path (`IOHEvaluator`, `BaselinePromptGenerator`);
for the multi-objective path used in the paper, substitute `MultiObjEvaluator`
(`llamevol/evaluator/multiobj_evaluator.py`) and `MultiObjectivePromptGenerator`.

Below is a simplified example demonstrating how to set up and run the LLaMEvol evolutionary process using the provided components. This example uses an `IOHEvaluator`, a `BaselinePromptGenerator`, a `gemini-2.0-flash` model via `LLMmanager`, and an `ESPopulation`.

```python
import logging
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.prompt_generators import BaselinePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger

# Configure logging
setup_logger(level=logging.INFO)

# 1. Instantiate Evaluator (Example: IOH BBOB)
evaluator = IOHEvaluator(budget=100, dim=5, problems=[2, 4, 6], instances=[[1]]*3, repeat=3)
evaluator.timeout = 30 * 60 # Set timeout(seconds) for each evaluation(including all tasks).

# 2. Instantiate Prompt Generator
prompt_generator = BaselinePromptGenerator()
prompt_generator.is_bo = True # Specify it's for Bayesian Optimization

# 3. Instantiate LLM Manager (Example: Google Gemini)
# Ensure API key is set via environment variable or passed directly
api_key = 'YOUR_API_KEY' # Replace with your actual key or load from env
llm_manager = LLMmanager(model_name='gemini-2.0-flash', api_key=api_key, client_str='google')

# 4. Instantiate Population (Example: (1+1)-ES)
es_options = {
    'n_parent': 1,
    'n_offspring': 1,
    'is_elitist': True,
    'log_dir': 'exp_es_search', # Directory to save logs
}
population = ESPopulation(
    n_parent=es_options['n_parent'], 
    n_offspring=es_options['n_offspring'], 
    use_elitism=es_options['is_elitist']
)
population.save_dir = es_options['log_dir']
population.name = f"evol_{es_options['n_parent']}+{es_options['n_offspring']}"

# 5. Instantiate LLaMEvol orchestrator
llamevol = LLaMEvol()

# 6. Run the evolution
llm_params = {'temperature': 0.7}
llamevol.run_evolutions(
    llm=llm_manager,
    evaluator=evaluator,
    prompt_generator=prompt_generator,
    population=population,
    n_population=5,  # Maximum number of generated individuals
    options={'llm_params': llm_params}
)

# 7. Save the final population
population.save(suffix='final')

print("Evolution finished. Results saved in:", population.log_dir)
```

The example above is single-objective. For the multi-objective path used in the paper,
swap `IOHEvaluator` for `MultiObjEvaluator` (`llamevol/evaluator/multiobj_evaluator.py`)
and `BaselinePromptGenerator` for `MultiObjectivePromptGenerator`; the runnable entry
point is `run_es_search.py` with `mode=mo`, configured by
`conf/config_paper_search.yaml` (see *Reproducing the paper* above).


## LLaMEvol
The `LLaMEvol` class (`llamevol/llamevol.py`) is the central orchestrator of the evolutionary algorithm. It coordinates the interactions between the LLM, Evaluator, Prompt Generator, and Population components to drive the search for optimal algorithms.

**Structure & Features:**
- **Main Loop:** Implements the core evolutionary loop (`run_evolutions`), managing generations and population size.
- **Component Integration:** Takes instances of `LLMmanager`, `AbstractEvaluator`, `PromptGenerator`, and `Population` as inputs, delegating specific tasks to each.
- **Task Determination:** Dynamically determines the appropriate task for the LLM based on the state of the parent individuals (e.g., `INITIALIZE_SOLUTION`, `FIX_ERRORS`, `OPTIMIZE_PERFORMANCE`) using `update_current_task`.
- **LLM Interaction:** Handles querying the LLM via the `LLMmanager`, including:
    - Constructing session messages based on prompts from the `PromptGenerator`.
    - Applying LLM parameters (temperature, top_k).
    - Managing retries (`n_retry`) in case of LLM or extraction failures.
    - Optional parallel querying using `concurrent.futures.ThreadPoolExecutor` (`n_query_threads`).
- **Evaluation Trigger:** Calls the `evaluate` method of the provided `Evaluator` on the code generated by the LLM.
- **Population Update:** Updates `Individual` objects within the `Population` with the results from the LLM (code, description) and Evaluator (fitness, feedback) using `_update_ind_and_handler`.
- **Token Tracking:** Logs prompt and response token counts per generation (`LLaMEvolTokenLogItem`).
- **Progression Control:** Iterates through generations until a target population size (`n_population`) is reached.

**Usage:**
1.  **Instantiate Components:** Create instances of `LLMmanager`, `AbstractEvaluator`, `PromptGenerator`, and `Population` configured for your specific task and resources.
2.  **Instantiate LLaMEvol:** Create an instance of the `LLaMEvol` class.
    ```python
    from llamevol import LLaMEvol
    llamevol = LLaMEvol()
    ```
3.  **Run Evolution:** Call the `run_evolutions` method, passing the instantiated components and desired parameters.
    ```python
    # Assuming llm, evaluator, prompt_generator, population are already created
    llamevol.run_evolutions(
        llm=llm_manager,
        evaluator=evaluator,
        prompt_generator=prompt_generator,
        population=population,
        n_population=20, # Maximum number of individuals
        n_retry=3,
        n_query_threads=4, # Number of parallel LLM queries
        options={'llm_params': {'temperature': 0.7}}
    )
    ```
4.  **Results:** The final population (containing evolved individuals and their performance) can be accessed and saved via the `Population` object after the run completes.

**Customization:**
- **Component Swapping:** The primary way to customize `LLaMEvol`'s behavior is by providing different implementations of its core components (LLM, Evaluator, Prompt Generator, Population). For example, using a different `Population` class changes the selection and generation strategy.
- **Configuration:** Adjust parameters passed to `run_evolutions`, such as `n_population`, `n_retry`, `n_query_threads`, and LLM-specific settings within the `options` dictionary.

## LLMmanager
This module (`llamevol/llm.py`) acts as a central manager for interacting with various Large Language Models (LLMs). 

**Features:**
- Provides a unified interface (`LLMmanager`) to connect to different LLM providers (Google GenAI, OpenAI, Anthropic, OpenRouter).
- Abstracts away the specific API details for each provider.
- Manages API keys and base URLs, primarily loaded from environment variables.
- Defines a standardized response object (`LLMClientResponse`) containing the generated text, token counts, and potential errors.
- Supports different LangChain-based client implementations for Google, OpenAI, Anthropic, and OpenRouter providers.

**Usage:**
1.  **Environment Variables(Optional):** Ensure the necessary API keys and base URLs for the desired LLMs are set as environment variables (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`, etc.). Copy and rename `.env.template` to `.env` and fill in the required keys.
    ```bash
    cp .env.template .env
    # Edit .env to add your API keys
    ```
2.  **Initialization:** Create an instance of `LLMmanager` by providing a `model_key` which corresponds to an entry in the `LLMS` dictionary within the script. Alternatively, you can manually specify `model_name`, `api_key`, `base_url`, and `client_str`. The mapping of `client_str` to the actual client class is handled in the `LLMmanager` constructor.
    ```python
    from llamevol.llm import LLMmanager

    # Using a predefined model key
    llm_manager = LLMmanager(model_key='llama3-70b-8192') 

    # Or manually configuring (example)
    # llm_manager = LLMmanager(model_name='some-model', api_key='YOUR_API_KEY', base_url='https://api.example.com/v1', client_str='openai')
    ```
3.  **Chat:** Use the `chat` method, passing a list of messages in the standard OpenAI format (list of dictionaries with 'role' and 'content').
    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the theory of relativity."}
    ]
    response = llm_manager.chat(messages, temperature=0.7)

    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Response Text: {response.text}")
        print(f"Prompt Tokens: {response.prompt_token_count}")
        print(f"Response Tokens: {response.response_token_count}")
    ```

**Customization:**
- **Adding Predefined Models:** To add support for a new model using an existing provider type, add an entry to the `LLMS` dictionary in `llm.py`. You'll need:
    - the model name recognized by the API
    - the environment variable name for the API key 
    - the environment variable name for the base URL (if applicable) 
    - a maximum interval value (**Deprecated**, designed for rate limiting and retries). 
    - the client type string (`'google'`, `'vertex'`, `'openai'`, `'openrouter'`, `'request'`, or `'anthropic'`). 

- **Adding New Providers:** 
    1. Install the corresponding `langchain-<provider>` package.
    2. Import the chat model class at the top of `llm.py`.
    3. Add a branch in `_create_model()` for the new `client_str`.
    4. Add the supported kwargs to `_SUPPORTED_KWARGS`.

## Prompt Generator
This component constructs the prompts sent to the LLM for generating or modifying optimization algorithms.

**Structure & Features:**
- **Abstract Base Classes:** Defines `PromptGenerator` and `ResponseHandler` abstract classes (`abstract_prompt_generator.py`) to ensure a consistent interface.
- **Concrete Implementations:** Provides specific generators like `BaselinePromptGenerator` (for generating algorithms from scratch), `BoTunerPromptGenerator` (for refining existing algorithms). 
- **Contextual Prompts:** Dynamically builds prompts incorporating problem descriptions, existing candidate solutions (code, descriptions, past performance), evaluation feedback (errors, performance metrics like AOC), and potentially information about the broader population of algorithms.
- **Task-Specific Instructions:** Generates detailed instructions for the LLM based on the task (e.g., "design a novel algorithm", "fix the error in this code", "optimize this algorithm based on feedback").
- **Response Parsing:** Each `PromptGenerator` has a corresponding `ResponseHandler` subclass responsible for parsing the LLM's structured output (e.g., extracting code blocks, justifications, pseudocode) using methods like `extract_response`.

**Usage:**
1.  **Instantiate:** Choose and instantiate a specific `PromptGenerator` subclass.
2.  **Generate Prompt:** Call the `get_prompt` method, passing the `GenerationTask`, problem description, and any relevant context (like candidate `ResponseHandler` objects or the `Population`).
3.  **Query LLM:** Use the returned system and user prompts with the `LLMmanager`.
4.  **Parse Response:** Get the corresponding `ResponseHandler` instance using `get_response_handler()` and use its `extract_response` method on the LLM's output string.

**Customization:**
- **New Strategies:** Create new subclasses inheriting from `PromptGenerator` and `ResponseHandler`.
- **Implement Methods:** Override methods like `get_prompt`, `task_description`, `task_instruction`, `response_format`, `evaluation_feedback_prompt` in your `PromptGenerator` subclass, and `extract_response` in your `ResponseHandler` subclass to define the new prompting logic and response parsing.

## Evaluator
The Evaluator component is responsible for executing the Python code generated by the LLM and assessing its performance on optimization tasks.

**Structure & Features:**
- **Abstract Base:** Defines `AbstractEvaluator` (`evaluator.py`) for a consistent interface.
- **Concrete Implementations:** Provides evaluators for standard benchmarks:
    - `IOHEvaluator` (`ioh_evaluator.py`): Evaluates algorithms on the IOHprofiler (BBOB) benchmark suite. Supports parallel execution across multiple problem instances and repetitions.
    - `RandomBoTorchTestEvaluator` (`random_botorch_evaluator.py`): Evaluates algorithms on synthetic test functions from the BoTorch library.
- **Code Execution:** Uses utilities in `exec_utils.py` (`default_exec`) to safely execute the generated Python code, capturing standard output, errors, and execution time. It handles budget constraints via `BOOverBudgetException`.
- **Result Tracking:** Employs `EvaluatorResult` and `EvaluatorBasicResult` (`evaluator_result.py`) to store detailed outcomes for each evaluation run, including:
    - Best function value found (`best_y`).
    - History of evaluated points (`x_hist`, `y_hist`).
    - Area Over the Convergence Curve (AOC), including log-scale AOC, calculated using `ConvergenceCurveAnalyzer`.
    - Execution time and any runtime errors.
- **BO Algorithm Introspection (Optional):** Uses `BOInjector` and `AlgorithmCritic` (`bo_injector.py`) to inject monitoring code specifically into Bayesian Optimization algorithms. This allows tracking internal metrics during the optimization run, such as:
    - Surrogate model R² score (on test and training data).
    - Surrogate model uncertainty.
    - Search space coverage metrics (grid-based, clustering-based using `CoverageCluster`).
    - Exploitation vs. Exploration metrics (distance to best points, acquisition score analysis via `EvaluatorSearchResult`).
- **Parallelism:** Supports parallel evaluation using `MPI` (as seen in `IOHEvaluator`). Specifically, `MPITaskManager` provides an MPI-based master-worker framework, which can be used to distribute evaluation tasks across multiple nodes. This is particularly useful for large-scale evaluations across distributed systems.

**Usage:**
1.  **Instantiate:** Create an instance of a specific evaluator subclass (e.g., `IOHEvaluator`) with configuration like budget, dimension, and target problems/instances.
2.  **Evaluate:** Call the `evaluate` method, providing the generated Python code string and the name of the main class within that code. Optional arguments control parallelism (`max_eval_workers`) and timeouts.
3.  **Process Results:** The `evaluate` method returns an `EvaluatorResult` object. This object contains a list of `EvaluatorBasicResult` objects, each holding the detailed metrics, history, and potential errors for a single evaluation run (e.g., one IOH instance).

**Customization:**
- **New Benchmarks:** Create a new class inheriting from `AbstractEvaluator`. Implement the required methods (`evaluate`, `problem_name`, etc.). You'll likely need a wrapper for your objective function (similar to `IOHObjectiveFn`) to manage budget and history tracking.
- **New Metrics:** Extend `EvaluatorBasicResult` or `EvaluatorSearchResult` to store additional metrics. Modify the relevant evaluator or create/modify an `ExecInjector` subclass (`exec_utils.py`, `bo_injector.py`) to compute and record these metrics during or after code execution.

## Population
The Population component (`llamevol/population/`) manages the collection of candidate algorithms (`Individual` objects) throughout the evolutionary process.

**Structure & Features:**
- **Abstract Base:** Defines `Population` (`population.py`) as an abstract base class, ensuring a consistent interface for different population management strategies. It includes common utilities like saving/loading populations (using `pickle`) and calculating diversity metrics.
- **Concrete Implementations:**
    - `ESPopulation` (`es_population.py`): Implements an Evolution Strategy-style population (e.g., (μ+λ) or (μ,λ)).
        - Manages individuals across discrete generations.
        - Supports configurable parent pool size (`n_parent`), offspring count (`n_offspring`), and parents per offspring (`n_parent_per_offspring`).
        - Handles selection for the next generation, including optional elitism (`use_elitism`).
        - Implements parent selection logic based on combinations and configurable crossover/mutation rates (`cross_over_rate`, `exclusive_operations`).
        - Allows plugging in custom parent selection (`get_parent_strategy`) and survival selection (`selection_strategy`) functions.
    - `IslandESPopulation` (`island_population.py`): Implements an island model using multiple `ESPopulation` instances.
        - Manages multiple sub-populations (islands) concurrently.
        - Introduces island lifecycles (`IslandStatus`: INITIAL, GROWING, MATURE, RESETING, KILLED) and geological ages (`IslandAge`: WARMUP, CAMBRIAN, NEOGENE) to control evolution dynamics.
        - Implements migration strategies between islands during specific ages (e.g., CAMBRIAN), potentially based on fitness and diversity (using `desc_similarity`).
        - Supports configurable migration parameters (`migration_batch`, `cyclic_migration`).
        - Allows islands to be reset or killed based on performance.
    - `SequencePopulation` (`sequence_population.py`): A simpler (potentially non-generational) population structure (currently basic).
- **Query Items:** Uses `PopulationQueryItem` to represent tasks for the main loop, specifying parent individuals for generating offspring.
- **Diversity Metrics:** Provides utility functions in `population.py` to assess population diversity:
    - `code_diff_similarity`: Based on line-by-line code differences.
    - `code_bert_similarity`: Uses CodeBERT embeddings for semantic code similarity.
    - `desc_similarity`: Uses sentence transformers on algorithm descriptions.
- **Persistence:** Populations can be saved to and loaded from disk using `pickle` via the `save()` and `load()` methods.

**Usage:**
1.  **Instantiate:** Create an instance of a specific population class (e.g., `ESPopulation`) with desired parameters (e.g., `n_parent`, `n_offspring`). Optionally provide custom strategy functions.
2.  **Get Tasks:** Call `get_offspring_queryitems()` to get a list of `PopulationQueryItem` objects. Each item indicates which parent(s) should be used to generate a new offspring.
3.  **Add Individuals:** After an offspring is generated and evaluated by the LLM and Evaluator, add the resulting `Individual` object to the population using `add_individual(individual, generation)`.
4.  **Advance Generation:** Call `select_next_generation()` to apply the survival selection mechanism and advance the population state to the next generation (primarily for `ESPopulation`).
5.  **Retrieve Data:** Access individuals using methods like `get_best_individual()`, `get_individuals(generation)`, `all_individuals()`.

**Customization:**
- **Strategies:** Implement custom functions for parent selection (`get_parent_strategy`) and survival selection (`selection_strategy`) and pass them to the constructor of `ESPopulation` or `IslandESPopulation`.
- **New Population Types:** Create a new class inheriting from `Population`. Implement all abstract methods (`get_population_size`, `add_individual`, `remove_individual`, `get_offspring_queryitems`, `get_current_generation`, `get_best_individual`, `all_individuals`) to define a completely new population management scheme.
- **Diversity Metrics:** Add new diversity calculation functions in `population.py` or elsewhere and integrate them into selection or migration strategies.

## Parallelism in IOHEvaluator

The `IOHEvaluator` supports several modes for parallelizing the evaluation of algorithms across different IOH problems, instances, and repetitions:

1.  **Sequential Execution:**
    - **How:** This is the default mode if no parallel options are explicitly enabled (i.e., `max_eval_workers` is set to 0 or less, and `use_mpi` and `use_mpi_future` are `False`).
    - **Description:** Each evaluation task (a specific problem/instance/repetition) is executed one after another in the main process.

2.  **Thread Pool Execution:**
    - **How:** Set `max_eval_workers` to a positive integer (e.g., `evaluator.max_eval_workers = 10`) and ensure `use_multi_process` is `False` (default).
    - **Description:** Uses Python's `concurrent.futures.ThreadPoolExecutor` to run evaluation tasks concurrently in multiple threads within the same process. 

3.  **Process Pool Execution:**
    - **How:** Set `max_eval_workers` to a positive integer and set `use_multi_process = True` (e.g., `evaluator.max_eval_workers = 10; evaluator.use_multi_process = True`).
    - **Description:** Uses Python's `concurrent.futures.ProcessPoolExecutor` to run evaluation tasks in separate processes. Suitable for the algorithm which don't use multiple cores effectively. 

4.  **MPI (Custom Task Manager):**
    - **How:** Set `use_mpi = True` (e.g., `evaluator.use_mpi = True`). Requires MPI environment, `mpi4py` installed and a specific command to run the script (e.g., `mpiexec python pyfile`).
    - **Description:** Utilizes a custom master-worker implementation (`MPITaskManager`) built on top of `mpi4py`. The main node(rank 0) distributes tasks to worker nodes(rank > 0). Suitable for distributed systems.

5.  **MPI (mpi4py.futures):**
    - **How:** Set `use_mpi_future = True` (e.g., `evaluator.use_mpi_future = True`). Requires MPI environment, `mpi4py` installed and a specific command to run the script (e.g., `mpiexec -n numprocs python -m mpi4py.futures pyfile`). The details of the command can be found in [the documentation of `mpi4py.futures`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#command-line). 
    - **Description:** Leverages `mpi4py.futures.MPIPoolExecutor` for a higher-level interface to MPI-based parallelism. Similar to the process pool but designed specifically for MPI environments.

**Configuration:**
These options are typically set as attributes on the `IOHEvaluator` instance *before* calling the `evaluate` method.

## Tests

```bash
make test
```
