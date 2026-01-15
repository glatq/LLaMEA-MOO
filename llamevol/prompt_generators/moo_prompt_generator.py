from .abstract_prompt_generator import PromptGenerator, ResponseHandler, GenerationTask
import torch
import re
import numpy as np
from ..evaluator import EvaluatorResult
from ..population import Population


class BaselineResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()
        self.desc = ""
        self.reason = ""

    def __to_json__(self):
        return {
            "desc": self.desc,
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response,
        }

    def extract_response(self, response: str, task: GenerationTask):
        if not response:
            return

        self.raw_response = response
        sections = ["Description", "Justification", "Code"]
        for section in sections:
            if section == "Code":
                self.code, err = self.extract_from_response(response, section)
                if err:
                    self.code, _ = self.extract_from_response(response, "Code2")
                self.code_name, _ = self.extract_from_response(response, "class_name")
            elif section == "Description":
                self.desc, _ = self.extract_from_response(response, section)
            elif section == "Justification":
                self.reason, _ = self.extract_from_response(response, section)

    def extract_from_response(
        self, response: str, section: str, pattern=None
    ) -> tuple[str, str]:
        error_str = ""
        res = ""
        ignore_case = True
        if pattern is None:
            if section == "class_name":
                pattern = r"```(?:python)?[\s\S]*?class\s+(\w+BO\w*):"
                ignore_case = False
            elif section == "Code":
                pattern = r"#\s*Code[\s\S]*```(?:python)?\s([\s\S]*?)```"
            elif section == "Code2":
                pattern = r"```(?:python)?\s([\s\S]*?)```"
            else:
                pattern = rf"#\s*{section}\s*([\s\S]*?)#\s"
                # pattern = rf"#\s*{section}\s*:\s*(.*)"
        match = re.search(pattern, response, re.IGNORECASE if ignore_case else 0)
        if match:
            res = match.group(1)
        else:
            error_str = f"{section} not found in the response."
        return res, error_str


class MultiObjectivePromptGenerator(PromptGenerator):
    def __init__(self):
        super().__init__()
        self.is_bo = False
        self.use_mini_bo = False
        self.problem_description = "pymoo library multi objective problems"

    def __str__(self):
        suffix = ""
        if self.is_bo:
            if self.use_mini_bo:
                suffix = "MiniBO"
            else:
                suffix = "BO"
        return f"{suffix}MultiObjectivePromptGenerator"

    def task_description(self, task: GenerationTask) -> str:
        if self.is_bo:
            return self.__bo_task_description(task)
        return self.__task_description(task)

    def __task_description(self, task):
        task_prompt = """
Your task is to write the multi-objective optimization algorithm in Python code. The code must provide a class with an init(self, budget, dim) method and a call(self, func) method. The call method should optimize the black-box function func using at most self.budget function evaluations. The func takes an array of shape (n_dims,) and returns an array of shape (M,) (one value per objective). One budget unit corresponds to one call of func (producing all M objectives). The search space bounds are provided via the `bounds` input (shape (2, dim)). The algorithm MUST use this `bounds` array for all sampling and clipping operations and MUST NOT assume or hard-code any bounds in the algorithm's logic. The dimensionality dim can vary.
As an expert in numpy, scipy, scikit-learn, torch, and gpytorch, you are allowed to use these libraries. Prefer lightweight models (for example, Gaussian processes or random forests with a sliding window of recent data) and avoid deep neural networks or very heavy models. If you use surrogate models, limit the training set size (for example, using only the most recent 200 points) so that fitting remains cheap. Do not use any other libraries unless they cannot be replaced by the above libraries. Do not remove the comments from the code. Name the class based on the characteristics of the algorithm with a template 'MOBO<Something>'.
The primary performance metric is Hypervolume (HV) maximization with respect to a fixed reference point. HV and the reference point are handled externally by the evaluator; your algorithm only needs to return a good approximation of the Pareto front (F_pareto, X_pareto). Do not compute the full hypervolume indicator inside the algorithm. 

CRITICAL ROBUSTNESS REQUIREMENT: The algorithm's internal search logic (especially the acquisition function and model fitting) MUST be stable and general across all objective scales.
Any scalarization method (e.g., Tchebycheff or weighted sum) MUST be applied to objectives that are normalized and scaled.
The normalization MUST be calculated dynamically based on the current observed objective ranges: using the minimum observed value (Ideal Point) and the maximum observed value (Nadir Point) from the archive (`self.y_all`) to scale all objectives to the $[0, 1]$ range.
This normalization is essential to prevent early pathological evaluations from destabilizing the acquisition function and steering the search to irrelevant regions.

If you use hypervolume-based acquisition (e.g., EHVI), you may choose a simple and lightweight internal reference point (for example based on the current observed nadir). This internal reference point is used only for acquisition scoring and does not need to match the evaluator’s reference point. Any HV-related computation used for acquisition must remain local and inexpensive (for example, small candidate sets), and must not compute HV over the entire archive.
You can assume typical problem dimensionalities between 2 and 20 and evaluation budgets between 50 and 500. The algorithm must remain computationally lightweight in Python. Its cost should scale approximately linearly with the number of evaluations and dimensions and should avoid O(budget²) operations in the main loop (for example, repeated full-archive dominance checks, full hypervolume calculations, or large nested Python loops).
Give a robust, computationally efficient multi-objective Bayesian Optimization algorithm to solve this task. It should be conceptually clear and not overly complex. Give the algorithm a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        return task_prompt

    def __bo_task_description(self, task):
        task_prompt = """
Your task is to write the multi-objective optimization algorithm in Python code. The code must provide a class with an init(self, budget, dim) method and a call(self, func) method. The call method should optimize the black-box function func using at most self.budget function evaluations. The func takes an array of shape (n_dims,) and returns an array of shape (M,) (one value per objective). One budget unit corresponds to one call of func (producing all M objectives). The search space bounds are provided via the `bounds` input (shape (2, dim)). The algorithm MUST use this `bounds` array for all sampling and clipping operations and MUST NOT assume or hard-code any bounds in the algorithm's logic. The dimensionality dim can vary.
As an expert in numpy, scipy, scikit-learn, torch, and gpytorch, you are allowed to use these libraries. Prefer lightweight models (for example, Gaussian processes or random forests with a sliding window of recent data) and avoid deep neural networks or very heavy models. If you use surrogate models, limit the training set size (for example, using only the most recent 200 points) so that fitting remains cheap. Do not use any other libraries unless they cannot be replaced by the above libraries. Do not remove the comments from the code. Name the class based on the characteristics of the algorithm with a template 'MOBO<Something>'.
The primary performance metric is Hypervolume (HV) maximization with respect to a fixed reference point. HV and the reference point are handled externally by the evaluator; your algorithm only needs to return a good approximation of the Pareto front (F_pareto, X_pareto). Do not compute the full hypervolume indicator inside the algorithm. 

CRITICAL ROBUSTNESS REQUIREMENT: The algorithm's internal search logic (especially the acquisition function and model fitting) MUST be stable and general across all objective scales.
Any scalarization method (e.g., Tchebycheff or weighted sum) MUST be applied to objectives that are normalized and scaled.
The normalization MUST be calculated dynamically based on the current observed objective ranges: using the minimum observed value (Ideal Point) and the maximum observed value (Nadir Point) from the archive (`self.y_all`) to scale all objectives to the $[0, 1]$ range.
This normalization is essential to prevent early pathological evaluations from destabilizing the acquisition function and steering the search to irrelevant regions.

If you use hypervolume-based acquisition (e.g., EHVI), you may choose a simple and lightweight internal reference point (for example based on the current observed nadir). This internal reference point is used only for acquisition scoring and does not need to match the evaluator’s reference point. Any HV-related computation used for acquisition must remain local and inexpensive (for example, small candidate sets), and must not compute HV over the entire archive.
You can assume typical problem dimensionalities between 2 and 20 and evaluation budgets between 50 and 500. The algorithm must remain computationally lightweight in Python. Its cost should scale approximately linearly with the number of evaluations and dimensions and should avoid O(budget²) operations in the main loop (for example, repeated full-archive dominance checks, full hypervolume calculations, or large nested Python loops).
Give a robust, computationally efficient multi-objective Bayesian Optimization algorithm to solve this task. It should be conceptually clear and not overly complex. Give the algorithm a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""

        if torch.cuda.is_available():
            task_prompt = """
Your task is to write the multi-objective optimization algorithm in Python code. The code must provide a class with an init(self, budget, dim) method and a call(self, func) method. The call method should optimize the black-box function func using at most self.budget function evaluations. The func takes an array of shape (n_dims,) and returns an array of shape (M,) (one value per objective). One budget unit corresponds to one call of func (producing all M objectives). The search space bounds are provided via the `bounds` input (shape (2, dim)). The algorithm MUST use this `bounds` array for all sampling and clipping operations and MUST NOT assume or hard-code any bounds in the algorithm's logic. The dimensionality dim can vary.
As an expert in numpy, scipy, scikit-learn, torch, and gpytorch, you are allowed to use these libraries. Prefer lightweight models (for example, Gaussian processes or random forests with a sliding window of recent data) and avoid deep neural networks or very heavy models. If you use surrogate models, limit the training set size (for example, using only the most recent 200 points) so that fitting remains cheap. Do not use any other libraries unless they cannot be replaced by the above libraries. Do not remove the comments from the code. Name the class based on the characteristics of the algorithm with a template 'MOBO<Something>'.
The primary performance metric is Hypervolume (HV) maximization with respect to a fixed reference point. HV and the reference point are handled externally by the evaluator; your algorithm only needs to return a good approximation of the Pareto front (F_pareto, X_pareto). Do not compute the full hypervolume indicator inside the algorithm. 

CRITICAL ROBUSTNESS REQUIREMENT: The algorithm's internal search logic (especially the acquisition function and model fitting) MUST be stable and general across all objective scales.
Any scalarization method (e.g., Tchebycheff or weighted sum) MUST be applied to objectives that are normalized and scaled.
The normalization MUST be calculated dynamically based on the current observed objective ranges: using the minimum observed value (Ideal Point) and the maximum observed value (Nadir Point) from the archive (`self.y_all`) to scale all objectives to the $[0, 1]$ range.
This normalization is essential to prevent early pathological evaluations from destabilizing the acquisition function and steering the search to irrelevant regions.

If you use hypervolume-based acquisition (e.g., EHVI), you may choose a simple and lightweight internal reference point (for example based on the current observed nadir). This internal reference point is used only for acquisition scoring and does not need to match the evaluator’s reference point. Any HV-related computation used for acquisition must remain local and inexpensive (for example, small candidate sets), and must not compute HV over the entire archive.
You can assume typical problem dimensionalities between 2 and 20 and evaluation budgets between 50 and 500. The algorithm must remain computationally lightweight in Python. Its cost should scale approximately linearly with the number of evaluations and dimensions and should avoid O(budget²) operations in the main loop (for example, repeated full-archive dominance checks, full hypervolume calculations, or large nested Python loops).
Give a robust, computationally efficient multi-objective Bayesian Optimization algorithm to solve this task. It should be conceptually clear and not overly complex. Give the algorithm a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.
"""
        return task_prompt

    def response_format(self, task: GenerationTask) -> str:
        output_format_prompt = """
Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>
"""
        return output_format_prompt

    def code_structure(self):
        if self.is_bo:
            if self.use_mini_bo:
                return self.__mini_bo_code_structure()
            return self.__bo_code_structure()
        return self.__code_structure()

    def __code_structure(self) -> str:
        return """
```python
import numpy as np 

class RandomSearchMO:
    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = bounds

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        # Pareto dominance for minimization
        return np.all(a <= b) and np.any(a < b)

    def __call__(self, func):
        X = []  # decision vectors in archive
        F = []  # objective vectors in archive

        low, high = self.bounds[0], self.bounds[1]

        for _ in range(self.budget):
            x = np.random.uniform(low, high)  # shape (dim,)
            f = np.asarray(func(x), dtype=float).ravel()  # shape (M,)

            # If dominated by any current archive point -> skip
            if any(self._dominates(fi, f) for fi in F):
                continue

            # Otherwise, remove points dominated by the newcomer
            keep = [i for i, fi in enumerate(F) if not self._dominates(f, fi)]
            if len(keep) != len(F):
                X = [X[i] for i in keep]
                F = [F[i] for i in keep]

            # Add newcomer
            X.append(x)
            F.append(f)

        # Return as numpy arrays, one row per solution
        if len(F) == 0:
            # No nondominated found (only possible if budget==0); fall back to empty arrays
            return np.empty((0,)), np.empty((0, self.dim))
        return np.vstack(F), np.vstack(X)
"""

    def __bo_code_structure(self) -> str:
        return """
```python
from collections.abc import Callable
from scipy.stats import qmc  # If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np


class <AlgorithmName>:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2, dim), bounds[0]: lower bound, bounds[1]: upper bound
        # The environment (evaluator / problem provider) should pass the true bounds.
        # Do NOT overwrite self.bounds with hard-coded values.
        if bounds is None:
            # Fallback: assume a simple [0, 1]^dim box if no bounds are provided.
            self.bounds = np.array([[0.0] * dim, [1.0] * dim], dtype=float)
        else:
            self.bounds = np.asarray(bounds, dtype=float)

        # The number of objectives (self.n_obj) is unknown a priori.
        # It MUST be inferred on the first call to func inside _evaluate_points.
        self.n_obj: int | None = None

        # X has shape (n_points, n_dims), y has shape (n_points, n_obj)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_evals = 0  # the number of function evaluations

        # Choose a reasonable number of initial evaluations.
        # Use a small, budget-aware design (for example, proportional to dim, but not more than a fraction of the budget).
        self.n_init = <your_strategy>

        # You may define internal batch size or sliding-window sizes here,
        # but do not add any other arguments without a default value.
        # Keep any additional state lightweight and inexpensive to update.

        # Do not add any other arguments without a default value

        def _sample_points(self, n_points: int) -> np.ndarray:
        # Sample n_points candidate points efficiently within self.bounds.
        # Use self.bounds[0] as lower bounds and self.bounds[1] as upper bounds.
        # Return array of shape (n_points, n_dims).


    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        # Fit and tune a lightweight surrogate model on (X, y).
        # Return the fitted model and store any required state on self.
        # Do not change the function signature.
        # Use a sliding window or otherwise limit the number of training points
        # so that model fitting remains computationally cheap.
        # Prefer simple models such as Gaussian processes or random forests over very heavy models.

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        # Implement a multi-objective acquisition function.
        # Prefer efficient scalarisation-based methods (for example ParEGO: random weight vectors + EI),
        # or lightweight approximations of HV improvement.
        # Do not compute full hypervolume over the entire archive inside this function.
        # Any HV-related computation used here must remain local and inexpensive
        # (for example, small candidate sets, small approximated Pareto subsets).
        # Calculate the acquisition function value for each point in X.
        # Return a 1-D score per candidate of shape (n_points,) for selection, using vectorised operations when possible.
        # All scalarisation weights MUST have length self.n_obj. Do NOT hard-code weights of size 2 or 3.

    def _select_next_points(self, batch_size: int) -> np.ndarray:
        # Select the next points to evaluate.
        # Use the acquisition function on a modest number of candidate points per iteration.
        # Keep candidate set sizes and any internal populations small (for example, on the order of tens, not hundreds or thousands),
        # and avoid deep nested loops or large inner optimisers.
        # The selection strategy can be any heuristic / evolutionary / mathematical / hybrid method,
        # but it must remain computationally cheap in Python.
        # If using scalarisation, you may sample or rotate weight vectors across iterations.
        # If using EHVI-style criteria, approximate HV improvement against a small current Pareto set only.
        # Return an array of shape (batch_size, n_dims).

    def _evaluate_points(self, func: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
        # Evaluate the points in X.
        # On the first evaluation, infer M from func(x). Set self.n_obj = y.shape[0] and enforce this dimension in all later operations.
        # This method must be the only place where func is called.
        # Respect the remaining budget: do not exceed self.budget evaluations in total.
        # Use simple, clear looping over points, and clip points to the bounds if necessary.
        # Update self.n_evals by the actual number of function calls performed.
        # Clip points to self.bounds before calling func, to respect the problem domain.
        # lb, ub = self.bounds
        # X_clipped = np.clip(X, lb, ub)
        # then call func on X_clipped
        # Return an array of shape (n_points, n_obj).

        self.n_evals += len(X)

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        # Update self.X and self.y with new evaluations.
        # Do not change the function signature.
        # Maintain/update the non-dominated archive efficiently and keep only Pareto-optimal points.
        # Implement archive maintenance incrementally: update dominance status using only the current archive and new points,
        # and avoid recomputing the full Pareto front from scratch with quadratic nested Python loops over all past points.
        # Dominance comparisons MUST use y.shape[1] == self.n_obj, never a fixed number of objectives.
        # Prefer vectorised dominance checks when possible.

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Main minimise optimisation loop.
        # func takes an array of shape (n_dims,) and returns np.ndarray of shape (M,) (one value per objective).
        # Do not call func directly; always use _evaluate_points so that the budget is respected.
        # Use an initial design phase, then iterate:
        #   fit/update the surrogate model on the current archive,
        #   use the acquisition function to select new points,
        #   evaluate them with _evaluate_points, and
        #   update the archive with _update_eval_points.
        # Keep per-iteration overhead small and avoid heavy nested optimisation inside this loop.
        # Return a tuple (F_pareto, X_pareto), where F_pareto has shape (K, n_obj)
        # and X_pareto has shape (K, n_dims) for the final non-dominated set.
        # The algorithm MUST remain correct for any number of objectives self.n_obj ≥ 2 without code changes.

        self._evaluate_points(...)
        self._update_eval_points(...)

        while self.n_evals < self.budget:
            # Optimisation loop:
            # 1. Fit or update the surrogate model using the current archive.
            # 2. Use the acquisition function and _select_next_points to pick new candidates.
            # 3. Evaluate candidates with _evaluate_points, respecting the remaining budget.
            # 4. Update the archive with _update_eval_points, keeping only non-dominated points.
            ...
            self._evaluate_points(...)
            self._update_eval_points(...)

        return F_pareto, X_pareto
"""

    def __mini_bo_code_structure(self) -> str:
        return """
```python
from collections.abc import Callable
import numpy as np

class <AlgorithmName>:
    def __init__(self, budget: int, dim: int, bounds: np.ndarray | None = None):
        self.budget = int(budget)
        self.dim = int(dim)
        # bounds has shape (2, <dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = bounds

        # X has shape (n_points, n_dims), y has shape (n_points, M)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.n_evals = 0  # number of function evaluations
        self.n_init = 0   # unused for pure random search; keep for API consistency

        # Do not add any other arguments without a default value

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        # Pareto dominance for minimization
        return np.all(a <= b) and np.any(a < b)

    def __call__(self, func: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # func: takes array of shape (n_dims,) and returns a 1-D np.ndarray of objectives (length M).
        # Return a tuple (self.y, self.X) = (F_nd, X_nd) with shapes (k, M) and (k, dim), respectively.
        low, high = self.bounds[0], self.bounds[1]

        X_archive: list[np.ndarray] = []
        F_archive: list[np.ndarray] = []

        while self.n_evals < self.budget:
            # Optimization
            pass
        
        # Persist history if desired
        if F_archive:
            self.X = np.vstack(X_archive)
            self.y = np.vstack(F_archive)
        else:
            self.X = np.empty((0, self.dim))
            self.y = np.empty((0,))
            
        return self.y, self.X
"""

    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options=None
    ) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        hvs = []
        grouped_hvs = []
        for _ in range(5):
            grouped_hvs.append([])
        for res in eval_res.result:
            # We stored negative HV as log_y_aoc in the evaluator.
            # Convert back to positive HV for human-readable feedback.
            hv = res.best_y
            hvs.append(hv)

            # Original ID format in the SO/IOH case was "<int>-<int>-<int>".
            # In the new MO/Pymoo case it is something like "zdt1-1-1" or "dtlz2-1-3".
            res_id = res.id or ""
            parts = res_id.split("-")

            raw_problem = parts[0] if len(parts) > 0 else ""
            raw_instance = parts[1] if len(parts) > 1 else ""
            raw_repeat = parts[2] if len(parts) > 2 else ""

            # Try to interpret the problem id as an integer (IOH-style).
            # If that fails, we keep it as a string (Pymoo-style).
            try:
                problem_num = int(raw_problem)
            except ValueError:
                problem_num = None

            try:
                instance_id = int(raw_instance) if raw_instance != "" else None
            except ValueError:
                instance_id = None

            try:
                repeat_id = int(raw_repeat) if raw_repeat != "" else None
            except ValueError:
                repeat_id = None

            # Grouping logic is meaningful only for the original IOH numeric ids.
            # For non-numeric ids (Pymoo problems), just assign them to the last group.
            if problem_num is not None:
                if problem_num <= 5:
                    group_idx = 0
                elif problem_num <= 9:
                    group_idx = 1
                elif problem_num <= 14:
                    group_idx = 2
                elif problem_num <= 19:
                    group_idx = 3
                else:
                    group_idx = 4
                problem_id_for_content = problem_num
            else:
                # Pymoo named problems, e.g. "zdt1", "dtlz2", etc.
                group_idx = 4
                problem_id_for_content = raw_problem

            content = {
                "problem_id": problem_id_for_content,
                "instance_id": instance_id,
                "repeat_id": repeat_id,
                "y_hv": hv,
            }
            grouped_hvs[group_idx].append(content)

        valid_hvs = [hv for hv in hvs if hv is not None]
        if not valid_hvs:
            hv_mean, hv_std = 0.0, 0.0
        else:
            hv_mean, hv_std = np.mean(valid_hvs), np.std(valid_hvs)

        separated_hvs = [content["y_hv"] for content in grouped_hvs[0]]
        separated_mean_hvs = np.mean(separated_hvs) if len(separated_hvs) > 0 else 0

        low_mod_hvs = [content["y_hv"] for content in grouped_hvs[1]]
        low_mod_mean_hvs = np.mean(low_mod_hvs) if len(low_mod_hvs) > 0 else 0

        high_uni_hvs = [content["y_hv"] for content in grouped_hvs[2]]
        high_uni_mean_hvs = np.mean(high_uni_hvs) if len(high_uni_hvs) > 0 else 0

        multi_adequate_hvs = [content["y_hv"] for content in grouped_hvs[3]]
        multi_adequate_mean_hvs = (
            np.mean(multi_adequate_hvs) if len(multi_adequate_hvs) > 0 else 0
        )

        multi_weak_hvs = [content["y_hv"] for content in grouped_hvs[4]]

        valid_weak_hvs = [hv for hv in multi_weak_hvs if hv is not None]
        if not valid_weak_hvs:
            multi_weak_mean_hvs = 0.0
        else:
            multi_weak_mean_hvs = np.mean(valid_weak_hvs)

        execution_time = eval_res.total_execution_time
        time_prompt = f"took {execution_time:0.2f} seconds to run."

        main_hv_prompt = f"""The algorithm {algorithm_name} got an average Hypervolume (HV, the larger the better) score of {hv_mean:0.4f} with standard deviation {hv_std:0.4f}.
        """
        # THIS ONE IS NOT USED IN final_feedback_prompt
        detailed_hv_prompt = f"""
        The mean HV score of the algorithm {algorithm_name} on Separable functions was {separated_mean_hvs:.04f}, on functions with low or moderate conditioning {low_mod_mean_hvs:.04f}, on functions with high conditioning and unimodal {high_uni_mean_hvs:.04f}, on Multi-modal functions with adequate global structure {multi_adequate_mean_hvs:.04f}, and on Multi-modal functions with weak global structure {multi_weak_mean_hvs:.04f}
        """

        final_feedback_prompt = f"{main_hv_prompt}\n{time_prompt}"

        return final_feedback_prompt

    def __get_candidate_prompt(self, candidate: BaselineResponseHandler) -> str:
        description = candidate.desc
        solution = f"```python\n{candidate.code}\n```"
        if candidate.error:
            if candidate.error_type == "NoCodeException":
                feedback = "No code was extracted. The code should be encapsulated with ``` in your response."
            else:
                feedback = f"An error occurred : {candidate.error}"
        else:
            feedback = self.evaluation_feedback_prompt(candidate.eval_result)

        return f"{description}\nWith code:\n{solution}\n{feedback}\n"

    def get_prompt(
        self,
        task: GenerationTask,
        problem_desc: str,
        candidates: list[BaselineResponseHandler] = None,
        population: Population = None,
        options: dict = None,
    ) -> tuple[str, str]:
        role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design a highly generalizable, scale-invariant multi-objective metaheuristic algorithm to solve arbitrary black-box optimization problems that are both effective and computationally lightweight in Python. The total runtime of the algorithm should be dominated by calls to the objective function, not by internal overhead."

        task_prompt = self.task_description(task)

        response_format_prompt = self.response_format(task=task)

        if task == GenerationTask.INITIALIZE_SOLUTION:
            pre_solution_prompt = ""
            if len(candidates) > 0:
                n_solution = len(candidates)
                pre_solution_prompt = f"{n_solution} algorithms have been designed. The new algorithm should be as **diverse** as possible from the previous ones on every aspect.\nIf the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n"
                for i, candidate in enumerate(candidates):
                    candidate_prompt = self.__get_candidate_prompt(candidate)
                    pre_solution_prompt += (
                        f"## {candidate.code_name}\n{candidate_prompt}\n"
                    )
                pre_solution_prompt += "\n"

            code_structure_prompt = (
                """The number of objectives M is unknown and must be inferred at runtime.
The algorithm MUST:
- detect M from the first evaluation of func,
- set self.n_obj = M exactly once in _evaluate_points,
- use self.n_obj everywhere instead of any hard-coded number,
- never assume M = 2 or M = 3.
A code structure guide is as follows and keep the comments from the guide when generating the code.\n"""
                + self.code_structure()
            )
            final_prompt = f"""{task_prompt}\n{pre_solution_prompt}\n{code_structure_prompt}\n{response_format_prompt}"""
        else:
            if len(candidates) > 1:
                crossover_operator = "Combine the selected solutions to create a new solution. Then refine the strategy of the new solution to improve it. If the errors from the previous algorithms are provided, analyze them. The new algorithm should be designed to avoid these errors.\n"

                selected_prompt = (
                    "The selected solutions to update are the evaluated algorithms:\n"
                )

                for candidate in candidates:
                    candidate_prompt = self.__get_candidate_prompt(candidate)
                    selected_prompt += f"## {candidate.code_name}\n{candidate_prompt}\n"

                selected_prompt += f"{crossover_operator}\n"
            else:
                candidate = candidates[0]
                candidate_prompt = self.__get_candidate_prompt(candidate)
                mutation_operator = (
                    "Refine the strategy of the selected solution to improve it."
                )

                selected_prompt = f"""The selected solution to update is the evaluated algorithm:\n{candidate_prompt}\n{mutation_operator}\n"""

            population_summary = ""
            if population is not None and population.get_population_size() > 0:
                current_population = population.get_individuals()
                population_summary = "The current population of algorithms already evaluated(name, score and runtime):\n"
                for ind in current_population:
                    handler = Population.get_handler_from_individual(ind)
                    if handler.eval_result is None:
                        continue
                    name = handler.code_name
                    score = handler.eval_result.score
                    runtime = handler.eval_result.total_execution_time
                    desc = handler.desc
                    population_summary += (
                        f"- {name}: {score:.4f}, {runtime:.2f} seconds\n"
                    )

            final_prompt = f"""{task_prompt}
{population_summary}

{selected_prompt}

{response_format_prompt}
"""
        return role_prompt, final_prompt

    def get_response_handler(self):
        return BaselineResponseHandler()
