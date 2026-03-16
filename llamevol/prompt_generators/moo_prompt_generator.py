from .abstract_prompt_generator import PromptGenerator, ResponseHandler, GenerationTask
import re
import numpy as np
from .load_default_prompt_configurations import load_default_moo_prompt_config
from ..evaluator import EvaluatorResult
from ..population import Population
from .prompt_strings import PromptStrings


class BaselineResponseHandler(ResponseHandler):
    def __init__(self):
        super().__init__()
        self.desc = ""
        self.reason = ""
        self.config_space = ""

    def __to_json__(self):
        return {
            "desc": self.desc,
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response,
            "config_space": self.config_space,
        }

    def extract_response(self, response: str, task: GenerationTask):
        if not response:
            return

        self.raw_response = response
        sections = ["Description", "Justification", "Code", "Space"]
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
            elif section == "Space":
                self.config_space, _ = self.extract_from_response(response, section)

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
                pattern = r"#\s*Code[\s\S]*?```(?:python)?\s([\s\S]*?)```"
            elif section == "Code2":
                pattern = r"```(?:python)?\s([\s\S]*?)```"
            elif section == "Space":
                pattern = r"#\s*Space[\s\S]*?```(?:python)?\s*([\s\S]*?)```"
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
    def __init__(self, conf=None):
        super().__init__()

        if conf is None:
            conf = load_default_moo_prompt_config()

        self.prompt_strings = PromptStrings(**conf["prompts"])
        self.is_bo = conf["is_bo"]
        self.use_mini_bo = conf["use_mini_bo"]
        self.use_cuda = conf["use_cuda"]
        self.problem_desc = conf["problem_desc"]

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

    def __bo_task_description(self, task):
        if self.use_cuda:
            lib_prompt = self.prompt_strings.bo_lib_prompt_gpu
        else:
            lib_prompt = self.prompt_strings.bo_lib_prompt_cpu

        return self.prompt_strings.bo_task_prompt_template.format(
            problem_desc=self.problem_desc, lib_prompt=lib_prompt
        )

    def __task_description(self, task: GenerationTask) -> str:
        return self.prompt_strings.general_task_prompt.format(
            problem_desc=self.problem_desc
        )

    def response_format(self, task: GenerationTask) -> str:
        return self.prompt_strings.output_format_prompt

    def code_structure(self):
        if self.is_bo:
            if self.use_mini_bo:
                return self.__mini_bo_code_structure()
            return self.__bo_code_structure()
        return self.__code_structure()

    def __code_structure(self) -> str:
        return self.prompt_strings.code_structure_template

    def __bo_code_structure(self) -> str:
        return self.prompt_strings.bo_code_structure_template

    def __mini_bo_code_structure(self) -> str:
        return self.prompt_strings.mini_bo_code_structure_template

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
        time_prompt = self.prompt_strings.time_prompt_template.format(
            execution_time=execution_time
        )

        main_hv_prompt = self.prompt_strings.main_aoc_prompt_template.format(
            algorithm_name=algorithm_name, hv_mean=hv_mean, hv_std=hv_std
        )

        # Add HPO feedback if incumbent hyperparameters are available
        hpo_prompt = ""
        if hasattr(eval_res, "metadata") and eval_res.metadata:
            if "incumbent" in eval_res.metadata and eval_res.metadata["incumbent"]:
                incumbent = eval_res.metadata["incumbent"]
                hpo_prompt = f"\nOptimized hyperparameters: {incumbent}"
            elif "hpo_error" in eval_res.metadata:
                hpo_prompt = f"\nNote: {eval_res.metadata['hpo_error']}"

        final_feedback_prompt = f"{main_hv_prompt}\n{time_prompt}{hpo_prompt}"

        return final_feedback_prompt

    def __get_candidate_prompt(self, candidate: BaselineResponseHandler) -> str:
        description = candidate.desc
        solution = self.prompt_strings.candidate_code_wrapper.format(
            code=candidate.code
        )
        if candidate.error:
            if candidate.error_type == "NoCodeException":
                feedback = self.prompt_strings.no_code_exception_msg
            else:
                feedback = self.prompt_strings.general_error_msg.format(
                    error=candidate.error
                )
        else:
            feedback = self.evaluation_feedback_prompt(candidate.eval_result)

        return self.prompt_strings.candidate_prompt_template.format(
            description=description, solution=solution, feedback=feedback
        )

    def get_prompt(
        self,
        task: GenerationTask,
        problem_desc: str,
        candidates: list[BaselineResponseHandler] = None,
        population: Population = None,
        options: dict = None,
    ) -> tuple[str, str]:
        role_prompt = self.prompt_strings.role_prompt

        task_prompt = self.task_description(task)

        response_format_prompt = self.response_format(task=task)

        if task == GenerationTask.INITIALIZE_SOLUTION:
            pre_solution_prompt = ""
            if len(candidates) > 0:
                n_solution = len(candidates)
                pre_solution_prompt = (
                    f"{n_solution} {self.prompt_strings.pre_solution_prompt_template}"
                )
                for i, candidate in enumerate(candidates):
                    candidate_prompt = self.__get_candidate_prompt(candidate)
                    pre_solution_prompt += (
                        f"## {candidate.code_name}\n{candidate_prompt}\n"
                    )
                pre_solution_prompt += "\n"

            code_structure_prompt = (
                self.prompt_strings.code_structure_intro + self.code_structure()
            )
            final_prompt = f"""{task_prompt}\n{pre_solution_prompt}\n{code_structure_prompt}\n{response_format_prompt}"""
        else:
            if len(candidates) > 1:
                crossover_operator = self.prompt_strings.crossover_operator

                selected_prompt = self.prompt_strings.selected_solutions_intro

                for candidate in candidates:
                    candidate_prompt = self.__get_candidate_prompt(candidate)
                    selected_prompt += f"## {candidate.code_name}\n{candidate_prompt}\n"

                selected_prompt += f"{crossover_operator}\n"
            else:
                candidate = candidates[0]
                candidate_prompt = self.__get_candidate_prompt(candidate)
                mutation_operator = self.prompt_strings.mutation_operator

                selected_prompt = f"""{self.prompt_strings.selected_solution_intro}{candidate_prompt}\n{mutation_operator}\n"""

            population_summary = ""
            if population is not None and population.get_population_size() > 0:
                current_population = population.get_individuals()
                population_summary = self.prompt_strings.population_summary_intro
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
