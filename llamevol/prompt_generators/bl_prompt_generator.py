import re
import numpy as np
from pydantic import BaseModel
from .load_default_prompt_configurations import load_default_bl_prompt_config
from .abstract_prompt_generator import (
    PromptGenerator,
    ResponseHandler,
    GenerationTask,
    EvaluatorResult,
)
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


class PromptStrings(BaseModel):
    role_prompt: str
    pre_solution_prompt_template: str
    code_structure_intro: str
    crossover_operator: str
    selected_solutions_intro: str
    selected_solution_intro: str
    mutation_operator: str
    population_summary_intro: str
    candidate_code_wrapper: str
    no_code_exception_msg: str
    general_error_msg: str
    candidate_prompt_template: str
    bo_lib_prompt_cpu: str
    bo_lib_prompt_gpu: str
    bo_task_prompt_template: str
    general_task_prompt: str
    output_format_prompt: str
    code_structure_template: str
    mini_bo_code_structure_template: str
    bo_code_structure_template: str
    time_prompt_template: str
    main_aoc_prompt_template: str


class BaselinePromptGenerator(PromptGenerator):
    def __init__(self, conf=None):
        super().__init__()

        if conf is None:
            conf = load_default_bl_prompt_config()

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
        return f"{suffix}BaselinePromptGenerator"

    # prompt generation
    def get_prompt(
        self,
        task: GenerationTask,
        problem_desc: str,
        candidates: list[BaselineResponseHandler] = None,
        population: Population = None,
        options: dict = None,
    ) -> tuple[str, str]:
        if task != GenerationTask.INITIALIZE_SOLUTION:
            if candidates is None or len(candidates) == 0:
                return "", ""

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

                    # pre_solution_prompt += f"- {candidate.desc}\n"
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
                        f"- {name}: {score:.4f}, {runtime:.2f} seconds, {desc}\n"
                    )

            final_prompt = f"""{task_prompt}
    {population_summary}

{selected_prompt}

    {response_format_prompt}
    """
        return role_prompt, final_prompt

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

    def __mini_bo_code_structure(self) -> str:
        return self.prompt_strings.mini_bo_code_structure_template

    def __bo_code_structure(self) -> str:
        return self.prompt_strings.bo_code_structure_template

    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options: dict = None
    ) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        aocs = []

        # While the grouping logic (grouped_aocs) was in the original function,
        # it doesn't seem to be used in the final return string.
        # I have preserved the calculation of 'aocs' which is used for the mean/std.

        for res in eval_res.result:
            aoc = res.log_y_aoc
            aocs.append(aoc)

        auc_mean = np.mean(aocs)
        auc_std = np.std(aocs)

        execution_time = eval_res.total_execution_time
        time_prompt = self.prompt_strings.time_prompt_template.format(
            execution_time=execution_time
        )

        main_aoc_prompt = self.prompt_strings.main_aoc_prompt_template.format(
            algorithm_name=algorithm_name, auc_mean=auc_mean, auc_std=auc_std
        )

        # Note: 'detailed_aoc_prompt' was calculated but not used in the original return statement.
        # I have omitted the calculation logic for the unused variables (separated_auc, etc.) to keep the refactor clean.
        # If you intend to use the detailed feedback later, we should add a template for it.

        final_feedback_prompt = f"{main_aoc_prompt}\n{time_prompt}"
        return final_feedback_prompt

    # Helper functions
    def get_response_handler(self):
        return BaselineResponseHandler()
