import numpy as np
from .load_default_prompt_configurations import load_default_bl_prompt_config
from .abstract_prompt_generator import (
    PromptGenerator,
    EvaluatorResult,
)
from .response_handler import BaselineResponseHandler
from .types import GenerationTask
from ..population import Population


class BaselinePromptGenerator(PromptGenerator):
    def __init__(self, conf=None):
        if conf is None:
            conf = load_default_bl_prompt_config()
        super().__init__(conf)

    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options: dict = None
    ) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        aocs = []

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

        final_feedback_prompt = f"{main_aoc_prompt}\n{time_prompt}"
        return final_feedback_prompt

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
            if candidates and len(candidates) > 0:
                n_solution = len(candidates)
                pre_solution_prompt = (
                    f"{n_solution} {self.prompt_strings.pre_solution_prompt_template}"
                )
                for i, candidate in enumerate(candidates):
                    candidate_prompt = self._get_candidate_prompt(candidate)
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
                    candidate_prompt = self._get_candidate_prompt(candidate)
                    selected_prompt += f"## {candidate.code_name}\n{candidate_prompt}\n"

                selected_prompt += f"{crossover_operator}\n"
            else:
                candidate = candidates[0]
                candidate_prompt = self._get_candidate_prompt(candidate)
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
