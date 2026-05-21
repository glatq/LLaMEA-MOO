from abc import ABC, abstractmethod
from llamevol.evaluator import EvaluatorResult
from .types import GenerationTask
from .prompt_strings import PromptStrings
from .bl_response_handler import BaselineResponseHandler
from ..population import Population


class ResponseImpReturnChecker(ABC):
    """Abstract base class for response return checkers."""

    @abstractmethod
    def __call__(self, imp_return: tuple) -> str:
        pass


class PromptGenerator(ABC):
    """Abstract base class for prompt generators."""

    def __init__(self, conf):
        # Safely handle Hydra DictConfig without strictly requiring omegaconf
        if type(conf).__name__ == "DictConfig":
            from omegaconf import OmegaConf

            conf = OmegaConf.to_container(conf, resolve=True)

        self.prompt_strings = PromptStrings(**conf["prompts"])
        self.problem_desc = conf.get("problem_desc", "")

    def __str__(self):
        return self.__class__.__name__

    def task_description(self, task: GenerationTask) -> str:
        parts = [
            self.prompt_strings.task_domain_prompt.format(
                problem_desc=self.problem_desc
            ),
            self.prompt_strings.lib_prompt,
            self.prompt_strings.task_mode_prompt,
        ]
        return "".join(p for p in parts if p)

    def task_instruction(self, task: GenerationTask) -> str:
        """explicit COT of the task accomplishment"""
        pass

    def code_structure(self) -> str:
        return self.prompt_strings.code_structure_template

    def response_format(self, task: GenerationTask) -> str:
        return self.prompt_strings.output_format_prompt

    def _get_candidate_prompt(self, candidate) -> str:
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

    def get_response_handler(self):
        return BaselineResponseHandler()

    def get_return_checker(self):
        return None

    def _format_population_entry(self, handler) -> str:
        name = handler.code_name
        score = handler.eval_result.score
        runtime = handler.eval_result.total_execution_time
        return f"- {name}: {score:.4f}, {runtime:.2f} seconds\n"

    def _format_feedback(self, eval_res, main_metric_prompt: str) -> str:
        execution_time = eval_res.total_execution_time
        time_prompt = self.prompt_strings.time_prompt_template.format(
            execution_time=execution_time
        )

        hpo_prompt = ""
        if hasattr(eval_res, "metadata") and eval_res.metadata:
            if "incumbent" in eval_res.metadata and eval_res.metadata["incumbent"]:
                incumbent = eval_res.metadata["incumbent"]
                hpo_prompt = f"\nOptimized hyperparameters: {incumbent}"
            elif "hpo_error" in eval_res.metadata:
                hpo_prompt = f"\nNote: {eval_res.metadata['hpo_error']}"

        return f"{main_metric_prompt}\n{time_prompt}{hpo_prompt}"

    @abstractmethod
    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options=None
    ) -> str:
        pass

    def get_prompt(
        self,
        task: GenerationTask,
        problem_desc: str,
        candidates=None,
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
                    population_summary += self._format_population_entry(handler)

            final_prompt = f"""{task_prompt}
{population_summary}

{selected_prompt}

{response_format_prompt}
"""
        return role_prompt, final_prompt
