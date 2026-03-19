from abc import ABC, abstractmethod
from llamevol.evaluator import EvaluatorResult
from .types import GenerationTask
from .prompt_strings import PromptStrings
from .response_handler import BaselineResponseHandler


class ResponseImpReturnChecker(ABC):
    """Abstract base class for response return checkers."""

    @abstractmethod
    def __call__(self, imp_return: tuple) -> str:
        pass


class PromptGenerator(ABC):
    """Abstract base class for prompt generators."""

    def __init__(self, conf=None):
        if conf is not None:
            self.prompt_strings = PromptStrings(**conf["prompts"])
            self.is_bo = conf["is_bo"]
            self.use_mini_bo = conf["use_mini_bo"]
            self.use_cuda = conf["use_cuda"]
            self.problem_desc = conf["problem_desc"]

    def __str__(self):
        suffix = ""
        if hasattr(self, "is_bo") and self.is_bo:
            if hasattr(self, "use_mini_bo") and self.use_mini_bo:
                suffix = "MiniBO"
            else:
                suffix = "BO"
        return f"{suffix}{self.__class__.__name__}"

    def task_description(self, task: GenerationTask) -> str:
        if self.is_bo:
            return self._bo_task_description(task)
        return self._task_description(task)

    def _bo_task_description(self, task):
        if self.use_cuda:
            lib_prompt = self.prompt_strings.bo_lib_prompt_gpu
        else:
            lib_prompt = self.prompt_strings.bo_lib_prompt_cpu

        return self.prompt_strings.bo_task_prompt_template.format(
            problem_desc=self.problem_desc, lib_prompt=lib_prompt
        )

    def _task_description(self, task: GenerationTask) -> str:
        return self.prompt_strings.general_task_prompt.format(
            problem_desc=self.problem_desc
        )

    def task_instruction(self, task: GenerationTask) -> str:
        """explicit COT of the task accomplishment"""
        pass

    def code_structure(self) -> str:
        if self.is_bo:
            if self.use_mini_bo:
                return self._mini_bo_code_structure()
            return self._bo_code_structure()
        return self._code_structure()

    def _code_structure(self) -> str:
        return self.prompt_strings.code_structure_template

    def _mini_bo_code_structure(self) -> str:
        return self.prompt_strings.mini_bo_code_structure_template

    def _bo_code_structure(self) -> str:
        return self.prompt_strings.bo_code_structure_template

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

    @abstractmethod
    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options=None
    ) -> str:
        pass

    @abstractmethod
    def get_prompt(
        self,
        task: GenerationTask,
        problem_desc: str,
        candidates=None,
        population=None,
        options: dict = None,
    ) -> tuple[str, str]:
        pass
