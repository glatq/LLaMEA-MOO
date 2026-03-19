from llamevol.prompt_generators.abstract_prompt_generator import (
    PromptGenerator,
    EvaluatorResult,
)
from llamevol.prompt_generators import ResponseHandler, GenerationTask


class DummyGenerator(PromptGenerator):
    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options=None
    ) -> str:
        return f"{eval_res.name}:{eval_res.score}"

    def get_prompt(
        self,
        task: GenerationTask,
        problem_desc: str,
        candidates=None,
        population=None,
        options=None,
    ) -> tuple[str, str]:
        return "SYS", "USER"

    def get_response_handler(self) -> ResponseHandler:
        return ResponseHandler()


class EmptyPopulation:
    def all_individuals(self):
        return []

    @staticmethod
    def get_handler_from_individual(ind):
        return ind
