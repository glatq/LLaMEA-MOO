from abc import ABC, abstractmethod

from llamevol.evaluator import EvaluatorResult
from llamevol.prompt_generators.types import GenerationTask


class ResponseHandler(ABC):
    """Abstract base class for response handler."""

    def __init__(self):
        self.sys_prompt = ""
        self.prompt = ""
        self.raw_response = ""
        self.llm_model = ""

        self.code = ""
        self.code_name = ""

        self.feedback = ""
        self.error = None
        self.error_type = None

        self._eval_result: EvaluatorResult = None

        self.parent_ids = []

        self.query_time = 0
        self.prompt_token_count = 0
        self.response_token_count = 0

    @property
    def eval_result(self) -> EvaluatorResult:
        return self._eval_result

    @eval_result.setter
    def eval_result(self, value: EvaluatorResult):
        if value is not None and value.error is not None:
            self.error = value.error
            self.error_type = value.error_type
        self._eval_result = value

    @abstractmethod
    def extract_response(self, response: str, task: GenerationTask) -> None:
        pass

    def __to_json__(self) -> dict:
        d = {
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response,
            "error": self.error,
            "error_type": self.error_type,
            "eval_result": self.eval_result.__to_json__(),
        }
        return d
