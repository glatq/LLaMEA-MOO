import re
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
