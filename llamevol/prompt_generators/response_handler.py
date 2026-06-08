import re
from dataclasses import dataclass, field

from llamevol.evaluator import EvaluatorResult
from llamevol.prompt_generators.types import GenerationTask


@dataclass
class SectionConfig:
    attr: str
    patterns: list[str]
    ignore_case: bool = True


class ResponseHandler:
    _sections: dict[str, SectionConfig] = {
        "Description": SectionConfig(
            attr="desc",
            patterns=[r"#\s*Description\s*([\s\S]*?)#\s"],
        ),
        "Justification": SectionConfig(
            attr="reason",
            patterns=[r"#\s*Justification\s*([\s\S]*?)#\s"],
        ),
        "Code": SectionConfig(
            attr="code",
            patterns=[
                r"#\s*Code[\s\S]*?```(?:python)?\s([\s\S]*?)```",
                r"```(?:python)?\s([\s\S]*?)```",
            ],
        ),
        "class_name": SectionConfig(
            attr="code_name",
            patterns=[r"```(?:python)?[\s\S]*?class\s+(\w+BO\w*):"],
            ignore_case=False,
        ),
        "Space": SectionConfig(
            attr="config_space",
            patterns=[r"#\s*Space[\s\S]*?```(?:python)?\s*([\s\S]*?)```"],
        ),
    }

    def __init__(self):
        self.sys_prompt = ""
        self.prompt = ""
        self.raw_response = ""
        self.llm_model = ""

        self.code = ""
        self.code_name = ""

        self.desc = ""
        self.reason = ""
        self.config_space = ""

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

    def extract_response(self, response: str, task: GenerationTask) -> None:
        if not response:
            return

        self.raw_response = response
        for section in self._sections:
            value, _ = self.extract_from_response(response, section)
            setattr(self, self._sections[section].attr, value)

    def extract_from_response(self, response: str, section: str) -> tuple[str, str]:
        config = self._sections.get(section)
        if config is None:
            return "", f"{section} not found in the response."

        flags = re.IGNORECASE if config.ignore_case else 0
        for p in config.patterns:
            match = re.search(p, response, flags)
            if match:
                return match.group(1), ""
        return "", f"{section} not found in the response."

    def __to_json__(self) -> dict:
        return {
            "desc": self.desc,
            "code": self.code,
            "code_name": self.code_name,
            "raw_response": self.raw_response,
            "config_space": self.config_space,
            "error": self.error,
            "error_type": self.error_type,
            "eval_result": self.eval_result.__to_json__() if self.eval_result else None,
        }
