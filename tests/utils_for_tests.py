import textwrap

from llamevol.evaluator.evaluator_result import EvaluatorBasicResult


def normalize_prompt(prompt_string):
    # 1. Remove common leading whitespace/indentation from multiline strings
    dedented = textwrap.dedent(prompt_string)

    # 2. Unify newline characters (e.g., convert Windows \r\n to Unix \n)
    normalized_newlines = dedented.replace("\r\n", "\n")

    # 3. Strip any leading/trailing newlines that might exist from the f-string definition
    # This specifically addresses the leading \n from f"""\n..."""
    return normalized_newlines.strip()


def make_basic_result(id_str: str, log_aoc: float) -> EvaluatorBasicResult:
    r = EvaluatorBasicResult()
    r.id = id_str
    r.log_y_aoc = log_aoc
    return r
