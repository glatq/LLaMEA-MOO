import numpy as np
from .abstract_prompt_generator import (
    PromptGenerator,
    EvaluatorResult,
)


class BaselinePromptGenerator(PromptGenerator):
    def __init__(self, conf):
        super().__init__(conf)

    def _format_population_entry(self, handler) -> str:
        name = handler.code_name
        score = handler.eval_result.score
        runtime = handler.eval_result.total_execution_time
        desc = handler.desc
        return f"- {name}: {score:.4f}, {runtime:.2f} seconds, {desc}\n"

    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options: dict = None
    ) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        aocs = [res.log_y_aoc for res in eval_res.result]
        auc_mean = np.mean(aocs)
        auc_std = np.std(aocs)

        main_aoc_prompt = self.prompt_strings.main_aoc_prompt_template.format(
            algorithm_name=algorithm_name, auc_mean=auc_mean, auc_std=auc_std
        )

        return self._format_feedback(eval_res, main_aoc_prompt)
