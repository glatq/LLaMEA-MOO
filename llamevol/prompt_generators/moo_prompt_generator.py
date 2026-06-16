from .response_handler import ResponseHandler
from .abstract_prompt_generator import PromptGenerator, _render
import numpy as np
from ..evaluator import EvaluatorResult


class MultiObjectivePromptGenerator(PromptGenerator):
    def __init__(self, conf):
        super().__init__(conf)

    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options=None
    ) -> str:
        if eval_res is None or len(eval_res.result) == 0:
            return ""

        algorithm_name = eval_res.name
        # best_y is stored as -HV (a per-point loss). Report the hypervolume as
        # +HV so the number matches "larger is better" and is consistent with
        # the population summary, which shows the (positive) run score.
        valid_hvs = [-res.best_y for res in eval_res.result if res.best_y is not None]
        if not valid_hvs:
            hv_mean, hv_std = 0.0, 0.0
        else:
            hv_mean, hv_std = np.mean(valid_hvs), np.std(valid_hvs)

        main_hv_prompt = _render(
            self.prompt_strings.main_aoc_prompt_template,
            algorithm_name=algorithm_name,
            hv_mean=f"{hv_mean:0.4f}",
            hv_std=f"{hv_std:0.4f}",
        )

        return self._format_feedback(eval_res, main_hv_prompt)

    def get_response_handler(self):
        return ResponseHandler()
