from .abstract_prompt_generator import PromptGenerator
from .response_handler import ResponseHandler
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
        hvs = []
        grouped_hvs = []
        for _ in range(5):
            grouped_hvs.append([])
        for res in eval_res.result:
            hv = res.best_y
            hvs.append(hv)

            res_id = res.id or ""
            parts = res_id.split("-")

            raw_problem = parts[0] if len(parts) > 0 else ""
            raw_instance = parts[1] if len(parts) > 1 else ""
            raw_repeat = parts[2] if len(parts) > 2 else ""

            try:
                problem_num = int(raw_problem)
            except ValueError:
                problem_num = None

            try:
                instance_id = int(raw_instance) if raw_instance != "" else None
            except ValueError:
                instance_id = None

            try:
                repeat_id = int(raw_repeat) if raw_repeat != "" else None
            except ValueError:
                repeat_id = None

            if problem_num is not None:
                if problem_num <= 5:
                    group_idx = 0
                elif problem_num <= 9:
                    group_idx = 1
                elif problem_num <= 14:
                    group_idx = 2
                elif problem_num <= 19:
                    group_idx = 3
                else:
                    group_idx = 4
                problem_id_for_content = problem_num
            else:
                group_idx = 4
                problem_id_for_content = raw_problem

            content = {
                "problem_id": problem_id_for_content,
                "instance_id": instance_id,
                "repeat_id": repeat_id,
                "y_hv": hv,
            }
            grouped_hvs[group_idx].append(content)

        valid_hvs = [hv for hv in hvs if hv is not None]
        if not valid_hvs:
            hv_mean, hv_std = 0.0, 0.0
        else:
            hv_mean, hv_std = np.mean(valid_hvs), np.std(valid_hvs)

        separated_hvs = [content["y_hv"] for content in grouped_hvs[0]]
        separated_mean_hvs = np.mean(separated_hvs) if len(separated_hvs) > 0 else 0

        low_mod_hvs = [content["y_hv"] for content in grouped_hvs[1]]
        low_mod_mean_hvs = np.mean(low_mod_hvs) if len(low_mod_hvs) > 0 else 0

        high_uni_hvs = [content["y_hv"] for content in grouped_hvs[2]]
        high_uni_mean_hvs = np.mean(high_uni_hvs) if len(high_uni_hvs) > 0 else 0

        multi_adequate_hvs = [content["y_hv"] for content in grouped_hvs[3]]
        multi_adequate_mean_hvs = (
            np.mean(multi_adequate_hvs) if len(multi_adequate_hvs) > 0 else 0
        )

        multi_weak_hvs = [content["y_hv"] for content in grouped_hvs[4]]

        valid_weak_hvs = [hv for hv in multi_weak_hvs if hv is not None]
        if not valid_weak_hvs:
            multi_weak_mean_hvs = 0.0
        else:
            multi_weak_mean_hvs = np.mean(valid_weak_hvs)

        main_hv_prompt = self.prompt_strings.main_aoc_prompt_template.format(
            algorithm_name=algorithm_name, hv_mean=hv_mean, hv_std=hv_std
        )

        return self._format_feedback(eval_res, main_hv_prompt)

    def get_response_handler(self):
        return ResponseHandler()
