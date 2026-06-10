"""Constrained multi-objective prompt generator.

Subclass of :class:`MultiObjectivePromptGenerator` that augments the evaluation
feedback with constraint information so the LLM can reason about feasibility:
per-run feasibility rate and mean constraint violation, plus aggregate guidance.

The parent's (unconstrained) hypervolume feedback and per-problem bucketing are
reused verbatim through ``super()`` and are NOT modified -- this class only
*appends* a feasibility section, and only when the evaluation actually carried
constraints (i.e. the data layer populated ``EvaluatorBasicResult.feasibility_rate``).
On a purely unconstrained evaluation it returns the base feedback unchanged, so
the generator is safe to use even if pointed at unconstrained problems.

Constraint conventions match the data layer (PR1): a point is feasible only when
every constraint satisfies ``G <= 0``, and its violation is ``cv = sum(max(0, G))``.
"""

import numpy as np

from ..evaluator import EvaluatorResult
from .moo_prompt_generator import MultiObjectivePromptGenerator


class ConstrainedMOOPromptGenerator(MultiObjectivePromptGenerator):
    """MOO prompt generator that adds feasibility-rate / constraint-violation feedback."""

    # Below this mean feasible fraction we explicitly nudge the LLM to prioritise
    # reaching the feasible region before refining objectives.
    LOW_FEASIBILITY_THRESHOLD = 0.5

    def evaluation_feedback_prompt(
        self, eval_res: EvaluatorResult, options=None
    ) -> str:
        base = super().evaluation_feedback_prompt(eval_res, options)
        if eval_res is None or not eval_res.result:
            return base

        rows = []
        feas_rates = []
        for res in eval_res.result:
            feas = getattr(res, "feasibility_rate", None)
            if feas is None:
                # Unconstrained rep: nothing constraint-specific to report.
                continue
            cv_hist = getattr(res, "cv_history", None)
            mean_cv = (
                float(np.mean(cv_hist))
                if cv_hist is not None and len(cv_hist) > 0
                else 0.0
            )
            feas_rates.append(float(feas))
            rows.append(
                f"- {res.name or '?'}: feasible fraction {feas:.1%}, "
                f"mean constraint violation {mean_cv:.4g}"
            )

        if not feas_rates:
            # Purely unconstrained evaluation -> leave the base feedback untouched.
            return base

        mean_feas = float(np.mean(feas_rates))
        section = (
            "\nConstraint feasibility feedback (a point is feasible only when every "
            "constraint satisfies G <= 0; only feasible points contribute to the "
            "Hypervolume score, and the constraint violation is cv = sum(max(0, G))).\n"
            f"Mean feasible fraction across runs: {mean_feas:.1%}.\n"
            "Per-run feasibility and mean constraint violation:\n"
            + "\n".join(rows)
            + "\n"
        )
        if mean_feas < self.LOW_FEASIBILITY_THRESHOLD:
            section += (
                "The feasible fraction is low: prioritize locating the feasible region "
                "(stronger constraint handling, e.g. feasibility-weighted selection or "
                "the Constraint Dominance Principle) before refining the objectives.\n"
            )
        return base + section
