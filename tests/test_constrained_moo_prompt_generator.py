"""PR5: ConstrainedMOOPromptGenerator appends feasibility feedback.

Verifies that the constrained generator:
  1. keeps the parent's (unconstrained) hypervolume feedback intact, and
  2. appends per-run feasibility rate + mean constraint violation computed from
     PR1's EvaluatorBasicResult.feasibility_rate / cv_history, with low-feasibility
     guidance, and
  3. leaves the base feedback untouched on a purely unconstrained evaluation.

The generator's logic is prompt-group agnostic, so the test composes the
existing ``moo_bo`` group just to obtain a valid PromptStrings config.
"""

import pathlib

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from llamevol.evaluator.evaluator_result import EvaluatorBasicResult, EvaluatorResult
from llamevol.prompt_generators.constrained_moo_prompt_generator import (
    ConstrainedMOOPromptGenerator,
)

_CONF = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


def _generator():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONF, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["prompts=moo_bo", "hardware=cpu", "mode=mo"],
        )
    return ConstrainedMOOPromptGenerator(cfg.prompts)


def _basic(name, best_y, feasibility_rate=None, cv=None):
    b = EvaluatorBasicResult()
    b.name = name
    b.best_y = best_y
    if feasibility_rate is not None:
        b.feasibility_rate = feasibility_rate
        b.cv_history = np.asarray(cv if cv is not None else [])
    return b


def _result(basics):
    er = EvaluatorResult()
    er.name = "MOBOConstrainedFoo"
    er.total_execution_time = 1.0
    er.result = basics
    return er


def test_appends_feasibility_section_with_low_feasibility_guidance():
    gen = _generator()
    er = _result(
        [
            _basic("mw1-rep1", -0.10, feasibility_rate=0.20, cv=[1.0, 0.0, 2.0]),
            _basic("ctp1-rep1", -0.05, feasibility_rate=0.40, cv=[0.5, 0.0]),
        ]
    )
    fb = gen.evaluation_feedback_prompt(er)

    # Parent's HV feedback is preserved.
    assert "Hypervolume" in fb
    # Feasibility section appended.
    assert "feasible fraction" in fb
    assert "mean constraint violation" in fb
    assert "mw1-rep1" in fb and "ctp1-rep1" in fb
    assert "Mean feasible fraction across runs: 30.0%" in fb  # mean(0.2, 0.4)
    # Low-feasibility (<50%) guidance fires.
    assert "prioritize locating the feasible region" in fb


def test_unconstrained_result_left_unchanged():
    gen = _generator()
    er = _result([_basic("zdt1-rep1", -0.7), _basic("zdt2-rep1", -0.6)])
    fb = gen.evaluation_feedback_prompt(er)

    # No constraint metadata -> identical to the parent's feedback.
    base = super(ConstrainedMOOPromptGenerator, gen).evaluation_feedback_prompt(er)
    assert fb == base
    assert "feasible fraction" not in fb
