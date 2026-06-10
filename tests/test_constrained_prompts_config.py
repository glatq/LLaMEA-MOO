"""PR4: the constrained MOO prompt group composes and the generator consumes it.

Guards two things:
  1. The composed ``moo_constrained_bo`` group exposes exactly the same
     PromptStrings field set as ``moo_bo`` -> generator construction cannot break
     on a missing/renamed key.
  2. The constraint-specific content (the (F, G) contract, feasible-HV metric,
     CDP guidance, 3-tuple return) is actually threaded into the prompts.
"""

import pathlib

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from llamevol.prompt_generators.moo_prompt_generator import (
    MultiObjectivePromptGenerator,
)
from llamevol.prompt_generators.types import GenerationTask

_CONF = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


def _compose(group):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONF, version_base=None):
        return compose(
            config_name="config",
            overrides=[f"prompts={group}", "hardware=cpu", "mode=mo"],
        )


def _subkeys(cfg):
    return set(OmegaConf.to_container(cfg.prompts, resolve=True)["prompts"].keys())


def test_constrained_group_matches_moo_bo_shape():
    # Identical PromptStrings field set -> generator construction can't break.
    assert _subkeys(_compose("moo_constrained_bo")) == _subkeys(_compose("moo_bo"))


def test_generator_builds_from_constrained_group():
    cfg = _compose("moo_constrained_bo")
    gen = MultiObjectivePromptGenerator(cfg.prompts)  # validates PromptStrings
    td = gen.task_description(GenerationTask.INITIALIZE_SOLUTION)
    cs = gen.prompt_strings.code_structure_template

    assert "constraint" in gen.prompt_strings.role_prompt.lower()
    assert "(F, G)" in td  # constrained func contract
    assert "FEASIBLE Hypervolume" in td  # feasible-HV metric
    assert "Constraint Dominance Principle" in td  # CDP handling guidance
    assert "G_pareto" in cs  # 3-tuple return in the scaffold
    assert "_violation" in cs  # constraint-violation helper in the scaffold
