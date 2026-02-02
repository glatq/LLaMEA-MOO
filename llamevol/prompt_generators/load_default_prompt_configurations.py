import yaml
from ..directories import prompt_generator_configs


def load_default_bl_prompt_config():
    cfg_path = prompt_generator_configs(filename="bl_prompt_generator.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    return cfg
