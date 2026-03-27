import logging
import getopt
import sys
from omegaconf import DictConfig
import hydra
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.prompt_generators.bl_prompt_generator import BaselinePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger


def get_IOHEvaluator(cfg):
    # HPO configuration (if enabled in config)
    use_hpo = cfg.so_search.get("use_hpo", False)
    hpo_trials = cfg.so_search.get("hpo_trials", 500)
    hpo_min_budget = cfg.so_search.get("hpo_min_budget", 50)
    hpo_max_budget = cfg.so_search.get("hpo_max_budget", 200)
    hpo_walltime = cfg.so_search.get("hpo_walltime", 3600)
    hpo_validation_budget = cfg.so_search.get("hpo_validation_budget", 20)

    evaluator = IOHEvaluator(
        budget=cfg.so_search.budget,
        dim=cfg.so_search.dim,
        problems=cfg.so_search.problems,
        instances=cfg.so_search.instances,
        repeat=cfg.so_search.repeat,
        use_hpo=use_hpo,
        hpo_trials=hpo_trials,
        hpo_min_budget=hpo_min_budget,
        hpo_max_budget=hpo_max_budget,
        hpo_walltime=hpo_walltime,
        hpo_validation_budget=hpo_validation_budget,
    )

    if use_hpo:
        logging.info("=" * 60)
        logging.info("SMAC HPO ENABLED")
        logging.info(f"  Trials: {hpo_trials}")
        logging.info(f"  Budget range: {hpo_min_budget}-{hpo_max_budget}")
        logging.info(f"  Walltime: {hpo_walltime}s")
        logging.info("=" * 60)

    return evaluator


# create an prompt generator
def get_bo_prompt_generator(prompts_cfg: DictConfig) -> BaselinePromptGenerator:
    prompt_generator = BaselinePromptGenerator(prompts_cfg)
    prompt_generator.is_bo = True
    return prompt_generator


def get_es_population(es_options):
    _n_parent = es_options["n_parent"]
    _n_offspring = es_options["n_offspring"]
    _is_elitist = es_options["is_elitist"]
    _n_parent_per_offspring = 2
    if _n_parent < 2:
        _n_parent_per_offspring = 1
    population = ESPopulation(
        n_parent=_n_parent,
        n_parent_per_offspring=_n_parent_per_offspring,
        n_offspring=_n_offspring,
        use_elitism=_is_elitist,
    )

    population.preorder_aware_init = (
        True  # pass the code of all solutions in the first generation to the LLM
    )
    population.save_per_generation = 1  # save population every generation
    population.debug_save_on_the_fly = True  # save every individual in the population
    population.save_dir = es_options["log_dir"]

    p_name = f"{_n_parent}+{_n_offspring}"
    if not _is_elitist:
        p_name = f"{_n_parent}-{_n_offspring}"

    population.name = f"evol_{p_name}"  # the name of the population will be used as the prefix of the log directory

    return population


@hydra.main(config_path="conf", config_name="config", version_base=None)
def run_exp(cfg: DictConfig):
    # create an IOHEvaluator
    evaluator = get_IOHEvaluator(cfg)
    evaluator.timeout = (
        cfg.so_search.evaluator_timeout
    )  # set the timeout(seconds) for each evaluation(all tasks)

    # create a prompt generator
    prompt_generator = get_bo_prompt_generator(cfg.prompts)

    # create a LLM Manager
    model_name = cfg.so_search.llm.model_name
    base_url = cfg.so_search.llm.base_url  # use default

    # choose the llm client, e.g. openai, google.
    # openai: OpenaiClient; google: google genai client; others: AISuiteClient
    client = cfg.so_search.llm.client
    api_key = cfg.so_search.llm.api_key

    llm = LLMmanager(
        model_name=model_name, api_key=api_key, base_url=base_url, client_str=client
    )

    # define ES parameters
    es_options = {
        "n_parent": cfg.so_search.n_parent,  # number of parents
        "n_offspring": cfg.so_search.n_offspring,  # number of offspring
        "is_elitist": cfg.so_search.is_elitist,  # whether to use elitist selection
        "log_dir": cfg.so_search.log_dir,  # directory to save logs
    }

    print(
        f"n_parents: {cfg.so_search.n_parent}, n_offspring: {cfg.so_search.n_offspring}, is_elitist: {cfg.so_search.is_elitist}, n_population: {cfg.so_search.n_population}, api_key: {cfg.so_search.llm.api_key}"
    )

    # create a ES Population
    population = get_es_population(es_options)

    # run the evolution
    llamevol = LLaMEvol()
    llm_params = {
        "temperature": cfg.so_search.llm.temperature,
        "top_k": cfg.so_search.llm.top_k,  #!!!! top_k sampling, which might not be supported by all LLMs
    }

    llamevol.run_evolutions(
        llm,
        evaluator,
        prompt_generator,
        population,
        n_population=cfg.so_search.n_population,
        options={"llm_params": llm_params},
    )

    population.save(suffix="final")


if __name__ == "__main__":
    setup_logger(level=logging.INFO)

    opts, args = getopt.getopt(
        sys.argv[1:],
        "p:o:k:en:",
    )

    for opt, arg in opts:
        if opt == "-p":
            n_parents = int(arg)
        elif opt == "-o":
            n_offspring = int(arg)
        elif opt == "-k":
            api_key = arg
        elif opt == "-e":
            is_elitist = True
        elif opt == "-n":
            n_population = int(arg)

    run_exp()
