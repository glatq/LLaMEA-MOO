import logging
import getopt
import sys
from omegaconf import DictConfig
import hydra
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
from llamevol.prompt_generators.bl_prompt_generator import BaselinePromptGenerator
from llamevol.prompt_generators.moo_prompt_generator import (
    MultiObjectivePromptGenerator,
)
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger
from pymoo.config import Config

Config.warnings["not_compiled"] = False


def get_IOHEvaluator(cfg):
    # HPO configuration (if enabled in config)
    use_hpo = cfg.so_search.get("use_hpo", False)
    hpo_trials = cfg.so_search.get("hpo_trials", 500)
    hpo_min_budget = cfg.so_search.get("hpo_min_budget", 50)
    hpo_max_budget = cfg.so_search.get("hpo_max_budget", 200)
    hpo_walltime = cfg.so_search.get("hpo_walltime", 3600)
    hpo_validation_budget = cfg.so_search.get("hpo_validation_budget", 20)
    hpo_n_problems = cfg.so_search.get("hpo_n_problems", None)
    hpo_n_workers = cfg.so_search.get("hpo_n_workers", 1)

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
        hpo_n_problems=hpo_n_problems,
        hpo_n_workers=hpo_n_workers,
    )

    return evaluator


def get_MOOEvaluator(cfg):
    budget = cfg.mo_search.budget
    repeat = cfg.mo_search.repeat
    timeout = cfg.mo_search.evaluator_timeout

    # HPO configuration (if enabled in config)
    use_hpo = cfg.mo_search.get("use_hpo", False)
    hpo_trials = cfg.mo_search.get("hpo_trials", 500)
    hpo_min_budget = cfg.mo_search.get("hpo_min_budget", 50)
    hpo_max_budget = cfg.mo_search.get("hpo_max_budget", 200)
    hpo_walltime = cfg.mo_search.get("hpo_walltime", 3600)
    hpo_validation_budget = cfg.mo_search.get("hpo_validation_budget", 20)
    hpo_n_problems = cfg.mo_search.get("hpo_n_problems", None)
    hpo_n_workers = cfg.mo_search.get("hpo_n_workers", 1)

    # Build MOOProblemSpec list from config
    problems = [
        MOOProblemSpec(
            name=p.name,
            dim=p.dim,
            n_obj=p.n_obj,
            ref_point=list(p.ref_point),
        )
        for p in cfg.mo_search.problems
    ]

    evaluator = MultiObjEvaluator(
        budget=budget,
        problems=problems,
        repeat=repeat,
        timeout=timeout,
        calculate_hv_history=False,
        use_hpo=use_hpo,
        hpo_trials=hpo_trials,
        hpo_min_budget=hpo_min_budget,
        hpo_max_budget=hpo_max_budget,
        hpo_walltime=hpo_walltime,
        hpo_validation_budget=hpo_validation_budget,
        hpo_n_problems=hpo_n_problems,
        hpo_n_workers=hpo_n_workers,
    )

    return evaluator


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
    mode = cfg.get("mode", "so")

    if mode == "mo":
        from pymoo.config import Config as PymooConfig

        PymooConfig.warnings["not_compiled"] = False

        search_cfg = cfg.mo_search
        evaluator = get_MOOEvaluator(cfg)
        prompt_generator = MultiObjectivePromptGenerator(cfg.prompts)
    elif mode == "so":
        search_cfg = cfg.so_search
        evaluator = get_IOHEvaluator(cfg)
        prompt_generator = BaselinePromptGenerator(cfg.prompts)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'so' or 'mo'.")

    evaluator.timeout = search_cfg.evaluator_timeout

    # Log HPO status for MO (SO evaluator logs this internally)
    if mode == "mo" and search_cfg.get("use_hpo", False):
        logging.info("=" * 60)
        logging.info("SMAC HPO ENABLED")
        logging.info(f"  Trials: {search_cfg.hpo_trials}")
        logging.info(
            f"  Budget range: {search_cfg.hpo_min_budget}-{search_cfg.hpo_max_budget}"
        )
        logging.info(f"  Walltime: {search_cfg.hpo_walltime}s")
        logging.info(f"  Workers: {search_cfg.get('hpo_n_workers', 1)}")
        logging.info("=" * 60)

    # create a LLM Manager
    llm = LLMmanager(
        model_name=search_cfg.llm.model_name,
        api_key=search_cfg.llm.api_key,
        base_url=search_cfg.llm.base_url,
        client_str=search_cfg.llm.client,
    )

    # define ES parameters
    es_options = {
        "n_parent": search_cfg.n_parent,
        "n_offspring": search_cfg.n_offspring,
        "is_elitist": search_cfg.is_elitist,
        "log_dir": search_cfg.log_dir,
    }

    print(
        f"Mode: {mode}, n_parents: {search_cfg.n_parent}, n_offspring: {search_cfg.n_offspring}, "
        f"is_elitist: {search_cfg.is_elitist}, n_population: {search_cfg.n_population}, "
        f"api_key: {search_cfg.llm.api_key}"
    )

    # create a ES Population
    population = get_es_population(es_options)

    # run the evolution
    llamevol = LLaMEvol()
    llm_params = {
        "temperature": search_cfg.llm.temperature,
        "top_k": search_cfg.llm.top_k,
    }

    llamevol.run_evolutions(
        llm,
        evaluator,
        prompt_generator,
        population,
        n_population=search_cfg.n_population,
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
