import logging
import getopt
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
from llamevol.prompt_generators import MultiObjectivePromptGenerator
from llamevol.population import ESPopulation
from llamevol.llm import LLMmanager
from llamevol import LLaMEvol
from llamevol.utils import setup_logger
from Experiments.plot_search_res import plot_search_result

from pymoo.config import Config

Config.warnings["not_compiled"] = False


def get_MOOEvaluator(cfg):
    # Default budget and dimension for MO experiments
    budget = cfg.mo_search.budget
    dim = cfg.mo_search.dtlz_dimension
    repeat = cfg.mo_search.repeat
    timeout = cfg.mo_search.evaluator_timeout

    # HPO configuration (if enabled in config)
    use_hpo = cfg.mo_search.get("use_hpo", False)
    hpo_trials = cfg.mo_search.get("hpo_trials", 500)
    hpo_min_budget = cfg.mo_search.get("hpo_min_budget", 50)
    hpo_max_budget = cfg.mo_search.get("hpo_max_budget", 200)
    hpo_walltime = cfg.mo_search.get("hpo_walltime", 3600)
    hpo_validation_budget = cfg.mo_search.get("hpo_validation_budget", 20)

    # Define the suite of 10 multi-objective benchmarks
    problems = [
        MOOProblemSpec(name="zdt1", dim=30, n_obj=2, ref_point=[1.1, 6.0]),
        MOOProblemSpec(name="zdt2", dim=30, n_obj=2, ref_point=[1.1, 7.0]),
        MOOProblemSpec(name="zdt3", dim=30, n_obj=2, ref_point=[1.1, 6.0]),
        MOOProblemSpec(name="zdt4", dim=10, n_obj=2, ref_point=[1.1, 160.0]),
        MOOProblemSpec(name="zdt6", dim=10, n_obj=2, ref_point=[1.1, 10.0]),
        MOOProblemSpec(name="dtlz1", dim=dim, n_obj=3, ref_point=[10.0, 10.0, 30.0]),
        MOOProblemSpec(name="dtlz2", dim=dim, n_obj=3, ref_point=[1.5, 1.5, 1.5]),
        MOOProblemSpec(name="dtlz3", dim=dim, n_obj=3, ref_point=[20.0, 30.0, 100.0]),
        MOOProblemSpec(name="dtlz4", dim=dim, n_obj=3, ref_point=[2.0, 1.2, 1.2]),
        MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0]),
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
    )

    if use_hpo:
        logging.info("=" * 60)
        logging.info("SMAC HPO ENABLED")
        logging.info(f"  Trials: {hpo_trials}")
        logging.info(f"  Budget range: {hpo_min_budget}-{hpo_max_budget}")
        logging.info(f"  Walltime: {hpo_walltime}s")
        logging.info("=" * 60)

    return evaluator


def get_mo_prompt_generator():
    prompt_generator = MultiObjectivePromptGenerator()
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

    # pass the code of all solutions in the first generation to the LLM
    population.preorder_aware_init = True

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
    evaluator = get_MOOEvaluator(cfg)
    evaluator.timeout = cfg.mo_search.evaluator_timeout

    # create a prompt generator
    prompt_generator = get_mo_prompt_generator()

    # create a LLM Manager
    model_name = cfg.mo_search.llm.model_name
    base_url = cfg.mo_search.llm.base_url

    # choose the llm client, e.g. openai, google.
    # openai: OpenaiClient; google: google genai client; others: AISuiteClient
    client = cfg.mo_search.llm.client
    api_key = cfg.mo_search.llm.api_key

    llm = LLMmanager(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        client_str=client,
    )

    # define ES parameters
    es_options = {
        "n_parent": cfg.mo_search.n_parent,  # number of parents
        "n_offspring": cfg.mo_search.n_offspring,  # number of offspring
        "is_elitist": cfg.mo_search.is_elitist,  # whether to use elitist selection
        "log_dir": cfg.mo_search.log_dir,  # directory to save logs
    }

    print(
        f"n_parents: {cfg.mo_search.n_parent}, n_offspring: {cfg.mo_search.n_offspring}, "
        f"elitist: {cfg.mo_search.is_elitist}, n_population: {cfg.mo_search.n_population}, api_key: {cfg.mo_search.llm.api_key}"
    )

    # create an ES Population
    population = get_es_population(es_options)

    # run the evolution
    llamevol = LLaMEvol()
    llm_params = {
        "temperature": cfg.mo_search.llm.temperature,
        "top_k": cfg.mo_search.llm.top_k,  # top_k sampling, which might not be supported by all LLMs
        # you can add other LLM parameters here if needed
    }

    llamevol.run_evolutions(
        llm,
        evaluator,
        prompt_generator,
        population,
        n_population=cfg.mo_search.n_population,
        options={"llm_params": llm_params},
    )

    population.save(suffix="final")


if __name__ == "__main__":
    setup_logger(level=logging.INFO)

    is_plotting = False

    opts, args = getopt.getopt(sys.argv[1:], "p:o:k:en:f")
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
        elif opt == "-f":
            is_ploting = True

    if is_plotting:
        # plot the search results: combine all the log files with 'final' suffix
        print("Plotting the MO search results...")
        log_dir = "exp_mo_es_search"
        plot_search_result(log_dir, fig_dir=log_dir)
        sys.exit(0)

    run_exp()
