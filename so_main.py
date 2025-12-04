import hydra
from omegaconf import DictConfig
import run_es_search


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_es_search.run_exp(cfg, 1, 1, True, None, n_population=40)


if __name__ == "__main__":
    main()
