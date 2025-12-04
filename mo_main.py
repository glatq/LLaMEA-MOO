from llamevol.utils import setup_logger
import logging
import run_mo_es_search


def main():
    setup_logger(level=logging.INFO)
    run_mo_es_search.run_exp(1, 1, True, None, n_population=40)


if __name__ == "__main__":
    main()
