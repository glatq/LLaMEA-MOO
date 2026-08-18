"""Task 4 (MOEAD-EI extension) -- run SMAC HPO on MOEAD-EI Hybrid, then benchmark
the tuned version.

MOEAD-EI Hybrid was the only highlighted algorithm whose SMAC HPO crashed during
generation (a population-size bug, fixed in MO bench/MOBO_MOEAD_EI_Hybrid_fixed.py),
so it was benchmarked with the LLM's DEFAULT hyperparameters. This script gives it
the same HPO treatment as the regular LLaMEA runs and re-benchmarks it, so we can
report tuned-vs-untuned for MOEAD-EI too.

It mirrors the generation HPO exactly (settings + problem subset read from
conf/config.yaml mo_search): SMAC AlgorithmConfigurationFacade, hpo_trials trials,
multi-fidelity min/max budget, the evenly-spaced hpo_n_problems subset of the
generation suite, and the same config space the LLM emitted during generation
(hardcoded in build_configspace(), so no auxiliary pkl is needed).

Two stages:
  A. SMAC HPO on the FIXED code  -> incumbent saved to benchmark_results/moead_hpo/
  B. Benchmark the tuned version  -> paper_data/benchmark_moead_tuned/
     (12 synthetic + 3 real, budget 400, 5 repeats; same problems/ref points as
      benchmark_stage2/3 via conf/benchmark_hpo_ablation.yaml)

Then compare tuned vs untuned with:  python analyze_moead_hpo.py

Run from the llamea-moo-3.12 env at the repo root (GCP VM is fine):
    python run_moead_hpo.py
"""

import json
import os
import pickle
import types

import numpy as np
from omegaconf import OmegaConf

# --- sklearn DTYPE shim BEFORE importing anything that imports smac ---------
# smac 2.4.0 does `from sklearn.tree._tree import DTYPE`, which sklearn >=1.9
# removed. Restore it (it was always np.float32) so smac imports. No-op on
# environments where DTYPE still exists (e.g. older sklearn).
import sklearn.tree._tree as _skt  # noqa: E402

if not hasattr(_skt, "DTYPE"):
    _skt.DTYPE = np.float32
    print("[shim] restored sklearn.tree._tree.DTYPE = np.float32")

from ConfigSpace import (  # noqa: E402
    ConfigurationSpace,
    Float,
    Integer,
    Categorical,
)
from llamevol.evaluator.multiobj_evaluator import MOOProblemSpec  # noqa: E402
from llamevol.evaluator.smac_hpo_wrapper import (  # noqa: E402
    SMACHPOConfig,
    run_smac_hpo_moo,
    validate_with_random_config,
    SMAC_AVAILABLE,
)
from benchmark_best_codes import benchmark_and_plot  # noqa: E402

# --- paths -----------------------------------------------------------------
FIXED_CODE = "MO bench/MOBO_MOEAD_EI_Hybrid_fixed.py"
CLS_NAME = "MOBO_MOEAD_EI_Hybrid_Fixed"
OUT_DIR = "benchmark_results/moead_hpo"
INCUMBENT_JSON = os.path.join(OUT_DIR, "incumbent.json")
INCUMBENT_PKL = os.path.join(OUT_DIR, f"{CLS_NAME}_incumbent_handler.pkl")
TUNED_BENCH_DIR = "benchmark_results/moead_tuned"


def build_configspace():
    """MOEAD-EI Hybrid's hyperparameter search space.

    Identical to the space the LLM emitted during generation (recovered from its
    saved response and hardcoded here so this script needs no auxiliary pkl).
    """
    cs = ConfigurationSpace()
    cs.add(
        [
            Integer("batch_size", (1, 10), default=6),
            Categorical("kernel_nu", [0.5, 1.5, 2.5], default=0.5),
            Float("n_init_ratio", (0.05, 0.5), default=0.275),
            Integer("n_model_points", (100, 300), default=200),
            Integer("n_offspring_surrogate", (100, 400), default=250),
            Integer("population_size", (20, 100), default=60),
            Float("rho", (0.001, 0.1), default=0.01, log=True),
        ]
    )
    return cs


def main():
    if not SMAC_AVAILABLE:
        raise SystemExit(
            "SMAC failed to import even after the DTYPE shim. Check the smac / "
            "ConfigSpace install in this env."
        )
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = OmegaConf.load("conf/config_paper_search.yaml").mo_search
    with open(FIXED_CODE, "r") as f:
        code = f.read()

    configspace = build_configspace()
    print(f"[hpo] config space: {len(configspace)} params: {list(configspace.keys())}")

    # Build the generation problem suite, then take the same evenly-spaced subset
    # SMAC tuned on during generation (np.linspace over indices).
    gen_problems = [
        MOOProblemSpec(
            name=p.name, dim=int(p.dim), n_obj=int(p.n_obj), ref_point=list(p.ref_point)
        )
        for p in cfg.problems
    ]
    n_hpo = int(cfg.get("hpo_n_problems", len(gen_problems)) or len(gen_problems))
    if n_hpo < len(gen_problems):
        idx = [int(i) for i in np.linspace(0, len(gen_problems) - 1, n_hpo)]
        hpo_specs = [gen_problems[i] for i in idx]
    else:
        hpo_specs = gen_problems
    print(f"[hpo] tuning on {len(hpo_specs)} problems: {[s.name for s in hpo_specs]}")

    hpo_config = SMACHPOConfig(
        n_trials=int(cfg.hpo_trials),
        min_budget=int(cfg.hpo_min_budget),
        max_budget=int(cfg.hpo_max_budget),
        walltime_limit=int(cfg.hpo_walltime),
        trial_walltime_limit=cfg.get("hpo_trial_walltime", None),
        n_workers=int(cfg.get("hpo_n_workers", 1)),
    )
    budget = int(cfg.budget)
    print(
        f"[hpo] SMACHPOConfig: trials={hpo_config.n_trials}, "
        f"fidelity={hpo_config.min_budget}-{hpo_config.max_budget}, "
        f"workers={hpo_config.n_workers}, per-instance budget={budget}"
    )

    # Sanity: validation should now PASS on the fixed code (it crashed at this
    # step during generation, which is why MOEAD-EI was never tuned).
    ok, err = validate_with_random_config(
        code=code,
        cls_name=CLS_NAME,
        configspace=configspace,
        problem_spec=hpo_specs[0],
        budget=int(cfg.get("hpo_validation_budget", 50)),
    )
    if not ok:
        raise SystemExit(f"[hpo] validation still fails on the fixed code: {err}")
    print("[hpo] validation passed on fixed code.")

    incumbent, inc_hv = run_smac_hpo_moo(
        code=code,
        cls_name=CLS_NAME,
        configspace=configspace,
        problem_specs=hpo_specs,
        budget=budget,
        hpo_config=hpo_config,
    )
    if not incumbent:
        raise SystemExit("[hpo] SMAC returned an empty incumbent; aborting.")
    print(f"[hpo] incumbent (HV~{inc_hv:.4f}): {incumbent}")

    # Save incumbent: a readable JSON + a benchmark-loadable handler pkl
    # (load_incumbent reads handler._eval_result.metadata['incumbent']).
    with open(INCUMBENT_JSON, "w") as f:
        json.dump({"incumbent": dict(incumbent), "incumbent_hv": inc_hv}, f, indent=2)
    handler = types.SimpleNamespace(
        _eval_result=types.SimpleNamespace(metadata={"incumbent": dict(incumbent)})
    )
    with open(INCUMBENT_PKL, "wb") as f:
        pickle.dump(handler, f)
    print(f"[hpo] saved incumbent -> {INCUMBENT_JSON} and {INCUMBENT_PKL}")

    # --- Stage B: benchmark the tuned version on 12 synth + 3 real ----------
    bcfg = OmegaConf.load("conf/benchmark_hpo_ablation.yaml")  # reuse the 15 problems
    bcfg.output_dir = TUNED_BENCH_DIR
    bcfg.algorithms = OmegaConf.create(
        [{"path": FIXED_CODE, "cls_name": CLS_NAME, "handler_pkl": INCUMBENT_PKL}]
    )
    print(f"[bench] benchmarking tuned {CLS_NAME} -> {TUNED_BENCH_DIR}")
    benchmark_and_plot(bcfg)
    print("[done] now run:  python analyze_moead_hpo.py")


if __name__ == "__main__":
    main()
