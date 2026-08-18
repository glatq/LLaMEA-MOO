"""Re-benchmark the SMAC-tuned MOEAD-EI Hybrid with INDEPENDENT repeats.

Why this exists: the first tuned-MOEAD benchmark ran on the GCP VM, where
Python 3.12 multiprocessing defaults to 'fork'. The MOEAD code uses the global
np.random stream with no per-repeat reseeding, so forked workers inherited the
same RNG state and the 5 "repeats" collided (identical clusters) -> invalid
variance / significance. macOS defaults to 'spawn' (independent), which is why
every other result is clean.

This script forces 'spawn' so repeats are independent on ANY platform, and turns
export ON so we also capture per-evaluation objectives / Pareto fronts (needed to
regenerate the Pareto-grid figures). It reuses the saved SMAC incumbent -- no new
HPO. The output dir is cleared first so results don't merge with the old
(compromised) run.

Run from the llamea-moo-3.12 env at the repo root (Mac or GCP):
    python run_moead_tuned_benchmark.py
"""

import multiprocessing
import os
import shutil

import numpy as np
import sklearn.tree._tree as _skt  # DTYPE shim, harmless if not needed

if not hasattr(_skt, "DTYPE"):
    _skt.DTYPE = np.float32

from omegaconf import OmegaConf
from benchmark_best_codes import benchmark_and_plot

FIXED_CODE = "MO bench/MOBO_MOEAD_EI_Hybrid_fixed.py"
CLS_NAME = "MOBO_MOEAD_EI_Hybrid_Fixed"
INCUMBENT_PKL = "benchmark_results/moead_hpo/MOBO_MOEAD_EI_Hybrid_Fixed_incumbent_handler.pkl"
OUT_DIR = "benchmark_results/moead_tuned"

if __name__ == "__main__":
    # Independent repeats on any platform (GCP/Linux defaults to fork otherwise).
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    print("multiprocessing start method:", multiprocessing.get_start_method())

    if not os.path.exists(INCUMBENT_PKL):
        raise SystemExit(
            f"Incumbent not found at {INCUMBENT_PKL}. Run run_moead_hpo.py first "
            "(or copy benchmark_results/moead_hpo/ here)."
        )

    # Clear the old (compromised) output so logs don't merge with it.
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        print(f"cleared {OUT_DIR}")

    cfg = OmegaConf.load("conf/benchmark_hpo_ablation.yaml")  # 15 problems + budget/repeat
    cfg.output_dir = OUT_DIR
    cfg.export = OmegaConf.create({"objectives": True, "pareto": True})  # for Pareto figs
    cfg.algorithms = OmegaConf.create(
        [{"path": FIXED_CODE, "cls_name": CLS_NAME, "handler_pkl": INCUMBENT_PKL}]
    )
    print(f"Re-benchmarking tuned {CLS_NAME} (spawn, export on) -> {OUT_DIR}")
    benchmark_and_plot(cfg)
    print("[done] download paper_data/benchmark_moead_tuned/ and verify repeats are distinct.")
