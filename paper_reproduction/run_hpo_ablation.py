"""Task 4 -- before/after SMAC-HPO ablation (the BEFORE half).

Runs the three systematically-generated algorithms with their LLM DEFAULT
hyperparameters (no SMAC incumbent) on the same 12 synthetic + 3 real-world
problems as the main benchmark, into paper_data/benchmark_hpo_default.

The AFTER half (SMAC-tuned) already lives in benchmark_stage2 (synthetic) and
benchmark_stage3 (real). After this run completes, compare with:

    python analyze_hpo_ablation.py

Run from the llamea-moo-3.12 env at the repo root:
    python run_hpo_ablation.py
"""

from omegaconf import OmegaConf
from benchmark_best_codes import benchmark_and_plot

if __name__ == "__main__":
    cfg = OmegaConf.load("conf/benchmark_hpo_ablation.yaml")
    print("HPO-ablation (default-config) benchmark:")
    print(OmegaConf.to_yaml(cfg))
    benchmark_and_plot(cfg)
