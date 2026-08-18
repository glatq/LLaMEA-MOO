"""Task 4 -- before/after SMAC-HPO comparison (the analysis half).

AFTER  (SMAC-tuned / incumbent)  = benchmark_stage2 (synthetic) + benchmark_stage3 (real).
BEFORE (LLM defaults)            = benchmark_hpo_default  (produced by run_hpo_ablation.py).

Both configs are normalized per problem by the SAME denominator used for the
paper's normalized-HV tables: the maximum mean-final-HV over the full benchmark
algorithm set on that problem. This puts the default config on the exact scale
of Table III, so the 'tuned' column reproduces the paper numbers and the delta
isolates the contribution of SMAC hyperparameter optimization.

Run after the BEFORE benchmark finishes:
    python analyze_hpo_ablation.py
"""

import os
import pandas as pd

ALGOS = {
    "MOBOImprovedScalarizedEI": "Improved-Scalarized-EI",
    "MOBORandomForestLCBPBI_CD": "RF-LCB-PBI",
    "MOBORandomForestParEGO_Batch_Refined": "RF-ParEGO-Batch",
}

DEFAULT_CSV = "paper_data/benchmark_hpo_default/hv_benchmark_log.csv"
TUNED = {
    "SYNTHETIC (12)": "paper_data/benchmark_stage2/hv_benchmark_log.csv",
    "REAL-WORLD (3)": "paper_data/benchmark_stage3/hv_benchmark_log.csv",
}


def mean_final_hv(csv):
    """Mean over repeats of the final-epoch HV, per (Algorithm, Problem)."""
    df = pd.read_csv(csv)
    fh = (
        df.sort_values("Epoch")
        .groupby(["Algorithm", "Problem", "Repeat"])
        .HV.last()
        .reset_index()
    )
    return fh.groupby(["Algorithm", "Problem"]).HV.mean().reset_index()


def compare(label, tuned_csv, default_csv):
    tuned = mean_final_hv(tuned_csv)
    default = mean_final_hv(default_csv)
    # per-problem normalizer = max mean-final-HV over ALL algos in the tuned run
    denom = tuned.groupby("Problem").HV.max()

    print(f"\n=== {label}  (normalized HV; per-problem-max denominator) ===")
    print(f"{'Algorithm':24s} {'default':>8s} {'tuned':>8s} {'delta':>8s} {'delta%':>8s}")
    rows = []
    for key, lab in ALGOS.items():
        t = tuned[tuned.Algorithm == key].set_index("Problem").HV / denom
        d = default[default.Algorithm == key].set_index("Problem").HV / denom
        common = t.index.intersection(d.index)
        if len(common) == 0:
            print(f"{lab:24s}  (no overlapping problems -- did the BEFORE run finish?)")
            continue
        tv = t.loc[common].mean()
        dv = d.loc[common].mean()
        pct = 100.0 * (tv - dv) / dv if dv else float("nan")
        rows.append((lab, dv, tv, tv - dv, pct))
        print(f"{lab:24s} {dv:8.3f} {tv:8.3f} {tv - dv:8.3f} {pct:7.1f}%")
    return rows


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_CSV):
        raise SystemExit(
            f"BEFORE results not found at {DEFAULT_CSV}. "
            "Run `python run_hpo_ablation.py` first."
        )
    for label, tuned_csv in TUNED.items():
        compare(label, tuned_csv, DEFAULT_CSV)
