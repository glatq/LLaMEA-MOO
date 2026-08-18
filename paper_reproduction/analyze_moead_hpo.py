"""Task 4 (MOEAD-EI extension) -- tuned vs untuned comparison for MOEAD-EI Hybrid.

UNTUNED (LLM defaults) = benchmark_moead_untuned. NOTE: the MOEAD-EI rows inside
benchmark_stage2/stage3 are the TUNED ones -- Table II reports the tuned configuration,
so the tuned rows replaced the untuned ones there. The untuned rows are therefore kept
in their own directory; reading them from stage2/3 would compare tuned against tuned.
TUNED   (SMAC incumbent) = benchmark_moead_tuned (from run_moead_hpo.py).

Both normalized per problem by the same denominator as the paper's normalized-HV
tables (max mean-final-HV over the full stage2/3 algorithm set on that problem),
so the untuned column matches Table III and the delta isolates SMAC's effect.

Run after run_moead_hpo.py finishes:
    python analyze_moead_hpo.py
"""

import os
import pandas as pd

CLS = "MOBO_MOEAD_EI_Hybrid_Fixed"
TUNED_CSV = "paper_data/benchmark_moead_tuned/hv_benchmark_log.csv"
UNTUNED_CSV = "paper_data/benchmark_moead_untuned/hv_benchmark_log.csv"
SCOPES = {
    "SYNTHETIC (12)": "paper_data/benchmark_stage2/hv_benchmark_log.csv",
    "REAL-WORLD (3)": "paper_data/benchmark_stage3/hv_benchmark_log.csv",
}


def mean_final_hv(csv):
    df = pd.read_csv(csv)
    fh = (
        df.sort_values("Epoch")
        .groupby(["Algorithm", "Problem", "Repeat"])
        .HV.last()
        .reset_index()
    )
    return fh.groupby(["Algorithm", "Problem"]).HV.mean().reset_index()


def main():
    if not os.path.exists(TUNED_CSV):
        raise SystemExit(f"Tuned results not found at {TUNED_CSV}. Run run_moead_hpo.py first.")
    tuned = mean_final_hv(TUNED_CSV)
    tuned = tuned[tuned.Algorithm == CLS].set_index("Problem").HV

    if not os.path.exists(UNTUNED_CSV):
        raise SystemExit(f"Untuned results not found at {UNTUNED_CSV}.")
    untuned_all = mean_final_hv(UNTUNED_CSV)
    untuned_all = untuned_all[untuned_all.Algorithm == CLS].set_index("Problem").HV

    for label, stage_csv in SCOPES.items():
        # denominator: per-problem max over the full Table II algorithm set
        allm = mean_final_hv(stage_csv)
        denom = allm.groupby("Problem").HV.max()
        untuned = untuned_all.loc[untuned_all.index.intersection(denom.index)]
        common = untuned.index.intersection(tuned.index)
        if len(common) == 0:
            print(f"\n=== {label} ===  no overlapping problems")
            continue
        u = (untuned.loc[common] / denom.loc[common]).mean()
        t = (tuned.loc[common] / denom.loc[common]).mean()
        pct = 100.0 * (t - u) / u if u else float("nan")
        print(f"\n=== {label}  (normalized HV; per-problem-max denominator) ===")
        print(f"  MOEAD-EI untuned (default): {u:.3f}")
        print(f"  MOEAD-EI tuned   (SMAC):    {t:.3f}")
        print(f"  delta: {t - u:+.3f}  ({pct:+.1f}%)")
        # per-problem detail
        print(f"  {'problem':8s} {'untuned':>8s} {'tuned':>8s} {'delta':>8s}")
        for p in common:
            uu = untuned.loc[p] / denom.loc[p]
            tt = tuned.loc[p] / denom.loc[p]
            print(f"  {p:8s} {uu:8.3f} {tt:8.3f} {tt - uu:+8.3f}")


if __name__ == "__main__":
    main()
