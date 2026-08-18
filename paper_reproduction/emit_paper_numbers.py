"""Emit paper-ready numbers from the benchmark CSVs.

Reads the three benchmark dirs and prints:
  * LaTeX rows for Table III (mean +/- std normalized HV: Phase 2 + Phase 3),
  * LaTeX rows for Table IV (synthetic per-problem raw HV) and V (RE per-problem),
  * an in-text checklist (Phase-1 top-3, synthetic/RE normalized HV, speedups,
    Friedman + Wilcoxon), so body-text numbers can be swept safely.

Best value per column/problem is wrapped in \\mathbf{}; generated algorithms
get a trailing \\dagger. Tables III is on normalized HV; IV/V on raw HV.

Usage:
    python emit_paper_numbers.py \
        stage1=paper_data/benchmark_stage1 \
        stage2=paper_data/benchmark_stage2 \
        stage3=paper_data/benchmark_stage3
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from omegaconf import OmegaConf

LABEL = {
    "MOBOImprovedScalarizedEI": "Improved-Scalarized-EI",
    "MOBORandomForestLCBPBI_CD": "RF-LCB-PBI",
    "MOBORandomForestParEGO_Batch_Refined": "RF-ParEGO-Batch",
    "MOBO_MOEAD_EI_Hybrid_Fixed": "MOEAD-EI Hybrid",
    "BoFireQparEGOWrapper": "qParEGO",
    "IOCSAMOCOBRAWrapper": "IOC-SAMO-COBRA",
    "NSGA2Wrapper": "NSGA-II",
    "NSGA3Wrapper": "NSGA-III",
    "RandomSearchMO": "Random Search",
}
GENERATED = {"MOBOImprovedScalarizedEI", "MOBORandomForestLCBPBI_CD",
             "MOBORandomForestParEGO_Batch_Refined", "MOBO_MOEAD_EI_Hybrid_Fixed"}
GROUPS = [
    ("LLaMEA-generated", ["MOBOImprovedScalarizedEI", "MOBORandomForestLCBPBI_CD",
                          "MOBORandomForestParEGO_Batch_Refined", "MOBO_MOEAD_EI_Hybrid_Fixed"]),
    ("SOTA Bayesian optimisation", ["BoFireQparEGOWrapper", "IOCSAMOCOBRAWrapper"]),
    ("Classical / baseline", ["NSGA2Wrapper", "NSGA3Wrapper", "RandomSearchMO"]),
]
SYNTH = [
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt6",
    "dtlz1",
    "dtlz2",
    "dtlz4",
    "dtlz7",
    "wfg4",
    "wfg7",
    "wfg9",
]
RE = ["re21", "re34", "re37"]


def _final(csv):
    df = pd.read_csv(csv)
    return (
        df.sort_values("Epoch")
        .groupby(["Algorithm", "Problem", "Repeat"])
        .HV.last()
        .reset_index()
    )


def raw_mean_std(csv):
    f = _final(csv)
    g = f.groupby(["Algorithm", "Problem"]).HV.agg(["mean", "std"])
    return g["mean"].unstack(), g["std"].unstack()


def norm_per_problem(csv):
    """Per (algo,problem) max-normalized mean HV -> DataFrame [Problem x Algorithm]."""
    f = _final(csv)
    m = f.groupby(["Algorithm", "Problem"]).HV.mean().unstack(0)  # rows=Problem
    return m.div(m.max(axis=1), axis=0)


def runtime_mean(csv):
    return pd.read_csv(csv).groupby("Algorithm").ExecutionTime.mean()


def name(a):
    return LABEL.get(a, a) + (r"$^\dagger$" if a in GENERATED else "")


def fmt(m, s, best):
    if np.isnan(m):
        return "--"
    inner = rf"{m:.3f} \pm {s:.3f}" if not np.isnan(s) else f"{m:.3f}"
    return f"$\\mathbf{{{inner}}}$" if best else f"${inner}$"


def algos_in_order(present):
    return [a for _, lst in GROUPS for a in lst if a in present]


def table_normalized(stage2, stage3):
    """Table III: per-algo mean +/- std normalized HV for Phase 2 (synth) and Phase 3 (RE)."""
    n2 = norm_per_problem(f"{stage2}/hv_benchmark_log.csv")  # rows=problem
    n3 = norm_per_problem(f"{stage3}/hv_benchmark_log.csv")
    algos = algos_in_order(set(n2.columns))
    m2, s2 = n2.mean(), n2.std()
    m3, s3 = n3.mean(), n3.std()
    best2 = m2[algos].idxmax()
    best3 = m3[algos].idxmax()
    print(
        "% ---- Table III (mean +/- std normalized HV; bold=best per column; dagger=generated) ----"
    )
    for gi, (gname, lst) in enumerate(GROUPS):
        print(f"% {gname}")
        for a in lst:
            if a not in n2.columns:
                continue
            print(
                f"{name(a):24s} & {fmt(m2[a], s2[a], a==best2)} & "
                f"{fmt(m3[a], s3[a], a==best3)} \\\\"
            )
    print()


def table_per_problem(csv, problems, title):
    means, stds = raw_mean_std(csv)
    algos = algos_in_order(set(means.index))
    best = {p: means.loc[algos, p].idxmax() for p in problems if p in means.columns}
    print(f"% ---- {title} (raw HV mean +/- std; bold=best per problem) ----")
    print("% columns: " + " & ".join(problems))
    for gname, lst in GROUPS:
        print(f"% {gname}")
        for a in lst:
            if a not in means.index:
                continue
            cells = []
            for p in problems:
                m = means.loc[a, p] if p in means.columns else np.nan
                s = stds.loc[a, p] if p in stds.columns else np.nan
                cells.append(fmt(m, s, best.get(p) == a))
            print(f"{name(a):24s} & " + " & ".join(cells) + " \\\\")
    print()


def in_text(stage1, stage2, stage3):
    print("% ================= IN-TEXT NUMBER CHECKLIST =================")
    # Phase-1 ranking (normalized HV over the 12 synthetic, the Phase-1 suite)
    n1 = (
        norm_per_problem(f"{stage1}/hv_benchmark_log.csv")
        .mean()
        .sort_values(ascending=False)
    )
    print("Phase-1 ranking (mean normalized HV, all nine generated):")
    for a, v in n1.items():
        print(f"   {LABEL.get(a, a.replace("MOBO", "").lstrip("_")):18s} {v:.3f}")
    top3 = list(n1.index[:3])
    print(f"   -> TOP-3: {[LABEL.get(a, a.replace("MOBO", "").lstrip("_")) for a in top3]}")

    # Phase-2 synthetic + Phase-3 RE normalized HV (headline accuracy)
    for tag, st, probs in [
        ("Phase-2 synthetic", stage2, SYNTH),
        ("Phase-3 real-world", stage3, RE),
    ]:
        nm = (
            norm_per_problem(f"{st}/hv_benchmark_log.csv")
            .mean()
            .sort_values(ascending=False)
        )
        print(f"\n{tag} mean normalized HV:")
        for a, v in nm.items():
            print(f"   {LABEL.get(a, a.replace("MOBO", "").lstrip("_")):18s} {v:.3f}")

    # Runtime + speedups
    print("\nMean runtime per run (s) and speedup vs MOEAD-EI Hybrid:")
    for tag, st in [("synthetic", stage2), ("real-world", stage3)]:
        rt = runtime_mean(f"{st}/runtime_log.csv")
        mo = rt.get("MOBO_MOEAD_EI_Hybrid_Fixed", np.nan)
        qp = rt.get("BoFireQparEGOWrapper", np.nan)
        print(
            f"   [{tag}] MOEAD {mo:,.1f}s | qParEGO {qp:,.1f}s | qParEGO/MOEAD = {qp/mo:.1f}x"
        )

    # Significance (synthetic)
    n2 = norm_per_problem(f"{stage2}/hv_benchmark_log.csv")
    algos = list(n2.columns)
    chi2, p = friedmanchisquare(*[n2[a].values for a in algos])
    ranks = np.vstack(
        [rankdata(-n2.loc[pr].values, method="average") for pr in n2.index]
    )
    mr = dict(zip(algos, ranks.mean(axis=0)))
    print(f"\nFriedman (synthetic): chi2={chi2:.3f}, p={p:.3e}")
    print(
        "Mean ranks:",
        {LABEL.get(a, a): round(mr[a], 2) for a in sorted(algos, key=lambda a: mr[a])},
    )
    ref = "MOBO_MOEAD_EI_Hybrid_Fixed"
    print(f"Wilcoxon (MOEAD vs each, two-sided p, synthetic):")
    for a in algos:
        if a == ref:
            continue
        try:
            _, pw = wilcoxon(n2[ref].values, n2[a].values)
        except ValueError:
            pw = 1.0
        print(f"   vs {LABEL.get(a, a.replace("MOBO", "").lstrip("_")):18s} p={pw:.3e}")


if __name__ == "__main__":
    cli = OmegaConf.from_cli(sys.argv[1:])
    s1 = cli.get("stage1", "paper_data/benchmark_stage1")
    s2 = cli.get("stage2", "paper_data/benchmark_stage2")
    s3 = cli.get("stage3", "paper_data/benchmark_stage3")
    table_normalized(s2, s3)
    table_per_problem(
        f"{s2}/hv_benchmark_log.csv", SYNTH, "Table IV: synthetic per-problem"
    )
    table_per_problem(f"{s3}/hv_benchmark_log.csv", RE, "Table V: RE per-problem")
    in_text(s1, s2, s3)
