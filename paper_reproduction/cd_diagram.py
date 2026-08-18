"""Critical-difference (Demsar 2006) diagram from a benchmark hv_benchmark_log.csv.

Mean ranks are computed per problem (higher final-epoch HV = better = rank 1),
averaged over problems. The Nemenyi critical difference is

    CD = q_alpha * sqrt( k (k+1) / (6 N) )

for k algorithms over N problems. Algorithms whose mean ranks differ by less
than CD are connected by a bar (statistically indistinguishable).

Publication output: titleless, large fonts, generous vertical spacing so the
figure stays legible when placed at \\textwidth in a two-column layout.

Usage:
    python cd_diagram.py csv=paper_data/benchmark_stage2/hv_benchmark_log.csv \
        out=paper_figures/fig7_cd_diagram.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os
_os.makedirs("paper_figures", exist_ok=True)  # gitignored; absent in a fresh clone

DISPLAY = {
    "MOBOImprovedScalarizedEI": "Improved-Scalarized-EI",
    "MOBORandomForestLCBPBI_CD": "RF-LCB-PBI",
    "MOBORandomForestParEGO_Batch_Refined": "RF-ParEGO-Batch",
    "MOBO_MOEAD_EI_Hybrid_Fixed": "MOEAD-EI Hybrid",
    "BoFireQparEGOWrapper": "qParEGO",
    "MOBOEA_HGBR": "HGBR",
    "MOBORFPBIUCB": "RF-PBI-UCB",
    "IOCSAMOCOBRAWrapper": "IOC-SAMO-COBRA",
    "NSGA2Wrapper": "NSGA-II",
    "NSGA3Wrapper": "NSGA-III",
    "RandomSearchMO": "Random Search",
}

# Studentized-range based critical values q_alpha for the two-tailed Nemenyi
# test at alpha = 0.05, indexed by number of algorithms k.
Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
}


def mean_ranks(csv_path):
    df = pd.read_csv(csv_path)
    final = (
        df.sort_values(["Algorithm", "Problem", "Repeat", "Epoch"])
        .groupby(["Algorithm", "Problem", "Repeat"])
        .HV.last()
        .reset_index()
    )
    mean_hv = final.groupby(["Algorithm", "Problem"]).HV.mean().reset_index()
    mat = mean_hv.pivot(index="Problem", columns="Algorithm", values="HV")
    ranks = mat.rank(axis=1, ascending=False)  # higher HV -> rank 1
    return ranks.mean(axis=0), mat.shape[0]


def cliques(ranks_sorted, cd):
    """Maximal groups of (rank-)adjacent methods within CD of each other."""
    k = len(ranks_sorted)
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks_sorted[j + 1] - ranks_sorted[i] <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
    out = []
    for g in groups:
        if not any(h != g and h[0] <= g[0] and h[1] >= g[1] for h in groups):
            out.append(g)
    return out


def plot_cd(avg, n_problems, out_path, fontsize=26):
    avg = avg.sort_values()  # best (lowest rank) first
    names = [DISPLAY.get(a, a) for a in avg.index]
    ranks = avg.to_numpy()
    k = len(ranks)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * n_problems))

    lo, hi = 1, k
    half = (k + 1) // 2
    row_h = 0.72
    fig_h = 2.8 + half * row_h
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-(half * row_h + 0.8), 1.7)
    ax.axis("off")
    # No invert: rank 1 (best) on the LEFT, ticks increase 1..k left-to-right
    # (per reviewer request -- more intuitive, fewer connector crossings).

    # main axis with integer ticks
    ax.plot([lo, hi], [0, 0], "k-", lw=3.6)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.12], "k-", lw=3.6)
        ax.text(r, 0.30, str(r), ha="center", va="bottom", fontsize=fontsize - 2)

    # connector lines + labels, split left/right
    for idx, (name, r) in enumerate(zip(names, ranks)):
        right_side = idx >= half
        row = idx if not right_side else idx - half
        y = -(row + 1) * row_h
        edge = (lo - 0.5) if not right_side else (hi + 0.5)
        ax.plot([r, r], [0, y], "k-", lw=4.6)
        ax.plot([r, edge], [y, y], "k-", lw=4.6)
        ha = "right" if not right_side else "left"
        ax.text(
            edge,
            y,
            f" {name} ({r:.2f}) " if not right_side else f" {name} ({r:.2f}) ",
            ha=ha,
            va="center",
            fontsize=fontsize,
        )

    # CD bar
    bar_y = 1.15
    ax.plot([lo, lo + cd], [bar_y, bar_y], "k-", lw=4.6)
    ax.plot([lo, lo], [bar_y - 0.10, bar_y + 0.10], "k-", lw=4.6)
    ax.plot([lo + cd, lo + cd], [bar_y - 0.10, bar_y + 0.10], "k-", lw=4.6)
    ax.text(
        lo + cd / 2,
        bar_y + 0.18,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=fontsize - 1,
    )

    # cliques (not-significantly-different groups)
    grps = cliques(ranks, cd)
    for gi, (i, j) in enumerate(grps):
        y = -0.28 - gi * 0.36
        ax.plot(
            [ranks[i] - 0.06, ranks[j] + 0.06],
            [y, y],
            color="crimson",
            lw=8.5,
            solid_capstyle="round",
        )

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}  (k={k}, N={n_problems}, CD={cd:.3f})")
    print("mean ranks:")
    for nm, r in zip(names, ranks):
        print(f"  {nm:18s} {r:.3f}")


def main():
    cli = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    csv = cli.get("csv", "paper_data/benchmark_stage2/hv_benchmark_log.csv")
    out = cli.get("out", "paper_figures/fig7_cd_diagram.png")
    avg, n = mean_ranks(csv)
    plot_cd(avg, n, out)


if __name__ == "__main__":
    main()
