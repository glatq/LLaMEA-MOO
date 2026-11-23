"""Time against accuracy for the GECCO paper (Figure 2), from benchmark_table.py's summary.csv.

One column wide, two panels stacked: the 12 synthetic problems and the 3 RE problems.
x: mean wall-clock time per run under the run controls (log scale); y: mean normalised
hypervolume (each problem's mean final HV divided by the best mean of any method on it).
The two generated algorithms are drawn in colour, the baselines in grey, every point
labelled with its method.

Usage, from the repo root:
    python paper_reproduction/benchmark_table.py out=X
    python paper_reproduction/time_accuracy_figure.py summary=X/summary.csv out=Y/time_accuracy.pdf
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DEFAULTS = {"summary": "benchmark_results/benchmark_table/summary.csv",
            "out": "benchmark_results/benchmark_table/time_accuracy.pdf"}

GENERATED = {"MOEAD-EI Hybrid": "#2a78d6", "Improved-Scalarized-EI": "#eb6834"}
BO = ["qNParEGO", "qEHVI", "MOEA/D-EGO", "ParEGO"]
SIMPLE = ["NSGA-II", "NSGA-III", "Random search"]
SHORT = {"Improved-Scalarized-EI": "ISEI"}
GREY = "#8a8984"
INK = "#2b2b29"
MUTED = "#6b6a66"
# Label offsets in points (dx, dy, horizontal alignment), per panel, to keep labels apart.
OFFSETS = {
    "synthetic": {"MOEAD-EI Hybrid": (7, -2, "left"), "Improved-Scalarized-EI": (7, -3, "left"),
                  "qNParEGO": (-2, 6, "center"), "qEHVI": (6, -3, "left"),
                  "MOEA/D-EGO": (6, -3, "left"), "ParEGO": (-6, -3, "right"),
                  "NSGA-II": (6, -3, "left"), "NSGA-III": (6, -3, "left"),
                  "Random search": (6, -3, "left")},
    "RE": {"MOEAD-EI Hybrid": (7, -3, "left"), "Improved-Scalarized-EI": (0, 6, "center"),
           "qNParEGO": (-6, -6, "right"), "qEHVI": (-6, -3, "right"),
           "MOEA/D-EGO": (0, -11, "center"), "ParEGO": (-6, -3, "right"),
           "NSGA-II": (6, 1, "left"), "NSGA-III": (6, -6, "left"),
           "Random search": (6, -3, "left")},
}


def draw(ax, s, col_hv, col_t, panel, title):
    for m in BO + SIMPLE:
        if m not in s.index:
            continue
        filled = m in BO
        ax.scatter(s.loc[m, col_t], s.loc[m, col_hv], s=26, marker="s" if filled else "o",
                   facecolor=GREY if filled else "white", edgecolor=GREY, linewidth=1.0, zorder=3)
    for m, c in GENERATED.items():
        if m in s.index:
            ax.scatter(s.loc[m, col_t], s.loc[m, col_hv], s=60, marker="D", facecolor=c,
                       edgecolor="white", linewidth=1.2, zorder=4)
    for m in s.index:
        dx, dy, ha = OFFSETS[panel].get(m, (6, -3, "left"))
        gen = m in GENERATED
        ax.annotate(SHORT.get(m, m), (s.loc[m, col_t], s.loc[m, col_hv]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=6.5,
                    color=INK if gen else MUTED, fontweight="bold" if gen else "normal")
    ax.set_xscale("log")
    ax.set_title(title, fontsize=7.5, loc="left", color=INK)
    ax.set_ylabel("Normalized HV", fontsize=7, color=INK)
    ax.tick_params(labelsize=6.5, colors=MUTED, length=2)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.set_xlim(0.08, 1e5)


def main():
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli(sys.argv[1:]))
    s = pd.read_csv(cfg.summary).set_index("Method")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.33, 4.3), sharex=True)
    draw(axes[0], s, "NormSynthetic", "TimeSynthetic", "synthetic", "(a) 12 synthetic problems")
    draw(axes[1], s, "NormRE", "TimeRE", "RE", "(b) 3 RE problems")
    axes[1].set_xlabel("Mean time per run (s, log scale)", fontsize=7, color=INK)
    fig.tight_layout(h_pad=0.8)
    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
