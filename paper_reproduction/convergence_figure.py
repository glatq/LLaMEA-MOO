"""Convergence of every benchmark method for the GECCO supplement, from the per-evaluation logs.

One panel per problem (15), mean hypervolume of the evaluated points against the number of
evaluations, over the same runs as benchmark_table.py (Table 2): ten runs per method and
problem. The per-evaluation logs come from the same folders as there, with these files:

    benchmark folders                      hv_benchmark_log.csv
    paper_data/ablation_{moead,isei}/       hv_baseline_log.csv (DDMTOLab's baseline runs,
                                            seeds 1-5; the per-evaluation logs of the other
                                            ablation runs are not committed)
    paper_data/ablation_*_extra/            hv_benchmark_log.csv (seeds 6-10)

A run that stopped at the time limit has fewer evaluations; a method's line ends where its
first run stopped, so every point of a line is a mean over all its runs. The script checks
that the last point of every run is the final hypervolume benchmark_table.py uses.

Usage, from the repo root:
    python paper_reproduction/convergence_figure.py out=X/convergence.pdf
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchmark_table as bt  # noqa: E402

DEFAULTS = {**bt.DEFAULTS, "out": "benchmark_results/benchmark_table/convergence.pdf"}

METHODS = [bt.HYBRID, bt.ISEI, "BoFireQparEGOWrapper", "BoFireQEHVIWrapper",
           "DDMTOLabMOEADEGO", "DDMTOLabParEGO", "NSGA2Wrapper", "NSGA3Wrapper",
           "RandomSearchMO"]
# Categorical slots 1-6 in fixed order for the generated algorithms and the BO baselines
# (the first two as in Figure 2); markers as a second cue, since lines cross. The simple
# baselines in grey, told apart by dash pattern.
STYLE = {
    bt.HYBRID: dict(color="#2a78d6", marker="D", lw=1.4, zorder=6),
    bt.ISEI: dict(color="#eb6834", marker="D", lw=1.4, zorder=5),
    "BoFireQparEGOWrapper": dict(color="#1baf7a", marker="o"),
    "BoFireQEHVIWrapper": dict(color="#eda100", marker="s"),
    "DDMTOLabMOEADEGO": dict(color="#e87ba4", marker="^"),
    "DDMTOLabParEGO": dict(color="#008300", marker="v"),
    "NSGA2Wrapper": dict(color="#8a8984", ls=(0, (4, 2))),
    "NSGA3Wrapper": dict(color="#8a8984", ls=(0, (1, 1.5))),
    "RandomSearchMO": dict(color="#b9b8b2", ls="-"),
}
INK, MUTED = "#2b2b29", "#6b6a66"


def _hv(d, name="hv_benchmark_log.csv", offset=0):
    h = pd.read_csv(bt._path(d) / name)
    return pd.DataFrame({"Method": h.Algorithm, "Problem": h.Problem,
                         "Run": h.Repeat.astype(int) + offset, "Epoch": h.Epoch.astype(int),
                         "HV": h.HV})


def collect(cfg):
    """Per-evaluation HV of every run that Table 2 uses."""
    parts = [_hv(d) for d in cfg.stage]
    rerun = [_hv(d) for d in cfg.qehvi_rerun]
    redone = {p for r in rerun for p in r.Problem.unique()}
    parts += [h[~h.Problem.isin(redone)] for h in (_hv(d) for d in cfg.qehvi)] + rerun
    for d in cfg.ablation:
        h = _hv(d, "hv_baseline_log.csv")
        parts.append(h[h.Run.isin(list(cfg.baseline_seeds))])
    parts += [_hv(d, offset=5) for d in cfg.extra]
    parts += [_hv(d) for d in cfg.extra_isei]
    for d in cfg.extra_ablation:
        h = _hv(d)
        parts.append(h[h.Run.isin(list(cfg.extra_seeds))])
    h = pd.concat(parts, ignore_index=True)
    h = h[h.Method.isin(METHODS) & h.Problem.isin(bt.PROBLEMS)]
    if h.duplicated(["Method", "Problem", "Run", "Epoch"]).any():
        raise SystemExit("duplicate evaluations in the logs")
    return h


def check(h, cfg):
    """The last point of every run must be the final HV of Table 2."""
    runs, _, _ = bt.collect(cfg)
    runs = runs[runs.Method.isin(METHODS)].set_index(["Method", "Problem", "Run"])
    last = h.sort_values("Epoch").groupby(["Method", "Problem", "Run"]).tail(1)
    last = last.set_index(["Method", "Problem", "Run"])
    if set(last.index) != set(runs.index):
        raise SystemExit("the per-evaluation logs and Table 2 cover different runs")
    gap = np.abs(last.HV - runs.HV.reindex(last.index)).max()
    if not gap <= 1e-9:
        raise SystemExit(f"final HV differs from Table 2's by up to {gap}")
    return len(last)


def curves(h):
    """Mean HV per method, problem and evaluation, up to the shortest run."""
    end = h.groupby(["Method", "Problem", "Run"]).Epoch.max().groupby(["Method", "Problem"]).min()
    h = h.join(end.rename("End"), on=["Method", "Problem"])
    h = h[h.Epoch <= h.End]
    return h.groupby(["Method", "Problem", "Epoch"]).HV.mean(), end


def main():
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli(sys.argv[1:]))
    h = collect(cfg)
    n = check(h, cfg)
    mean, end = curves(h)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(5, 3, figsize=(7.0, 7.7))
    for ax, prob in zip(axes.ravel(), bt.PROBLEMS):
        for m in METHODS:
            s = dict(STYLE[m])
            y = mean.loc[(m, prob)]
            ax.plot(y.index, y.values, color=s["color"], ls=s.get("ls", "-"),
                    lw=s.get("lw", 1.0), zorder=s.get("zorder", 3),
                    marker=s.get("marker"), markevery=(40, 80), markersize=3.2,
                    markeredgewidth=0)
            if end[(m, prob)] < 400:
                ax.plot(y.index[-1], y.values[-1], marker="x", color=s["color"], ms=4.5,
                        mew=1.2, zorder=7)
        ax.set_title(prob.upper(), fontsize=7.5, loc="left", color=INK, pad=2)
        ax.tick_params(labelsize=6, colors=MUTED, length=2)
        ax.grid(True, color="#e4e3df", linewidth=0.5, zorder=0)
        ax.set_xlim(0, 400)
        ax.set_xticks([0, 100, 200, 300, 400])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b2")
    for ax in axes[-1]:
        ax.set_xlabel("Evaluations", fontsize=7, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean HV", fontsize=7, color=INK)
    handles = [plt.Line2D([], [], color=STYLE[m]["color"], ls=STYLE[m].get("ls", "-"),
                          lw=STYLE[m].get("lw", 1.0), marker=STYLE[m].get("marker"),
                          markersize=3.5, markeredgewidth=0) for m in METHODS]
    fig.legend(handles, [bt.LABEL[m] for m in METHODS], loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.9, w_pad=1.0)
    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    short = end[end < 400]
    print(f"{n} runs; lines ending before 400 evaluations: "
          + ", ".join(f"{bt.LABEL[m]} {p.upper()} {e}" for (m, p), e in short.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
