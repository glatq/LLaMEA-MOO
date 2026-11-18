"""Analyse a Fawcett & Hoos ablation (``ablation/run_ablation.py``).

    python paper_reproduction/analyze_ablation.py [config=conf/ablation.yaml]
        [results=paper_data/ablation_moead] [out=paper_figures/ablation_moead]

``config`` gives the pair, the seeds and the problems; ``results`` the directory
holding the runner's logs (default: the config's ``output_dir``). Writes to ``out``:

* ``path.csv``         every round of both paths on the path seeds: each candidate's
                       average rank and the one kept (``path_log.csv``);
* ``steps.csv``        each step along both paths, on the validation seeds;
* ``vs_end.csv``       each configuration on the paths against the end algorithm,
                       on the validation seeds;
* ``per_problem.csv``  mean and standard deviation of the final HV of each
                       configuration on the paths, per problem, validation seeds;
* ``baseline.csv``     DDMTOLab's own start algorithm against the modular start and
                       the end, per problem, on the path seeds;
* ``runtime.csv``      mean run time per configuration;
* ``summary.md``       the numbers quoted in the paper;
* ``ablation_path.{pdf,png}``   median HV change from the start along both paths;
* ``ablation_steps.{pdf,png}``  each step's HV change on each problem.

A step of the forward path puts one block of the end algorithm into the current
configuration (``+search``); a step of the backward path puts the start
algorithm's block back (``-search``: the end's search taken out).

Statistics. Per problem: the two-sided Mann-Whitney U test on the final HV of the
seeds, alpha 0.05. Across problems: the relative difference of mean final HV,
``(b - a) / |a|``, on each problem, and the two-sided Wilcoxon signed-rank test on
those differences (relative, so that problems with a large HV scale do not
dominate); Holm's correction over the steps of one direction, and over the
configurations compared with the end.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

from ablation.config import BLOCKS, END, ISEI_BLOCKS, ISEI_END, ISEI_START, START  # noqa: E402

PAIRS = {
    "moead": dict(blocks=BLOCKS, start=START, end=END, baseline="DDMTOLabMOEADEGO",
                  start_name="MOEA/D-EGO", end_name="MOEAD-EI Hybrid"),
    "isei": dict(blocks=ISEI_BLOCKS, start=ISEI_START, end=ISEI_END, baseline="DDMTOLabParEGO",
                 start_name="ParEGO", end_name="Improved-Scalarized-EI"),
}
ALPHA = 0.05
FORWARD, BACKWARD = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
WORSE, NO_CHANGE, BETTER = "#e34948", "#f0efec", "#2a78d6"  # diverging pair, gray midpoint
INK, MUTED = "#0b0b0b", "#52514e"


def holm(p):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def sign(p_value, rel):
    """``+`` / ``-`` for a significant difference, ``=`` otherwise."""
    return ("+" if rel > 0 else "-") if p_value < ALPHA else "="


class Runs:
    """The final HV of every seed, per (configuration, problem), for a set of seeds."""

    def __init__(self, status, seeds, problems):
        s = status[status.Repeat.isin(seeds)]
        self.n_seeds, self.problems = len(seeds), list(problems)
        self.hv = {k: g.FinalHV.to_numpy() for k, g in s.groupby(["Algorithm", "Problem"])}

    def has(self, config):
        return all(len(self.hv.get((config, p), ())) == self.n_seeds for p in self.problems)

    def values(self, config, problem):
        v = self.hv.get((config, problem))
        if v is None or len(v) != self.n_seeds:
            raise SystemExit(f"{config} on {problem}: {0 if v is None else len(v)} runs, "
                             f"expected {self.n_seeds}")
        return v

    def mean(self, config):
        return pd.Series({p: self.values(config, p).mean() for p in self.problems})

    def compare(self, a, b):
        """``b`` against ``a``: summary over problems, relative differences, per-problem p."""
        ma, mb = self.mean(a), self.mean(b)
        rel = (mb - ma) / ma.abs()
        mw = pd.Series({p: mannwhitneyu(self.values(b, p), self.values(a, p),
                                        alternative="two-sided").pvalue for p in self.problems})
        sig = mw < ALPHA
        p = 1.0 if np.allclose(rel, 0) else wilcoxon(rel).pvalue
        return dict(Higher=int((rel > 0).sum()), MedianRelPct=100 * rel.median(),
                    MeanRelPct=100 * rel.mean(), Wilcoxon=p,
                    SigBetter=int((sig & (rel > 0)).sum()), SigWorse=int((sig & (rel < 0)).sum())), rel, mw


def path_configs(path_log, start, end):
    """Each direction's kept configurations in order, from its first: ``[(block, config)]``."""
    out = {}
    for d, first in (("forward", start), ("backward", end)):
        kept = path_log[(path_log.Direction == d) & path_log.Chosen].sort_values("Round")
        out[d] = [(None, first)] + list(zip(kept.ChangedBlock.astype(int), kept.Candidate))
    return out


def step_label(direction, block):
    return ("+" if direction == "forward" else "-") + block


# ------------------------------------------------------------------- tables
def steps_table(runs, paths, blocks):
    rows, rels, mws = [], {}, {}
    for d, seq in paths.items():
        part = []
        for r, ((_, a), (blk, b)) in enumerate(zip(seq, seq[1:]), start=1):
            stats, rel, mw = runs.compare(a, b)
            label = step_label(d, blocks[blk - 1])
            part.append(dict(Direction=d, Round=r, Block=blocks[blk - 1], Step=label,
                             From=a, To=b, **stats))
            rels[(d, r)], mws[(d, r)] = rel, mw
        for row, adj in zip(part, holm([x["Wilcoxon"] for x in part])):
            row["Holm"] = adj
        rows += part
    return pd.DataFrame(rows), rels, mws


def vs_end_table(runs, configs, end, blocks):
    means = pd.DataFrame({c: runs.mean(c) for c in configs}).T  # configuration x problem
    rank = means.rank(axis=0, ascending=False, method="average").mean(axis=1)
    rows = []
    for c in configs:
        if c == end:
            continue
        stats, _, _ = runs.compare(end, c)
        parts = ",".join(b for b, x in zip(blocks, c) if x == "T") or "-"
        rows.append(dict(Configuration=c, EndBlocks=parts, AvgRank=rank[c], **stats))
    df = pd.DataFrame(rows)
    df["Holm"] = holm(df.Wilcoxon)
    return df.sort_values("AvgRank").reset_index(drop=True), rank


def per_problem_table(runs, configs, start, end):
    rows = []
    for c in configs:
        _, rel_s, mw_s = runs.compare(start, c)
        _, rel_e, mw_e = runs.compare(end, c)
        for p in runs.problems:
            v = runs.values(c, p)
            rows.append(dict(Configuration=c, Problem=p, Mean=v.mean(), SD=v.std(ddof=1),
                             VsStart=sign(mw_s[p], rel_s[p]), VsEnd=sign(mw_e[p], rel_e[p])))
    return pd.DataFrame(rows)


def baseline_table(status, runtime, problems, seeds, pair):
    """DDMTOLab's start algorithm against the modular start and the end (path seeds)."""
    runs = Runs(status, seeds, problems)
    base, start, end = pair["baseline"], pair["start"], pair["end"]
    if not runs.has(base):
        return None, {}
    s_stats, s_rel, s_mw = runs.compare(base, start)
    e_stats, e_rel, e_mw = runs.compare(base, end)
    mb, ms, me = runs.mean(base), runs.mean(start), runs.mean(end)
    table = pd.DataFrame({
        "Problem": problems, "DDMTOLab": mb.values, "ModularStart": ms.values,
        "StartRelPct": 100 * s_rel.values, "StartMannWhitney": s_mw.values,
        "End": me.values, "EndRelPct": 100 * e_rel.values, "EndMannWhitney": e_mw.values,
    })
    rt = runtime[runtime.Repeat.isin(seeds)].groupby("Algorithm").ExecutionTime.mean()
    return table, dict(start=s_stats, end=e_stats,
                       runtime={k: rt.get(k, np.nan) for k in (base, start, end)})


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    return plt


def plot_path(runs, paths, steps, blocks, pair, out):
    """Median HV change from the start (validation seeds) along both paths."""
    plt = _style()
    start, end = pair["start"], pair["end"]
    end_level = runs.compare(start, end)[0]["MedianRelPct"]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), sharey=True)
    for ax, d, color in zip(axes, ("forward", "backward"), (FORWARD, BACKWARD)):
        seq = paths[d]
        y = [runs.compare(start, c)[0]["MedianRelPct"] for _, c in seq]
        x = np.arange(len(seq))
        sig = steps[steps.Direction == d].set_index("Round").Holm < ALPHA
        ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axhline(end_level, color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
        for level, name in ((0, pair["start_name"]), (end_level, pair["end_name"])):
            ax.text(len(seq) - 0.3, level, name, ha="left", va="center", fontsize=6, color=MUTED,
                    clip_on=False)
        ax.plot(x, y, color=color, lw=2, zorder=2)
        for i in x:
            filled = i > 0 and bool(sig.get(i, False))
            ax.plot(i, y[i], "o", ms=5.5, mec=color, mfc=color if filled else "white", mew=1.5, zorder=3)
        first = pair["start_name"] if d == "forward" else pair["end_name"]
        ax.set_xticks(x)
        ax.set_xticklabels([first] + [step_label(d, blocks[b - 1]) for b, _ in seq[1:]],
                           rotation=25, ha="right")
        for lab, i in zip(ax.get_xticklabels(), x):
            if i > 0 and bool(sig.get(i, False)):
                lab.set_color(INK)
                lab.set_fontweight("bold")
        ax.set_xlim(-0.4, len(seq) - 0.4)
        ax.set_title(f"{d} path: {first} → "
                     f"{pair['end_name'] if d == 'forward' else pair['start_name']}", loc="left", color=INK)
        ax.set_ylabel(f"median HV change\nvs {pair['start_name']} (%)")
    axes[0].margins(y=0.15)
    fig.text(0.99, 0.0, "filled marker, bold step: significant over problems "
             "(Wilcoxon on relative HV, Holm, α = 0.05)", ha="right", va="bottom",
             fontsize=6, color=MUTED)
    fig.tight_layout(h_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ablation_path.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_steps(steps, rels, mws, problems, out):
    """Each step's relative HV change on each problem, both directions."""
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", [WORSE, NO_CHANGE, BETTER])
    vals = np.concatenate([100 * r.values for r in rels.values()])
    lim = max(2.0, float(np.ceil(np.percentile(np.abs(vals), 95))))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    n = {d: (steps.Direction == d).sum() for d in ("forward", "backward")}
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 0.22 * (n["forward"] + n["backward"]) + 1.3),
                             gridspec_kw=dict(height_ratios=[n["forward"], n["backward"]]))
    for ax, d in zip(axes, ("forward", "backward")):
        s = steps[steps.Direction == d].sort_values("Round")
        M = np.array([100 * rels[(d, r)].reindex(problems).values for r in s.Round])
        P = np.array([mws[(d, r)].reindex(problems).values for r in s.Round])
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if P[i, j] < ALPHA:
                    ax.text(j, i, "+" if M[i, j] > 0 else "−", ha="center", va="center",
                            fontsize=7, color="white" if abs(M[i, j]) > 0.6 * lim else INK)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f"{lab}{' *' if h < ALPHA else ''}" for lab, h in zip(s.Step, s.Holm)])
        ax.set_xticks(range(len(problems)))
        ax.set_xticklabels([p.upper() for p in problems] if d == "backward" else [])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{d} path", loc="left", color=INK)
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015, extend="both")
    cb.set_label("HV change of the step (%)")
    cb.outline.set_visible(False)
    fig.text(0.01, 0.0, "+ / −: significant on that problem (Mann-Whitney, α = 0.05);  "
             "*: significant over problems (Wilcoxon, Holm)", fontsize=6, color=MUTED)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ablation_steps.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- driver
def main(cfg):
    pair = PAIRS[cfg.get("pair", "moead")]
    blocks, start, end = pair["blocks"], pair["start"], pair["end"]
    results = pathlib.Path(cfg.get("results", None) or cfg.output_dir)
    if not results.is_absolute():
        results = ROOT / results
    out = pathlib.Path(cfg.get("out", None) or results / "analysis")
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    problems = [p.name for p in cfg.problems]
    budget = int(cfg.budget)

    status = pd.read_csv(results / "run_status_log.csv")
    runtime = pd.read_csv(results / "runtime_log.csv")
    path_log = pd.read_csv(results / "path_log.csv")
    bad = status[(status.Evaluations < budget) | status.TimedOut.astype(bool) | status.Error.notna()]
    fallbacks = status.AskFallbacks.fillna(0)

    path_log.to_csv(out / "path.csv", index=False)
    paths = path_configs(path_log, start, end)
    configs = list(dict.fromkeys(c for seq in paths.values() for _, c in seq))
    val = Runs(status, list(cfg.validation_seeds), problems)

    steps, rels, mws = steps_table(val, paths, blocks)
    steps.to_csv(out / "steps.csv", index=False)
    vs_end, rank = vs_end_table(val, configs, end, blocks)
    vs_end.to_csv(out / "vs_end.csv", index=False)
    per_problem_table(val, configs, start, end).to_csv(out / "per_problem.csv", index=False)
    rt = runtime.groupby("Algorithm").ExecutionTime.agg(["mean", "median", "count"])
    rt.to_csv(out / "runtime.csv")
    base, base_stats = baseline_table(status, runtime, problems, list(cfg.path_seeds), pair)
    if base is not None:
        base.to_csv(out / "baseline.csv", index=False)

    plot_path(val, paths, steps, blocks, pair, out)
    plot_steps(steps, rels, mws, problems, out)

    # ------------------------------------------------------------- summary
    def stat_line(s):
        return (f"higher on {s['Higher']}/{len(problems)}, median {s['MedianRelPct']:+.2f}%, "
                f"Wilcoxon p={s['Wilcoxon']:.3g}; per problem {s['SigBetter']} better / "
                f"{s['SigWorse']} worse")

    lines = [f"# Ablation {pair['start_name']} ({start}) -> {pair['end_name']} ({end})", "",
             f"Runs: {len(status)} ({status.Algorithm.nunique()} configurations), "
             f"failed: {len(bad)}, runs with a model-fit fallback: {int((fallbacks > 0).sum())}.",
             f"Path seeds {list(cfg.path_seeds)}, validation seeds {list(cfg.validation_seeds)}; "
             f"{len(configs)} configurations on the paths.", ""]
    pilot = results / "path_log_pilot.csv"
    if pilot.exists():
        pl = pd.read_csv(pilot)
        agree = {d: (set(pl[(pl.Direction == d) & pl.Chosen].Candidate)
                     == set(path_log[(path_log.Direction == d) & (path_log.Round == 1)
                                     & path_log.Chosen].Candidate)) for d in ("forward", "backward")}
        lines += [f"Pilot (seed 1) round-1 picks equal the path's: {agree}.", ""]
    for d in ("forward", "backward"):
        s = steps[steps.Direction == d]
        lines += [f"## {d} path (validation seeds)", "",
                  "| round | step | configuration | higher | median | Wilcoxon p | Holm | "
                  "per problem better/worse |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r.Round} | {r.Step} | {r.To} | {r.Higher}/{len(problems)} | "
                  f"{r.MedianRelPct:+.2f}% | {r.Wilcoxon:.3g} | {r.Holm:.3g} | {r.SigBetter}/{r.SigWorse} |"
                  for r in s.itertuples()]
        lines.append("")
    best = vs_end.iloc[0]
    lines += [f"## Path configurations against {pair['end_name']} (validation seeds)", "",
              f"Average rank among the {len(configs)} path configurations: {pair['end_name']} "
              f"{rank[end]:.2f}; best {best.Configuration} ({best.EndBlocks}) {best.AvgRank:.2f}.", "",
              "| configuration | end's blocks | avg rank | higher | median | Wilcoxon p | Holm | "
              "per problem better/worse |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.Configuration} | {r.EndBlocks} | {r.AvgRank:.2f} | {r.Higher}/{len(problems)} | "
              f"{r.MedianRelPct:+.2f}% | {r.Wilcoxon:.3g} | {r.Holm:.3g} | {r.SigBetter}/{r.SigWorse} |"
              for r in vs_end.itertuples()]
    lines.append("")
    if base is not None:
        r = base_stats["runtime"]
        lines += [f"## DDMTOLab's {pair['start_name']} (path seeds)", "",
                  f"Modular start vs DDMTOLab: {stat_line(base_stats['start'])}.",
                  f"{pair['end_name']} vs DDMTOLab: {stat_line(base_stats['end'])}.",
                  f"Mean run time (s): DDMTOLab {r[pair['baseline']]:.0f}, modular start "
                  f"{r[start]:.0f}, {pair['end_name']} {r[end]:.0f}.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    cli = OmegaConf.from_cli(sys.argv[1:])
    base_cfg = OmegaConf.load(cli.pop("config", str(ROOT / "conf" / "ablation.yaml")))
    main(OmegaConf.merge(base_cfg, cli))
