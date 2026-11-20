"""Table 2 of the GECCO paper: the two generated algorithms against the baselines.

Collects the final hypervolume of every run (method, problem, run) from the logs:

    paper_data/benchmark_stage{2,3}/          MOEAD-EI Hybrid, Improved-Scalarized-EI (its
                                              explicit seeds), qNParEGO, NSGA-II, NSGA-III and
                                              random search, runs 1-5
    paper_data/benchmark_qehvi_stage{2,3}/    qEHVI, runs 1-5; a problem found in a
                                              qehvi_rerun folder is taken from there instead
    paper_data/ablation_moead/, ablation_isei/  DDMTOLab's MOEA/D-EGO and ParEGO (the ablation
                                              runner's baseline stage), seeds 1-5
    the extra folders                         runs 6-10 of every method (README, "Five more
                                              runs"), when present

A folder that does not exist is skipped and named in summary.md. IOC-SAMO-COBRA is left out
(README); the RF variants come in with rf=true. Run times come only from runs made under the
run controls (one thread per run, one run per physical core): the timing pass, qEHVI, the
ablation baselines and the extra runs. The original Phase 2 and 3 runs are not timed.

Each problem's mean HVs are divided by the best mean of any method on that problem
(normalised HV; qEHVI's HV on DTLZ1 is 0, so a relative difference to it is undefined).
For each generated algorithm G against each other method B:
  * across problems: two-sided Wilcoxon signed-rank test on the 15 differences of
    normalised HV, Holm over all these comparisons;
  * per problem: two-sided Mann-Whitney U test on the runs, Holm over the 15 problems of
    the pair.

Usage, from the repo root:
    python paper_reproduction/benchmark_table.py out=benchmark_results/table2 [rf=true]
"""

import pathlib
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]

SYNTH = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz1", "dtlz2", "dtlz4", "dtlz7",
         "wfg4", "wfg7", "wfg9"]
RE = ["re21", "re34", "re37"]
PROBLEMS = SYNTH + RE

HYBRID, ISEI = "MOBO_MOEAD_EI_Hybrid_Fixed", "MOBOImprovedScalarizedEI"
RF = ["MOBORandomForestLCBPBI_CD", "MOBORandomForestParEGO_Batch_Refined"]
BASELINES = ["BoFireQparEGOWrapper", "BoFireQEHVIWrapper", "DDMTOLabMOEADEGO",
             "DDMTOLabParEGO", "NSGA2Wrapper", "NSGA3Wrapper", "RandomSearchMO"]
DDMTOLAB = ["DDMTOLabMOEADEGO", "DDMTOLabParEGO"]
LEFT_OUT = {"IOCSAMOCOBRAWrapper"}
LABEL = {
    HYBRID: "MOEAD-EI Hybrid", ISEI: "Improved-Scalarized-EI",
    "MOBORandomForestLCBPBI_CD": "RF-LCB-PBI",
    "MOBORandomForestParEGO_Batch_Refined": "RF-ParEGO-Batch",
    "BoFireQparEGOWrapper": "qNParEGO", "BoFireQEHVIWrapper": "qEHVI",
    "DDMTOLabMOEADEGO": "MOEA/D-EGO", "DDMTOLabParEGO": "ParEGO",
    "NSGA2Wrapper": "NSGA-II", "NSGA3Wrapper": "NSGA-III", "RandomSearchMO": "Random search",
}

DEFAULTS = {
    "stage": ["paper_data/benchmark_stage2", "paper_data/benchmark_stage3"],
    "qehvi": ["paper_data/benchmark_qehvi_stage2", "paper_data/benchmark_qehvi_stage3"],
    "qehvi_rerun": ["paper_data/benchmark_qehvi_rerun_wfg7",
                    "paper_data/benchmark_qehvi_rerun_re37"],
    "ablation": ["paper_data/ablation_moead", "paper_data/ablation_isei"],
    "baseline_seeds": [1, 2, 3, 4, 5],
    "timing": ["paper_data/benchmark_timing_stage2", "paper_data/benchmark_timing_stage3"],
    "extra": ["paper_data/benchmark_extra_stage2", "paper_data/benchmark_extra_stage3"],
    "extra_isei": ["paper_data/benchmark_extra_isei_syn", "paper_data/benchmark_extra_isei_real"],
    "extra_ablation": ["paper_data/ablation_moead_extra", "paper_data/ablation_isei_extra"],
    "extra_seeds": [6, 7, 8, 9, 10],
    "rf": False,
    "alpha": 0.05,
    "out": "benchmark_results/table2",
}


def holm(p):
    """Holm-adjusted p values, in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (len(p) - i) * p[k])
        adj[k] = min(1.0, running)
    return adj


def _path(d):
    p = pathlib.Path(d)
    return p if p.is_absolute() else ROOT / p


def bench_runs(d, offset=0):
    """Final HV, evaluations and time-limit flag of each run in a benchmark folder."""
    d = _path(d)
    h = pd.read_csv(d / "hv_benchmark_log.csv")
    last = h.sort_values("Epoch").groupby(["Algorithm", "Problem", "Repeat"]).tail(1)
    runs = pd.DataFrame({"Method": last.Algorithm, "Problem": last.Problem,
                         "Run": last.Repeat.astype(int) + offset, "HV": last.HV,
                         "Evaluations": last.Epoch.astype(int)})
    runs["TimedOut"] = False
    status = d / "run_status_log.csv"
    if status.exists():
        st = pd.read_csv(status)
        flag = {(r.Algorithm, r.Problem, int(r.Repeat) + offset): bool(r.TimedOut)
                for r in st.itertuples()}
        runs["TimedOut"] = [flag.get(k, False) for k in
                            zip(runs.Method, runs.Problem, runs.Run)]
    return runs


def bench_times(d, offset=0):
    t = pd.read_csv(_path(d) / "runtime_log.csv")
    return pd.DataFrame({"Method": t.Algorithm, "Problem": t.Problem,
                         "Run": t.Repeat.astype(int) + offset, "Time": t.ExecutionTime})


def ablation_runs(d, seeds):
    """DDMTOLab's baselines from an ablation folder (exact runs; Repeat is the seed)."""
    d = _path(d)
    st = pd.read_csv(d / "run_status_log.csv")
    st = st[st.Algorithm.isin(DDMTOLAB) & st.Repeat.isin(seeds)]
    runs = pd.DataFrame({"Method": st.Algorithm, "Problem": st.Problem,
                         "Run": st.Repeat.astype(int), "HV": st.FinalHV,
                         "Evaluations": st.Evaluations, "TimedOut": st.TimedOut.astype(bool)})
    t = pd.read_csv(d / "runtime_log.csv")
    t = t[t.Algorithm.isin(DDMTOLAB) & t.Repeat.isin(seeds)]
    times = pd.DataFrame({"Method": t.Algorithm, "Problem": t.Problem,
                          "Run": t.Repeat.astype(int), "Time": t.ExecutionTime})
    return runs, times


def collect(cfg):
    """All runs and all controlled run times, plus the folders that were missing."""
    runs, times, missing = [], [], []

    def have(d):
        if (_path(d) / "run_status_log.csv").exists() or (_path(d) / "hv_benchmark_log.csv").exists():
            return True
        missing.append(str(d))
        return False

    for d in cfg.stage:
        if have(d):
            runs.append(bench_runs(d))
    rerun = [bench_runs(d) for d in cfg.qehvi_rerun if have(d)]
    redone = {p for r in rerun for p in r.Problem.unique()}
    for d in cfg.qehvi:
        if have(d):
            r, t = bench_runs(d), bench_times(d)
            runs.append(r[~r.Problem.isin(redone)])
            times.append(t[~t.Problem.isin(redone)])
    runs += rerun
    times += [bench_times(d) for d in cfg.qehvi_rerun if (_path(d) / "runtime_log.csv").exists()]
    for d in cfg.ablation:
        if have(d):
            r, t = ablation_runs(d, list(cfg.baseline_seeds))
            runs.append(r)
            times.append(t)
    for d in cfg.timing:
        if have(d):
            times.append(bench_times(d))
    for d in cfg.extra:
        if have(d):
            runs.append(bench_runs(d, offset=5))
            times.append(bench_times(d, offset=5))
    for d in cfg.extra_isei:
        if have(d):
            runs.append(bench_runs(d))
            times.append(bench_times(d))
    for d in cfg.extra_ablation:
        if have(d):
            r, t = ablation_runs(d, list(cfg.extra_seeds))
            runs.append(r)
            times.append(t)

    runs = pd.concat(runs, ignore_index=True)
    times = pd.concat(times, ignore_index=True) if times else pd.DataFrame(
        columns=["Method", "Problem", "Run", "Time"])
    keep = [HYBRID, ISEI] + (RF if cfg.rf else []) + BASELINES
    runs = runs[runs.Method.isin(keep) & runs.Problem.isin(PROBLEMS)]
    times = times[times.Method.isin(keep) & times.Problem.isin(PROBLEMS)]
    dup = runs.duplicated(["Method", "Problem", "Run"])
    if dup.any():
        raise SystemExit(f"duplicate runs:\n{runs[dup].head()}")
    return runs, times, missing


def analyse(runs, times, alpha=0.05):
    """Per-problem statistics, the method summary and the tests."""
    present = [m for m in [HYBRID, ISEI] + RF + BASELINES if m in set(runs.Method)]
    complete = [m for m in present
                if set(runs[runs.Method == m].Problem) >= set(PROBLEMS)]
    stats = runs.groupby(["Method", "Problem"]).HV.agg(["mean", "std", "count"])
    mean = stats["mean"].unstack(0).reindex(index=PROBLEMS, columns=complete)
    norm = mean.div(mean.max(axis=1), axis=0)
    rank = mean.rank(axis=1, ascending=False)

    tm = times.groupby(["Method", "Problem"]).Time.mean().unstack(0)
    summary = pd.DataFrame({
        "NormSynthetic": norm.loc[SYNTH].mean(),
        "NormRE": norm.loc[RE].mean(),
        "AvgRank": rank.mean(),
        "TimeSynthetic": tm.reindex(index=SYNTH, columns=complete).mean(),
        "TimeRE": tm.reindex(index=RE, columns=complete).mean(),
        "Runs": runs.groupby("Method").Run.count().reindex(complete),
        "TimedOutRuns": runs.groupby("Method").TimedOut.sum().reindex(complete).astype(int),
    })

    pairs = [(g, b) for g in (HYBRID, ISEI) if g in complete
             for b in complete if b != g and not (g == ISEI and b == HYBRID)]
    tests = []
    for g, b in pairs:
        diff = norm[g] - norm[b]
        p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        per = []
        for prob in PROBLEMS:
            x = runs[(runs.Method == g) & (runs.Problem == prob)].HV
            y = runs[(runs.Method == b) & (runs.Problem == prob)].HV
            same = x.nunique() == 1 and y.nunique() == 1 and x.iloc[0] == y.iloc[0]
            per.append(1.0 if same else mannwhitneyu(x, y, alternative="two-sided").pvalue)
        adj = holm(per)
        up = (mean[g] > mean[b]).to_numpy()
        tests.append({
            "Generated": g, "Other": b, "Higher": int((diff > 0).sum()),
            "MedianDiffPct": 100 * float(np.median(diff)), "Wilcoxon": p,
            "BetterRaw": int(((np.array(per) < alpha) & up).sum()),
            "WorseRaw": int(((np.array(per) < alpha) & ~up).sum()),
            "BetterHolm": int(((adj < alpha) & up).sum()),
            "WorseHolm": int(((adj < alpha) & ~up).sum()),
            "MinRuns": int(min(stats.loc[g]["count"].min(), stats.loc[b]["count"].min())),
        })
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["Holm"] = holm(tests.Wilcoxon)
    per_problem = (stats.reset_index().rename(columns={"mean": "Mean", "std": "SD",
                                                       "count": "Runs"}))
    per_problem["Norm"] = [norm.at[p, m] if m in norm.columns else np.nan
                           for m, p in zip(per_problem.Method, per_problem.Problem)]
    return summary, tests, per_problem, complete


def _name(m):
    return LABEL.get(m, m)


def _p(x):
    return f"{x:.2g}" if x >= 0.001 else f"{x:.1e}"


def write(out, runs, times, missing, summary, tests, per_problem, complete):
    out = _path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary.rename(index=_name).to_csv(out / "summary.csv")
    tests.assign(Generated=tests.Generated.map(_name), Other=tests.Other.map(_name)).to_csv(
        out / "tests.csv", index=False)
    per_problem.assign(Method=per_problem.Method.map(_name)).to_csv(out / "per_problem.csv",
                                                                    index=False)
    absent = [m for m in [HYBRID, ISEI] + BASELINES if m not in complete]
    lines = ["# Table 2: generated algorithms against the baselines", ""]
    lines.append(f"Runs per method and problem: {sorted(set(per_problem.Runs.astype(int)))}; "
                 f"methods without all 15 problems (left out): "
                 f"{', '.join(_name(m) for m in absent) or 'none'}.")
    lines.append(f"Missing folders: {', '.join(missing) or 'none'}.")
    lines.append("")
    lines.append("| method | norm. HV synthetic | norm. HV RE | avg rank | time synthetic (s) "
                 "| time RE (s) | runs | runs at the time limit |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m, r in summary.iterrows():
        lines.append(f"| {_name(m)} | {r.NormSynthetic:.3f} | {r.NormRE:.3f} | {r.AvgRank:.2f} "
                     f"| {r.TimeSynthetic:.0f} | {r.TimeRE:.0f} | {int(r.Runs)} "
                     f"| {int(r.TimedOutRuns)} |")
    lines += ["", "## Tests (Wilcoxon over the 15 problems on normalised HV, Holm over all rows; "
              "per problem Mann-Whitney, Holm over the 15 problems of the pair)", "",
              "| generated | against | higher on | median diff (pp) | Wilcoxon p | Holm "
              "| per problem better/worse (raw) | (Holm) | min runs |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in tests.itertuples():
        lines.append(f"| {_name(r.Generated)} | {_name(r.Other)} | {r.Higher}/15 "
                     f"| {r.MedianDiffPct:+.2f} | {_p(r.Wilcoxon)} | {_p(r.Holm)} "
                     f"| {r.BetterRaw}/{r.WorseRaw} | {r.BetterHolm}/{r.WorseHolm} "
                     f"| {r.MinRuns} |")
    lines += ["", "## Run time ratios (mean time per run, controlled runs)", ""]
    for g in (HYBRID, ISEI):
        if g not in summary.index:
            continue
        for b in summary.index:
            if b in (HYBRID, ISEI):
                continue
            rs = summary.at[b, "TimeSynthetic"] / summary.at[g, "TimeSynthetic"]
            rr = summary.at[b, "TimeRE"] / summary.at[g, "TimeRE"]
            lines.append(f"- {_name(b)} / {_name(g)}: synthetic {rs:.1f}x, RE {rr:.1f}x")
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    # LaTeX tabular for the main table, one row per method (\input it inside a table).
    tex = ["% Generated by paper_reproduction/benchmark_table.py; do not edit by hand.",
           "\\begin{tabular}{@{}l r r r r r@{}}",
           "\\toprule",
           "& \\multicolumn{2}{c}{Norm.\\ HV} & & \\multicolumn{2}{c}{Time per run (s)} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}",
           "Method & Synth. & RE & Rank & Synth. & RE \\\\",
           "\\midrule"]
    best = {c: summary[c].max() for c in ("NormSynthetic", "NormRE")}
    best["AvgRank"] = summary["AvgRank"].min()

    def cell(c, v, fmt):
        s = format(v, fmt)
        return f"\\textbf{{{s}}}" if np.isclose(v, best[c]) else s

    def secs(v):
        return "$<$1" if v < 1 else f"{v:,.0f}"

    group = None
    for m, r in summary.iterrows():
        g = 0 if m in [HYBRID, ISEI] + RF else 1 if m in BASELINES[:4] else 2
        if group is not None and g != group:
            tex.append("\\midrule")
        group = g
        tex.append(f"{_name(m)} & {cell('NormSynthetic', r.NormSynthetic, '.3f')} & "
                   f"{cell('NormRE', r.NormRE, '.3f')} & {cell('AvgRank', r.AvgRank, '.2f')} & "
                   f"{secs(r.TimeSynthetic)} & {secs(r.TimeRE)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {out}")


def main():
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli(sys.argv[1:]))
    runs, times, missing = collect(cfg)
    summary, tests, per_problem, complete = analyse(runs, times, cfg.alpha)
    write(cfg.out, runs, times, missing, summary, tests, per_problem, complete)


if __name__ == "__main__":
    main()
