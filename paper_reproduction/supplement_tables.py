"""LaTeX tables of an ablation for the GECCO supplement, from analyze_ablation.py's output.

Reads path.csv, steps.csv, vs_end.csv and baseline.csv of one analysed ablation and
writes these tabulars into out=:

    <name>_rounds.tex   average rank (path seeds) of every candidate of every round, per
                        direction; the chosen one in bold, ties marked
    <name>_steps.tex    every step of both paths on the validation seeds
    <name>_vs_end.tex   every path configuration against the end algorithm
    <name>_confirm.tex  the confirmation per problem, if confirm.csv exists

Usage, from the repo root:
    python paper_reproduction/analyze_ablation.py results=paper_data/ablation_moead out=X
    python paper_reproduction/supplement_tables.py analysis=X name=moead out=Y
"""

import math
import pathlib
import sys

import pandas as pd
from omegaconf import OmegaConf

DEFAULTS = {"analysis": "benchmark_results/ablation_fast/analysis", "name": "moead",
            "out": "benchmark_results/supplement_tables"}


def _p(x):
    return f"{x:.3f}" if x >= 0.001 else f"{x:.0e}"


def rounds_table(path):
    """Rows: rounds; columns: the block changed; cell: average rank of that candidate."""
    blocks = sorted(path.ChangedBlock.unique())
    out = []
    for direction in ("forward", "backward"):
        d = path[path.Direction == direction]
        out.append(f"\\multicolumn{{{len(blocks) + 1}}}{{@{{}}l}}{{\\emph{{{direction} path}}}} \\\\")
        for rnd, g in d.groupby("Round"):
            cells = []
            for b in blocks:
                r = g[g.ChangedBlock == b]
                if r.empty:
                    cells.append("")
                    continue
                r = r.iloc[0]
                s = f"{r.AvgRank:.2f}"
                if r.Chosen:
                    s = f"\\textbf{{{s}}}" + ("$^{\\ast}$" if r.Tie else "")
                cells.append(s)
            out.append(f"{rnd} & " + " & ".join(cells) + " \\\\")
        if direction == "forward":
            out.append("\\midrule")
    head = ("\\begin{tabular}{@{}r" + "r" * len(blocks) + "@{}}\n\\toprule\n"
            "Round & " + " & ".join(str(b) for b in blocks) + " \\\\\n\\midrule")
    return head + "\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n"


def steps_table(steps):
    rows = []
    for direction in ("forward", "backward"):
        d = steps[steps.Direction == direction].sort_values("Round")
        rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{direction} path}}}} \\\\")
        for r in d.itertuples():
            rows.append(f"{r.Round} & {r.Step} & \\texttt{{{r.To}}} & {r.Higher}/15 & "
                        f"{r.MedianRelPct:+.2f} & {_p(r.Wilcoxon)} & {_p(r.Holm)} & "
                        f"{r.SigBetter}/{r.SigWorse} \\\\")
        if direction == "forward":
            rows.append("\\midrule")
    head = ("\\begin{tabular}{@{}r l l r r r r r@{}}\n\\toprule\n"
            "Round & Step & Configuration & Higher & Median (\\%) & $p$ & Holm & Better/worse \\\\\n"
            "\\midrule")
    return head + "\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def vs_end_table(vs_end):
    rows = [f"\\texttt{{{r.Configuration}}} & {r.AvgRank:.2f} & {r.Higher}/15 & "
            f"{r.MedianRelPct:+.2f} & {_p(r.Wilcoxon)} & {_p(r.Holm)} & "
            f"{r.SigBetter}/{r.SigWorse} \\\\" for r in vs_end.itertuples()]
    head = ("\\begin{tabular}{@{}l r r r r r r@{}}\n\\toprule\n"
            "Configuration & Rank & Higher & Median (\\%) & $p$ & Holm & Better/worse \\\\\n"
            "\\midrule")
    return head + "\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def _mean_sd(m, sd):
    """Mean to four significant digits and the SD to the same decimals."""
    d = max(0, 3 - math.floor(math.log10(abs(m)))) if m else 3
    return f"{m:.{d}f} ({sd:.{d}f})"


def confirm_table(confirm):
    """Per problem: mean (SD) final hypervolume of the confirmed configuration and of the
    end algorithm on the confirmation seeds, relative difference, Mann-Whitney p and its
    Holm adjustment over the problems. Bold: the higher mean when Holm's p < 0.05."""
    rows = []
    for r in confirm.itertuples():
        a, b = _mean_sd(r.Config, r.ConfigSD), _mean_sd(r.Against, r.AgainstSD)
        if r.Holm < 0.05:
            if r.Config > r.Against:
                a = f"\\textbf{{{a}}}"
            else:
                b = f"\\textbf{{{b}}}"
        name = r.Problem.upper()
        rows.append(f"{name} & {a} & {b} & {r.RelPct:+.2f} & {_p(r.MannWhitney)} & "
                    f"{_p(r.Holm)} \\\\")
    head = ("\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\n"
            "Problem & Configuration & End algorithm & Difference (\\%) & $p$ & Holm \\\\\n"
            "\\midrule")
    return head + "\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli(sys.argv[1:]))
    a, out = pathlib.Path(cfg.analysis), pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    note = "% Generated by paper_reproduction/supplement_tables.py; do not edit by hand.\n"
    (out / f"{cfg.name}_rounds.tex").write_text(note + rounds_table(pd.read_csv(a / "path.csv")))
    (out / f"{cfg.name}_steps.tex").write_text(note + steps_table(pd.read_csv(a / "steps.csv")))
    (out / f"{cfg.name}_vs_end.tex").write_text(note + vs_end_table(pd.read_csv(a / "vs_end.csv")))
    if (a / "confirm.csv").exists():
        (out / f"{cfg.name}_confirm.tex").write_text(
            note + confirm_table(pd.read_csv(a / "confirm.csv")))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
