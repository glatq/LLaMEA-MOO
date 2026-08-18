"""
Generate convergence + Pareto + runtime plots from benchmark CSVs.

Usage:
    python plot_from_csv.py csv=paper_data/benchmark_stage2/hv_benchmark_log.csv
    python plot_from_csv.py csv=<path> output_dir=<path> dpi=300
    python plot_from_csv.py runtime_csv=<path>          # runtime-only mode
    python plot_from_csv.py pareto_csv=<path>           # Pareto-only mode

Optional filters/overrides:
    algorithms='[MOBO_MOEAD_EI_Hybrid_Fixed, MOBORFPBIUCB, qParEGOWrapper]'   # subset & order
    problems='[zdt1, dtlz2]'                                                   # subset
    labels='{NSGA2Wrapper: NSGA-II, custom: My Algo}'                          # extra label overrides
    runtime_csv=<path>                                                         # explicit runtime CSV
    pareto_csv=<path>                                                          # explicit Pareto CSV
    objectives_csv=<path>                                                      # raw objectives (Pareto fallback)

If `runtime_csv` / `pareto_csv` are not specified, the script looks for
`runtime_log.csv` / `pareto_front_log.csv` in the same directory as `csv`
and uses them automatically. For Pareto plots, if `pareto_front_log.csv` is
absent but `objectives_log.csv` is present, fronts are computed from the raw
objectives.

CSV schemas (produced by benchmark_best_codes.py):
    hv_benchmark_log.csv: Algorithm, Problem, Repeat, Epoch, HV
    runtime_log.csv:      Algorithm, Problem, Repeat, ExecutionTime
    pareto_front_log.csv: Algorithm, Problem, PointIdx, f1..fM
    objectives_log.csv:   Algorithm, Problem, Repeat, Eval, f1..fM
    feasibility_log.csv:  Algorithm, Problem, Repeat, Eval, CV, FeasibleFraction

Notes:
    - Pareto front plots are produced for bi-objective problems (f1, f2).
      Problems with a different objective count are skipped.
    - Default display-name map mirrors the one in benchmark_best_codes.py.
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from omegaconf import OmegaConf

DEFAULT_LABEL_MAP = {
    "NSGA2Wrapper": "NSGA-II",
    "NSGA3Wrapper": "NSGA-III",
    "IOCSAMOCOBRAWrapper": "IOC-SAMO-COBRA",
    "qParEGOWrapper": "qParEGO",
    "BoFireQparEGOWrapper": "qParEGO",
    "qEHVIWrapper": "qEHVI",
    "RandomSearchMO": "Random Search",
    "MOBO_MOEAD_EI_Hybrid_Fixed": "MOEAD-EI Hybrid",
    "MOBOEA_HGBR": "HGBR",
    "MOBORFPBIUCB": "RF-PBI-UCB",
}


def make_convergence_plots(
    csv_path: str,
    output_dir: str | None = None,
    dpi: int = 150,
    algorithms: list[str] | None = None,
    problems: list[str] | None = None,
    extra_labels: dict | None = None,
    figsize: tuple = (10, 6),
    fontsize: int = 11,
    no_title: bool = False,
    legend_ncol: int = 1,
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Resolve labels
    label_map = dict(DEFAULT_LABEL_MAP)
    if extra_labels:
        label_map.update(extra_labels)

    # Resolve algorithm order/filter
    all_algos = list(df["Algorithm"].unique())
    if algorithms:
        plot_algos = [a for a in algorithms if a in all_algos]
        missing = [a for a in algorithms if a not in all_algos]
        if missing:
            print(f"Warning: requested algorithms not in CSV: {missing}")
    else:
        plot_algos = all_algos

    # Resolve problem filter
    all_problems = list(df["Problem"].unique())
    plot_problems = [
        p for p in (problems if problems else all_problems) if p in all_problems
    ]

    print(f"CSV:        {csv_path}")
    print(f"Output dir: {output_dir}")
    print(f"Algorithms ({len(plot_algos)}): {plot_algos}")
    print(f"Problems   ({len(plot_problems)}): {plot_problems}")

    for prob in plot_problems:
        sub = df[df["Problem"] == prob]
        if sub.empty:
            continue

        plt.figure(figsize=figsize)
        plt.rcParams.update({"font.size": fontsize})
        plotted_any = False

        for algo in plot_algos:
            algo_df = sub[sub["Algorithm"] == algo]
            if algo_df.empty:
                continue

            # Pivot to (Repeat, Epoch) -> HV, then mean/std across repeats
            pivoted = algo_df.pivot_table(
                index="Repeat", columns="Epoch", values="HV"
            ).sort_index(axis=1)

            # Forward-fill within a repeat in case of missing trailing epochs
            pivoted = pivoted.ffill(axis=1)

            epochs = pivoted.columns.values
            mean_hv = pivoted.mean(axis=0).values
            std_hv = pivoted.std(axis=0).values

            label = label_map.get(algo, algo)
            line = plt.plot(epochs, mean_hv, label=label, linewidth=2)
            plt.fill_between(
                epochs,
                mean_hv - std_hv,
                mean_hv + std_hv,
                alpha=0.2,
                color=line[0].get_color(),
            )
            plotted_any = True

        if plotted_any:
            if not no_title:
                plt.title(f"Convergence: {prob.upper()}", fontsize=fontsize + 1)
            plt.xlabel("Evaluations", fontsize=fontsize)
            plt.ylabel("Hypervolume (HV)", fontsize=fontsize)
            plt.tick_params(labelsize=fontsize - 1)
            plt.legend(loc="lower right", fontsize=fontsize - 2, ncol=legend_ncol)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            fig_path = os.path.join(output_dir, f"plot_{prob}_convergence.png")
            plt.savefig(fig_path, dpi=dpi)
            print(f"Saved: {fig_path}")

        plt.close()


def make_feasibility_plots(
    csv_path: str,
    output_dir: str | None = None,
    dpi: int = 150,
    algorithms: list[str] | None = None,
    problems: list[str] | None = None,
    extra_labels: dict | None = None,
    figsize: tuple = (10, 6),
    fontsize: int = 11,
    no_title: bool = False,
    legend_ncol: int = 1,
):
    """Feasible-fraction convergence from ``feasibility_log.csv``.

    Columns: Algorithm, Problem, Repeat, Eval, CV, FeasibleFraction. One figure
    per problem showing the running feasible fraction (mean +/- std across
    repeats) vs evaluations. Only produced for constrained benchmarks.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    label_map = dict(DEFAULT_LABEL_MAP)
    if extra_labels:
        label_map.update(extra_labels)

    all_algos = list(df["Algorithm"].unique())
    plot_algos = [a for a in algorithms if a in all_algos] if algorithms else all_algos

    all_problems = list(df["Problem"].unique())
    plot_problems = [
        p for p in (problems if problems else all_problems) if p in all_problems
    ]

    print(f"Feasibility CSV: {csv_path}")
    print(f"Output dir:      {output_dir}")

    for prob in plot_problems:
        sub = df[df["Problem"] == prob]
        if sub.empty:
            continue

        plt.figure(figsize=figsize)
        plt.rcParams.update({"font.size": fontsize})
        plotted_any = False

        for algo in plot_algos:
            algo_df = sub[sub["Algorithm"] == algo]
            if algo_df.empty:
                continue

            # Pivot to (Repeat, Eval) -> FeasibleFraction, then mean/std across repeats
            pivoted = algo_df.pivot_table(
                index="Repeat", columns="Eval", values="FeasibleFraction"
            ).sort_index(axis=1)
            pivoted = pivoted.ffill(axis=1)

            evals = pivoted.columns.values
            mean_f = pivoted.mean(axis=0).values
            std_f = pivoted.std(axis=0).values

            label = label_map.get(algo, algo)
            line = plt.plot(evals, mean_f, label=label, linewidth=2)
            plt.fill_between(
                evals,
                mean_f - std_f,
                mean_f + std_f,
                alpha=0.2,
                color=line[0].get_color(),
            )
            plotted_any = True

        if plotted_any:
            if not no_title:
                plt.title(f"Feasibility: {prob.upper()}", fontsize=fontsize + 1)
            plt.xlabel("Evaluations", fontsize=fontsize)
            plt.ylabel("Feasible fraction (cv == 0)", fontsize=fontsize)
            plt.ylim(-0.02, 1.02)
            plt.tick_params(labelsize=fontsize - 1)
            plt.legend(loc="lower right", fontsize=fontsize - 2, ncol=legend_ncol)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            fig_path = os.path.join(output_dir, f"plot_{prob}_feasibility.png")
            plt.savefig(fig_path, dpi=dpi)
            print(f"Saved: {fig_path}")

        plt.close()


def _objective_columns(df):
    """Return objective columns (f1, f2, ...) present in `df`, in numeric order."""
    cols = [c for c in df.columns if re.fullmatch(r"f\d+", str(c))]
    return sorted(cols, key=lambda c: int(c[1:]))


def _nondominated_mask(Y):
    """Boolean mask of non-dominated rows in `Y` (minimization, any n_obj)."""
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # Points that dominate row i: component-wise <= and strictly < somewhere
        dominates_i = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)
        dominates_i[i] = False
        if dominates_i.any():
            keep[i] = False
    return keep


def _pareto_2d(
    sub,
    prob,
    present,
    plot_algos,
    label_map,
    use_raw,
    output_dir,
    figsize,
    fontsize,
    no_title,
    dpi,
):
    """Bi-objective Pareto front: sorted markers + connecting dashes."""
    f1c, f2c = present
    plt.figure(figsize=figsize)
    plt.rcParams.update({"font.size": fontsize})
    plotted_any = False

    for algo in plot_algos:
        algo_df = sub[sub["Algorithm"] == algo]
        if algo_df.empty:
            continue
        Y = algo_df[[f1c, f2c]].dropna().to_numpy(dtype=float)
        if Y.size == 0:
            continue
        if use_raw:
            Y = Y[_nondominated_mask(Y)]
        Y = Y[Y[:, 0].argsort()]
        label = label_map.get(algo, algo)
        plt.plot(Y[:, 0], Y[:, 1], "o--", label=label, markersize=5)
        plotted_any = True

    if plotted_any:
        if not no_title:
            plt.title(f"Pareto Front: {prob.upper()}", fontsize=fontsize + 1)
        plt.xlabel("Objective 1 ($f_1$)", fontsize=fontsize)
        plt.ylabel("Objective 2 ($f_2$)", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize - 1)
        plt.legend(fontsize=fontsize - 2)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"pareto_{prob}.png")
        plt.savefig(fig_path, dpi=dpi)
        print(f"Saved: {fig_path}")
    plt.close()


def _pareto_3d(
    sub,
    prob,
    present,
    plot_algos,
    label_map,
    use_raw,
    output_dir,
    figsize,
    fontsize,
    no_title,
    dpi,
):
    """Tri-objective Pareto front: 3D scatter, one marker style per algorithm."""
    f1c, f2c, f3c = present
    markers = ["o", "^", "s", "D", "v", "P", "X", "*"]
    fig = plt.figure(figsize=figsize)
    plt.rcParams.update({"font.size": fontsize})
    ax = fig.add_subplot(111, projection="3d")
    plotted_any = False

    for i, algo in enumerate(plot_algos):
        algo_df = sub[sub["Algorithm"] == algo]
        if algo_df.empty:
            continue
        Y = algo_df[[f1c, f2c, f3c]].dropna().to_numpy(dtype=float)
        if Y.size == 0:
            continue
        if use_raw:
            Y = Y[_nondominated_mask(Y)]
        label = label_map.get(algo, algo)
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            Y[:, 2],
            label=label,
            marker=markers[i % len(markers)],
            s=14,
            alpha=0.6,
            edgecolors="none",
            depthshade=True,
        )
        plotted_any = True

    if plotted_any:
        if not no_title:
            ax.set_title(f"Pareto Front: {prob.upper()}", fontsize=fontsize + 1)
        ax.set_xlabel("$f_1$", fontsize=fontsize, labelpad=3)
        ax.set_ylabel("$f_2$", fontsize=fontsize, labelpad=3)
        ax.set_zlabel("$f_3$", fontsize=fontsize, labelpad=3)
        ax.tick_params(labelsize=fontsize - 3)
        ax.view_init(elev=22, azim=45)
        ax.legend(fontsize=fontsize - 3, loc="upper left")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"pareto_{prob}.png")
        plt.savefig(fig_path, dpi=dpi)
        print(f"Saved: {fig_path}")
    plt.close()


def make_pareto_plots(
    pareto_csv_path: str | None,
    output_dir: str,
    dpi: int = 300,
    algorithms: list[str] | None = None,
    problems: list[str] | None = None,
    extra_labels: dict | None = None,
    objectives_csv_path: str | None = None,
    figsize: tuple = (8, 6),
    fontsize: int = 11,
    no_title: bool = False,
):
    """One Pareto-front PNG per problem (`pareto_<prob>.png`).

    Bi-objective problems get a 2D front; tri-objective problems get a 3D
    scatter. Prefers `pareto_front_log.csv` (front points already extracted).
    Falls back to `objectives_log.csv`, computing the combined non-dominated
    front per (algorithm, problem) across repeats.
    """
    use_raw = False
    if pareto_csv_path and os.path.isfile(pareto_csv_path):
        df = pd.read_csv(pareto_csv_path)
        src = pareto_csv_path
    elif objectives_csv_path and os.path.isfile(objectives_csv_path):
        df = pd.read_csv(objectives_csv_path)
        src = objectives_csv_path
        use_raw = True
    else:
        print(
            "No Pareto/objective CSV found, skipping Pareto plots "
            f"(looked for: {pareto_csv_path}, {objectives_csv_path})"
        )
        return

    obj_cols = _objective_columns(df)
    if len(obj_cols) < 2:
        print(f"Pareto CSV {src} has < 2 objective columns; skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    label_map = dict(DEFAULT_LABEL_MAP)
    if extra_labels:
        label_map.update(extra_labels)

    all_algos = list(df["Algorithm"].unique())
    plot_algos = [a for a in algorithms if a in all_algos] if algorithms else all_algos

    all_problems = list(df["Problem"].unique())
    plot_problems = [
        p for p in (problems if problems else all_problems) if p in all_problems
    ]

    print(f"Pareto CSV: {src}{' (raw objectives)' if use_raw else ''}")
    print(f"Output dir: {output_dir}")

    for prob in plot_problems:
        sub = df[df["Problem"] == prob]
        if sub.empty:
            continue

        # Objective columns actually populated for this problem
        present = [c for c in obj_cols if sub[c].notna().any()]
        if len(present) == 2:
            _pareto_2d(
                sub,
                prob,
                present,
                plot_algos,
                label_map,
                use_raw,
                output_dir,
                figsize,
                fontsize,
                no_title,
                dpi,
            )
        elif len(present) == 3:
            _pareto_3d(
                sub,
                prob,
                present,
                plot_algos,
                label_map,
                use_raw,
                output_dir,
                figsize,
                fontsize,
                no_title,
                dpi,
            )
        else:
            print(
                f"Skipping Pareto plot for {prob} "
                f"(n_obj={len(present)} not in {{2, 3}})"
            )


def make_runtime_plots(
    runtime_csv_path: str,
    output_dir: str,
    dpi: int = 150,
    algorithms: list[str] | None = None,
    problems: list[str] | None = None,
    extra_labels: dict | None = None,
):
    """Two figures from runtime_log.csv:
    1. Per-problem bar chart (one PNG per problem) — mean runtime ± std across repeats
    2. Summary: mean runtime per algorithm averaged across all problems
    """
    if not os.path.isfile(runtime_csv_path):
        print(f"Runtime CSV not found, skipping runtime plots: {runtime_csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(runtime_csv_path)

    label_map = dict(DEFAULT_LABEL_MAP)
    if extra_labels:
        label_map.update(extra_labels)

    all_algos = list(df["Algorithm"].unique())
    if algorithms:
        plot_algos = [a for a in algorithms if a in all_algos]
    else:
        plot_algos = all_algos

    all_problems = list(df["Problem"].unique())
    plot_problems = [
        p for p in (problems if problems else all_problems) if p in all_problems
    ]

    print(f"Runtime CSV: {runtime_csv_path}")

    # 1. Per-problem bar charts
    for prob in plot_problems:
        sub = df[df["Problem"] == prob]
        if sub.empty:
            continue

        means, stds, labels_for_x = [], [], []
        for algo in plot_algos:
            algo_df = sub[sub["Algorithm"] == algo]
            if algo_df.empty:
                continue
            means.append(algo_df["ExecutionTime"].mean())
            stds.append(algo_df["ExecutionTime"].std() if len(algo_df) > 1 else 0.0)
            labels_for_x.append(label_map.get(algo, algo))

        if not means:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_for_x, rotation=30, ha="right")
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(
            f"Runtime per algorithm: {prob.upper()}  (mean ± std over repeats)"
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"runtime_{prob}.png")
        plt.savefig(fig_path, dpi=dpi)
        print(f"Saved: {fig_path}")
        plt.close()

    # 2. Summary: mean runtime per algorithm averaged across problems
    summary_means, summary_stds, summary_labels = [], [], []
    for algo in plot_algos:
        algo_df = df[df["Algorithm"] == algo]
        if algo_df.empty:
            continue
        # Per-problem mean first, then mean of those means (so problems are weighted equally)
        per_problem = algo_df.groupby("Problem")["ExecutionTime"].mean()
        summary_means.append(per_problem.mean())
        summary_stds.append(per_problem.std() if len(per_problem) > 1 else 0.0)
        summary_labels.append(label_map.get(algo, algo))

    if summary_means:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(summary_means))
        ax.bar(x, summary_means, yerr=summary_stds, capsize=4, alpha=0.85, color="C0")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_labels, rotation=30, ha="right")
        ax.set_ylabel("Wall-clock time per problem (s)")
        ax.set_title("Runtime summary: mean ± std across problems")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, "runtime_summary.png")
        plt.savefig(fig_path, dpi=dpi)
        print(f"Saved: {fig_path}")
        plt.close()


if __name__ == "__main__":
    cli = OmegaConf.from_cli(sys.argv[1:])

    csv_path = cli.get("csv", None)
    runtime_csv_path = cli.get("runtime_csv", None)
    pareto_csv_path = cli.get("pareto_csv", None)
    objectives_csv_path = cli.get("objectives_csv", None)
    feasibility_csv_path = cli.get("feasibility_csv", None)

    if (
        csv_path is None
        and runtime_csv_path is None
        and pareto_csv_path is None
        and feasibility_csv_path is None
    ):
        print(__doc__)
        sys.exit(
            "ERROR: pass csv=<path>, runtime_csv=<path>, pareto_csv=<path>, "
            "and/or feasibility_csv=<path>"
        )

    output_dir = cli.get("output_dir", None)
    dpi = int(cli.get("dpi", 150))
    algos = list(cli.algorithms) if "algorithms" in cli else None
    probs = list(cli.problems) if "problems" in cli else None
    extra_labels = OmegaConf.to_container(cli.labels) if "labels" in cli else None
    figsize_raw = cli.get("figsize", None)
    figsize = tuple(figsize_raw) if figsize_raw is not None else (10, 6)
    fontsize = int(cli.get("fontsize", 11))
    no_title = bool(cli.get("no_title", False))
    legend_ncol = int(cli.get("legend_ncol", 1))

    if csv_path is not None:
        make_convergence_plots(
            csv_path=str(csv_path),
            output_dir=output_dir,
            dpi=dpi,
            algorithms=algos,
            problems=probs,
            extra_labels=extra_labels,
            figsize=figsize,
            fontsize=fontsize,
            no_title=no_title,
            legend_ncol=legend_ncol,
        )
        # Auto-detect sibling CSVs next to the HV CSV if not explicitly provided
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if runtime_csv_path is None:
            candidate = os.path.join(csv_dir, "runtime_log.csv")
            if os.path.isfile(candidate):
                runtime_csv_path = candidate
        if pareto_csv_path is None:
            candidate = os.path.join(csv_dir, "pareto_front_log.csv")
            if os.path.isfile(candidate):
                pareto_csv_path = candidate
        if objectives_csv_path is None:
            candidate = os.path.join(csv_dir, "objectives_log.csv")
            if os.path.isfile(candidate):
                objectives_csv_path = candidate
        if feasibility_csv_path is None:
            candidate = os.path.join(csv_dir, "feasibility_log.csv")
            if os.path.isfile(candidate):
                feasibility_csv_path = candidate

    if feasibility_csv_path is not None:
        feas_out = output_dir or os.path.dirname(os.path.abspath(feasibility_csv_path))
        make_feasibility_plots(
            csv_path=str(feasibility_csv_path),
            output_dir=feas_out,
            dpi=dpi,
            algorithms=algos,
            problems=probs,
            extra_labels=extra_labels,
            figsize=figsize,
            fontsize=fontsize,
            no_title=no_title,
            legend_ncol=legend_ncol,
        )

    if runtime_csv_path is not None:
        rt_out = output_dir or os.path.dirname(os.path.abspath(runtime_csv_path))
        make_runtime_plots(
            runtime_csv_path=str(runtime_csv_path),
            output_dir=rt_out,
            dpi=dpi,
            algorithms=algos,
            problems=probs,
            extra_labels=extra_labels,
        )

    if pareto_csv_path is not None or objectives_csv_path is not None:
        ref_path = pareto_csv_path or objectives_csv_path
        pareto_out = output_dir or os.path.dirname(os.path.abspath(ref_path))
        make_pareto_plots(
            pareto_csv_path=str(pareto_csv_path) if pareto_csv_path else None,
            output_dir=pareto_out,
            dpi=int(cli.get("pareto_dpi", 300)),
            algorithms=algos,
            problems=probs,
            extra_labels=extra_labels,
            objectives_csv_path=(
                str(objectives_csv_path) if objectives_csv_path else None
            ),
            figsize=tuple(cli.get("pareto_figsize", (8, 6))),
            fontsize=fontsize,
            no_title=no_title,
        )
