import os
import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.config import Config

Config.warnings["not_compiled"] = False


def load_incumbent(handler_pkl_path):
    """Load SMAC incumbent hyperparameters from a handler pkl file."""
    try:
        with open(handler_pkl_path, "rb") as f:
            handler = pickle.load(f)
        incumbent = handler._eval_result.metadata.get("incumbent")
        if incumbent:
            print(f"    Loaded incumbent from {handler_pkl_path}: {incumbent}")
        return incumbent
    except Exception as e:
        print(f"    Warning: Could not load incumbent from {handler_pkl_path}: {e}")
        return None


def _append_rows_to_csv(path, rows):
    """Append `rows` (list of dicts) to the CSV at `path`, creating it if needed.

    Concatenation (rather than a plain file append) lets pandas align columns,
    so problems with different objective counts (f1..fM) stay consistent.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def feasible_fraction_curve(cv_history):
    """Running fraction of feasible (cv == 0) evaluations after each evaluation.

    cv_history is the per-evaluation constraint violation cv = sum(max(0, G)).
    Returns an array of the same length whose i-th entry is the fraction of the
    first (i+1) evaluations that were feasible -- a feasibility "convergence"
    curve analogous to the HV curve.
    """
    cv = np.asarray(cv_history, dtype=float)
    if cv.size == 0:
        return np.empty(0)
    feasible = (cv <= 0.0).astype(float)
    return np.cumsum(feasible) / np.arange(1, cv.size + 1)


def benchmark_and_plot(cfg):
    budget = cfg.budget
    repeat = cfg.repeat
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Pick the suite by mode so constrained and unconstrained run as two separate
    # experiments (constrained problems return (F, G) and must not be fed to
    # unconstrained algorithms, and vice versa). constrained=true uses the
    # constrained_problems / constrained_algorithms sections of benchmark.yaml.
    constrained = bool(cfg.get("constrained", False))
    problem_cfg = cfg.constrained_problems if constrained else cfg.problems
    algorithm_cfg = cfg.constrained_algorithms if constrained else cfg.algorithms
    print(
        f"Benchmark mode: {'CONSTRAINED' if constrained else 'unconstrained'} "
        f"({len(problem_cfg)} problems, {len(algorithm_cfg)} algorithms)"
    )

    # Build problem specs
    problems = [
        MOOProblemSpec(
            name=p.name, dim=p.dim, n_obj=p.n_obj, ref_point=list(p.ref_point)
        )
        for p in problem_cfg
    ]

    # Initialize evaluator
    timeout = cfg.get("timeout", 1800)
    evaluator = MultiObjEvaluator(
        budget=budget,
        problems=problems,
        repeat=repeat,
        calculate_hv_history=True,
        timeout=timeout,
    )

    all_results = {}
    log_file = os.path.join(output_dir, "hv_benchmark_log.csv")
    obj_file = os.path.join(output_dir, "objectives_log.csv")
    pareto_file = os.path.join(output_dir, "pareto_front_log.csv")
    runtime_file = os.path.join(output_dir, "runtime_log.csv")
    # Constraint-aware logs (written only for constrained problems, where the
    # evaluator populated per-evaluation cv_history / feasibility_rate).
    feas_file = os.path.join(output_dir, "feasibility_log.csv")
    feas_summary_file = os.path.join(output_dir, "feasibility_summary.csv")

    # Export configuration (objective/Pareto CSVs so figures can be
    # regenerated from data, like the HV convergence curves already are).
    export_cfg = cfg.get("export", {}) or {}
    export_objectives = bool(export_cfg.get("objectives", True))
    export_pareto = bool(export_cfg.get("pareto", True))

    # Shared non-dominated sorter (used for Pareto export and Pareto plots)
    nds = NonDominatedSorting()

    # Run benchmarking
    for algo_cfg in algorithm_cfg:
        path = algo_cfg.path
        cls_name = algo_cfg.cls_name

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        with open(path, "r") as f:
            code_content = f.read()

        # Load incumbent from handler pkl if specified
        init_kwargs = None
        if "handler_pkl" in algo_cfg and algo_cfg.handler_pkl:
            init_kwargs = load_incumbent(algo_cfg.handler_pkl)

        print(f"--- Benchmarking Algorithm: {cls_name} ---")
        res = evaluator.evaluate(
            code=code_content, cls_name=cls_name, cls_init_kwargs=init_kwargs
        )
        all_results[cls_name] = res

        # Collect HV history (rows) + raw objectives (obj_rows) in one pass.
        # raw_y_by_problem accumulates every objective vector across repeats so
        # we can compute one combined non-dominated front per problem.
        rows = []
        obj_rows = []
        runtime_rows = []
        feas_rows = []
        feas_summary_rows = []
        raw_y_by_problem = {}
        for run in res.result:
            if "-rep" in run.name:
                prob_name, rep_str = run.name.split("-rep")
                rep_val = int(rep_str)
            else:
                prob_name, rep_val = run.name, 1

            # Per-run wall-clock time (for the time-accuracy comparison)
            exec_time = getattr(run, "execution_time", None)
            if exec_time is not None:
                runtime_rows.append(
                    {
                        "Algorithm": cls_name,
                        "Problem": prob_name,
                        "Repeat": rep_val,
                        "ExecutionTime": float(exec_time),
                    }
                )

            if hasattr(run, "hv_hist") and run.hv_hist is not None:
                for epoch, hv_value in enumerate(run.hv_hist):
                    rows.append(
                        {
                            "Algorithm": cls_name,
                            "Problem": prob_name,
                            "Repeat": rep_val,
                            "Epoch": epoch + 1,
                            "HV": hv_value,
                        }
                    )

            # Constraint feasibility (only present for constrained problems).
            cv_hist = getattr(run, "cv_history", None)
            if cv_hist is not None and len(cv_hist) > 0:
                cv_arr = np.asarray(cv_hist, dtype=float)
                frac_curve = feasible_fraction_curve(cv_arr)
                for eval_idx, (cv_val, frac) in enumerate(zip(cv_arr, frac_curve)):
                    feas_rows.append(
                        {
                            "Algorithm": cls_name,
                            "Problem": prob_name,
                            "Repeat": rep_val,
                            "Eval": eval_idx + 1,
                            "CV": float(cv_val),
                            "FeasibleFraction": float(frac),
                        }
                    )
                feas_rate = getattr(run, "feasibility_rate", None)
                if feas_rate is None:
                    feas_rate = float(np.mean(cv_arr <= 0.0))
                feas_summary_rows.append(
                    {
                        "Algorithm": cls_name,
                        "Problem": prob_name,
                        "Repeat": rep_val,
                        "FeasibilityRate": float(feas_rate),
                        "MeanCV": float(np.mean(cv_arr)),
                    }
                )

            raw_y = getattr(run, "raw_y_hist", None)
            if raw_y is not None and len(raw_y) > 0:
                raw_y = np.asarray(raw_y, dtype=float)
                if export_objectives:
                    for eval_idx, yv in enumerate(raw_y):
                        obj_row = {
                            "Algorithm": cls_name,
                            "Problem": prob_name,
                            "Repeat": rep_val,
                            "Eval": eval_idx + 1,
                        }
                        for j, val in enumerate(yv):
                            obj_row[f"f{j + 1}"] = float(val)
                        obj_rows.append(obj_row)
                if export_pareto:
                    raw_y_by_problem.setdefault(prob_name, []).append(raw_y)

        # Write HV history (existing schema/behaviour preserved)
        current_df = pd.DataFrame(rows)
        if os.path.exists(log_file):
            existing_df = pd.read_csv(log_file)
            full_df = pd.concat([existing_df, current_df], ignore_index=True)
            full_df = full_df.drop_duplicates(
                subset=["Algorithm", "Problem", "Repeat", "Epoch"]
            )
        else:
            full_df = current_df

        full_df = full_df.sort_values(by=["Algorithm", "Problem", "Repeat", "Epoch"])
        full_df.to_csv(log_file, index=False)
        print(f"Updated and sorted {log_file} with results for {cls_name}")

        # Write raw objective vectors (one row per evaluation)
        if export_objectives:
            _append_rows_to_csv(obj_file, obj_rows)
            if obj_rows:
                print(f"Updated {obj_file} with raw objectives for {cls_name}")

        # Write combined non-dominated front per problem (sorted by f1)
        if export_pareto:
            pareto_rows = []
            for prob_name, ys in raw_y_by_problem.items():
                Y_combined = np.vstack(ys)
                front_idx = nds.do(Y_combined, only_non_dominated_front=True)
                pf = Y_combined[front_idx]
                pf = pf[pf[:, 0].argsort()]
                for k, pt in enumerate(pf):
                    pareto_row = {
                        "Algorithm": cls_name,
                        "Problem": prob_name,
                        "PointIdx": k,
                    }
                    for j, val in enumerate(pt):
                        pareto_row[f"f{j + 1}"] = float(val)
                    pareto_rows.append(pareto_row)
            _append_rows_to_csv(pareto_file, pareto_rows)
            if pareto_rows:
                print(f"Updated {pareto_file} with Pareto front for {cls_name}")

        # Write per-run wall-clock times (time-accuracy comparison)
        _append_rows_to_csv(runtime_file, runtime_rows)
        if runtime_rows:
            print(f"Updated {runtime_file} with runtimes for {cls_name}")

        # Write constraint feasibility logs (only for constrained problems)
        _append_rows_to_csv(feas_file, feas_rows)
        _append_rows_to_csv(feas_summary_file, feas_summary_rows)
        if feas_rows:
            print(
                f"Updated {feas_file}/{feas_summary_file} with feasibility for {cls_name}"
            )

    # Plotting
    save_figs = cfg.plotting.save_figures
    show_figs = cfg.plotting.show_figures

    # Convergence plots
    for spec in problems:
        plt.figure(figsize=(10, 6))
        plot_added = False

        for cls_name, res in all_results.items():
            problem_runs = [
                run
                for run in res.result
                if run.name.startswith(spec.name)
                and hasattr(run, "hv_hist")
                and run.hv_hist is not None
            ]

            if not problem_runs:
                continue

            hists = [run.hv_hist for run in problem_runs]
            max_len = max(len(h) for h in hists)
            padded_hists = []
            for h in hists:
                if len(h) < max_len:
                    padded_hists.append(np.pad(h, (0, max_len - len(h)), mode="edge"))
                else:
                    padded_hists.append(h)

            data = np.array(padded_hists)
            mean_hv = np.mean(data, axis=0)
            std_hv = np.std(data, axis=0)
            epochs = np.arange(1, len(mean_hv) + 1)

            line = plt.plot(epochs, mean_hv, label=f"{cls_name}", linewidth=2)
            plt.fill_between(
                epochs,
                mean_hv - std_hv,
                mean_hv + std_hv,
                alpha=0.2,
                color=line[0].get_color(),
            )
            plot_added = True

        if plot_added:
            plt.title(f"Convergence Comparison: {spec.name.upper()}")
            plt.xlabel("Evaluations (Epochs)")
            plt.ylabel("Hypervolume (HV)")
            plt.legend(loc="lower right")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            if save_figs:
                fig_path = os.path.join(output_dir, f"plot_{spec.name}_convergence.png")
                plt.savefig(fig_path, dpi=cfg.plotting.convergence_dpi)
                print(f"Saved plot: {fig_path}")
            if show_figs:
                plt.show()
            else:
                plt.close()
        else:
            plt.close()

    # Pareto front plots (reuses the `nds` sorter created above)
    for spec in problems:
        if spec.n_obj != 2:
            print(f"Skipping Pareto plot for {spec.name} (n_obj != 2)")
            continue

        plt.figure(figsize=(8, 6))
        plot_exists = False

        for cls_name, res in all_results.items():
            all_y = [
                run.raw_y_hist
                for run in res.result
                if run.name.startswith(spec.name) and hasattr(run, "raw_y_hist")
            ]

            if not all_y:
                continue

            Y_combined = np.vstack(all_y)
            front_idx = nds.do(Y_combined, only_non_dominated_front=True)
            pf = Y_combined[front_idx]
            pf = pf[pf[:, 0].argsort()]

            plt.plot(pf[:, 0], pf[:, 1], "o--", label=cls_name, markersize=5)
            plot_exists = True

        if plot_exists:
            plt.title(f"Pareto Front Comparison: {spec.name.upper()}")
            plt.xlabel("Objective 1 ($f_1$)")
            plt.ylabel("Objective 2 ($f_2$)")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()

            if save_figs:
                fig_path = os.path.join(output_dir, f"pareto_{spec.name}.png")
                plt.savefig(fig_path, dpi=cfg.plotting.pareto_dpi)
                print(f"Saved Pareto figure to {fig_path}")
            if show_figs:
                plt.show()
            else:
                plt.close()
        else:
            plt.close()


if __name__ == "__main__":
    # Load config: conf/benchmark.yaml + optional CLI overrides
    base_cfg = OmegaConf.load("conf/benchmark.yaml")

    # Support CLI overrides: python benchmark_best_codes.py budget=200 repeat=10
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    print("Benchmark config:")
    print(OmegaConf.to_yaml(cfg))

    benchmark_and_plot(cfg)
