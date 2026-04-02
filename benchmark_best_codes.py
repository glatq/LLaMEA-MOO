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


def benchmark_and_plot(cfg):
    budget = cfg.budget
    repeat = cfg.repeat
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Build problem specs
    problems = [
        MOOProblemSpec(
            name=p.name, dim=p.dim, n_obj=p.n_obj, ref_point=list(p.ref_point)
        )
        for p in cfg.problems
    ]

    # Initialize evaluator
    evaluator = MultiObjEvaluator(
        budget=budget, problems=problems, repeat=repeat, calculate_hv_history=True
    )

    all_results = {}
    log_file = os.path.join(output_dir, "hv_benchmark_log.csv")

    # Run benchmarking
    for algo_cfg in cfg.algorithms:
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

        # Log HV history to CSV
        rows = []
        for run in res.result:
            if "-rep" in run.name:
                prob_name, rep_str = run.name.split("-rep")
                rep_val = int(rep_str)
            else:
                prob_name, rep_val = run.name, 1

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

    # Pareto front plots
    nds = NonDominatedSorting()

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
