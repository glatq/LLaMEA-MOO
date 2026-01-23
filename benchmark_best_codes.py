import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.config import Config

Config.warnings["not_compiled"] = False
os.makedirs("benchmark_results", exist_ok=True)


def benchmark_and_plot():
    budget = 10
    repeat = 3

    problems = [
        MOOProblemSpec(name="bnh", dim=2, n_obj=2, ref_point=[140.0, 50.0]),
        MOOProblemSpec(name="osy", dim=6, n_obj=2, ref_point=[0.0, 386.0]),
        MOOProblemSpec(name="ctp1", dim=2, n_obj=2, ref_point=[1.0, 2.0]),
        MOOProblemSpec(name="tnk", dim=2, n_obj=2, ref_point=[2.0, 2.0]),
        MOOProblemSpec(name="mw1", dim=8, n_obj=2, ref_point=[1.0, 7.0]),
        MOOProblemSpec(name="mw2", dim=6, n_obj=2, ref_point=[1.0, 7.0]),
        MOOProblemSpec(name="mw3", dim=6, n_obj=2, ref_point=[1.0, 7.0]),
        MOOProblemSpec(name="mw11", dim=6, n_obj=2, ref_point=[30.0, 30.0]),
    ]

    algorithms = [
        (
            "exp_mo_es_search/ESPopulation_evol_1+1_0114165323/27-28_MOBO_HybridCandidateEI_EnhancedGPR_-3420.4316.py",
            "MOBO_HybridCandidateEI_EnhancedGPR",
        ),
        (
            "exp_mo_es_search/ESPopulation_evol_4+16_0115153530/0-4_MOBO_KNN_EHVI_-3015.6484.py",
            "MOBO_KNN_EHVI",
        ),
        ("MORS_baseline.py", "RandomSearchMO"),
    ]

    # 3. Initialize Evaluator
    evaluator = MultiObjEvaluator(
        budget=budget, problems=problems, repeat=repeat, calculate_hv_history=True
    )

    all_results = {}

    # 4. Run Benchmarking
    for path, cls_name in algorithms:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        with open(path, "r") as f:
            code_content = f.read()

        print(f"--- Benchmarking Algorithm: {cls_name} ---")
        res = evaluator.evaluate(code=code_content, cls_name=cls_name)
        all_results[cls_name] = res

        log_file = "benchmark_results/hv_benchmark_log.csv"
        rows = []

        for run in res.result:
            # Extract names. run.name is "bnh-rep1", "osy-rep2", etc.
            if "-rep" in run.name:
                prob_name, rep_str = run.name.split("-rep")
                rep_val = int(rep_str)
            else:
                prob_name, rep_val = run.name, 1  # fallback

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

        # 2. Combine with existing log if it exists to ensure GLOBAL sorting
        if os.path.exists(log_file):
            existing_df = pd.read_csv(log_file)
            # Combine current results with previous ones
            full_df = pd.concat([existing_df, current_df], ignore_index=True)
            # Optional: Remove duplicates in case of rerun
            full_df = full_df.drop_duplicates(
                subset=["Algorithm", "Problem", "Repeat", "Epoch"]
            )
        else:
            full_df = current_df

        # 3. Perform Global Sorting: Problem > Repeat > Epoch
        # Adding 'Algorithm' to the end keeps comparisons together
        full_df = full_df.sort_values(by=["Algorithm", "Problem", "Repeat", "Epoch"])

        # 4. Save the fully sorted file (Overwrite, don't append)
        full_df.to_csv(log_file, index=False)

        print(f"Updated and sorted {log_file} with results for {cls_name}")

    # 5. Plotting: One figure per problem
    for spec in problems:
        plt.figure(figsize=(10, 6))
        plot_added = False

        for cls_name, res in all_results.items():
            # Collect all hv_hist arrays for this specific problem
            # Note: run.name is formatted as "{problem_name}-rep{N}"
            problem_runs = [
                run
                for run in res.result
                if run.name.startswith(spec.name)
                and hasattr(run, "hv_hist")
                and run.hv_hist is not None
            ]

            if not problem_runs:
                continue

            # Extract histories
            hists = [run.hv_hist for run in problem_runs]

            # Handle variable lengths (padding with the last value if a run stopped early)
            max_len = max(len(h) for h in hists)
            padded_hists = []
            for h in hists:
                if len(h) < max_len:
                    padded_hists.append(np.pad(h, (0, max_len - len(h)), mode="edge"))
                else:
                    padded_hists.append(h)

            # Calculate Statistics
            data = np.array(padded_hists)
            mean_hv = np.mean(data, axis=0)
            std_hv = np.std(data, axis=0)
            epochs = np.arange(1, len(mean_hv) + 1)

            # Plotting Mean and Std Shadow
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

            # Save the figure
            fig_path = f"benchmark_results/plot_{spec.name}_convergence.png"
            plt.savefig(fig_path)
            print(f"Saved plot: {fig_path}")
            plt.show()
        else:
            plt.close()

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

            # Note: We don't need 'line =' here since we aren't reuse the object
            plt.plot(pf[:, 0], pf[:, 1], "o--", label=cls_name, markersize=5)
            plot_exists = True

        if plot_exists:
            plt.title(f"Pareto Front Comparison: {spec.name.upper()}")
            plt.xlabel("Objective 1 ($f_1$)")
            plt.ylabel("Objective 2 ($f_2$)")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.7)
            plt.tight_layout()

            # --- SAVE BEFORE SHOW ---
            fig_path = f"benchmark_results/pareto_{spec.name}.png"
            plt.savefig(fig_path, dpi=300)  # dpi=300 for high quality
            print(f"Saved Pareto figure to {fig_path}")

            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    benchmark_and_plot()
