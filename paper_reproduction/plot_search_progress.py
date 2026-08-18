"""Fig. 2: LLaMEA search progress -- best-so-far search fitness over the first 100
algorithms evaluated, for all nine evolutionary runs.

Reads paper_data/search_history.csv, which was exported from the saved population
checkpoints of the nine runs (columns: run, strategy, generation, index, algorithm,
fitness). The original version of this script parsed the per-individual .py
filenames inside the run directories; those trees are ~24 GB and are not part of
this repository. The checkpoint export is also strictly more complete: in
exp_mo_1plus1_run3 only 59 of the 100 evaluated individuals have a .py on disk,
because duplicate-filtered candidates are never written out. That does not change
the curve -- a duplicate inherits the cached fitness of an earlier individual and so
can never raise the best-so-far, and both sources agree on the final value (0.4775)
-- but the CSV gives the exact step positions.

Run from the repo root:  python paper_reproduction/plot_search_progress.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13,
})

CSV = "paper_data/search_history.csv"
OUT = "paper_figures/fig2_llamea_search_progress_9runs.png"
XCUT = 100
COLOR = {"(1+1)-ES": "tab:blue", "(4+16)-ES": "tab:red", "(8,16)-ES": "tab:green"}
STYLES = ["-", "--", "-."]

os.makedirs("paper_figures", exist_ok=True)
df = pd.read_csv(CSV)

plt.figure(figsize=(6.8, 5.0))
seen, cnt = set(), {}
for strategy in ["(1+1)-ES", "(4+16)-ES", "(8,16)-ES"]:
    sub = df[df.strategy == strategy]
    for run in sorted(sub.run.unique()):
        r = sub[sub.run == run].sort_values("index")
        xs, ys, cur = [], [], -np.inf
        for idx, fit in zip(r["index"], r.fitness):
            if pd.notna(fit):
                cur = max(cur, float(fit))
            if cur > -np.inf:
                xs.append(int(idx)); ys.append(cur)
        xs, ys = np.array(xs), np.array(ys)
        keep = xs <= XCUT
        n = cnt.get(strategy, 0); cnt[strategy] = n + 1
        plt.step(xs[keep], ys[keep], where="post", color=COLOR[strategy],
                 linestyle=STYLES[n % 3], linewidth=2.5, alpha=0.85,
                 label=(strategy if strategy not in seen else None))
        seen.add(strategy)

plt.title("LLaMEA Search Progress", fontweight="bold")
plt.xlabel("Number of algorithms evaluated")
plt.ylabel("Best-so-far HV score")
plt.xlim(0, XCUT)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close()
print("saved", OUT)
