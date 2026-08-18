"""Section VI-A: the recorded ancestry of the three systematically-selected
algorithms -- the parent chain from each design back to its lineage root.

By default this reads paper_data/lineage.csv, exported from the population
checkpoints of the generation runs. To re-derive it from scratch you need the full
run trees (each checkpoint is ~76 MB and they are not part of this repository);
point `runs=` at a directory containing them:

    python paper_reproduction/trace_lineage.py
    python paper_reproduction/trace_lineage.py runs=/path/to/paper_results

Note on MOEAD-EI Hybrid: it came from a separate development run (April 2026), not
from the nine runs reported in the paper, and its checkpoint can no longer be
loaded by the current code (it references a prompt-generator module that has since
been removed). Its lineage is therefore not reproducible here; its benchmark
results are, via MO bench/MOBO_MOEAD_EI_Hybrid_fixed.py.
"""
import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

CSV = "paper_data/lineage.csv"
# algorithm -> (run directory name, search fitness of the target individual)
TARGETS = {
    "MOBOImprovedScalarizedEI": ("exp_mo_4plus16_run2", 0.6981),
    "MOBORandomForestLCBPBI_CD": ("exp_mo_4plus16_run3", 0.5295),
    "MOBORandomForestParEGO_Batch_Refined": ("exp_mo_4plus16_run1", 0.5158),
}


def from_csv(path):
    import pandas as pd

    df = pd.read_csv(path)
    for algo, g in df.groupby("algorithm", sort=False):
        g = g.sort_values("step")
        root, final = g.fitness.iloc[0], g.fitness.iloc[-1]
        print(f"\n===== {algo}  ({g.run.iloc[0]})")
        print(f"      {len(g)} lineage steps, generations "
              f"{g.generation.iloc[0]}->{g.generation.iloc[-1]}, "
              f"fitness {root:.4f} -> {final:.4f}  (+{100 * (final / root - 1):.1f}%)")
        for _, r in g.iterrows():
            op = "root" if r.n_parents == 0 else ("mutation" if r.n_parents == 1 else "crossover")
            print(f"  step {r.step}  [gen {r.generation:>2}]  {r.ancestor:<44s} "
                  f"{r.fitness:.4f}  ({op})")


def from_checkpoints(root):
    for algo, (run, fit) in TARGETS.items():
        subs = glob.glob(os.path.join(root, run, "ESPopulation_*"))
        if not subs:
            print(f"\n===== {algo}: no run directory under {os.path.join(root, run)}")
            continue
        cps = sorted(glob.glob(os.path.join(subs[0], "*heckpoint*.pkl")),
                     key=lambda p: int(p.split("checkpoint_")[1].split("_")[0]))
        pop = pickle.load(open(cps[-1], "rb"))
        inds = list(pop.all_individuals())
        by_id = {i.id: i for i in inds}
        cands = [i for i in inds if i.name == algo and i.fitness is not None]
        cands.sort(key=lambda i: abs(i.fitness - fit))
        chain, cur, seen = [], cands[0], set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            chain.append(cur)
            pids = cur.parent_id or []
            cur = by_id.get(pids[0]) if pids else None
        chain.reverse()
        print(f"\n===== {algo}  ({run})  {len(chain)} lineage steps")
        for step, c in enumerate(chain):
            n = len(c.parent_id or [])
            op = "root" if n == 0 else ("mutation" if n == 1 else "crossover")
            print(f"  step {step}  [gen {c.generation:>2}]  {c.name:<44s} "
                  f"{c.fitness:.4f}  ({op})")


if __name__ == "__main__":
    runs = None
    for a in sys.argv[1:]:
        if a.startswith("runs="):
            runs = a.split("=", 1)[1]
    if runs:
        sys.path.insert(0, ".")  # the llamevol package, needed to unpickle populations
        from_checkpoints(runs)
    elif os.path.exists(CSV):
        from_csv(CSV)
    else:
        sys.exit(f"{CSV} not found; pass runs=<dir> to trace from population checkpoints")
