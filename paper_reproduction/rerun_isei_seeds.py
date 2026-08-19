"""Re-run Improved-Scalarized-EI with 5 explicit seeds so it gets genuine per-repeat
variance (Section VI-B of the paper).

Why this exists: Improved-Scalarized-EI is the only one of the nine generated algorithms
with a `random_seed` parameter (default 42) feeding its own np.random.default_rng, so it
is deterministic -- every repeat of the normal benchmark returns an identical
hypervolume and reports zero variance. Passing five explicit seeds restores a fair
variance estimate. Every Improved-Scalarized-EI number in the paper, in Phase 1, 2 and
3 alike, comes from this script, not from conf/benchmark_phase*.yaml (which deliberately
exclude it).

Reuses MultiObjEvaluator.evaluate (repeat=1 per seed) so HV is computed identically to
the main benchmark. Writes its own output dirs AND appends its rows into the matching
phase dirs, so no manual merging is needed:

    synthetic (12 problems) -> benchmark_results/phase1 and benchmark_results/phase2
    real-world (3 problems) -> benchmark_results/phase3

Run from the repo root, after the phase benchmarks:
    python paper_reproduction/rerun_isei_seeds.py
"""
import os, numpy as np, pandas as pd
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec
import yaml
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ALGO="generated_algorithms/MOBOImprovedScalarizedEI.py"
INCUMBENTS="generated_algorithms/incumbents.yaml"
CLS="MOBOImprovedScalarizedEI"
SEEDS=[42,1,2,3,4]
SYN=[("zdt1",30,2,[1.1,7.2]),("zdt2",30,2,[1.1,8.0]),("zdt3",30,2,[1.1,7.2]),("zdt4",10,2,[1.1,300.0]),
     ("zdt6",10,2,[1.1,11.0]),("dtlz1",5,3,[10.0,10.0,10.0]),("dtlz2",20,3,[3.0,3.0,3.0]),
     ("dtlz4",5,3,[1.1,1.1,1.1]),("dtlz7",20,3,[1.1,1.1,28.0]),("wfg4",6,2,[2.2,4.4]),
     ("wfg7",10,3,[2.2,4.4,6.6]),("wfg9",6,2,[2.2,4.4])]
RE=[("re21",4,2,[2851.9,0.037]),("re34",5,3,[1862.3,12.17,0.196]),("re37",4,3,[0.887,0.913,1.012])]
code=open(ALGO).read()
incumbent=yaml.safe_load(open(INCUMBENTS))["MOBOImprovedScalarizedEI"]["init_kwargs"] or {}
nds=NonDominatedSorting()

KEYS={"hv_benchmark_log":["Algorithm","Problem","Repeat","Epoch"],
      "objectives_log":["Algorithm","Problem","Repeat","Eval"],
      "runtime_log":["Algorithm","Problem","Repeat"],
      "pareto_front_log":["Algorithm","Problem","PointIdx"]}

def _append_into(path, df, name):
    """Append df into an existing benchmark CSV, de-duplicating on that file's keys.

    Mirrors benchmark_best_codes.py's append-and-dedup behaviour, so a phase dir can be
    built from several invocations. Keeps the LAST occurrence, so re-running this script
    replaces its own earlier rows rather than doubling them.
    """
    if df.empty:
        return
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        keys=[k for k in KEYS[name] if k in df.columns]
        if keys:
            df = df.drop_duplicates(subset=keys, keep="last")
        if set(KEYS[name][:3]).issubset(df.columns):
            df = df.sort_values(by=[c for c in KEYS[name] if c in df.columns])
    df.to_csv(path, index=False)


def run(specs, outdir, targets=None):
    os.makedirs(outdir, exist_ok=True)
    problems=[MOOProblemSpec(name=n,dim=d,n_obj=o,ref_point=r) for (n,d,o,r) in specs]
    hv_rows,obj_rows,rt_rows=[],[],[]; raw_by_prob={}
    for ri,seed in enumerate(SEEDS, start=1):
        ev=MultiObjEvaluator(budget=400,problems=problems,repeat=1,calculate_hv_history=True,
                             timeout=21600,use_multiprocessing=True)
        res=ev.evaluate(code=code,cls_name=CLS,cls_init_kwargs={**incumbent,"random_seed":seed})
        for r in res.result:
            prob=r.name.split("-rep")[0]
            et=getattr(r,"execution_time",None)
            if et is not None: rt_rows.append({"Algorithm":CLS,"Problem":prob,"Repeat":ri,"ExecutionTime":float(et)})
            hv=getattr(r,"hv_hist",None)
            if hv is not None:
                for e,v in enumerate(hv): hv_rows.append({"Algorithm":CLS,"Problem":prob,"Repeat":ri,"Epoch":e+1,"HV":v})
            ry=getattr(r,"raw_y_hist",None)
            if ry is not None and len(ry)>0:
                ry=np.asarray(ry,dtype=float)
                for ei,yv in enumerate(ry):
                    row={"Algorithm":CLS,"Problem":prob,"Repeat":ri,"Eval":ei+1}
                    for j,val in enumerate(yv): row[f"f{j+1}"]=float(val)
                    obj_rows.append(row)
                raw_by_prob.setdefault(prob,[]).append(ry)
        print(f"  seed {seed} done ({len(res.result)} problems)", flush=True)
    # combined ND pareto front per problem (across the 5 seeds)
    par_rows=[]
    for prob,arrs in raw_by_prob.items():
        Y=np.vstack(arrs); front=Y[nds.do(Y, only_non_dominated_front=True)]
        for pi,pt in enumerate(front):
            row={"Algorithm":CLS,"Problem":prob,"PointIdx":pi}
            for j,val in enumerate(pt): row[f"f{j+1}"]=float(val)
            par_rows.append(row)
    frames={"hv_benchmark_log":pd.DataFrame(hv_rows),"objectives_log":pd.DataFrame(obj_rows),
            "runtime_log":pd.DataFrame(rt_rows),"pareto_front_log":pd.DataFrame(par_rows)}
    for name,df in frames.items():
        df.to_csv(f"{outdir}/{name}.csv",index=False)
    for tgt in (targets or []):
        if not os.path.isdir(tgt):
            print(f"  [skip] {tgt} does not exist (run the phase benchmark first)")
            continue
        for name,df in frames.items():
            _append_into(os.path.join(tgt,f"{name}.csv"), df, name)
        print(f"  [merged] Improved-Scalarized-EI rows appended into {tgt}")
    # quick determinism check vs seed 42
    f=pd.DataFrame(hv_rows); f=f[f.Epoch==400]
    print(f"  [{outdir}] final-HV spread per problem (seed order {SEEDS}):")
    for prob in f.Problem.unique():
        vals=f[f.Problem==prob].sort_values("Repeat").HV.round(3).tolist()
        print(f"    {prob:7s} {vals}")

if __name__ == "__main__":
    print("=== SYNTHETIC ===")
    run(SYN, "benchmark_results/isei_rerun_syn",
        targets=["benchmark_results/phase1", "benchmark_results/phase2"])
    print("=== REAL ===")
    run(RE, "benchmark_results/isei_rerun_real",
        targets=["benchmark_results/phase3"])
    print("DONE")
