"""Re-run Improved-Scalarized-EI with 5 explicit seeds (42 + 4 others) so it gets
genuine per-repeat variance. Reuses MultiObjEvaluator.evaluate (repeat=1 per seed)
so HV is computed identically to the main benchmark. Emits standard CSVs."""
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

def run(specs, outdir):
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
    pd.DataFrame(hv_rows).to_csv(f"{outdir}/hv_benchmark_log.csv",index=False)
    pd.DataFrame(obj_rows).to_csv(f"{outdir}/objectives_log.csv",index=False)
    pd.DataFrame(rt_rows).to_csv(f"{outdir}/runtime_log.csv",index=False)
    pd.DataFrame(par_rows).to_csv(f"{outdir}/pareto_front_log.csv",index=False)
    # quick determinism check vs seed 42
    f=pd.DataFrame(hv_rows); f=f[f.Epoch==400]
    print(f"  [{outdir}] final-HV spread per problem (seed order {SEEDS}):")
    for prob in f.Problem.unique():
        vals=f[f.Problem==prob].sort_values("Repeat").HV.round(3).tolist()
        print(f"    {prob:7s} {vals}")

if __name__ == "__main__":
    print("=== SYNTHETIC ==="); run(SYN,"benchmark_results/isei_rerun_syn")
    print("=== REAL ===");      run(RE,"benchmark_results/isei_rerun_real")
    print("DONE")
