"""Fig 6: time-accuracy trade-off, 1x2 (synthetic | real), distinct marker per
algorithm, single shared legend below, square-ish panels, large fonts."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os as _os
_os.makedirs("paper_figures", exist_ok=True)  # gitignored; absent in a fresh clone
plt.rcParams.update({"font.size":19,"axes.titlesize":21,"axes.labelsize":19,
    "xtick.labelsize":16,"ytick.labelsize":16,"legend.fontsize":17})
ORDER=["MOBOImprovedScalarizedEI","MOBORandomForestLCBPBI_CD","MOBORandomForestParEGO_Batch_Refined",
       "MOBO_MOEAD_EI_Hybrid_Fixed","BoFireQparEGOWrapper","IOCSAMOCOBRAWrapper","NSGA2Wrapper",
       "NSGA3Wrapper","RandomSearchMO"]
LBL={"MOBOImprovedScalarizedEI":"Improved-Scalarized-EI","MOBORandomForestLCBPBI_CD":"RF-LCB-PBI",
     "MOBORandomForestParEGO_Batch_Refined":"RF-ParEGO-Batch","BoFireQparEGOWrapper":"qParEGO",
     "IOCSAMOCOBRAWrapper":"IOC-SAMO-COBRA","NSGA2Wrapper":"NSGA-II","NSGA3Wrapper":"NSGA-III",
     "RandomSearchMO":"Random Search","MOBO_MOEAD_EI_Hybrid_Fixed":"MOEAD-EI Hybrid"}
MARK={"MOBOImprovedScalarizedEI":"*","MOBORandomForestLCBPBI_CD":"P",
      "MOBORandomForestParEGO_Batch_Refined":"X","BoFireQparEGOWrapper":"s",
      "IOCSAMOCOBRAWrapper":"D","NSGA2Wrapper":"^","NSGA3Wrapper":"v","RandomSearchMO":"o",
      "MOBO_MOEAD_EI_Hybrid_Fixed":"p"}
COL={a:plt.cm.tab10(i) for i,a in enumerate(ORDER)}
SZ={"*":420,"P":230,"X":230,"s":190,"D":180,"^":210,"v":210,"o":190,"p":300}
def normhv(csv):
    df=pd.read_csv(csv); f=df.sort_values("Epoch").groupby(["Algorithm","Problem","Repeat"]).HV.last().reset_index()
    m=f.groupby(["Algorithm","Problem"]).HV.mean().unstack(0); return m.div(m.max(axis=1),axis=0).mean()
def runtime(csv): return pd.read_csv(csv).groupby("Algorithm").ExecutionTime.mean()
fig,axes=plt.subplots(1,2,figsize=(12.5,6.2))
for ax,(stage,letter,tag) in zip(axes,[("benchmark_stage2","a","Synthetic"),("benchmark_stage3","b","Real-world")]):
    hv=normhv(f"paper_data/{stage}/hv_benchmark_log.csv"); rt=runtime(f"paper_data/{stage}/runtime_log.csv")
    for a in ORDER:
        if a not in hv.index: continue
        ax.scatter(rt[a],hv[a],s=SZ[MARK[a]],marker=MARK[a],c=[COL[a]],edgecolor="k",linewidth=0.8,zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("Mean wall-clock per run (s)"); ax.set_ylabel("Mean normalized HV")
    ax.set_title(f"({letter}) {tag}"); ax.grid(True,ls="--",alpha=0.5)
handles=[mlines.Line2D([],[],color=COL[a],marker=MARK[a],linestyle="None",markersize=15,
         markeredgecolor="k",label=LBL[a]) for a in ORDER]
fig.legend(handles=handles,loc="lower center",ncol=5,bbox_to_anchor=(0.5,-0.02),frameon=True)
fig.subplots_adjust(bottom=0.32,wspace=0.28,top=0.92,left=0.08,right=0.98)
fig.savefig("paper_figures/fig6_time_accuracy.png",dpi=200,bbox_inches="tight"); plt.close()
print("saved paper_figures/fig6_time_accuracy.png")
