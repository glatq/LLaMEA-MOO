"""2x2 convergence grid with a single shared legend below (Figs 3 & 4)."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import os as _os
_os.makedirs("paper_figures", exist_ok=True)  # gitignored; absent in a fresh clone
plt.rcParams.update({"font.size":19,"axes.titlesize":20,"axes.labelsize":18,
    "xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":17})
LBL={"MOBOImprovedScalarizedEI":"Improved-Scalarized-EI","MOBORandomForestLCBPBI_CD":"RF-LCB-PBI",
     "MOBORandomForestParEGO_Batch_Refined":"RF-ParEGO-Batch","BoFireQparEGOWrapper":"qParEGO",
     "IOCSAMOCOBRAWrapper":"IOC-SAMO-COBRA","NSGA2Wrapper":"NSGA-II","NSGA3Wrapper":"NSGA-III",
     "RandomSearchMO":"Random Search","MOBO_MOEAD_EI_Hybrid_Fixed":"MOEAD-EI Hybrid"}
def lab(a): return LBL.get(a, a.replace("MOBO","").lstrip("_"))

def grid(csv, problems, out, order, ncol, bottom, nrows=2, ncols=2, figsize=(13,9), hspace=0.32, left=0.08, legend_y=0.0):
    df=pd.read_csv(csv)
    algos=[a for a in order if a in df.Algorithm.unique()]
    colors={a:plt.cm.tab10(i%10) for i,a in enumerate(algos)}
    fig,axes=plt.subplots(nrows,ncols,figsize=figsize)
    axlist=axes.ravel() if hasattr(axes,"ravel") else [axes]
    for ax,prob,letter in zip(axlist,problems,"abcdefghijkl"):
        d=df[df.Problem==prob]
        for a in algos:
            g=d[d.Algorithm==a].groupby("Epoch").HV
            m,s=g.mean(),g.std()
            ax.plot(m.index,m.values,color=colors[a],lw=2,label=lab(a))
            ax.fill_between(m.index,(m-s).values,(m+s).values,color=colors[a],alpha=0.15)
        ax.set_title(f"({letter}) {prob.upper()}"); ax.set_xlabel("Evaluations")
        ax.set_ylabel("Hypervolume (HV)"); ax.grid(True,ls="--",alpha=0.4)
    h,l=axlist[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=ncol,bbox_to_anchor=(0.5,legend_y),frameon=True)
    fig.subplots_adjust(bottom=bottom,hspace=hspace,wspace=0.27,top=0.96,left=left,right=0.98)
    fig.savefig(out,dpi=200,bbox_inches="tight"); plt.close(); print("saved",out)

GEN_ORDER=["MOBOImprovedScalarizedEI","MOBORandomForestLCBPBI_CD","MOBORandomForestParEGO_Batch_Refined",
           "MOBOESILimitedGP","MOBO_GP_Tchebycheff_Refined","MOBORobustLCBFPS","MOBORefinedTchebyLCBGP",
           "MOBOEnsembleRidge_MPFDUWS","MOBO_GP_WeightedUCB"]
P2_ORDER=["MOBOImprovedScalarizedEI","MOBORandomForestLCBPBI_CD","MOBORandomForestParEGO_Batch_Refined",
          "MOBO_MOEAD_EI_Hybrid_Fixed","BoFireQparEGOWrapper","IOCSAMOCOBRAWrapper","NSGA2Wrapper",
          "NSGA3Wrapper","RandomSearchMO"]
# Fig 3 = Phase 1 (generated)
grid("paper_data/benchmark_stage1/hv_benchmark_log.csv",["zdt1","dtlz2","dtlz1","wfg7"],
     "paper_figures/fig3_phase1_grid.png",GEN_ORDER,ncol=3,bottom=0.22,legend_y=-0.03)
# Fig 4 = Phase 2 (8 algos)
grid("paper_data/benchmark_stage2/hv_benchmark_log.csv",["zdt6","dtlz4","zdt4","dtlz7"],
     "paper_figures/fig4_phase2_grid.png",P2_ORDER,ncol=5,bottom=0.20,legend_y=-0.03)
# Fig 5 = Phase 3 real (3x1, single column, legend below)
grid("paper_data/benchmark_stage3/hv_benchmark_log.csv",["re21","re34","re37"],
     "paper_figures/fig5_phase3_grid.png",P2_ORDER,ncol=2,bottom=0.22,
     nrows=3,ncols=1,figsize=(6.5,12.5),hspace=0.42,left=0.15,legend_y=-0.045)
# Appendix all-12 grids (4 rows x 3 cols), single legend below
SYN12=["zdt1","zdt2","zdt3","zdt4","zdt6","dtlz1","dtlz2","dtlz4","dtlz7","wfg4","wfg7","wfg9"]
# Fig 8 = Phase 1 appendix (9 generated)
grid("paper_data/benchmark_stage1/hv_benchmark_log.csv",SYN12,
     "paper_figures/fig8_phase1_all.png",GEN_ORDER,ncol=3,bottom=0.11,
     nrows=4,ncols=3,figsize=(16,18),hspace=0.40,left=0.07,legend_y=-0.025)
# Fig 9 = Phase 2 appendix (8 algos)
grid("paper_data/benchmark_stage2/hv_benchmark_log.csv",SYN12,
     "paper_figures/fig9_phase2_all.png",P2_ORDER,ncol=5,bottom=0.10,
     nrows=4,ncols=3,figsize=(16,18),hspace=0.40,left=0.07,legend_y=-0.02)
