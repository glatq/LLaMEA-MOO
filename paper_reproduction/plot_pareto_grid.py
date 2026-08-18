"""Fig 10 (synthetic, 4x3) and Fig 11 (real, 1x3): non-dominated fronts as a
grid with ONE shared legend below. 2D problems -> sorted markers + dashes;
3D problems -> 3D scatter. Curated set: new top-3 + MOEAD-EI + qParEGO."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os as _os
_os.makedirs("paper_figures", exist_ok=True)  # gitignored; absent in a fresh clone
plt.rcParams.update({"font.size":16,"axes.titlesize":18,"axes.labelsize":16,
    "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":18})
ALGOS=["MOBOImprovedScalarizedEI","MOBORandomForestLCBPBI_CD","MOBORandomForestParEGO_Batch_Refined",
       "MOBO_MOEAD_EI_Hybrid_Fixed","BoFireQparEGOWrapper"]
LBL={"MOBOImprovedScalarizedEI":"Improved-Scalarized-EI","MOBORandomForestLCBPBI_CD":"RF-LCB-PBI",
     "MOBORandomForestParEGO_Batch_Refined":"RF-ParEGO-Batch","MOBO_MOEAD_EI_Hybrid_Fixed":"MOEAD-EI Hybrid",
     "BoFireQparEGOWrapper":"qParEGO"}
COL={"MOBOImprovedScalarizedEI":"tab:blue","MOBORandomForestLCBPBI_CD":"tab:orange",
     "MOBORandomForestParEGO_Batch_Refined":"tab:green","MOBO_MOEAD_EI_Hybrid_Fixed":"tab:red",
     "BoFireQparEGOWrapper":"tab:purple"}
MARK={"MOBOImprovedScalarizedEI":"*","MOBORandomForestLCBPBI_CD":"P",
      "MOBORandomForestParEGO_Batch_Refined":"X","MOBO_MOEAD_EI_Hybrid_Fixed":"p","BoFireQparEGOWrapper":"s"}
NOBJ={"zdt1":2,"zdt2":2,"zdt3":2,"zdt4":2,"zdt6":2,"dtlz1":3,"dtlz2":3,"dtlz4":3,"dtlz7":3,
      "wfg4":2,"wfg7":3,"wfg9":2,"re21":2,"re34":3,"re37":3}

def pareto_grid(csv, problems, nrows, ncols, out, figsize, legend_y, bottom, ncol=5):
    df=pd.read_csv(csv); fig=plt.figure(figsize=figsize)
    for i,(prob,letter) in enumerate(zip(problems,"abcdefghijkl")):
        nobj=NOBJ[prob]; sub=df[df.Problem==prob]
        ax=fig.add_subplot(nrows,ncols,i+1,projection="3d" if nobj==3 else None)
        for a in ALGOS:
            d=sub[sub.Algorithm==a]
            if d.empty: continue
            if nobj==2:
                Y=d[["f1","f2"]].dropna().to_numpy(float)
                if not len(Y): continue
                Y=Y[Y[:,0].argsort()]
                ax.plot(Y[:,0],Y[:,1],linestyle="--",marker=MARK[a],color=COL[a],markersize=5,linewidth=1.2)
            else:
                Y=d[["f1","f2","f3"]].dropna().to_numpy(float)
                if not len(Y): continue
                ax.scatter(Y[:,0],Y[:,1],Y[:,2],marker=MARK[a],color=COL[a],s=14,alpha=0.6,edgecolors="none")
        ttl=f"({letter}) {prob.upper()}"+(" (3 obj.)" if nobj==3 else "")
        ax.set_title(ttl)
        ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")
        if nobj==2: ax.grid(True,ls=":",alpha=0.7)
        else: ax.set_zlabel("$f_3$"); ax.view_init(elev=22,azim=45); ax.tick_params(labelsize=10)
    handles=[mlines.Line2D([],[],color=COL[a],marker=MARK[a],linestyle="None",markersize=15,label=LBL[a]) for a in ALGOS]
    fig.legend(handles=handles,loc="lower center",ncol=ncol,bbox_to_anchor=(0.5,legend_y),frameon=True)
    fig.subplots_adjust(bottom=bottom,hspace=0.38,wspace=0.30,top=0.96,left=0.07,right=0.97)
    fig.savefig(out,dpi=200,bbox_inches="tight"); plt.close(); print("saved",out)

pareto_grid("paper_data/benchmark_stage2/pareto_front_log.csv",
            ["zdt1","zdt2","zdt3","zdt4","zdt6","dtlz1","dtlz2","dtlz4","dtlz7","wfg4","wfg7","wfg9"],
            4,3,"paper_figures/fig10_pareto_synth.png",(16,19),legend_y=-0.015,bottom=0.10)
pareto_grid("paper_data/benchmark_stage3/pareto_front_log.csv",
            ["re21","re34","re37"],1,3,"paper_figures/fig11_pareto_real.png",(16,5.8),legend_y=-0.10,bottom=0.26)
