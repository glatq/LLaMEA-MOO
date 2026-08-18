"""Statistical significance tests for the benchmark HV results.

Two modes:

mode=demsar (default) -- Demsar (2006) suite-level analysis over the problems:
  * Friedman omnibus test across all algorithms (+ mean ranks).
  * Pairwise Wilcoxon signed-rank tests of a reference algorithm vs each other
    algorithm, paired by problem, with Holm correction.
  Per-problem performance = mean final-epoch HV over repeats, then
  max-normalized per problem (Friedman is rank-based so this only affects
  Wilcoxon).

mode=blade -- BLADE-style per-problem head-to-head (no correction), matching
  the protocol in the BLADE benchmark paper (arXiv:2504.20183). For each
  problem a two-sample Welch t-test over repeats compares each generated
  algorithm against the SOTA reference (qParEGO); we report on how many
  problems the generated algorithm is significantly better / worse. The same
  per-problem test is applied to computational time (runtime_log.csv).

Usage:
    python significance_tests.py csv=paper_data/benchmark_stage2/hv_benchmark_log.csv
    python significance_tests.py mode=blade csv=paper_data/benchmark_stage2/hv_benchmark_log.csv
    python significance_tests.py mode=blade csv=<hv.csv> time=<runtime_log.csv> verbose=true
    python significance_tests.py csv=<path> reference=MOBO_MOEAD_EI_Hybrid_Fixed alpha=0.05
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, ttest_ind
from omegaconf import OmegaConf

LABELS = {
    "MOBO_MOEAD_EI_Hybrid_Fixed": "MOEAD-EI Hybrid",
    "MOBOImprovedScalarizedEI": "Improved-Scalarized-EI",
    "MOBORandomForestLCBPBI_CD": "RF-LCB-PBI",
    "MOBORandomForestParEGO_Batch_Refined": "RF-ParEGO-Batch",
    "MOBOEA_HGBR": "HGBR",
    "MOBORFPBIUCB": "RF-PBI-UCB",
    "BoFireQparEGOWrapper": "qParEGO",
    "RandomSearchMO": "Random Search",
    "NSGA2Wrapper": "NSGA-II",
    "NSGA3Wrapper": "NSGA-III",
    "IOCSAMOCOBRAWrapper": "IOC-SAMO-COBRA",
}

# Generated algorithms compared head-to-head against the SOTA reference.
GENERATED_DEFAULT = [
    "MOBO_MOEAD_EI_Hybrid_Fixed",
    "MOBOImprovedScalarizedEI",
    "MOBORandomForestLCBPBI_CD",
    "MOBORandomForestParEGO_Batch_Refined",
]
REFERENCE_DEFAULT = "BoFireQparEGOWrapper"  # qParEGO (SOTA)


def per_problem_matrix(csv_path):
    """Return DataFrame indexed by Problem, columns = Algorithm, values =
    per-problem max-normalized mean final HV."""
    df = pd.read_csv(csv_path)
    final = (
        df.sort_values("Epoch")
        .groupby(["Algorithm", "Problem", "Repeat"])
        .HV.last()
        .reset_index()
    )
    mean_hv = final.groupby(["Algorithm", "Problem"]).HV.mean().reset_index()
    mat = mean_hv.pivot(index="Problem", columns="Algorithm", values="HV")
    # max-normalize PER PROBLEM: divide each row (problem) by the max across
    # algorithms on that problem. (Row-wise division preserves within-problem
    # ordering, so Friedman ranks are unaffected; it only rescales for Wilcoxon.)
    mat = mat.div(mat.max(axis=1), axis=0)
    return mat


def holm(pvals):
    """Holm step-down adjusted p-values for a list of raw p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)  # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj


def _final_values(csv_path, value_col):
    """Per (Algorithm, Problem, Repeat) value. For HV, take the final epoch."""
    df = pd.read_csv(csv_path)
    if value_col == "HV" and "Epoch" in df.columns:
        df = (
            df.sort_values("Epoch")
            .groupby(["Algorithm", "Problem", "Repeat"])
            .HV.last()
            .reset_index()
        )
    return df


def per_problem_tests(csv_path, value_col, reference, generated, alpha=0.05,
                      higher_is_better=True):
    """BLADE-style per-problem comparison (no multiple-comparison correction).

    For every problem, run a two-sample Welch t-test (independent samples over
    repeats) of each generated algorithm vs the reference. Count the problems
    where the generated algorithm is significantly better / worse / not
    different, and the median per-problem ratio in the generated algorithm's
    favour (HV: gen/ref; time: ref/gen = speed-up).
    """
    df = _final_values(csv_path, value_col)
    probs = sorted(df.Problem.unique())
    out = {}
    for gen in generated:
        better = worse = ns = 0
        rows, ratios = [], []
        for p in probs:
            a = df[(df.Algorithm == gen) & (df.Problem == p)][value_col].values
            b = df[(df.Algorithm == reference) & (df.Problem == p)][value_col].values
            if len(a) < 2 or len(b) < 2:
                ns += 1
                continue
            _, pv = ttest_ind(a, b, equal_var=False)
            ma, mb = a.mean(), b.mean()
            gen_better = (ma > mb) if higher_is_better else (ma < mb)
            if pv < alpha and gen_better:
                better += 1
                tag = "+"
            elif pv < alpha and not gen_better:
                worse += 1
                tag = "-"
            else:
                ns += 1
                tag = "."
            ratios.append((ma / mb) if higher_is_better else (mb / ma))
            rows.append((p, ma, mb, pv, tag))
        out[gen] = dict(
            better=better, worse=worse, ns=ns, n=len(probs),
            rows=rows,
            median_ratio=float(np.median(ratios)) if ratios else float("nan"),
        )
    return out


def run_blade(hv_csv, time_csv=None, reference=REFERENCE_DEFAULT,
              generated=None, alpha=0.05, verbose=False):
    """BLADE-style report: per-problem t-tests on HV and (optionally) time."""
    generated = generated or GENERATED_DEFAULT
    name = lambda a: LABELS.get(a, a)
    print(f"\nBLADE-style per-problem tests (Welch t-test, no correction)")
    print(f"Reference = {name(reference)}  |  alpha={alpha}")

    print("\n--- Final hypervolume (higher = better) ---")
    res = per_problem_tests(hv_csv, "HV", reference, generated, alpha, True)
    for gen in generated:
        r = res[gen]
        print(
            f"  {name(gen):22s}: sig-better {r['better']}/{r['n']}, "
            f"sig-worse {r['worse']}/{r['n']}, n.s. {r['ns']}/{r['n']}  "
            f"(median {r['median_ratio']:.2f}x HV)"
        )
        if verbose:
            for p, ma, mb, pv, tag in r["rows"]:
                print(f"      [{tag}] {p:7s} gen={ma:.4f} ref={mb:.4f} p={pv:.4f}")

    if time_csv:
        print("\n--- Computational time (lower = better) ---")
        rest = per_problem_tests(time_csv, "ExecutionTime", reference,
                                 generated, alpha, False)
        for gen in generated:
            r = rest[gen]
            print(
                f"  {name(gen):22s}: sig-faster {r['better']}/{r['n']}, "
                f"sig-slower {r['worse']}/{r['n']}, n.s. {r['ns']}/{r['n']}  "
                f"(median {r['median_ratio']:.1f}x faster)"
            )
            if verbose:
                for p, ma, mb, pv, tag in r["rows"]:
                    print(f"      [{tag}] {p:7s} gen={ma:.1f}s ref={mb:.1f}s p={pv:.4f}")


def run(csv_path, reference=None, alpha=0.05):
    mat = per_problem_matrix(csv_path)
    algos = list(mat.columns)
    n_problems = mat.shape[0]
    name = lambda a: LABELS.get(a, a)

    print(f"\nCSV: {csv_path}")
    print(f"Problems: {n_problems}  |  Algorithms: {len(algos)}")
    if n_problems < 6:
        print(
            f"  WARNING: only {n_problems} problems — rank tests are severely "
            "underpowered (Wilcoxon cannot reach p<0.05 below ~6 pairs). "
            "Treat these results as descriptive only."
        )

    # --- Friedman omnibus + mean ranks (rank 1 = best per problem) ---
    cols = [mat[a].values for a in algos]
    chi2, p_fried = friedmanchisquare(*cols)
    # ranks per problem: higher HV better -> rank by -value
    ranks = np.vstack(
        [rankdata(-mat.loc[p].values, method="average") for p in mat.index]
    )
    mean_ranks = ranks.mean(axis=0)
    print(
        f"\nFriedman: chi2={chi2:.3f}, p={p_fried:.3e}  "
        f"({'significant' if p_fried < alpha else 'n.s.'} at alpha={alpha})"
    )
    print("Mean ranks (lower = better):")
    for a, r in sorted(zip(algos, mean_ranks), key=lambda x: x[1]):
        print(f"  {name(a):18s} {r:.3f}")

    # --- reference choice: best mean rank if not given ---
    if reference is None:
        reference = algos[int(np.argmin(mean_ranks))]
    print(
        f"\nPairwise Wilcoxon signed-rank vs reference = {name(reference)} "
        f"(Holm-corrected, two-sided):"
    )
    others = [a for a in algos if a != reference]
    raw = []
    for a in others:
        x, y = mat[reference].values, mat[a].values
        try:
            _, p = wilcoxon(x, y)  # two-sided
        except ValueError:
            p = 1.0  # all-zero differences
        raw.append(p)
    adj = holm(raw)
    for a, pr, pa in sorted(zip(others, raw, adj), key=lambda t: t[2]):
        better = "higher" if mat[reference].mean() > mat[a].mean() else "lower"
        sig = "*" if pa < alpha else " "
        print(
            f"  {sig} {name(reference)} vs {name(a):18s} "
            f"p_raw={pr:.3e}  p_holm={pa:.3e}  ({name(reference)} {better})"
        )
    print(
        f"  (* = significant at alpha={alpha}; reference is {better if others else ''})"
    )


def _parse_list(val):
    if val is None:
        return None
    if isinstance(val, str):
        return [s for s in val.split(",") if s]
    return list(val)


if __name__ == "__main__":
    cli = OmegaConf.from_cli(sys.argv[1:])
    csv = cli.get("csv", "paper_data/benchmark_stage2/hv_benchmark_log.csv")
    if not csv:
        print(__doc__)
        sys.exit("ERROR: pass csv=<hv_benchmark_log.csv>")
    mode = str(cli.get("mode", "demsar")).lower()
    alpha = float(cli.get("alpha", 0.05))

    if mode == "blade":
        time_csv = cli.get("time", None)
        if time_csv is None and "hv_benchmark_log.csv" in str(csv):
            cand = str(csv).replace("hv_benchmark_log.csv", "runtime_log.csv")
            if os.path.exists(cand):
                time_csv = cand
        run_blade(
            str(csv),
            time_csv=str(time_csv) if time_csv else None,
            reference=str(cli.get("reference", REFERENCE_DEFAULT)),
            generated=_parse_list(cli.get("generated", None)),
            alpha=alpha,
            verbose=bool(cli.get("verbose", False)),
        )
    else:
        run(
            str(csv),
            reference=cli.get("reference", None),
            alpha=alpha,
        )
