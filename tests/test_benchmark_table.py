"""paper_reproduction/benchmark_table.py on small fake logs."""

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bt", ROOT / "paper_reproduction" / "benchmark_table.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def _bench(d, methods, runs=5, epochs=4, timed_out=()):
    """A benchmark folder: HV grows to (level + run/100) over the epochs."""
    d.mkdir(parents=True)
    hv, rt, st = [], [], []
    for m, level in methods.items():
        for p in bt.PROBLEMS:
            for r in range(1, runs + 1):
                final = level + r / 100
                for e in range(1, epochs + 1):
                    hv.append({"Algorithm": m, "Problem": p, "Repeat": r, "Epoch": e,
                               "HV": final * e / epochs})
                rt.append({"Algorithm": m, "Problem": p, "Repeat": r, "ExecutionTime": 10.0})
                st.append({"Algorithm": m, "Problem": p, "Repeat": r, "Evaluations": epochs,
                           "TimedOut": (m, p) in timed_out})
    pd.DataFrame(hv).to_csv(d / "hv_benchmark_log.csv", index=False)
    pd.DataFrame(rt).to_csv(d / "runtime_log.csv", index=False)
    pd.DataFrame(st).to_csv(d / "run_status_log.csv", index=False)


def _ablation(d, seeds, level=1.0):
    d.mkdir(parents=True)
    st = [{"Algorithm": "DDMTOLabMOEADEGO", "Problem": p, "Repeat": s, "Evaluations": 400,
           "FinalHV": level + s / 100, "TimedOut": False} for p in bt.PROBLEMS for s in seeds]
    st += [{"Algorithm": "SSSSSSSSSS", "Problem": p, "Repeat": s, "Evaluations": 400,
            "FinalHV": 9.0, "TimedOut": False} for p in bt.PROBLEMS for s in seeds]
    pd.DataFrame(st).to_csv(d / "run_status_log.csv", index=False)
    pd.DataFrame([{"Algorithm": r["Algorithm"], "Problem": r["Problem"], "Repeat": r["Repeat"],
                   "ExecutionTime": 100.0} for r in st]).to_csv(d / "runtime_log.csv", index=False)


@pytest.fixture
def cfg(tmp_path):
    _bench(tmp_path / "stage", {bt.HYBRID: 2.0, bt.ISEI: 1.5, "RandomSearchMO": 0.5,
                                "IOCSAMOCOBRAWrapper": 3.0})
    _bench(tmp_path / "qehvi", {"BoFireQEHVIWrapper": 1.8}, timed_out={("BoFireQEHVIWrapper", "wfg7")})
    _bench(tmp_path / "timing", {bt.HYBRID: 0, bt.ISEI: 0, "RandomSearchMO": 0}, runs=1)
    _ablation(tmp_path / "abl", [1, 2, 3, 4, 5])
    return bt.OmegaConf.merge(bt.OmegaConf.create(bt.DEFAULTS), {
        "stage": [str(tmp_path / "stage")], "qehvi": [str(tmp_path / "qehvi")],
        "qehvi_rerun": [str(tmp_path / "none")], "ablation": [str(tmp_path / "abl")],
        "timing": [str(tmp_path / "timing")], "extra": [], "extra_isei": [], "extra_ablation": [],
        "out": str(tmp_path / "out")})


def test_collects_every_source_and_leaves_cobra_out(cfg):
    runs, times, missing = bt.collect(cfg)
    assert "IOCSAMOCOBRAWrapper" not in set(runs.Method)
    assert set(runs.Method) == {bt.HYBRID, bt.ISEI, "RandomSearchMO", "BoFireQEHVIWrapper",
                                "DDMTOLabMOEADEGO"}
    assert (runs.groupby(["Method", "Problem"]).size() == 5).all()
    assert runs[runs.Method == "DDMTOLabMOEADEGO"].HV.max() == pytest.approx(1.05)
    assert any("none" in m for m in missing)
    # Times only from controlled runs: the timing pass, qEHVI and the ablation baseline.
    assert set(times.Method) == {bt.HYBRID, bt.ISEI, "RandomSearchMO", "BoFireQEHVIWrapper",
                                 "DDMTOLabMOEADEGO"}


def test_rerun_replaces_its_problems_and_extra_runs_add_runs_6_to_10(cfg, tmp_path):
    rerun = tmp_path / "rerun"
    _bench(rerun, {"BoFireQEHVIWrapper": 5.0})
    h = pd.read_csv(rerun / "hv_benchmark_log.csv")
    h[h.Problem == "wfg7"].to_csv(rerun / "hv_benchmark_log.csv", index=False)
    _bench(tmp_path / "extra", {bt.HYBRID: 2.0, "RandomSearchMO": 0.5})
    cfg.qehvi_rerun = [str(rerun)]
    cfg.extra = [str(tmp_path / "extra")]
    runs, _, _ = bt.collect(cfg)
    q = runs[runs.Method == "BoFireQEHVIWrapper"]
    assert q[q.Problem == "wfg7"].HV.min() == pytest.approx(5.01)
    assert len(q[q.Problem == "wfg7"]) == 5
    assert sorted(runs[(runs.Method == bt.HYBRID) & (runs.Problem == "zdt1")].Run) == list(range(1, 11))


def test_statistics_and_outputs(cfg):
    runs, times, missing = bt.collect(cfg)
    summary, tests, per_problem, complete = bt.analyse(runs, times)
    assert summary.at[bt.HYBRID, "NormSynthetic"] == pytest.approx(1.0)
    assert summary.at[bt.HYBRID, "AvgRank"] == pytest.approx(1.0)
    assert summary.at["BoFireQEHVIWrapper", "TimedOutRuns"] == 5
    row = tests[(tests.Generated == bt.HYBRID) & (tests.Other == "RandomSearchMO")].iloc[0]
    # Better on all 15 problems: the smallest two-sided Wilcoxon p for n=15, and every
    # per-problem Mann-Whitney (5 against 5, p=0.0079) fails Holm over 15 problems.
    assert row.Higher == 15 and row.Wilcoxon == pytest.approx(2 / 2**15)
    assert row.BetterRaw == 15 and row.BetterHolm == 0
    assert not ((tests.Generated == bt.ISEI) & (tests.Other == bt.HYBRID)).any()
    bt.write(cfg.out, runs, times, missing, summary, tests, per_problem, complete)
    tex = (pathlib.Path(cfg.out) / "table.tex").read_text()
    assert tex.count("\\midrule") == 3 and "\\end{tabular}" in tex and "\\textbf{1.000}" in tex


def test_holm_matches_the_step_down_rule():
    assert np.allclose(bt.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
