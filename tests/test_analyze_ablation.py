"""The ablation analysis: Holm's correction, the comparisons, and a full run on
synthetic logs in the runner's format."""

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from ablation.config import BLOCKS, END, START, neighbours

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytest.importorskip("matplotlib")
_spec = importlib.util.spec_from_file_location("analyze_ablation", ROOT / "paper_reproduction" / "analyze_ablation.py")
aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aa)

PROBLEMS = ["p1", "p2", "p3", "p4", "p5", "p6"]


def test_holm():
    adj = aa.holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
    assert np.all(aa.holm(np.ones(3)) == 1)


def _fake_results(tmp_path, effect_block=8, confirm=()):
    """Logs of a full run: block ``effect_block`` (0-based) adds 10% HV on every problem.
    ``confirm``: ``(configurations, seeds)`` of a confirmation."""
    rng = np.random.default_rng(0)
    rows, path_rows = [], []

    def hv(config, problem, seed):
        base = 1.0 + PROBLEMS.index(problem)
        return base * (1.1 if config[effect_block] == "T" else 1.0) * (1 + 0.001 * rng.standard_normal())

    def add(config, seeds):
        for p in PROBLEMS:
            for s in seeds:
                rows.append(dict(Algorithm=config, Problem=p, Repeat=s, Evaluations=400,
                                 FinalHV=hv(config, p, s), FastPredict=True, TimedOut=False,
                                 AskFallbacks=0, ErrorType=None, Error=None, FirstFallbackError=None))

    on_path = set()
    for d, (cur, target) in {"forward": (START, END), "backward": (END, START)}.items():
        on_path.add(cur)
        r = 1
        while cur != target:
            cands = neighbours(cur, target)
            # the effect block first, then block order
            chosen = next((c for b, c in cands if b == effect_block), cands[0][1])
            for b, c in cands:
                add(c, [1, 2])
                path_rows.append(dict(Direction=d, Round=r, Candidate=c, ChangedBlock=b + 1,
                                      AvgRank=1.0 if c == chosen else 2.0, Chosen=c == chosen, Tie=False))
            cur, r = chosen, r + 1
            on_path.add(cur)
    for c in on_path:
        add(c, [3, 4, 5, 6, 7, 8])
    add(START, [1, 2])
    add(aa.PAIRS["moead"]["baseline"], [1, 2])
    for configs, seeds in confirm:
        for c in configs:
            add(c, seeds)
    status = pd.DataFrame(rows).drop_duplicates(["Algorithm", "Problem", "Repeat"])
    status.to_csv(tmp_path / "run_status_log.csv", index=False)
    status.assign(ExecutionTime=1.0)[["Algorithm", "Problem", "Repeat", "ExecutionTime"]].to_csv(
        tmp_path / "runtime_log.csv", index=False)
    pd.DataFrame(path_rows).to_csv(tmp_path / "path_log.csv", index=False)
    return tmp_path


def test_full_analysis_on_synthetic_logs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _fake_results(results, effect_block=8)
    cfg = OmegaConf.create(dict(
        pair="moead", budget=400, results=str(results), out=str(tmp_path / "out"),
        path_seeds=[1, 2], validation_seeds=[3, 4, 5, 6, 7, 8],
        problems=[dict(name=p) for p in PROBLEMS]))
    aa.main(cfg)
    out = tmp_path / "out"
    for f in ("path.csv", "steps.csv", "vs_end.csv", "per_problem.csv", "baseline.csv", "runtime.csv",
              "summary.md", "ablation_path.pdf", "ablation_steps.png"):
        assert (out / f).exists(), f
    steps = pd.read_csv(out / "steps.csv")
    assert len(steps) == 2 * len(BLOCKS)
    assert (steps.Holm >= steps.Wilcoxon - 1e-12).all()
    fwd = steps[steps.Direction == "forward"].set_index("Round")
    assert fwd.loc[1, "Step"] == "+search" and fwd.loc[1, "Higher"] == len(PROBLEMS)
    assert fwd.loc[1, "MedianRelPct"] == pytest.approx(10, abs=0.5)
    assert fwd.loc[1, "SigBetter"] == len(PROBLEMS)  # six seeds each: Mann-Whitney p = 0.002
    others = steps[steps.Block != "search"]
    assert (others.MedianRelPct.abs() < 0.5).all() and (others.SigBetter + others.SigWorse <= 2).all()
    vs_end = pd.read_csv(out / "vs_end.csv")
    has_search = vs_end.Configuration.str[8] == "T"
    assert (vs_end[has_search].MedianRelPct.abs() < 0.5).all()
    assert vs_end[~has_search].MedianRelPct.between(-10, -8).all()  # 1/1.1 - 1 = -9.1%
    assert START in set(vs_end.Configuration)


def test_confirmation_is_one_test_on_its_own_seeds(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    with_search, without = "SSSSSSSSTS", START
    _fake_results(results, effect_block=8, confirm=[((with_search, without), [21, 22, 23, 24])])
    base = dict(pair="moead", budget=400, results=str(results), path_seeds=[1, 2],
                validation_seeds=[3, 4, 5, 6, 7, 8], problems=[dict(name=p) for p in PROBLEMS])
    aa.main(OmegaConf.create(dict(base, out=str(tmp_path / "plain"))))
    plain = (tmp_path / "plain" / "summary.md").read_text()
    cfg = OmegaConf.create(dict(base, out=str(tmp_path / "out"),
                                confirm=dict(config=with_search, against=without, seeds=[21, 22, 23, 24])))
    aa.main(cfg)
    out = tmp_path / "out"
    # the ablation's own numbers do not see the confirmation's runs
    for f in ("steps.csv", "vs_end.csv", "per_problem.csv", "runtime.csv"):
        assert (out / f).read_text() == (tmp_path / "plain" / f).read_text(), f
    summary = (out / "summary.md").read_text()
    assert summary.startswith(plain.split("## Confirmation")[0].rstrip())
    conf = pd.read_csv(out / "confirm.csv")
    assert list(conf.Problem) == PROBLEMS
    assert conf.RelPct.between(9, 11).all()
    assert (conf.MannWhitney < 0.05).all()  # four seeds each: p = 0.029 per problem
    assert (conf.Holm >= conf.MannWhitney).all() and not (conf.Holm < 0.05).any()  # but 6 x 0.029 > 0.05
    assert "seeds 21-24" in summary
    assert "Wilcoxon p=0.0312 (pre-specified, alpha 0.05): confirmed" in summary
    assert "after Holm over the problems: 0 better / 0 worse" in summary


def test_confirmation_waits_for_its_runs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _fake_results(results, effect_block=8, confirm=[(("SSSSSSSSTS",), [21, 22])])  # one side only
    aa.main(OmegaConf.create(dict(
        pair="moead", budget=400, results=str(results), out=str(tmp_path / "out"),
        path_seeds=[1, 2], validation_seeds=[3, 4, 5, 6, 7, 8], problems=[dict(name=p) for p in PROBLEMS],
        confirm=dict(config="SSSSSSSSTS", against=START, seeds=[21, 22]))))
    assert not (tmp_path / "out" / "confirm.csv").exists()
    assert "Runs incomplete" in (tmp_path / "out" / "summary.md").read_text()
