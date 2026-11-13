"""The ablation runner's path logic, with a fake run in place of the algorithm.

Each block gets an importance weight and a configuration's HV is the sum of the
weights of its T blocks. Going S -> T, each round should therefore add the most
important remaining block; going T -> S it should remove the least important
one first; and two equally important blocks should be taken in block order and
logged as a tie.
"""

import concurrent.futures as cf
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from ablation import END, START
from ablation import run_ablation as ra

WEIGHTS = [5, 1, 9, 3, 3, 7, 2, 8, 4, 6]  # blocks 4 and 5 are equally important


def _fake_run(task):
    algo, spec, seed, budget, threads, run_timeout, fast = task
    hv = sum(w for w, c in zip(WEIGHTS, algo) if c == "T") + (100 if spec.name == "b" else 0)
    basic = SimpleNamespace(hv_hist=[0.0, float(hv)], execution_time=0.01, n_evals=2,
                            timed_out=False, diagnostics={}, error=None, error_type=None)
    return algo, spec.name, seed, basic


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "_run", _fake_run)
    monkeypatch.setattr(ra, "_executor", lambda n: cf.ThreadPoolExecutor(max_workers=n))
    monkeypatch.setattr(ra, "_write_environment", lambda *a, **k: None)

    def go(stage, fast_predict=False):
        cfg = OmegaConf.create({
            "stage": stage, "budget": 2, "output_dir": str(tmp_path), "fast_predict": fast_predict,
            "pilot_seeds": [1], "path_seeds": [1, 2], "validation_seeds": [3],
            "max_workers": 4, "threads_per_run": 1, "run_timeout": None,
            "problems": [{"name": "a", "dim": 2, "n_obj": 2, "ref_point": [1.0, 1.0]},
                         {"name": "b", "dim": 2, "n_obj": 2, "ref_point": [1.0, 1.0]}],
        })
        ra.run(cfg)
        return tmp_path

    return go


def _blocks_in_order(path, direction):
    chosen = path[(path.Direction == direction) & path.Chosen].sort_values("Round")
    return chosen.ChangedBlock.tolist()


def test_paths_follow_block_importance_with_block_order_ties(runner):
    out = runner("path")
    path = pd.read_csv(out / "path_log.csv")
    by_weight = sorted(range(1, 11), key=lambda b: (-WEIGHTS[b - 1], b))
    assert _blocks_in_order(path, "forward") == by_weight
    assert _blocks_in_order(path, "backward") == sorted(range(1, 11), key=lambda b: (WEIGHTS[b - 1], b))
    # every round of each direction tried every remaining block once
    assert path.groupby("Direction").size().tolist() == [55, 55]
    # the round where blocks 4 and 5 were both open and best is logged as a tie
    tie_rounds = path[path.Tie].groupby(["Direction", "Round"]).ChangedBlock.apply(sorted)
    assert all(blocks == [4, 5] for blocks in tie_rounds)
    assert len(tie_rounds) == 2


def test_runner_resumes_without_repeating_runs_or_decisions(runner):
    out = runner("path")
    n_runs = len(pd.read_csv(out / "run_status_log.csv"))
    n_rows = len(pd.read_csv(out / "path_log.csv"))
    runner("path")
    assert len(pd.read_csv(out / "run_status_log.csv")) == n_runs
    assert len(pd.read_csv(out / "path_log.csv")) == n_rows


def test_runner_refuses_to_mix_exact_and_fast_runs(runner):
    out = runner("pilot")
    with pytest.raises(SystemExit, match="fast_predict"):
        runner("pilot", fast_predict=True)
    assert not pd.read_csv(out / "run_status_log.csv").FastPredict.any()


def test_pilot_runs_the_ends_and_round_one_on_one_seed(runner):
    out = runner("pilot")
    status = pd.read_csv(out / "run_status_log.csv")
    assert set(status.Repeat) == {1}
    assert status.Algorithm.nunique() == 2 + 10 + 10  # two ends, round 1 of both directions
    assert (out / "path_log_pilot.csv").exists() and not (out / "path_log.csv").exists()


def test_validation_reruns_exactly_the_path_configurations(runner):
    out = runner("path")
    runner("validate")
    status = pd.read_csv(out / "run_status_log.csv")
    path = pd.read_csv(out / "path_log.csv")
    on_path = set(path.loc[path.Chosen, "Candidate"]) | {START, END}
    assert set(status.loc[status.Repeat == 3, "Algorithm"]) == on_path
    # each path has 11 configurations; with this additive fake the two paths
    # retrace each other except around the tie, so most are shared
    for direction, start in (("forward", START), ("backward", END)):
        steps = {start} | set(path.loc[path.Chosen & (path.Direction == direction), "Candidate"])
        assert len(steps) == 11


def test_one_real_run_goes_through_the_benchmark_run_function():
    from llamevol.evaluator.multiobj_evaluator import MOOProblemSpec

    spec = MOOProblemSpec(name="zdt1", dim=3, n_obj=2, ref_point=[1.1, 7.2])
    algo, problem, seed, basic = ra._run((END, spec, 2, 20, 1, None, False))
    assert (algo, problem, seed) == (END, "zdt1", 2)
    assert basic.error is None and basic.n_evals == 20
    assert len(basic.hv_hist) == 20 and basic.torch_threads == 1
    assert basic.diagnostics["config"] == END
