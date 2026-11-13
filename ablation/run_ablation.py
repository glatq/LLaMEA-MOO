"""Run the Fawcett & Hoos ablation between MOEA/D-EGO and MOEAD-EI Hybrid.

    python -m ablation.run_ablation stage=<stage> [config=conf/ablation.yaml] [key=value ...]

Stages (each resumable: runs already in ``run_status_log.csv`` are skipped):

* ``pilot``     both ends and round 1 of both directions, on ``pilot_seeds``;
* ``path``      both ablation paths in full, on ``path_seeds``. Each round runs every
                configuration one block closer to the other end, then keeps the one
                with the best average rank of mean HV over the problems (ties: the
                lower block number). Every candidate and decision goes to
                ``path_log.csv``;
* ``validate``  the configurations on both paths, on ``validation_seeds`` (the
                numbers reported, free of the bias of picking the best of a round);
* ``baseline``  DDMTOLab's MOEA/D-EGO itself (``MO bench/ddmtolab_moead_ego.py``),
                on ``path_seeds``, for the main tables.

Every run goes through the benchmark's own per-run function
(``multiobj_evaluator._run_single_moo_rep``): one thread per run, one run per
physical core, the per-run time limit, HV after every evaluation. The logs have
the benchmark's columns, with the configuration string as ``Algorithm``.

``fast_predict`` (config) makes MOEA/D-EGO's two inner loops use closed-form
predictions from the fitted models (``ablation.fast_predict``). An output directory
holds runs of one setting only; the runner refuses to mix them. The baseline is
DDMTOLab's own code, always exact, and is logged with ``FastPredict`` false.

The pilot's runs are the seed-1 runs of round 1, so ``path`` reuses them.
"""

import concurrent.futures as cf
import os
import pathlib
import sys
import time
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from ablation.config import END, START, neighbours  # noqa: E402
from llamevol.evaluator.multiobj_evaluator import (  # noqa: E402
    MOOProblemSpec,
    _run_single_moo_rep,
    physical_core_count,
)

BASELINE = "DDMTOLabMOEADEGO"
DIRECTIONS = {"forward": (START, END), "backward": (END, START)}
MODULAR_CODE = (
    "import sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "from ablation.modular import ModularMOEAD\n"
)


# ------------------------------------------------------------------ one run
def _run(task):
    """Worker: one (algorithm, problem, seed) run. Returns the run result."""
    algo, spec, seed, budget, threads, run_timeout, fast = task
    if algo == BASELINE:
        code = (ROOT / "MO bench" / "ddmtolab_moead_ego.py").read_text()
        cls_name, kwargs = "DDMTOLabMOEADEGOWrapper", {"seed": seed}
    else:
        code, cls_name = MODULAR_CODE, "ModularMOEAD"
        kwargs = {"config": algo, "seed": seed, "fast_predict": fast}
    basic = _run_single_moo_rep(
        spec, seed - 1, budget, code, cls_name,
        cls=None, cls_init_kwargs=kwargs, cls_call_kwargs=None, injector=None,
        stop_event=None, calculate_hv_history=True,
        threads_per_run=threads, run_timeout=run_timeout,
    )
    return algo, spec.name, seed, basic


# ------------------------------------------------------------------- logging
class Logs:
    """Append-only CSV logs in ``output_dir`` (benchmark schema)."""

    def __init__(self, output_dir, fast_predict=False):
        self.dir = pathlib.Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fast_predict = bool(fast_predict)
        self.hv = self.dir / "hv_benchmark_log.csv"
        self.runtime = self.dir / "runtime_log.csv"
        self.status = self.dir / "run_status_log.csv"
        self.final = {}  # (algo, problem, seed) -> final HV
        if self.status.exists():
            st = pd.read_csv(self.status)
            done = st["FastPredict"].astype(bool) if "FastPredict" in st else pd.Series(False, index=st.index)
            modular = st["Algorithm"] != BASELINE  # DDMTOLab's own runs are always exact
            if (done[modular] != self.fast_predict).any():
                raise SystemExit(
                    f"{self.status} holds runs with fast_predict={not self.fast_predict}; "
                    f"use another output_dir for fast_predict={self.fast_predict}"
                )
            for r in st.itertuples():
                self.final[(r.Algorithm, r.Problem, int(r.Repeat))] = r.FinalHV

    @staticmethod
    def _append(path, rows):
        if rows:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)

    def record(self, algo, problem, seed, basic):
        hv = getattr(basic, "hv_hist", None)
        hv = [] if hv is None else list(hv)
        final = float(hv[-1]) if hv else 0.0  # a run that produced nothing scores worst
        diag = getattr(basic, "diagnostics", None) or {}
        self._append(self.hv, [
            {"Algorithm": algo, "Problem": problem, "Repeat": seed, "Epoch": e + 1, "HV": v}
            for e, v in enumerate(hv)
        ])
        self._append(self.runtime, [{
            "Algorithm": algo, "Problem": problem, "Repeat": seed,
            "ExecutionTime": float(basic.execution_time),
        }])
        self._append(self.status, [{
            "Algorithm": algo, "Problem": problem, "Repeat": seed,
            "Evaluations": getattr(basic, "n_evals", None), "FinalHV": final,
            "FastPredict": self.fast_predict and algo != BASELINE,
            "TimedOut": bool(getattr(basic, "timed_out", False)),
            "AskFallbacks": diag.get("ask_fallbacks"), "ErrorType": basic.error_type,
            "Error": (str(basic.error)[:300] if basic.error else None),
            "FirstFallbackError": diag.get("first_error"),
        }])
        self.final[(algo, problem, seed)] = final


# ------------------------------------------------------------ path decision
def choose(candidates, logs, problems, seeds):
    """The candidate with the best average rank of mean final HV over the problems.

    ``candidates`` are ``(changed_block, config)`` pairs. Ties go to the lower
    block number (the block earlier in the algorithm's flow).
    """
    configs = [c for _, c in candidates]
    mean = pd.DataFrame(
        {p.name: [np.mean([logs.final[(c, p.name, s)] for s in seeds]) for c in configs] for p in problems},
        index=configs,
    )
    avg = mean.rank(axis=0, ascending=False, method="average").mean(axis=1)
    best = avg.min()
    tied = [(b, c) for b, c in candidates if np.isclose(avg[c], best)]
    block, chosen = min(tied)
    return block, chosen, avg, len(tied) > 1


# -------------------------------------------------------------------- driver
def run(cfg):
    problems = [
        MOOProblemSpec(name=p.name, dim=p.dim, n_obj=p.n_obj, ref_point=list(p.ref_point))
        for p in cfg.problems
    ]
    budget = int(cfg.budget)
    threads = int(cfg.get("threads_per_run", 1))
    run_timeout = cfg.get("run_timeout", None)
    max_workers = cfg.get("max_workers", None) or physical_core_count()
    stage = cfg.stage
    fast = bool(cfg.get("fast_predict", False))
    logs = Logs(cfg.output_dir, fast)
    _write_environment(logs.dir, cfg, max_workers, threads, run_timeout)
    seeds = list({"pilot": cfg.pilot_seeds, "path": cfg.path_seeds,
                  "validate": cfg.validation_seeds, "baseline": cfg.path_seeds}[stage])
    # Path decisions: the pilot's (one seed) are kept apart from the real ones.
    path_log = logs.dir / ("path_log_pilot.csv" if stage == "pilot" else "path_log.csv")
    decided = {}
    if path_log.exists() and stage in ("pilot", "path"):
        old = pd.read_csv(path_log)
        for r in old[old.Chosen].itertuples():
            decided[(r.Direction, int(r.Round))] = r.Candidate

    def complete(algos):
        return all((a, p.name, s) in logs.final for a in algos for p in problems for s in seeds)

    with _executor(max_workers) as pool:
        pending = {}  # future -> (algo, problem, seed)

        def submit(algos):
            queued = set(pending.values())
            for a in algos:
                for p in problems:
                    for s in seeds:
                        key = (a, p.name, s)
                        if key not in logs.final and key not in queued:
                            task = (a, p, s, budget, threads, run_timeout, fast)
                            pending[pool.submit(_run, task)] = key
                            queued.add(key)

        state = {}
        if stage == "baseline":
            submit([BASELINE])
        elif stage == "validate":
            path = pd.read_csv(logs.dir / "path_log.csv")
            submit(sorted(set(path.loc[path.Chosen, "Candidate"]) | {START, END}))
        else:
            submit([START, END])
            max_rounds = 1 if stage == "pilot" else len(START)
            state = {d: {"current": a, "target": b, "round": 1, "done": False}
                     for d, (a, b) in DIRECTIONS.items()}
            for s in state.values():
                submit([c for _, c in neighbours(s["current"], s["target"])])

        def advance():
            """Decide every direction whose current round has all its runs."""
            for d, s in state.items():
                while not s["done"]:
                    cands = neighbours(s["current"], s["target"])
                    if not complete([c for _, c in cands]):
                        break
                    block, chosen, avg, tie = choose(cands, logs, problems, seeds)
                    if (d, s["round"]) in decided:
                        chosen = decided[(d, s["round"])]  # replay a logged decision
                    else:
                        logs._append(path_log, [
                            {"Direction": d, "Round": s["round"], "Candidate": c, "ChangedBlock": b + 1,
                             "AvgRank": avg[c], "Chosen": c == chosen,
                             "Tie": bool(tie and np.isclose(avg[c], avg.min()))}
                            for b, c in cands
                        ])
                        print(f"== {d} round {s['round']}: keep {chosen} (block {block + 1}"
                              f"{', tie' if tie else ''})", flush=True)
                    s["current"], s["round"] = chosen, s["round"] + 1
                    if s["current"] == s["target"] or s["round"] > max_rounds:
                        s["done"] = True
                    else:
                        submit([c for _, c in neighbours(s["current"], s["target"])])

        advance()
        while pending:
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                algo, prob, seed = pending.pop(fut)
                try:
                    _, _, _, basic = fut.result()
                except Exception as e:  # the worker itself failed
                    basic = SimpleNamespace(hv_hist=None, execution_time=float("nan"),
                                            error=f"{type(e).__name__}: {e}", error_type="WorkerError")
                logs.record(algo, prob, seed, basic)
                print(f"[{time.strftime('%H:%M:%S')}] {algo} {prob} seed {seed}: "
                      f"HV {logs.final[(algo, prob, seed)]:.6g} ({len(pending)} pending)", flush=True)
            advance()


def _executor(max_workers):
    """The pool the runs go to (tests swap in a thread pool)."""
    return cf.ProcessPoolExecutor(max_workers=max_workers)


def _write_environment(out_dir, cfg, max_workers, threads, run_timeout):
    from benchmark_best_codes import write_environment

    ev = SimpleNamespace(max_workers=max_workers, threads_per_run=threads,
                         run_timeout=run_timeout, timeout=None)
    write_environment(os.path.join(out_dir, f"environment_{cfg.stage}.json"), ev,
                      SimpleNamespace(budget=cfg.budget, repeat=len(cfg.path_seeds)))


if __name__ == "__main__":
    cli = OmegaConf.from_cli(sys.argv[1:])
    base = OmegaConf.load(cli.pop("config", str(ROOT / "conf" / "ablation.yaml")))
    cfg = OmegaConf.merge(base, cli)
    if cfg.get("stage") not in ("pilot", "path", "validate", "baseline"):
        raise SystemExit("stage=pilot|path|validate|baseline is required")
    run(cfg)
