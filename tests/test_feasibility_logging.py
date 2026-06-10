"""PR3: feasibility logging + plotting.

Covers the pure feasible-fraction curve used by the benchmark logger and a smoke
test that the feasibility convergence plot renders from a synthetic CSV.

Note: ``plot_from_csv.py`` is a gitignored, local-only analysis script, so the
plotting smoke test is skipped when it is absent (e.g. on a clean checkout/CI).
The benchmark-logging test does not depend on it.
"""

import importlib.util
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless rendering before pyplot is imported anywhere

import numpy as np
import pandas as pd
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PLOT_SCRIPT = _ROOT / "plot_from_csv.py"


def _load(module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, _ROOT / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# feasible_fraction_curve (benchmark_best_codes)                              #
# --------------------------------------------------------------------------- #
def test_feasible_fraction_curve_values():
    bench = _load("benchmark_best_codes")
    # cv: [1, 0, 0, 2] -> feasible [0,1,1,0] -> cumfrac [0, 1/2, 2/3, 2/4]
    curve = bench.feasible_fraction_curve([1.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(curve, [0.0, 0.5, 2 / 3, 0.5])


def test_feasible_fraction_curve_edges():
    bench = _load("benchmark_best_codes")
    np.testing.assert_array_equal(bench.feasible_fraction_curve([]), np.empty(0))
    # all feasible -> all ones; all infeasible -> all zeros
    np.testing.assert_allclose(bench.feasible_fraction_curve([0.0, 0.0]), [1.0, 1.0])
    np.testing.assert_allclose(bench.feasible_fraction_curve([3.0, 1.0]), [0.0, 0.0])


# --------------------------------------------------------------------------- #
# make_feasibility_plots smoke                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _PLOT_SCRIPT.is_file(),
    reason="plot_from_csv.py is a gitignored local-only analysis script",
)
def test_make_feasibility_plots_writes_png(tmp_path):
    plot = _load("plot_from_csv")
    # Two repeats of a constrained run that becomes feasible over time.
    rows = []
    for rep in (1, 2):
        for ev, frac in enumerate([0.0, 0.5, 0.66, 0.75], start=1):
            rows.append(
                {
                    "Algorithm": "MOBOFoo",
                    "Problem": "mw1",
                    "Repeat": rep,
                    "Eval": ev,
                    "CV": max(0.0, 1.0 - 0.25 * ev),
                    "FeasibleFraction": frac,
                }
            )
    csv_path = tmp_path / "feasibility_log.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    plot.make_feasibility_plots(str(csv_path), output_dir=str(tmp_path))
    assert (tmp_path / "plot_mw1_feasibility.png").is_file()
