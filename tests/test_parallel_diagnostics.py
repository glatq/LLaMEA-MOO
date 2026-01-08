import time
import os
import pytest
from llamevol.evaluator.multiobj_evaluator import MultiObjEvaluator, MOOProblemSpec

# Diagnostic algorithm that logs its own timing
DIAGNOSTIC_CODE = """
import numpy as np
import time
import os
from datetime import datetime

class DiagnosticOptimizer:
    def __init__(self, budget, dim, bounds=None):
        self.budget = budget

    def __call__(self, func):
        # Using print(..., flush=True) helps see the "burst" in the terminal
        print(f"\\n[Worker {os.getpid()}] START at {datetime.now().strftime('%H:%M:%S.%f')}", flush=True)
        time.sleep(5) 
        for _ in range(self.budget):
            func(np.zeros(1))
        return None
"""


def test_m2_core_saturation():
    """
    Verifies that all 10 cores on the M2 are utilized.
    10 tasks @ 5s each = 50s total work.
    In parallel, this should take ~7-9s (5s work + 2-3s Mac overhead).
    """
    num_cores = os.cpu_count()  # Should be 10 on your MBP
    problem = [MOOProblemSpec(name="zdt1", dim=2, n_obj=2)]

    # We set repeat to exactly the number of cores
    evaluator = MultiObjEvaluator(
        budget=5, problems=problem, repeat=num_cores, timeout=30
    )

    start_time = time.perf_counter()
    res = evaluator.evaluate(code=DIAGNOSTIC_CODE, cls_name="DiagnosticOptimizer")
    total_duration = time.perf_counter() - start_time

    # 1. Check integrity
    assert len(res.result) == num_cores
    assert res.error is None

    # 2. Check Parallelism
    # If it were sequential, it would take at least (10 cores * 5s) = 50s.
    # If it takes < 15s, it is impossible for it to be sequential.
    assert total_duration < 20.0, (
        f"Saturation test failed. Took {total_duration:.2f}s for {num_cores * 5}s of work. "
        "Either cores aren't being used or system overhead is extreme."
    )

    print(f"\n[M2 Check] Total duration: {total_duration:.2f}s")
    print(f"[M2 Check] Efficiency: {((num_cores * 5) / total_duration):.1f}x speedup")


def test_parallel_execution_speed():
    """
    Checks if the evaluator is faster than a sequential baseline.
    3 reps * 2s work per rep = 6s of pure work.
    """
    problem = [MOOProblemSpec(name="zdt1", dim=2, n_obj=2)]
    evaluator = MultiObjEvaluator(budget=5, problems=problem, repeat=3, timeout=15)

    start_time = time.perf_counter()
    res = evaluator.evaluate(code=DIAGNOSTIC_CODE, cls_name="DiagnosticOptimizer")
    total_duration = time.perf_counter() - start_time

    # 1. Calculate the pure 'work' done inside workers
    individual_times = [r.execution_time for r in res.result]
    sum_of_work = sum(individual_times)  # Should be ~6.0s

    # 2. Define a Sequential Baseline
    # In a sequential world, you pay the 'Startup Tax' for every single task.
    # On M2/macOS, this tax is roughly 0.5s-1.5s per task.
    startup_tax_estimate = 1.0 * len(res.result)
    sequential_baseline = sum_of_work + startup_tax_estimate

    # 3. Assertion: Are we faster than the baseline?
    # Even with overhead, parallel should beat a sequential run on an M2.
    assert total_duration < sequential_baseline, (
        f"Execution was not efficient. Total ({total_duration:.2f}s) "
        f"is not faster than Sequential Baseline ({sequential_baseline:.2f}s)."
    )

    # Verify we didn't just fail upward
    assert res.error is None
    assert len(res.result) == 3


def test_global_timeout_enforcement():
    """
    Verify that if the total batch work (6s) exceeds the global timeout (3s),
    the evaluator kills the processes and returns a TimeoutError.
    """
    problem = [MOOProblemSpec(name="zdt1", dim=2, n_obj=2)]
    # Total work is 6s, but we only give it 3s.
    evaluator = MultiObjEvaluator(budget=5, problems=problem, repeat=3, timeout=3)

    start_time = time.perf_counter()
    res = evaluator.evaluate(code=DIAGNOSTIC_CODE, cls_name="DiagnosticOptimizer")
    duration = time.perf_counter() - start_time

    # ASSERTIONS
    assert res.error_type == "TimeoutError"
    # Result list should be empty (or at least incomplete) because of the break/shutdown
    assert len(res.result) < 3

    # The duration should be very close to our 3s timeout, not the 6s work time.
    assert (
        2.5 <= duration <= 4.5
    ), f"Timeout not enforced correctly. Took {duration:.2f}s."
    print(f"\nTimeout check passed: Caught TimeoutError at {duration:.2f}s.")


def test_repetition_integrity():
    problem = [MOOProblemSpec(name="zdt1", dim=2, n_obj=2)]
    evaluator = MultiObjEvaluator(budget=5, problems=problem, repeat=2, timeout=20)

    res = evaluator.evaluate(code=DIAGNOSTIC_CODE, cls_name="DiagnosticOptimizer")

    # This is the "Safety Catch"
    if not res.result:
        # If the list is empty, it means something went wrong in the background
        pytest.fail(
            f"Evaluator returned no results. Error: {res.error} (Type: {res.error_type})"
        )

    names = [r.name for r in res.result]
    assert "zdt1-rep1" in names
    assert "zdt1-rep2" in names
