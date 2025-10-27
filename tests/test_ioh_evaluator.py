import pytest
from llamevol.evaluator.evaluator_result import EvaluatorResult
from llamevol.evaluator.ioh_evaluator import IOHEvaluator
import concurrent.futures
from types import SimpleNamespace
import types


@pytest.fixture
def ioh_eval():
    ioheval = IOHEvaluator(
        problems=[
            2,
            4,
            6,
            8,
            12,
            14,
            18,
            15,
            21,
            23,
        ],
        instances=[
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ],
    )
    return ioheval


def test_init_(ioh_eval):
    assert isinstance(ioh_eval, IOHEvaluator)

    actual_dim = ioh_eval.dim
    expected_dim = 5
    assert expected_dim == actual_dim

    actual_budget = ioh_eval.budget
    expected_budget = 40
    assert expected_budget == actual_budget

    actual_problems = ioh_eval.problems
    expected_problems = [
        2,
        4,
        6,
        8,
        12,
        14,
        18,
        15,
        21,
        23,
    ]
    assert expected_problems == actual_problems

    actual_instances = [[1]] * len(actual_problems)
    expected_instances = [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ]
    assert expected_instances == actual_instances

    actual_repeat = ioh_eval.reapeat
    expected_repeat = 1
    assert expected_repeat == actual_repeat


def test_eval_budget(ioh_eval):
    actual = ioh_eval.eval_bugdet()
    expected = 40
    assert expected == actual


def test_problem_name(ioh_eval):
    actual = ioh_eval.problem_name()
    expected_problems = [
        2,
        4,
        6,
        8,
        12,
        14,
        18,
        15,
        21,
        23,
    ]
    expected = "f2_f4_f6_f8_f12_f14_f18_f15_f21_f23"
    assert expected == actual


def test_problem_prompt(ioh_eval):
    actual = ioh_eval.problem_prompt()
    expected = "Problems from the BBOB test suite with dimensions 5\n"
    assert expected == actual


class FakeTaskManager:
    def __init__(self, seq):
        self.seq = list(seq)
        self.shutdown_called = None

    def as_completed(self, futures, timeout=None):
        for f in self.seq:
            yield f

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_called = (wait, cancel_futures)


def make_finished_future(val):
    f = concurrent.futures.Future()
    f.set_result(val)
    return f


def test_start_as_completed(ioh_eval, monkeypatch):
    def ok(self, *res):
        return SimpleNamespace(error=None, error_type=None)

    monkeypatch.setattr(
        ioh_eval,
        f"_{ioh_eval.__class__.__name__}__process_results",
        types.MethodType(ok, ioh_eval),
    )

    f1, f2 = make_finished_future(("a",)), make_finished_future(("b",))
    futures = {f1: 1, f2: 2}
    tm = FakeTaskManager([f1, f2])

    eval_result = EvaluatorResult()
    eval_result.result = []

    out = ioh_eval.start_as_completed(
        eval_result,
        futures,
        timeout=5,
        task_manager=tm,
        cls_name="C",
        interval=1,
        total_tasks=2,
    )

    assert out is eval_result
    assert out.error is None
    assert len(out.result) == 2
    assert tm.shutdown_called == (True, False)


def make_basic(ok=True):
    # tiny struct matching what __process_results returns
    if ok:
        return SimpleNamespace(
            error=None,
            error_type=None,
            log_y_aoc=1.0,
            log_y_aoc_ioh=2.0,
            execution_time=0.1,
        )
    return SimpleNamespace(
        error=RuntimeError("boom"),
        error_type="RuntimeError",
        log_y_aoc=0.0,
        log_y_aoc_ioh=0.0,
        execution_time=0.0,
    )


# tiny struct matching what __process_results returns
def make_basic(ok=True):
    if ok:
        return SimpleNamespace(
            error=None,
            error_type=None,
            log_y_aoc=1.0,
            log_y_aoc_ioh=2.0,
            execution_time=0.1,
        )
    return SimpleNamespace(
        error=RuntimeError("boom"),
        error_type="RuntimeError",
        log_y_aoc=0.0,
        log_y_aoc_ioh=0.0,
        execution_time=0.0,
    )


def test_evaluate_stops_on_error(ioh_eval, monkeypatch):
    obj = ioh_eval
    obj.gpu_name = None
    obj.ignore_over_budget = False
    obj.use_mpi = False
    obj.use_mpi_future = False
    obj.use_multi_process = False
    obj.max_eval_workers = 0  # force sequential branch
    obj.timeout = 30

    # drive three tasks
    obj.obj_fn_params = [{"idx": i} for i in range(3)]
    obj._logging_eval_process = types.MethodType(lambda self, *_: None, obj)
    obj._check_timeout = types.MethodType(lambda self, *_: False, obj)

    # Patch the module-level helper correctly on the module where it's defined
    import llamevol.evaluator.ioh_evaluator as mod

    def fake_eval_block(**kwargs):
        # we don't care about the returned tuple contents since __process_results is faked
        return ("dummy",), None, None, 0.0, None, None

    monkeypatch.setattr(mod, "ioh_evaluate_block", fake_eval_block, raising=True)

    # First processed result is OK, second is error -> evaluate() should stop
    calls = {"n": 0}

    def __proc(self, *res):
        calls["n"] += 1
        return make_basic(ok=(calls["n"] == 1))

    mangled = f"_{obj.__class__.__name__}__process_results"
    setattr(obj, mangled, types.MethodType(__proc, obj))

    out = obj.evaluate(code="print(1)", cls_name="MyCls")

    assert out.error is not None
    assert out.error_type == "RuntimeError"
    assert len(out.result) == 1  # stopped after first OK then error
    assert out.score == 0.0  # error path zeroes aggregates
    assert out.total_execution_time == 0.0
