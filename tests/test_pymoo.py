from pymoo.indicators.hv import HV
from pymoo.problems import get_problem
import numpy as np


def test_problem_load():
    problem = get_problem("zdt1")
    assert problem.name() == "ZDT1"


def test_hv_calculation():
    ref_point = np.array([10, 10])
    f_pareto = np.array([[2, 8], [4, 4], [8, 2]])
    hv = HV(ref_point=ref_point)
    assert hv(f_pareto) == 44.0
