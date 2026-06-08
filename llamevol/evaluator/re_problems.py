"""Real-world engineering benchmark problems (RE suite) used in Phase 3.

Only the three *unconstrained* problems used in the paper are included
(RE21, RE34, RE37), taken verbatim from Tanabe & Ishibuchi (2020), "An
easy-to-use real-world multi-objective optimization problem suite", Applied
Soft Computing -- https://github.com/ryojitanabe/reproblems .

Each class exposes the minimal interface ``PymooMOProvider`` needs:
``n_obj``, ``xl``, ``xu`` and ``evaluate(X, return_values_of=["F"]) -> [F]``
(matching the subset of the pymoo Problem API the provider's wrapper calls).
"""

import numpy as np


class _REProblem:
    """Base: batches the single-point objective formula to the pymoo-like API."""

    problem_name = ""
    n_obj = 0

    def _evaluate_one(self, x):
        raise NotImplementedError

    def evaluate(self, X, return_values_of=("F",), **kwargs):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = np.array([self._evaluate_one(row) for row in X], dtype=float)
        return [F]


class RE21(_REProblem):
    """Four bar truss design (2 objectives, 4 variables)."""

    problem_name = "RE21"
    n_obj = 2

    def __init__(self):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        self.xu = np.full(4, 3 * tmp_val)
        self.xl = np.array(
            [tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]
        )

    def _evaluate_one(self, x):
        x1, x2, x3, x4 = x
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0
        f1 = L * ((2 * x1) + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)
        f2 = ((F * L) / E) * (
            (2.0 / x1)
            + (2.0 * np.sqrt(2.0) / x2)
            - (2.0 * np.sqrt(2.0) / x3)
            + (2.0 / x4)
        )
        return np.array([f1, f2])


class RE34(_REProblem):
    """Vehicle crashworthiness design (3 objectives, 5 variables)."""

    problem_name = "RE34"
    n_obj = 3

    def __init__(self):
        self.xl = np.full(5, 1.0)
        self.xu = np.full(5, 3.0)

    def _evaluate_one(self, x):
        x1, x2, x3, x4, x5 = x
        f1 = (
            1640.2823
            + (2.3573285 * x1)
            + (2.3220035 * x2)
            + (4.5688768 * x3)
            + (7.7213633 * x4)
            + (4.4559504 * x5)
        )
        f2 = (
            6.5856
            + (1.15 * x1)
            - (1.0427 * x2)
            + (0.9738 * x3)
            + (0.8364 * x4)
            - (0.3695 * x1 * x4)
            + (0.0861 * x1 * x5)
            + (0.3628 * x2 * x4)
            - (0.1106 * x1 * x1)
            - (0.3437 * x3 * x3)
            + (0.1764 * x4 * x4)
        )
        f3 = (
            -0.0551
            + (0.0181 * x1)
            + (0.1024 * x2)
            + (0.0421 * x3)
            - (0.0073 * x1 * x2)
            + (0.024 * x2 * x3)
            - (0.0118 * x2 * x4)
            - (0.0204 * x3 * x4)
            - (0.008 * x3 * x5)
            - (0.0241 * x2 * x2)
            + (0.0109 * x4 * x4)
        )
        return np.array([f1, f2, f3])


class RE37(_REProblem):
    """Rocket injector design (3 objectives, 4 variables)."""

    problem_name = "RE37"
    n_obj = 3

    def __init__(self):
        self.xl = np.full(4, 0.0)
        self.xu = np.full(4, 1.0)

    def _evaluate_one(self, x):
        xAlpha, xHA, xOA, xOPTT = x
        f1 = (
            0.692
            + (0.477 * xAlpha)
            - (0.687 * xHA)
            - (0.080 * xOA)
            - (0.0650 * xOPTT)
            - (0.167 * xAlpha * xAlpha)
            - (0.0129 * xHA * xAlpha)
            + (0.0796 * xHA * xHA)
            - (0.0634 * xOA * xAlpha)
            - (0.0257 * xOA * xHA)
            + (0.0877 * xOA * xOA)
            - (0.0521 * xOPTT * xAlpha)
            + (0.00156 * xOPTT * xHA)
            + (0.00198 * xOPTT * xOA)
            + (0.0184 * xOPTT * xOPTT)
        )
        f2 = (
            0.153
            - (0.322 * xAlpha)
            + (0.396 * xHA)
            + (0.424 * xOA)
            + (0.0226 * xOPTT)
            + (0.175 * xAlpha * xAlpha)
            + (0.0185 * xHA * xAlpha)
            - (0.0701 * xHA * xHA)
            - (0.251 * xOA * xAlpha)
            + (0.179 * xOA * xHA)
            + (0.0150 * xOA * xOA)
            + (0.0134 * xOPTT * xAlpha)
            + (0.0296 * xOPTT * xHA)
            + (0.0752 * xOPTT * xOA)
            + (0.0192 * xOPTT * xOPTT)
        )
        f3 = (
            0.370
            - (0.205 * xAlpha)
            + (0.0307 * xHA)
            + (0.108 * xOA)
            + (1.019 * xOPTT)
            - (0.135 * xAlpha * xAlpha)
            + (0.0141 * xHA * xAlpha)
            + (0.0998 * xHA * xHA)
            + (0.208 * xOA * xAlpha)
            - (0.0301 * xOA * xHA)
            - (0.226 * xOA * xOA)
            + (0.353 * xOPTT * xAlpha)
            - (0.0497 * xOPTT * xOA)
            - (0.423 * xOPTT * xOPTT)
            + (0.202 * xHA * xAlpha * xAlpha)
            - (0.281 * xOA * xAlpha * xAlpha)
            - (0.342 * xHA * xHA * xAlpha)
            - (0.245 * xHA * xHA * xOA)
            + (0.281 * xOA * xOA * xHA)
            - (0.184 * xOPTT * xOPTT * xAlpha)
            - (0.281 * xHA * xAlpha * xOA)
        )
        return np.array([f1, f2, f3])


# Registry of the supported RE problems (lower-case id -> class)
RE_PROBLEMS = {"re21": RE21, "re34": RE34, "re37": RE37}


def get_re_problem(problem_id):
    """Return an RE problem instance for `problem_id`, or None if not an RE id."""
    cls = RE_PROBLEMS.get(str(problem_id).lower())
    return cls() if cls is not None else None
