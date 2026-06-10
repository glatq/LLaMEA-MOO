"""Real-world engineering benchmark problems (RE / CRE suites).

Taken from Tanabe & Ishibuchi (2020), "An easy-to-use real-world multi-objective
optimization problem suite", Applied Soft Computing --
https://github.com/ryojitanabe/reproblems .

Two families are provided:

* **RE21, RE34, RE37** -- the unconstrained real-world problems used in Paper 1.
* **CRE21--CRE25, CRE31, CRE32, CRE51** -- the constrained suite (added for the
  constrained extension). These expose genuine inequality constraints; Tanabe's
  ``g >= 0`` feasible convention is converted to the pymoo / PR1 ``G <= 0``
  feasible convention (see the CRE section below).

Each class exposes the minimal interface ``PymooMOProvider`` needs: ``n_obj``,
``n_ieq_constr``, ``xl``, ``xu`` and ``evaluate(X, return_values_of=...)``.
Unconstrained problems return ``[F]`` (default); constrained problems also
support ``return_values_of=["F", "G"]`` -> ``[F, G]``, matching the subset of
the pymoo Problem API the provider's wrapper calls.
"""

import numpy as np


class _REProblem:
    """Base: batches the single-point formula to the pymoo-like API.

    Unconstrained problems implement ``_evaluate_one(x) -> F`` and leave
    ``n_ieq_constr == 0`` (legacy behaviour, unchanged).

    Constrained problems set ``n_ieq_constr > 0`` and implement
    ``_evaluate_fg_one(x) -> (F, G)`` instead, where ``G`` is the *signed*
    inequality-constraint vector in the ``G <= 0`` feasible convention (pymoo /
    PR1 standard). The provider auto-detects constraints via ``n_ieq_constr``
    and ``evaluate(..., return_values_of=["F","G"])`` then returns ``[F, G]``.
    """

    problem_name = ""
    n_obj = 0
    # Number of inequality constraints (G <= 0). 0 == unconstrained.
    # The attribute name matches pymoo so PymooMOProvider auto-detects it.
    n_ieq_constr = 0

    def _evaluate_one(self, x):
        """Objectives F for a single point (unconstrained problems)."""
        raise NotImplementedError

    def _evaluate_fg_one(self, x):
        """Objectives F and signed constraints G (G <= 0 feasible) for one point.

        Only implemented by constrained subclasses (n_ieq_constr > 0).
        """
        raise NotImplementedError

    def evaluate(self, X, return_values_of=("F",), **kwargs):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.n_ieq_constr > 0:
            pairs = [self._evaluate_fg_one(row) for row in X]
            F = np.array([p[0] for p in pairs], dtype=float)
            G = np.array([p[1] for p in pairs], dtype=float)
        else:
            F = np.array([self._evaluate_one(row) for row in X], dtype=float)
            G = np.zeros((X.shape[0], 0), dtype=float)

        out = {"F": F, "G": G}
        # Return requested quantities in the requested order, mirroring pymoo's
        # `evaluate(..., return_values_of=[...])`. Default ("F",) -> [F], so the
        # existing unconstrained call sites (`evaluate(x)[0]`) are unchanged.
        return [out[k] for k in return_values_of]


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


# --------------------------------------------------------------------------- #
# Constrained RE problems (CRE suite, Tanabe & Ishibuchi 2020).                #
#                                                                              #
# Objectives are transcribed verbatim from reproblem_python_ver/reproblem.py.  #
# Tanabe's code uses the convention "g >= 0 means feasible" and returns the    #
# clipped violation max(0, -g). We instead expose the *signed* constraint in   #
# the pymoo / PR1 convention "G <= 0 means feasible" by returning G = -g, so   #
# constraint-aware algorithms and the qNEHVI baseline can model the real       #
# (signed) constraint function. cv = sum(max(0, G)) then equals Tanabe's total #
# violation, and feasibility (cv == 0) is identical.                           #
# --------------------------------------------------------------------------- #


class CRE21(_REProblem):
    """Two bar truss design (2 objectives, 3 variables, 3 constraints)."""

    problem_name = "CRE21"
    n_obj = 2
    n_ieq_constr = 3

    def __init__(self):
        self.xl = np.array([0.00001, 0.00001, 1.0])
        self.xu = np.array([100.0, 100.0, 3.0])

    def _evaluate_fg_one(self, x):
        x1, x2, x3 = x
        f0 = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        f1 = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
        g0 = 0.1 - f0
        g1 = 100000.0 - f1
        g2 = 100000.0 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        return np.array([f0, f1]), -np.array([g0, g1, g2])


class CRE22(_REProblem):
    """Welded beam design (2 objectives, 4 variables, 4 constraints)."""

    problem_name = "CRE22"
    n_obj = 2
    n_ieq_constr = 4

    def __init__(self):
        self.xl = np.array([0.125, 0.1, 0.1, 0.125])
        self.xu = np.array([5.0, 10.0, 10.0, 5.0])

    def _evaluate_fg_one(self, x):
        x1, x2, x3, x4 = x
        P = 6000.0
        L = 14.0
        E = 30 * 1e6
        G_mod = 12 * 1e6  # shear modulus (Tanabe's local `G`; renamed)
        tauMax = 13600.0
        sigmaMax = 30000.0

        f0 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f1 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar
        tauDashDash = (M * R) / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = (
            tauDash * tauDash
            + ((2 * tauDash * tauDashDash * x2) / (2 * R))
            + (tauDashDash * tauDashDash)
        )
        tau = np.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = (
            4.013
            * E
            * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0)
            / (L * L)
        )
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G_mod))
        PC = tmpVar * (1 - tmpVar2)

        g0 = tauMax - tau
        g1 = sigmaMax - sigma
        g2 = x4 - x1
        g3 = PC - P
        return np.array([f0, f1]), -np.array([g0, g1, g2, g3])


class CRE23(_REProblem):
    """Disc brake design (2 objectives, 4 variables, 4 constraints)."""

    problem_name = "CRE23"
    n_obj = 2
    n_ieq_constr = 4

    def __init__(self):
        self.xl = np.array([55.0, 75.0, 1000.0, 11.0])
        self.xu = np.array([80.0, 110.0, 3000.0, 20.0])

    def _evaluate_fg_one(self, x):
        x1, x2, x3, x4 = x
        f0 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        f1 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (
            x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)
        )
        g0 = (x2 - x1) - 20.0
        g1 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g2 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power(
            (x2 * x2 - x1 * x1), 2
        )
        g3 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (
            x2 * x2 - x1 * x1
        ) - 900.0
        return np.array([f0, f1]), -np.array([g0, g1, g2, g3])


class CRE24(_REProblem):
    """Speed reducer design (2 objectives, 7 variables, 11 constraints).

    x3 is an integer variable (number of teeth); rounded as in Tanabe's code.
    """

    problem_name = "CRE24"
    n_obj = 2
    n_ieq_constr = 11

    def __init__(self):
        self.xl = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0])
        self.xu = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])

    def _evaluate_fg_one(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = np.round(x[2])
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        f0 = (
            0.7854
            * x1
            * (x2 * x2)
            * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934)
            - 1.508 * x1 * (x6 * x6 + x7 * x7)
            + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7)
            + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
        )
        tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
        f1 = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        g = np.zeros(11)
        g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g[4] = -(x2 * x3) + 40.0
        g[5] = -(x1 / x2) + 12.0
        g[6] = -5.0 + (x1 / x2)
        g[7] = -1.9 + x4 - 1.5 * x6
        g[8] = -1.9 + x5 - 1.1 * x7
        g[9] = -f1 + 1300.0
        tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
        return np.array([f0, f1]), -g


class CRE25(_REProblem):
    """Gear train design (2 objectives, 4 integer variables, 1 constraint)."""

    problem_name = "CRE25"
    n_obj = 2
    n_ieq_constr = 1

    def __init__(self):
        self.xl = np.full(4, 12.0)
        self.xu = np.full(4, 60.0)

    def _evaluate_fg_one(self, x):
        # All four variables must take integer values.
        x1 = np.round(x[0])
        x2 = np.round(x[1])
        x3 = np.round(x[2])
        x4 = np.round(x[3])

        f0 = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        f1 = max([x1, x2, x3, x4])
        g0 = 0.5 - (f0 / 6.931)
        return np.array([f0, f1]), -np.array([g0])


class CRE31(_REProblem):
    """Car side-impact design (3 objectives, 7 variables, 10 constraints)."""

    problem_name = "CRE31"
    n_obj = 3
    n_ieq_constr = 10

    def __init__(self):
        self.xl = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4])
        self.xu = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2])

    def _evaluate_fg_one(self, x):
        x1, x2, x3, x4, x5, x6, x7 = x
        f0 = (
            1.98
            + 4.9 * x1
            + 6.67 * x2
            + 6.98 * x3
            + 4.01 * x4
            + 1.78 * x5
            + 0.00001 * x6
            + 2.73 * x7
        )
        f1 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f2 = 0.5 * (Vmbp + Vfd)

        g = np.zeros(10)
        g[0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[1] = 0.32 - (
            0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0154464 * x6
        )
        g[2] = 0.32 - (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2 * x2
        )
        g[3] = 0.32 - (
            0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2
        )
        g[4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[5] = 32 - (
            33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728
        )
        g[6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[7] = 4 - f1
        g[8] = 9.9 - Vmbp
        g[9] = 15.7 - Vfd
        return np.array([f0, f1, f2]), -g


class CRE32(_REProblem):
    """Conceptual marine (bulk carrier) design (3 objectives, 6 vars, 9 constraints).

    The third objective is annual cargo transport capacity, handled as a
    minimization of its negation (f2 = -annual_cargo), as in Tanabe's code.
    """

    problem_name = "CRE32"
    n_obj = 3
    n_ieq_constr = 9

    def __init__(self):
        self.xl = np.array([150.0, 20.0, 13.0, 10.0, 14.0, 0.63])
        self.xu = np.array([274.32, 32.31, 25.0, 11.71, 18.0, 0.75])

    def _evaluate_fg_one(self, x):
        x_L, x_B, x_D, x_T, x_Vk, x_CB = x

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g_acc = 9.8065  # gravity (Tanabe's local `g`; renamed)
        Fn = V / np.power(g_acc * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (
            a + (b * Fn)
        )
        outfit_weight = (
            1.0
            * np.power(x_L, 0.8)
            * np.power(x_B, 0.6)
            * np.power(x_D, 0.3)
            * np.power(x_CB, 0.1)
        )
        steel_weight = (
            0.034
            * np.power(x_L, 1.7)
            * np.power(x_B, 0.7)
            * np.power(x_D, 0.4)
            * np.power(x_CB, 0.5)
        )
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            (2000.0 * np.power(steel_weight, 0.85))
            + (3500.0 * outfit_weight)
            + (2400.0 * np.power(power, 0.8))
        )
        capital_costs = 0.2 * ship_cost
        DWT = displacement - light_ship_weight
        running_costs = 40000.0 * np.power(DWT, 0.3)
        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0
        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)
        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)
        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)
        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f0 = annual_costs / annual_cargo
        f1 = light_ship_weight
        f2 = -annual_cargo

        c = np.zeros(9)
        c[0] = (x_L / x_B) - 6.0
        c[1] = -(x_L / x_D) + 15.0
        c[2] = -(x_L / x_T) + 19.0
        c[3] = 0.45 * np.power(DWT, 0.31) - x_T
        c[4] = 0.7 * x_D + 0.7 - x_T
        c[5] = 500000.0 - DWT
        c[6] = DWT - 3000.0
        c[7] = 0.32 - Fn
        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        c[8] = (KB + BMT - KG) - (0.07 * x_B)
        return np.array([f0, f1, f2]), -c


class CRE51(_REProblem):
    """Water resource planning (5 objectives, 3 variables, 7 constraints)."""

    problem_name = "CRE51"
    n_obj = 5
    n_ieq_constr = 7

    def __init__(self):
        self.xl = np.array([0.01, 0.01, 0.01])
        self.xu = np.array([0.45, 0.10, 0.10])

    def _evaluate_fg_one(self, x):
        x0, x1, x2 = x[0], x[1], x[2]
        f0 = 106780.37 * (x1 + x2) + 61704.67
        f1 = 3000 * x0
        f2 = 305700 * 2289 * x1 / np.power(0.06 * 2289, 0.65)
        f3 = 250 * 2289 * np.exp(-39.75 * x1 + 9.9 * x2 + 2.74)
        f4 = 25 * (1.39 / (x0 * x1) + 4940 * x2 - 80)

        g = np.zeros(7)
        g[0] = 1 - (0.00139 / (x0 * x1) + 4.94 * x2 - 0.08)
        g[1] = 1 - (0.000306 / (x0 * x1) + 1.082 * x2 - 0.0986)
        g[2] = 50000 - (12.307 / (x0 * x1) + 49408.24 * x2 + 4051.02)
        g[3] = 16000 - (2.098 / (x0 * x1) + 8046.33 * x2 - 696.71)
        g[4] = 10000 - (2.138 / (x0 * x1) + 7883.39 * x2 - 705.04)
        g[5] = 2000 - (0.417 * x0 * x1 + 1721.26 * x2 - 136.54)
        g[6] = 550 - (0.164 / (x0 * x1) + 631.13 * x2 - 54.48)
        return np.array([f0, f1, f2, f3, f4]), -g


def compute_ref_point(
    problem,
    n_samples=10000,
    percentile=95.0,
    scale=1.1,
    seed=0,
    feasible_only=False,
):
    """Reference point for hypervolume, per the paper's recipe.

    Returns ``scale * percentile-of-objectives`` over ``n_samples`` uniform
    random samples in the box bounds (default: 1.1 x 95th percentile per
    objective). For constrained problems, ``feasible_only=True`` restricts the
    percentile to feasible samples (more representative of the feasible HV that
    is actually scored); it falls back to all samples if none are feasible.

    Deterministic given ``seed`` so the resulting reference points are
    reproducible.
    """
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    rng = np.random.default_rng(seed)
    X = xl + (xu - xl) * rng.random((n_samples, len(xl)))

    res = problem.evaluate(X, return_values_of=["F", "G"])
    F = np.asarray(res[0], dtype=float)
    if feasible_only and getattr(problem, "n_ieq_constr", 0) > 0:
        G = np.asarray(res[1], dtype=float)
        feasible = np.maximum(0.0, G).sum(axis=1) == 0
        if feasible.any():
            F = F[feasible]
    return scale * np.percentile(F, percentile, axis=0)


# Registry of the supported RE / CRE problems (lower-case id -> class).
# RE21/34/37 are the unconstrained real-world problems (Paper 1); the CRE
# problems are the constrained suite added for the constrained extension.
RE_PROBLEMS = {
    "re21": RE21,
    "re34": RE34,
    "re37": RE37,
    "cre21": CRE21,
    "cre22": CRE22,
    "cre23": CRE23,
    "cre24": CRE24,
    "cre25": CRE25,
    "cre31": CRE31,
    "cre32": CRE32,
    "cre51": CRE51,
}


def get_re_problem(problem_id):
    """Return an RE/CRE problem instance for `problem_id`, or None if unknown."""
    cls = RE_PROBLEMS.get(str(problem_id).lower())
    return cls() if cls is not None else None
