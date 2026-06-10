"""Shared multi-objective scoring helpers.

Centralizes hypervolume / feasible-hypervolume computation so the evaluator
(``multiobj_evaluator``) and the SMAC HPO wrapper (``smac_hpo_wrapper``) score
runs through *identical* logic. It lives in its own module to avoid a circular
import: ``multiobj_evaluator`` already imports ``smac_hpo_wrapper``, so the
helper cannot live in either of them.

Constraint convention follows pymoo and the field standard: ``G <= 0`` means a
constraint is satisfied, so the per-point constraint violation is

    cv = sum(max(0, G))    (summed over the constraint dimension)

and a point is feasible iff ``cv == 0``. No sign flipping is performed anywhere.

Backward compatibility: on the unconstrained path (``G is None`` or empty),
``score_moo_run`` returns exactly the normalized hypervolume the evaluator
computed before this module existed, so Paper 1 numbers are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Score ceiling for a run that produced *zero* feasible points, before the soft
# constraint-violation gradient is applied. Kept tiny so that any feasible run
# (whose score comes from a real hypervolume, typically orders of magnitude
# larger) outranks every fully-infeasible run, while SMAC/ES still see a usable
# gradient among infeasible configurations.
DEFAULT_INFEASIBLE_EPSILON = 1e-3


@dataclass
class MOOScore:
    """Outcome of scoring a single multi-objective run.

    Attributes:
        score: Normalized hypervolume (higher is better, ~[0, 1]). For a
            fully-infeasible constrained run this is the soft fallback value.
        feasibility_rate: Fraction of evaluated points with ``cv == 0``.
            Always 1.0 on the unconstrained path.
        cv: Per-evaluation constraint violation, shape ``(n_points,)``.
            All zeros on the unconstrained path.
        n_feasible: Number of feasible evaluated points.
    """

    score: float
    feasibility_rate: float
    cv: np.ndarray
    n_feasible: int


def constraint_violation(G) -> np.ndarray:
    """Per-row constraint violation ``cv = sum(max(0, G))`` (``G <= 0`` feasible).

    Returns a 1-D array with one entry per row of ``G``. An empty/zero-column
    ``G`` yields an all-zeros vector (no constraints -> no violation).
    """
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    if G.size == 0:
        return np.zeros(G.shape[0] if G.ndim >= 1 else 0)
    return np.maximum(0.0, G).sum(axis=1)


def raw_hv(Y, ref_point) -> float:
    """Hypervolume of the non-dominated front of ``Y`` w.r.t. ``ref_point``.

    Un-normalized. Returns 0.0 for an empty ``Y``.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return 0.0
    ref_point = np.asarray(ref_point, dtype=float)
    front_idx = NonDominatedSorting().do(Y, only_non_dominated_front=True)
    return float(HV(ref_point=ref_point)(Y[front_idx]))


def normalized_hv(Y, ref_point) -> float:
    """``raw_hv`` divided by the reference volume ``prod(ref_point)``.

    This reproduces the exact normalization the evaluator used previously so the
    unconstrained scoring path is numerically unchanged.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    hv = raw_hv(Y, ref_point)
    ref_volume = float(np.prod(ref_point))
    return hv / ref_volume if ref_volume > 0 else hv


def score_moo_run(
    Y,
    ref_point,
    G=None,
    epsilon: float = DEFAULT_INFEASIBLE_EPSILON,
) -> MOOScore:
    """Score a single multi-objective run.

    Unconstrained (``G`` is None or has no columns): normalized HV over all
    points -- identical to the legacy behaviour.

    Constrained: normalized HV restricted to feasible (``cv == 0``) points. If
    no feasible point exists, fall back to a tiny gradient

        epsilon * (1 - mean_relative_cv)

    where ``mean_relative_cv = mean(cv / max(cv))`` lies in ``[0, 1]``, so the
    optimizer still sees improvement as solutions approach feasibility. Any
    feasible run outranks any fully-infeasible run.
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0] if Y.ndim >= 1 else 0

    has_constraints = G is not None and np.asarray(G).size > 0
    if not has_constraints:
        return MOOScore(
            score=normalized_hv(Y, ref_point),
            feasibility_rate=1.0,
            cv=np.zeros(n),
            n_feasible=n,
        )

    cv = constraint_violation(G)
    feasible = cv <= 0.0
    n_feasible = int(feasible.sum())
    feasibility_rate = float(feasible.mean()) if cv.size else 0.0

    if n_feasible > 0:
        score = normalized_hv(Y[feasible], ref_point)
    else:
        max_cv = float(cv.max()) if cv.size else 0.0
        mean_relative_cv = float((cv / max_cv).mean()) if max_cv > 0 else 0.0
        score = epsilon * (1.0 - mean_relative_cv)

    return MOOScore(
        score=score,
        feasibility_rate=feasibility_rate,
        cv=cv,
        n_feasible=n_feasible,
    )
