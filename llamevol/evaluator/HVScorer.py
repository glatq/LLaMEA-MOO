# HVScorer.py
from __future__ import annotations
from pymoo.indicators.hv import HV
from typing import Dict, List, Optional
import numpy as np


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict Pareto dominance (minimization)."""
    return np.all(a <= b) and np.any(a < b)


class HVScorer:
    """
    Convert a sequence of M-objective values F (evaluation order) into:
      - scalar y_hv (final HV or AoHV),
      - hv_curve (HV after each accepted non-dominated insertion),
      - Pareto archive indices and front,
      - fitness suitable for a minimize-only pipeline: fitness = -y_hv
    """

    def __init__(
        self,
        ref_point: Optional[np.ndarray] = None,
        use_aohv: bool = False,
        ref_margin: float = 0.10,
    ):
        """
        Parameters
        ----------
        ref_point : Optional[np.ndarray]
            If provided, use as the HV reference point (minimization).
        use_aohv : bool
            If True, optimize area-under-HV curve; else final HV only.
        ref_margin : float
            Fallback margin when inferring reference point from data.
        """
        self._ref_point = (
            None if ref_point is None else np.asarray(ref_point, float).ravel()
        )
        self._use_aohv = bool(use_aohv)
        self._ref_margin = float(ref_margin)

    def _infer_ref(self, F: np.ndarray) -> np.ndarray:
        """Adaptive reference point strictly dominated by observed region."""
        ymax = np.max(F, axis=0)
        ymin = np.min(F, axis=0)
        span = (ymax - ymin) + 1e-9
        return ymax + self._ref_margin * span

    def score(self, F: np.ndarray) -> Dict[str, object]:
        """
        Parameters
        ----------
        F : np.ndarray
            Shape (n_evals, M). Minimization assumed.

        Returns
        -------
        dict with keys:
          fitness: float        # negative scalar for minimize-only pipelines
          y_hv: float           # positive scalar HV or AoHV
          hv_curve: List[float]
          pareto_idx: List[int]
          pareto_F: np.ndarray  # (|Pareto|, M)
          ref_point: np.ndarray # used reference point
        """
        F = np.asarray(F, float)
        if F.size == 0:
            return dict(
                fitness=0.0,
                y_hv=0.0,
                hv_curve=[],
                pareto_idx=[],
                pareto_F=np.empty((0, 0)),
                ref_point=self._ref_point
                if self._ref_point is not None
                else np.array([]),
            )

        ref = self._ref_point if self._ref_point is not None else self._infer_ref(F)
        hv_indicator = HV(ref_point=ref)

        cur_idx: List[int] = []
        hv_curve: List[float] = []

        for i, f in enumerate(F):
            if any(_dominates(F[j], f) for j in cur_idx):
                hv_curve.append(hv_curve[-1] if hv_curve else 0.0)
                continue
            # prune dominated archive points, add newcomer
            cur_idx = [j for j in cur_idx if not _dominates(f, F[j])]
            cur_idx.append(i)
            hv_curve.append(float(hv_indicator(F[cur_idx])))

        pareto_idx = cur_idx
        pareto_F = F[pareto_idx] if pareto_idx else np.empty((0, F.shape[1]))

        if self._use_aohv:
            # Normalize by length to be comparable across budgets
            y_hv = float(np.trapz(hv_curve, dx=1) / max(1, len(hv_curve)))
        else:
            y_hv = hv_curve[-1] if hv_curve else 0.0

        fitness = -float(y_hv)  # keep the global contract: lower is better

        return dict(
            fitness=fitness,
            y_hv=float(y_hv),
            hv_curve=hv_curve,
            pareto_idx=pareto_idx,
            pareto_F=pareto_F,
            ref_point=ref,
        )
