"""Fawcett & Hoos ablation between MOEA/D-EGO and MOEAD-EI Hybrid.

``ModularMOEAD`` builds either algorithm, or any mix of the two, from ten blocks
named by a configuration string (``ablation.config``).
"""

from .config import BLOCKS, END, START, neighbours, parse_config
from .modular import HYBRID_TUNED, ModularMOEAD

__all__ = ["BLOCKS", "END", "START", "HYBRID_TUNED", "ModularMOEAD", "neighbours", "parse_config"]
