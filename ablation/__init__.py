"""Fawcett & Hoos ablations between two pairs of algorithms.

``ModularMOEAD`` builds MOEA/D-EGO, MOEAD-EI Hybrid or any mix of the two from ten
blocks; ``ablation.parego_isei.ModularParEGOISEI`` does the same for ParEGO and
Improved-Scalarized-EI from seven. Both are named by a configuration string
(``ablation.config``).
"""

from .config import BLOCKS, END, ISEI_BLOCKS, ISEI_END, ISEI_START, START, neighbours, parse_config
from .modular import HYBRID_TUNED, ModularMOEAD

__all__ = ["BLOCKS", "END", "START", "HYBRID_TUNED", "ModularMOEAD", "neighbours", "parse_config",
           "ISEI_BLOCKS", "ISEI_END", "ISEI_START"]
