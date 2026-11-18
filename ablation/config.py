"""Configuration strings of the modular algorithms.

A configuration names, for each block, which of the two algorithms it comes from:
``S`` (the start of the ablation path) or ``T`` (the end). It is written as a
string of S/T letters in block order.

Two ablations use them:

* MOEA/D-EGO as implemented in DDMTOLab (S) and the tuned MOEAD-EI Hybrid (T), ten
  blocks (``BLOCKS``): ``SSSSSSSSSS`` is MOEA/D-EGO, ``TTTTTTTTTT`` the Hybrid;
* ParEGO as implemented in DDMTOLab (S) and the tuned Improved-Scalarized-EI (T),
  seven blocks (``ISEI_BLOCKS``): ``SSSSSSS`` is ParEGO, ``TTTTTTT`` ISEI.
"""

BLOCKS = (
    "init",           # 1  initial design
    "surrogate",      # 2  surrogate model
    "training",       # 3  training data (all points + local models / sliding window)
    "scaling",        # 4  objective scaling
    "weights",        # 5  subproblem weight vectors
    "scalarisation",  # 6  Tchebycheff / augmented Tchebycheff
    "ideal",          # 7  ideal point
    "acquisition",    # 8  expected Tchebycheff improvement / EI of the largest term
    "search",         # 9  candidate search
    "selection",      # 10 batch selection
)

START = "S" * len(BLOCKS)  # MOEA/D-EGO
END = "T" * len(BLOCKS)    # MOEAD-EI Hybrid

ISEI_BLOCKS = (
    "init",           # 1  initial design: Latin hypercube 11d-1 / Sobol 2d+1
    "weights",        # 2  weight vectors: random from a uniform set / a Sobol set in turn
    "scalarisation",  # 3  augmented Tchebycheff / weighted sum (each with its normalisation)
    "modelled",       # 4  one GP on the scalarised value / one GP per objective
    "gp",             # 5  BoTorch SingleTaskGP / scikit-learn Matern-5/2 + noise
    "training",       # 6  all points / the most recent points
    "search",         # 7  LogEI by BoTorch's optimiser / EI by ISEI's genetic algorithm
)

ISEI_START = "S" * len(ISEI_BLOCKS)  # ParEGO
ISEI_END = "T" * len(ISEI_BLOCKS)    # Improved-Scalarized-EI


def parse_config(config: str, blocks=BLOCKS) -> dict:
    """Map an S/T string, one letter per block, to ``{block: "S" | "T"}``."""
    config = str(config).strip().upper()
    if len(config) != len(blocks) or set(config) - {"S", "T"}:
        raise ValueError(
            f"config must be {len(blocks)} letters of S/T in block order "
            f"{blocks}, got {config!r}"
        )
    return dict(zip(blocks, config))


def neighbours(config: str, towards: str) -> list:
    """Configurations one block closer to ``towards`` (one ablation round).

    Returns ``(block_index, config)`` pairs, in block order. Works for either
    ablation: both strings must be S/T letters of the same length.
    """
    config, towards = str(config).strip().upper(), str(towards).strip().upper()
    if len(config) != len(towards) or set(config + towards) - {"S", "T"}:
        raise ValueError(f"configs must be S/T strings of one length, got {config!r}, {towards!r}")
    out = []
    for i, (a, b) in enumerate(zip(config, towards)):
        if a != b:
            out.append((i, config[:i] + b + config[i + 1:]))
    return out
