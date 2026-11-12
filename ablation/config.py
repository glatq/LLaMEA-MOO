"""Configuration strings of the modular algorithm.

A configuration names, for each of the ten blocks, which of the two algorithms it
comes from: ``S`` (the start of the ablation path, MOEA/D-EGO as implemented in
DDMTOLab) or ``T`` (the end of the path, the tuned MOEAD-EI Hybrid). It is written
as a ten-letter string in block order, so ``SSSSSSSSSS`` is MOEA/D-EGO and
``TTTTTTTTTT`` is MOEAD-EI Hybrid.
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


def parse_config(config: str) -> dict:
    """Map a ten-letter S/T string to ``{block: "S" | "T"}``."""
    config = str(config).strip().upper()
    if len(config) != len(BLOCKS) or set(config) - {"S", "T"}:
        raise ValueError(
            f"config must be {len(BLOCKS)} letters of S/T in block order "
            f"{BLOCKS}, got {config!r}"
        )
    return dict(zip(BLOCKS, config))


def neighbours(config: str, towards: str) -> list:
    """Configurations one block closer to ``towards`` (one ablation round).

    Returns ``(block_index, config)`` pairs, in block order.
    """
    parse_config(config)
    parse_config(towards)
    out = []
    for i, (a, b) in enumerate(zip(config, towards)):
        if a != b:
            out.append((i, config[:i] + b + config[i + 1:]))
    return out
