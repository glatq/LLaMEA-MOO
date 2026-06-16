"""Regression guard: non-finite fitness must never survive selection.

A MOO individual that errors on every problem gets ``score == nan`` (no valid
per-problem result), so ``ind.fitness`` is ``nan``. ``nan`` corrupts the elitist
sort -- every ``nan`` comparison is False -- so before the fix such broken
algorithms slipped into the kept parents (observed live: 2 of 4 parents were
all-errored ``nan`` individuals), poisoning the breeding pool. Selection must
treat ``nan``/``inf``/``None`` as worst so failed runs are never bred from.
"""

import math

from llamevol.individual import Individual
from llamevol.population import ESPopulation


def _select(specs, n_parent):
    pop = ESPopulation(
        n_parent=n_parent,
        n_parent_per_offspring=1,
        n_offspring=max(1, n_parent),
        use_elitism=True,
    )
    for name, fit in specs:
        ind = Individual()
        ind.name = name
        ind.fitness = fit
        pop.add_individual(ind, generation=0)
    pop.select_next_generation()
    return {pop.individuals[i].name for i in pop.selected_generations[-1]}


def test_nan_fitness_never_selected_over_valid():
    survivors = _select(
        [
            ("good", 0.5),
            ("ok", 0.4),
            ("broken", float("nan")),
            ("meh", 0.3),
        ],
        n_parent=2,
    )
    assert survivors == {"good", "ok"}
    assert "broken" not in survivors


def test_neg_inf_and_none_fitness_never_selected():
    survivors = _select(
        [
            ("good", 0.5),
            ("failed", float("-inf")),
            ("never_evaluated", None),
            ("ok", 0.4),
        ],
        n_parent=2,
    )
    assert survivors == {"good", "ok"}


def test_one_plus_one_keeps_valid_over_nan_offspring():
    # The (1+1) case the bug hit hardest: a broken offspring must not displace a
    # valid parent.
    survivors = _select(
        [("parent_hv0.6", 0.6), ("broken_offspring", float("nan"))], n_parent=1
    )
    assert survivors == {"parent_hv0.6"}


def test_all_broken_population_does_not_crash():
    # Degenerate case: every individual failed. Selection must not raise and must
    # still return n_parent ids (there is simply no good choice).
    survivors = _select([("a", float("nan")), ("b", float("nan"))], n_parent=1)
    assert len(survivors) == 1
